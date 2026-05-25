#!/usr/bin/env bash
# Gemma installer — plně automatický, idempotentní.
#
# Detekuje distro (Arch / Debian / Ubuntu), nainstaluje systémové balíky,
# Ollamu, modely, whisper.cpp, Python venv + závislosti. NEinstaluje NVIDIA
# ovladač (vyžaduje reboot, riziko bricknutí systému) — jen ověří `nvidia-smi`
# a pokud chybí, dá ti instrukce.
#
# Použití:
#   ./scripts/install.sh                 # plná instalace, default modely
#   ./scripts/install.sh --skip-models   # přeskočí ollama pull (rychlé re-runs)
#   ./scripts/install.sh --no-optional   # přeskočí Brave/Claude/symlink prompty
#   ./scripts/install.sh --help
#
# Skript je idempotentní: pokud něco už máš, přeskočí to. Lze pustit víckrát.
set -euo pipefail

# ──────────────────── Colors + helpers ─────────────────────────────────────
if [[ -t 1 ]]; then
  C_RED=$'\e[31m'; C_GREEN=$'\e[32m'; C_YELLOW=$'\e[33m'
  C_BLUE=$'\e[34m'; C_BOLD=$'\e[1m'; C_RESET=$'\e[0m'
else
  C_RED=''; C_GREEN=''; C_YELLOW=''; C_BLUE=''; C_BOLD=''; C_RESET=''
fi

step()   { echo; echo "${C_BOLD}${C_BLUE}▶ $*${C_RESET}"; }
ok()     { echo "  ${C_GREEN}✓${C_RESET} $*"; }
warn()   { echo "  ${C_YELLOW}⚠${C_RESET} $*"; }
err()    { echo "  ${C_RED}✗${C_RESET} $*" >&2; }
fatal()  { err "$*"; exit 1; }
skip()   { echo "  ${C_YELLOW}…${C_RESET} $* ${C_YELLOW}(skip — už hotovo)${C_RESET}"; }

# ──────────────────── Args ─────────────────────────────────────────────────
SKIP_MODELS=0
NO_OPTIONAL=0
for arg in "$@"; do
  case "$arg" in
    --skip-models) SKIP_MODELS=1 ;;
    --no-optional) NO_OPTIONAL=1 ;;
    -h|--help)
      sed -n '2,15p' "$0" | sed 's/^# \?//'
      exit 0 ;;
    *) fatal "Neznámá volba: $arg (zkus --help)" ;;
  esac
done

# ──────────────────── Repo root ────────────────────────────────────────────
ROOT="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"
cd "$ROOT"
ok "Pracuju v: $ROOT"

# ──────────────────── 0. Sanity checks ─────────────────────────────────────
step "0. Sanity checks (root user, internet)"

[[ $EUID -ne 0 ]] || fatal "Nespouštěj jako root. Skript si sudo vyžádá sám."

if ! ping -c1 -W2 1.1.1.1 >/dev/null 2>&1; then
  warn "Nezdá se, že máš internet. Pokračuju, ale stahování může selhat."
else
  ok "Internet funguje."
fi

# ──────────────────── 1. Detekce distra ────────────────────────────────────
step "1. Detekce distribuce"

# Čteme /etc/os-release (ID + ID_LIKE) místo /etc/arch-release nebo
# /etc/debian_version — pokrývá deriváty (Pop_OS, Mint, Manjaro, EndeavourOS, …).
DISTRO=""
if [[ -r /etc/os-release ]]; then
  # shellcheck disable=SC1091
  source /etc/os-release
  case " ${ID:-} ${ID_LIKE:-} " in
    *" arch "*)
      DISTRO="arch" ;;
    *" debian "*|*" ubuntu "*)
      DISTRO="debian" ;;
  esac
fi
[[ -n "$DISTRO" ]] || fatal "Nepodporovaná distribuce. Skript umí jen Arch a Debian/Ubuntu (vč. derivátů). Zkus manuální instalaci podle README."

# systemd check — Ollama install skript + my služby na něm závisí.
[[ -d /run/systemd/system ]] || fatal "Skript vyžaduje systemd init (ollama servis). Pokud máš jiný init, instaluj manuálně."

ok "Detekováno: ${PRETTY_NAME:-$DISTRO} (handler=$DISTRO)"

# ──────────────────── 2. Systémové balíky ──────────────────────────────────
step "2. Systémové balíky (base-devel, git, curl, ffmpeg, tmux, python, cmake)"

# `curl` MUSÍ být v required check — bez něj později spadne Ollama install
# tichým crashem. (Gemini/Codex review 2026-05-21).
need_install=()
for cmd in git curl ffmpeg tmux cmake python3 pip; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    need_install+=("$cmd")
  fi
done

if [[ ${#need_install[@]} -eq 0 ]]; then
  ok "Všechny systémové nástroje už máš."
else
  warn "Chybí: ${need_install[*]} — instaluju..."
  if [[ "$DISTRO" == "arch" ]]; then
    sudo pacman -S --needed --noconfirm base-devel git curl ca-certificates \
      ffmpeg tmux python python-pip cmake
  else
    sudo apt update
    sudo apt install -y build-essential git curl ca-certificates ffmpeg tmux \
      python3 python3-venv python3-pip cmake
  fi
  ok "Hotovo."
fi

# Python 3.11 přesně (Chatterbox TTS waity: 3.11 - 3.13 funguje, 3.14 ne).
# Hledáme explicitně, nepoužíváme `python3` (system může být cokoli).
# Codex review: každého kandidáta MUSÍME ověřit že má venv + ensurepip,
# jinak `python3.11 -m venv` selže až za 30 minut během instalace.
PY_BIN=""
for cand in python3.11 python3.12 python3.13; do
  if ! command -v "$cand" >/dev/null 2>&1; then continue; fi
  if ! "$cand" -c 'import venv, ensurepip' >/dev/null 2>&1; then
    warn "$cand existuje, ale chybí mu venv/ensurepip modul — přeskakuju"
    continue
  fi
  PY_BIN="$cand"
  break
done
if [[ -z "$PY_BIN" ]]; then
  if [[ "$DISTRO" == "arch" ]]; then
    fatal "Potřebuju python3.11-3.13 s venv+ensurepip. Arch má jen 'python' (= aktuální). Zkus 'sudo pacman -S python311' z AUR, nebo si zbuilduj přes pyenv."
  else
    fatal "Potřebuju python3.11-3.13 + venv. 'sudo apt install python3.11 python3.11-venv'"
  fi
fi
PY_VERSION=$("$PY_BIN" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")')
ok "Python $PY_VERSION ($PY_BIN) OK."

# ──────────────────── 3. NVIDIA + CUDA check (NO auto-install) ─────────────
step "3. NVIDIA driver + CUDA (jen kontrola, neinstaluji)"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  err "Chybí nvidia-smi → NVIDIA ovladač nainstalovaný není."
  cat <<'EOF'

  Skript ZÁMĚRNĚ neinstaluje NVIDIA driver — vyžaduje to reboot a může
  rozbít X/Wayland, pokud to uděláš v půlce instalace. Udělej to ručně:

  Arch:    sudo pacman -S nvidia nvidia-utils cuda && reboot
  Ubuntu:  sudo ubuntu-drivers autoinstall && sudo apt install nvidia-cuda-toolkit && reboot

  Pak spusť install.sh znova.
EOF
  exit 1
fi

# `|| true` chrání před pipefail trap pokud nvidia-smi vrátí nenulový exit
# (např. driver/library mismatch po update bez rebootu).
GPU_INFO=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -1 || true)
[[ -n "$GPU_INFO" ]] || fatal "nvidia-smi je nainstalován, ale neumí dotaz GPU (driver/library mismatch?). Zkus reboot, pak install.sh znova."
ok "GPU: $GPU_INFO"

if ! command -v nvcc >/dev/null 2>&1; then
  warn "nvcc nenalezen — CUDA toolkit asi chybí (PyTorch ho ale nepotřebuje)."
  warn "Pro whisper.cpp s CUDA akcelerací: sudo pacman -S cuda (Arch) / nvidia-cuda-toolkit (Ubuntu)"
else
  ok "CUDA toolkit: $(nvcc --version | grep release | awk '{print $5, $6}' | tr -d ',')"
fi

# ──────────────────── 4. Ollama ────────────────────────────────────────────
step "4. Ollama (LLM runtime)"

if command -v ollama >/dev/null 2>&1; then
  # `ollama --version` může psát na stderr nebo daemon nepřipojený warning —
  # `|| true` chrání před pipefail trap pokud grep -v nevyfiltruje nic.
  OLLAMA_VER=$(ollama --version 2>/dev/null | grep -v "^Warning" | head -1 || true)
  [[ -n "$OLLAMA_VER" ]] || OLLAMA_VER="(verze neznámá)"
  skip "Ollama už nainstalovaná ($OLLAMA_VER)"
else
  # Bezpečnější varianta `curl | sh`: stáhneme do tmp, zobrazíme hash a pak
  # spustíme. Pořád to není pinning na konkrétní commit, ale alespoň user
  # vidí co se chystá běžet (může si pak install.sh prohlédnout).
  TMP_INSTALL=$(mktemp -t ollama-install-XXXXXX.sh)
  trap 'rm -f "$TMP_INSTALL"' EXIT
  warn "Stahuju Ollama install skript z ollama.com..."
  curl -fsSL https://ollama.com/install.sh -o "$TMP_INSTALL"
  ok "Stáhnuto do $TMP_INSTALL (sha256: $(sha256sum "$TMP_INSTALL" | awk '{print $1}' | head -c 16)…)"
  warn "Spouštím install skript..."
  sh "$TMP_INSTALL"
  rm -f "$TMP_INSTALL"
  trap - EXIT
  ok "Hotovo."
fi

# Spustit službu (idempotent)
if systemctl is-active --quiet ollama; then
  ok "Ollama služba běží."
else
  sudo systemctl enable --now ollama
  ok "Spustil jsem Ollama službu."
fi

# Krátký delay, ollama API potřebuje moment na start.
for i in 1 2 3 4 5; do
  if curl -sf http://localhost:11434/api/tags >/dev/null 2>&1; then
    break
  fi
  sleep 1
done
curl -sf http://localhost:11434/api/tags >/dev/null 2>&1 \
  || warn "Ollama API neodpovídá na :11434 — možná je potřeba moment, zkus 'systemctl status ollama'."

# ──────────────────── 5. Modely ────────────────────────────────────────────
step "5. Modely (gemma4-e4b-32k, gemma4-26b-32k)"

# Sentinel directory pro install state (mtime markery, build logy).
# Vytváříme bez ohledu na --skip-models, protože step 6 ho používá pro CMAKE_LOG.
STATE_DIR="$ROOT/voice/.install-state"
mkdir -p "$STATE_DIR"

if [[ "$SKIP_MODELS" == "1" ]]; then
  warn "Přeskočeno (--skip-models)."
else
  MODELS_TO_BUILD=(
    "gemma4-e4b-32k:modelfiles/gemma4-e4b-32k.Modelfile"
    "gemma4-26b-32k:modelfiles/gemma4-26b-32k.Modelfile"
  )

  # `ollama list` může selhat pokud daemon ještě startuje; chytíme to do
  # proměnné s `|| true` aby pipefail nezabil celý skript.
  if ! EXISTING=$(ollama list 2>/dev/null); then
    fatal "Ollama daemon neodpovídá na 'ollama list'. Zkus 'systemctl status ollama' a pusť install.sh znova."
  fi
  EXISTING=$(awk 'NR>1 {print $1}' <<<"$EXISTING" || true)

  for entry in "${MODELS_TO_BUILD[@]}"; do
    tag="${entry%%:*}"
    file="${entry##*:}"
    [[ -f "$file" ]] || { warn "Modelfile $file neexistuje, přeskakuju ${tag}"; continue; }

    stamp_file="$STATE_DIR/model-${tag}.mtime"
    current_mtime=$(stat -c %Y "$file" 2>/dev/null || echo 0)
    stored_mtime=$(cat "$stamp_file" 2>/dev/null || echo "")

    if grep -q "^${tag}" <<<"$EXISTING" 2>/dev/null \
       && [[ "$current_mtime" == "$stored_mtime" ]]; then
      skip "Model ${tag} už je v Ollamě (Modelfile beze změny)"
    else
      if grep -q "^${tag}" <<<"$EXISTING" 2>/dev/null; then
        warn "Modelfile ${file} změněn — přestavuji ${tag}..."
      else
        warn "Stahuji ${tag} (~10-17 GB, 10-20 minut)..."
      fi
      ollama create "$tag" -f "$file"
      echo "$current_mtime" > "$stamp_file"
      ok "${tag} hotovo."
    fi
  done
fi

# ──────────────────── 6. whisper.cpp (STT) ─────────────────────────────────
step "6. whisper.cpp (STT engine)"

CMAKE_LOG="$STATE_DIR/whisper-cmake.log"

if [[ -d whisper.cpp/build ]] && [[ -f whisper.cpp/build/bin/whisper-cli ]]; then
  skip "whisper.cpp už zbuildován"
else
  # Detekce partial clone (interrupted git clone může nechat adresář).
  # Validní whisper.cpp checkout MUSÍ mít .git/ + CMakeLists.txt.
  if [[ -d whisper.cpp ]] && { [[ ! -d whisper.cpp/.git ]] || [[ ! -f whisper.cpp/CMakeLists.txt ]]; }; then
    warn "Detekován partial whisper.cpp clone — přesouvám stranou..."
    mv whisper.cpp "whisper.cpp.broken.$(date +%s)"
  fi
  if [[ ! -d whisper.cpp ]]; then
    # Klonujeme do tmp dirname, po úspěchu přejmenujeme — atomic safe.
    TMP_CLONE="whisper.cpp.cloning.$$"
    git clone https://github.com/ggerganov/whisper.cpp.git "$TMP_CLONE"
    mv "$TMP_CLONE" whisper.cpp
  fi
  cd whisper.cpp
  # CUDA build pokud máme nvcc, jinak CPU.
  # Output do log file aby silent failure nezůstalo skryté (Gemini review).
  if command -v nvcc >/dev/null 2>&1; then
    warn "Build s CUDA akcelerací (log: $CMAKE_LOG)..."
    cmake -B build -DGGML_CUDA=1 &>"$CMAKE_LOG" \
      || fatal "cmake configure selhal. Zkontroluj $CMAKE_LOG"
  else
    warn "Build CPU-only (chybí nvcc, log: $CMAKE_LOG)..."
    cmake -B build &>"$CMAKE_LOG" \
      || fatal "cmake configure selhal. Zkontroluj $CMAKE_LOG"
  fi
  cmake --build build -j --config Release &>>"$CMAKE_LOG" \
    || fatal "cmake build selhal. Zkontroluj $CMAKE_LOG"
  cd "$ROOT"
  ok "Hotovo."
fi

# Model — large-v3-turbo (cca 1.5 GB). Sentinel file `.complete` zabrání
# že partial/corrupt download se na rerun bere jako hotovo.
WHISPER_MODEL="whisper.cpp/models/ggml-large-v3-turbo.bin"
WHISPER_SENTINEL="${WHISPER_MODEL}.complete"

if [[ -f "$WHISPER_MODEL" ]] && [[ -f "$WHISPER_SENTINEL" ]]; then
  skip "Whisper model large-v3-turbo už stažen"
else
  [[ -f "$WHISPER_MODEL" ]] && warn "Existující soubor bez .complete sentinel — mažu (může být partial)..."
  rm -f "$WHISPER_MODEL"
  warn "Stahuju Whisper large-v3-turbo (~1.5 GB)..."
  cd whisper.cpp
  bash ./models/download-ggml-model.sh large-v3-turbo
  cd "$ROOT"
  # Sentinel = atomic marker že download dokončil
  if [[ -f "$WHISPER_MODEL" ]] && [[ $(stat -c %s "$WHISPER_MODEL") -gt 100000000 ]]; then
    touch "$WHISPER_SENTINEL"
    ok "Hotovo."
  else
    fatal "Whisper model nestáhl správně (chybí nebo příliš malý). Zkus znova."
  fi
fi

# ──────────────────── 7. Python venv ───────────────────────────────────────
step "7. Python venv + závislosti (PyTorch, FastAPI, Chatterbox TTS, pyte)"

VENV="voice/.venv-tts"

# Check pro chatterbox: glob přes všechny python3.X složky uvnitř lib/
CHATTERBOX_INSTALLED=0
if [[ -f "$VENV/bin/uvicorn" ]]; then
  for sp in "$VENV"/lib/python*/site-packages/chatterbox; do
    [[ -d "$sp" ]] && CHATTERBOX_INSTALLED=1 && break
  done
fi

if [[ "$CHATTERBOX_INSTALLED" == "1" ]]; then
  skip "Venv už hotový (chatterbox-tts nalezen)"
else
  if [[ ! -d "$VENV" ]]; then
    warn "Vytvářím venv v $VENV pomocí $PY_BIN..."
    "$PY_BIN" -m venv "$VENV"
  fi

  # Aktivace pro tento skript (subshell)
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
  pip install --upgrade pip >/dev/null

  warn "Instaluju PyTorch (CUDA 12.8 wheels, ~2.5 GB)..."
  pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128 \
    >/dev/null

  warn "Instaluju server stack (FastAPI, httpx, pyte, pytest)..."
  pip install fastapi 'uvicorn[standard]' httpx pydantic python-multipart \
    pyte pytest pytest-asyncio pytest-timeout >/dev/null

  warn "Instaluju Chatterbox TTS..."
  pip install chatterbox-tts >/dev/null

  deactivate
  ok "Venv hotový."
fi

# ──────────────────── 8. Volitelné: Brave Search API ───────────────────────
if [[ "$NO_OPTIONAL" != "1" ]]; then
  step "8. (Volitelné) Brave Search API klíč"

  if [[ -f "$HOME/.brave-search-api" ]]; then
    skip "~/.brave-search-api už existuje"
  else
    echo "  Brave Search má 2000 dotazů zdarma měsíčně. Klíč získáš na:"
    echo "  https://api.search.brave.com/  (klíč začíná BSA-...)"
    echo
    # `-s` skryje klíč na obrazovce (citlivý secret). `-p` neumí kombinaci
    # s -s na všech shellech, proto si vypisujeme prompt manuálně.
    printf '  Vlož klíč (Enter = přeskočit): '
    read -r -s brave_key
    echo
    if [[ -n "${brave_key:-}" ]]; then
      # Subshell pro umask — neovlivní zbytek skriptu (Gemini review HIGH).
      ( umask 077; echo "$brave_key" > "$HOME/.brave-search-api" )
      chmod 600 "$HOME/.brave-search-api"
      ok "Uložen do ~/.brave-search-api (chmod 600)."
    else
      warn "Přeskočeno — web_search tool bude vracet chybu, vše ostatní funguje."
    fi
    unset brave_key
  fi
fi

# ──────────────────── 9. Volitelné: Claude Code CLI ────────────────────────
if [[ "$NO_OPTIONAL" != "1" ]]; then
  step "9. (Volitelné) Claude Code CLI (pro claude mode + ask_claude tool)"

  if command -v claude >/dev/null 2>&1; then
    skip "Claude CLI už nainstalován ($(claude --version 2>&1 | head -1))"
    if claude auth status 2>&1 | grep -qi "logged in\|authenticated"; then
      ok "Claude je přihlášený."
    else
      warn "Claude CLI tu je, ale není přihlášený. Spusť ručně: 'claude auth login'"
    fi
  else
    read -r -p "  Nainstalovat Claude Code CLI? [Y/n] " yn
    if [[ ! "$yn" =~ ^[Nn] ]]; then
      # Stáhneme do tmp + ukáže se hash — uživatel může zkontrolovat
      # před spuštěním (curl|sh trade-off, viz Ollama install výše).
      TMP_CLAUDE=$(mktemp -t claude-install-XXXXXX.sh)
      trap 'rm -f "$TMP_CLAUDE"' EXIT
      curl -fsSL https://claude.ai/install.sh -o "$TMP_CLAUDE"
      ok "Stáhnuto (sha256: $(sha256sum "$TMP_CLAUDE" | awk '{print $1}' | head -c 16)…)"
      sh "$TMP_CLAUDE"
      rm -f "$TMP_CLAUDE"
      trap - EXIT
      ok "Hotovo. PO instalaci spusť ručně: 'claude auth login' (otevře browser)."
    else
      warn "Přeskočeno — claude mode + ask_claude tool budou vracet chybu."
    fi
  fi
fi

# ──────────────────── 10. Volitelné: ~/bin/gemma symlink ───────────────────
if [[ "$NO_OPTIONAL" != "1" ]]; then
  step "10. (Volitelné) Symlink ~/bin/gemma"

  if [[ -L "$HOME/bin/gemma" ]] && [[ "$(readlink "$HOME/bin/gemma")" == "$ROOT/scripts/gemma" ]]; then
    skip "Symlink už existuje a ukazuje správně"
  elif [[ -e "$HOME/bin/gemma" ]]; then
    warn "~/bin/gemma už existuje, ale ukazuje jinam — nepřepisuju."
  else
    read -r -p "  Vytvořit symlink ~/bin/gemma → scripts/gemma? [Y/n] " yn
    if [[ ! "$yn" =~ ^[Nn] ]]; then
      mkdir -p "$HOME/bin"
      ln -s "$ROOT/scripts/gemma" "$HOME/bin/gemma"
      ok "Hotovo."
      if [[ ":$PATH:" != *":$HOME/bin:"* ]]; then
        warn "~/bin není v \$PATH. Přidej do ~/.bashrc / ~/.zshrc:"
        echo "      export PATH=\"\$HOME/bin:\$PATH\""
      fi
    fi
  fi
fi

# ──────────────────── Hotovo ───────────────────────────────────────────────
echo
echo "${C_GREEN}${C_BOLD}═══════════════════════════════════════════════════════════════${C_RESET}"
echo "${C_GREEN}${C_BOLD}  ✓ Instalace hotová.${C_RESET}"
echo "${C_GREEN}${C_BOLD}═══════════════════════════════════════════════════════════════${C_RESET}"
echo
echo "Spuštění:"
echo
echo "  ${C_BOLD}# Vytvoř si pracovní adresář (sandbox root, NE \$HOME):${C_RESET}"
echo "  mkdir -p ~/git/github/muj-projekt && cd ~/git/github/muj-projekt"
echo
echo "  ${C_BOLD}# Spustit:${C_RESET}"
if [[ -L "$HOME/bin/gemma" ]]; then
  echo "  gemma"
else
  echo "  $ROOT/scripts/agent.sh"
fi
echo
echo "  ${C_BOLD}# Pak otevři v prohlížeči:${C_RESET}"
echo "  http://127.0.0.1:8080"
echo
echo "Pro claude mode v tmux adapteru (persistent sessions):"
echo "  AGENT_CLAUDE_BRIDGE_MODE=tmux gemma"
echo
