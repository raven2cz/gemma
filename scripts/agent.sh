#!/usr/bin/env bash
# Spouštěč Gemma voice + agent webapp.
#
# WORKDIR = aktuální adresář ($PWD), kde skript spouštíš. Agent dostane
# tuhle cestu jako sandbox root pro read/write/bash AUTO.
#
# Použití:
#   ./scripts/agent.sh                # WORKDIR = $PWD, port 8080
#   ./scripts/agent.sh --dangerous    # ASK → AUTO (destructive stále vyžaduje frázi)
#   ./scripts/agent.sh --port 9000    # vlastní port
#   ./scripts/agent.sh --host 0.0.0.0 # bind na všechny interfacy
set -euo pipefail

ROOT="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"
WORKDIR="$PWD"
PORT=8080
HOST=127.0.0.1
DANGEROUS=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port)       PORT="$2"; shift 2 ;;
    --host)       HOST="$2"; shift 2 ;;
    --dangerous)  DANGEROUS=1; shift ;;
    -h|--help)
      sed -n '2,11p' "$0" | sed 's/^# \?//'; exit 0 ;;
    *)
      echo "Neznámá volba: $1" >&2
      echo "Použij -h pro nápovědu." >&2
      exit 2
      ;;
  esac
done

# Sanity checks ---------------------------------------------------------
if [[ ! -x "$ROOT/voice/.venv-tts/bin/uvicorn" ]]; then
  echo "Venv chybí: $ROOT/voice/.venv-tts/bin/uvicorn neexistuje." >&2
  echo "Setup: cd $ROOT/voice && python -m venv .venv-tts && .venv-tts/bin/pip install -r requirements.txt" >&2
  exit 1
fi

if ! curl -sf http://localhost:11434/api/tags >/dev/null 2>&1; then
  echo "Ollama neběží na :11434 — spusť: sudo systemctl start ollama" >&2
  exit 1
fi

# Free port -----------------------------------------------------------
# Najde proces poslouchající na PORT. Pokud je to NAŠE uvicorn webapp z této
# venv (zombie po pádu klienta / Ctrl-C v jiném shellu), zabije ho. Cizí
# proces = error PŘED jakýmkoli kill (žádný kill našich pokud mezi listenery
# je i jeden cizí).
free_port() {
  local port="$1"
  local lines
  lines=$(ss -lntpH "sport = :$port" 2>/dev/null || true)
  if [[ -z "$lines" ]]; then return 0; fi

  local pids
  pids=$(printf '%s\n' "$lines" | grep -oE 'pid=[0-9]+' | cut -d= -f2 | sort -u)
  if [[ -z "$pids" ]]; then
    echo "Port $port obsazený, ale PID nelze zjistit (běží pod jiným userem?)." >&2
    return 1
  fi

  local our_venv="$ROOT/voice/.venv-tts/bin"

  # Helper: zachytí (pid, start_time_jiffies) páry pro racy guard při SIGKILL.
  # `/proc/$pid/stat` field 22 = starttime since boot (jiffies). PID recycling
  # ho NEzachová → safe odlišovač "pořád stejný proces".
  pid_starttime() {
    awk '{print $22}' "/proc/$1/stat" 2>/dev/null || echo ""
  }

  # Pass 1 — validace všech PIDs. Žádné kill, jen klasifikace. Pokud mezi
  # listenery je CIZÍ proces, error rovnou (jinak bychom mu mohli zabít
  # našeho souseda a pak teprve hlásit chybu — porušení principu).
  local pid cmdline starttime
  local -a our_pids=()
  local -a our_starttimes=()
  for pid in $pids; do
    cmdline=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || echo "")
    # Naše = uvicorn z této venv (NEstačí jen "uvicorn") + voice.webapp.server.
    if [[ "$cmdline" == *"$our_venv"* && "$cmdline" == *"voice.webapp.server"* ]]; then
      starttime=$(pid_starttime "$pid")
      our_pids+=("$pid")
      our_starttimes+=("$starttime")
    else
      echo "Port $port drží CIZÍ proces (PID $pid): $cmdline" >&2
      echo "Buď ho zabij ručně, nebo použij --port <jiný>." >&2
      exit 1
    fi
  done

  if [[ ${#our_pids[@]} -eq 0 ]]; then return 0; fi

  # Pass 2 — SIGTERM našim PIDs. Race-safe: ověř, že PID stále má identický
  # starttime (= je to pořád náš proces, ne PID recycle).
  local idx
  for idx in "${!our_pids[@]}"; do
    pid="${our_pids[$idx]}"
    starttime="${our_starttimes[$idx]}"
    local now_st
    now_st=$(pid_starttime "$pid")
    if [[ "$now_st" != "$starttime" || -z "$now_st" ]]; then
      # Proces už skončil (PID možná recyklovaný) — neposílat signál.
      continue
    fi
    echo ">>> Port $port obsadila stará webapp (PID $pid) — zabíjím..."
    kill -TERM "$pid" 2>/dev/null || true
  done

  # Wait up to 5s for graceful shutdown.
  local i
  for i in 1 2 3 4 5 6 7 8 9 10; do
    sleep 0.5
    if [[ -z "$(ss -lntH "sport = :$port" 2>/dev/null)" ]]; then return 0; fi
  done

  # SIGKILL fallback s race guard — re-verify starttime před každým KILL.
  # Pokud PID byl recyklovaný (jiný starttime), žádný signál.
  for idx in "${!our_pids[@]}"; do
    pid="${our_pids[$idx]}"
    starttime="${our_starttimes[$idx]}"
    local now_st
    now_st=$(pid_starttime "$pid")
    if [[ -z "$now_st" ]]; then continue; fi  # už neexistuje
    if [[ "$now_st" != "$starttime" ]]; then
      echo "PID $pid byl recyklovaný (jiný starttime) — neposílám SIGKILL." >&2
      continue
    fi
    echo ">>> SIGTERM nezabralo, SIGKILL PID $pid" >&2
    kill -KILL "$pid" 2>/dev/null || true
  done
  sleep 0.5
  if [[ -n "$(ss -lntH "sport = :$port" 2>/dev/null)" ]]; then
    echo "Port $port se nepodařilo uvolnit." >&2
    exit 1
  fi
}
free_port "$PORT"

# Spuštění -------------------------------------------------------------
export AGENT_WORKDIR="$WORKDIR"
if [[ "$DANGEROUS" == "1" ]]; then
  export AGENT_DANGEROUS=1
  echo "⚠️  DANGEROUS MODE — ASK rozhodnutí skipována, destructive stále vyžaduje frázi"
fi
echo ">>> WORKDIR: $WORKDIR"
echo ">>> URL:     http://$HOST:$PORT"

cd "$ROOT"
exec ./voice/.venv-tts/bin/uvicorn \
    voice.webapp.server:app --host "$HOST" --port "$PORT"
