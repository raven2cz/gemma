/**
 * Voice chat - main app.
 *
 * State machine:  idle → recording → transcribing → thinking → speaking → idle
 * Any state → error (banner, retry)
 */

// ──────────── Diagnostic logging (musí být PRVNÍ věc před import) ────────────
// Capture všechny JS errors + unhandled promise rejections + log key init steps.
// Bez tohoto, když init shodí (např. missing element, syntax error, network
// fail), UI zůstane prázdné s `…` placeholders a žádná diagnostic info.
const _APP_INIT_LOG = [];
function logInit(stage, detail) {
  const ts = new Date().toISOString().slice(11, 23);
  const msg = `[init ${ts}] ${stage}` + (detail ? `: ${detail}` : '');
  _APP_INIT_LOG.push(msg);
  console.log(msg);
}
function logError(label, err) {
  const detail = err && err.stack ? err.stack : String(err);
  const msg = `[ERROR ${label}] ${detail}`;
  _APP_INIT_LOG.push(msg);
  console.error(msg);
  // Send to server for webapp.log (best-effort, ne await).
  try {
    fetch('/api/client_log', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ level: 'error', label, detail, log: _APP_INIT_LOG.slice(-20) }),
    }).catch(() => {});
  } catch {}
  // Visible banner v UI aby user věděl že něco selhalo i bez DevTools.
  showInitErrorBanner(label, detail);
}
function showInitErrorBanner(label, detail) {
  try {
    if (document.getElementById('init-error-banner')) return;
    const banner = document.createElement('div');
    banner.id = 'init-error-banner';
    banner.style.cssText = (
      'position:fixed;top:0;left:0;right:0;z-index:99999;'
      + 'background:#7c2d12;color:#fff;padding:10px 16px;'
      + 'font-family:monospace;font-size:12px;border-bottom:2px solid #ef4444;'
    );
    banner.innerHTML = `<strong>App init error [${label}]:</strong> `
      + `<span style="opacity:0.9">${(detail || '').slice(0, 400)}</span> `
      + `<button onclick="this.parentNode.remove()" style="float:right;background:transparent;border:1px solid #fff;color:#fff;padding:2px 8px;cursor:pointer">×</button>`;
    document.body.insertBefore(banner, document.body.firstChild);
  } catch {}
}
window.addEventListener('error', (ev) => {
  logError('window.onerror',
    `${ev.message} at ${ev.filename}:${ev.lineno}:${ev.colno}`
    + (ev.error && ev.error.stack ? '\n' + ev.error.stack : ''));
});
window.addEventListener('unhandledrejection', (ev) => {
  logError('unhandledrejection', ev.reason);
});
logInit('script loaded');

import { GlowingOrb } from './orb.js';

const AVATARS = {
  [GlowingOrb.meta.id]: GlowingOrb,
};

// ──────────── DOM
const $ = (id) => document.getElementById(id);
const modelSelect = $('model-select');
const brandModel = $('brand-model');
const brandWorkdir = $('brand-workdir');

/** Synchronizuj brand badge v topbaru s aktuálně zvoleným modelem. Voláno
 * po loadModels() a po každé change události na dropdownu. Badge má title
 * = celý model string (pro hover, pokud je text useknutý ellipsis). */
function syncBrandModel() {
  if (!brandModel) return;
  const m = modelSelect.value || '(none)';
  brandModel.textContent = m;
  brandModel.title = `Model: ${m} (změníš v ⚙ Konfigurace → model)`;
}
const voiceSelect = $('voice-select');
const refSelect = $('ref-select');
const refField = $('ref-field');
const refClearBtn = $('ref-clear');
const langSelect = $('lang-select');
const fastToggle = $('fast-toggle');
const vadToggle = $('vad-toggle');
const ttsToggle = $('tts-toggle');
const ttsQuickToggle = $('tts-quick');
const streamToggle = $('stream-toggle');
const clearBtn = $('clear-btn');
const healthDot = $('health-dot');
const settingsBtn = $('settings-btn');
const settingsPanel = $('settings-panel');
const settingsBackdrop = $('settings-backdrop');
const settingsClose = $('settings-close');
const micBtn = $('mic-btn');
const stopBtn = $('stop-btn');
const sendBtn = $('send-btn');
const composer = $('composer');
const textInput = $('text-input');
const statusText = $('status-text');
const phaseLabel = $('phase-label');
const transcript = $('transcript');
const errorToast = $('error-toast');
const audioEl = $('tts-audio');
const orbCanvas = $('orb-canvas');
const orbToggle = $('orb-toggle');
const stageEl = document.querySelector('.stage');
const modeToggle = $('mode-toggle');
const approvalModal = $('approval-modal');
const approvalSummary = $('approval-summary');
const approvalRisk = $('approval-risk');
const approvalArgs = $('approval-args');
const approvalExplicit = $('approval-explicit');
const approvalPhraseInput = $('approval-phrase-input');
const approvalAllowBtn = $('approval-allow-btn');
const approvalDenyBtn = $('approval-deny-btn');
const approvalMicBtn = $('approval-mic-btn');
const approvalForm = $('approval-form');

// ──────────── State
const state = {
  phase: 'idle',
  messages: [],       // {role, content}
  audioCtx: null,
  micStream: null,
  mediaRecorder: null,
  recordedChunks: [],
  vad: null,
  chatAbort: null,
  ttsAbort: null,
  currentAssistantEl: null,
  assistantBuffer: '',
  // Jazyk: 'auto' = detekce serverem; 'cs'|'en' = force override.
  // `lastLang` drží poslední použitý output lang (pro UI stav + hysteresi).
  langOverride: localStorage.getItem('langOverride') || 'auto',
  lastLang: 'cs',
  // Input mode určuje intent: 'mic' → chceme voice out, 'text' → jen text.
  // Nastavuje se v okamžiku odeslání (snapshot).
  inputMode: 'mic',
  // Globální TTS master switch. Nepovolí TTS ani pro mic vstup.
  voiceEnabled: localStorage.getItem('tts') !== '0',
  // Streaming TTS chunking (Phase B): per-sentence synth + playback queue.
  streamTTSEnabled: localStorage.getItem('streamTTS') !== '0',
  // Agent-mode TTS scope: 'final' = jen finální odpověď po toolech (default),
  // 'off' = bez TTS v agent módu. V chat módu ignorováno (chat má vlastní stream_tts).
  ttsScope: (localStorage.getItem('ttsScope') === 'off') ? 'off' : 'final',
  // Per-turn kontext (audio queue, cancel flag, turn id). Inicializuje runTurn().
  turnCtx: null,
  // Conversation mode. 'chat' = klasický single-turn LLM; 'agent' = tool-calling
  // smyčka. Frontend stateless ze strany serveru - persistuje se v localStorage,
  // server jen čte z body.mode na každém /api/turn requestu.
  mode: localStorage.getItem('mode') || 'chat',
};

// ──────────── Avatar
// Pokud zařízení nepodporuje WebGL2 (starší Safari, vypnutá HW akcelerace),
// attach() vyhodí. Bez guardu by celý app.js padl a UI zůstalo mrtvé, takže
// sem máme fallback stub avatar, který nekreslí ale splňuje API kontrakt.
class NullAvatar {
  attach() {}
  detach() {}
  resize() {}
  setPhase() {}
  setAnalyser() {}
}

let avatar;
let avatarAlive = false;
try {
  avatar = new (AVATARS['orb'])();
  avatar.attach(orbCanvas);
  avatarAlive = true;
} catch (e) {
  console.warn('avatar attach failed, using null avatar:', e);
  avatar = new NullAvatar();
  // Schovej canvas container - bez WebGL je tam černé místo.
  if (stageEl) stageEl.classList.add('orb-unavailable');
}

function fitCanvas() {
  if (!avatarAlive) return;
  const rect = orbCanvas.getBoundingClientRect();
  avatar.resize(rect.width, rect.height, window.devicePixelRatio || 1);
}
if (avatarAlive) {
  new ResizeObserver(fitCanvas).observe(orbCanvas);
  fitCanvas();
}

// ──────────── Phase setter
//
// Jediná source of truth pro UI stavy svázané s fází.
// - state.phase: logický stav (idle | recording | transcribing | thinking |
//   synthesizing | speaking | error)
// - composer busy: během všech non-idle/error fází
// - stop button visible: během thinking | synthesizing | speaking
// - mic recording class: během recording
// - send button disabled: když je composer busy nebo textarea prázdná
// Avatar: synthesizing se mapuje na thinking (vizuálně), jinak 1:1.
const PHASE_LABELS = {
  idle: 'ready',
  recording: 'poslouchám…',
  transcribing: 'přepisuji…',
  thinking: 'přemýšlím…',
  approval: 'čekám na potvrzení…',
  synthesizing: 'syntetizuji hlas…',
  speaking: 'mluvím…',
  error: 'chyba',
};

function setPhase(phase) {
  state.phase = phase;

  const avatarPhase =
    phase === 'transcribing' || phase === 'synthesizing' || phase === 'approval'
      ? 'thinking' : phase;
  avatar.setPhase(avatarPhase);

  phaseLabel.textContent = phase;
  statusText.textContent = PHASE_LABELS[phase] || phase;

  const busy = phase !== 'idle' && phase !== 'error';

  composer.classList.toggle('busy', busy);
  // V `approval` fázi VYŽADUJEME aktivní mic + text input, aby user mohl říct
  // / napsat "ano povoluju" / "ne". Bez toho by setPhase('approval') zablokoval
  // jediné cesty, jak voice/text intercept spustit (mic + text submit handler).
  textInput.disabled = busy && phase !== 'approval';
  // Send ↔ Stop swap: v busy fázi (recording / transcribing / thinking /
  // synthesizing / speaking) schováme Send a ukážeme Stop na stejném slotu.
  // `hidden` attr odstraní z tab orderu i a11y stromu (nejen display:none).
  const wasBusy = sendBtn.hidden;
  stopBtn.hidden = !busy;
  sendBtn.hidden = busy;
  micBtn.classList.toggle('recording', phase === 'recording');
  micBtn.disabled = busy && phase !== 'recording' && phase !== 'approval';
  updateSendButton();
  // Keyboard UX: když fokus byl na Send a ten zmizí, přesuň ho na Stop,
  // ať user může Stop odpálit Enterem bez mouse/tab navigace.
  if (busy && !wasBusy && document.activeElement === sendBtn) {
    stopBtn.focus();
  }
}

function updateSendButton() {
  const submittable = state.phase === 'idle' || state.phase === 'error'
                      || state.phase === 'approval';
  sendBtn.disabled = !submittable || textInput.value.trim().length === 0;
}

// ──────────── Error UI
let errorTimer = 0;
function showError(msg, opts = {}) {
  // sticky=true → zůstane dokud user neclickne (pro chyby z agent loopu /
  // serveru, kde user musí mít čas přečíst detail). Default = 6s auto-dismiss.
  const sticky = opts.sticky === true;
  errorToast.textContent = msg;
  errorToast.hidden = false;
  errorToast.classList.toggle('sticky', sticky);
  clearTimeout(errorTimer);
  if (!sticky) {
    errorTimer = setTimeout(() => (errorToast.hidden = true), 6000);
  } else {
    errorTimer = 0;
  }
  console.error(msg);
}

// ──────────── Markdown rendering (marked + highlight.js)
// Assistant body projde markdown parserem s code highlightingem. User body
// zůstává plain text (hlasem nikdo markdown nepíše a nechceme XSS vektor).
function configureMarked() {
  // DOMPurify: u každého <a target=...> doplň rel="noopener noreferrer",
  // ať reverse tabnabbing neprojde.
  if (window.DOMPurify && !window.DOMPurify._relHookInstalled) {
    window.DOMPurify.addHook('afterSanitizeAttributes', (node) => {
      if (node.tagName === 'A' && node.hasAttribute('target')) {
        node.setAttribute('rel', 'noopener noreferrer');
      }
    });
    window.DOMPurify._relHookInstalled = true;
  }
  if (!window.marked) return;
  const renderer = new window.marked.Renderer();
  // Inject copy-button nad code block pro UX
  renderer.code = function (code, infostring) {
    let langStr = '';
    let content = code;
    if (typeof code === 'object' && code !== null) {
      langStr = (code.lang || '').match(/\S*/)?.[0] || '';
      content = code.text ?? '';
    } else {
      langStr = (infostring || '').match(/\S*/)?.[0] || '';
    }
    const lang = langStr || 'plaintext';
    let highlighted = escapeHtml(content);
    if (window.hljs) {
      try {
        const res = window.hljs.getLanguage(lang)
          ? window.hljs.highlight(content, { language: lang, ignoreIllegals: true })
          : window.hljs.highlightAuto(content);
        highlighted = res.value;
      } catch { /* fallback na escaped */ }
    }
    return `<div class="code-block">
      <div class="code-head">
        <span class="code-lang">${escapeHtml(lang)}</span>
        <button class="code-copy" type="button" title="Kopírovat">copy</button>
      </div>
      <pre><code class="hljs language-${escapeHtml(lang)}">${highlighted}</code></pre>
    </div>`;
  };
  window.marked.setOptions({
    gfm: true,
    breaks: true,
    renderer,
  });
}

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

// Assistant output projde marked → DOMPurify. Lokální LLM sice neposílá škodlivý
// HTML záměrně, ale umí vyplivnout `<img onerror=...>` nebo `javascript:` link
// (prompt injection z user content → echo v odpovědi). Bez sanitizace by se to
// přímo exekutovalo přes .innerHTML.
function renderMarkdown(text) {
  if (!window.marked) return escapeHtml(text).replace(/\n/g, '<br>');
  let html;
  try {
    html = window.marked.parse(text);
  } catch {
    return escapeHtml(text).replace(/\n/g, '<br>');
  }
  if (window.DOMPurify) {
    return window.DOMPurify.sanitize(html, {
      ADD_ATTR: ['target'],  // markdown autolinks mohou mít target="_blank"
    });
  }
  // Fallback: raději nic nevykreslit než nevysanitizované HTML.
  console.warn('DOMPurify missing; falling back to escaped text');
  return escapeHtml(text).replace(/\n/g, '<br>');
}

// ──────────── Transcript helpers
function clearWelcome() {
  const w = transcript.querySelector('.welcome');
  if (w) w.remove();
}

function addMessage(role, content = '') {
  clearWelcome();
  const el = document.createElement('div');
  el.className = `message ${role}`;
  el.innerHTML = `<div class="role">${role}</div><div class="body"></div>`;
  const body = el.querySelector('.body');
  if (role === 'assistant') {
    body.innerHTML = renderMarkdown(content);
  } else {
    body.textContent = content;
  }
  transcript.appendChild(el);
  transcript.scrollTop = transcript.scrollHeight;
  return el;
}

// Claude mode helpers
function ensureAssistantBubble(ctx) {
  if (state.currentAssistantEl) return;
  state.currentAssistantEl = addMessage('assistant', '');
}

function renderClaudeResultBlock(ev) {
  if (!ev) return null;
  const wrap = document.createElement('div');
  wrap.className = 'claude-result-card';
  wrap.dataset.mode = ev.mode || 'consult';
  const head = document.createElement('div');
  head.className = 'claude-result-head';
  const cost = ev.total_cost_usd != null ? `$${Number(ev.total_cost_usd).toFixed(4)}` : '-';
  const dur = ev.duration_ms != null ? `${(ev.duration_ms / 1000).toFixed(1)}s` : '-';
  const tools = (ev.tool_uses || []).join(', ') || '-';
  head.innerHTML = `
    <span class="claude-result-badge">🤖 ${escapeHtml(ev.model || 'claude')} · ${escapeHtml(ev.mode || 'consult')}</span>
    <span class="claude-result-meta">cost ${cost} · ${dur} · tools: ${escapeHtml(tools)} · adapter: ${escapeHtml(ev.adapter || '?')}</span>
  `;
  wrap.appendChild(head);
  if (ev.text) {
    const body = document.createElement('div');
    body.className = 'claude-result-body';
    body.textContent = ev.text;
    wrap.appendChild(body);
  }
  if (!ev.ok && ev.error) {
    const err = document.createElement('div');
    err.className = 'claude-result-error';
    err.textContent = `Error: ${ev.error}`;
    wrap.appendChild(err);
  }
  return wrap;
}

// Rerender throttled přes rAF - markdown parsing každý token je zbytečné,
// stačí aktualizovat max raz za frame (~16 ms). Idempotentní: parser dostane
// aktuální state.assistantBuffer a vyrenderuje.
let _appendPending = false;
function appendToLast(_token) {
  if (!state.currentAssistantEl) return;
  if (_appendPending) return;
  _appendPending = true;
  requestAnimationFrame(() => {
    _appendPending = false;
    if (!state.currentAssistantEl) return;
    const body = state.currentAssistantEl.querySelector('.body');
    body.innerHTML = renderMarkdown(state.assistantBuffer);
    transcript.scrollTop = transcript.scrollHeight;
  });
}

// ──────────── Health + model list
let healthPollTimer = null;
async function loadHealth() {
  let ok = false;
  try {
    const r = await fetch('/api/health');
    const h = await r.json();
    ok = !!(h.ollama && h.whisper_bin && h.ffmpeg && h.tts_ready && h.cuda);
    healthDot.classList.toggle('ok', ok);
    healthDot.classList.toggle('bad', !ok);
    healthDot.title = Object.entries(h).map(([k, v]) => `${k}: ${v ? '✓' : '✗'}`).join('\n');
    if (!h.ollama) showError('Ollama není dostupná na :11434. Spusť: sudo systemctl start ollama');
    if (!h.ffmpeg) showError('ffmpeg chybí. Nainstaluj: sudo pacman -S ffmpeg');

    // Sandbox root badge v topbaru. Červený pulse pokud == HOME (kritické).
    if (brandWorkdir && h.workdir) {
      brandWorkdir.textContent = h.workdir;
      brandWorkdir.title = `Agent sandbox root: ${h.workdir}\nVšechny fs tooly operují relativně k téhle cestě.`;
      brandWorkdir.classList.toggle('danger', !!h.workdir_is_home);
      if (h.workdir_is_home) {
        showError(
          `NEBEZPEČNÉ: WORKDIR = HOME (${h.workdir}). Agent může psát kamkoliv ` +
          `pod ~. Vypni server, přejdi do projektu a spusť \`gemma\` znova.`,
          { sticky: true }
        );
      }
    }
    const dangerBadge = $('dangerous-badge');
    if (dangerBadge) {
      dangerBadge.hidden = !h.dangerous_mode;
      if (h.dangerous_mode && h.agent_workdir) {
        dangerBadge.title = `Dangerous mode aktivní (workdir: ${h.agent_workdir}) - ASK skipována, destructive stále vyžaduje frázi`;
      }
    }
  } catch (e) {
    healthDot.classList.remove('ok');
    healthDot.classList.add('bad');
    showError('Server neodpovídá.');
  }
  // Pokud něco chybí (typicky TTS dosud loaduje, nebo Ollama restartuje),
  // poll dokud se to nevzpamatuje. Interval ~3 s stačí.
  if (!ok && !healthPollTimer) {
    healthPollTimer = setInterval(loadHealth, 3000);
  } else if (ok && healthPollTimer) {
    clearInterval(healthPollTimer);
    healthPollTimer = null;
  }
}

async function loadModels() {
  try {
    const r = await fetch('/api/models');
    const { models } = await r.json();
    modelSelect.innerHTML = '';
    for (const m of models) {
      const opt = document.createElement('option');
      opt.value = opt.textContent = m;
      modelSelect.appendChild(opt);
    }

    // Resolve uložený model proti current Ollama list. Ollama vrací modely
    // se sufixem `:latest` (např. "gemma4-e4b-32k:latest"); legacy localStorage
    // mohlo držet verzi bez sufixu z předchozí verze appky. Fallback chain:
    //   1) saved exact match
    //   2) saved + ":latest"
    //   3) saved bez ":latest" (case opačný)
    //   4) preferovaný default "gemma4-e4b-32k" v různých variantách
    //   5) první dostupný (cokoli aby dropdown nebyl prázdný)
    const saved = localStorage.getItem('model') || '';
    const tryMatch = (cand) => cand && models.includes(cand) ? cand : null;
    const resolved =
      tryMatch(saved)
      || tryMatch(saved + ':latest')
      || tryMatch(saved.replace(/:latest$/, ''))
      || tryMatch('gemma4-e4b-32k:latest')
      || tryMatch('gemma4-e4b-32k')
      || tryMatch('gemma4:e4b')
      || (models[0] || '');
    if (resolved) {
      modelSelect.value = resolved;
      // Persistuj rozhodnutí, aby další load nemusel resolve-ovat znova.
      localStorage.setItem('model', resolved);
    }
    syncBrandModel();
    console.info(`[model] resolved=${resolved} (saved=${JSON.stringify(saved)})`);
  } catch (e) {
    showError(`Nelze načíst modely: ${e.message}`);
  }
}

async function loadVoices() {
  // Voice families (nový model): backend vrátí [{family, langs}]. Každá family
  // zastřeší per-lang varianty; backend per-turn vybere správnou podle detekce.
  // Legacy explicit `ref` držíme v hidden field pro power-usery (žádný lang
  // fallback, přesný soubor).
  try {
    const [vr, rr] = await Promise.all([
      fetch('/api/voices').then(r => r.json()),
      fetch('/api/refs').then(r => r.json()),
    ]);
    const voices = vr.voices || [];
    const refs = rr.refs || [];

    // Voice select
    voiceSelect.innerHTML = '';
    for (const v of voices) {
      const opt = document.createElement('option');
      opt.value = v.family;
      const coverage = v.langs.includes('universal') ? 'universal' : v.langs.join('/');
      opt.textContent = `${v.family} · ${coverage}`;
      opt.dataset.coverage = coverage;
      voiceSelect.appendChild(opt);
    }

    // Legacy ref select (pod advanced)
    refSelect.innerHTML = '';
    for (const f of refs) {
      const opt = document.createElement('option');
      opt.value = opt.textContent = f;
      refSelect.appendChild(opt);
    }

    // Migrace: starý localStorage['ref'] (= filename) → voice family.
    // POZOR: savedRef čteme AŽ PO migraci, protože ta může `refExplicit` zapsat
    // (oldRef bez rozpoznatelné family). Bez toho by první turn jel přes voice
    // family a explicit override by se aktivoval až po reloadu.
    const oldRef = localStorage.getItem('ref');
    const savedVoice = localStorage.getItem('voice');
    const voiceValues = voices.map(v => v.family);

    if (savedVoice && voiceValues.includes(savedVoice)) {
      voiceSelect.value = savedVoice;
    } else if (oldRef) {
      // Pokus migrovat: `ref_v1.wav` → family `v1`, `ref_female_cs.wav` → `female`.
      const m = oldRef.match(/^ref_([a-zA-Z0-9_-]+?)(?:_(?:cs|en))?\.wav$/);
      if (m && voiceValues.includes(m[1])) {
        voiceSelect.value = m[1];
        localStorage.setItem('voice', m[1]);
      } else if (refs.includes(oldRef)) {
        // Nepoznaná family, zachovej jako explicit ref.
        localStorage.setItem('refExplicit', oldRef);
      }
      localStorage.removeItem('ref');
    } else if (voiceValues.includes('female')) {
      voiceSelect.value = 'female';
    }

    // Explicit ref mode (čti POST-migrace).
    const savedRef = localStorage.getItem('refExplicit');
    if (savedRef && refs.includes(savedRef)) {
      refSelect.value = savedRef;
      refField.hidden = false;
    }
  } catch (e) {
    showError(`Nelze načíst hlasy: ${e.message}`);
  }
}

// ──────────── AudioContext (only after user gesture)
async function ensureAudioCtx() {
  if (!state.audioCtx) {
    state.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  }
  if (state.audioCtx.state === 'suspended') await state.audioCtx.resume();
  return state.audioCtx;
}

// ──────────── Mic + VAD
async function startMic() {
  await ensureAudioCtx();
  if (!state.micStream) {
    state.micStream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    });
  }
  return state.micStream;
}

function stopMicTracks() {
  if (state.micStream) {
    state.micStream.getTracks().forEach((t) => t.stop());
    state.micStream = null;
  }
}

// Simple RMS VAD: once speech detected, wait for silence timeout to auto-stop.
class SimpleVAD {
  constructor(stream, audioCtx, { onStart, onEnd, silenceMs = 1500, threshold = 0.015 }) {
    this.src = audioCtx.createMediaStreamSource(stream);
    this.analyser = audioCtx.createAnalyser();
    this.analyser.fftSize = 1024;
    this.src.connect(this.analyser);
    this.buf = new Float32Array(this.analyser.fftSize);
    this.onStart = onStart;
    this.onEnd = onEnd;
    this.silenceMs = silenceMs;
    this.threshold = threshold;
    this.speaking = false;
    this.lastLoudAt = 0;
    this.running = false;
  }
  start() {
    this.running = true;
    this._tick();
  }
  stop() {
    this.running = false;
  }
  _tick = () => {
    if (!this.running) return;
    this.analyser.getFloatTimeDomainData(this.buf);
    let sum = 0;
    for (let i = 0; i < this.buf.length; i++) sum += this.buf[i] * this.buf[i];
    const rms = Math.sqrt(sum / this.buf.length);
    const now = performance.now();
    if (rms > this.threshold) {
      if (!this.speaking) {
        this.speaking = true;
        this.onStart?.();
      }
      this.lastLoudAt = now;
    } else if (this.speaking && now - this.lastLoudAt > this.silenceMs) {
      this.speaking = false;
      this.onEnd?.();
      return;  // stop ticking; caller decides to restart
    }
    requestAnimationFrame(this._tick);
  };
  getAnalyser() { return this.analyser; }
}

async function beginRecording({ auto }) {
  clearError();
  try {
    const stream = await startMic();
    const ctx = await ensureAudioCtx();

    state.recordedChunks = [];
    const mr = new MediaRecorder(stream, { mimeType: 'audio/webm;codecs=opus' });
    mr.ondataavailable = (e) => { if (e.data && e.data.size) state.recordedChunks.push(e.data); };
    mr.onstop = () => finishRecording();
    state.mediaRecorder = mr;
    // Snapshot pending approval (turnId + approvalId) - pokud se mezitím modal
    // zavře/změní, finishRecording() to detekuje a discard-ne transkripci místo
    // toho, aby ji omylem poslal na neaktivní approval nebo do nového turnu.
    state.recordingApprovalSnap = _pendingApproval
      ? { turnId: _pendingApproval.turnId, approvalId: _pendingApproval.approvalId }
      : null;
    mr.start(250);

    setPhase('recording');

    // Connect analyser for orb while recording
    if (auto) {
      const vad = new SimpleVAD(stream, ctx, {
        onStart: () => { /* already recording */ },
        onEnd: () => stopRecording(),
      });
      state.vad = vad;
      vad.start();
      avatar.setAnalyser(vad.getAnalyser());
    } else {
      // push-to-talk: still need an analyser for orb visualization
      const a = ctx.createAnalyser();
      a.fftSize = 1024;
      ctx.createMediaStreamSource(stream).connect(a);
      avatar.setAnalyser(a);
    }
  } catch (e) {
    showError(`Mikrofon: ${e.message}`);
    setPhase('idle');
  }
}

function stopRecording() {
  if (state.vad) { state.vad.stop(); state.vad = null; }
  if (state.mediaRecorder && state.mediaRecorder.state === 'recording') {
    try { state.mediaRecorder.requestData(); } catch {}
    state.mediaRecorder.stop();
  }
  // Fáze přechází na 'transcribing' v finishRecording() → mic .recording
  // třída se strhne přes setPhase.
}

async function finishRecording() {
  const blob = new Blob(state.recordedChunks, { type: 'audio/webm' });
  state.recordedChunks = [];
  stopMicTracks();
  avatar.setAnalyser(null);

  if (blob.size < 1000) {
    setPhase('idle');
    showError('Nahrávka moc krátká.');
    return;
  }

  setPhase('transcribing');
  let userText = '';
  try {
    const fd = new FormData();
    fd.append('audio', blob, 'rec.webm');
    const r = await fetch('/api/transcribe', { method: 'POST', body: fd });
    if (!r.ok) throw new Error(await r.text());
    const { text } = await r.json();
    userText = (text || '').trim();
  } catch (e) {
    showError(`Přepis selhal: ${e.message}`);
    // Pokud běží approval modal, drž fázi 'approval' (user může zkusit znovu).
    setPhase(_pendingApproval !== null ? 'approval' : 'idle');
    return;
  }

  if (!userText) {
    showError('Nic se nerozpoznalo.');
    setPhase(_pendingApproval !== null ? 'approval' : 'idle');
    return;
  }

  state.inputMode = 'mic';

  // Voice approval intercept: pokud běží modal čekající na phrase, route
  // tam místo do /api/turn. Race-guard: snapshot ID na startu recordingu
  // a při finishi STRIKTNĚ ověř shodu (snap může být null pokud recording
  // začal PŘED otevřením modalu - taky discard, jinak by neúmyslná
  // transkripce poslala approve nahodile do právě otevřeného modalu).
  if (_pendingApproval !== null) {
    const snap = state.recordingApprovalSnap;
    const matches = snap
      && snap.turnId === _pendingApproval.turnId
      && snap.approvalId === _pendingApproval.approvalId;
    if (!matches) {
      showError(
        'Mikrofon byl spuštěn před schvalovacím modalem - transkripci ignoruji. '
        + 'Stiskni mic znovu a řekni „ano povoluju" / „ne".',
        { sticky: true }
      );
      setPhase('approval');
      return;
    }
    handleVoiceApproval(userText);
    return;
  }
  // Pokud recording začal pro approval ALE modal mezitím zavřel (user kliknul
  // Allow/Deny/Esc během recording), zahoď transkripci - jinak by šla jako
  // nový chat turn. Codex audit HIGH: defense in depth nad běžným resetem
  // `recordingApprovalSnap` až v beginRecording.
  if (state.recordingApprovalSnap !== null) {
    state.recordingApprovalSnap = null;
    showError('Modal byl mezitím zavřen - nahrávku ignoruji.');
    setPhase('idle');
    return;
  }

  // Voice intent: "agent mód" / "chat mód" → přepne mode bez LLM kola.
  if (handleModeSwitchIntent(userText)) return;

  addMessage('user', userText);
  state.messages.push({ role: 'user', content: userText });

  // Next: LLM (mic vstup → chceme TTS pokud není globálně vypnutý)
  await runTurn();
}

/** Voice approval handler - mapuje transcript na approve/deny rozhodnutí.
 * No-match: ukáže transcript v modal phrase inputu + toast, user může opravit
 * ručně nebo nahrát znovu. Transcript se NEPŘIDÁVÁ do chat historie. */
function handleVoiceApproval(userText) {
  if (_pendingApproval === null) {
    setPhase('idle');
    return;
  }
  const result = classifyApprovalUtterance(userText, _pendingApproval.requiresExplicit);
  if (result === null) {
    // No-match: ukaž transcript v modal inputu + toast.
    if (_pendingApproval.requiresExplicit && approvalPhraseInput) {
      approvalPhraseInput.value = userText;
      // Re-evaluate allow button enable state (delegováno na onPhraseInput).
      approvalPhraseInput.dispatchEvent(new Event('input'));
    }
    showError(
      _pendingApproval.requiresExplicit
        ? `Nerozumím, řekni „ano povoluju" pro povolení nebo „ne" pro zamítnutí. (transcript: „${userText}")`
        : `Nerozumím, řekni „ano" / „ne". (transcript: „${userText}")`,
      { sticky: true }
    );
    // Modal pořád otevřený - phase zpátky na approval, ať mic/text input
    // zůstanou aktivní pro další pokus.
    setPhase('approval');
    return;
  }
  // Atomic resolve - finish() volá cleanup + resolve, nuluje _pendingApproval.
  // setPhase přepne caller v stream loop ('approval' → 'thinking').
  _pendingApproval.finish(result);
}

// Měkký cap: posledních 40 tahů (20 párů). Všechny naše modely mají num_ctx=32768,
// což je ~24k slov čistého textu - 40 tahů se tam bohatě vejde i s dlouhými odpověďmi.
const MAX_TURNS = 40;
function trimMessages() {
  if (state.messages.length > MAX_TURNS) {
    state.messages = state.messages.slice(-MAX_TURNS);
  }
}

// ──────────── Audio pipeline (streaming TTS)
//
// Jedno <audio> = jeden bound MediaElementSource (bind jen 1×, jinak crash).
// `audioEl.preload="auto"` + `audioEl.src = url` → browser si chunk natáhne
// přes HTTP cache. Backend nemaže audio soubory při GETu, takže retry/Range
// requests fungují nativně - žádný blob-prefetch už není potřeba.
// Single source of truth pro pipeline stav: state.turnCtx.

function ensureAudioElementWired() {
  if (audioEl._connected) return audioEl._analyser;
  const ctx = state.audioCtx;
  if (!ctx) return null;
  const srcNode = ctx.createMediaElementSource(audioEl);
  const analyser = ctx.createAnalyser();
  analyser.fftSize = 1024;
  srcNode.connect(analyser);
  analyser.connect(ctx.destination);
  audioEl._analyser = analyser;
  audioEl._connected = true;
  return analyser;
}

function newTurnCtx() {
  return {
    id: null,
    audioQueue: [],          // items: { seq, url }
    audioPlaying: false,
    textDone: false,      // server poslal 'lang' (LLM stream + queued chunks done)
    streamDone: false,    // server poslal 'done' nebo 'canceled'
    canceled: false,
    finished: false,      // už jsme zavolali finishTurnCtx
  };
}

function enqueueAudio(ctx, seq, url) {
  if (ctx.canceled) return;
  // Cache warming: nový Audio(url).load() prefetchne WAV do HTTP cache
  // bez playbacku. Bez toho by server-side post-complete cleanup (watchdog
  // smaže tmpdir 60 s po `done`) mohl ukousnout ještě nepřehrané chunky
  // dlouhé fronty - teď je frontend stáhne hned a má je lokálně.
  try {
    const pre = new Audio();
    pre.preload = 'auto';
    pre.src = url;
    // load() kickne fetch; nepotřebujeme výsledek držet, browser cache stačí.
    pre.load();
  } catch {}
  ctx.audioQueue.push({ seq, url });
  ctx.audioQueue.sort((a, b) => a.seq - b.seq);
  playNextInCtx(ctx);
}

async function playNextInCtx(ctx) {
  // Re-entrancy guard: pokud už hrajeme, nechej `ended` handler řetězit dál.
  // Flag musí být nastaven SYNCHRONNĚ PŘED jakýmkoli await - jinak by dva
  // souběžné vstupy (enqueueAudio + 'ended') mohly oba dojít k shift().
  if (ctx.audioPlaying) return;
  if (ctx.canceled || ctx.audioQueue.length === 0) {
    maybeFinishTurnCtx(ctx);
    return;
  }
  ctx.audioPlaying = true;
  const item = ctx.audioQueue.shift();
  const analyser = ensureAudioElementWired();
  if (analyser) avatar.setAnalyser(analyser);

  audioEl.src = item.url;
  if (state.phase !== 'speaking') setPhase('speaking');
  try {
    await audioEl.play();
  } catch (e) {
    console.warn('audio.play failed:', e);
    ctx.audioPlaying = false;
    maybeFinishTurnCtx(ctx);
  }
}

function maybeFinishTurnCtx(ctx) {
  if (ctx.finished) return;
  if (!ctx.streamDone) return;
  if (ctx.audioPlaying) return;
  if (ctx.audioQueue.length > 0) return;
  finishTurnCtx(ctx);
}

function finishTurnCtx(ctx) {
  ctx.finished = true;
  avatar.setAnalyser(null);
  setPhase('idle');
}

// ──────────── LLM chat (NDJSON stream) + (optional) streaming TTS
//
// runTurn otevře /api/turn (jedno spojení, heterogenní NDJSON eventy):
//   text.delta → token do buffer
//   lang_hint  → server-side preload (nic neposíláme)
//   audio      → enqueue pro playback
//   lang       → finální jazyk
//   done       → server skončil (streamDone)
//   canceled   → server přerušil
//   error      → chyba
//
// Flagy wantTTS/streamTTS se snapshotují na začátku turnu; toggle během
// streamu platí až pro další turn (žádná race).
async function runTurn() {
  const wantTTS = state.voiceEnabled && state.inputMode === 'mic';
  const streamTTS = state.streamTTSEnabled;

  trimMessages();
  setPhase('thinking');
  const assistantEl = addMessage('assistant', '');
  assistantEl.classList.add('streaming');
  state.currentAssistantEl = assistantEl;
  state.assistantBuffer = '';

  const ctx = newTurnCtx();
  state.turnCtx = ctx;

  const abort = new AbortController();
  state.chatAbort = abort;

  let outLang = state.langOverride === 'auto' ? state.lastLang : state.langOverride;

  try {
    const r = await fetch('/api/turn', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({
        model: modelSelect.value,
        mode: state.mode,
        messages: state.messages,
        // Voice family (auto per-turn lang resolve) má přednost; explicit ref
        // je legacy override pro power-usery (deterministický, bez lang swapu).
        voice: refField.hidden ? voiceSelect.value : '',
        ref: refField.hidden ? '' : refSelect.value,
        fast: fastToggle.checked,
        lang_override: state.langOverride,
        prev_lang: state.lastLang,
        want_tts: wantTTS,
        stream_tts: streamTTS,
        // Agent-mode TTS scope. V chat módu ignorováno server-side.
        tts_scope: state.ttsScope,
        // Claude-mode model selection (opus/sonnet/haiku).
        claude_model: state.mode === 'claude'
          ? (document.getElementById('claude-model-select')?.value || localStorage.getItem('claudeModel') || 'opus')
          : undefined,
      }),
      signal: abort.signal,
    });
    if (!r.ok || !r.body) throw new Error(`HTTP ${r.status}: ${await r.text()}`);

    // Vezmi turn_id z response header (server ho může sdílet; zatím ho tam
    // neposílá - použijeme URL z prvního audio eventu jako fallback).
    ctx.id = r.headers.get('x-turn-id') || null;

    const reader = r.body.getReader();
    const dec = new TextDecoder();
    let buf = '';
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      const lines = buf.split('\n');
      buf = lines.pop() || '';
      for (const line of lines) {
        if (!line.trim()) continue;
        let ev;
        try { ev = JSON.parse(line); } catch { continue; }
        switch (ev.type) {
          case 'user_lang':
            break;
          case 'text': {
            if (ev.delta) {
              // Po tool_call se assistant bubble vynulovala - vytvoř novou
              // pro další text deltas. Buffer je per-bubble, ne per-turn.
              if (!state.currentAssistantEl) {
                const newBubble = addMessage('assistant', '');
                newBubble.classList.add('streaming');
                state.currentAssistantEl = newBubble;
                state.assistantBuffer = '';
              }
              state.assistantBuffer += ev.delta;
              appendToLast();
            }
            break;
          }
          case 'lang_hint':
            outLang = ev.lang;
            break;
          case 'chunk':
            // Non-speakable blok (code) - tokeny už v text.delta, ignoruj.
            break;
          case 'tool_call':
            // Agent mode: nový tool call. Karta v transcriptu, "running" stav.
            appendToolCard(ev);
            // Pokud byla nějaká bublina rozjetá s textem, příští text delta
            // vyrobí novou (currentAssistantEl už je null po appendToolCard).
            break;
          case 'approval_required': {
            // Agent mode: schvalování. UI modal + povolený mic/text submit pro
            // hlasovou nebo textovou approval frázi (intercept v finishRecording
            // a handleTextSubmit).
            setToolCardAwaiting(ev.tool_call_id);
            setPhase('approval');
            const { decision, phrase } = await showApprovalModal(ev, ctx.id);
            // Po resolve modal: vrátit phase zpět na thinking (server pokračuje
            // tool execution / dalším round).
            setPhase('thinking');
            const ok = await sendApprovalDecision(ctx.id, ev.approval_id, decision, phrase);
            // Pokud server odmítl (např. destruktivní bez správné fráze),
            // nahlas to a abortni turn - server čeká na další POST a nic
            // jiného mu neulehčí.
            if (!ok) {
              try { await fetch(`/api/turn/${ctx.id}/cancel`, { method: 'POST' }); } catch {}
              ctx.canceled = true;
              ctx.streamDone = true;
              break;
            }
            const card = findToolCard(ev.tool_call_id);
            if (card) {
              card.dataset.status = 'running';
              card.querySelector('.tool-card-status').textContent =
                decision === 'approve' ? 'running…' : 'denied';
            }
            break;
          }
          case 'tool_result': {
            // ev.content je už-stringified (často JSON). Pokud server označil
            // ok=false a content obsahuje "denied", reflectni to v kartě.
            const isDenied = !ev.ok && /"denied"|"user denied"/i.test(ev.content || '');
            fillToolResult({ ...ev, denied: isDenied });
            // ask_claude result má speciální grafický blok pod tool kartou -
            // user vidí celý Claudův text, NEZpracovává se přes TTS (jen v UI).
            maybeRenderClaudeResultCard(ev);
            break;
          }
          case 'tool_progress': {
            // Subagent (typicky ask_claude) emituje průběžné status updaty.
            // UI text only, žádné TTS.
            updateToolProgress(ev);
            break;
          }
          case 'audio_filler': {
            // Phase 9.3: tool běží > 2s. Krátká fráze přes prohlížečové
            // speechSynthesis (cs-CZ). Hlavní TTS pipeline je zaneprázdněná
            // odpovědí agenta; tady chceme jen rychlou hluboko-latentní indikaci.
            playAudioFiller();
            break;
          }
          case 'agent_error':
            throw new Error(ev.msg || 'agent error');
          case 'agent_canceled':
            ctx.canceled = true;
            ctx.streamDone = true;
            break;
          case 'agent_done':
            // Server ještě pošle 'done' event jako canonical terminator.
            break;
          case 'claude_turn_started': {
            // Claude mode: server začal volat adapter.ask().
            setPhase('thinking');
            const msg = `🤖 ${ev.model || 'claude'} · ${ev.mode || 'consult'}`;
            ctx.claudeStartedAt = Date.now();
            // Add inline status to current assistant message
            ensureAssistantBubble(ctx);
            if (state.currentAssistantEl) {
              const ind = document.createElement('div');
              ind.className = 'claude-status-indicator';
              ind.textContent = msg;
              state.currentAssistantEl.appendChild(ind);
            }
            break;
          }
          case 'claude_result': {
            // Final result z Claude adapter - render jako claude_result_card.
            ctx.streamDone = true;
            const card = renderClaudeResultBlock(ev);
            if (card) {
              const stage = document.querySelector('.stage');
              if (stage) stage.appendChild(card);
            }
            // Add text do assistant message pokud máme
            if (ev.ok && ev.text) {
              if (state.currentAssistantEl) {
                const body = document.createElement('div');
                body.className = 'claude-assistant-text';
                body.textContent = ev.text;
                state.currentAssistantEl.appendChild(body);
              }
              state.messages.push({ role: 'assistant', content: ev.text });
            } else if (!ev.ok) {
              addMessage('assistant', `Chyba: ${ev.error || 'unknown'}`);
            }
            // Refresh permission badge (state mohl být upgrade-ovaný)
            refreshClaudePermBadge();
            setPhase('idle');
            break;
          }
          case 'claude_approval_required': {
            // Server-side gate: user chce edit ale nedal "ano povoluju".
            // Zobrazit jako system message (ne modal - mode-level toggle).
            const note = `🔒 Claude session je v read-only módu. Pro editaci napiš nebo řekni "${ev.required_phrase}" + tvoji akci.`;
            addMessage('assistant', note);
            setPhase('idle');
            break;
          }
          case 'claude_session_dead': {
            const note = `⚠️ Claude session ukončena (${ev.msg}). Při dalším dotazu se založí nová.`;
            addMessage('assistant', note);
            setPhase('idle');
            break;
          }
          case 'audio': {
            // Codex audit HIGH: zruš audio_filler (speechSynthesis) PŘED tím,
            // než pustíme server-side TTS. Jinak by hlasy spolu mluvily.
            try { window.speechSynthesis?.cancel(); } catch {}
            if (!ctx.id) {
              // Derivuj turn id z URL: /api/turn/<id>/audio/<seq>.wav
              const m = /\/api\/turn\/([a-f0-9]{16})\//.exec(ev.url);
              if (m) ctx.id = m[1];
            }
            if (state.phase === 'thinking') setPhase('synthesizing');
            enqueueAudio(ctx, ev.seq, ev.url);
            break;
          }
          case 'audio_error':
            console.warn(`tts chunk ${ev.seq} failed: ${ev.msg}`);
            break;
          case 'lang':
            outLang = ev.lang;
            ctx.textDone = true;
            break;
          case 'error':
            throw new Error(ev.msg || 'server error');
          case 'canceled':
            ctx.canceled = true;
            ctx.streamDone = true;
            break;
          case 'done':
            ctx.streamDone = true;
            break;
        }
      }
    }
  } catch (e) {
    if (e.name !== 'AbortError') {
      // Sticky = user musí mít čas přečíst (typicky stack trace nebo ollama body).
      // Console log s plnou exception pro DevTools dive.
      console.error('Turn error:', e);
      showError(`Turn: ${e.message}`, { sticky: true });
    }
    ctx.canceled = true;
    ctx.streamDone = true;
    assistantEl.classList.remove('streaming');
    state.chatAbort = null;
    maybeFinishTurnCtx(ctx);
    return;
  }

  assistantEl.classList.remove('streaming');
  if (state.currentAssistantEl) state.currentAssistantEl.classList.remove('streaming');
  state.chatAbort = null;

  // Agent mode: state.messages přepíšeme kompletní server history (vč.
  // tool_calls + tool_result), aby další turn LLM viděl tool kontext.
  // Bez toho by každý turn ztratil paměť na předchozí tooly.
  if (state.mode === 'agent' && ctx.id) {
    try {
      const r = await fetch(`/api/turn/${ctx.id}/messages`);
      if (r.ok) {
        const data = await r.json();
        if (data && data.status === 'ok' && Array.isArray(data.messages)) {
          state.messages = data.messages.filter((m) => m && m.role !== 'system');
          persistMessages();
          state.lastLang = outLang;
          ctx.streamDone = true;
          maybeFinishTurnCtx(ctx);
          return;
        }
      }
    } catch (e) {
      console.warn('agent history fetch failed:', e);
    }
    // Fallback: nepodařilo se stáhnout - alespoň ulož final text.
    if (state.assistantBuffer.trim()) {
      state.messages.push({ role: 'assistant', content: state.assistantBuffer });
      persistMessages();
    }
    state.lastLang = outLang;
    ctx.streamDone = true;
    maybeFinishTurnCtx(ctx);
    return;
  }

  if (!state.assistantBuffer.trim()) {
    ctx.streamDone = true;
    maybeFinishTurnCtx(ctx);
    return;
  }

  // Persist assistant message (oba módy - voice i text).
  state.messages.push({ role: 'assistant', content: state.assistantBuffer });
  persistMessages();
  state.lastLang = outLang;

  if (!wantTTS) {
    ctx.streamDone = true;
    finishTurnCtx(ctx);
    return;
  }

  // Když je stream hotový a queue prázdná (krátká odpověď / cache už hrála) - finish.
  maybeFinishTurnCtx(ctx);
}

// ──────────── Stop button
function stopEverything() {
  const ctx = state.turnCtx;
  if (ctx) {
    ctx.canceled = true;
    ctx.streamDone = true;
    ctx.audioQueue.length = 0;
    if (ctx.id) {
      fetch(`/api/turn/${ctx.id}/cancel`, { method: 'POST' }).catch(() => {});
    } else {
      // Fallback (turn id ještě nevypadl z prvního audio eventu):
      fetch('/api/tts/cancel', { method: 'POST' }).catch(() => {});
    }
  }
  if (state.chatAbort) state.chatAbort.abort();
  if (state.mediaRecorder && state.mediaRecorder.state === 'recording') {
    // Odpoj onstop → finishRecording() nesubmitne zahozenou nahrávku
    // a nespustí nový turn po tom, co uživatel mačkne Stop.
    try { state.mediaRecorder.onstop = null; } catch {}
    stopRecording();
    state.recordedChunks = [];
    stopMicTracks();
    avatar.setAnalyser(null);
  }
  try { audioEl.pause(); audioEl.currentTime = 0; } catch {}
  if (ctx) finishTurnCtx(ctx);
  else {
    avatar.setAnalyser(null);
    setPhase('idle');
  }
}

// Audio element: jeden globální `ended` listener - řetězí přehrávání.
audioEl.addEventListener('ended', () => {
  const ctx = state.turnCtx;
  if (!ctx) return;
  ctx.audioPlaying = false;
  if (ctx.canceled) { maybeFinishTurnCtx(ctx); return; }
  if (ctx.audioQueue.length > 0) {
    playNextInCtx(ctx);
  } else {
    maybeFinishTurnCtx(ctx);
  }
});

// ──────────── Persistence
function persistMessages() {
  try {
    const trimmed = state.messages.slice(-20); // posledních 20 turnů
    localStorage.setItem('messages', JSON.stringify(trimmed));
  } catch {}
}
function restoreMessages() {
  try {
    const raw = localStorage.getItem('messages');
    if (!raw) return;
    const arr = JSON.parse(raw);
    if (!Array.isArray(arr)) return;
    state.messages = arr;
    for (const m of arr) {
      if (m.role === 'user' || m.role === 'assistant') {
        // Tool calls (assistant s tool_calls) ukážeme jen jako bublinu -
        // pro Phase 1 nepřehráváme tool karty z historie. Detailní replay
        // přidáme později, až bude tool history rich (Phase 2+).
        const content = typeof m.content === 'string' ? m.content : '';
        if (content || m.role === 'user') {
          addMessage(m.role, content);
        }
      }
    }
  } catch {}
}

// ──────────── Mode toggle (chat → agent → claude → chat)
function applyMode(mode) {
  if (!['chat', 'agent', 'claude'].includes(mode)) mode = 'chat';
  state.mode = mode;
  localStorage.setItem('mode', mode);
  if (modeToggle) {
    modeToggle.dataset.mode = mode;
    const label = modeToggle.querySelector('.mode-chip-label');
    if (label) label.textContent = mode;
    const titles = {
      chat: 'chat mode (klik → agent)',
      agent: 'agent mode (klik → claude)',
      claude: 'claude mode (klik → chat)',
    };
    modeToggle.title = titles[mode];
  }
  document.body.dataset.mode = mode;
  // Claude model selector + permission badge - jen v claude mode
  const modelSelect = document.getElementById('claude-model-select');
  const permBadge = document.getElementById('claude-perm-badge');
  if (modelSelect) {
    modelSelect.hidden = mode !== 'claude';
  }
  if (permBadge) {
    permBadge.hidden = mode !== 'claude';
  }
  if (mode === 'claude') {
    refreshClaudePermBadge();
  }
}

function toggleMode() {
  const next = { chat: 'agent', agent: 'claude', claude: 'chat' }[state.mode] || 'chat';
  applyMode(next);
  playModeBeep(next);
}

// Refresh permission badge - poll server pro current state.
async function refreshClaudePermBadge() {
  const badge = document.getElementById('claude-perm-badge');
  if (!badge) return;
  try {
    const r = await fetch('/api/claude_ui_state');
    if (!r.ok) return;
    const state = await r.json();
    const perm = state.permission_mode || 'consult';
    badge.dataset.perm = perm;
    badge.textContent = perm === 'edit' ? '✏️ edit allowed' : '🔒 read-only';
    badge.title = perm === 'edit'
      ? `editace povolena (approved ${new Date((state.approved_at || 0) * 1000).toLocaleTimeString()})`
      : 'jen čtení; pro editaci napiš "ano povoluju" + akci';
  } catch (e) {
    console.warn('claude perm badge refresh failed', e);
  }
}

// Voice/text intent: rozpozná příkaz typu "agent mód", "přepni do chatu" atd.
// Vrací "agent" | "chat" | "claude" | null. Match je úmyslně přísný.
const _RE_INTENT_AGENT = /^\s*(?:p(?:ř|r)epni(?:\s+(?:do|na))?\s+|aktivuj\s+|spus(?:t|ť)\s+|zapni\s+|jdi\s+do\s+)?(?:agent(?:n(?:í|i))?(?:[\s-]*(?:m(?:ó|o)d|m(?:ó|o)du|re(?:ž|z)im(?:u)?|mode))?|agent[au]?)\s*\.?\s*$/i;
const _RE_INTENT_CHAT = /^\s*(?:p(?:ř|r)epni(?:\s+(?:do|na|zp(?:ě|e)t\s+do))?\s+|zp(?:ě|e)t\s+(?:do|na)\s+|jdi\s+(?:do|zp(?:ě|e)t\s+do)\s+)?(?:chat[auem]?(?:[\s-]*(?:m(?:ó|o)d|m(?:ó|o)du|re(?:ž|z)im(?:u)?|mode))?|norm(?:á|a)ln(?:í|i)(?:[\s-]*(?:m(?:ó|o)d|re(?:ž|z)im))?)\s*\.?\s*$/i;
// Claude mode switch: "přepni na claude", "použij opus/sonnet/haiku", "spusť claude"
const _RE_INTENT_CLAUDE = /^\s*(?:p(?:ř|r)epni(?:\s+(?:do|na))?\s+|pou(?:ž|z)ij\s+|spus(?:t|ť)\s+|aktivuj\s+|jdi\s+do\s+)?(?:claude|claud[au]?|opus[au]?|sonnet[au]?|sonet[au]?|haik[uau]?)\s*(?:m(?:ó|o)d[au]?)?\s*\.?\s*$/i;
// Detekce konkrétního modelu pokud user zmínil
const _RE_CLAUDE_OPUS = /\bopus\w*\b/i;
const _RE_CLAUDE_SONNET = /\b(?:sonnet|sonet)\w*\b/i;
const _RE_CLAUDE_HAIKU = /\bhaik\w*\b/i;

function tryModeSwitchIntent(text) {
  if (!text) return null;
  if (_RE_INTENT_AGENT.test(text)) return 'agent';
  if (_RE_INTENT_CHAT.test(text)) return 'chat';
  if (_RE_INTENT_CLAUDE.test(text)) return 'claude';
  return null;
}

function detectClaudeModelFromText(text) {
  if (!text) return null;
  if (_RE_CLAUDE_OPUS.test(text)) return 'opus';
  if (_RE_CLAUDE_SONNET.test(text)) return 'sonnet';
  if (_RE_CLAUDE_HAIKU.test(text)) return 'haik' && 'haiku';
  return null;
}

// Provede přepnutí na základě voice/text intentu. Vrátí true pokud
// se přepnulo (caller pak NEPOSÍLÁ message do LLM).
function handleModeSwitchIntent(userText) {
  const target = tryModeSwitchIntent(userText);
  if (!target) return false;
  // Pokud claude target + user řekl specific model, uložit
  if (target === 'claude') {
    const modelHint = detectClaudeModelFromText(userText);
    if (modelHint) {
      const sel = document.getElementById('claude-model-select');
      if (sel) sel.value = modelHint;
      localStorage.setItem('claudeModel', modelHint);
    }
  }
  if (state.mode !== target) {
    applyMode(target);
    playModeBeep(target);
  }
  // Echo do UI ať vidíš co se stalo. Žádné LLM volání, žádný state.messages push.
  addMessage('user', userText);
  const replies = {
    agent: 'Přepnuto do agent módu.',
    chat: 'Přepnuto do chat módu.',
    claude: 'Přepnuto do Claude módu.',
  };
  const reply = replies[target];
  addMessage('assistant', reply);
  // Voice response: pokud máš TTS zapnuté a vstup byl hlasový, ozvi se.
  if (state.voiceEnabled && state.inputMode === 'mic') {
    try {
      const u = new SpeechSynthesisUtterance(reply);
      u.lang = 'cs-CZ';
      u.rate = 1.05;
      window.speechSynthesis.speak(u);
    } catch {}
  }
  setPhase('idle');
  return true;
}

// Web Audio API - krátký tón při přepnutí mode. Agent = vyšší (880 Hz, "up"),
// chat = nižší (440 Hz, "down"). User-gesture safe (toggleMode je vždy z kliku).
let _beepCtx = null;
function playModeBeep(mode) {
  try {
    const AC = window.AudioContext || window.webkitAudioContext;
    if (!AC) return;
    _beepCtx = _beepCtx || new AC();
    const ctx = _beepCtx;
    if (ctx.state === 'suspended') ctx.resume();
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = 'sine';
    const FREQS = { chat: 440, agent: 880, claude: 660 };
    osc.frequency.value = FREQS[mode] || 440;
    osc.connect(gain).connect(ctx.destination);
    const now = ctx.currentTime;
    gain.gain.setValueAtTime(0.0001, now);
    gain.gain.exponentialRampToValueAtTime(0.18, now + 0.01);
    gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.18);
    osc.start(now);
    osc.stop(now + 0.2);
  } catch {}
}

// Phase 9.3: Audio filler - krátké rotující CZ fráze přes prohlížečové
// speechSynthesis. Throttled aby série rychlých tool callů nezahltila audio.
const _FILLER_PHRASES = ['Moment.', 'Hledám.', 'Pracuju na tom.', 'Chvilku.'];
let _fillerLastAt = 0;
let _fillerIdx = 0;
function playAudioFiller() {
  try {
    const now = Date.now();
    // Throttle: max jeden filler za 4s, ať to není opakované drmolení.
    if (now - _fillerLastAt < 4000) return;
    _fillerLastAt = now;
    if (!('speechSynthesis' in window)) return;
    // Pokud běží hlavní TTS audio, neskákej do něj.
    const mainAudio = document.getElementById('tts-audio');
    if (mainAudio && !mainAudio.paused && !mainAudio.ended) return;
    const phrase = _FILLER_PHRASES[_fillerIdx % _FILLER_PHRASES.length];
    _fillerIdx++;
    const u = new SpeechSynthesisUtterance(phrase);
    u.lang = 'cs-CZ';
    u.rate = 1.05;
    u.volume = 0.6;
    // Vyber CZ hlas pokud existuje, jinak default.
    const voices = window.speechSynthesis.getVoices();
    const cs = voices.find(v => /^cs/i.test(v.lang));
    if (cs) u.voice = cs;
    window.speechSynthesis.cancel(); // zruš případné staré utterance
    window.speechSynthesis.speak(u);
  } catch {}
}

if (modeToggle) {
  modeToggle.addEventListener('click', toggleMode);
}

// Claude model selector change → persist + send v dalším turnu
const _claudeModelSel = document.getElementById('claude-model-select');
if (_claudeModelSel) {
  const saved = localStorage.getItem('claudeModel');
  if (saved && ['opus', 'sonnet', 'haiku'].includes(saved)) {
    _claudeModelSel.value = saved;
  }
  _claudeModelSel.addEventListener('change', () => {
    localStorage.setItem('claudeModel', _claudeModelSel.value);
  });
}

// Click on permission badge → fetch + reset prompt
const _claudePermBadge = document.getElementById('claude-perm-badge');
if (_claudePermBadge) {
  _claudePermBadge.addEventListener('click', async () => {
    if (_claudePermBadge.dataset.perm === 'edit') {
      // Allow user to revoke edit permission (= reset to consult)
      if (confirm('Zrušit oprávnění k editaci? Aktuální Claude session bude ukončena (permission_mode je immutable per process).')) {
        await fetch('/api/claude_ui_state/reset', { method: 'POST' });
        refreshClaudePermBadge();
      }
    } else {
      alert('Pro povolení editace napiš/řekni "ano povoluju" + tvoji akci v dalším dotazu.');
    }
  });
}

// ──────────── Tool cards (agent mode)
function summarizeArgs(args) {
  if (!args || typeof args !== 'object') return '';
  const entries = Object.entries(args);
  if (entries.length === 0) return '';
  const parts = entries.slice(0, 3).map(([k, v]) => {
    let val = typeof v === 'string' ? v : JSON.stringify(v);
    if (val && val.length > 60) val = val.slice(0, 57) + '…';
    return `${k}=${val}`;
  });
  if (entries.length > 3) parts.push('…');
  return parts.join('  ');
}

function appendToolCard(ev) {
  clearWelcome();
  // Pokud běží streaming assistant bubble a je prázdná, odstraň ji -
  // jinak vznikne "ghost" bublina mezi userem a tool kartou.
  if (state.currentAssistantEl && !state.assistantBuffer.trim()) {
    state.currentAssistantEl.remove();
    state.currentAssistantEl = null;
  } else if (state.currentAssistantEl) {
    // Bublina měla text - finalizuj a začni novou až přijdou další text deltas.
    state.currentAssistantEl.classList.remove('streaming');
    state.currentAssistantEl = null;
    state.assistantBuffer = '';
  }
  const el = document.createElement('div');
  el.className = 'tool-card';
  el.dataset.tcid = ev.id;
  el.dataset.status = 'running';
  el.innerHTML = `
    <div class="tool-card-header">
      <span class="tool-card-icon">⚙</span>
      <span class="tool-card-name"></span>
      <span class="tool-card-summary"></span>
      <span class="tool-card-status">running…</span>
    </div>
    <details class="tool-card-details">
      <summary>detaily</summary>
      <pre class="tool-card-args"></pre>
      <pre class="tool-card-result" hidden></pre>
    </details>
  `;
  el.querySelector('.tool-card-name').textContent = ev.name;
  el.querySelector('.tool-card-summary').textContent = summarizeArgs(ev.args || {});
  el.querySelector('.tool-card-args').textContent = JSON.stringify(ev.args || {}, null, 2);
  transcript.appendChild(el);
  transcript.scrollTop = transcript.scrollHeight;
  return el;
}

function findToolCard(tcid) {
  return transcript.querySelector(`.tool-card[data-tcid="${CSS.escape(tcid)}"]`);
}

function setToolCardAwaiting(tcid) {
  const card = findToolCard(tcid);
  if (!card) return;
  card.dataset.status = 'awaiting';
  card.querySelector('.tool-card-status').textContent = 'čeká na schválení…';
}

function fillToolResult(ev) {
  const card = findToolCard(ev.id);
  if (!card) return;
  card.dataset.status = ev.ok ? 'done' : (ev.denied ? 'denied' : 'error');
  card.querySelector('.tool-card-status').textContent =
    ev.ok ? 'done' : (ev.denied ? 'denied' : 'error');
  let resultText = ev.content || '';
  try {
    const parsed = JSON.parse(resultText);
    resultText = JSON.stringify(parsed, null, 2);
  } catch {
    // Plain text - leave as-is.
  }
  const pre = card.querySelector('.tool-card-result');
  pre.textContent = resultText;
  pre.hidden = false;
}

// Limit progress eventů zobrazených v UI per tool card. Bez něj by ask_claude
// s desítkami tool_uses zaplnil DOM. Posledních N stačí pro orientaci.
const _PROGRESS_LOG_MAX = 12;

const _STAGE_ICONS = {
  started: '▶',       // ▶
  thinking: '…',      // …
  tool_use: '⚙',      // ⚙
  tool_result: '✓',   // ✓
  text: '✍',          // ✍
  cost: '$',
};

/** Update progress activity log na tool kartě podle tool_progress eventu.
 * Místo overwriting textu udržujeme scrolling log posledních N eventů,
 * každý s ikonou stage + jeho message/tool_name. User vidí kontinuálně co
 * Claude dělá (Read X, Edit Y, Bash Z, ...). */
function updateToolProgress(ev) {
  const card = findToolCard(ev.tool_call_id);
  if (!card) return;
  let logEl = card.querySelector('.tool-card-progress-log');
  if (!logEl) {
    logEl = document.createElement('div');
    logEl.className = 'tool-card-progress-log';
    // Vložit pod status, před result.
    const status = card.querySelector('.tool-card-status');
    if (status && status.parentNode) {
      status.parentNode.insertBefore(logEl, status.nextSibling);
    } else {
      card.appendChild(logEl);
    }
  }
  const p = ev.payload || {};
  const stage = p.stage || 'work';
  const icon = _STAGE_ICONS[stage] || '•';  // bullet fallback

  // Sestav label: stage + message/tool_name detail
  let detail = '';
  if (stage === 'tool_use') {
    // message obsahuje "Read /file.py" nebo "Bash: ls -la" etc.
    detail = p.message || p.tool_name || '';
  } else if (stage === 'tool_result') {
    detail = (p.ok === false) ? 'tool selhal' : (p.message || 'tool OK');
  } else if (stage === 'text') {
    const t = (p.text || '').replace(/\s+/g, ' ').slice(0, 80);
    detail = t ? `"${t}${(p.text || '').length > 80 ? '…' : ''}"` : '';
  } else if (stage === 'cost') {
    const cost = p.cost_usd != null ? `$${Number(p.cost_usd).toFixed(4)}` : '?';
    const dur = p.duration_ms != null ? `${(p.duration_ms / 1000).toFixed(1)}s` : '?';
    detail = `${cost} · ${dur}`;
  } else {
    detail = p.message || '';
  }

  const row = document.createElement('div');
  row.className = 'tool-card-progress-row';
  row.dataset.stage = stage;
  const stageEl = document.createElement('span');
  stageEl.className = 'tool-card-progress-icon';
  stageEl.textContent = icon;
  const labelEl = document.createElement('span');
  labelEl.className = 'tool-card-progress-text';
  labelEl.textContent = detail ? `${stage}: ${detail}` : stage;
  row.appendChild(stageEl);
  row.appendChild(labelEl);
  logEl.appendChild(row);

  // Trim na posledních N
  while (logEl.childElementCount > _PROGRESS_LOG_MAX) {
    logEl.removeChild(logEl.firstChild);
  }
  // Autoscroll na nejnovější
  logEl.scrollTop = logEl.scrollHeight;
}

/** Pokud tool_result je z ask_claude, vykresli pod kartu speciální blok
 * s celým Claudovým textem (ne procesován TTS). Jinak nic. */
function maybeRenderClaudeResultCard(ev) {
  if (!ev.ok || !ev.content) return;
  let payload;
  try {
    payload = JSON.parse(ev.content);
  } catch { return; }
  // Detekce ask_claude payload tvaru: má `text` + `model` + `mode` keys.
  if (!payload || typeof payload !== 'object') return;
  if (typeof payload.text !== 'string' || !payload.model || !payload.mode) return;

  const card = findToolCard(ev.id);
  if (!card) return;

  // Plný Claudův text + metadata (cost, duration, model, session, tool_uses).
  const wrap = document.createElement('div');
  wrap.className = 'claude-result-card';
  wrap.dataset.mode = payload.mode;

  const head = document.createElement('div');
  head.className = 'claude-result-head';
  const cost = payload.total_cost_usd != null
    ? `$${payload.total_cost_usd.toFixed(4)}`
    : '-';
  const dur = payload.duration_ms != null ? `${(payload.duration_ms / 1000).toFixed(1)}s` : '-';
  const tools = (payload.tool_uses || []).join(', ') || '-';
  head.innerHTML = `
    <span class="claude-result-badge">🤖 Claude · ${escapeHtml(payload.model)} · ${escapeHtml(payload.mode)}</span>
    <span class="claude-result-meta">cost ${cost} · ${dur} · tools: ${escapeHtml(tools)}</span>
  `;
  wrap.appendChild(head);

  const body = document.createElement('div');
  body.className = 'claude-result-body';
  body.textContent = payload.text;
  wrap.appendChild(body);

  // Vložit pod tool kartu.
  if (card.parentNode) {
    card.parentNode.insertBefore(wrap, card.nextSibling);
  }
}

function escapeHtml(s) {
  return String(s ?? '').replace(/[&<>"']/g, c => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  }[c]));
}

// ──────────── Approval modal
// Approval phrase config - fetchnuto z `/api/approval_config` při startu UI,
// fallback constants (musí být v sync s voice/agent/config.py). Server je
// authoritative; tyto fallbacky kryjí init před prvním fetchem.
let _approvalConfig = {
  approve_phrases: ['ano', 'jo', 'ok', 'okej', 'okay', 'povol', 'povoluju',
                    'jasně', 'jasne', 'fajn', 'yes'],
  deny_phrases: ['ne', 'stop', 'zruš', 'zrus', 'nepovoluju', 'nechci', 'no'],
  destructive_phrase: 'ano povoluju',
};
fetch('/api/approval_config').then(r => r.ok ? r.json() : null).then(cfg => {
  if (cfg && cfg.approve_phrases && cfg.deny_phrases && cfg.destructive_phrase) {
    _approvalConfig = cfg;
  }
}).catch(() => {/* keep fallback */});

// Normalizace transkripce z Whisperu pro phrase match: lowercase, strip
// interpunkce, collapse whitespace. ZACHOVÁVÁ diakritiku ("zruš", "jasně").
function _normalizeVoicePhrase(text) {
  return (text || '')
    .toLowerCase()
    .replace(/[.,!?;:()"„""'`]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

// Najde whole-word match v normalized textu (boundary = whitespace nebo začátek/konec).
function _containsWord(normalized, phrase) {
  const padded = ` ${normalized} `;
  return padded.indexOf(` ${phrase} `) !== -1;
}

// Najde substring (pro multi-word phrase typu "ano povoluju").
function _containsPhrase(normalized, phrase) {
  return normalized.includes(phrase);
}

/**
 * Mapuje voice/text input na approval rozhodnutí. Konflikt (approve i deny)
 * vyhrává DENY (codex: safer). Destructive vyžaduje phrase-match, ne intent.
 *
 * @return {{decision: 'approve'|'deny', phrase: string} | null} null = no match
 */
function classifyApprovalUtterance(text, requiresExplicit) {
  const norm = _normalizeVoicePhrase(text);
  if (!norm) return null;

  const hasDeny = _approvalConfig.deny_phrases.some(p => _containsWord(norm, p));
  if (hasDeny) return { decision: 'deny', phrase: '' };

  if (requiresExplicit) {
    // Destructive: STRICT match - normalized text musí být přesně rovný
    // canonical frázi. Codex audit HIGH: contains/substring chytí false
    // positive jako "tak jsem řekl 'ano povoluju' nikdy", což by povolilo
    // destruktivní akci ze špatného kontextu. Pokud whisper přidá prefix
    // ("říkám ano povoluju"), no-match → fallback do input fieldu pro
    // ruční kontrolu, ne automatický approve.
    if (norm === _approvalConfig.destructive_phrase) {
      // Posíláme SERVERY očekávanou canonical formu, ne user text.
      return { decision: 'approve', phrase: _approvalConfig.destructive_phrase };
    }
    return null;
  }
  // Non-destructive: intent based, libovolná approve fráze.
  const hasApprove = _approvalConfig.approve_phrases.some(p => _containsWord(norm, p));
  if (hasApprove) return { decision: 'approve', phrase: '' };
  return null;
}

let _pendingApproval = null;  // { finish(result), requiresExplicit, turnId, approvalId }

function showApprovalModal(ev, turnId) {
  return new Promise((resolve) => {
    approvalSummary.textContent = ev.summary || `${ev.tool}`;
    const risk = ev.risk || 'low';
    approvalRisk.dataset.risk = risk;
    approvalRisk.textContent = `risk: ${risk}`;
    approvalArgs.textContent = JSON.stringify(ev.args || {}, null, 2);
    approvalExplicit.hidden = !ev.requires_explicit;
    approvalPhraseInput.value = '';

    const requiresExplicit = !!ev.requires_explicit;
    approvalAllowBtn.disabled = requiresExplicit;

    const onAllow = () => {
      let phrase = '';
      if (requiresExplicit) {
        phrase = (approvalPhraseInput.value || '').trim().toLowerCase();
        if (phrase !== 'ano povoluju') return;  // safety: button should be disabled
      }
      cleanup();
      resolve({ decision: 'approve', phrase });
    };
    const onDeny = () => { cleanup(); resolve({ decision: 'deny', phrase: '' }); };
    const onPhraseInput = () => {
      const phrase = (approvalPhraseInput.value || '').trim().toLowerCase();
      approvalAllowBtn.disabled = requiresExplicit && phrase !== 'ano povoluju';
    };
    const onCancel = (e) => { e.preventDefault(); cleanup(); resolve({ decision: 'deny', phrase: '' }); };

    // Mic listener cleanup se přiřadí později (po jeho registraci níž).
    let onMicClick = null;
    function cleanup() {
      approvalAllowBtn.removeEventListener('click', onAllow);
      approvalDenyBtn.removeEventListener('click', onDeny);
      approvalPhraseInput.removeEventListener('input', onPhraseInput);
      approvalModal.removeEventListener('cancel', onCancel);
      approvalForm?.removeEventListener('submit', onFormSubmit);
      if (onMicClick && approvalMicBtn) {
        approvalMicBtn.removeEventListener('click', onMicClick);
        approvalMicBtn.classList.remove('recording');
      }
      // Codex audit HIGH: pokud user kliknul Allow/Deny/Esc během approval
      // recordingu, modal zavřeme; mic by ale jinak doběhl (push-to-talk)
      // a triggernul finishRecording() už mimo intercept, NEBO by zůstal
      // viset s otevřenými tracks. Unconditional discard pokrývá i stav
      // race kdy stop() už proběhl ale onstop event čeká ve frontě
      // (mediaRecorder.state === 'inactive', ale onstop dorazí). Snap=null
      // garantuje, že iter-5 guard ve `finishRecording()` cokoli protlačeného
      // taky zahodí.
      if (state.mediaRecorder) {
        try { state.mediaRecorder.onstop = null; } catch {}
        if (state.mediaRecorder.state === 'recording') {
          try { state.mediaRecorder.stop(); } catch {}
        }
        state.recordedChunks = [];
        stopMicTracks();
        avatar.setAnalyser(null);
        if (state.vad) { state.vad.stop(); state.vad = null; }
      }
      state.recordingApprovalSnap = null;
      if (approvalModal.open) approvalModal.close();
      _pendingApproval = null;
    }

    // Enter v phrase inputu spustí form submit. Bez explicitního handleru by
    // default dialog form behavior modal zavřel bez resolve → promise hangne,
    // turn se zasekne. Codex audit HIGH: preventDefault + delegace na onAllow.
    const onFormSubmit = (e) => {
      e.preventDefault();
      onAllow();  // honoruje requires_explicit + phrase match check uvnitř
    };

    approvalAllowBtn.addEventListener('click', onAllow);
    approvalDenyBtn.addEventListener('click', onDeny);
    approvalPhraseInput.addEventListener('input', onPhraseInput);
    approvalModal.addEventListener('cancel', onCancel);
    approvalForm?.addEventListener('submit', onFormSubmit);

    // Mic UVNITŘ modalu - `<dialog>::showModal()` udělá zbytek stránky inert,
    // takže mic v hlavním composeru by nešel kliknout. Tlačítko v modalu má
    // vlastní recording lifecycle; finishRecording() routuje přes
    // _pendingApproval do classifyApprovalUtterance.
    onMicClick = async () => {
      if (state.phase === 'recording') {
        stopRecording();
        return;
      }
      if (state.phase === 'approval' || state.phase === 'idle') {
        approvalMicBtn?.classList.add('recording');
        try {
          await beginRecording({ auto: vadToggle.checked });
        } catch (e) {
          approvalMicBtn?.classList.remove('recording');
          showError(`Mikrofon: ${e.message}`);
        }
      }
    };
    if (approvalMicBtn) approvalMicBtn.addEventListener('click', onMicClick);

    // `finish` umožní voice/text handlerům vyřešit modal bez prokliku tlačítek.
    // Cleanup nuluje _pendingApproval atomicky před resolve (žádný double-submit).
    const finish = (result) => { cleanup(); resolve(result); };
    _pendingApproval = {
      finish,
      requiresExplicit,
      turnId,
      approvalId: ev.approval_id,
    };

    approvalModal.showModal();
    setTimeout(() => approvalDenyBtn.focus(), 0);
  });
}

async function sendApprovalDecision(turnId, approvalId, decision, phrase) {
  const body = { decision };
  if (phrase) body.phrase = phrase;
  try {
    const r = await fetch(`/api/turn/${turnId}/approval/${approvalId}`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!r.ok) {
      // Server odmítl (např. 400 destructive bez phrase, 409 already canceled).
      // Vrátíme false ať volající modal znovu zobrazí, případně cancel-ne turn.
      console.warn('approval POST rejected', r.status);
      showError(`Schválení odmítnuto serverem (${r.status}).`);
      return false;
    }
    return true;
  } catch (e) {
    console.warn('approval POST failed:', e);
    showError('Chyba sítě při odeslání schválení.');
    return false;
  }
}

function clearConversation() {
  // Pokud běží turn, nejdřív ho přerušíme - jinak by stream dál zapisoval
  // do detached assistantEl a výsledek by se smazal, ale server by pořád
  // generoval TTS a drze dohráván do `audioEl`. Stop = single source of
  // truth pro zrušení pipeline.
  if (state.phase !== 'idle' && state.phase !== 'error') {
    stopEverything();
  }
  state.messages = [];
  state.currentAssistantEl = null;
  state.assistantBuffer = '';
  localStorage.removeItem('messages');
  transcript.innerHTML = `
    <div class="welcome">
      <p class="welcome-title">napiš zprávu nebo zmáčkni mikrofon</p>
      <p class="welcome-sub">mic = hlasová odpověď · text = textová odpověď · stop přeruší vše</p>
    </div>`;
}

// ──────────── Text composer
function autoGrowTextarea() {
  textInput.style.height = 'auto';
  const max = parseInt(getComputedStyle(textInput).maxHeight, 10) || 160;
  textInput.style.height = Math.min(textInput.scrollHeight, max) + 'px';
}

async function handleTextSubmit() {
  const text = textInput.value.trim();
  if (!text) return;
  // 'approval' phase: text submit povolen pro textovou approval frázi
  // (intercept přebírá _pendingApproval handler níž).
  if (state.phase !== 'idle' && state.phase !== 'error' && state.phase !== 'approval') return;

  // Předběžně resume AudioContext - pokud bude TTS hrát, autoplay policy
  // vyžaduje gesto před prvním AudioContext.resume(). Submit click je
  // platné gesto.
  ensureAudioCtx().catch(() => {});

  textInput.value = '';
  autoGrowTextarea();

  state.inputMode = 'text';

  // Approval intercept: pokud běží modal čekající na rozhodnutí, route text
  // tam místo do /api/turn (jinak by nový turn rozbil starý waiting).
  if (_pendingApproval !== null) {
    handleVoiceApproval(text);
    return;
  }

  // Text intent: "agent mód" / "chat mód" → přepne mode bez LLM kola.
  if (handleModeSwitchIntent(text)) return;

  addMessage('user', text);
  state.messages.push({ role: 'user', content: text });

  await runTurn();
}

function clearError() {
  errorToast.hidden = true;
  errorToast.classList.remove('sticky');
  clearTimeout(errorTimer);
}

errorToast.addEventListener('click', () => {
  if (errorToast.classList.contains('sticky')) clearError();
});

// ──────────── Events
micBtn.addEventListener('click', async () => {
  if (state.phase === 'idle') {
    await beginRecording({ auto: vadToggle.checked });
  } else if (state.phase === 'recording') {
    stopRecording();
  }
});

stopBtn.addEventListener('click', stopEverything);
clearBtn.addEventListener('click', clearConversation);

// ──────────── Settings panel (slide-over)
//
// A11y kontrakt:
//  - role="dialog" + aria-modal + aria-labelledby (v HTML)
//  - pozadí (topbar/stage/bottombar) dostane `inert` při open → screen reader
//    i klávesnice ho ignorují, nelze tam Tab-outem odejít
//  - focus trap: Tab/Shift+Tab cyklicky přes focusable elementy panelu
//  - focus se po zavření vrací na element, který dialog otevřel
let settingsLastFocus = null;

// Všechny interaktivní oblasti MIMO panel. Při otevření je přetížíme `inert`.
const SETTINGS_INERT_SELECTORS = ['.topbar', '.stage', '.bottombar', '.error-toast'];

function getSettingsInertTargets() {
  return SETTINGS_INERT_SELECTORS
    .map((s) => document.querySelector(s))
    .filter((el) => el);
}

// Focusable prvky uvnitř panelu - pouze nedisabled, visible. Neviditelné inputy
// (ref-field když hidden) filtrujeme přes offsetParent.
const FOCUSABLE_SEL = 'a[href], button:not([disabled]), textarea:not([disabled]), input:not([disabled]), select:not([disabled]), [tabindex]:not([tabindex="-1"])';

function getSettingsFocusable() {
  return Array.from(settingsPanel.querySelectorAll(FOCUSABLE_SEL))
    .filter((el) => !el.hasAttribute('disabled') && el.offsetParent !== null);
}

function onSettingsKeydown(e) {
  if (e.key === 'Escape') {
    e.preventDefault();
    closeSettings();
    return;
  }
  if (e.key !== 'Tab') return;
  const focusable = getSettingsFocusable();
  if (focusable.length === 0) {
    e.preventDefault();
    return;
  }
  const first = focusable[0];
  const last = focusable[focusable.length - 1];
  const active = document.activeElement;
  if (e.shiftKey && active === first) {
    e.preventDefault();
    last.focus();
  } else if (!e.shiftKey && active === last) {
    e.preventDefault();
    first.focus();
  } else if (!settingsPanel.contains(active)) {
    // Focus byl mimo panel (např. po attach inertu) → přitáhni zpět.
    e.preventDefault();
    first.focus();
  }
}

function openSettings() {
  settingsLastFocus = document.activeElement;
  settingsBackdrop.hidden = false;
  settingsPanel.hidden = false;
  settingsPanel.setAttribute('aria-hidden', 'false');
  settingsBtn.setAttribute('aria-expanded', 'true');
  for (const el of getSettingsInertTargets()) el.setAttribute('inert', '');
  // rAF aby transition startla z initial stavu
  requestAnimationFrame(() => {
    settingsBackdrop.classList.add('open');
    settingsPanel.classList.add('open');
  });
  settingsPanel.addEventListener('keydown', onSettingsKeydown);
  settingsClose.focus();
}

function closeSettings() {
  settingsBackdrop.classList.remove('open');
  settingsPanel.classList.remove('open');
  settingsPanel.setAttribute('aria-hidden', 'true');
  settingsBtn.setAttribute('aria-expanded', 'false');
  settingsPanel.removeEventListener('keydown', onSettingsKeydown);
  for (const el of getSettingsInertTargets()) el.removeAttribute('inert');
  // Počkáme na transition, pak hidden (odstraní z tab orderu).
  setTimeout(() => {
    if (!settingsPanel.classList.contains('open')) {
      settingsPanel.hidden = true;
      settingsBackdrop.hidden = true;
    }
  }, 340);
  if (settingsLastFocus && typeof settingsLastFocus.focus === 'function') {
    settingsLastFocus.focus();
  }
}

settingsBtn.addEventListener('click', () => {
  if (settingsPanel.classList.contains('open')) closeSettings();
  else openSettings();
});
settingsClose.addEventListener('click', closeSettings);
settingsBackdrop.addEventListener('click', closeSettings);

// Composer - form submit (Enter bez shift)
composer.addEventListener('submit', (e) => {
  e.preventDefault();
  handleTextSubmit();
});
textInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    handleTextSubmit();
  }
});
textInput.addEventListener('input', () => {
  autoGrowTextarea();
  updateSendButton();
});

modelSelect.addEventListener('change', () => {
  localStorage.setItem('model', modelSelect.value);
  syncBrandModel();
  console.info(`[model] changed → ${modelSelect.value}`);
});
voiceSelect.addEventListener('change', () => localStorage.setItem('voice', voiceSelect.value));
refSelect.addEventListener('change', () => {
  localStorage.setItem('refExplicit', refSelect.value);
});
refClearBtn.addEventListener('click', () => {
  // Vrátí se k voice family režimu; explicit ref zapomene.
  localStorage.removeItem('refExplicit');
  refField.hidden = true;
});
fastToggle.addEventListener('change', () => localStorage.setItem('fast', fastToggle.checked ? '1' : '0'));
vadToggle.addEventListener('change', () => localStorage.setItem('vad', vadToggle.checked ? '1' : '0'));
function setVoiceEnabled(v) {
  state.voiceEnabled = v;
  ttsToggle.checked = v;
  ttsQuickToggle.checked = v;
  localStorage.setItem('tts', v ? '1' : '0');
}
ttsToggle.addEventListener('change', () => setVoiceEnabled(ttsToggle.checked));
ttsQuickToggle.addEventListener('change', () => setVoiceEnabled(ttsQuickToggle.checked));
streamToggle.addEventListener('change', () => {
  state.streamTTSEnabled = streamToggle.checked;
  localStorage.setItem('streamTTS', streamToggle.checked ? '1' : '0');
});
const ttsScopeSelect = $('tts-scope');
if (ttsScopeSelect) {
  ttsScopeSelect.value = state.ttsScope;
  ttsScopeSelect.addEventListener('change', () => {
    const v = ttsScopeSelect.value === 'off' ? 'off' : 'final';
    state.ttsScope = v;
    localStorage.setItem('ttsScope', v);
  });
}
langSelect.addEventListener('change', () => {
  state.langOverride = langSelect.value;
  localStorage.setItem('langOverride', state.langOverride);
  // Voice families se podle jazyka neorezávají - family sama obsahuje per-lang
  // varianty, backend vybere tu správnou. Refresh seznamu tu není třeba.
});

// Restore toggles
if (localStorage.getItem('fast') === '1') fastToggle.checked = true;
if (localStorage.getItem('vad') === '0') vadToggle.checked = false;
ttsToggle.checked = state.voiceEnabled;
ttsQuickToggle.checked = state.voiceEnabled;
streamToggle.checked = state.streamTTSEnabled;
langSelect.value = state.langOverride;

// Orb collapse - schová animovaného avatara do úzkého pruhu u levého kraje.
// ResizeObserver na canvasu se postará o resize WebGL viewportu během tranzice;
// fallback setTimeout pro jistotu po dokončení animace (320 ms).
function setOrbCollapsed(collapsed, { persist = true } = {}) {
  stageEl.classList.toggle('orb-collapsed', collapsed);
  orbToggle.title = collapsed ? 'zobrazit avatar' : 'skrýt avatar';
  if (persist) localStorage.setItem('orbCollapsed', collapsed ? '1' : '0');
  setTimeout(fitCanvas, 350);
}
if (localStorage.getItem('orbCollapsed') === '1') setOrbCollapsed(true, { persist: false });
orbToggle.addEventListener('click', () => {
  setOrbCollapsed(!stageEl.classList.contains('orb-collapsed'));
});

// Keyboard: space = start/stop recording - ale nikdy neukradne input v
// textarea/input/select/button/contenteditable (jinak by nešlo napsat mezeru,
// ani přepnout checkbox).
const INTERACTIVE_SEL = 'textarea,input,select,button,[contenteditable="true"]';
document.addEventListener('keydown', (e) => {
  if (e.code !== 'Space') return;
  if (e.target && typeof e.target.closest === 'function' && e.target.closest(INTERACTIVE_SEL)) return;
  e.preventDefault();
  if (state.phase === 'idle') micBtn.click();
  else if (state.phase === 'recording') stopRecording();
  else if (!stopBtn.hidden) stopEverything();
});

// Copy tlačítko v code block - delegovaný listener (funguje i na dynamicky
// vložené bloky z rerenderu během streamingu).
transcript.addEventListener('click', (e) => {
  const btn = e.target.closest('.code-copy');
  if (!btn) return;
  const pre = btn.closest('.code-block')?.querySelector('pre code');
  if (!pre) return;
  navigator.clipboard.writeText(pre.textContent).then(() => {
    btn.textContent = 'copied';
    btn.classList.add('copied');
    setTimeout(() => {
      btn.textContent = 'copy';
      btn.classList.remove('copied');
    }, 1500);
  }).catch(() => {
    btn.textContent = 'failed';
    setTimeout(() => (btn.textContent = 'copy'), 1500);
  });
});

// ──────────── Init (s comprehensive logging - bez něj UI selže tiše)
(async () => {
  logInit('init: start');
  try {
    logInit('init: configureMarked');
    configureMarked();

    logInit('init: autoGrowTextarea');
    autoGrowTextarea();

    logInit('init: setPhase(idle)');
    setPhase('idle');

    logInit('init: applyMode', state.mode);
    applyMode(state.mode);

    logInit('init: loadHealth');
    await loadHealth();
    logInit('init: loadHealth OK');

    logInit('init: loadModels + loadVoices parallel');
    await Promise.all([loadModels(), loadVoices()]);
    logInit('init: models+voices OK');

    logInit('init: restoreMessages');
    restoreMessages();

    logInit('init: COMPLETE');
  } catch (err) {
    logError('init', err);
  }
})();
