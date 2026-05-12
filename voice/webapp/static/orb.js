/**
 * GlowingOrb — default Avatar implementace.
 * WebGL2 fragment shader na fullscreen quad. Čtyři fáze crossfadeované přes
 * 320 ms. Reaguje na AnalyserNode amplitudu (listening/speaking).
 */
import { Avatar } from './avatar_api.js';

const VERT_SRC = `#version 300 es
in vec2 aPos;
out vec2 vUV;
void main() {
  vUV = aPos * 0.5 + 0.5;
  gl_Position = vec4(aPos, 0.0, 1.0);
}`;

const FRAG_SRC = `#version 300 es
precision highp float;
in vec2 vUV;
out vec4 outColor;

uniform vec2 uRes;
uniform float uTime;
uniform float uAmp;          // 0..1 smoothed amplitude
uniform vec4 uPhaseMix;      // weights (idle, listening, thinking, speaking), sum=1

// Palettes
const vec3 PAL_IDLE_A  = vec3(0.35, 0.12, 0.78);
const vec3 PAL_IDLE_B  = vec3(0.18, 0.55, 1.00);
const vec3 PAL_LIST_A  = vec3(0.18, 0.90, 0.55);
const vec3 PAL_LIST_B  = vec3(0.40, 1.00, 0.85);
const vec3 PAL_THINK_A = vec3(1.00, 0.22, 0.80);
const vec3 PAL_THINK_B = vec3(0.25, 0.85, 1.00);
const vec3 PAL_SPK_A   = vec3(1.00, 0.52, 0.15);
const vec3 PAL_SPK_B   = vec3(1.00, 0.85, 0.35);

// Hash + gradient noise + FBM (iq style)
vec2 hash2(vec2 p) {
  p = vec2(dot(p, vec2(127.1, 311.7)), dot(p, vec2(269.5, 183.3)));
  return fract(sin(p) * 43758.5453) * 2.0 - 1.0;
}
float noise(vec2 p) {
  vec2 i = floor(p);
  vec2 f = fract(p);
  vec2 u = f * f * (3.0 - 2.0 * f);
  return mix(
    mix(dot(hash2(i + vec2(0,0)), f - vec2(0,0)),
        dot(hash2(i + vec2(1,0)), f - vec2(1,0)), u.x),
    mix(dot(hash2(i + vec2(0,1)), f - vec2(0,1)),
        dot(hash2(i + vec2(1,1)), f - vec2(1,1)), u.x),
    u.y
  );
}
float fbm(vec2 p) {
  float v = 0.0;
  float a = 0.5;
  for (int i = 0; i < 5; i++) {
    v += a * noise(p);
    p *= 2.02;
    a *= 0.5;
  }
  return v;
}

// 2D rotation
mat2 rot(float a) { return mat2(cos(a), -sin(a), sin(a), cos(a)); }

vec3 phasePalette(float r, float angle, float t) {
  // Idle: slow radial breathing, blue-violet
  vec3 idle = mix(PAL_IDLE_A, PAL_IDLE_B, 0.5 + 0.5 * sin(t * 0.45 + r * 2.6));
  // Listening: green teal with radial sweep
  vec3 listen = mix(PAL_LIST_A, PAL_LIST_B, 0.5 + 0.5 * sin(t * 1.8 + r * 4.0 + angle * 0.6));
  // Thinking: iridescent rainbow rotating
  float iridescence = 0.5 + 0.5 * sin(angle * 3.0 + t * 2.2 + r * 5.0);
  vec3 think = mix(PAL_THINK_A, PAL_THINK_B, iridescence);
  // Speaking: amber/gold
  vec3 spk = mix(PAL_SPK_A, PAL_SPK_B, 0.5 + 0.5 * sin(t * 3.5 + r * 8.0));
  return idle * uPhaseMix.x + listen * uPhaseMix.y + think * uPhaseMix.z + spk * uPhaseMix.w;
}

void main() {
  vec2 uv = (vUV - 0.5) * 2.0;
  float aspect = uRes.x / uRes.y;
  uv.x *= aspect;
  // fit orb consistently even on wide canvases
  float scale = 1.0 / max(0.9, min(1.3, 1.0));
  uv *= scale;

  float t = uTime;
  float r = length(uv);
  float angle = atan(uv.y, uv.x);

  // Rotating noise domain for liquid look
  vec2 p1 = rot(t * 0.12) * uv * 2.2;
  vec2 p2 = rot(-t * 0.18) * uv * 4.5;
  float n = fbm(p1 + vec2(t * 0.25, 0.0));
  float n2 = fbm(p2 - vec2(0.0, t * 0.35));
  float noiseMix = n * 0.65 + n2 * 0.35;

  // Amplitude boost targets listen+speak phases
  float ampPhase = uPhaseMix.y + uPhaseMix.w;
  float amp = uAmp * ampPhase;

  // Surface displacement: makes the sphere "breathe" + react to audio
  float surfaceWobble = noiseMix * 0.09 + amp * 0.12;
  float thinkPulse = uPhaseMix.z * (0.5 + 0.5 * sin(t * 3.0)) * 0.04;
  float rDisp = r - surfaceWobble - thinkPulse;

  // Core sphere — soft edge
  float coreRadius = 0.52;
  float core = smoothstep(coreRadius + 0.12, coreRadius - 0.05, rDisp);

  // Outer glow halo — large, soft
  float glow = smoothstep(1.3, coreRadius, rDisp);
  glow = pow(glow, 2.4) * (0.55 + 0.45 * uPhaseMix.w);

  // Thinking: rotating conic shimmer inside the sphere
  float conic = 0.5 + 0.5 * sin(angle * 5.0 + t * 1.8 + noiseMix * 4.0);
  float thinkShine = conic * uPhaseMix.z * smoothstep(coreRadius, 0.0, r) * 0.45;

  // Listening: outer ring reactive to mic amplitude
  float listenRing = exp(-pow((r - coreRadius - 0.08 - amp * 0.05) * 10.0, 2.0));
  listenRing *= uPhaseMix.y * (0.3 + uAmp * 1.2);

  // Speaking: concentric ripples emanating outward, audio-driven
  float ripples = sin(r * 22.0 - t * 7.0) * 0.5 + 0.5;
  ripples *= exp(-pow((r - coreRadius - 0.15) * 4.5, 2.0));
  ripples *= uPhaseMix.w * (0.2 + uAmp * 1.6);

  // Total intensity
  float intensity = core + glow * 0.5 + thinkShine + listenRing + ripples;
  // Surface texture variation
  intensity *= 1.0 + noiseMix * 0.18;

  vec3 col = phasePalette(rDisp, angle, t) * intensity;

  // Center bloom kick
  col += vec3(0.25, 0.22, 0.38) * pow(max(0.0, 1.0 - r * 1.6), 4.0) * 0.6;

  // Gentle vignette to blend into background
  col *= 1.0 - smoothstep(1.1, 1.9, length(uv)) * 0.75;

  // Subtle chromatic shimmer for speaking phase
  col.r += uPhaseMix.w * uAmp * 0.08 * sin(t * 10.0 + r * 6.0);
  col.b += uPhaseMix.z * 0.08 * sin(t * 5.0 + angle * 3.0);

  // Tone/gamma
  col = col / (col + 1.0);         // Reinhard
  col = pow(col, vec3(1.0 / 1.05)); // tiny gamma bump

  outColor = vec4(col, 1.0);
}`;

const PHASES = ['idle', 'listening', 'thinking', 'speaking'];
const CROSSFADE_MS = 320;

function compileShader(gl, type, src) {
  const sh = gl.createShader(type);
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
    const info = gl.getShaderInfoLog(sh);
    gl.deleteShader(sh);
    throw new Error('Shader compile: ' + info);
  }
  return sh;
}

function linkProgram(gl, vs, fs) {
  const p = gl.createProgram();
  gl.attachShader(p, vs);
  gl.attachShader(p, fs);
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
    const info = gl.getProgramInfoLog(p);
    gl.deleteProgram(p);
    throw new Error('Program link: ' + info);
  }
  return p;
}

export class GlowingOrb extends Avatar {
  static meta = { id: 'orb', label: 'Zářící koule' };

  constructor() {
    super();
    this.gl = null;
    this.program = null;
    this.vao = null;
    this.uniforms = {};
    this.rafId = 0;
    this.startTime = performance.now();
    this.analyser = null;
    this.audioBuf = null;
    this.smoothedAmp = 0;

    // Phase weights (idle, listening, thinking, speaking)
    this.phaseWeights = [1, 0, 0, 0];
    this.phaseTargets = [1, 0, 0, 0];
    this.phaseTransitionStart = 0;
    this.phaseTransitionFrom = [1, 0, 0, 0];

    // prefers-reduced-motion: freezneme `uTime` → shader nebude běžet ambient
    // animace (noise, ripple, shimmer), jen crossfade mezi fázemi a amplituda
    // z mic zůstanou. Screen-reader/motion-sensitive užovatelé tak dostanou
    // klidný orb bez nutnosti WebGL vypínat úplně.
    this.mqReduced = window.matchMedia
      ? window.matchMedia('(prefers-reduced-motion: reduce)')
      : null;
  }

  attach(canvas) {
    this.canvas = canvas;
    const gl = canvas.getContext('webgl2', { antialias: true, alpha: false, premultipliedAlpha: false });
    if (!gl) throw new Error('WebGL2 není k dispozici');
    this.gl = gl;

    const vs = compileShader(gl, gl.VERTEX_SHADER, VERT_SRC);
    const fs = compileShader(gl, gl.FRAGMENT_SHADER, FRAG_SRC);
    this.program = linkProgram(gl, vs, fs);
    gl.deleteShader(vs);
    gl.deleteShader(fs);

    // Fullscreen quad
    const vao = gl.createVertexArray();
    gl.bindVertexArray(vao);
    const vbo = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
    gl.bufferData(
      gl.ARRAY_BUFFER,
      new Float32Array([-1, -1, 3, -1, -1, 3]),  // big triangle trick
      gl.STATIC_DRAW,
    );
    const loc = gl.getAttribLocation(this.program, 'aPos');
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
    this.vao = vao;

    this.uniforms = {
      uRes: gl.getUniformLocation(this.program, 'uRes'),
      uTime: gl.getUniformLocation(this.program, 'uTime'),
      uAmp: gl.getUniformLocation(this.program, 'uAmp'),
      uPhaseMix: gl.getUniformLocation(this.program, 'uPhaseMix'),
    };

    gl.clearColor(0, 0, 0, 0);
    this._loop();
  }

  detach() {
    cancelAnimationFrame(this.rafId);
    this.rafId = 0;
    const gl = this.gl;
    if (!gl) return;
    gl.deleteProgram(this.program);
    gl.deleteVertexArray(this.vao);
    this.gl = null;
  }

  resize(cssWidth, cssHeight, dpr) {
    if (!this.canvas) return;
    const w = Math.max(1, Math.floor(cssWidth * dpr));
    const h = Math.max(1, Math.floor(cssHeight * dpr));
    // Pozor: nastavení canvas.width / .height vždy vymaže WebGL drawing buffer,
    // i když je hodnota stejná (HTML spec). Měníme jen při skutečné změně,
    // jinak by rAF smyčka kreslila do plátna, které je hned vzápětí zase vymazáno.
    if (this.canvas.width !== w) this.canvas.width = w;
    if (this.canvas.height !== h) this.canvas.height = h;
    if (this.gl) this.gl.viewport(0, 0, w, h);
  }

  setPhase(phase) {
    const idx = PHASES.indexOf(phase);
    if (idx < 0) return;
    const target = [0, 0, 0, 0];
    target[idx] = 1;
    if (target.every((v, i) => v === this.phaseTargets[i])) return;
    this.phaseTransitionFrom = [...this.phaseWeights];
    this.phaseTargets = target;
    this.phaseTransitionStart = performance.now();
  }

  setAnalyser(analyser) {
    this.analyser = analyser;
    if (analyser) {
      this.audioBuf = new Uint8Array(analyser.fftSize);
    } else {
      this.audioBuf = null;
    }
  }

  _updatePhaseMix(now) {
    const dt = now - this.phaseTransitionStart;
    const t = Math.min(1, dt / CROSSFADE_MS);
    // smoothstep
    const e = t * t * (3 - 2 * t);
    for (let i = 0; i < 4; i++) {
      this.phaseWeights[i] = this.phaseTransitionFrom[i] + (this.phaseTargets[i] - this.phaseTransitionFrom[i]) * e;
    }
  }

  _readAmplitude() {
    if (!this.analyser || !this.audioBuf) return 0;
    this.analyser.getByteTimeDomainData(this.audioBuf);
    let sum = 0;
    for (let i = 0; i < this.audioBuf.length; i++) {
      const v = (this.audioBuf[i] - 128) / 128;
      sum += v * v;
    }
    return Math.sqrt(sum / this.audioBuf.length);
  }

  _loop = () => {
    const gl = this.gl;
    if (!gl) return;
    const now = performance.now();
    this._updatePhaseMix(now);

    // Smooth amplitude (EMA, faster attack than release)
    const raw = this._readAmplitude();
    const mapped = Math.min(1, raw * 4.0);  // boost typical speech levels
    const k = mapped > this.smoothedAmp ? 0.35 : 0.08;
    this.smoothedAmp = this.smoothedAmp + (mapped - this.smoothedAmp) * k;

    gl.useProgram(this.program);
    gl.bindVertexArray(this.vao);
    gl.uniform2f(this.uniforms.uRes, this.canvas.width, this.canvas.height);
    // Reduced motion → zmrazený čas = žádné time-based shader efekty.
    const timeSec = this.mqReduced && this.mqReduced.matches
      ? 0
      : (now - this.startTime) / 1000;
    gl.uniform1f(this.uniforms.uTime, timeSec);
    gl.uniform1f(this.uniforms.uAmp, this.smoothedAmp);
    gl.uniform4f(this.uniforms.uPhaseMix, ...this.phaseWeights);
    gl.drawArrays(gl.TRIANGLES, 0, 3);

    this.rafId = requestAnimationFrame(this._loop);
  };
}
