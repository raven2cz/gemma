/**
 * Abstract Avatar plugin interface.
 *
 * Vytvoř vlastní třídu (dědí/implementuje stejné API), zaregistruj ji
 * v app.js do mapy `AVATARS` a automaticky se objeví v UI picker.
 *
 * Každý avatar si sám vybere renderovací kontext (2D, WebGL, WebGL2)
 * v `attach(canvas)`.
 */
export class Avatar {
  /** Metadata pro UI picker. Přepiš v potomkovi. */
  static meta = { id: 'base', label: 'base avatar' };

  /**
   * Připojí avatar k canvasu. Zavolej jednou. Startuje render loop.
   * @param {HTMLCanvasElement} canvas
   */
  attach(canvas) { throw new Error('not implemented'); }

  /** Odpojí avatar, zruší RAF, uvolní GL zdroje. */
  detach() { throw new Error('not implemented'); }

  /**
   * Hook pro resize + HiDPI. App.js ho volá z ResizeObserveru.
   * @param {number} cssWidth
   * @param {number} cssHeight
   * @param {number} dpr
   */
  resize(cssWidth, cssHeight, dpr) { throw new Error('not implemented'); }

  /**
   * Nastavení aktuální fáze. Implementace by měla crossfade přes ~300 ms.
   * @param {'idle'|'listening'|'thinking'|'speaking'|'error'} phase
   */
  setPhase(phase) { throw new Error('not implemented'); }

  /**
   * Volitelné: připoj WebAudio AnalyserNode pro amplitudově-reaktivní efekty.
   * Implementace typicky volá getByteFrequencyData nebo getFloatTimeDomainData
   * v render loopu.
   * @param {AnalyserNode|null} analyser
   */
  setAnalyser(analyser) { /* optional */ }
}
