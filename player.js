/**
 * player.js — StreamingPlayer: gapless scheduling of streamed PCM chunks with
 * a click-to-seek waveform, moving playhead, elapsed/total time display, and
 * one-shot previews. Owns the 24 kHz playback AudioContext.
 */

// ─── Waveform drawing (shared by studio player, library cards, wizard) ─────

export function drawBarVisualizer(canvas, pcm, numBars, totalSamples) {
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 2;
  // Use offsetWidth if available (visible), otherwise use pre-set canvas.width
  const ow = canvas.offsetWidth;
  const oh = canvas.offsetHeight;
  const w = ow > 0 ? ow * dpr : canvas.width;
  const h = oh > 0 ? oh * dpr : canvas.height;
  canvas.width = w;
  canvas.height = h;
  ctx.clearRect(0, 0, w, h);
  if (w === 0 || h === 0) return;

  if (!numBars) numBars = Math.min(150, Math.floor(w / 4));
  const total = totalSamples || pcm.length;
  const totalBar = w / numBars;
  const gap = Math.max(1, Math.floor(totalBar * 0.25));
  const barWidth = totalBar - gap;

  for (let i = 0; i < numBars; i++) {
    const sampleIdx = Math.floor(i * total / numBars);
    const x = i * totalBar;

    // Not-yet-generated region (streaming): flat gray stub
    if (sampleIdx >= pcm.length) {
      ctx.shadowBlur = 0;
      ctx.fillStyle = 'rgba(148, 163, 184, 0.18)';
      ctx.fillRect(x, (h - 2) / 2, barWidth, 2);
      continue;
    }

    const windowSize = Math.max(1, Math.floor(total / numBars));
    let rms = 0;
    for (let j = 0; j < windowSize; j++) {
      const v = pcm[sampleIdx + j] || 0;
      rms += v * v;
    }
    rms = Math.sqrt(rms / windowSize);

    const barH = Math.max(2, rms * h * 2);
    const y = (h - barH) / 2;
    const intensity = Math.min(1, rms * 5);

    // Green neon gradient per bar
    const grd = ctx.createLinearGradient(x, y, x, y + barH);
    grd.addColorStop(0, `rgba(74, 222, 128, ${0.4 + intensity * 0.6})`);
    grd.addColorStop(0.5, `rgba(34, 197, 94, ${0.6 + intensity * 0.4})`);
    grd.addColorStop(1, `rgba(74, 222, 128, ${0.4 + intensity * 0.6})`);
    ctx.fillStyle = grd;

    // Neon glow
    ctx.shadowColor = `rgba(74, 222, 128, ${intensity * 0.8})`;
    ctx.shadowBlur = intensity * 10;

    // Rounded rect
    const r = Math.min(barWidth / 2, 3);
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.lineTo(x + barWidth - r, y);
    ctx.quadraticCurveTo(x + barWidth, y, x + barWidth, y + r);
    ctx.lineTo(x + barWidth, y + barH - r);
    ctx.quadraticCurveTo(x + barWidth, y + barH, x + barWidth - r, y + barH);
    ctx.lineTo(x + r, y + barH);
    ctx.quadraticCurveTo(x, y + barH, x, y + barH - r);
    ctx.lineTo(x, y + r);
    ctx.quadraticCurveTo(x, y, x + r, y);
    ctx.closePath();
    ctx.fill();
  }
  ctx.shadowBlur = 0;
}

export function drawMiniWaveform(canvas, pcm) {
  drawBarVisualizer(canvas, pcm, 60);
}

function fmtTime(seconds) {
  const s = Math.max(0, seconds);
  return `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`;
}

// ─── StreamingPlayer ────────────────────────────────────────────────────────

export class StreamingPlayer {
  constructor({ canvas, playheadEl, timeEl, bufferingEl, sampleRate = 24000, gapSeconds = 0.15, onStateChange = () => {}, onAutoplayBlocked = null } = {}) {
    this.canvas = canvas;
    this.playheadEl = playheadEl;
    this.timeEl = timeEl;
    this.bufferingEl = bufferingEl;
    this.sr = sampleRate;
    this.gapSamples = Math.round(gapSeconds * sampleRate);
    this.onStateChange = onStateChange;
    this.onAutoplayBlocked = onAutoplayBlocked;

    this._ctx = null;
    this._combined = new Float32Array(0); // grows chunk by chunk (incl. gaps)
    this._generated = 0;                  // samples available so far
    this._estTotal = 0;                   // estimated final length while streaming
    this._streaming = false;
    this._playing = false;
    this._spans = [];                     // { ctxStart, sampleStart, sampleEnd }
    this._sources = [];                   // live AudioBufferSourceNodes
    this._nextCtxTime = 0;                // when the next chunk's audio should start
    this._epoch = 0;                      // invalidates stale onended callbacks
    this._raf = 0;
    this._lastSample = 0;
    this._lastTimeUpdate = 0;
    this._lastAriaUpdate = 0;
    this._oneShot = null;                 // { src, onended }

    if (canvas) {
      canvas.addEventListener('click', (e) => {
        if (this._generated === 0) return;
        const rect = canvas.getBoundingClientRect();
        if (rect.width > 0) this.seekToFraction((e.clientX - rect.left) / rect.width);
      });
      canvas.addEventListener('keydown', (e) => {
        if (this._generated === 0) return;
        const total = this._displayTotal();
        if (e.key === 'ArrowLeft') { e.preventDefault(); this.seekToFraction((this._lastSample - 5 * this.sr) / total); }
        else if (e.key === 'ArrowRight') { e.preventDefault(); this.seekToFraction((this._lastSample + 5 * this.sr) / total); }
        else if (e.key === 'Home') { e.preventDefault(); this.seekToFraction(0); }
      });
    }
  }

  getAudioCtx() {
    if (!this._ctx || this._ctx.state === 'closed') this._ctx = new AudioContext({ sampleRate: this.sr });
    if (this._ctx.state === 'suspended') this._ctx.resume().catch(() => {});
    return this._ctx;
  }

  isPlaying() { return this._playing; }
  isStreaming() { return this._streaming; }
  hasAudio() { return this._generated > 0; }
  getCombined() { return this._combined; }

  dimPrevious() { if (this.canvas) this.canvas.style.opacity = '0.35'; }
  undim() { if (this.canvas) this.canvas.style.opacity = ''; }

  // ── Streaming ──

  beginStream({ estTotalSamples = 0 } = {}) {
    this._stopAllSources();
    this.stopOneShot();
    this._combined = new Float32Array(0);
    this._generated = 0;
    this._estTotal = estTotalSamples;
    this._streaming = true;
    this._playing = true;
    this._nextCtxTime = 0;
    this._lastSample = 0;
    this.undim();
    if (this.playheadEl) {
      this.playheadEl.classList.remove('hidden');
      this.playheadEl.style.transform = 'translateX(0px)';
    }
    this._drawWave();
    this._updateTimeText(true);
    this._startRaf();
    this.onStateChange('playing');
    // Autoplay policy: generation is user-initiated so the context is normally
    // running; if it stays suspended, let the app show a "click to play" hint.
    const ctx = this.getAudioCtx();
    setTimeout(() => {
      if (ctx.state === 'suspended' && this.onAutoplayBlocked) this.onAutoplayBlocked();
    }, 400);
  }

  appendChunk(pcm) {
    if (!this._streaming) return;
    if (this._generated > 0) this._appendToCombined(new Float32Array(this.gapSamples));
    const sampleStart = this._generated;
    this._appendToCombined(pcm);
    if (this._playing) {
      const ctx = this.getAudioCtx();
      const startAt = Math.max(ctx.currentTime + 0.02, this._nextCtxTime);
      this._scheduleSpan(pcm, sampleStart, startAt);
      this._nextCtxTime = startAt + pcm.length / this.sr + this.gapSamples / this.sr;
    }
    this._drawWave();
  }

  /** Finish the stream; returns the combined Float32Array (chunks + gaps). */
  endStream() {
    this._streaming = false;
    this._estTotal = this._generated;
    if (this.bufferingEl) this.bufferingEl.classList.add('hidden');
    this._drawWave();
    this._updateTimeText(true);
    if (this._playing) this._startRaf();
    else this.onStateChange('idle');
    return this._combined;
  }

  // ── Static playback (replay / restored generations) ──

  loadStatic(pcm) {
    this._stopAllSources();
    this.stopOneShot();
    this._streaming = false;
    this._playing = false;
    this._combined = new Float32Array(pcm);
    this._generated = this._combined.length;
    this._estTotal = this._generated;
    this._lastSample = 0;
    if (this.playheadEl) this.playheadEl.classList.add('hidden');
    if (this.bufferingEl) this.bufferingEl.classList.add('hidden');
    this._drawWave();
    this._updateTimeText(true);
    this.onStateChange('idle');
  }

  play(fromSample = 0) {
    if (this._generated === 0) return;
    this._startFrom(Math.floor(fromSample));
  }

  stop() {
    this.stopOneShot();
    if (!this._playing) return;
    this._stopAllSources();
    this._playing = false;
    if (this.bufferingEl) this.bufferingEl.classList.add('hidden');
    this._updateTimeText(true);
    this.onStateChange('idle');
  }

  seekToFraction(fraction) {
    if (this._generated === 0) return;
    const total = this._displayTotal();
    const target = Math.round(Math.min(Math.max(fraction, 0), 1) * total);
    this._startFrom(target);
  }

  _startFrom(sample) {
    const target = Math.min(Math.max(0, sample), this._generated - 1);
    this.stopOneShot();
    this._stopAllSources();
    const ctx = this.getAudioCtx();
    const startAt = ctx.currentTime + 0.02;
    this._scheduleSpan(this._combined.subarray(target, this._generated), target, startAt);
    this._nextCtxTime = startAt + (this._generated - target) / this.sr + this.gapSamples / this.sr;
    this._playing = true;
    this._lastSample = target;
    if (this.playheadEl) this.playheadEl.classList.remove('hidden');
    this._updatePlayhead(target);
    this._startRaf();
    this.onStateChange('playing');
  }

  // ── One-shot previews (library cards, wizard, voice tests) ──

  playOneShot(pcm, sampleRate = this.sr, onended = null) {
    this.stopOneShot();
    if (this._playing) this.stop();
    const ctx = this.getAudioCtx();
    const buf = ctx.createBuffer(1, pcm.length, sampleRate);
    buf.getChannelData(0).set(pcm);
    const src = ctx.createBufferSource();
    src.buffer = buf;
    src.connect(ctx.destination);
    const entry = { src, onended };
    src.onended = () => {
      if (this._oneShot !== entry) return;
      this._oneShot = null;
      if (onended) onended();
    };
    this._oneShot = entry;
    src.start();
  }

  stopOneShot() {
    if (!this._oneShot) return;
    const { src, onended } = this._oneShot;
    this._oneShot = null;
    try { src.onended = null; src.stop(); } catch { /* already stopped */ }
    if (onended) onended();
  }

  isOneShotPlaying() { return !!this._oneShot; }

  // ── Internals ──

  _appendToCombined(pcm) {
    const next = new Float32Array(this._combined.length + pcm.length);
    next.set(this._combined, 0);
    next.set(pcm, this._combined.length);
    this._combined = next;
    this._generated = next.length;
  }

  _scheduleSpan(pcm, sampleStart, startAt) {
    const ctx = this.getAudioCtx();
    const buf = ctx.createBuffer(1, pcm.length, this.sr);
    buf.getChannelData(0).set(pcm);
    const src = ctx.createBufferSource();
    src.buffer = buf;
    src.connect(ctx.destination);
    const epoch = this._epoch;
    // Belt-and-braces end detection (rAF is throttled in background tabs)
    src.onended = () => { if (epoch === this._epoch) this._maybeFinishPlayback(); };
    src.start(startAt);
    this._sources.push(src);
    this._spans.push({ ctxStart: startAt, sampleStart, sampleEnd: sampleStart + pcm.length });
  }

  _stopAllSources() {
    this._epoch++;
    for (const s of this._sources) {
      try { s.onended = null; s.stop(); } catch { /* already stopped */ }
    }
    this._sources = [];
    this._spans = [];
  }

  _displayTotal() {
    return (this._streaming ? Math.max(this._estTotal, this._generated) : this._generated) || 1;
  }

  _lastSpanEnd() {
    const s = this._spans[this._spans.length - 1];
    return s ? s.ctxStart + (s.sampleEnd - s.sampleStart) / this.sr : 0;
  }

  _computeSample(now) {
    const spans = this._spans;
    if (!spans.length) return this._lastSample;
    if (now < spans[0].ctxStart) return spans[0].sampleStart;
    for (let i = 0; i < spans.length; i++) {
      const s = spans[i];
      const end = s.ctxStart + (s.sampleEnd - s.sampleStart) / this.sr;
      if (now < end) {
        if (now >= s.ctxStart) return s.sampleStart + (now - s.ctxStart) * this.sr;
        // In the scheduled gap between the previous span and this one
        const prev = spans[i - 1];
        if (!prev) return s.sampleStart;
        const prevEnd = prev.ctxStart + (prev.sampleEnd - prev.sampleStart) / this.sr;
        return prev.sampleEnd + (now - prevEnd) * this.sr;
      }
    }
    const last = spans[spans.length - 1];
    const lastEnd = last.ctxStart + (last.sampleEnd - last.sampleStart) / this.sr;
    return last.sampleEnd + (now - lastEnd) * this.sr;
  }

  _maybeFinishPlayback() {
    if (this._streaming || !this._playing) return;
    if (!this._ctx || !this._spans.length) return;
    if (this._ctx.currentTime >= this._lastSpanEnd() - 0.05) this._finishPlayback();
  }

  _finishPlayback() {
    this._epoch++;
    this._sources = [];
    this._spans = [];
    this._playing = false;
    this._lastSample = this._generated;
    this._updatePlayhead(this._generated);
    this._updateTimeText(true);
    this.onStateChange('idle');
  }

  _startRaf() {
    if (this._raf) return;
    const tick = () => {
      this._raf = 0;
      const ctx = this._ctx;
      if (!ctx) return;
      if (this._playing) {
        const now = ctx.currentTime;
        let sample = this._computeSample(now);
        const pastEnd = this._spans.length > 0 && now > this._lastSpanEnd() + 0.02;
        if (pastEnd) {
          if (this._streaming) {
            // Underrun: playback caught up with generation
            sample = this._generated;
            if (this.bufferingEl) this.bufferingEl.classList.remove('hidden');
          } else {
            this._finishPlayback();
            return;
          }
        } else if (this.bufferingEl && !this.bufferingEl.classList.contains('hidden')) {
          this.bufferingEl.classList.add('hidden');
        }
        this._lastSample = Math.min(Math.max(0, sample), this._generated);
        this._updatePlayhead(this._lastSample);
        this._updateTimeText();
      }
      if (this._playing || this._streaming) this._raf = requestAnimationFrame(tick);
    };
    this._raf = requestAnimationFrame(tick);
  }

  _updatePlayhead(sample) {
    if (!this.playheadEl || !this.canvas) return;
    const w = this.canvas.clientWidth;
    if (w <= 0) return;
    const px = Math.min(1, sample / this._displayTotal()) * w;
    this.playheadEl.style.transform = `translateX(${px.toFixed(1)}px)`;
    const nowMs = performance.now();
    if (nowMs - this._lastAriaUpdate > 1000) {
      this._lastAriaUpdate = nowMs;
      this.canvas.setAttribute('aria-valuenow', String(Math.round((sample / this._displayTotal()) * 100)));
    }
  }

  _updateTimeText(force = false) {
    if (!this.timeEl) return;
    const nowMs = performance.now();
    if (!force && nowMs - this._lastTimeUpdate < 250) return;
    this._lastTimeUpdate = nowMs;
    if (this._generated === 0 && !this._streaming) { this.timeEl.textContent = ''; return; }
    const totalStr = (this._streaming ? '~' : '') + fmtTime(this._displayTotal() / this.sr);
    this.timeEl.textContent = `${fmtTime(this._lastSample / this.sr)} / ${totalStr}`;
  }

  _drawWave() {
    if (!this.canvas) return;
    if (this._streaming) drawBarVisualizer(this.canvas, this._combined, null, this._displayTotal());
    else drawBarVisualizer(this.canvas, this._combined);
  }

  /** Redraw after container resize / view switches. */
  redraw() {
    this._drawWave();
    this._updatePlayhead(this._lastSample);
  }
}
