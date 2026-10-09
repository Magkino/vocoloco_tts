/**
 * trim-editor.js — TrimEditor: choose the part of a longer clip to use as a
 * voice reference. Drag the start/end handles or the whole window, click
 * outside it to move it there, preview with a playhead. A newly placed
 * selection snaps to the quietest points nearby so it doesn't cut mid-word;
 * handle and window drags after that are exact. Clips longer than the main
 * view get an overview strip for navigation.
 */

import { drawBarVisualizer } from './player.js';
import { quietestPoint, defaultSelection } from './audio-postprocess.js?v=2';

const VIEW_S = 45;      // main view width for long clips (room around a 30 s selection)
const SNAP_S = 0.15;    // snap search radius around a released edge
const MIN_LEN_S = 0.5;  // the handles can't cross
const STEP_S = 0.1;     // arrow-key nudge (Shift: 1 s)
const LOUD_DB = -40;    // a 10 ms frame this loud counts as speech

export class TrimEditor {
  constructor({ canvas, overviewCanvas = null, sampleRate = 24000, minS = 3, maxS = 15, onChange = () => {} }) {
    this.canvas = canvas;
    this.overview = overviewCanvas;
    this.sr = sampleRate;
    this.minS = minS;
    this.maxS = maxS;
    this.onChange = onChange;
    this.pcm = null;
    this.start = 0;
    this.end = 0;
    this.viewStart = 0;
    this.viewEnd = 0;
    this._loud = null; // Uint8Array per 10 ms frame

    this.main = canvas.parentElement; // positioned wrapper (.trim-main)
    const add = (cls) => { const d = document.createElement('div'); d.className = cls; this.main.appendChild(d); return d; };
    this.shadeL = add('trim-shade');
    this.shadeR = add('trim-shade');
    this.selEl = add('trim-sel');
    this.hStart = add('trim-handle');
    this.hEnd = add('trim-handle');
    this.playhead = add('trim-playhead hidden');
    for (const [h, which, label] of [[this.hStart, 'start', 'Selection start'], [this.hEnd, 'end', 'Selection end']]) {
      h.tabIndex = 0;
      h.setAttribute('role', 'slider');
      h.setAttribute('aria-label', label);
      h.addEventListener('keydown', (e) => this._onKey(e, which));
    }
    this.main.addEventListener('pointerdown', (e) => this._onPointerDown(e));

    if (overviewCanvas) {
      const wrap = overviewCanvas.parentElement;
      this.ovWin = document.createElement('div');
      this.ovWin.className = 'trim-ov-window';
      wrap.appendChild(this.ovWin);
      wrap.addEventListener('pointerdown', (e) => this._onOverviewDown(e));
    }
  }

  /** Show `pcm` with `sel` ({ start, end } in samples) or an automatic default selection. */
  load(pcm, sel = null) {
    this.pcm = pcm;
    const s = sel || defaultSelection(pcm, this.sr, { minS: this.minS, maxS: this.maxS });
    this.start = s.start;
    this.end = s.end;
    this._computeLoudness();
    this._long = pcm.length > VIEW_S * this.sr;
    if (this.overview) this.overview.parentElement.classList.toggle('hidden', !this._long);
    this._fitView(true);
    this.redraw();
    this._emit(true);
  }

  getSelection() { return { start: this.start, end: this.end }; }

  /** True when the selection leaves out speech, i.e. the transcript probably needs editing. */
  cutsSpeech() {
    if (!this._loud) return false;
    const frame = Math.round(this.sr / 100);
    const a = Math.floor(this.start / frame), b = Math.ceil(this.end / frame);
    for (let i = 0; i < this._loud.length; i++) if ((i < a || i >= b) && this._loud[i]) return true;
    return false;
  }

  /** Playhead position in samples, or null to hide it. */
  setPlayhead(sample) {
    const vl = this.viewEnd - this.viewStart;
    if (sample == null || !vl || sample < this.viewStart || sample > this.viewEnd) {
      this.playhead.classList.add('hidden');
      return;
    }
    this.playhead.classList.remove('hidden');
    this.playhead.style.left = `${((sample - this.viewStart) / vl) * 100}%`;
  }

  /** Full redraw (after load / resize). */
  redraw() {
    if (!this.pcm) return;
    // The overview spans the whole clip (up to 30 min), so it's only drawn here
    if (this.overview && this._long) drawBarVisualizer(this.overview, this.pcm, null);
    this._drawMain();
  }

  // ── Internals ──

  _drawMain() {
    drawBarVisualizer(this.canvas, this.pcm.subarray(this.viewStart, this.viewEnd));
    this._layout();
  }

  _computeLoudness() {
    const frame = Math.round(this.sr / 100);
    const gate = Math.pow(10, LOUD_DB / 20) ** 2 * frame;
    const n = Math.floor(this.pcm.length / frame);
    this._loud = new Uint8Array(n);
    for (let f = 0; f < n; f++) {
      let s = 0;
      for (let i = f * frame, e = i + frame; i < e; i++) s += this.pcm[i] * this.pcm[i];
      this._loud[f] = s >= gate ? 1 : 0;
    }
  }

  _fitView(force = false) {
    const len = this.pcm.length;
    if (!this._long) { this.viewStart = 0; this.viewEnd = len; return false; }
    if (!force && this.start >= this.viewStart && this.end <= this.viewEnd) return false;
    const w = VIEW_S * this.sr;
    const center = (this.start + this.end) / 2;
    this.viewStart = Math.round(Math.min(Math.max(0, center - w / 2), len - w));
    this.viewEnd = this.viewStart + w;
    return true;
  }

  _layout() {
    const vl = this.viewEnd - this.viewStart;
    const frac = (x) => Math.min(1, Math.max(0, (x - this.viewStart) / vl));
    const x0 = frac(this.start) * 100, x1 = frac(this.end) * 100;
    this.shadeL.style.left = '0%';
    this.shadeL.style.width = `${x0}%`;
    this.shadeR.style.left = `${x1}%`;
    this.shadeR.style.width = `${100 - x1}%`;
    this.selEl.style.left = `${x0}%`;
    this.selEl.style.width = `${x1 - x0}%`;
    this.hStart.style.left = `${x0}%`;
    this.hEnd.style.left = `${x1}%`;
    const total = this.pcm.length / this.sr;
    for (const [h, v] of [[this.hStart, this.start], [this.hEnd, this.end]]) {
      h.setAttribute('aria-valuemin', '0');
      h.setAttribute('aria-valuemax', total.toFixed(1));
      h.setAttribute('aria-valuenow', (v / this.sr).toFixed(1));
      h.setAttribute('aria-valuetext', `${(v / this.sr).toFixed(1)} seconds`);
    }
    if (this.ovWin && this._long) {
      this.ovWin.style.left = `${(this.start / this.pcm.length) * 100}%`;
      this.ovWin.style.width = `${((this.end - this.start) / this.pcm.length) * 100}%`;
    }
  }

  _emit(final) { this.onChange(this.getSelection(), { final }); }

  _update(final = false) {
    if (this._fitView()) this._drawMain(); else this._layout();
    this._emit(final);
  }

  _setStart(s) {
    const minLen = MIN_LEN_S * this.sr, maxLen = this.maxS * this.sr;
    s = Math.round(Math.min(Math.max(0, s), this.end - minLen));
    this.start = Math.max(s, this.end - maxLen);
  }

  _setEnd(s) {
    const minLen = MIN_LEN_S * this.sr, maxLen = this.maxS * this.sr;
    s = Math.round(Math.max(Math.min(this.pcm.length, s), this.start + minLen));
    this.end = Math.min(s, this.start + maxLen);
  }

  _moveTo(start) {
    const len = this.end - this.start;
    this.start = Math.round(Math.min(Math.max(0, start), this.pcm.length - len));
    this.end = this.start + len;
  }

  // Snap released edges to the quietest nearby point, unless that breaks the length limits
  _snap(which) {
    const r = SNAP_S * this.sr, maxLen = this.maxS * this.sr, minLen = MIN_LEN_S * this.sr;
    if (which !== 'end') {
      const s = quietestPoint(this.pcm, this.sr, this.start - r, this.start + r);
      if (this.end - s <= maxLen && this.end - s >= minLen) this.start = s;
    }
    if (which !== 'start') {
      const e = quietestPoint(this.pcm, this.sr, this.end - r, this.end + r);
      if (e - this.start <= maxLen && e - this.start >= minLen) this.end = e;
    }
  }

  _sampleAt(clientX) {
    const r = this.main.getBoundingClientRect();
    const f = Math.min(1, Math.max(0, (clientX - r.left) / r.width));
    return Math.round(this.viewStart + f * (this.viewEnd - this.viewStart));
  }

  _onPointerDown(e) {
    if (!this.pcm || e.button > 0) return;
    e.preventDefault();
    const s0 = this._sampleAt(e.clientX);
    const mode = e.target === this.hStart ? 'start'
      : e.target === this.hEnd ? 'end'
      : (s0 > this.start && s0 < this.end) ? 'move' : 'new';
    const orig = { start: this.start, x: e.clientX, s0 };
    let moved = false;
    if (mode === 'start' || mode === 'end') (mode === 'start' ? this.hStart : this.hEnd).focus({ preventScroll: true });
    this.main.setPointerCapture(e.pointerId);

    const onMove = (ev) => {
      if (!moved && Math.abs(ev.clientX - orig.x) < 3) return;
      moved = true;
      const s = this._sampleAt(ev.clientX);
      if (mode === 'start') this._setStart(s);
      else if (mode === 'end') this._setEnd(s);
      else if (mode === 'move') this._moveTo(orig.start + (s - orig.s0));
      else {
        // drag out a new selection from where the pointer went down
        this.start = Math.min(orig.s0, s);
        this.end = Math.max(orig.s0, s);
        if (s < orig.s0) this._setStart(this.start); else this._setEnd(this.end);
      }
      this._update(false);
    };
    const onUp = () => {
      this.main.removeEventListener('pointermove', onMove);
      this.main.removeEventListener('pointerup', onUp);
      this.main.removeEventListener('pointercancel', onUp);
      // A newly placed selection snaps to quiet points; dragging a handle or the
      // window afterwards is exact, so small adjustments stay where they're dropped
      if (mode === 'new') {
        if (!moved) this._moveTo(s0 - (this.end - this.start) / 2);
        this._snap('both');
      }
      this._update(true);
    };
    this.main.addEventListener('pointermove', onMove);
    this.main.addEventListener('pointerup', onUp);
    this.main.addEventListener('pointercancel', onUp);
  }

  _onOverviewDown(e) {
    if (!this.pcm || e.button > 0) return;
    e.preventDefault();
    const wrap = this.overview.parentElement;
    const centerAt = (clientX) => {
      const r = wrap.getBoundingClientRect();
      const f = Math.min(1, Math.max(0, (clientX - r.left) / r.width));
      this._moveTo(f * this.pcm.length - (this.end - this.start) / 2);
      this._fitView(true);
      this._drawMain();
      this._emit(false);
    };
    wrap.setPointerCapture(e.pointerId);
    centerAt(e.clientX);
    const onMove = (ev) => centerAt(ev.clientX);
    const onUp = () => {
      wrap.removeEventListener('pointermove', onMove);
      wrap.removeEventListener('pointerup', onUp);
      wrap.removeEventListener('pointercancel', onUp);
      this._snap('both');
      this._update(true);
    };
    wrap.addEventListener('pointermove', onMove);
    wrap.addEventListener('pointerup', onUp);
    wrap.addEventListener('pointercancel', onUp);
  }

  _onKey(e, which) {
    if (!this.pcm) return;
    const step = (e.shiftKey ? 1 : STEP_S) * this.sr;
    const cur = which === 'start' ? this.start : this.end;
    let next;
    if (e.key === 'ArrowLeft' || e.key === 'ArrowDown') next = cur - step;
    else if (e.key === 'ArrowRight' || e.key === 'ArrowUp') next = cur + step;
    else if (e.key === 'Home') next = 0;
    else if (e.key === 'End') next = this.pcm.length;
    else return;
    e.preventDefault();
    if (which === 'start') this._setStart(next); else this._setEnd(next);
    this._update(true);
  }
}
