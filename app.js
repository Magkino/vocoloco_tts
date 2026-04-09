/**
 * OmniVoice — Main application (redesigned)
 */

const MODEL_BASE_URL = 'https://huggingface.co/Gigsu/vocoloco-onnx/resolve/main';

// ─── DOM — all original IDs preserved ──────────────────────────────────────
const textEl = document.getElementById('text-input');
const qualityEl = document.getElementById('quality');
const generateBtn = document.getElementById('generate-btn');
const statusEl = document.getElementById('status');
const playerEl = document.getElementById('player');
const waveformEl = document.getElementById('waveform');
const refAudioEl = document.getElementById('ref-audio');
const refTextEl = document.getElementById('ref-text');
const micBtn = document.getElementById('mic-btn');
const stopBtn = document.getElementById('stop-btn');
const recInfo = document.getElementById('rec-info');
const recTimeEl = document.getElementById('rec-time');
const refPreview = document.getElementById('ref-preview');
const cloneToggle = document.getElementById('clone-toggle');
const cloneBody = document.getElementById('clone-body');
const saveVoiceNameEl = document.getElementById('save-voice-name');
const saveVoiceBtn = document.getElementById('save-voice-btn');
const savedVoicesEl = document.getElementById('saved-voices');
const genderRow = document.getElementById('gender-row');
const pitchRow = document.getElementById('pitch-row');

// New DOM elements
const voiceBadgeText = document.getElementById('voice-badge-text');
const charCount = document.getElementById('char-count');
const progressBar = document.getElementById('progress-bar');
const replayBtn = document.getElementById('replay-btn');
const downloadBtn = document.getElementById('download-btn');
const playerDuration = document.getElementById('player-duration');
const historySectionEl = document.getElementById('history-section');
const historyScrollEl = document.getElementById('history-scroll');
const clonePanel = document.getElementById('clone-panel');
const newCloneBtn = document.getElementById('new-clone-btn');

// ─── State ────────────────────────────────────────────────────────────────
let ttsWorker = null;
let audioCtx = null;
let isReady = false;
let isGenerating = false;
let mediaRecorder = null;
let recordedChunks = [];
let recordedBlob = null;
let recTimer = null;
let cloneActive = false;
let selectedSavedVoice = null;
let currentSource = null; // currently playing AudioBufferSourceNode

// Last generated audio for replay/download
let lastPcm = null;
let lastSampleRate = 24000;
let lastText = '';

// Generation history (persisted in IndexedDB)
let history = [];
const MAX_HISTORY = 50;
const HISTORY_DB = 'omnivoice-history';

function openHistoryDB() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(HISTORY_DB, 1);
    req.onupgradeneeded = () => req.result.createObjectStore('items', { keyPath: 'id', autoIncrement: true });
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

async function loadHistory() {
  const db = await openHistoryDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction('items', 'readonly');
    const req = tx.objectStore('items').getAll();
    req.onsuccess = () => {
      const items = req.result.map(item => ({
        ...item,
        pcm: new Float32Array(item.pcm),
      }));
      items.sort((a, b) => b.timestamp - a.timestamp);
      resolve(items.slice(0, MAX_HISTORY));
    };
    req.onerror = () => reject(req.error);
  });
}

async function saveHistoryItem(item) {
  const db = await openHistoryDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction('items', 'readwrite');
    // Store pcm as regular array for IndexedDB compat
    tx.objectStore('items').put({
      text: item.text,
      pcm: Array.from(item.pcm),
      sampleRate: item.sampleRate,
      duration: item.duration,
      timestamp: item.timestamp,
    });
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

async function deleteHistoryItem(id) {
  const db = await openHistoryDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction('items', 'readwrite');
    tx.objectStore('items').delete(id);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

// Clone panel state
let clonePanelOpen = false;

// Voice mode: 'design' | 'clone-saved' | 'clone-new'
// (simplified from tab-based approach)

// ─── Char count ───────────────────────────────────────────────────────────
textEl.addEventListener('input', () => {
  charCount.textContent = textEl.value.length;
});

// ─── Toggle button logic ───────────────────────────────────────────────────

function initToggleRow(row) {
  row.querySelectorAll('.toggle-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      row.querySelectorAll('.toggle-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      // Deselect saved voice when changing design params
      if (selectedSavedVoice) {
        selectedSavedVoice = null;
        cloneActive = false;
        cloneToggle.classList.remove('on');
        renderSavedVoices();
      }
      updateVoiceBadge();
    });
  });
}
initToggleRow(genderRow);
initToggleRow(pitchRow);

// Quality toggle row (syncs with hidden <select> for worker compat)
const qualityRow = document.getElementById('quality-row');
if (qualityRow) {
  qualityRow.querySelectorAll('.toggle-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      qualityRow.querySelectorAll('.toggle-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      // Sync to hidden select
      qualityEl.value = btn.dataset.val;
    });
  });
}

function getToggleVal(row) {
  const active = row.querySelector('.toggle-btn.active');
  return active ? active.dataset.val : '';
}

function getToggleLabel(row) {
  const active = row.querySelector('.toggle-btn.active');
  return active ? active.textContent.trim() : '';
}

function buildInstruct() {
  const parts = [getToggleVal(genderRow), getToggleVal(pitchRow)].filter(Boolean);
  return parts.length ? parts.join(', ') : null;
}

// ─── Deterministic seed from instruct string ──────────────────────────────
// Hash instruct string to a stable integer seed for reproducible voice design
function hashInstruct(instruct) {
  if (!instruct) return 0;
  let hash = 0;
  for (let i = 0; i < instruct.length; i++) {
    const ch = instruct.charCodeAt(i);
    hash = ((hash << 5) - hash) + ch;
    hash = hash & hash; // Convert to 32-bit integer
  }
  return Math.abs(hash);
}

// ─── Voice Badge ───────────────────────────────────────────────────────────

function isCloneMode() {
  return !!(selectedSavedVoice || (clonePanelOpen && (recordedBlob || (refAudioEl.files && refAudioEl.files.length > 0))));
}

function updateVoiceUI() {
  const cloning = isCloneMode();

  // Dim voice design controls when a cloned voice is active
  genderRow.style.opacity = cloning ? '0.3' : '1';
  genderRow.style.pointerEvents = cloning ? 'none' : 'auto';
  pitchRow.style.opacity = cloning ? '0.3' : '1';
  pitchRow.style.pointerEvents = cloning ? 'none' : 'auto';
  // Also dim labels
  genderRow.parentElement.querySelector('label').style.opacity = cloning ? '0.3' : '1';
  pitchRow.parentElement.querySelector('label').style.opacity = cloning ? '0.3' : '1';

  // Update badge and button
  if (selectedSavedVoice) {
    voiceBadgeText.textContent = `Using cloned voice: "${selectedSavedVoice.name}"`;
    voiceBadgeText.style.color = '#4ade80';
    generateBtn.textContent = `Generate as "${selectedSavedVoice.name}"`;
  } else if (clonePanelOpen && (recordedBlob || (refAudioEl.files && refAudioEl.files.length > 0))) {
    voiceBadgeText.textContent = 'Using new cloned voice';
    voiceBadgeText.style.color = '#4ade80';
    generateBtn.textContent = 'Generate with Clone';
  } else {
    const gender = getToggleVal(genderRow);
    const pitch = getToggleVal(pitchRow);
    voiceBadgeText.textContent = `${gender}, ${pitch}`;
    voiceBadgeText.style.color = '';
    generateBtn.textContent = 'Generate Speech';
  }
}

// Alias for compat
function updateVoiceBadge() { updateVoiceUI(); }

// ─── Clone Panel Toggle ───────────────────────────────────────────────────

newCloneBtn.addEventListener('click', () => {
  clonePanelOpen = !clonePanelOpen;
  clonePanel.classList.toggle('open', clonePanelOpen);
  if (clonePanelOpen) {
    cloneActive = true;
    cloneToggle.classList.add('on');
    cloneBody.classList.add('show');
  } else {
    // Only deactivate clone if no saved voice is selected
    if (!selectedSavedVoice) {
      cloneActive = false;
      cloneToggle.classList.remove('on');
    }
  }
  updateVoiceBadge();
});

// ─── IndexedDB for saved voices ────────────────────────────────────────────

const VOICE_DB = 'omnivoice-voices';
function openVoiceDB() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(VOICE_DB, 1);
    req.onupgradeneeded = () => req.result.createObjectStore('voices', { keyPath: 'id' });
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}
async function getSavedVoices() {
  const db = await openVoiceDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction('voices', 'readonly');
    const req = tx.objectStore('voices').getAll();
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}
async function saveVoice(voice) {
  const db = await openVoiceDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction('voices', 'readwrite');
    tx.objectStore('voices').put(voice);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}
async function deleteVoice(id) {
  const db = await openVoiceDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction('voices', 'readwrite');
    tx.objectStore('voices').delete(id);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

// ─── AudioContext ──────────────────────────────────────────────────────────

function getAudioCtx() {
  if (!audioCtx || audioCtx.state === 'closed') audioCtx = new AudioContext({ sampleRate: 24000 });
  if (audioCtx.state === 'suspended') audioCtx.resume();
  return audioCtx;
}

// ─── Progress Bar ─────────────────────────────────────────────────────────

function showProgress(mode) {
  progressBar.classList.remove('hidden');
  if (mode === 'indeterminate') {
    progressBar.classList.add('indeterminate');
    progressBar.style.width = '';
  } else {
    progressBar.classList.remove('indeterminate');
  }
}

function setProgressPercent(pct) {
  progressBar.classList.remove('indeterminate');
  progressBar.style.width = Math.min(100, pct) + '%';
}

function hideProgress() {
  progressBar.classList.add('hidden');
  progressBar.classList.remove('indeterminate');
  progressBar.style.width = '0%';
}

// ─── Worker ────────────────────────────────────────────────────────────────

function initWorker() {
  ttsWorker = new Worker('workers/tts-worker.js?v=2', { type: 'module' });
  ttsWorker.onmessage = (e) => {
    const msg = e.data;
    switch (msg.type) {
      case 'progress':
        setStatus(msg.detail);
        // Try to parse step progress for bar
        const stepMatch = msg.detail?.match?.(/(\d+)\s*\/\s*(\d+)/);
        if (stepMatch) {
          const [, current, total] = stepMatch;
          setProgressPercent((parseInt(current) / parseInt(total)) * 100);
        } else {
          showProgress('indeterminate');
        }
        break;
      case 'ready':
        isReady = true;
        setStatus('Ready');
        hideProgress();
        generateBtn.disabled = false;
        textEl.disabled = false;
        refAudioEl.disabled = false;
        refTextEl.disabled = false;
        micBtn.disabled = false;
        saveVoiceNameEl.disabled = false;
        saveVoiceBtn.disabled = false;
        break;
      case 'audio':
        onAudio(msg.pcm, msg.sampleRate);
        break;
      case 'error':
        if (msg.message === 'DESKTOP_ONLY') {
          const sep = location.search ? '&' : '?';
          document.querySelector('main').innerHTML = `
            <div style="text-align:center;padding:60px 24px;">
              <div style="font-size:48px;margin-bottom:16px;">🖥️</div>
              <h2 style="font-size:22px;font-weight:700;color:white;margin-bottom:8px;">Desktop Recommended</h2>
              <p style="color:#94a3b8;font-size:14px;line-height:1.6;max-width:340px;margin:0 auto 20px;">
                VocoLoco downloads ~3 GB of model data and requires significant GPU/VRAM to run inference. This will most likely not work on mobile devices.
              </p>
              <button onclick="location.href=location.href+'${sep}force=1'" style="padding:10px 24px;border-radius:10px;background:#1e293b;border:1px solid #2d3748;color:#94a3b8;font-size:13px;cursor:pointer;transition:all 0.15s;">Try anyway (not recommended)</button>
            </div>`;
          return;
        }
        setStatus(msg.message);
        console.error(msg.message);
        isGenerating = false;
        generateBtn.disabled = false;
        hideProgress();
        break;
    }
  };
  const forceLoad = new URLSearchParams(window.location.search).has('force');
  ttsWorker.postMessage({ type: 'init', modelBaseUrl: MODEL_BASE_URL, force: forceLoad });
  showProgress('indeterminate');
}

// ─── Audio playback ────────────────────────────────────────────────────────

function onAudio(pcm, sampleRate) {
  // Switch to studio view if not already there
  switchView('studio');

  // Store for replay/download
  lastPcm = new Float32Array(pcm);
  lastSampleRate = sampleRate;
  lastText = textEl.value.trim();

  const duration = pcm.length / sampleRate;

  // Add to history
  addToHistory(lastText, lastPcm, lastSampleRate, duration);

  // Show player FIRST, force reflow, then draw waveform
  const studioPlayer = document.getElementById('studio-player');
  if (studioPlayer) {
    studioPlayer.classList.remove('hidden');
    studioPlayer.style.display = 'block';
    // Force layout reflow so canvas gets real dimensions
    void studioPlayer.offsetHeight;
  }

  drawWaveform(pcm);
  playerDuration.textContent = duration.toFixed(1) + 's';

  // Play audio
  playPcm(pcm, sampleRate);

  // Scroll player into view
  if (studioPlayer) studioPlayer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

  setStatus(`Playing ${duration.toFixed(1)}s`);
  hideProgress();
  isGenerating = false;
  generateBtn.disabled = false;
}

function playPcm(pcm, sampleRate) {
  // Stop any currently playing audio
  if (currentSource) {
    try { currentSource.stop(); } catch (e) { /* ignore */ }
    currentSource = null;
  }

  const ctx = getAudioCtx();
  const buf = ctx.createBuffer(1, pcm.length, sampleRate);
  buf.getChannelData(0).set(pcm);
  const src = ctx.createBufferSource();
  src.buffer = buf;
  src.connect(ctx.destination);
  src.onended = () => {
    currentSource = null;
    if (!isGenerating) setStatus('Ready');
  };
  currentSource = src;
  src.start();
}

function drawBarVisualizer(canvas, pcm, numBars) {
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
  const totalBar = w / numBars;
  const gap = Math.max(1, Math.floor(totalBar * 0.25));
  const barWidth = totalBar - gap;

  for (let i = 0; i < numBars; i++) {
    const sampleIdx = Math.floor(i * pcm.length / numBars);
    const windowSize = Math.max(1, Math.floor(pcm.length / numBars));
    let rms = 0;
    for (let j = 0; j < windowSize; j++) {
      const v = pcm[sampleIdx + j] || 0;
      rms += v * v;
    }
    rms = Math.sqrt(rms / windowSize);

    const barH = Math.max(2, rms * h * 2);
    const x = i * totalBar;
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

function drawWaveform(pcm) {
  drawBarVisualizer(waveformEl, pcm);
}

// ─── WAV Download ──────────────────────────────────────────────────────────

function downloadWav(pcm, sampleRate, filename) {
  const numSamples = pcm.length;
  const bytesPerSample = 2;
  const dataSize = numSamples * bytesPerSample;
  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);
  const writeStr = (o, s) => { for (let i = 0; i < s.length; i++) view.setUint8(o + i, s.charCodeAt(i)); };
  writeStr(0, 'RIFF');
  view.setUint32(4, 36 + dataSize, true);
  writeStr(8, 'WAVE');
  writeStr(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * bytesPerSample, true);
  view.setUint16(32, bytesPerSample, true);
  view.setUint16(34, 16, true);
  writeStr(36, 'data');
  view.setUint32(40, dataSize, true);
  for (let i = 0; i < numSamples; i++) {
    view.setInt16(44 + i * 2, Math.max(-1, Math.min(1, pcm[i])) * 0x7FFF, true);
  }
  const blob = new Blob([buffer], { type: 'audio/wav' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

// ─── Replay / Download buttons ─────────────────────────────────────────────

replayBtn.addEventListener('click', () => {
  if (lastPcm) {
    playPcm(lastPcm, lastSampleRate);
    setStatus(`Replaying ${(lastPcm.length / lastSampleRate).toFixed(1)}s`);
  }
});

// Save generated audio as a reusable voice
const saveGenVoiceBtn = document.getElementById('save-gen-voice-btn');
saveGenVoiceBtn.addEventListener('click', async () => {
  if (!lastPcm) return;
  const name = prompt('Name this voice:');
  if (!name || !name.trim()) return;
  await saveVoice({
    id: 'v-' + Date.now(),
    name: name.trim(),
    refAudio: new Float32Array(lastPcm),
    refText: lastText || null,
  });
  selectedSavedVoice = (await getSavedVoices()).at(-1);
  cloneActive = true;
  cloneToggle.classList.add('on');
  await renderSavedVoices();
  updateVoiceBadge();
  setStatus(`Voice "${name.trim()}" saved`);
});

downloadBtn.addEventListener('click', () => {
  if (lastPcm) {
    const safeName = (lastText || 'omnivoice').slice(0, 40).replace(/[^a-zA-Z0-9 ]/g, '').trim().replace(/\s+/g, '_') || 'omnivoice';
    downloadWav(lastPcm, lastSampleRate, `${safeName}.wav`);
  }
});

// ─── Generation History ────────────────────────────────────────────────────

async function addToHistory(text, pcm, sampleRate, duration) {
  const item = { text, pcm: new Float32Array(pcm), sampleRate, duration, timestamp: Date.now() };
  history.unshift(item);
  if (history.length > MAX_HISTORY) history.pop();
  renderHistory();
  updateLibraryBadge();
  try { await saveHistoryItem(item); } catch (e) { console.warn('Failed to persist history:', e); }
}

function renderHistory() {
  if (history.length === 0) {
    historySectionEl.classList.add('hidden');
    return;
  }
  historySectionEl.classList.remove('hidden');
  historyScrollEl.innerHTML = '';

  history.forEach((item, idx) => {
    const card = document.createElement('div');
    card.className = 'history-card';
    card.style.animationDelay = `${idx * 0.05}s`;

    const snippet = item.text.length > 30 ? item.text.slice(0, 30) + '...' : item.text;

    // Mini waveform canvas
    const canvas = document.createElement('canvas');
    canvas.width = 360; canvas.height = 60;
    canvas.style.cssText = 'width:100%;height:30px;border-radius:6px;background:#0f131a;margin-bottom:8px;';
    drawMiniWaveform(canvas, item.pcm);

    card.innerHTML = `
      <div style="font-size:11px;color:#94a3b8;margin-bottom:4px;line-height:1.3;overflow:hidden;white-space:nowrap;text-overflow:ellipsis;">${escapeHtml(snippet)}</div>
    `;
    card.insertBefore(canvas, card.firstChild);

    const meta = document.createElement('div');
    meta.style.cssText = 'display:flex;align-items:center;justify-content:space-between;margin-top:6px;';
    meta.innerHTML = `
      <span style="font-size:10px;color:#4ade80;font-weight:600;">${item.duration.toFixed(1)}s</span>
      <div style="display:flex;gap:4px;">
        <button data-action="replay" style="padding:3px 10px;border-radius:6px;background:#273c38;color:#4ade80;border:1px solid rgba(74,222,128,0.3);font-size:10px;font-weight:600;cursor:pointer;transition:all 0.15s;">&#9654;</button>
        <button data-action="download" style="padding:3px 10px;border-radius:6px;background:#1e293b;color:#9ca3af;border:1px solid #2d3748;font-size:10px;font-weight:600;cursor:pointer;transition:all 0.15s;">&#8595;</button>
      </div>
    `;
    card.appendChild(meta);

    card.querySelector('[data-action="replay"]').addEventListener('click', (e) => {
      e.stopPropagation();
      playPcm(item.pcm, item.sampleRate);
      setStatus(`Replaying ${item.duration.toFixed(1)}s`);
    });

    card.querySelector('[data-action="download"]').addEventListener('click', (e) => {
      e.stopPropagation();
      const safeName = (item.text || 'omnivoice').slice(0, 40).replace(/[^a-zA-Z0-9 ]/g, '').trim().replace(/\s+/g, '_') || 'omnivoice';
      downloadWav(item.pcm, item.sampleRate, `${safeName}.wav`);
    });

    // Click card to replay
    card.addEventListener('click', () => {
      playPcm(item.pcm, item.sampleRate);
      setStatus(`Replaying ${item.duration.toFixed(1)}s`);
    });

    historyScrollEl.appendChild(card);
  });
}

function drawMiniWaveform(canvas, pcm) {
  drawBarVisualizer(canvas, pcm, 60);
}

function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

// ─── Decode ref audio to 24kHz mono ────────────────────────────────────────

async function decodeRefAudio(file) {
  const ctx = getAudioCtx();
  const buf = await ctx.decodeAudioData(await file.arrayBuffer());
  const numSamples = Math.round(buf.duration * 24000);
  const offline = new OfflineAudioContext(1, numSamples, 24000);
  const src = offline.createBufferSource();
  src.buffer = buf;
  src.connect(offline.destination);
  src.start();
  return (await offline.startRendering()).getChannelData(0);
}

// ─── Generate ──────────────────────────────────────────────────────────────

async function generate() {
  const text = textEl.value.trim();
  if (!text || !isReady || isGenerating) return;
  if (text.length > 500) { setStatus('Text too long (max 500 characters)'); return; }
  getAudioCtx();
  isGenerating = true;
  generateBtn.disabled = true;
  setStatus('Preparing...');
  showProgress('indeterminate');

  const instruct = buildInstruct();

  const msg = {
    type: 'synthesize',
    text,
    lang: null,
    numStep: parseInt(qualityEl.value),
    guidanceScale: 3.0,
    tShift: 0.05,
    speed: 1.0,
    instruct: instruct,
    seed: null, // no seed — let the model pick a random voice each time
  };

  // Voice cloning (saved voice or new clone)
  if (selectedSavedVoice) {
    msg.refAudio = selectedSavedVoice.refAudio;
    msg.refText = selectedSavedVoice.refText;
    msg.instruct = null;
    msg.seed = null;
  } else if (clonePanelOpen) {
    const refSource = recordedBlob || refAudioEl.files[0];
    if (refSource) {
      setStatus('Decoding reference audio...');
      msg.refAudio = await decodeRefAudio(refSource);
      msg.refText = refTextEl.value.trim() || null;
      msg.instruct = null;
      msg.seed = null;
    }
  }

  if (msg.refAudio instanceof Float32Array) msg.refAudio = new Float32Array(msg.refAudio);
  const transfers = msg.refAudio ? [msg.refAudio.buffer] : [];
  ttsWorker.postMessage(msg, transfers);
}

// ─── Mic recording ─────────────────────────────────────────────────────────

async function startRecording() {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  recordedChunks = [];
  recordedBlob = null;
  selectedSavedVoice = null;
  renderSavedVoices();
  mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm;codecs=opus' });
  mediaRecorder.ondataavailable = (e) => { if (e.data.size > 0) recordedChunks.push(e.data); };
  mediaRecorder.onstop = () => {
    stream.getTracks().forEach(t => t.stop());
    recordedBlob = new Blob(recordedChunks, { type: 'audio/webm' });
    refPreview.src = URL.createObjectURL(recordedBlob);
    refPreview.style.display = 'block';
    refAudioEl.value = '';
    recInfo.style.display = 'none';
    micBtn.classList.remove('recording');
    clearInterval(recTimer);
    updateVoiceBadge();
  };
  mediaRecorder.start();
  recInfo.style.display = 'flex';
  micBtn.classList.add('recording');
  let sec = 0;
  recTimeEl.textContent = '0s';
  recTimer = setInterval(() => {
    sec++;
    recTimeEl.textContent = `${sec}s`;
    if (sec >= 20) stopRecording();
  }, 1000);
}

function stopRecording() {
  if (mediaRecorder?.state === 'recording') mediaRecorder.stop();
}

micBtn.addEventListener('click', () => {
  if (micBtn.classList.contains('recording')) stopRecording();
  else startRecording();
});
stopBtn.addEventListener('click', stopRecording);
refAudioEl.addEventListener('change', () => {
  recordedBlob = null;
  selectedSavedVoice = null;
  refPreview.style.display = 'none';
  renderSavedVoices();
  updateVoiceBadge();
});

// ─── Save / load voices ────────────────────────────────────────────────────

saveVoiceBtn.addEventListener('click', async () => {
  const name = saveVoiceNameEl.value.trim();
  if (!name) return;
  const refSource = recordedBlob || refAudioEl.files[0];
  if (!refSource) { setStatus('Record or upload audio first'); return; }
  const pcm = await decodeRefAudio(refSource);
  await saveVoice({
    id: 'v-' + Date.now(),
    name,
    refAudio: pcm,
    refText: refTextEl.value.trim() || null,
  });
  saveVoiceNameEl.value = '';
  await renderSavedVoices();
  setStatus('Voice saved');

  // Select the new voice
  const voices = await getSavedVoices();
  if (voices.length > 0) {
    const newest = voices[voices.length - 1];
    selectedSavedVoice = newest;
    cloneActive = true;
    cloneToggle.classList.add('on');
    // Close clone panel
    clonePanelOpen = false;
    clonePanel.classList.remove('open');
    await renderSavedVoices();
    updateVoiceBadge();
  }
});

async function renderSavedVoices() {
  const voices = await getSavedVoices();
  savedVoicesEl.innerHTML = '';

  for (const v of voices) {
    const el = document.createElement('div');
    el.className = 'saved-voice' + (selectedSavedVoice?.id === v.id ? ' active' : '');
    el.innerHTML = `<span>${escapeHtml(v.name)}</span><button class="del">&times;</button>`;
    el.querySelector('.del').onclick = async (e) => {
      e.stopPropagation();
      await deleteVoice(v.id);
      if (selectedSavedVoice?.id === v.id) {
        selectedSavedVoice = null;
        cloneActive = false;
        cloneToggle.classList.remove('on');
        updateVoiceBadge();
      }
      renderSavedVoices();
    };
    el.addEventListener('click', () => {
      if (selectedSavedVoice?.id === v.id) {
        selectedSavedVoice = null;
        cloneActive = false;
        cloneToggle.classList.remove('on');
      } else {
        selectedSavedVoice = v;
        cloneActive = true;
        cloneToggle.classList.add('on');
      }
      renderSavedVoices();
      updateVoiceBadge();
    });
    savedVoicesEl.appendChild(el);
  }
}

// ─── UI ────────────────────────────────────────────────────────────────────

function setStatus(text) {
  statusEl.textContent = text;
}

generateBtn.addEventListener('click', generate);
textEl.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    generate();
  }
});

// ─── View Navigation (sidebar + mobile tabs) ─────────────────────────────

function switchView(viewName) {
  document.querySelectorAll('.app-view').forEach(v => v.style.display = 'none');
  const target = document.getElementById('view-' + viewName);
  if (target) target.style.display = 'flex';

  document.querySelectorAll('.sidebar-tab').forEach(t => t.classList.remove('active-tab'));
  const st = document.querySelector(`.sidebar-tab[data-view="${viewName}"]`);
  if (st) st.classList.add('active-tab');

  document.querySelectorAll('.mobile-tab').forEach(t => t.classList.remove('active-tab'));
  const mt = document.querySelector(`.mobile-tab[data-view="${viewName}"]`);
  if (mt) mt.classList.add('active-tab');

  if (viewName === 'library') {
    renderLibrary();
    requestAnimationFrame(() => {
      document.querySelectorAll('#library-list canvas').forEach((c, i) => {
        if (history[i]) drawMiniWaveform(c, history[i].pcm);
      });
    });
  }
  if (viewName === 'settings' && typeof calculateStorage === 'function') {
    calculateStorage();
  }
}

document.querySelectorAll('.sidebar-tab').forEach(tab => {
  tab.addEventListener('click', (e) => { e.preventDefault(); switchView(tab.dataset.view); });
});
document.querySelectorAll('.mobile-tab').forEach(tab => {
  tab.addEventListener('click', () => switchView(tab.dataset.view));
});

function renderLibrary() {
  const listEl = document.getElementById('library-list');
  const totalEl = document.getElementById('library-total');

  if (history.length === 0) {
    listEl.innerHTML = '<div class="text-center text-omni-text-muted text-sm py-12">No generations yet. Create something in the Studio!</div>';
    totalEl.textContent = '0 generations';
    return;
  }

  totalEl.textContent = `${history.length} generation${history.length !== 1 ? 's' : ''}`;
  listEl.innerHTML = '';

  history.forEach((item) => {
    const el = document.createElement('div');
    el.className = 'library-item';

    const snippet = item.text.length > 80 ? item.text.slice(0, 80) + '...' : item.text;
    const diff = Date.now() - item.timestamp;
    const timeAgo = diff < 60000 ? 'just now' : diff < 3600000 ? `${Math.floor(diff/60000)}m ago` : diff < 86400000 ? `${Math.floor(diff/3600000)}h ago` : new Date(item.timestamp).toLocaleDateString();

    // Build info section first
    const info = document.createElement('div');
    info.style.cssText = 'display:flex;align-items:center;justify-content:space-between;gap:12px;';
    info.innerHTML = `
      <div style="flex:1;min-width:0;">
        <div style="font-size:13px;color:#e5e7eb;line-height:1.4;margin-bottom:4px;overflow:hidden;white-space:nowrap;text-overflow:ellipsis;">${escapeHtml(snippet)}</div>
        <div style="font-size:11px;color:#64748b;">${item.duration.toFixed(1)}s &middot; ${timeAgo}</div>
      </div>
      <div style="display:flex;gap:6px;flex-shrink:0;">
        <button data-action="replay" style="padding:6px 14px;border-radius:8px;background:#273c38;color:#4ade80;border:1px solid rgba(74,222,128,0.3);font-size:12px;font-weight:600;cursor:pointer;transition:all 0.15s;">&#9654; Play</button>
        <button data-action="download" style="padding:6px 14px;border-radius:8px;background:#1e293b;color:#9ca3af;border:1px solid #2d3748;font-size:12px;font-weight:600;cursor:pointer;transition:all 0.15s;">&#8595; WAV</button>
      </div>
    `;

    // Canvas AFTER innerHTML so it doesn't get destroyed
    const canvas = document.createElement('canvas');
    canvas.width = 600; canvas.height = 60;
    canvas.style.cssText = 'width:100%;height:36px;border-radius:8px;background:#0a0e14;margin-bottom:10px;';

    el.appendChild(canvas);
    el.appendChild(info);

    info.querySelector('[data-action="replay"]').addEventListener('click', () => {
      playPcm(item.pcm, item.sampleRate);
      setStatus(`Replaying ${item.duration.toFixed(1)}s`);
    });
    info.querySelector('[data-action="download"]').addEventListener('click', () => {
      const safeName = (item.text || 'omnivoice').slice(0, 40).replace(/[^a-zA-Z0-9 ]/g, '').trim().replace(/\s+/g, '_') || 'omnivoice';
      downloadWav(item.pcm, item.sampleRate, `${safeName}.wav`);
    });

    // Append to DOM first, then draw (canvas needs real dimensions)
    listEl.appendChild(el);
    // drawMiniWaveform uses canvas.width/height (set explicitly), not offsetWidth
    drawMiniWaveform(canvas, item.pcm);
  });
}

function updateLibraryBadge() {
  const b = document.getElementById('library-count');
  if (b) { b.textContent = history.length; b.classList.toggle('has-items', history.length > 0); }
}

// ─── Init ──────────────────────────────────────────────────────────────────

(async () => {
  await renderSavedVoices();
  updateVoiceBadge();
  // Load persisted history
  try {
    history = await loadHistory();
    renderHistory();
    updateLibraryBadge();
  } catch (e) { console.warn('Failed to load history:', e); }
  initWorker();
})();
