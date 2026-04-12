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
const loadingOverlay = document.getElementById('loading-overlay');
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
      updateTimeEstimate();
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

  // Update button text based on voice mode (badge removed, button text is enough)
  if (selectedSavedVoice) {
    generateBtn.textContent = `Generate as "${selectedSavedVoice.name}"`;
  } else if (clonePanelOpen && (recordedBlob || (refAudioEl.files && refAudioEl.files.length > 0))) {
    generateBtn.textContent = 'Generate with Clone';
  } else {
    generateBtn.textContent = 'Generate Speech';
  }
  updateTimeEstimate();
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

// ─── Ready handler (backend badge, CPU warning, enable UI) ──���─────────────

function enableUI() {
  isReady = true;
  setStatus('Ready');
  hideProgress();
  if (loadingOverlay) {
    loadingOverlay.style.transition = 'opacity 0.3s';
    loadingOverlay.style.opacity = '0';
    setTimeout(() => { loadingOverlay.remove(); }, 300);
  }
  generateBtn.disabled = false;
  textEl.disabled = false;
  refAudioEl.disabled = false;
  refTextEl.disabled = false;
  micBtn.disabled = false;
  saveVoiceNameEl.disabled = false;
  saveVoiceBtn.disabled = false;
}

function onReady(backend) {
  // Show backend badge
  const badge = document.getElementById('backend-badge');
  if (badge) {
    const isGpu = backend === 'webgpu';
    badge.textContent = isGpu ? 'GPU' : 'CPU';
    badge.classList.remove('hidden');
    if (isGpu) {
      badge.style.color = '#4ade80';
      badge.style.borderColor = 'rgba(74,222,128,0.4)';
      badge.style.background = 'rgba(74,222,128,0.1)';
    } else {
      badge.style.color = '#f59e0b';
      badge.style.borderColor = 'rgba(245,158,11,0.4)';
      badge.style.background = 'rgba(245,158,11,0.1)';
    }
  }

  enableUI();
}

// ─── Worker ───────────────────────────────────────────────────────���────────

function initWorker() {
  ttsWorker = new Worker('workers/tts-worker.js?v=3', { type: 'module' });
  ttsWorker.onmessage = (e) => {
    const msg = e.data;
    switch (msg.type) {
      case 'progress':
        setStatus(msg.detail);
        // Try to parse step progress for bar + track ms/step
        const stepMatch = msg.detail?.match?.(/Step (\d+)\/(\d+) \((\d+)ms\)/);
        if (stepMatch) {
          const [, current, total, stepMs] = stepMatch;
          setProgressPercent((parseInt(current) / parseInt(total)) * 100);
          msPerStep = parseInt(stepMs);
        } else {
          const genericMatch = msg.detail?.match?.(/(\d+)\s*\/\s*(\d+)/);
          if (genericMatch) {
            const [, current, total] = genericMatch;
            setProgressPercent((parseInt(current) / parseInt(total)) * 100);
          } else {
            showProgress('indeterminate');
          }
        }
        break;
      case 'ready':
        onReady(msg.backend);
        break;
      case 'audio':
        onAudio(msg.pcm, msg.sampleRate);
        break;
      case 'error':
        setStatus(msg.message);
        console.error(msg.message);
        isGenerating = false;
        setGenerating(false);
        hideProgress();
        break;
    }
  };
  ttsWorker.postMessage({ type: 'init', modelBaseUrl: MODEL_BASE_URL, forceCPU: location.search.includes('cpu') });
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
    void studioPlayer.offsetHeight; // reflow so canvas gets real dimensions
  }

  drawWaveform(pcm);
  playerDuration.textContent = duration.toFixed(1) + 's';

  // Play audio and scroll into view
  playPcm(pcm, sampleRate);
  if (studioPlayer) studioPlayer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

  setStatus(`Playing ${duration.toFixed(1)}s`);
  hideProgress();
  isGenerating = false;
  setGenerating(false);
}

function updateReplayBtn() {
  if (!replayBtn) return;
  if (currentSource) {
    replayBtn.innerHTML = '&#9632; Stop';
    replayBtn.classList.add('bg-red-500/20', 'border-red-400/40', 'text-red-400');
    replayBtn.classList.remove('bg-omni-active-bg', 'text-omni-neon', 'border-omni-neon/40', 'shadow-neon');
  } else {
    replayBtn.innerHTML = '&#9654; Replay';
    replayBtn.classList.remove('bg-red-500/20', 'border-red-400/40', 'text-red-400');
    replayBtn.classList.add('bg-omni-active-bg', 'text-omni-neon', 'border-omni-neon/40', 'shadow-neon');
  }
}

function stopPlayback() {
  if (currentSource) {
    try { currentSource.stop(); } catch (e) { /* ignore */ }
    currentSource = null;
    updateReplayBtn();
    if (!isGenerating) setStatus('Ready');
  }
}

function playPcm(pcm, sampleRate) {
  stopPlayback();

  const ctx = getAudioCtx();
  const buf = ctx.createBuffer(1, pcm.length, sampleRate);
  buf.getChannelData(0).set(pcm);
  const src = ctx.createBufferSource();
  src.buffer = buf;
  src.connect(ctx.destination);
  src.onended = () => {
    currentSource = null;
    updateReplayBtn();
    if (!isGenerating) setStatus('Ready');
  };
  currentSource = src;
  src.start();
  updateReplayBtn();
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

// ─── MP3 Download (with ID3v2 provenance metadata) ─────────────────────────

function buildId3v2Tag() {
  const enc = new TextEncoder();
  // ID3v2.3 text frames: [frameId, text]
  const textFrames = [
    ['TSSE', 'VocoLoco (OmniVoice TTS, browser-based)'],
    ['TCON', 'AI-generated speech'],
    ['TDRC', new Date().toISOString()],
  ];
  // COMM frame (comment) has special structure
  const comment = 'AI-generated synthetic speech. This audio was produced entirely by an artificial intelligence text-to-speech model. EU AI Act Art. 50 \u2014 not a recording of a human voice.';

  // Calculate total size
  let framesSize = 0;
  const builtFrames = [];
  for (const [id, text] of textFrames) {
    const textBytes = enc.encode(text);
    const frameDataSize = 1 + textBytes.length; // encoding byte + text
    framesSize += 10 + frameDataSize;
    builtFrames.push({ id, encoding: 3, data: textBytes, size: frameDataSize }); // 3 = UTF-8
  }
  // COMM frame: encoding(1) + lang(3) + short desc null-term + text
  const commBytes = enc.encode(comment);
  const commDataSize = 1 + 3 + 1 + commBytes.length; // encoding + "eng" + \0 (empty short desc) + text
  framesSize += 10 + commDataSize;

  const totalSize = 10 + framesSize; // ID3 header + frames
  const buf = new Uint8Array(totalSize);
  const view = new DataView(buf.buffer);

  // ID3v2.3 header
  buf[0] = 0x49; buf[1] = 0x44; buf[2] = 0x33; // "ID3"
  buf[3] = 3; buf[4] = 0; // version 2.3
  buf[5] = 0; // flags
  // Size as syncsafe integer (28 bits across 4 bytes, MSB of each byte is 0)
  const s = framesSize;
  buf[6] = (s >> 21) & 0x7F;
  buf[7] = (s >> 14) & 0x7F;
  buf[8] = (s >> 7) & 0x7F;
  buf[9] = s & 0x7F;

  let off = 10;
  // Write text frames
  for (const frame of builtFrames) {
    buf.set(enc.encode(frame.id), off); // frame ID (4 bytes)
    view.setUint32(off + 4, frame.size); // size (big-endian)
    buf[off + 8] = 0; buf[off + 9] = 0; // flags
    buf[off + 10] = frame.encoding; // UTF-8
    buf.set(frame.data, off + 11);
    off += 10 + frame.size;
  }
  // Write COMM frame
  buf.set(enc.encode('COMM'), off);
  view.setUint32(off + 4, commDataSize);
  buf[off + 8] = 0; buf[off + 9] = 0;
  buf[off + 10] = 3; // UTF-8
  buf[off + 11] = 0x65; buf[off + 12] = 0x6E; buf[off + 13] = 0x67; // "eng"
  buf[off + 14] = 0; // empty short description (null terminator)
  buf.set(commBytes, off + 15);

  return buf;
}

function downloadMp3(pcm, sampleRate, filename) {
  // Convert Float32 PCM to Int16
  const samples = new Int16Array(pcm.length);
  for (let i = 0; i < pcm.length; i++) {
    samples[i] = Math.max(-32768, Math.min(32767, Math.round(pcm[i] * 32767)));
  }

  // Encode MP3 using lamejs (mono, 128kbps)
  const encoder = new lamejs.Mp3Encoder(1, sampleRate, 128);
  const chunkSize = 1152;
  const mp3Parts = [];
  for (let i = 0; i < samples.length; i += chunkSize) {
    const chunk = samples.subarray(i, Math.min(i + chunkSize, samples.length));
    const mp3buf = encoder.encodeBuffer(chunk);
    if (mp3buf.length > 0) mp3Parts.push(mp3buf);
  }
  const flush = encoder.flush();
  if (flush.length > 0) mp3Parts.push(flush);

  // Build ID3v2 tag
  const id3 = buildId3v2Tag();

  // Concatenate: ID3 tag + MP3 data
  const blob = new Blob([id3, ...mp3Parts], { type: 'audio/mpeg' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

// ─── Replay / Download buttons ─────────────────────────────────────────────

replayBtn.addEventListener('click', () => {
  if (currentSource) {
    stopPlayback();
    return;
  }
  if (lastPcm) {
    playPcm(lastPcm, lastSampleRate);
    setStatus(`Playing ${(lastPcm.length / lastSampleRate).toFixed(1)}s`);
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
    const words = (lastText || 'vocoloco').slice(0, 50).replace(/[^a-zA-Z0-9 ]/g, '').trim().replace(/\s+/g, '_') || 'vocoloco';
    const ts = new Date().toISOString().slice(0, 16).replace(/[T:]/g, '-');
    downloadMp3(lastPcm, lastSampleRate, `${words}_${ts}.mp3`);
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
      const words = (item.text || 'vocoloco').slice(0, 50).replace(/[^a-zA-Z0-9 ]/g, '').trim().replace(/\s+/g, '_') || 'vocoloco';
      const ts = new Date(item.timestamp).toISOString().slice(0, 16).replace(/[T:]/g, '-');
      downloadMp3(item.pcm, item.sampleRate, `${words}_${ts}.mp3`);
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
  const pcm = (await offline.startRendering()).getChannelData(0);

  // Trim silence from both ends (same logic as postProcessAudio in worker)
  const thresh = 0.005, margin = Math.floor(24000 * 0.02);
  let start = 0, end = pcm.length;
  for (let i = 0; i < pcm.length; i++) if (Math.abs(pcm[i]) > thresh) { start = Math.max(0, i - margin); break; }
  for (let i = pcm.length - 1; i >= 0; i--) if (Math.abs(pcm[i]) > thresh) { end = Math.min(pcm.length, i + margin); break; }
  return pcm.slice(start, end);
}

// ─── Generate ──────────────────────────────────────────────────────────────

const timeEstimateEl = document.getElementById('time-estimate');
const generateBtnDefaultClasses = generateBtn.className; // save original classes

// Estimated ms/step based on last generation (updated after each run)
let msPerStep = null;

function updateTimeEstimate() {
  if (!timeEstimateEl) return;
  const text = textEl.value.trim();
  const steps = parseInt(qualityEl.value);
  if (!text || !msPerStep) { timeEstimateEl.classList.add('hidden'); return; }
  const estSec = (steps * msPerStep / 1000).toFixed(0);
  timeEstimateEl.textContent = `Estimated: ~${estSec}s`;
  timeEstimateEl.classList.remove('hidden');
}

function setGenerating(active) {
  if (active) {
    // Transform generate button into cancel button
    generateBtn.className = generateBtnDefaultClasses;
    generateBtn.classList.add('btn-cancel');
    generateBtn.textContent = 'Cancel';
    generateBtn.disabled = false; // must be clickable as cancel
    generateBtn.onclick = cancelGeneration;
    timeEstimateEl.classList.add('hidden');
  } else {
    // Restore generate button
    generateBtn.className = generateBtnDefaultClasses;
    generateBtn.onclick = null; // remove cancel handler, use the addEventListener below
    generateBtn.disabled = !isReady;
    updateVoiceBadge(); // restores button text
  }
}

function cancelGeneration() {
  if (!isGenerating) return;
  ttsWorker.terminate();
  isGenerating = false;
  isReady = false; // worker is being recreated
  setGenerating(false);
  generateBtn.disabled = true; // disabled until new worker is ready
  hideProgress();
  setStatus('Cancelled — reloading...');
  initWorker();
}

async function generate() {
  const text = textEl.value.trim();
  if (!text || !isReady || isGenerating) return;
  if (text.length > 500) { setStatus('Text too long (max 500 characters)'); return; }

  // Warn when history is at capacity
  if (history.length >= MAX_HISTORY) {
    const proceed = confirm(`You have ${MAX_HISTORY} generations stored. Generating a new one will remove the oldest. Save any you want to keep first.\n\nContinue?`);
    if (!proceed) return;
  }

  getAudioCtx();
  isGenerating = true;
  setGenerating(true);
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

// ─── Clone panel helpers ──────────────────────────────────────────────────

const cloneTips = document.getElementById('clone-tips');
const cloneAudioInfo = document.getElementById('clone-audio-info');
const cloneDuration = document.getElementById('clone-duration');
const cloneDurationWarn = document.getElementById('clone-duration-warn');
const cloneSaveSection = document.getElementById('clone-save-section');
const cloneClearBtn = document.getElementById('clone-clear-btn');
const micLabel = document.getElementById('mic-label');
const recHint = document.getElementById('rec-hint');

function showCloneAudio(blob) {
  refPreview.src = URL.createObjectURL(blob);
  // Decode to get trimmed duration and validate
  decodeRefAudio(blob).then(pcm => {
    const dur = pcm.length / 24000;
    cloneDuration.textContent = `${dur.toFixed(1)}s (trimmed)`;
    // Duration warnings
    if (dur < 3) {
      cloneDurationWarn.textContent = 'Recording too short — aim for 5-10 seconds for best results.';
      cloneDurationWarn.classList.remove('hidden');
    } else if (dur > 15) {
      cloneDurationWarn.textContent = 'Recording is long — only the first ~10s improve quality, the rest increases generation time.';
      cloneDurationWarn.classList.remove('hidden');
    } else {
      cloneDurationWarn.classList.add('hidden');
    }
  });
  // Show audio info, hide tips
  cloneAudioInfo.classList.remove('hidden');
  cloneSaveSection.classList.remove('hidden');
  if (cloneTips) cloneTips.classList.add('hidden');
  // Auto-focus name input
  saveVoiceNameEl.focus();
  updateVoiceBadge();
}

function clearCloneAudio() {
  recordedBlob = null;
  refAudioEl.value = '';
  selectedSavedVoice = null;
  refPreview.src = '';
  cloneAudioInfo.classList.add('hidden');
  cloneSaveSection.classList.add('hidden');
  cloneDurationWarn.classList.add('hidden');
  if (cloneTips) cloneTips.classList.remove('hidden');
  renderSavedVoices();
  updateVoiceBadge();
}

cloneClearBtn.addEventListener('click', clearCloneAudio);

// ─── Mic recording ─────────────────────────────────────────────────────────

let countdownActive = false;

async function startRecording() {
  // Get mic permission first (prompt appears immediately)
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

  // 3-second countdown overlay
  countdownActive = true;
  micBtn.disabled = true;
  const panel = document.getElementById('clone-panel');
  const overlay = document.createElement('div');
  overlay.style.cssText = 'position:absolute;inset:0;z-index:20;display:flex;flex-direction:column;align-items:center;justify-content:center;background:rgba(15,19,26,0.92);border-radius:12px;backdrop-filter:blur(4px);';
  overlay.innerHTML = '<div id="countdown-num" style="font-size:64px;font-weight:800;color:#4ade80;line-height:1;transition:transform 0.15s,opacity 0.15s;">3</div><div style="color:#94a3b8;font-size:13px;margin-top:8px;">Get ready to speak...</div><button id="countdown-cancel" style="margin-top:16px;color:#64748b;font-size:12px;cursor:pointer;background:none;border:none;text-decoration:underline;">Cancel</button>';
  panel.style.position = 'relative';
  panel.querySelector('div').appendChild(overlay);
  const numEl = document.getElementById('countdown-num');
  document.getElementById('countdown-cancel').addEventListener('click', () => { countdownActive = false; });
  for (let i = 3; i >= 1; i--) {
    numEl.textContent = i;
    numEl.style.transform = 'scale(1.3)'; numEl.style.opacity = '0.5';
    requestAnimationFrame(() => { numEl.style.transform = 'scale(1)'; numEl.style.opacity = '1'; });
    await new Promise(r => setTimeout(r, 1000));
    if (!countdownActive) { overlay.remove(); stream.getTracks().forEach(t => t.stop()); micBtn.disabled = false; return; }
  }
  overlay.remove();
  countdownActive = false;
  micBtn.disabled = false;

  // Start recording
  recordedChunks = [];
  recordedBlob = null;
  selectedSavedVoice = null;
  renderSavedVoices();
  mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm;codecs=opus' });
  mediaRecorder.ondataavailable = (e) => { if (e.data.size > 0) recordedChunks.push(e.data); };
  mediaRecorder.onstop = () => {
    stream.getTracks().forEach(t => t.stop());
    recordedBlob = new Blob(recordedChunks, { type: 'audio/webm' });
    recInfo.classList.add('hidden');
    micBtn.classList.remove('recording');
    micLabel.textContent = 'Record';
    clearInterval(recTimer);
    showCloneAudio(recordedBlob);
  };
  mediaRecorder.start();
  recInfo.classList.remove('hidden');
  micBtn.classList.add('recording');
  micLabel.textContent = 'Recording...';
  if (recHint) recHint.textContent = '';
  let sec = 0;
  recTimeEl.textContent = '0s';
  recTimer = setInterval(() => {
    sec++;
    recTimeEl.textContent = `${sec}s`;
    if (sec === 10 && recHint) recHint.textContent = 'Optimal length reached';
    if (sec >= 15) stopRecording();
  }, 1000);
}

function stopRecording() {
  if (mediaRecorder?.state === 'recording') mediaRecorder.stop();
}

micBtn.addEventListener('click', () => {
  if (countdownActive) { countdownActive = false; return; }
  if (micBtn.classList.contains('recording')) stopRecording();
  else startRecording();
});
stopBtn.addEventListener('click', stopRecording);
refAudioEl.addEventListener('change', () => {
  recordedBlob = null;
  selectedSavedVoice = null;
  const file = refAudioEl.files[0];
  if (file) showCloneAudio(file);
  else clearCloneAudio();
});

// ─── Save / load voices ────────────────────────────────────────────────────

saveVoiceBtn.addEventListener('click', async () => {
  const name = saveVoiceNameEl.value.trim();
  if (!name) { setStatus('Add a name to save this voice'); saveVoiceNameEl.focus(); return; }
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
  setStatus(`Voice "${name}" saved`);

  // Select the new voice and close panel
  const voices = await getSavedVoices();
  if (voices.length > 0) {
    const newest = voices[voices.length - 1];
    selectedSavedVoice = newest;
    cloneActive = true;
    cloneToggle.classList.add('on');
    clonePanelOpen = false;
    clonePanel.classList.remove('open');
    clearCloneAudio();
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

// Ctrl/Cmd+Enter to generate
textEl.addEventListener('keydown', (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
    e.preventDefault();
    generate();
  }
});

// Update time estimate as user types
textEl.addEventListener('input', updateTimeEstimate);
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
        <button data-action="download" style="padding:6px 14px;border-radius:8px;background:#1e293b;color:#9ca3af;border:1px solid #2d3748;font-size:12px;font-weight:600;cursor:pointer;transition:all 0.15s;">&#8595; MP3</button>
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
      const words = (item.text || 'vocoloco').slice(0, 50).replace(/[^a-zA-Z0-9 ]/g, '').trim().replace(/\s+/g, '_') || 'vocoloco';
      const ts = new Date(item.timestamp).toISOString().slice(0, 16).replace(/[T:]/g, '-');
      downloadMp3(item.pcm, item.sampleRate, `${words}_${ts}.mp3`);
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
