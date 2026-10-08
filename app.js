/**
 * VocoLoco — Main application
 * Streams generation sentence-by-sentence, manages voices (guided recording
 * wizard + cached voice tokens), the library, settings, and all studio UI.
 */

import { toast, confirmDialog, isDialogOpen } from './ui-dialogs.js';
import { StreamingPlayer, drawBarVisualizer, drawMiniWaveform } from './player.js';
import { chunkText } from './text-chunker.js';
import { TrimEditor } from './trim-editor.js';
import { prepareReference, defaultSelection, REF_PREP_VERSION } from './audio-postprocess.js?v=2';

const MODEL_BASE_URL = 'https://huggingface.co/Gigsu/vocoloco-onnx/resolve/main';
// Local dev: a complete mirror in ./models (scripts/fetch-models.sh) replaces
// Hugging Face, so reloads don't re-download ~3 GB. Never used in production.
const LOCAL_MODEL_URL = new URL('models', location.href).href;
const MAX_TEXT_LEN = 2000;
// OmniVoice's sampling defaults (num_step comes from the quality toggle)
const GEN_PARAMS = { guidanceScale: 2.0, tShift: 0.1 };

// ─── Guided recording scripts ───────────────────────────────────────────────
// Reading one of these aloud makes the transcript exact by construction.

const VOICE_SCRIPTS = [
  { id: 'en-village', lang: 'en', label: 'Village morning',
    text: 'The quick autumn breeze swept golden leaves across the quiet village square. Somewhere nearby a church bell rang twice, and the smell of fresh bread drifted over from the bakery.' },
  { id: 'en-storm', lang: 'en', label: 'Thunderstorm',
    text: 'Have you ever watched a thunderstorm roll in from the sea? First the air goes completely still, then a low rumble builds, and suddenly bright flashes dance across the horizon.' },
  { id: 'en-coffee', lang: 'en', label: 'Morning routine',
    text: 'I usually start my day with a strong cup of coffee and a short walk outside. It clears my head, wakes up my voice, and puts me in a surprisingly good mood.' },
  { id: 'de-markt', lang: 'de', label: 'Auf dem Marktplatz',
    text: 'Der alte Marktplatz liegt ruhig in der Morgensonne. Ein Duft von frischem Brot und Kaffee zieht durch die Gassen, während die ersten Händler ihre bunten Stände aufbauen.' },
  { id: 'de-gewitter', lang: 'de', label: 'Sommergewitter',
    text: 'Hast du schon einmal ein Sommergewitter beobachtet? Zuerst wird die Luft ganz still, dann grollt der Donner leise in der Ferne, und plötzlich zucken helle Blitze über den Himmel.' },
];

const TEST_SENTENCES = {
  en: "Hey, this is my new cloned voice. I think it sounds pretty close, don't you?",
  de: 'Hallo, das ist meine neue geklonte Stimme. Klingt ziemlich echt, oder?',
};

// ─── DOM ────────────────────────────────────────────────────────────────────

const $ = (id) => document.getElementById(id);

const textEl = $('text-input');
const qualityEl = $('quality');
const generateBtn = $('generate-btn');
const statusEl = $('status');
const waveformEl = $('waveform');
const genderRow = $('gender-row');
const pitchRow = $('pitch-row');
const qualityRow = $('quality-row');
const charCount = $('char-count');
const progressBar = $('progress-bar');
const replayBtn = $('replay-btn');
const downloadBtn = $('download-btn');
const saveGenVoiceBtn = $('save-gen-voice-btn');
const voiceLockBtn = $('voice-lock-btn');
const playerControls = $('player-controls');
const playerDuration = $('player-duration');
const loadingOverlay = $('loading-overlay');
const loadingTitle = $('loading-title');
const loadingSub = $('loading-sub');
const loadingTrack = $('loading-track');
const loadingFill = $('loading-fill');
const loadingStats = $('loading-stats');
const studioVoicePicker = $('studio-voice-picker');
const manageVoicesLink = $('manage-voices-link');

// Voices view + wizard
const voicesHome = $('voices-home');
const voicesList = $('voices-list');
const voicesEmpty = $('voices-empty');
const voicesEncoderNote = $('voices-encoder-note');
const newVoiceBtn = $('new-voice-btn');
const voicesEmptyCta = $('voices-empty-cta');
const voiceWizard = $('voice-wizard');
const wizTitle = $('wiz-title');
const wizBackBtn = $('wiz-back-btn');
const wizCloseBtn = $('wiz-close-btn');
const wizMethodRecord = $('wiz-method-record');
const wizMethodUpload = $('wiz-method-upload');
const wizScriptList = $('wiz-script-list');
const wizCustomWrap = $('wiz-custom-wrap');
const wizCustomText = $('wiz-custom-text');
const wizScriptNext = $('wiz-script-next');
const wizMicStatus = $('wiz-mic-status');
const wizMicSelect = $('wiz-mic-select');
const wizMicError = $('wiz-mic-error');
const wizMicErrorText = $('wiz-mic-error-text');
const wizMicRetry = $('wiz-mic-retry');
const wizMicFallbackUpload = $('wiz-mic-fallback-upload');
const wizScriptDisplay = $('wiz-script-display');
const wizLiveWave = $('wiz-live-wave');
const wizLevelFill = $('wiz-level-fill');
const wizRecFill = $('wiz-rec-fill');
const wizRecElapsed = $('wiz-rec-elapsed');
const wizRecZone = $('wiz-rec-zone');
const wizSilenceHint = $('wiz-silence-hint');
const wizRecordBtn = $('wiz-record-btn');
const wizRecordBtnInner = $('wiz-record-btn-inner');
const wizRecordLabel = $('wiz-record-label');
const wizDropzone = $('wiz-dropzone');
const wizFileInput = $('wiz-file-input');
const wizUploadError = $('wiz-upload-error');
const wizReviewWave = $('wiz-review-wave');
const wizReviewPlay = $('wiz-review-play');
const wizReviewDuration = $('wiz-review-duration');
const wizReviewVerdict = $('wiz-review-verdict');
const wizTrimOverview = $('wiz-trim-overview');
const wizTrimInfo = $('wiz-trim-info');
const wizTranscript = $('wiz-transcript');
const wizTranscriptHint = $('wiz-transcript-hint');
const wizVoiceName = $('wiz-voice-name');
const wizReviewBack = $('wiz-review-back');
const wizSaveBtn = $('wiz-save-btn');
const wizSaveBlocker = $('wiz-save-blocker');
const wizSavingStatus = $('wiz-saving-status');
const wizDoneSub = $('wiz-done-sub');
const wizDoneNote = $('wiz-done-note');
const wizTestBtn = $('wiz-test-btn');
const wizGotoStudio = $('wiz-goto-studio');

const wizRecordBtnIdleHTML = wizRecordBtnInner ? wizRecordBtnInner.innerHTML : '';
const WIZ_RECORD_IDLE_LABEL = 'Tap to record — starts after a 3-2-1 countdown';
const WIZ_ZONE_IDLE_LABEL = 'Ready when you are — aim for 5-12 seconds';

// ─── State ──────────────────────────────────────────────────────────────────

let ttsWorker = null;
let isReady = false;
let isGenerating = false;
let encoderAvailable = true;
let selectedSavedVoice = null;
let voicesCache = [];
let currentView = 'studio';

// Streaming generation state
let stream = null;
let streamCounter = 0;
let sessionChainRef = null; // { tokens, tokenCount, text, pcm } from the last design-mode generation
let voiceLocked = false;
let lastGen = null;         // { voiceName, first: { pcm, text }, chainRef }

// Last generated audio for replay/download
let lastPcm = null;
let lastSampleRate = 24000;
let lastText = '';

// Generation history (persisted in IndexedDB)
let history = [];
const MAX_HISTORY = 50;
const HISTORY_DB = 'omnivoice-history';
const VOICE_DB = 'omnivoice-voices';
const QUALITY_LABELS = { 8: 'Fast', 16: 'Good', 20: 'High', 32: 'Best' };

// ─── IndexedDB plumbing ─────────────────────────────────────────────────────
// Open (or create/heal) a database that must contain `storeName`. Handles a
// DB that exists without the store (e.g. recreated by a versionless open
// after a delete) by bumping the version. Closes on versionchange so a
// pending deleteDatabase can't dead-lock behind this connection.
function openStoreDB(dbName, storeName, options) {
  return new Promise((resolve, reject) => {
    const attempt = (version) => {
      const req = version ? indexedDB.open(dbName, version) : indexedDB.open(dbName);
      req.onupgradeneeded = () => {
        if (!req.result.objectStoreNames.contains(storeName)) {
          req.result.createObjectStore(storeName, options);
        }
      };
      req.onsuccess = () => {
        const db = req.result;
        db.onversionchange = () => db.close();
        if (db.objectStoreNames.contains(storeName)) { resolve(db); return; }
        const next = db.version + 1; // self-heal: DB exists but store is missing
        db.close();
        attempt(next);
      };
      req.onerror = () => reject(req.error);
      req.onblocked = () => { /* waits for other connections; they close on versionchange */ };
    };
    attempt();
  });
}

// One transaction per call, connection CLOSED afterwards — leaked connections
// would block deleteDatabase forever. `fn(store, setResult)` may register
// request handlers; the promise resolves with whatever setResult captured.
async function withStore(dbName, storeName, options, mode, fn) {
  const db = await openStoreDB(dbName, storeName, options);
  return new Promise((resolve, reject) => {
    let out;
    const tx = db.transaction(storeName, mode);
    tx.oncomplete = () => { db.close(); resolve(out); };
    tx.onerror = () => { db.close(); reject(tx.error); };
    tx.onabort = () => { db.close(); reject(tx.error || new Error('transaction aborted')); };
    try {
      fn(tx.objectStore(storeName), (v) => { out = v; });
    } catch (e) {
      try { tx.abort(); } catch { /* already aborted */ }
      db.close();
      reject(e);
    }
  });
}

const historyStore = (mode, fn) => withStore(HISTORY_DB, 'items', { keyPath: 'id', autoIncrement: true }, mode, fn);
const voicesStore = (mode, fn) => withStore(VOICE_DB, 'voices', { keyPath: 'id' }, mode, fn);

// ─── History IndexedDB ──────────────────────────────────────────────────────

async function loadHistory() {
  const rows = await historyStore('readonly', (store, setResult) => {
    const req = store.getAll();
    req.onsuccess = () => setResult(req.result);
  });
  const items = (rows || []).map(item => ({
    ...item,
    // Read-path compat: legacy rows stored pcm as number[]
    pcm: item.pcm instanceof Float32Array ? item.pcm : new Float32Array(item.pcm),
  }));
  items.sort((a, b) => b.timestamp - a.timestamp);
  return items.slice(0, MAX_HISTORY);
}

async function saveHistoryItem(item) {
  await historyStore('readwrite', (store) => {
    const req = store.put({
      text: item.text,
      pcm: item.pcm, // Float32Array — IndexedDB structured clone handles typed arrays
      sampleRate: item.sampleRate,
      duration: item.duration,
      timestamp: item.timestamp,
      voiceName: item.voiceName ?? null,
      quality: item.quality ?? null,
      cancelled: !!item.cancelled,
    });
    req.onsuccess = () => { item.id = req.result; };
  });
}

async function deleteHistoryItem(id) {
  await historyStore('readwrite', (store) => store.delete(id));
}

async function clearHistoryStore() {
  await historyStore('readwrite', (store) => store.clear());
}

// Keep the DB bounded: delete oldest rows beyond `max` (autoIncrement keys
// grow chronologically, so the smallest keys are the oldest generations).
async function pruneHistoryDB(max = MAX_HISTORY) {
  try {
    await historyStore('readwrite', (store) => {
      const countReq = store.count();
      countReq.onsuccess = () => {
        let excess = countReq.result - max;
        if (excess <= 0) return;
        const cursorReq = store.openCursor();
        cursorReq.onsuccess = () => {
          const cur = cursorReq.result;
          if (cur && excess > 0) { cur.delete(); excess--; cur.continue(); }
        };
      };
    });
  } catch (e) { console.warn('History prune failed:', e); }
}

// ─── Voices IndexedDB (schema v2 with lazy migration) ──────────────────────

async function getSavedVoices() {
  const rows = await voicesStore('readonly', (store, setResult) => {
    const req = store.getAll();
    req.onsuccess = () => setResult(req.result);
  });
  return rows || [];
}

async function saveVoice(voice) {
  await voicesStore('readwrite', (store) => store.put(voice));
}

async function deleteVoice(id) {
  await voicesStore('readwrite', (store) => store.delete(id));
}

async function clearVoicesStore() {
  await voicesStore('readwrite', (store) => store.clear());
}

// Persist freshly encoded tokens WITHOUT clobbering concurrent edits (e.g. a
// rename committed while a slow CPU encode was running): re-fetch and merge.
async function persistVoiceTokens(voiceId, tokens, tokenCount, extra = null) {
  const fresh = (await getSavedVoices()).find(v => v.id === voiceId);
  if (!fresh) return null; // deleted meanwhile
  const record = normalizeVoiceRecord(fresh);
  Object.assign(record, extra, { tokens, tokenCount });
  await saveVoice(record);
  const cached = voicesCache.find(v => v.id === voiceId);
  if (cached) Object.assign(cached, extra, { tokens, tokenCount });
  return record;
}

// Fill v1 records (id/name/refAudio/refText) up to the v2 shape in memory.
function normalizeVoiceRecord(v) {
  if (!v) return v;
  if (v.tokens === undefined) v.tokens = null;
  if (v.tokenCount === undefined) v.tokenCount = v.tokens ? Math.floor(v.tokens.length / 8) : null;
  if (!v.duration) v.duration = v.refAudio ? v.refAudio.length / 24000 : 0;
  if (!v.createdAt) {
    const ts = parseInt(String(v.id).slice(2), 10);
    v.createdAt = Number.isFinite(ts) && ts > 0 ? ts : Date.now();
  }
  if (v.scriptId === undefined) v.scriptId = null;
  if (v.lang === undefined) v.lang = null;
  if (!v.source) v.source = 'uploaded';
  v.version = 2;
  return v;
}

async function refreshVoices() {
  try {
    const raw = await getSavedVoices();
    voicesCache = raw.map(normalizeVoiceRecord).sort((a, b) => b.createdAt - a.createdAt);
  } catch (e) {
    console.warn('Failed to load voices:', e);
    voicesCache = [];
  }
  // Keep the selection pointing at the fresh record object
  if (selectedSavedVoice) {
    selectedSavedVoice = voicesCache.find(v => v.id === selectedSavedVoice.id) || null;
  }
  return voicesCache;
}

// ─── Worker encode requests ─────────────────────────────────────────────────

let nextEncodeId = 1;
const pendingEncodes = new Map(); // requestId → { resolve, reject }

function requestEncode(pcmF32) {
  return new Promise((resolve, reject) => {
    if (!ttsWorker) { reject(new Error('Engine not running')); return; }
    const requestId = nextEncodeId++;
    pendingEncodes.set(requestId, { resolve, reject });
    const refAudio = new Float32Array(pcmF32); // copy — the stored record keeps its buffer
    ttsWorker.postMessage({ type: 'encode-reference', requestId, refAudio }, [refAudio.buffer]);
  });
}

function rejectPendingEncodes(reason) {
  for (const [, p] of pendingEncodes) p.reject(new Error(reason));
  pendingEncodes.clear();
}

// Encode a voice's PCM into cached tokens exactly once — concurrent callers
// (background migration, voice selection, generate) share the same promise.
const voiceEncodePromises = new Map(); // voiceId → Promise<voice>

// Voices saved before reference prep v2 were cut right after the last loud
// sample (often keeping the click of the stop button). Every chunk continues
// from the end of the reference, so that edge came back as a small sound at
// the start of each chunk. Such voices are re-prepared and re-encoded once.
function needsRefPrep(voice) {
  return !!voice.refAudio && (voice.refPrep || 1) < REF_PREP_VERSION;
}

function encodeVoiceShared(voice) {
  let p = voiceEncodePromises.get(voice.id);
  if (!p) {
    p = (async () => {
      const prep = needsRefPrep(voice);
      const refAudio = prep ? prepareReference(voice.refAudio, 24000) : voice.refAudio;
      const r = await requestEncode(refAudio);
      const tokens = r.tokens instanceof Int32Array ? r.tokens : new Int32Array(r.tokens);
      const extra = prep ? { refAudio, duration: refAudio.length / 24000, refPrep: REF_PREP_VERSION } : null;
      await persistVoiceTokens(voice.id, tokens, r.tokenCount, extra);
      Object.assign(voice, extra, { tokens, tokenCount: r.tokenCount });
      return voice;
    })();
    p.finally(() => voiceEncodePromises.delete(voice.id)).catch(() => {});
    voiceEncodePromises.set(voice.id, p);
  }
  return p;
}

// Lazy migration wrapper with guards (never during generation).
async function ensureVoiceTokens(voice) {
  if (!voice || (voice.tokens && !needsRefPrep(voice)) || !isReady || !encoderAvailable || isGenerating) return;
  try {
    await encodeVoiceShared(voice);
    if (currentView === 'voices') renderVoicesView();
  } catch (e) {
    console.warn('Background voice encode failed:', e.message);
  }
}

async function migrateVoicesInBackground() {
  for (const v of voicesCache) {
    if (isGenerating) return;
    if (!v.tokens || needsRefPrep(v)) await ensureVoiceTokens(v);
  }
}

// ─── Toggle rows (accessible radiogroups) ───────────────────────────────────

function initToggleRow(row, { label, onChange } = {}) {
  if (!row) return;
  row.setAttribute('role', 'radiogroup');
  if (label) row.setAttribute('aria-label', label);
  const btns = Array.from(row.querySelectorAll('.toggle-btn'));
  const apply = (btn) => {
    btns.forEach(b => {
      const on = b === btn;
      b.classList.toggle('active', on);
      b.setAttribute('aria-checked', on ? 'true' : 'false');
      b.tabIndex = on ? 0 : -1;
    });
  };
  btns.forEach((btn, idx) => {
    btn.setAttribute('role', 'radio');
    const on = btn.classList.contains('active');
    btn.setAttribute('aria-checked', on ? 'true' : 'false');
    btn.tabIndex = on ? 0 : -1;
    btn.addEventListener('click', () => {
      if (btn.classList.contains('active')) return;
      apply(btn);
      if (onChange) onChange(btn.dataset.val, btn);
    });
    btn.addEventListener('keydown', (e) => {
      let target = null;
      if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') target = btns[(idx - 1 + btns.length) % btns.length];
      else if (e.key === 'ArrowRight' || e.key === 'ArrowDown') target = btns[(idx + 1) % btns.length];
      else if (e.key === 'Home') target = btns[0];
      else if (e.key === 'End') target = btns[btns.length - 1];
      else if (e.key === ' ' || e.key === 'Enter') target = btn;
      else return;
      e.preventDefault();
      target.focus();
      if (!target.classList.contains('active')) {
        apply(target);
        if (onChange) onChange(target.dataset.val, target);
      }
    });
  });
}

function onDesignChange() {
  if (isGenerating) return; // controls are locked during generation (incl. keyboard)
  if (selectedSavedVoice) selectVoice(null);
  if (voiceLocked || sessionChainRef) {
    if (voiceLocked) toast('Voice lock cleared — settings changed');
    voiceLocked = false;
    sessionChainRef = null;
    updateVoiceLockBtn();
  }
}

initToggleRow(genderRow, { label: 'Gender', onChange: onDesignChange });
initToggleRow(pitchRow, { label: 'Pitch', onChange: onDesignChange });
initToggleRow(qualityRow, { label: 'Quality', onChange: (val) => { if (!isGenerating) qualityEl.value = val; } });

function getToggleVal(row) {
  const active = row.querySelector('.toggle-btn.active');
  return active ? active.dataset.val : '';
}

function buildInstruct() {
  const parts = [getToggleVal(genderRow), getToggleVal(pitchRow)].filter(Boolean);
  return parts.length ? parts.join(', ') : null;
}

// ─── Char count ─────────────────────────────────────────────────────────────

textEl.addEventListener('input', () => {
  const len = textEl.value.length;
  charCount.textContent = len;
  charCount.parentElement.classList.toggle('char-count-warn', len >= MAX_TEXT_LEN * 0.9);
});

// ─── Voice mode UI ──────────────────────────────────────────────────────────

function isCloneMode() {
  return !!selectedSavedVoice;
}

function updateVoiceUI() {
  const cloning = isCloneMode();
  if (!isGenerating) {
    genderRow.style.opacity = cloning ? '0.3' : '1';
    genderRow.style.pointerEvents = cloning ? 'none' : 'auto';
    pitchRow.style.opacity = cloning ? '0.3' : '1';
    pitchRow.style.pointerEvents = cloning ? 'none' : 'auto';
    const gLabel = genderRow.parentElement.querySelector('label');
    const pLabel = pitchRow.parentElement.querySelector('label');
    if (gLabel) gLabel.style.opacity = cloning ? '0.3' : '1';
    if (pLabel) pLabel.style.opacity = cloning ? '0.3' : '1';
    generateBtn.textContent = selectedSavedVoice ? `Generate as "${selectedSavedVoice.name}"` : 'Generate Speech';
  }
  updateVoiceLockBtn();
}

function updateVoiceLockBtn() {
  if (!voiceLockBtn) return;
  const show = !selectedSavedVoice && !!sessionChainRef;
  voiceLockBtn.classList.toggle('hidden', !show);
  voiceLockBtn.classList.toggle('locked', voiceLocked);
  voiceLockBtn.setAttribute('aria-pressed', voiceLocked ? 'true' : 'false');
  voiceLockBtn.textContent = voiceLocked ? '\u{1F512} Voice locked' : '\u{1F512} Keep this voice';
}

if (voiceLockBtn) {
  voiceLockBtn.addEventListener('click', () => {
    if (!sessionChainRef) return;
    voiceLocked = !voiceLocked;
    updateVoiceLockBtn();
    toast(voiceLocked
      ? 'Voice locked — the next generations reuse this exact voice'
      : 'Voice lock off — each generation gets a fresh voice');
  });
}

function selectVoice(v) {
  selectedSavedVoice = v;
  renderStudioVoicePicker();
  if (currentView === 'voices') renderVoicesView();
  updateVoiceUI();
  if (v) ensureVoiceTokens(v);
}

// ─── Progress bar ───────────────────────────────────────────────────────────

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
  progressBar.classList.remove('hidden');
  progressBar.classList.remove('indeterminate');
  progressBar.style.width = Math.min(100, pct) + '%';
}

function hideProgress() {
  progressBar.classList.add('hidden');
  progressBar.classList.remove('indeterminate');
  progressBar.style.width = '0%';
}

// ─── Loading overlay / first-run experience ─────────────────────────────────

let rateSamples = [];
let lastLoadingTextUpdate = 0;

function onDownloadPlan(msg) {
  rateSamples = [];
  if (!loadingOverlay || !loadingOverlay.isConnected) return;
  if (msg.filesToDownload > 0) {
    if (msg.firstRun) {
      loadingTitle.textContent = 'First-time setup';
      loadingSub.textContent = 'VocoLoco runs a 600M-parameter AI voice model directly in your browser — nothing you type or generate ever leaves your device. The model (~3 GB) downloads once and is stored locally for instant starts next time. Keep this tab open; this can take a few minutes.';
    } else {
      loadingTitle.textContent = 'Resuming download';
      loadingSub.textContent = msg.cachedBytes > 0
        ? `${formatBytes(msg.cachedBytes)} already saved — fetching the rest.`
        : 'Fetching the remaining model files.';
    }
    if (msg.totalBytes) loadingTrack.classList.remove('hidden');
  } else {
    loadingTitle.textContent = 'Loading VocoLoco…';
    loadingSub.textContent = '';
  }
}

function onDownloadProgress(msg) {
  if (msg.loadedBytes != null && msg.totalBytes) {
    const pct = (msg.loadedBytes / msg.totalBytes) * 100;
    setProgressPercent(pct);
    if (loadingFill && loadingFill.isConnected) loadingFill.style.width = pct.toFixed(1) + '%';
    const now = performance.now();
    rateSamples.push({ t: now, bytes: msg.loadedBytes });
    while (rateSamples.length > 2 && now - rateSamples[0].t > 12000) rateSamples.shift();
    if (now - lastLoadingTextUpdate > 2000) {
      lastLoadingTextUpdate = now;
      let suffix = '';
      const first = rateSamples[0];
      if (rateSamples.length > 1 && now - first.t > 1500) {
        const rate = (msg.loadedBytes - first.bytes) / ((now - first.t) / 1000);
        if (rate > 1) {
          suffix = ` · ${formatBytes(rate)}/s`;
          const etaS = (msg.totalBytes - msg.loadedBytes) / rate;
          suffix += etaS < 60 ? ' · under a minute left' : ` · about ${Math.round(etaS / 60)} min left`;
        }
      }
      const line = `${formatBytes(msg.loadedBytes)} of ${formatBytes(msg.totalBytes)}${suffix}`;
      if (loadingStats && loadingStats.isConnected) loadingStats.textContent = line;
      setStatus(`Downloading models — ${line}`);
    }
  } else {
    showProgress('indeterminate');
    const now = performance.now();
    if (now - lastLoadingTextUpdate > 2000) {
      lastLoadingTextUpdate = now;
      const line = msg.fileIndex
        ? `File ${msg.fileIndex}/${msg.fileCount} — ${msg.detail || 'downloading…'}`
        : (msg.detail || 'Downloading models…');
      if (loadingStats && loadingStats.isConnected) loadingStats.textContent = line;
      setStatus(line);
    }
  }
}

function onLoadingPhase(msg) {
  showProgress('indeterminate');
  const warmingUp = ['session-main', 'session-decoder', 'session-encoder', 'warmup'].includes(msg.phase);
  if (warmingUp) {
    setStatus('Warming up the model…');
    if (loadingOverlay && loadingOverlay.isConnected) {
      loadingTitle.textContent = 'Warming up the model…';
      loadingSub.textContent = 'First start after a download takes a little longer while GPU shaders compile (~10–30 s).';
      loadingTrack.classList.add('hidden');
      loadingStats.textContent = '';
    }
  } else {
    setStatus(msg.detail || 'Loading…');
  }
}

// ─── Ready handler ──────────────────────────────────────────────────────────

function enableUI() {
  isReady = true;
  setStatus('Ready');
  hideProgress();
  if (loadingOverlay && loadingOverlay.isConnected) {
    loadingOverlay.style.transition = 'opacity 0.3s';
    loadingOverlay.style.opacity = '0';
    setTimeout(() => loadingOverlay.remove(), 300);
  }
  generateBtn.disabled = false;
  textEl.disabled = false;
  if (currentView === 'studio') textEl.focus({ preventScroll: true });
  if (currentView === 'voices') renderVoicesView();
  // Models finished loading while the user sat on the wizard's done step
  if (wiz.step === 'done' && wizTestBtn) {
    wizTestBtn.disabled = isGenerating;
    wizTestBtn.title = '';
  }
}

function onReady(backend) {
  backendIsCpu = backend !== 'webgpu';
  const badge = $('backend-badge');
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
  migrateVoicesInBackground();
}

// ─── Worker ─────────────────────────────────────────────────────────────────

let pendingJob = null; // { jobId, resolve }

function resolveJob(result) {
  const p = pendingJob;
  pendingJob = null;
  if (p) p.resolve(result);
}

function sendJob(msg) {
  return new Promise((resolve) => {
    pendingJob = { jobId: msg.jobId, resolve };
    const transfers = [];
    if (msg.refAudio) transfers.push(msg.refAudio.buffer);
    if (msg.refTokens) transfers.push(msg.refTokens.buffer);
    ttsWorker.postMessage(msg, transfers);
  });
}

function initWorker() {
  rejectPendingEncodes('Engine restarted');
  resolveJob({ kind: 'cancelled' });
  ttsWorker = new Worker('workers/tts-worker.js?v=5', { type: 'module' });
  ttsWorker.onerror = (e) => {
    console.error('Worker error:', e);
    setStatus('Engine failed to load — check the console and reload the page.');
    hideProgress();
    if (loadingOverlay && loadingOverlay.isConnected) {
      loadingTitle.textContent = 'Something went wrong';
      loadingSub.textContent = e.message || 'The speech engine could not start. Reload the page to try again.';
      const spinner = $('loading-spinner');
      if (spinner) spinner.style.display = 'none';
    }
  };
  ttsWorker.onmessage = (e) => {
    const msg = e.data;
    switch (msg.type) {
      case 'plan':
        onDownloadPlan(msg);
        break;
      case 'progress':
        if (msg.stage === 'downloading') onDownloadProgress(msg);
        else if (msg.stage === 'loading') onLoadingPhase(msg);
        else if (msg.stage === 'generating') onGenProgress(msg);
        else if (msg.stage === 'warning') toast(msg.detail, { type: 'error', duration: 6000 });
        else setStatus(msg.detail || '');
        break;
      case 'ready':
        encoderAvailable = msg.encoderAvailable !== false;
        onReady(msg.backend);
        break;
      case 'audio':
        resolveJob({ kind: 'audio', pcm: msg.pcm, sampleRate: msg.sampleRate, peak: msg.peak, tokens: msg.tokens || null, tokenCount: msg.tokenCount || 0 });
        break;
      case 'cancelled':
        resolveJob({ kind: 'cancelled' });
        break;
      case 'encoded': {
        const p = pendingEncodes.get(msg.requestId);
        if (p) { pendingEncodes.delete(msg.requestId); p.resolve(msg); }
        break;
      }
      case 'encode-error': {
        const p = pendingEncodes.get(msg.requestId);
        if (p) { pendingEncodes.delete(msg.requestId); p.reject(Object.assign(new Error(msg.message), { code: msg.code })); }
        break;
      }
      case 'error':
        if (pendingJob) {
          resolveJob({ kind: 'error', message: msg.message });
        } else {
          console.error(msg.message);
          setStatus(msg.message);
          hideProgress();
          if (loadingOverlay && loadingOverlay.isConnected) {
            loadingTitle.textContent = 'Something went wrong';
            loadingSub.textContent = msg.message;
            const spinner = $('loading-spinner');
            if (spinner) spinner.style.display = 'none';
          }
        }
        break;
    }
  };
  const worker = ttsWorker;
  resolveModelBaseUrl().then((modelBaseUrl) => {
    worker.postMessage({ type: 'init', modelBaseUrl, forceCPU: location.search.includes('cpu') });
  });
  showProgress('indeterminate');
}

let modelBaseUrlPromise = null;

function resolveModelBaseUrl() {
  if (!modelBaseUrlPromise) {
    modelBaseUrlPromise = (async () => {
      if (!['localhost', '127.0.0.1', '[::1]'].includes(location.hostname)) return MODEL_BASE_URL;
      try {
        // The fetch script writes the manifest last, so its presence means the mirror is complete
        const r = await fetch(`${LOCAL_MODEL_URL}/omnivoice-main-manifest.json`, { method: 'HEAD' });
        if (r.ok) {
          console.log(`[init] Using local model mirror: ${LOCAL_MODEL_URL}`);
          return LOCAL_MODEL_URL;
        }
      } catch { /* no mirror */ }
      return MODEL_BASE_URL;
    })();
  }
  return modelBaseUrlPromise;
}

// ─── Player ─────────────────────────────────────────────────────────────────

const player = new StreamingPlayer({
  canvas: waveformEl,
  playheadEl: $('playhead'),
  timeEl: playerDuration,
  bufferingEl: $('buffering-chip'),
  sampleRate: 24000,
  gapSeconds: 0.15,
  onStateChange: (state) => updateReplayBtn(state === 'playing'),
  onAutoplayBlocked: () => {
    toast('Click anywhere to start audio playback');
    const resume = () => { player.getAudioCtx(); document.removeEventListener('pointerdown', resume); };
    document.addEventListener('pointerdown', resume);
  },
});

function updateReplayBtn(playing) {
  if (!replayBtn) return;
  if (playing) {
    replayBtn.innerHTML = '&#9632; Stop';
    replayBtn.classList.add('bg-red-500/20', 'border-red-400/40', 'text-red-400');
    replayBtn.classList.remove('bg-omni-active-bg', 'text-omni-neon', 'border-omni-neon/40', 'shadow-neon');
  } else {
    replayBtn.innerHTML = '&#9654; Replay';
    replayBtn.classList.remove('bg-red-500/20', 'border-red-400/40', 'text-red-400');
    replayBtn.classList.add('bg-omni-active-bg', 'text-omni-neon', 'border-omni-neon/40', 'shadow-neon');
  }
}

function enablePlayerControls() {
  if (playerControls) {
    playerControls.style.opacity = '1';
    playerControls.style.pointerEvents = 'auto';
  }
}

replayBtn.addEventListener('click', () => {
  if (player.isPlaying()) {
    player.stop();
    if (!isGenerating) setStatus('Ready');
    return;
  }
  if (player.hasAudio()) {
    player.play(0);
  } else if (lastPcm) {
    player.loadStatic(lastPcm);
    player.play(0);
  }
});

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
  const comment = 'AI-generated synthetic speech. This audio was produced entirely by an artificial intelligence text-to-speech model. EU AI Act Art. 50 — not a recording of a human voice.';

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

function mp3Filename(text, timestamp) {
  const words = (text || 'vocoloco').slice(0, 50).replace(/[^a-zA-Z0-9 ]/g, '').trim().replace(/\s+/g, '_') || 'vocoloco';
  const ts = new Date(timestamp || Date.now()).toISOString().slice(0, 16).replace(/[T:]/g, '-');
  return `${words}_${ts}.mp3`;
}

downloadBtn.addEventListener('click', () => {
  if (lastPcm) downloadMp3(lastPcm, lastSampleRate, mp3Filename(lastText));
});

// Save part of the last generation as a reusable voice — opens the wizard's
// trim/review step on the whole generation with chunk 1 preselected (the
// combined audio starts with chunk 1). Design-mode generations carry chunk
// 1's exact (pcm, text, tokens) triple, so keeping that selection needs no
// encoder run at all.
saveGenVoiceBtn.addEventListener('click', () => {
  if (!lastPcm) return;
  const preset = {
    source: 'generated',
    pcm: new Float32Array(lastPcm),
    transcript: lastText.replace(/ …$/, ''),
  };
  if (lastGen && lastGen.first) {
    preset.selection = { start: 0, end: lastGen.first.pcm.length };
    preset.transcript = lastGen.first.text;
    if (!lastGen.voiceName && lastGen.chainRef) {
      preset.tokens = lastGen.chainRef.tokens;
      preset.tokenCount = lastGen.chainRef.tokenCount;
    }
  }
  switchView('voices');
  openVoiceWizard('review', preset);
});

// ─── Generation history / library ───────────────────────────────────────────

async function addToHistory(text, pcm, sampleRate, duration, meta = {}) {
  const item = {
    text,
    pcm: new Float32Array(pcm),
    sampleRate,
    duration,
    timestamp: Date.now(),
    voiceName: meta.voiceName ?? null,
    quality: meta.quality ?? null,
    cancelled: !!meta.cancelled,
  };
  history.unshift(item);
  if (history.length > MAX_HISTORY) history.pop();
  updateLibraryBadge();
  if (currentView === 'library') renderLibrary();
  try {
    await saveHistoryItem(item);
    await pruneHistoryDB();
  } catch (e) { console.warn('Failed to persist history:', e); }
}

function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

function timeAgoLabel(timestamp) {
  const diff = Date.now() - timestamp;
  if (diff < 60000) return 'just now';
  if (diff < 3600000) return `${Math.floor(diff / 60000)}m ago`;
  if (diff < 86400000) return `${Math.floor(diff / 3600000)}h ago`;
  return new Date(timestamp).toLocaleDateString();
}

function renderLibrary() {
  const listEl = $('library-list');
  const totalEl = $('library-total');

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
    const metaParts = [`${item.duration.toFixed(1)}s`];
    metaParts.push(item.voiceName ? escapeHtml(item.voiceName) : 'Designed voice');
    if (item.quality && QUALITY_LABELS[item.quality]) metaParts.push(QUALITY_LABELS[item.quality]);
    metaParts.push(timeAgoLabel(item.timestamp));
    const partialChip = item.cancelled
      ? ' <span style="font-size:9px;font-weight:700;text-transform:uppercase;color:#f59e0b;background:rgba(245,158,11,0.1);border:1px solid rgba(245,158,11,0.3);padding:1px 6px;border-radius:9999px;">partial</span>'
      : '';

    const info = document.createElement('div');
    info.style.cssText = 'display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap;';
    info.innerHTML = `
      <div style="flex:1;min-width:0;">
        <div style="font-size:13px;color:#e5e7eb;line-height:1.4;margin-bottom:4px;overflow:hidden;white-space:nowrap;text-overflow:ellipsis;">${escapeHtml(snippet)}</div>
        <div style="font-size:11px;color:#64748b;">${metaParts.join(' &middot; ')}${partialChip}</div>
      </div>
      <div style="display:flex;gap:6px;flex-shrink:0;flex-wrap:wrap;">
        <button data-action="replay" style="padding:6px 14px;border-radius:8px;background:#273c38;color:#4ade80;border:1px solid rgba(74,222,128,0.3);font-size:12px;font-weight:600;cursor:pointer;transition:all 0.15s;">&#9654; Play</button>
        <button data-action="download" style="padding:6px 14px;border-radius:8px;background:#1e293b;color:#9ca3af;border:1px solid #2d3748;font-size:12px;font-weight:600;cursor:pointer;transition:all 0.15s;">&#8595; MP3</button>
        <button data-action="use-text" style="padding:6px 14px;border-radius:8px;background:#1e293b;color:#9ca3af;border:1px solid #2d3748;font-size:12px;font-weight:600;cursor:pointer;transition:all 0.15s;" aria-label="Load this text into the Studio">Use text</button>
        <button data-action="delete" style="padding:6px 14px;border-radius:8px;background:rgba(239,68,68,0.08);color:#f87171;border:1px solid rgba(239,68,68,0.25);font-size:12px;font-weight:600;cursor:pointer;transition:all 0.15s;" aria-label="Delete this generation">Delete</button>
      </div>
    `;

    const canvas = document.createElement('canvas');
    canvas.width = 600; canvas.height = 60;
    canvas.style.cssText = 'width:100%;height:36px;border-radius:8px;background:#0a0e14;margin-bottom:10px;';

    el.appendChild(canvas);
    el.appendChild(info);

    info.querySelector('[data-action="replay"]').addEventListener('click', () => {
      player.playOneShot(item.pcm, item.sampleRate);
      setStatus(`Replaying ${item.duration.toFixed(1)}s`);
    });
    info.querySelector('[data-action="download"]').addEventListener('click', () => {
      downloadMp3(item.pcm, item.sampleRate, mp3Filename(item.text, item.timestamp));
    });
    info.querySelector('[data-action="use-text"]').addEventListener('click', () => {
      textEl.value = item.text.replace(/ …$/, '');
      textEl.dispatchEvent(new Event('input'));
      switchView('studio');
      textEl.focus({ preventScroll: true });
      toast('Text loaded into Studio');
    });
    info.querySelector('[data-action="delete"]').addEventListener('click', async () => {
      const ok = await confirmDialog({
        title: 'Delete this generation?',
        body: 'This cannot be undone.',
        confirmLabel: 'Delete',
        danger: true,
      });
      if (!ok) return;
      try { if (item.id != null) await deleteHistoryItem(item.id); } catch (e) { console.warn('Delete failed:', e); }
      history = history.filter(h => h !== item);
      renderLibrary();
      updateLibraryBadge();
      toast('Deleted', { type: 'success' });
    });

    listEl.appendChild(el);
    drawMiniWaveform(canvas, item.pcm);
  });
}

function updateLibraryBadge() {
  const b = $('library-count');
  if (b) { b.textContent = history.length; b.classList.toggle('has-items', history.length > 0); }
}

// ─── Decode ref audio to 24kHz mono ─────────────────────────────────────────

async function decodeRefAudio(file) {
  const ctx = player.getAudioCtx();
  const buf = await ctx.decodeAudioData(await file.arrayBuffer());
  const numSamples = Math.round(buf.duration * 24000);
  const offline = new OfflineAudioContext(1, numSamples, 24000);
  const src = offline.createBufferSource();
  src.buffer = buf;
  src.connect(offline.destination);
  src.start();
  // Untrimmed: the user picks the part in the trim editor, prepareReference()
  // cleans the edges when the voice is saved
  return (await offline.startRendering()).getChannelData(0);
}

// ─── Streaming generation ───────────────────────────────────────────────────

const generateBtnDefaultClasses = generateBtn.className;

function setGenerating(active) {
  if (active) {
    generateBtn.className = generateBtnDefaultClasses;
    generateBtn.classList.add('btn-cancel');
    generateBtn.textContent = 'Cancel';
    generateBtn.disabled = false;
    generateBtn.onclick = cancelGeneration;
  } else {
    generateBtn.className = generateBtnDefaultClasses;
    generateBtn.onclick = null;
    generateBtn.disabled = !isReady;
    updateVoiceUI();
  }
}

function lockVoiceControls(locked) {
  for (const el of [genderRow, pitchRow, qualityRow, studioVoicePicker]) {
    if (!el) continue;
    el.style.pointerEvents = locked ? 'none' : 'auto';
    el.style.opacity = locked ? '0.5' : '1';
    el.setAttribute('aria-disabled', locked ? 'true' : 'false');
  }
  if (!locked) updateVoiceUI(); // restores clone-mode dimming
}

async function generate() {
  const text = textEl.value.trim();
  if (!text || !isReady || isGenerating) return;
  if (text.length > MAX_TEXT_LEN) {
    toast(`Text too long (max ${MAX_TEXT_LEN} characters)`, { type: 'error' });
    return;
  }

  if (history.length >= MAX_HISTORY) {
    const proceed = await confirmDialog({
      title: 'Library is full',
      body: `You have ${MAX_HISTORY} saved generations. Starting a new one removes the oldest automatically. Download anything you want to keep first.`,
      confirmLabel: 'Generate anyway',
    });
    if (!proceed || isGenerating) return;
  }

  const chunks = chunkText(text);
  if (chunks.length === 0) {
    toast('Nothing to speak — the text contains no readable characters.', { type: 'error' });
    return;
  }

  player.getAudioCtx(); // unlock audio within the user gesture

  isGenerating = true;
  setGenerating(true);
  lockVoiceControls(true);
  player.dimPrevious();
  setStatus('Preparing…');
  showProgress('indeterminate');

  const voice = selectedSavedVoice;
  stream = {
    id: 'g' + (++streamCounter),
    chunks,
    results: [],
    cancelRequested: false,
    totalTokens: chunks.reduce((a, c) => a + c.estTokens, 0),
    doneTokens: 0,
    gain: null,
    chainRef: null,
    quality: parseInt(qualityEl.value),
    voiceName: voice ? voice.name : null,
    hardKillTimer: null,
    lastStepMs: null,
    emaMsPerToken: null,
  };

  // Resolve the voice reference ONCE for the whole stream
  const ref = { refTokens: null, refText: null, instruct: null };
  try {
    if (voice) {
      ref.refText = voice.refText;
      if (voice.tokens && !needsRefPrep(voice)) {
        ref.refTokens = voice.tokens;
      } else if (encoderAvailable) {
        setStatus('Encoding reference voice… (one-time per voice)');
        await encodeVoiceShared(voice); // shares any encode already in flight
        ref.refTokens = voice.tokens;
        if (currentView === 'voices') renderVoicesView();
      } else if (voice.tokens) {
        ref.refTokens = voice.tokens; // older reference prep, but no encoder on this device
      } else {
        toast(`"${voice.name}" isn't set up on this device (voice encoder unavailable) — generating with a default voice instead.`, { type: 'error', duration: 6000 });
        ref.refText = null;
      }
    } else if (voiceLocked && sessionChainRef) {
      ref.refTokens = sessionChainRef.tokens;
      ref.refText = sessionChainRef.text;
    } else {
      ref.instruct = buildInstruct();
    }
  } catch (err) {
    toast('Could not prepare the voice: ' + err.message, { type: 'error' });
    finishStream();
    return;
  }
  if (!stream || stream.cancelRequested) { finishStream(); return; }

  if (backendIsCpu && !cpuHintShown) {
    cpuHintShown = true;
    toast('CPU mode: generation is much slower — the first audio can take several minutes.', { duration: 8000 });
  }

  await runStream(ref);
}

let backendIsCpu = false;
let cpuHintShown = false;

async function runStream(ref) {
  const s = stream;
  for (let i = 0; i < s.chunks.length; i++) {
    if (s.cancelRequested) break;
    const res = await sendJob(buildChunkMsg(i, ref));
    if (res.kind === 'cancelled') break;
    if (res.kind === 'error') {
      toast('Generation failed: ' + String(res.message).split('\n')[0], { type: 'error', duration: 6000 });
      break;
    }
    onChunkAudio(i, res);
  }
  finishStream();
}

function buildChunkMsg(i, ref) {
  const s = stream;
  const msg = {
    type: 'synthesize',
    jobId: `${s.id}-${i}`,
    text: s.chunks[i].text,
    lang: null,
    numStep: s.quality,
    ...GEN_PARAMS,
    speed: 1.0,
    seed: null,
    instruct: null,
    normalize: false,
  };
  if (ref.refTokens) {
    // Cloned / locked voice: same reference for every chunk
    msg.refTokens = new Int32Array(ref.refTokens); // copy — cached buffer must survive transfer
    msg.refText = ref.refText;
  } else if (i === 0) {
    // Design mode, first chunk: let the model pick a voice, capture its tokens
    msg.instruct = ref.instruct;
    msg.returnTokens = true;
  } else if (s.chainRef) {
    // Design mode, later chunks: chain chunk 1's voice
    msg.refTokens = new Int32Array(s.chainRef.tokens);
    msg.refText = s.chainRef.text;
  } else {
    msg.instruct = ref.instruct;
  }
  return msg;
}

// Silence between chunks. Pauses after a full stop inside generated chunks
// measure ~0.4-0.75 s (each chunk also keeps up to 100 ms of its own
// lead/tail); a chunk split mid-sentence only gets a comma-length pause.
function joinPauseSeconds(prevText) {
  return /[.!?…。！？]["'”’)\]]*$/.test(prevText.trim()) ? 0.4 : 0.15;
}

function onChunkAudio(i, res) {
  const s = stream;
  const pcm = res.pcm; // transferred — ours to mutate

  // Uniform loudness: gain derived from chunk 1's peak, clip-guarded per chunk
  if (s.gain == null) s.gain = res.peak > 1e-6 ? 0.5 / res.peak : 1;
  let g = s.gain;
  if (res.peak > 1e-6 && res.peak * g > 0.98) g = 0.98 / res.peak;
  if (g !== 1) for (let k = 0; k < pcm.length; k++) pcm[k] *= g;

  if (i === 0) {
    if (currentView !== 'studio') switchView('studio');
    player.beginStream({ estTotalSamples: Math.round((s.totalTokens / 25) * 24000) });
    if (res.tokens) {
      s.chainRef = { tokens: res.tokens, tokenCount: res.tokenCount, text: s.chunks[0].text };
    }
    enablePlayerControls();
  }
  player.appendChunk(pcm, i > 0 ? joinPauseSeconds(s.chunks[i - 1].text) : 0);
  s.results.push({ text: s.chunks[i].text, pcm });
  s.doneTokens += s.chunks[i].estTokens;
}

function finishStream() {
  const s = stream;
  if (s && s.hardKillTimer) clearTimeout(s.hardKillTimer);

  if (s && s.results.length > 0) {
    const partial = s.cancelRequested && s.results.length < s.chunks.length;
    const combined = player.endStream();
    lastPcm = combined;
    lastSampleRate = 24000;
    lastText = s.results.map(r => r.text).join(' ') + (partial ? ' …' : '');
    const duration = combined.length / 24000;
    addToHistory(lastText, combined, 24000, duration, {
      voiceName: s.voiceName,
      quality: s.quality,
      cancelled: partial,
    });
    enablePlayerControls();
    lastGen = {
      voiceName: s.voiceName,
      first: { pcm: s.results[0].pcm, text: s.results[0].text },
      chainRef: s.chainRef,
    };
    if (!s.voiceName && s.chainRef) {
      sessionChainRef = { ...s.chainRef, pcm: s.results[0].pcm };
    }
    setStatus(partial
      ? `Stopped — kept ${s.results.length} of ${s.chunks.length} parts`
      : `Playing ${duration.toFixed(1)}s`);
  } else if (s) {
    player.undim();
    setStatus(s.cancelRequested ? 'Cancelled' : 'Ready');
  }

  isGenerating = false;
  stream = null;
  setGenerating(false);
  lockVoiceControls(false);
  updateVoiceLockBtn();
  hideProgress();
  migrateVoicesInBackground();
}

function onGenProgress(msg) {
  if (!stream) {
    setStatus(`Generating — step ${msg.step}/${msg.numStep}`);
    setProgressPercent((msg.step / msg.numStep) * 100);
    return;
  }
  const s = stream;
  s.lastStepMs = msg.stepMs;
  const idx = Math.min(s.results.length, s.chunks.length - 1);
  const chunkTokens = s.chunks[idx].estTokens;
  const frac = msg.step / msg.numStep;
  setProgressPercent(((s.doneTokens + chunkTokens * frac) / s.totalTokens) * 100);

  // ETA from an EMA of ms-per-token
  const msPerToken = (msg.stepMs * msg.numStep) / Math.max(1, chunkTokens);
  s.emaMsPerToken = s.emaMsPerToken == null ? msPerToken : s.emaMsPerToken * 0.7 + msPerToken * 0.3;
  const remaining = Math.max(0, s.totalTokens - s.doneTokens - chunkTokens * frac);
  const etaMs = remaining * s.emaMsPerToken;
  const etaStr = etaMs > 10000
    ? (etaMs < 90000 ? ` · ~${Math.round(etaMs / 1000)}s left` : ` · ~${Math.round(etaMs / 60000)} min left`)
    : '';

  if (s.chunks.length > 1) {
    const prefix = s.results.length > 0 ? 'Playing · generating' : 'Generating';
    setStatus(`${prefix} part ${idx + 1} of ${s.chunks.length} — step ${msg.step}/${msg.numStep}${etaStr}`);
  } else {
    setStatus(`Generating — step ${msg.step}/${msg.numStep}${etaStr}`);
  }
}

function cancelGeneration() {
  if (!isGenerating) return;
  if (stream) {
    if (stream.cancelRequested) return;
    stream.cancelRequested = true; // the chunk loop stops sending new jobs
    // The worker can only honor a cancel at a step boundary, so the fallback
    // must outlast at least one full step — CPU steps can take minutes.
    const measured = stream.lastStepMs;
    const timeout = measured
      ? Math.max(4000, Math.round(measured * 2) + 2000)
      : (backendIsCpu ? 60000 : 5000);
    stream.hardKillTimer = setTimeout(hardKillWorker, timeout);
  }
  ttsWorker.postMessage({ type: 'cancel' });
  generateBtn.textContent = 'Stopping…';
  generateBtn.disabled = true;
  setStatus('Stopping after the current step…');
}

// Fallback when the worker doesn't confirm the cancel in time (e.g. a
// minutes-long CPU step): kill it and reinitialize. Completed chunks are
// already on this side and are kept either way.
function hardKillWorker() {
  rejectPendingEncodes('Engine restarted');
  try { ttsWorker.terminate(); } catch { /* already dead */ }
  isReady = false;
  resolveJob({ kind: 'cancelled' });
  setStatus('Cancelled — restarting engine…');
  initWorker();
}

generateBtn.addEventListener('click', generate);

// Enter generates (Shift+Enter = newline, Ctrl/Cmd+Enter also generates)
textEl.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && (e.ctrlKey || e.metaKey || !e.shiftKey)) {
    e.preventDefault();
    generate();
  }
});

// ─── Voice test generation (wizard + voice cards) ───────────────────────────

async function testVoice(voice, btn) {
  if (!isReady || isGenerating || !voice) return;
  const text = TEST_SENTENCES[voice.lang] || TEST_SENTENCES.en;
  isGenerating = true;
  setGenerating(true);
  const orig = btn ? btn.textContent : null;
  if (btn) { btn.disabled = true; btn.textContent = 'Generating…'; }
  setStatus('Testing voice…');
  showProgress('indeterminate');

  const msg = {
    type: 'synthesize',
    jobId: 'test-' + Date.now(),
    text,
    lang: null,
    numStep: 8, // fast — this is a preview
    ...GEN_PARAMS,
    speed: 1.0,
    instruct: null,
    seed: null,
    refText: voice.refText,
    normalize: true,
  };
  if (voice.tokens) msg.refTokens = new Int32Array(voice.tokens);
  else msg.refAudio = new Float32Array(voice.refAudio);

  const res = await sendJob(msg);

  isGenerating = false;
  setGenerating(false);
  hideProgress();
  if (btn) { btn.disabled = false; btn.textContent = orig; }
  if (res.kind === 'audio') {
    player.playOneShot(res.pcm, res.sampleRate);
    setStatus(`Voice test — ${(res.pcm.length / res.sampleRate).toFixed(1)}s`);
  } else if (res.kind === 'error') {
    toast('Test failed: ' + String(res.message).split('\n')[0], { type: 'error' });
    setStatus('Ready');
  } else {
    setStatus('Cancelled');
  }
}

// Shared preview toggle for stored voice PCM (wizard review + voice cards)
let previewBtn = null;

function togglePcmPreview(pcm, btn, labelPlay = '▶ Play', labelStop = '■ Stop') {
  if (previewBtn === btn && player.isOneShotPlaying()) {
    player.stopOneShot(); // onended restores the label
    return;
  }
  previewBtn = btn;
  btn.textContent = labelStop;
  player.playOneShot(pcm, 24000, () => {
    btn.textContent = labelPlay;
    if (previewBtn === btn) previewBtn = null;
  });
}

// ─── Voices view ────────────────────────────────────────────────────────────

function updateVoicesBadge() {
  const b = $('voices-count');
  if (b) { b.textContent = voicesCache.length; b.classList.toggle('has-items', voicesCache.length > 0); }
}

async function renderVoicesView() {
  await refreshVoices();
  updateVoicesBadge();
  if (voicesEncoderNote) voicesEncoderNote.classList.toggle('hidden', encoderAvailable);
  const has = voicesCache.length > 0;
  if (voicesEmpty) voicesEmpty.style.display = has ? 'none' : 'flex';
  if (!voicesList) return;
  voicesList.style.display = has ? 'flex' : 'none';
  voicesList.innerHTML = '';
  for (const v of voicesCache) voicesList.appendChild(buildVoiceCard(v));
}

function buildVoiceCard(v) {
  const card = document.createElement('div');
  card.className = 'voice-card' + (selectedSavedVoice?.id === v.id ? ' selected' : '');

  const header = document.createElement('div');
  header.style.cssText = 'display:flex;align-items:center;gap:8px;';

  const nameSpan = document.createElement('span');
  nameSpan.className = 'vc-name';
  nameSpan.textContent = v.name;
  nameSpan.style.cssText = 'flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;';

  const renameBtn = document.createElement('button');
  renameBtn.className = 'vc-btn';
  renameBtn.textContent = '✎';
  renameBtn.setAttribute('aria-label', `Rename voice "${v.name}"`);
  renameBtn.title = 'Rename';

  header.append(nameSpan, renameBtn);

  if (!v.tokens) {
    const pending = document.createElement('span');
    pending.className = 'vc-pending';
    pending.textContent = 'setting up…';
    pending.title = 'The voice is analyzed once in the background to make generation fast.';
    header.appendChild(pending);
  }

  const meta = document.createElement('div');
  meta.className = 'vc-meta';
  const srcLabel = { recorded: 'Recorded', uploaded: 'Uploaded', generated: 'From generation' }[v.source] || 'Voice';
  meta.textContent = `${(v.duration || 0).toFixed(1)}s · ${new Date(v.createdAt).toLocaleDateString()} · ${srcLabel}`;

  const btnRow = document.createElement('div');
  btnRow.style.cssText = 'display:flex;gap:6px;flex-wrap:wrap;';

  const playBtn = document.createElement('button');
  playBtn.className = 'vc-btn';
  playBtn.textContent = '▶ Play';
  playBtn.setAttribute('aria-label', `Play the recording of "${v.name}"`);
  playBtn.addEventListener('click', (e) => { e.stopPropagation(); togglePcmPreview(v.refAudio, playBtn); });

  const testBtn = document.createElement('button');
  testBtn.className = 'vc-btn primary';
  testBtn.textContent = 'Test';
  testBtn.disabled = !isReady;
  testBtn.title = isReady ? 'Generate a short sample with this voice' : 'Models still loading…';
  testBtn.addEventListener('click', (e) => { e.stopPropagation(); testVoice(v, testBtn); });

  const delBtn = document.createElement('button');
  delBtn.className = 'vc-btn danger';
  delBtn.textContent = 'Delete';
  delBtn.setAttribute('aria-label', `Delete voice "${v.name}"`);
  delBtn.addEventListener('click', async (e) => {
    e.stopPropagation();
    const ok = await confirmDialog({
      title: 'Delete voice?',
      body: `"${v.name}" will be removed permanently.`,
      confirmLabel: 'Delete',
      danger: true,
    });
    if (!ok) return;
    try { await deleteVoice(v.id); } catch (err) { toast('Delete failed: ' + err.message, { type: 'error' }); return; }
    if (selectedSavedVoice?.id === v.id) selectedSavedVoice = null;
    await renderVoicesView();
    renderStudioVoicePicker();
    updateVoiceUI();
    toast(`Voice "${v.name}" deleted`, { type: 'success' });
  });

  btnRow.append(playBtn, testBtn, delBtn);
  card.append(header, meta, btnRow);

  // Inline rename
  renameBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    const input = document.createElement('input');
    input.className = 'voice-name-input';
    input.value = v.name;
    input.maxLength = 40;
    input.setAttribute('aria-label', 'New voice name');
    header.replaceChild(input, nameSpan);
    input.focus();
    input.select();
    let done = false;
    const commit = async () => {
      if (done) return;
      done = true;
      const newName = input.value.trim();
      if (newName && newName !== v.name) {
        v.name = newName;
        try { await saveVoice(v); } catch (err) { console.warn('Rename failed:', err); }
        renderStudioVoicePicker();
        updateVoiceUI();
      }
      renderVoicesView();
    };
    const revert = () => { if (done) return; done = true; renderVoicesView(); };
    input.addEventListener('keydown', (ev) => {
      ev.stopPropagation();
      if (ev.key === 'Enter') commit();
      else if (ev.key === 'Escape') revert();
    });
    input.addEventListener('blur', commit);
    input.addEventListener('click', (ev) => ev.stopPropagation());
  });

  card.addEventListener('click', () => {
    selectVoice(selectedSavedVoice?.id === v.id ? null : v);
  });

  return card;
}

// ─── Studio voice picker ────────────────────────────────────────────────────

function renderStudioVoicePicker() {
  if (!studioVoicePicker) return;
  studioVoicePicker.innerHTML = '';
  const mkPill = (label, active, onClick) => {
    const el = document.createElement('div');
    el.className = 'saved-voice' + (active ? ' active' : '');
    el.setAttribute('role', 'button');
    el.tabIndex = 0;
    el.textContent = label;
    el.addEventListener('click', onClick);
    el.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onClick(); }
    });
    return el;
  };
  studioVoicePicker.appendChild(mkPill('Default voices', !selectedSavedVoice, () => selectVoice(null)));
  for (const v of voicesCache) {
    studioVoicePicker.appendChild(
      mkPill(v.name, selectedSavedVoice?.id === v.id, () => selectVoice(selectedSavedVoice?.id === v.id ? null : v))
    );
  }
  const addPill = mkPill('+ New voice', false, () => {
    switchView('voices');
    openVoiceWizard('method');
  });
  addPill.style.borderStyle = 'dashed';
  addPill.style.color = '#9ca3af';
  studioVoicePicker.appendChild(addPill);
}

// Click + Enter/Space activation for role="button" non-buttons
function onActivate(el, fn) {
  if (!el) return;
  el.addEventListener('click', fn);
  el.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fn(); }
  });
}

onActivate(manageVoicesLink, () => switchView('voices'));

// ─── Voice wizard ───────────────────────────────────────────────────────────

const wiz = {
  step: null,           // 'method' | 'script' | 'record' | 'upload' | 'review' | 'saving' | 'done'
  method: null,
  source: null,         // 'recorded' | 'uploaded' | 'generated'
  scriptId: null,
  lang: null,
  scriptText: '',
  deviceId: null,
  stream: null, micCtx: null, srcNode: null, analyser: null, raf: 0,
  recorder: null, chunks: [], recStartTs: 0, recording: false, counting: false,
  levelRing: new Float32Array(100),
  sourcePcm: null,      // the full recording / upload / generation
  sel: null,            // { start, end } in samples — the part used as the voice
  pcm: null, duration: 0, // the selected part
  presetTokens: null, presetTokenCount: 0, presetSel: null, // tokens only valid for presetSel
  savedVoiceId: null,
};

const MAX_SOURCE_S = 600; // longer files: only the first 10 minutes are loaded
const MAX_REF_S = 15;     // longer references don't improve cloning, they only slow every step

const trimEditor = wizReviewWave ? new TrimEditor({
  canvas: wizReviewWave,
  overviewCanvas: wizTrimOverview,
  sampleRate: 24000,
  minS: 3,
  maxS: MAX_REF_S,
  onChange: (sel, { final }) => onTrimChange(sel, final),
}) : null;

function fmtClock(s) {
  return `${Math.floor(s / 60)}:${(s % 60).toFixed(1).padStart(4, '0')}`;
}

// New source audio for the review step; the selection defaults to the best
// <= 15 s of speech (or `presetSel`, e.g. chunk 1 of a generation).
function wizSetSource(pcm, presetSel = null) {
  const cap = MAX_SOURCE_S * 24000;
  if (pcm.length > cap) {
    pcm = pcm.slice(0, cap);
    toast('Only the first 10 minutes of this file were loaded.');
  }
  wiz.sourcePcm = pcm;
  wiz.presetSel = presetSel;
  wiz.sel = presetSel || defaultSelection(pcm, 24000, { minS: 3, maxS: MAX_REF_S });
  wiz.pcm = pcm.slice(wiz.sel.start, wiz.sel.end);
  wiz.duration = wiz.pcm.length / 24000;
}

function onTrimChange(sel, final) {
  if (!wiz.sourcePcm) return;
  wiz.sel = sel;
  wiz.pcm = wiz.sourcePcm.slice(sel.start, sel.end);
  wiz.duration = wiz.pcm.length / 24000;
  stopReviewPreview();
  updateReviewVerdict();
  if (final) updateTrimInfo();
}

function updateTrimInfo() {
  if (!wizTrimInfo || !wiz.sel || !wiz.sourcePcm) return;
  const total = wiz.sourcePcm.length / 24000;
  const range = `${fmtClock(wiz.sel.start / 24000)} – ${fmtClock(wiz.sel.end / 24000)} of ${fmtClock(total)}`;
  wizTrimInfo.textContent = `${range} · Drag the green handles or the window to choose the part to clone (3–${MAX_REF_S} s)`;
  const atPreset = wiz.presetSel && wiz.sel.start === wiz.presetSel.start && wiz.sel.end === wiz.presetSel.end;
  if (wizTranscriptHint && trimEditor && !atPreset && trimEditor.cutsSpeech()) {
    wizTranscriptHint.textContent = 'You trimmed the clip: the transcript must contain exactly the words spoken in the selected part.';
  } else if (wizTranscriptHint) {
    wizTranscriptHint.textContent = defaultTranscriptHint();
  }
}

function defaultTranscriptHint() {
  return wiz.source === 'recorded'
    ? 'Pre-filled from your script. Fix any words you changed while reading.'
    : wiz.source === 'uploaded'
      ? 'Type the exact words spoken in the selected part — this noticeably improves cloning accuracy.'
      : 'Pre-filled with the generated text.';
}

// Preview of the selection with a moving playhead
let reviewPlayheadRaf = 0;

function stopReviewPreview() {
  if (previewBtn === wizReviewPlay && player.isOneShotPlaying()) player.stopOneShot();
  cancelAnimationFrame(reviewPlayheadRaf);
  reviewPlayheadRaf = 0;
  if (trimEditor) trimEditor.setPlayhead(null);
}

function startReviewPlayhead() {
  const ctx = player.getAudioCtx();
  const t0 = ctx.currentTime;
  const startSample = wiz.sel.start;
  const tick = () => {
    if (previewBtn !== wizReviewPlay || !player.isOneShotPlaying()) {
      reviewPlayheadRaf = 0;
      trimEditor.setPlayhead(null);
      return;
    }
    trimEditor.setPlayhead(startSample + (ctx.currentTime - t0) * 24000);
    reviewPlayheadRaf = requestAnimationFrame(tick);
  };
  cancelAnimationFrame(reviewPlayheadRaf);
  reviewPlayheadRaf = requestAnimationFrame(tick);
}

window.addEventListener('resize', () => {
  if (wiz.step === 'review' && trimEditor && wiz.sourcePcm) trimEditor.redraw();
});

const WIZ_TITLES = {
  method: 'New voice',
  script: 'Pick a script',
  record: 'Record your voice',
  upload: 'Upload audio',
  review: 'Review & save',
  saving: 'Saving…',
  done: 'Done',
};

function wizReset() {
  wiz.step = null;
  wiz.method = null;
  wiz.source = null;
  wiz.scriptId = null;
  wiz.lang = null;
  wiz.scriptText = '';
  stopReviewPreview();
  wiz.sourcePcm = null;
  wiz.sel = null;
  wiz.pcm = null;
  wiz.duration = 0;
  wiz.presetTokens = null;
  wiz.presetTokenCount = 0;
  wiz.presetSel = null;
  wiz.savedVoiceId = null;
  wiz.levelRing.fill(0);
  if (wizTranscript) wizTranscript.value = '';
  if (wizVoiceName) wizVoiceName.value = '';
  if (wizCustomText) wizCustomText.value = '';
  if (wizCustomWrap) wizCustomWrap.classList.add('hidden');
  if (wizUploadError) wizUploadError.classList.add('hidden');
  if (wizSaveBlocker) wizSaveBlocker.classList.add('hidden');
  if (wizSaveBtn) wizSaveBtn.disabled = false;
}

function openVoiceWizard(startStep = 'method', preset = null) {
  wizStopMedia();
  wizReset();
  if (preset) {
    wiz.source = preset.source || 'generated';
    if (preset.pcm) wizSetSource(preset.pcm, preset.selection || null);
    wiz.presetTokens = preset.tokens || null;
    wiz.presetTokenCount = preset.tokenCount || 0;
    if (wizTranscript) wizTranscript.value = preset.transcript || '';
  }
  if (voicesHome) voicesHome.style.display = 'none';
  if (voiceWizard) voiceWizard.style.display = 'flex';
  wizGoto(startStep);
}

function closeVoiceWizard() {
  wizStopMedia();
  wizReset();
  if (voiceWizard) voiceWizard.style.display = 'none';
  if (voicesHome) voicesHome.style.display = 'flex';
  renderVoicesView();
}

function wizGoto(step) {
  if (wiz.step === 'record' && step !== 'record') wizStopMedia();
  wiz.step = step;
  document.querySelectorAll('#voice-wizard .wizard-step').forEach(el => el.classList.remove('active'));
  const el = $('wiz-step-' + step);
  if (el) el.classList.add('active');
  if (wizTitle) wizTitle.textContent = WIZ_TITLES[step] || 'New voice';
  if (wizBackBtn) wizBackBtn.style.visibility = step === 'saving' ? 'hidden' : 'visible';
  if (wizCloseBtn) wizCloseBtn.style.visibility = step === 'saving' ? 'hidden' : 'visible';

  if (step === 'script') renderScriptList();
  if (step === 'record') startMicCheck(wiz.deviceId || undefined);
  if (step === 'review') {
    requestAnimationFrame(() => {
      if (wiz.sourcePcm && trimEditor) trimEditor.load(wiz.sourcePcm, wiz.sel);
    });
    updateReviewVerdict();
    wizReviewBack.textContent = wiz.source === 'recorded' ? '← Re-record'
      : wiz.source === 'uploaded' ? '← Choose another file'
      : 'Cancel';
    wizTranscriptHint.textContent = defaultTranscriptHint();
    if (!wizVoiceName.value) wizVoiceName.focus({ preventScroll: true });
  } else {
    stopReviewPreview();
  }
}

function wizBack() {
  switch (wiz.step) {
    case 'method': closeVoiceWizard(); break;
    case 'script': wizGoto('method'); break;
    case 'record': wizGoto('script'); break;
    case 'upload': wizGoto('method'); break;
    case 'review':
      if (wiz.source === 'recorded') wizGoto('record');
      else if (wiz.source === 'uploaded') wizGoto('upload');
      else closeVoiceWizard();
      break;
    case 'done': closeVoiceWizard(); break;
    default: closeVoiceWizard();
  }
}

if (wizBackBtn) wizBackBtn.addEventListener('click', wizBack);
if (wizCloseBtn) wizCloseBtn.addEventListener('click', closeVoiceWizard);
if (wizReviewBack) wizReviewBack.addEventListener('click', wizBack);
if (newVoiceBtn) newVoiceBtn.addEventListener('click', () => openVoiceWizard('method'));
if (voicesEmptyCta) voicesEmptyCta.addEventListener('click', () => openVoiceWizard('method'));
onActivate(wizMethodRecord, () => { wiz.method = 'record'; wizGoto('script'); });
onActivate(wizMethodUpload, () => { wiz.method = 'upload'; wizGoto('upload'); });

// ── Script picker ──

function renderScriptList() {
  if (!wizScriptList) return;
  wizScriptList.innerHTML = '';
  const mkCard = (id, label, langTag, preview) => {
    const card = document.createElement('div');
    card.className = 'script-card' + (wiz.scriptId === id ? ' selected' : '');
    card.dataset.scriptId = id;
    card.setAttribute('role', 'button');
    card.tabIndex = 0;
    const head = document.createElement('div');
    head.className = 'sc-head';
    const title = document.createElement('span');
    title.textContent = label;
    head.appendChild(title);
    if (langTag) {
      const lt = document.createElement('span');
      lt.className = 'sc-lang';
      lt.textContent = langTag;
      head.appendChild(lt);
    }
    const prev = document.createElement('div');
    prev.className = 'sc-preview';
    prev.textContent = preview;
    card.append(head, prev);
    const select = () => selectScript(id);
    card.addEventListener('click', select);
    card.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); select(); } });
    return card;
  };
  for (const s of VOICE_SCRIPTS) wizScriptList.appendChild(mkCard(s.id, s.label, s.lang.toUpperCase(), s.text));
  wizScriptList.appendChild(mkCard('custom', 'Use your own text', null, 'Write 2-3 sentences and read them aloud.'));
  updateScriptNext();
}

function selectScript(id) {
  wiz.scriptId = id;
  const script = VOICE_SCRIPTS.find(x => x.id === id);
  wiz.lang = script ? script.lang : null;
  wiz.scriptText = script ? script.text : wizCustomText.value.trim();
  wizCustomWrap.classList.toggle('hidden', id !== 'custom');
  document.querySelectorAll('#wiz-script-list .script-card').forEach(c => {
    c.classList.toggle('selected', c.dataset.scriptId === id);
  });
  updateScriptNext();
  if (id === 'custom') wizCustomText.focus({ preventScroll: true });
}

function updateScriptNext() {
  if (!wizScriptNext) return;
  wizScriptNext.disabled = !(wiz.scriptId && (wiz.scriptId !== 'custom' || wiz.scriptText.length >= 20));
}

if (wizCustomText) {
  wizCustomText.addEventListener('input', () => {
    if (wiz.scriptId === 'custom') {
      wiz.scriptText = wizCustomText.value.trim();
      updateScriptNext();
    }
  });
}

if (wizScriptNext) {
  wizScriptNext.addEventListener('click', () => {
    if (wizScriptNext.disabled) return;
    wizScriptDisplay.textContent = wiz.scriptText;
    wizGoto('record');
  });
}

// ── Mic check + live metering ──

function setMicStatus(live) {
  if (wizMicStatus) wizMicStatus.classList.toggle('off', !live);
}

function hideMicError() {
  if (wizMicError) wizMicError.classList.add('hidden');
}

function showMicError(err) {
  setMicStatus(false);
  if (wizRecordBtn) wizRecordBtn.disabled = true;
  let text;
  if (err && (err.name === 'NotAllowedError' || err.name === 'SecurityError')) {
    text = "Microphone access was denied. Allow the microphone in your browser's site settings, then try again.";
  } else if (err && (err.name === 'NotFoundError' || err.name === 'OverconstrainedError')) {
    text = 'No microphone found. Plug one in, or upload an audio file instead.';
  } else {
    text = "Couldn't start the microphone: " + (err && err.message ? err.message : 'unknown error');
  }
  if (wizMicErrorText) wizMicErrorText.textContent = text;
  if (wizMicError) wizMicError.classList.remove('hidden');
}

let micCheckSeq = 0;

async function startMicCheck(deviceId) {
  const seq = ++micCheckSeq; // invalidates overlapping calls (double retry, quick device switch)
  wizStopMedia();
  hideMicError();
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: deviceId ? { deviceId: { exact: deviceId } } : true,
    });
  } catch (err) {
    if (seq === micCheckSeq) showMicError(err);
    return;
  }
  if (seq !== micCheckSeq || wiz.step !== 'record' || currentView !== 'voices') {
    // Superseded by a newer mic check, or the user navigated away while the
    // permission prompt was open — release this stream, it is not tracked.
    stream.getTracks().forEach(t => t.stop());
    return;
  }
  wiz.stream = stream;
  const track = wiz.stream.getAudioTracks()[0];
  wiz.deviceId = deviceId || (track && track.getSettings ? track.getSettings().deviceId : null) || null;
  await populateMicSelect();
  if (seq !== micCheckSeq) return; // superseded during enumeration; its wizStopMedia released our stream
  // Separate default-rate context for analysis only (playback ctx is pinned to 24 kHz)
  wiz.micCtx = new AudioContext();
  if (wiz.micCtx.state === 'suspended') wiz.micCtx.resume().catch(() => {});
  wiz.srcNode = wiz.micCtx.createMediaStreamSource(wiz.stream);
  wiz.analyser = wiz.micCtx.createAnalyser();
  wiz.analyser.fftSize = 2048;
  wiz.srcNode.connect(wiz.analyser);
  if (track) track.addEventListener('ended', onMicTrackEnded);
  if (navigator.mediaDevices.addEventListener) {
    navigator.mediaDevices.addEventListener('devicechange', populateMicSelect);
  }
  setMicStatus(true);
  if (wizRecordBtn) wizRecordBtn.disabled = false;
  wiz.levelRing.fill(0);
  startMeterLoop();
}

function onMicTrackEnded() {
  if (wiz.recording) stopWizRecording();
  else setMicStatus(false);
}

async function populateMicSelect() {
  if (!wizMicSelect) return;
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const inputs = devices.filter(d => d.kind === 'audioinput');
    wizMicSelect.innerHTML = '';
    inputs.forEach((d, i) => {
      const opt = document.createElement('option');
      opt.value = d.deviceId;
      opt.textContent = d.label || `Microphone ${i + 1}`;
      wizMicSelect.appendChild(opt);
    });
    if (wiz.deviceId) wizMicSelect.value = wiz.deviceId;
  } catch { /* enumeration unavailable */ }
}

if (wizMicSelect) {
  wizMicSelect.addEventListener('change', () => startMicCheck(wizMicSelect.value || undefined));
}
if (wizMicRetry) wizMicRetry.addEventListener('click', () => startMicCheck(wiz.deviceId || undefined));
if (wizMicFallbackUpload) wizMicFallbackUpload.addEventListener('click', () => { wiz.method = 'upload'; wizGoto('upload'); });

function recZoneLabel(elapsed) {
  if (elapsed < 3) return 'Keep going — at least 3 seconds needed';
  if (elapsed < 5) return 'Usable — a few more seconds is better';
  if (elapsed < 12) return 'Good length — stop whenever you finish the script';
  return 'Long enough — stopping automatically at 15s';
}

function startMeterLoop() {
  if (!wiz.analyser) return;
  const buf = new Float32Array(wiz.analyser.fftSize);
  const ring = wiz.levelRing;
  let lastRingPush = 0;
  const silenceWindow = []; // { t, rms }

  const tick = (now) => {
    wiz.raf = 0;
    if (!wiz.analyser) return;
    wiz.analyser.getFloatTimeDomainData(buf);
    let sum = 0;
    for (let i = 0; i < buf.length; i++) sum += buf[i] * buf[i];
    const rms = Math.sqrt(sum / buf.length);

    // Level bar
    if (wizLevelFill) {
      wizLevelFill.style.width = Math.min(100, rms * 400) + '%';
      wizLevelFill.classList.toggle('hot', rms > 0.35);
    }

    // Rolling live waveform (bars scroll left)
    if (now - lastRingPush > 50) {
      lastRingPush = now;
      ring.copyWithin(0, 1);
      ring[ring.length - 1] = rms;
      drawLiveBars(wizLiveWave, ring);
    }

    // Silence detection over the last ~2 s
    silenceWindow.push({ t: now, rms });
    while (silenceWindow.length && now - silenceWindow[0].t > 2200) silenceWindow.shift();
    const span = silenceWindow.length ? now - silenceWindow[0].t : 0;
    const allQuiet = span >= 2000 && silenceWindow.every(s => s.rms < 0.01);
    if (wizSilenceHint) {
      wizSilenceHint.classList.toggle('hidden', !allQuiet);
      if (allQuiet) {
        wizSilenceHint.textContent = wiz.recording
          ? "We can't hear you — check the microphone above or speak louder."
          : 'No signal from this microphone. Try another one from the list above.';
      }
    }

    // Recording timeline
    if (wiz.recording) {
      const elapsed = (now - wiz.recStartTs) / 1000;
      if (wizRecElapsed) wizRecElapsed.textContent = elapsed.toFixed(1) + 's';
      if (wizRecFill) wizRecFill.style.width = Math.min(100, (elapsed / 15) * 100) + '%';
      if (wizRecZone) wizRecZone.textContent = recZoneLabel(elapsed);
      if (elapsed >= 15) stopWizRecording();
    }

    wiz.raf = requestAnimationFrame(tick);
  };
  cancelAnimationFrame(wiz.raf);
  wiz.raf = requestAnimationFrame(tick);
}

// Live level bars in the same neon style as the output waveform
function drawLiveBars(canvas, ring) {
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 2;
  const w = (canvas.offsetWidth || 300) * dpr;
  const h = (canvas.offsetHeight || 72) * dpr;
  canvas.width = w;
  canvas.height = h;
  ctx.clearRect(0, 0, w, h);
  const n = ring.length;
  const totalBar = w / n;
  const gap = Math.max(1, Math.floor(totalBar * 0.25));
  const barWidth = Math.max(1, totalBar - gap);
  for (let i = 0; i < n; i++) {
    const rms = ring[i];
    const barH = Math.max(2, Math.min(h * 0.92, rms * h * 2.5));
    const x = i * totalBar;
    const y = (h - barH) / 2;
    const intensity = Math.min(1, rms * 5);
    ctx.fillStyle = `rgba(74, 222, 128, ${0.35 + intensity * 0.65})`;
    ctx.shadowColor = `rgba(74, 222, 128, ${intensity * 0.8})`;
    ctx.shadowBlur = intensity * 10;
    ctx.fillRect(x, y, barWidth, barH);
  }
  ctx.shadowBlur = 0;
}

// ── Record button: idle → 3-2-1 countdown → recording → stop ──

if (wizRecordBtn) {
  wizRecordBtn.addEventListener('click', () => {
    if (wiz.recording) { stopWizRecording(); return; }
    if (wiz.counting) { cancelWizCountdown(); return; }
    if (!wiz.stream) { startMicCheck(wiz.deviceId || undefined); return; }
    startWizCountdown();
  });
}

function cancelWizCountdown() {
  wiz.counting = false;
  if (wizRecordBtn) wizRecordBtn.classList.remove('counting');
  if (wizRecordBtnInner) wizRecordBtnInner.innerHTML = wizRecordBtnIdleHTML;
  if (wizRecordLabel) wizRecordLabel.textContent = WIZ_RECORD_IDLE_LABEL;
}

async function startWizCountdown() {
  if (!wiz.stream) return;
  wiz.counting = true;
  wizRecordBtn.classList.add('counting');
  wizRecordLabel.textContent = 'Starting… tap to cancel';
  for (let i = 3; i >= 1; i--) {
    if (!wiz.counting) return;
    wizRecordBtnInner.textContent = String(i);
    await new Promise(r => setTimeout(r, 1000));
  }
  if (!wiz.counting) return;
  wiz.counting = false;
  wizRecordBtn.classList.remove('counting');
  startWizRecording();
}

function startWizRecording() {
  if (!wiz.stream || !wiz.stream.active) {
    // Mic was unplugged/revoked during the countdown
    cancelWizCountdown();
    showMicError({ name: 'NotFoundError' });
    return;
  }
  wiz.chunks = [];
  let opts;
  try {
    opts = MediaRecorder.isTypeSupported('audio/webm;codecs=opus') ? { mimeType: 'audio/webm;codecs=opus' } : undefined;
  } catch { opts = undefined; }
  try {
    wiz.recorder = new MediaRecorder(wiz.stream, opts);
    wiz.recorder.ondataavailable = (e) => { if (e.data.size > 0) wiz.chunks.push(e.data); };
    wiz.recorder.onstop = onWizRecordingStopped;
    wiz.recorder.start();
  } catch (e) {
    wiz.recorder = null;
    cancelWizCountdown();
    showMicError(e);
    return;
  }
  wiz.recStartTs = performance.now();
  wiz.recording = true;
  wizRecordBtn.classList.add('recording');
  wizRecordBtnInner.textContent = '■';
  wizRecordLabel.textContent = 'Recording — tap to stop';
  if (wizRecZone) wizRecZone.textContent = recZoneLabel(0);
}

function stopWizRecording() {
  if (wiz.recorder && wiz.recorder.state === 'recording') {
    wiz.recording = false;
    try { wiz.recorder.stop(); } catch { /* already stopped */ }
  }
}

async function onWizRecordingStopped() {
  wiz.recording = false;
  wiz.recorder = null;
  if (wizRecordBtn) {
    wizRecordBtn.classList.remove('recording');
    wizRecordBtnInner.innerHTML = wizRecordBtnIdleHTML;
    wizRecordLabel.textContent = WIZ_RECORD_IDLE_LABEL;
  }
  const blob = new Blob(wiz.chunks, { type: 'audio/webm' });
  wiz.chunks = [];
  if (blob.size === 0) return;
  let pcm;
  try {
    pcm = await decodeRefAudio(blob);
  } catch (e) {
    toast("Couldn't process the recording: " + e.message, { type: 'error' });
    return;
  }
  wiz.source = 'recorded';
  wiz.presetTokens = null;
  wizSetSource(pcm);
  wizTranscript.value = wiz.scriptText || '';
  wizGoto('review');
}

function wizStopMedia() {
  cancelAnimationFrame(wiz.raf);
  wiz.raf = 0;
  wiz.counting = false;
  if (wiz.recorder && wiz.recorder.state === 'recording') {
    wiz.recorder.ondataavailable = null;
    wiz.recorder.onstop = null;
    try { wiz.recorder.stop(); } catch { /* already stopped */ }
  }
  wiz.recorder = null;
  wiz.chunks = [];
  wiz.recording = false;
  if (wiz.srcNode) { try { wiz.srcNode.disconnect(); } catch { /* detached */ } wiz.srcNode = null; }
  wiz.analyser = null;
  if (wiz.micCtx) { try { wiz.micCtx.close(); } catch { /* closed */ } wiz.micCtx = null; }
  if (wiz.stream) { wiz.stream.getTracks().forEach(t => t.stop()); wiz.stream = null; }
  if (navigator.mediaDevices && navigator.mediaDevices.removeEventListener) {
    navigator.mediaDevices.removeEventListener('devicechange', populateMicSelect);
  }
  if (wizRecordBtn) {
    wizRecordBtn.classList.remove('recording', 'counting');
    if (wizRecordBtnInner) wizRecordBtnInner.innerHTML = wizRecordBtnIdleHTML;
    if (wizRecordLabel) wizRecordLabel.textContent = WIZ_RECORD_IDLE_LABEL;
    if (wizRecElapsed) wizRecElapsed.textContent = '0.0s';
    if (wizRecFill) wizRecFill.style.width = '0%';
    if (wizRecZone) wizRecZone.textContent = WIZ_ZONE_IDLE_LABEL;
    if (wizSilenceHint) wizSilenceHint.classList.add('hidden');
    if (wizLevelFill) { wizLevelFill.style.width = '0%'; wizLevelFill.classList.remove('hot'); }
  }
  setMicStatus(false);
}

window.addEventListener('beforeunload', wizStopMedia);

// ── Upload step ──

function showUploadError(text) {
  if (wizUploadError) {
    wizUploadError.textContent = text;
    wizUploadError.classList.remove('hidden');
  }
}

async function handleWizFile(file) {
  if (!file) return;
  if (wizUploadError) wizUploadError.classList.add('hidden');
  if (file.size > 25 * 1024 * 1024) {
    showUploadError('File too large — use a short clip under 25 MB.');
    return;
  }
  let pcm;
  try {
    pcm = await decodeRefAudio(file);
  } catch {
    showUploadError("Couldn't read this file. Use MP3, WAV, M4A, or OGG.");
    return;
  }
  wiz.source = 'uploaded';
  wiz.presetTokens = null;
  wizSetSource(pcm);
  wiz.scriptId = null;
  wiz.lang = null;
  wizTranscript.value = '';
  if (wizFileInput) wizFileInput.value = '';
  wizGoto('review');
}

if (wizDropzone) {
  wizDropzone.addEventListener('click', () => wizFileInput.click());
  wizDropzone.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); wizFileInput.click(); } });
  wizDropzone.addEventListener('dragover', (e) => { e.preventDefault(); wizDropzone.classList.add('dragging'); });
  wizDropzone.addEventListener('dragleave', () => wizDropzone.classList.remove('dragging'));
  wizDropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    wizDropzone.classList.remove('dragging');
    handleWizFile(e.dataTransfer.files && e.dataTransfer.files[0]);
  });
}
if (wizFileInput) wizFileInput.addEventListener('change', () => handleWizFile(wizFileInput.files[0]));

// ── Review step ──

function updateReviewVerdict() {
  const d = wiz.duration || 0;
  if (wizReviewDuration) wizReviewDuration.textContent = d.toFixed(1) + 's';
  let cls, label, blocker = null;
  if (d < 3) { cls = 'bad'; label = 'Too short'; blocker = 'Too short to clone — use at least 3 seconds of speech.'; }
  else if (d < 5) { cls = 'warn'; label = 'Usable'; }
  else if (d <= 12) { cls = 'good'; label = 'Good length'; }
  else { cls = 'warn'; label = 'Long'; }
  if (wizReviewVerdict) {
    wizReviewVerdict.className = 'verdict-chip ' + cls;
    wizReviewVerdict.textContent = label;
  }
  if (wizSaveBtn) wizSaveBtn.disabled = d < 3;
  if (wizSaveBlocker) {
    if (blocker) { wizSaveBlocker.textContent = blocker; wizSaveBlocker.classList.remove('hidden'); }
    else wizSaveBlocker.classList.add('hidden');
  }
}

if (wizReviewPlay) {
  wizReviewPlay.addEventListener('click', () => {
    if (!wiz.pcm) return;
    togglePcmPreview(wiz.pcm, wizReviewPlay);
    if (trimEditor && previewBtn === wizReviewPlay && player.isOneShotPlaying()) startReviewPlayhead();
  });
}

// ── Save flow (persist first, encode after — a voice can never be lost) ──

async function wizSave() {
  const name = wizVoiceName.value.trim();
  if (!name) {
    wizSaveBlocker.textContent = 'Give the voice a name.';
    wizSaveBlocker.classList.remove('hidden');
    wizVoiceName.focus();
    return;
  }
  if (!wiz.pcm || wiz.duration < 3) { updateReviewVerdict(); return; }
  wizSaveBtn.disabled = true;
  stopReviewPreview();

  // Preset tokens (chunk 1 of a design-mode generation) only match their own selection
  const keepTokens = !!(wiz.presetTokens && wiz.presetSel && wiz.sel
    && wiz.sel.start === wiz.presetSel.start && wiz.sel.end === wiz.presetSel.end);
  const refAudio = keepTokens ? wiz.pcm : prepareReference(wiz.pcm, 24000);

  const record = {
    id: 'v-' + Date.now(),
    version: 2,
    name,
    refAudio,
    refText: wizTranscript.value.trim() || null,
    tokens: keepTokens ? new Int32Array(wiz.presetTokens) : null,
    tokenCount: keepTokens ? wiz.presetTokenCount : null,
    duration: refAudio.length / 24000,
    refPrep: REF_PREP_VERSION,
    createdAt: Date.now(),
    scriptId: wiz.scriptId,
    lang: wiz.lang,
    source: wiz.source || 'recorded',
  };

  wizSavingStatus.textContent = 'Saving…';
  wizGoto('saving');
  try {
    await saveVoice(record);
  } catch (e) {
    wizGoto('review');
    wizSaveBtn.disabled = false;
    toast('Could not save the voice: ' + e.message, { type: 'error' });
    return;
  }
  wiz.savedVoiceId = record.id;

  let encodeNote = null;
  if (!record.tokens) {
    if (isReady && encoderAvailable) {
      try {
        wizSavingStatus.textContent = 'Analyzing voice… (one-time setup)';
        await encodeVoiceShared(record); // persists via merge, shares in-flight work
      } catch (e) {
        encodeNote = e.code === 'encoder-unavailable'
          ? 'This device could not run the voice encoder. The recording is saved and will finish setting up on a device with more memory.'
          : 'Voice analysis failed — it will be retried automatically before the next use.';
      }
    } else if (!isReady) {
      encodeNote = 'Models are still loading — the voice will finish setting up automatically.';
    } else {
      encodeNote = 'The voice encoder is unavailable on this device. The recording is saved; cloning it here may be limited.';
    }
  }

  await refreshVoices();
  selectedSavedVoice = voicesCache.find(v => v.id === record.id) || null;
  renderStudioVoicePicker();
  updateVoiceUI();
  updateVoicesBadge();

  wizDoneSub.textContent = `"${name}" is ready to use.`;
  wizDoneNote.textContent = encodeNote || '';
  wizDoneNote.classList.toggle('hidden', !encodeNote);
  wizTestBtn.disabled = !isReady || isGenerating;
  wizTestBtn.title = !isReady ? 'Models still loading…' : '';
  wizSaveBtn.disabled = false;
  wizGoto('done');
  toast(`Voice "${name}" saved`, { type: 'success' });
}

if (wizSaveBtn) wizSaveBtn.addEventListener('click', wizSave);
if (wizVoiceName) {
  wizVoiceName.addEventListener('keydown', (e) => { if (e.key === 'Enter') { e.preventDefault(); wizSave(); } });
}
if (wizTestBtn) {
  wizTestBtn.addEventListener('click', () => {
    const v = voicesCache.find(x => x.id === wiz.savedVoiceId);
    if (v) testVoice(v, wizTestBtn);
  });
}
if (wizGotoStudio) {
  wizGotoStudio.addEventListener('click', () => {
    closeVoiceWizard();
    switchView('studio');
  });
}

// ─── UI ─────────────────────────────────────────────────────────────────────

function setStatus(text) {
  statusEl.textContent = text;
}

// ─── View navigation (sidebar + mobile tabs) ────────────────────────────────

function switchView(viewName) {
  if (currentView === 'voices' && viewName !== 'voices') wizStopMedia();

  document.querySelectorAll('.app-view').forEach(v => { v.style.display = 'none'; });
  const target = $('view-' + viewName);
  if (target) target.style.display = 'flex';

  document.querySelectorAll('.sidebar-tab, .mobile-tab').forEach(t => {
    const active = t.dataset.view === viewName;
    t.classList.toggle('active-tab', active);
    t.setAttribute('aria-selected', active ? 'true' : 'false');
    t.tabIndex = active ? 0 : -1;
  });

  currentView = viewName;

  if (viewName === 'library') {
    renderLibrary();
    requestAnimationFrame(() => {
      document.querySelectorAll('#library-list canvas').forEach((c, i) => {
        if (history[i]) drawMiniWaveform(c, history[i].pcm);
      });
    });
  }
  if (viewName === 'settings') calculateStorage();
  if (viewName === 'voices') {
    renderVoicesView();
    if (voiceWizard && voiceWizard.style.display !== 'none' && wiz.step === 'record' && !wiz.stream) {
      startMicCheck(wiz.deviceId || undefined);
    }
  }
  if (viewName === 'studio') requestAnimationFrame(() => player.redraw());

  if (target) target.focus({ preventScroll: true });
}

function initTabGroup(selector, orientation) {
  const tabs = Array.from(document.querySelectorAll(selector));
  const container = tabs[0] ? tabs[0].parentElement : null;
  if (container) {
    container.setAttribute('role', 'tablist');
    container.setAttribute('aria-orientation', orientation);
  }
  tabs.forEach((tab, idx) => {
    tab.setAttribute('role', 'tab');
    tab.setAttribute('aria-selected', tab.classList.contains('active-tab') ? 'true' : 'false');
    tab.tabIndex = tab.classList.contains('active-tab') ? 0 : -1;
    tab.addEventListener('click', (e) => { e.preventDefault(); switchView(tab.dataset.view); });
    tab.addEventListener('keydown', (e) => {
      const prev = orientation === 'vertical' ? 'ArrowUp' : 'ArrowLeft';
      const next = orientation === 'vertical' ? 'ArrowDown' : 'ArrowRight';
      let target = null;
      if (e.key === prev) target = tabs[(idx - 1 + tabs.length) % tabs.length];
      else if (e.key === next) target = tabs[(idx + 1) % tabs.length];
      else if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); switchView(tab.dataset.view); return; }
      else return;
      e.preventDefault();
      target.focus();
      switchView(target.dataset.view);
    });
  });
}

initTabGroup('.sidebar-tab', 'vertical');
initTabGroup('.mobile-tab', 'horizontal');

// ─── Global keyboard shortcuts ──────────────────────────────────────────────

document.addEventListener('keydown', (e) => {
  if (e.key !== 'Escape') return;
  if (isDialogOpen()) return; // the dialog's own capture-phase handler closes it
  if (isGenerating) { cancelGeneration(); return; }
  if (player.isPlaying() || player.isOneShotPlaying()) {
    player.stop();
    setStatus('Ready');
  }
});

// ─── Settings: storage usage + data management ──────────────────────────────
// (moved from the inline index.html script so it can use the dialog module)

function formatBytes(bytes) {
  if (!bytes || bytes <= 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.min(sizes.length - 1, Math.floor(Math.log(bytes) / Math.log(k)));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
}

async function getDBSize(dbName) {
  // Never CREATE a database just to measure it — a versionless open on a
  // freshly deleted DB would recreate it with zero object stores.
  try {
    if (indexedDB.databases) {
      const list = await indexedDB.databases();
      if (!list.some(d => d.name === dbName)) return 0;
    }
  } catch { /* enumeration unsupported — openStoreDB self-heals anyway */ }
  return new Promise((resolve) => {
    const req = indexedDB.open(dbName);
    req.onsuccess = () => {
      const db = req.result;
      const storeNames = Array.from(db.objectStoreNames);
      if (storeNames.length === 0) { db.close(); resolve(0); return; }
      const tx = db.transaction(storeNames, 'readonly');
      let total = 0;
      let pending = storeNames.length;
      storeNames.forEach(name => {
        const store = tx.objectStore(name);
        const r = store.getAll();
        r.onsuccess = () => {
          for (const val of r.result) {
            if (val instanceof ArrayBuffer) total += val.byteLength;
            else if (val instanceof Blob) total += val.size;
            else if (val && val.byteLength !== undefined) total += val.byteLength;
            else if (val && typeof val === 'object') {
              for (const key of Object.keys(val)) {
                const field = val[key];
                if (field && field.byteLength !== undefined) total += field.byteLength;
                else if (typeof field === 'string') total += field.length * 2;
                else total += 8;
              }
            } else total += 8;
          }
          if (--pending === 0) { db.close(); resolve(total); }
        };
        r.onerror = () => { if (--pending === 0) { db.close(); resolve(total); } };
      });
    };
    req.onerror = () => resolve(0);
  });
}

async function getCacheAPISize(cacheName) {
  try {
    const cache = await caches.open(cacheName);
    const keys = await cache.keys();
    let total = 0;
    for (const req of keys) {
      const resp = await cache.match(req);
      if (resp) {
        const len = parseInt(resp.headers.get('Content-Length') || '0', 10);
        if (len > 0) total += len;
        else total += (await resp.blob()).size;
      }
    }
    return total;
  } catch { return 0; }
}

async function calculateStorage() {
  const modelsEl = $('storage-models');
  if (!modelsEl) return;
  const modelsSize = await getCacheAPISize('omnivoice-models-v1');
  const historySize = await getDBSize(HISTORY_DB);
  const voicesSize = await getDBSize(VOICE_DB);
  modelsEl.textContent = formatBytes(modelsSize);
  $('storage-history').textContent = formatBytes(historySize);
  $('storage-voices').textContent = formatBytes(voicesSize);
  $('storage-total').textContent = formatBytes(modelsSize + historySize + voicesSize);
}

function clearDB(dbName) {
  return new Promise((resolve) => {
    const req = indexedDB.deleteDatabase(dbName);
    req.onsuccess = () => resolve();
    req.onerror = () => resolve();
    req.onblocked = () => resolve();
  });
}

const clearCacheBtn = $('clear-cache-btn');
if (clearCacheBtn) {
  clearCacheBtn.addEventListener('click', async () => {
    const ok = await confirmDialog({
      title: 'Clear cached models?',
      body: 'You will need to re-download ~3 GB on next load.',
      confirmLabel: 'Clear models',
      danger: true,
    });
    if (!ok) return;
    await caches.delete('omnivoice-models-v1');
    await clearDB('omnivoice-cache'); // legacy cache cleanup
    calculateStorage();
    toast('Cached models cleared', { type: 'success' });
  });
}

const clearHistoryBtn = $('clear-history-btn');
if (clearHistoryBtn) {
  clearHistoryBtn.addEventListener('click', async () => {
    const ok = await confirmDialog({
      title: 'Clear generation history?',
      body: 'All saved generations will be deleted. This cannot be undone.',
      confirmLabel: 'Delete all',
      danger: true,
    });
    if (!ok) return;
    try { await clearHistoryStore(); } catch (e) { console.warn('Clear history failed:', e); }
    history = [];
    updateLibraryBadge();
    if (currentView === 'library') renderLibrary();
    calculateStorage();
    toast('Generation history cleared', { type: 'success' });
  });
}

const clearVoicesBtn = $('clear-voices-btn');
if (clearVoicesBtn) {
  clearVoicesBtn.addEventListener('click', async () => {
    const ok = await confirmDialog({
      title: 'Clear saved voices?',
      body: 'All cloned voices will be deleted. This cannot be undone.',
      confirmLabel: 'Delete voices',
      danger: true,
    });
    if (!ok) return;
    try { await clearVoicesStore(); } catch (e) { console.warn('Clear voices failed:', e); }
    selectedSavedVoice = null;
    await refreshVoices();
    renderStudioVoicePicker();
    updateVoiceUI();
    updateVoicesBadge();
    if (currentView === 'voices') renderVoicesView();
    calculateStorage();
    toast('Saved voices cleared', { type: 'success' });
  });
}

const clearAllBtn = $('clear-all-btn');
if (clearAllBtn) {
  clearAllBtn.addEventListener('click', async () => {
    const ok = await confirmDialog({
      title: 'Clear ALL local data?',
      body: 'Models, history, and voices will be deleted. Models (~3 GB) will re-download on next use.',
      confirmLabel: 'Clear everything',
      danger: true,
    });
    if (!ok) return;
    await caches.delete('omnivoice-models-v1');
    await clearDB('omnivoice-cache');
    await clearDB(HISTORY_DB);
    await clearDB(VOICE_DB);
    location.reload();
  });
}

// ─── Init ───────────────────────────────────────────────────────────────────

(async () => {
  await refreshVoices();
  renderStudioVoicePicker();
  updateVoicesBadge();
  updateVoiceUI();
  try {
    history = await loadHistory();
    updateLibraryBadge();
    pruneHistoryDB(); // clean up legacy unbounded rows, fire-and-forget
  } catch (e) { console.warn('Failed to load history:', e); }
  initWorker();
})();
