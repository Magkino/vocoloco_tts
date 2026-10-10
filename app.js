/**
 * VocoLoco — Main application
 * Streams generation sentence-by-sentence, manages voices (voice wizard +
 * cached voice tokens), the library, settings, and all studio UI.
 */

import { toast, confirmDialog, isDialogOpen } from './ui-dialogs.js?v=2';
import { StreamingPlayer, drawBarVisualizer, drawMiniWaveform } from './player.js';
import { chunkText } from './text-chunker.js?v=3';
import { TrimEditor } from './trim-editor.js?v=3';
import { prepareReference, defaultSelection, REF_PREP_VERSION } from './audio-postprocess.js?v=3';
import { diskUsage, clearDisk } from './workers/model-files.js?v=4';
import { refInfoKey } from './workers/voice-pitch.js';

const MODEL_BASE_URL = 'https://huggingface.co/Gigsu/vocoloco-onnx/resolve/main';
// Anywhere but the official site (local development, self-hosting): a complete mirror in ./models
// (scripts/fetch-models.sh) replaces Hugging Face, so reloads don't re-download gigabytes
const LOCAL_MODEL_URL = new URL('models', location.href).href;
// Phones and tablets: their browsers kill a tab that holds much more than ~1.5 GB (iPhone), so
// one big model is in memory at a time there (tts-worker.js, lowMemory; Whisper, requestTranscript)
const LOW_MEMORY = /Android|iPhone|iPad|iPod/i.test(navigator.userAgent)
  || (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1) || !!navigator.userAgentData?.mobile;
const MAX_TEXT_LEN = 2000;
// OmniVoice's sampling defaults (num_step comes from the quality toggle)
const GEN_PARAMS = { guidanceScale: 2.0, tShift: 0.1 };

// "Test this voice", in the voice's language when known (detected by Whisper)
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
const speedRow = $('speed-row');
const speedEl = $('speed');
const speedValue = $('speed-value');
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
const loadingActions = $('loading-actions');
const downloadStartBtn = $('download-start');
const downloadLaterBtn = $('download-later');
const downloadSpeechOnlyBtn = $('download-speech-only');
const loadRetryBtn = $('load-retry');
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
const wizMicStatus = $('wiz-mic-status');
const wizMicSelect = $('wiz-mic-select');
const wizMicError = $('wiz-mic-error');
const wizMicErrorText = $('wiz-mic-error-text');
const wizMicRetry = $('wiz-mic-retry');
const wizMicFallbackUpload = $('wiz-mic-fallback-upload');
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
const wizAsrBtn = $('wiz-asr-btn');
const wizAsrLabel = $('wiz-asr-label');
const wizAsrReserve = $('wiz-asr-reserve');
const wizVoiceName = $('wiz-voice-name');
const wizReviewBack = $('wiz-review-back');
const wizSaveBtn = $('wiz-save-btn');
const wizSaveBlocker = $('wiz-save-blocker');
const wizSavingStatus = $('wiz-saving-status');
const wizSavingPct = $('wiz-saving-pct');
const wizDoneSub = $('wiz-done-sub');
const wizDoneNote = $('wiz-done-note');
const wizTestBtn = $('wiz-test-btn');
const wizGotoStudio = $('wiz-goto-studio');

const wizRecordBtnIdleHTML = wizRecordBtnInner ? wizRecordBtnInner.innerHTML : '';
const WIZ_RECORD_IDLE_LABEL = 'Tap to record, starts after a 3-2-1 countdown';

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
let sessionChainRef = null; // { tokens, tokenCount, rateTokens, text, pcm } from the last design-mode generation
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

// The worker measures a reference's pitch on its decoded audio (tts-worker.js,
// referenceInfo); kept with the saved voice whose tokens it was measured on, so that decode runs
// once per voice instead of once per session. Re-encoded tokens get a new key and a new measurement.
async function persistRefInfo({ key, pitch }) {
  for (const v of voicesCache) {
    if (!v.tokens || v.refInfo?.key === key || refInfoKey(v.tokens, 8) !== key) continue;
    const refInfo = { key, pitch };
    const fresh = (await getSavedVoices()).find((s) => s.id === v.id);
    if (fresh?.tokens && refInfoKey(fresh.tokens, 8) === key) await saveVoice(Object.assign(normalizeVoiceRecord(fresh), { refInfo }));
    v.refInfo = refInfo;
  }
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

// The voice encoder only analyzes new voices: never downloaded silently
let encoderCached = false;   // complete in the model cache (reported by the worker)
let encoderBytes = null;
let encoderConsent = false;  // the user said yes this session (ONNX Runtime; the engine: ensureCloning)
let speechBackend = null;    // 'engine' or ONNX Runtime's, once the speech model is ready

function canEncodeSilently() { return encoderCached || encoderConsent; }
const engineBackend = () => (speechBackend ? speechBackend === 'engine' : !!navigator.gpu);

async function confirmEncoderDownload() {
  if (canEncodeSilently()) return true;
  if (engineBackend()) return ensureCloning();
  const size = encoderBytes ? formatBytes(encoderBytes) : 'about 0.6 GB';
  encoderConsent = await confirmDialog({
    title: 'Download voice encoder?',
    body: `Needed once to clone voices.${mobileDataNote()}`,
    confirmLabel: `Download ${size}`,
    cancelLabel: 'Not now',
  });
  return encoderConsent;
}

// ─── Voice cloning models ───────────────────────────────────────────────────
// Cloning a voice takes two models besides the speech model: Whisper, which writes down what the
// recording says, and the voice encoder, which turns it into the voice's tokens. Both come as one
// download (download-worker.js): offered with the speech model on the first visit, or before the
// first new voice. Nothing in the voice wizard asks again.

const CLONING_FILES = ['engine/whisper-i8.bin', 'engine/encoder-f32.bin'];
const CLONING_KEY = 'vocoloco-cloning'; // '1': chosen, so an interrupted download goes on by itself
const cloning = { total: 0, stored: 0, loaded: 0, run: null }; // bytes; run: the download (Promise)
let cloningDialogUpdate = null; // the progress dialog's, while it is open

const cloningComplete = () => cloning.total > 0 && cloning.stored >= cloning.total;
const cloningLeft = () => (cloning.total ? cloning.total - cloning.stored : 1.5e9);

// A download-worker.js job: its final message (progress messages go to onProgress)
function downloadJob(type, urls, onProgress) {
  return new Promise((resolve, reject) => {
    const w = new Worker('workers/download-worker.js?v=1', { type: 'module' });
    w.onmessage = ({ data: m }) => {
      if (m.type === 'progress') { onProgress?.(m); return; }
      w.terminate();
      if (m.type === 'error') reject(new Error(m.message)); else resolve(m);
    };
    w.onerror = (e) => { w.terminate(); reject(new Error(e.message || 'The download could not start')); };
    w.postMessage({ type, urls });
  });
}

async function cloningUrls() {
  const base = await resolveModelBaseUrl();
  return CLONING_FILES.map((f) => `${base}/${f}`);
}

function markCloningStored() {
  encoderCached = true;
  localStorage.setItem(ASR_CACHED_KEY, '1');
}

// How much of the two is stored
async function checkCloning() {
  const s = await downloadJob('inspect', await cloningUrls());
  Object.assign(cloning, { total: s.totalBytes, stored: s.storedBytes });
  if (cloningComplete()) markCloningStored();
}

// The download, one at a time; progress in the first visit's box, the progress dialog or the
// activity pill
function downloadCloning() {
  localStorage.setItem(CLONING_KEY, '1');
  keepStorage();
  cloning.run ??= (async () => {
    await downloadJob('download', await cloningUrls(), (m) => {
      Object.assign(cloning, { loaded: m.loadedBytes, total: m.totalBytes });
      onCloningProgress();
    });
    cloning.stored = cloning.total;
    markCloningStored();
  })().finally(() => {
    cloning.run = null;
    if (!combinedDownload && activityLabel === 'Downloading voice cloning' && !isGenerating) hideProgress();
  });
  return cloning.run;
}

function onCloningProgress() {
  if (combinedDownload) { showCombinedProgress(); return; }
  activityLabel = 'Downloading voice cloning';
  if (cloning.total) setProgressPercent((cloning.loaded / cloning.total) * 100);
  cloningDialogUpdate?.();
}

// Before a new voice: the cloning models, asked for once and downloaded in one go. False when
// the user says no or lets it go on in the background
async function ensureCloning() {
  if (!engineBackend()) return true; // ONNX Runtime: its encoder asks on its own (confirmEncoderDownload)
  try { await checkCloning(); } catch { /* offline: say so below if anything is missing */ }
  if (cloningComplete()) return true;
  if (!cloning.run) {
    const ok = await confirmDialog({
      title: 'Download voice cloning?',
      body: `Cloning a voice needs two more models: speech recognition, which writes down what your recording says, and the voice encoder. One-time download, stored on this device.${mobileDataNote()}`,
      confirmLabel: `Download ${formatBytes(cloningLeft())}`,
      cancelLabel: 'Not now',
    });
    if (!ok) return false;
  }
  return cloningProgressDialog();
}

// Progress of the cloning download in a dialog; true once it is done, false when the user lets it
// go on in the background (the activity pill shows it, a toast says when it's ready)
function cloningProgressDialog() {
  return new Promise((resolve) => {
    const overlay = document.createElement('div');
    overlay.className = 'vl-overlay';
    const dialog = document.createElement('div');
    dialog.className = 'vl-dialog';
    dialog.setAttribute('role', 'dialog');
    dialog.setAttribute('aria-modal', 'true');
    dialog.setAttribute('aria-labelledby', 'vl-cloning-title');
    const title = document.createElement('h3');
    title.id = 'vl-cloning-title';
    title.textContent = 'Downloading voice cloning';
    const body = document.createElement('p');
    body.textContent = 'Keep this tab open. The voice wizard opens when it is done.';
    const track = document.createElement('div');
    track.className = 'vl-progress';
    const fill = document.createElement('div');
    track.append(fill);
    const stats = document.createElement('p');
    stats.className = 'vl-progress-stats';
    const actions = document.createElement('div');
    actions.className = 'vl-actions';
    const hide = document.createElement('button');
    hide.className = 'vl-btn vl-btn-cancel';
    hide.textContent = 'Continue in background';
    actions.append(hide);
    dialog.append(title, body, track, stats, actions);
    overlay.append(dialog);
    document.body.append(overlay);
    hide.focus();
    cloningDialogUpdate = () => {
      fill.style.width = `${cloning.total ? (cloning.loaded / cloning.total) * 100 : 0}%`;
      stats.textContent = cloning.total ? `${formatBytes(cloning.loaded)} of ${formatBytes(cloning.total)}` : 'Starting…';
    };
    cloningDialogUpdate();
    let open = true;
    const close = (result) => {
      if (!open) return;
      open = false;
      cloningDialogUpdate = null;
      overlay.remove();
      resolve(result);
    };
    hide.addEventListener('click', () => close(false));
    downloadCloning().then(() => {
      if (open) close(true);
      else toast('Voice cloning is ready', { type: 'success' });
    }, (e) => {
      close(false);
      toast(`The voice cloning download failed: ${e.message}`, { type: 'error' });
    });
  });
}

async function requestEncode(pcmF32) {
  if (!(await confirmEncoderDownload())) {
    throw Object.assign(new Error('The voice encoder was not downloaded'), { code: 'encoder-declined' });
  }
  setBusy('voice', true);
  // phones: after any transcription (Whisper's worker ends with it, see requestTranscript)
  if (LOW_MEMORY) return encodeAlone(pcmF32).finally(() => setBusy('voice', false));
  setActivity('Analyzing voice');
  showProgress('indeterminate');
  return new Promise((resolve, reject) => {
    if (!ttsWorker) { reject(new Error('Engine not running')); return; }
    const requestId = nextEncodeId++;
    pendingEncodes.set(requestId, { resolve, reject });
    const refAudio = new Float32Array(pcmF32); // copy — the stored record keeps its buffer
    ttsWorker.postMessage({ type: 'encode-reference', requestId, refAudio }, [refAudio.buffer]);
  }).finally(() => {
    setBusy('voice', false);
    if (!isGenerating) hideProgress();
  });
}

// Phones: the voice encoder in a worker of its own, ended right after, while the speech model's
// worker is stopped (parkSpeech, encoder-worker.js)
function encodeAlone(pcmF32) {
  return onGpu(async () => {
    await parkSpeech();
    const base = await resolveModelBaseUrl();
    setStatus('Analyzing voice…');
    setActivity('Analyzing voice');
    showProgress('indeterminate');
    const w = new Worker('workers/encoder-worker.js?v=1', { type: 'module' });
    try {
      return await new Promise((resolve, reject) => {
        w.onmessage = ({ data: m }) => {
          if (m.type === 'progress') onEncoderDownload(m);
          else if (m.type === 'encoded') resolve(m);
          else if (m.type === 'error') reject(Object.assign(new Error(m.message), { code: m.code }));
        };
        w.onerror = (e) => reject(new Error(e.message || 'The voice encoder could not start'));
        const pcm = new Float32Array(pcmF32); // copy — the stored record keeps its buffer
        w.postMessage({ type: 'encode', id: 1, pcm, base }, [pcm.buffer]);
      });
    } catch (e) {
      if (e.code === 'models-gone') modelsGone();
      throw e;
    } finally {
      w.terminate();
      if (!isGenerating) { setStatus('Ready'); hideProgress(); }
    }
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
  if (!canEncodeSilently()) return; // background work never asks for the download
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
    if (voiceLocked) toast('Voice lock cleared: settings changed');
    voiceLocked = false;
    sessionChainRef = null;
    updateVoiceLockBtn();
  }
}

initToggleRow(genderRow, { label: 'Gender', onChange: onDesignChange });
initToggleRow(pitchRow, { label: 'Pitch', onChange: onDesignChange });
initToggleRow(qualityRow, { label: 'Quality', onChange: (val) => { if (!isGenerating) qualityEl.value = val; } });

// Speed: the worker stretches the finished audio (pitch unchanged). The model
// itself always runs at its natural pace: giving it more time per sentence
// (OmniVoice's speed < 1) changed cloned voices' pitch and timbre.
// The slider runs -100..100 so 1.00× sits in the middle: the left half covers
// SPEED_MIN..1, the right half 1..SPEED_MAX.
const SPEED_KEY = 'vocoloco-tempo';
const SPEED_MIN = 0.85, SPEED_MAX = 1.1;

function positionToSpeed(p) {
  const f = Math.max(-1, Math.min(1, p / 100));
  return Math.round((1 + f * (f < 0 ? 1 - SPEED_MIN : SPEED_MAX - 1)) * 100) / 100;
}

function speedToPosition(v) {
  return Math.round(100 * (v - 1) / (v < 1 ? 1 - SPEED_MIN : SPEED_MAX - 1));
}

function currentSpeed() {
  const p = speedEl ? parseFloat(speedEl.value) : NaN;
  return Number.isFinite(p) ? positionToSpeed(p) : 1;
}

function showSpeed() {
  const label = `${currentSpeed().toFixed(2)}×`;
  if (speedValue) speedValue.textContent = label;
  if (speedEl) speedEl.setAttribute('aria-valuetext', label);
}

if (speedEl) {
  const saved = parseFloat(localStorage.getItem(SPEED_KEY));
  if (saved >= SPEED_MIN && saved <= SPEED_MAX) speedEl.value = String(speedToPosition(saved));
  showSpeed();
  speedEl.addEventListener('input', () => {
    showSpeed();
    localStorage.setItem(SPEED_KEY, String(currentSpeed()));
  });
  // Double-click: back to 1.00×
  speedEl.addEventListener('dblclick', () => {
    if (speedEl.disabled) return;
    speedEl.value = '0';
    speedEl.dispatchEvent(new Event('input'));
  });
}

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

// ─── Pause buttons: [pause], [pause 2s] at the cursor (or over the selection) ──

function insertTag(tag) {
  if (textEl.disabled) return;
  const start = textEl.selectionStart ?? textEl.value.length, end = textEl.selectionEnd ?? start;
  const before = textEl.value.slice(0, start), after = textEl.value.slice(end);
  const text = (before && !/\s$/.test(before) ? ' ' : '') + tag + (after && !/^\s/.test(after) ? ' ' : '');
  if (before.length + text.length + after.length > MAX_TEXT_LEN) { toast('The text is at its maximum length.', { type: 'error' }); return; }
  textEl.focus();
  textEl.setRangeText(text, start, end, 'end');
  textEl.dispatchEvent(new Event('input'));
}
const tagBar = $('tag-bar');
// pointerdown would move the focus (and on phones the cursor) out of the text box
tagBar.addEventListener('pointerdown', (e) => { if (e.target.closest('button')) e.preventDefault(); });
tagBar.addEventListener('click', (e) => { const b = e.target.closest('button[data-tag]'); if (b) insertTag(b.dataset.tag); });

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
      ? 'Voice locked: the next generations reuse this voice'
      : 'Voice lock off: each generation gets a fresh voice');
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
// The Studio's bar, and the activity pill in the top bar, seen from every tab (on a phone the
// Studio is out of sight while a voice is tested): what runs (setActivity) and how far.

const activityEl = $('activity');
const activityText = $('activity-text');
let activityLabel = 'Working';
let activityPct = null; // null: no measure yet
let busyBtn = null; // the button whose job runs (a voice's Test): shows the same

function renderActivity() {
  const text = activityPct == null ? `${activityLabel}…` : `${activityLabel} ${Math.floor(activityPct)}%`;
  activityText.textContent = text;
  activityEl.style.setProperty('--pct', `${activityPct ?? 0}%`);
  if (busyBtn) {
    busyBtn.textContent = activityPct == null ? `${activityLabel.split(' ')[0]}…` : `${Math.floor(activityPct)}%`; // a narrow button
    busyBtn.style.setProperty('--pct', `${activityPct ?? 0}%`);
  }
}

function setActivity(label) {
  activityLabel = label;
  activityPct = null;
  renderActivity();
}

function showProgress(mode) {
  progressBar.classList.remove('hidden');
  if (mode === 'indeterminate') {
    progressBar.classList.add('indeterminate');
    progressBar.style.width = '';
    activityPct = null;
  } else {
    progressBar.classList.remove('indeterminate');
  }
  activityEl.classList.remove('hidden');
  renderActivity();
}

function setProgressPercent(pct) {
  progressBar.classList.remove('hidden');
  progressBar.classList.remove('indeterminate');
  progressBar.style.width = Math.min(100, pct) + '%';
  activityPct = Math.min(100, pct);
  activityEl.classList.remove('hidden');
  renderActivity();
}

function hideProgress() {
  progressBar.classList.add('hidden');
  progressBar.classList.remove('indeterminate');
  progressBar.style.width = '0%';
  activityEl.classList.add('hidden');
}

// ─── Loading overlay / first-run experience ─────────────────────────────────

let rateSamples = [];
let lastLoadingTextUpdate = 0;

// ─── Download consent ───────────────────────────────────────────────────────
// The worker waits for 'start-download' before fetching anything (people on
// mobile data or short on storage may not want gigabytes right now).

let downloadPlan = null;

function mobileDataNote() {
  const c = navigator.connection; // Chromium only; absent elsewhere
  return c && (c.type === 'cellular' || c.saveData) ? ' Mobile data: Wi-Fi recommended.' : '';
}

function downloadSizeText(plan) {
  if (plan.totalBytes) return formatBytes(plan.totalBytes - plan.cachedBytes);
  const onnx = plan.backend ? plan.backend === 'onnx' : !navigator.gpu; // a first visit's estimate (worker init)
  return onnx ? 'about 2.4 GB' : 'about 1.2 GB';
}

// The overlay covers the text box; longer messages (narrow screens) grow the box
function fitLoadingOverlay() {
  if (!loadingOverlay || !loadingOverlay.isConnected) return;
  const box = loadingOverlay.parentElement;
  box.dataset.baseMinHeight ??= box.style.minHeight; // the inline 180px
  const cs = getComputedStyle(loadingOverlay);
  const kids = [...loadingOverlay.children].filter(el => getComputedStyle(el).display !== 'none');
  const need = kids.reduce((s, el) => s + el.offsetHeight, 0)
    + (parseFloat(cs.rowGap) || 0) * Math.max(0, kids.length - 1)
    + parseFloat(cs.paddingTop) + parseFloat(cs.paddingBottom);
  box.style.minHeight = box.dataset.baseMinHeight;
  if (need > box.clientHeight) box.style.minHeight = `${Math.ceil(need)}px`;
}
window.addEventListener('resize', fitLoadingOverlay);

function askModelDownload(plan, later = false) {
  const size = downloadSizeText(plan);
  if (!loadingOverlay || !loadingOverlay.isConnected) {
    // Engine restarted after the overlay was gone: ask in a dialog instead
    confirmDialog({
      title: 'Download voice model?',
      body: `One-time download, stored on this device.${mobileDataNote()}`,
      confirmLabel: `Download ${size}`,
      cancelLabel: 'Not now',
    }).then((ok) => {
      if (ok) startModelDownload();
      else setStatus('Voice model not downloaded');
    });
    return;
  }
  const spinner = $('loading-spinner');
  if (spinner) spinner.style.display = 'none';
  loadingTrack.classList.add('hidden');
  loadingStats.textContent = '';
  if (later) {
    loadingTitle.textContent = 'Voice model not downloaded';
    loadingSub.textContent = '';
  } else if (plan.cachedBytes > 0) {
    loadingTitle.textContent = 'Finish download?';
    loadingSub.textContent = `${size} left.`;
  } else if (offerCloning) {
    loadingTitle.textContent = 'Download the models?';
    loadingSub.textContent = 'One-time download, stored on this device. Voice cloning adds speech recognition and the voice encoder.';
  } else {
    loadingTitle.textContent = 'Download voice model?';
    loadingSub.textContent = 'One-time download, stored on this device.';
  }
  loadingSub.textContent = (loadingSub.textContent + mobileDataNote()).trim();
  // with cloning: everything at once, or the speech model alone
  const speechBytes = plan.totalBytes ? plan.totalBytes - plan.cachedBytes : 1.2e9;
  downloadStartBtn.textContent = offerCloning ? `Download all · ${plan.totalBytes ? '' : 'about '}${formatBytes(speechBytes + cloningLeft())}` : `Download ${size}`;
  downloadSpeechOnlyBtn.textContent = `Speech only · ${size}`;
  downloadSpeechOnlyBtn.classList.toggle('hidden', !offerCloning);
  downloadLaterBtn.classList.toggle('hidden', later);
  loadingActions.classList.remove('hidden');
  hideProgress();
  setStatus(later ? 'Voice model not downloaded' : 'Download needed');
  fitLoadingOverlay();
}

// Downloaded models are best-effort storage unless the site asks to keep them, and browsers
// clear that when a device runs low on space (an iPhone did, right after a 0.9 GB download).
// Chrome and Safari decide without asking; Firefox asks once.
async function keepStorage() {
  try {
    if (!navigator.storage?.persist || await navigator.storage.persisted()) return;
    if (!await navigator.storage.persist()) console.warn('[storage] not persistent: the browser may clear the models when space runs low');
  } catch { /* not supported */ }
}

// The browser deleted the models while the app was open (a read found the file gone): say so,
// and start the engine again, which asks before downloading
let modelsGoneAsked = false;
function modelsGone() {
  if (modelsGoneAsked) return;
  modelsGoneAsked = true;
  confirmDialog({
    title: 'The models were deleted',
    body: 'The browser removed the downloaded models from this device, which browsers do when storage runs low. Download them again?',
    confirmLabel: 'Download again', cancelLabel: 'Later',
  }).then((ok) => {
    modelsGoneAsked = false;
    if (!ok) return;
    rejectPendingEncodes('Engine restarted');
    try { ttsWorker.terminate(); } catch { /* already dead */ }
    isReady = false;
    initWorker();
  });
}

// withCloning: the voice cloning models too, as one download (see offerCloning)
function startModelDownload(withCloning = false) {
  keepStorage();
  const both = withCloning && offerCloning;
  if (loadingOverlay && loadingOverlay.isConnected) {
    loadingActions.classList.add('hidden');
    const spinner = $('loading-spinner');
    if (spinner) spinner.style.display = '';
    loadingTitle.textContent = downloadPlan.cachedBytes > 0 ? 'Resuming download' : both ? 'Downloading the models' : 'Downloading voice model';
    loadingSub.textContent = 'Keep this tab open.';
    if (downloadPlan.totalBytes) loadingTrack.classList.remove('hidden');
    fitLoadingOverlay();
  }
  setActivity('Downloading models');
  showProgress('indeterminate');
  setStatus('Downloading…');
  if (both) {
    combinedDownload = { loaded: downloadPlan.cachedBytes || 0, total: downloadPlan.totalBytes || 0 };
    downloadCloning().catch((e) => toast(`The voice cloning download failed: ${e.message}`, { type: 'error' }));
  } else if (offerCloning) localStorage.setItem(CLONING_KEY, '0');
  ttsWorker.postMessage({ type: 'start-download' });
}

downloadStartBtn?.addEventListener('click', () => startModelDownload(true));
downloadSpeechOnlyBtn?.addEventListener('click', () => startModelDownload(false));
downloadLaterBtn?.addEventListener('click', () => askModelDownload(downloadPlan, true));

// The first visit's question offers voice cloning too (when the engine runs and it isn't stored)
let offerCloning = false;

async function onDownloadPlan(msg) {
  rateSamples = [];
  downloadPlan = msg;
  if (msg.filesToDownload > 0) {
    offerCloning = false;
    if (msg.backend === 'engine') {
      try { await checkCloning(); offerCloning = !cloningComplete(); } catch { /* no size: speech only */ }
    }
    askModelDownload(msg);
  } else if (loadingOverlay && loadingOverlay.isConnected) {
    loadingTitle.textContent = 'Loading VocoLoco…';
    loadingSub.textContent = '';
  }
}

// Time left as 45s, 4m 10s or 1h 5m
function formatDuration(s) {
  s = Math.max(1, Math.round(s));
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60);
  return h ? `${h}h ${m}m` : m ? `${m}m ${s % 60}s` : `${s}s`;
}

// First visit with voice cloning chosen: the speech model's download and the cloning one show as
// one, and the app opens once both are done
let combinedDownload = null; // { loaded, total } of the speech model's part

function showCombinedProgress() {
  const c = combinedDownload;
  renderDownloadProgress({ loadedBytes: c.loaded + cloning.loaded, totalBytes: c.total && cloning.total ? c.total + cloning.total : null });
}

function onDownloadProgress(msg) {
  if (!combinedDownload) { renderDownloadProgress(msg); return; }
  Object.assign(combinedDownload, { loaded: msg.loadedBytes ?? 0, total: msg.totalBytes ?? 0 });
  showCombinedProgress();
}

function renderDownloadProgress(msg) {
  if (msg.loadedBytes != null && msg.totalBytes) {
    const pct = (msg.loadedBytes / msg.totalBytes) * 100;
    setProgressPercent(pct);
    if (loadingFill && loadingFill.isConnected) {
      loadingTrack.classList.remove('hidden'); // a first visit's consent had no size yet
      loadingFill.style.width = pct.toFixed(1) + '%';
    }
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
          suffix += ` · ${formatDuration(etaS)}`;
        }
      }
      const line = `${formatBytes(msg.loadedBytes)} of ${formatBytes(msg.totalBytes)}${suffix}`;
      if (loadingStats && loadingStats.isConnected) loadingStats.textContent = line;
      setStatus(`Downloading models: ${line}`);
    }
  } else {
    showProgress('indeterminate');
    const now = performance.now();
    if (now - lastLoadingTextUpdate > 2000) {
      lastLoadingTextUpdate = now;
      const line = msg.fileIndex
        ? `File ${msg.fileIndex}/${msg.fileCount}: ${msg.detail || 'downloading…'}`
        : (msg.detail || 'Downloading models…');
      if (loadingStats && loadingStats.isConnected) loadingStats.textContent = line;
      setStatus(line);
    }
  }
}

function onLoadingPhase(msg) {
  if (combinedDownload && cloning.run) return; // the box keeps showing the download (see 'ready')
  setActivity('Loading the model');
  showProgress('indeterminate');
  const warmingUp = ['session-main', 'session-decoder', 'session-encoder', 'warmup'].includes(msg.phase);
  if (warmingUp) {
    setStatus('Warming up the model…');
    if (loadingOverlay && loadingOverlay.isConnected) {
      loadingTitle.textContent = 'Warming up the model…';
      loadingSub.textContent = 'First start after a download takes a little longer while GPU shaders compile (~10–30 s).';
      loadingTrack.classList.add('hidden');
      loadingStats.textContent = '';
      fitLoadingOverlay();
    }
  } else {
    setStatus(msg.detail || 'Loading…');
    if (loadingOverlay && loadingOverlay.isConnected) loadingStats.textContent = msg.detail || '';
  }
}

// The voice encoder downloads on first use (cloning), not with the main model
function onEncoderDownload(msg) {
  if (msg.done) encoderCached = true;
  if (wiz.step === 'saving') {
    if (msg.done) setWizSavingStatus('Analyzing voice… (one-time setup)');
    else setWizSavingStatus('Downloading the voice encoder', msg.totalBytes
      ? Math.min(100, Math.floor((msg.loadedBytes / msg.totalBytes) * 100)) + '%'
      : Math.round(msg.loadedBytes / 1e6) + ' MB');
  }
  if (msg.done) setActivity('Analyzing voice');
  else if (msg.totalBytes) { activityLabel = 'Downloading encoder'; setProgressPercent((msg.loadedBytes / msg.totalBytes) * 100); }
  if (!isGenerating) setStatus(msg.done ? 'Ready' : msg.detail);
}

// ─── Ready handler ──────────────────────────────────────────────────────────

function enableUI() {
  isReady = true;
  statusEl.classList.remove('vl-sr-only'); // shown in the loading box until now
  if (!isGenerating) { // else a generation waited for the model (resumeSpeech) and goes on now
    setStatus('Ready');
    hideProgress();
  }
  if (loadingOverlay && loadingOverlay.isConnected) {
    loadingOverlay.style.transition = 'opacity 0.3s';
    loadingOverlay.style.opacity = '0';
    setTimeout(() => {
      const box = loadingOverlay.parentElement;
      if (box.dataset.baseMinHeight != null) box.style.minHeight = box.dataset.baseMinHeight;
      loadingOverlay.remove();
    }, 300);
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
  const isGpu = backend === 'webgpu' || backend === 'engine';
  backendIsCpu = !isGpu;
  const badge = $('backend-badge');
  if (badge) {
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

// ─── Memory check ───────────────────────────────────────────────────────────
// The app holds up to MEMORY_NEED_MB of GPU memory at once on a phone (a generation with a long
// reference: 1.20 GB measured; Whisper takes ~0.35 GB), and iPhones kill a tab at ~1.5 GB. Before the
// first download the worker checks that it can hold that much (tts-worker.js, probeMemory).
// Where it can't, iOS kills the tab instead of saying so: the check is noted as running first,
// and a page that finds the note on load knows how the last one ended.
const MEMORY_NEED_MB = 1280;
const MEMORY_KEY = 'vocoloco-memory'; // { running | passed | failed: MB }
// Set while a model loads or works, cleared when done or when the page is left: still set on
// the next load means the system closed the tab mid-way, almost always for memory
const BUSY_KEY = 'vocoloco-busy';
const ASR_OFF_KEY = 'vocoloco-asr-off-2'; // Whisper got the tab killed here: no auto-transcription (-2: the engine's)

let memoryCheck = {};
try { memoryCheck = JSON.parse(localStorage.getItem(MEMORY_KEY)) || {}; } catch { /* none yet */ }
if (memoryCheck.running) saveMemoryCheck({ failed: memoryCheck.running }); // the tab died during the check
const busy = new Set(localStorage.getItem(BUSY_KEY)?.split(',') ?? []);
const closedWhile = [...busy];
busy.clear();
localStorage.removeItem(BUSY_KEY);

function saveMemoryCheck(state) {
  memoryCheck = state;
  localStorage.setItem(MEMORY_KEY, JSON.stringify(state));
}

function setBusy(what, on) {
  if (on) busy.add(what); else busy.delete(what);
  if (busy.size) localStorage.setItem(BUSY_KEY, [...busy].join(','));
  else localStorage.removeItem(BUSY_KEY);
}
// A page that is left (reload, close) wasn't killed: neither marker may outlive it
window.addEventListener('pagehide', () => {
  localStorage.removeItem(BUSY_KEY);
  if (memoryCheck.running) localStorage.removeItem(MEMORY_KEY);
});
window.addEventListener('pageshow', () => {
  if (busy.size) localStorage.setItem(BUSY_KEY, [...busy].join(','));
  if (memoryCheck.running) saveMemoryCheck(memoryCheck);
});

// The last page was closed by the system: say what it was doing, and don't run into it again
if (closedWhile.length) {
  const doing = { load: 'loading the voice model', generate: 'generating speech', voice: 'analyzing a voice', transcribe: 'transcribing a recording' };
  const what = closedWhile.map((k) => doing[k]).filter(Boolean).join(' and ') || 'working';
  if (closedWhile.includes('load')) saveMemoryCheck({ failed: MEMORY_NEED_MB });
  if (closedWhile.includes('transcribe') && LOW_MEMORY) localStorage.setItem(ASR_OFF_KEY, '1');
  const hint = closedWhile.includes('transcribe') && LOW_MEMORY
    ? ' Automatic transcription is off on this device now: type what the recording says.'
    : closedWhile.includes('load') ? '' : ' Close other apps and tabs; shorter texts and recordings need less memory.';
  confirmDialog({
    title: 'The browser closed VocoLoco',
    body: `It was ${what} when this device ran out of memory.${hint}`,
    confirmLabel: 'OK', cancelLabel: null,
  });
}

// Shown instead of the model when this device can't hold it (see above)
function showMemoryFailure() {
  showLoadFailure('Not enough memory',
    `The voice model needs about ${formatBytes(MEMORY_NEED_MB * 1048576)} of memory, and this device can't spare that right now. Close other apps and browser tabs, then check again.`,
    { label: 'Check again', run: () => { saveMemoryCheck({}); location.reload(); } });
}

// The loading overlay (or a dialog once it's gone) explains why there is no model
function showLoadFailure(title, text, action = null) {
  setStatus(title);
  hideProgress();
  if (!loadingOverlay || !loadingOverlay.isConnected) {
    confirmDialog({ title, body: text, confirmLabel: action?.label ?? 'OK', cancelLabel: action ? 'Close' : null })
      .then((ok) => { if (ok) action?.run(); });
    return;
  }
  loadingTitle.textContent = title;
  loadingSub.textContent = text;
  const spinner = $('loading-spinner');
  if (spinner) spinner.style.display = 'none';
  loadingTrack.classList.add('hidden');
  loadingStats.textContent = '';
  downloadStartBtn.classList.add('hidden');
  downloadLaterBtn.classList.add('hidden');
  loadRetryBtn.classList.toggle('hidden', !action);
  loadRetryBtn.textContent = action?.label ?? '';
  loadRetryBtn.onclick = action?.run ?? null;
  loadingActions.classList.toggle('hidden', !action);
  fitLoadingOverlay();
}

// ─── Worker ─────────────────────────────────────────────────────────────────

let pendingJob = null; // { jobId, resolve }
let workerAnswered = false; // any message from the current worker yet (initWorker)

function resolveJob(result) {
  const p = pendingJob;
  pendingJob = null;
  if (p) p.resolve(result);
}

function sendJob(msg) {
  setBusy('generate', true);
  const run = async () => {
    if (!await resumeSpeech()) return { kind: 'error', message: 'The voice model could not start again.' };
    return new Promise((resolve) => {
      pendingJob = { jobId: msg.jobId, resolve };
      const transfers = [];
      if (msg.refAudio) transfers.push(msg.refAudio.buffer);
      if (msg.refTokens) transfers.push(msg.refTokens.buffer);
      ttsWorker.postMessage(msg, transfers);
    });
  };
  return (LOW_MEMORY ? onGpu(run) : run()).finally(() => setBusy('generate', false));
}

function initWorker() {
  rejectPendingEncodes('Engine restarted');
  resolveJob({ kind: 'cancelled' });
  speechParked = null; // a new worker either way
  ttsWorker = new Worker('workers/tts-worker.js?v=11', { type: 'module' });
  workerAnswered = false;
  ttsWorker.onerror = (e) => {
    console.error('Worker error:', e);
    setBusy('load', false);
    if (onSpeechReady) { isReady = false; onSpeechReady(false); onSpeechReady = null; }
    showLoadFailure('Something went wrong', e.message || 'The speech engine could not start. Reload the page to try again.');
  };
  ttsWorker.onmessage = (e) => {
    const msg = e.data;
    workerAnswered = true;
    switch (msg.type) {
      case 'plan':
        onDownloadPlan(msg);
        break;
      case 'progress':
        // Marked from here on (not during the download): the weights going into memory is
        // where a phone that can't hold them gets the tab killed
        if (msg.phase === 'session-main') setBusy('load', true);
        if (msg.stage === 'downloading') onDownloadProgress(msg);
        else if (msg.stage === 'loading') onLoadingPhase(msg);
        else if (msg.stage === 'generating') onGenProgress(msg);
        else if (msg.stage === 'warning') toast(msg.detail, { type: 'error', duration: 6000 });
        else if (msg.stage === 'encoder-download') onEncoderDownload(msg);
        else setStatus(msg.detail || '');
        break;
      case 'memory-check':
        if (msg.state === 'start') { // noted before the worker allocates anything
          saveMemoryCheck({ running: MEMORY_NEED_MB });
          ttsWorker.postMessage({ type: 'memory-check-go' });
          break;
        }
        saveMemoryCheck(msg.ok ? { passed: MEMORY_NEED_MB } : { failed: MEMORY_NEED_MB });
        if (!msg.ok) { setBusy('load', false); showMemoryFailure(); }
        break;
      case 'ready': {
        setBusy('load', false);
        encoderAvailable = msg.encoderAvailable !== false;
        encoderCached ||= !!msg.encoderCached;
        encoderBytes = msg.encoderBytes || null;
        speechBackend = msg.backend;
        const ready = () => {
          onReady(msg.backend);
          onSpeechReady?.(true);
          onSpeechReady = null;
        };
        if (combinedDownload && cloning.run) {
          // the first visit's download goes on with voice cloning; the app opens after it
          if (loadingOverlay?.isConnected) loadingTitle.textContent = 'Downloading voice cloning';
          showCombinedProgress();
          cloning.run.catch(() => {}).finally(() => { combinedDownload = null; ready(); });
        } else {
          combinedDownload = null;
          ready();
          // chosen before but not finished (tab closed mid-way): goes on by itself
          if (msg.backend === 'engine' && localStorage.getItem(CLONING_KEY) === '1' && !cloning.run) {
            checkCloning().then(() => { if (!cloningComplete()) return downloadCloning(); }).catch(() => {});
          }
        }
        keepStorage(); // also for models stored before the app asked
        break;
      }
      case 'released':
        onModelReleased?.();
        onModelReleased = null;
        break;
      case 'encoder-status':
        encoderAvailable = msg.available;
        if (currentView === 'voices') renderVoicesView();
        break;
      case 'audio':
        resolveJob({ kind: 'audio', pcm: msg.pcm, sampleRate: msg.sampleRate, peak: msg.peak, tokens: msg.tokens || null, tokenCount: msg.tokenCount || 0, rateTokens: msg.rateTokens ?? null });
        break;
      case 'cancelled':
        resolveJob({ kind: 'cancelled' });
        break;
      case 'ref-info':
        persistRefInfo(msg).catch((err) => console.warn('Saving the reference info failed:', err.message));
        break;
      case 'encoded': {
        const p = pendingEncodes.get(msg.requestId);
        if (p) { pendingEncodes.delete(msg.requestId); p.resolve(msg); }
        break;
      }
      case 'encode-error': {
        const p = pendingEncodes.get(msg.requestId);
        if (p) { pendingEncodes.delete(msg.requestId); p.reject(Object.assign(new Error(msg.message), { code: msg.code })); }
        if (msg.code === 'models-gone') modelsGone();
        break;
      }
      case 'error':
        if (msg.code === 'models-gone') modelsGone();
        if (msg.code === 'gpu-lost') {
          console.warn(msg.message);
          rejectPendingEncodes(msg.message);
          resolveJob({ kind: 'error', message: msg.message });
          try { ttsWorker.terminate(); } catch { /* already dead */ }
          isReady = false;
          toast('The GPU stopped. Restarting the engine…', { type: 'error' });
          initWorker();
          break;
        }
        if (pendingJob) {
          resolveJob({ kind: 'error', message: msg.message });
        } else {
          console.error(msg.message);
          setBusy('load', false);
          if (onSpeechReady) { isReady = false; onSpeechReady(false); onSpeechReady = null; }
          showLoadFailure(msg.code === 'no-engine' ? "This browser isn't supported" : 'Something went wrong', msg.message);
        }
        break;
    }
  };
  // A device that couldn't hold the model gets no download (see showMemoryFailure)
  if (memoryCheck.failed >= MEMORY_NEED_MB) { showMemoryFailure(); return; }
  const worker = ttsWorker;
  resolveModelBaseUrl().then((modelBaseUrl) => {
    const check = !(memoryCheck.passed >= MEMORY_NEED_MB);
    worker.postMessage({
      type: 'init', modelBaseUrl, forceCPU: location.search.includes('cpu'), confirmDownload: true,
      lowMemory: LOW_MEMORY, probeBytes: check ? MEMORY_NEED_MB * 1048576 : 0,
    });
    // The worker answers init right away; silence means its scripts never finished loading
    setTimeout(() => {
      if (ttsWorker !== worker || workerAnswered) return;
      showLoadFailure("VocoLoco couldn't start", "The speech engine's scripts didn't finish loading. Check your connection, then reload the page.",
        { label: 'Reload', run: () => location.reload() });
    }, 45e3);
  });
  setActivity('Loading the model');
  showProgress('indeterminate');
}

// Phones and tablets (LOW_MEMORY): one big model on the GPU at a time. WebKit gives a worker's GPU
// memory back only when the worker ends (freed buffers stay held while the bind groups that used
// them wait for garbage collection), so models don't unload, their workers end: the speech
// model's while Whisper or the voice encoder runs (encoder-worker.js), and it starts again for
// the next generation. These phases take turns (onGpu).
let gpuTurn = Promise.resolve();
function onGpu(fn) {
  const p = gpuTurn.then(fn);
  gpuTurn = p.catch(() => {});
  return p;
}

let speechParked = null; // Promise: the speech model's worker stopping or stopped (initWorker clears it)
let onModelReleased = null;
function parkSpeech() {
  if (!LOW_MEMORY || !isReady) return Promise.resolve();
  speechParked ??= new Promise((resolve) => {
    // the worker answers once its current job is done
    onModelReleased = () => {
      try { ttsWorker.terminate(); } catch { /* already dead */ }
      resolve();
    };
    ttsWorker.postMessage({ type: 'release-model' });
  });
  return speechParked;
}

// The speech model's worker again if it was stopped; false if it couldn't start
let onSpeechReady = null;
async function resumeSpeech() {
  if (!speechParked) return true;
  await speechParked;
  return new Promise((resolve) => { initWorker(); onSpeechReady = resolve; });
}

let modelBaseUrlPromise = null;

function resolveModelBaseUrl() {
  if (!modelBaseUrlPromise) {
    modelBaseUrlPromise = (async () => {
      // Anywhere but the official site (local development, self-hosting): a complete models/
      // mirror next to the app (scripts/fetch-models.sh) is used instead of Hugging Face
      if (location.hostname.endsWith('.github.io')) return MODEL_BASE_URL;
      let local;
      try {
        // The fetch script writes the manifest last, so its presence means the mirror is complete
        local = (await fetch(`${LOCAL_MODEL_URL}/omnivoice-main-manifest.json`, { method: 'HEAD' })).ok;
        localStorage.setItem('vocoloco-local-mirror', local ? '1' : '');
      } catch { // offline: as last time (the models stored come from there)
        local = localStorage.getItem('vocoloco-local-mirror') === '1';
      }
      return local ? LOCAL_MODEL_URL : MODEL_BASE_URL;
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
  const comment = 'AI-generated synthetic speech. This audio was produced entirely by an artificial intelligence text-to-speech model. EU AI Act Art. 50: not a recording of a human voice.';

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

// Encoding blocks the page for a moment (seconds for long audio on a phone): the button says so
// first, after one painted frame (still within the tap, which the download needs)
async function saveMp3(btn, pcm, sampleRate, filename) {
  const label = btn.textContent;
  btn.textContent = 'Saving…';
  btn.disabled = true;
  await new Promise((r) => requestAnimationFrame(() => setTimeout(r, 0)));
  try {
    downloadMp3(pcm, sampleRate, filename);
  } finally {
    btn.textContent = label;
    btn.disabled = false;
  }
}

downloadBtn.addEventListener('click', () => {
  if (lastPcm) saveMp3(downloadBtn, lastPcm, lastSampleRate, mp3Filename(lastText));
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
      saveMp3(info.querySelector('[data-action="download"]'), item.pcm, item.sampleRate, mp3Filename(item.text, item.timestamp));
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
  // Resample no more than wizSetSource keeps (+1 s so it can tell the file was cut)
  const numSamples = Math.round(Math.min(buf.duration, MAX_SOURCE_S + 1) * 24000);
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
  if (speedEl) speedEl.disabled = locked;
  for (const el of [genderRow, pitchRow, qualityRow, speedRow, studioVoicePicker]) {
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

  // A cloned voice's own pace sizes the parts (text-chunker.js)
  const paceVoice = selectedSavedVoice?.tokens ? selectedSavedVoice : null;
  const chunks = chunkText(text, paceVoice ? { refText: paceVoice.refText, refTokens: paceVoice.tokens.length / 8 } : {});
  if (chunks.length === 0) {
    toast('Nothing to speak: the text contains no readable characters.', { type: 'error' });
    return;
  }

  player.getAudioCtx(); // unlock audio within the user gesture

  isGenerating = true;
  setGenerating(true);
  lockVoiceControls(true);
  player.dimPrevious();
  setStatus('Preparing…');
  setActivity('Generating');
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
    tempo: currentSpeed(),
    voiceName: voice ? voice.name : null,
    hardKillTimer: null,
    lastStepMs: null,
    emaMsPerToken: null,
  };


  // Resolve the voice reference ONCE for the whole stream
  const ref = { refTokens: null, refText: null, refRateTokens: null, refInfo: null, instruct: null };
  try {
    if (voice) {
      ref.refText = voice.refText;
      // Older reference prep: re-analyze only if the encoder is already here
      if (voice.tokens && (!needsRefPrep(voice) || !canEncodeSilently())) {
        ref.refTokens = voice.tokens;
      } else if (encoderAvailable) {
        setStatus('Encoding reference voice… (one-time per voice)');
        await encodeVoiceShared(voice); // shares any encode already in flight
        ref.refTokens = voice.tokens;
        if (currentView === 'voices') renderVoicesView();
      } else if (voice.tokens) {
        ref.refTokens = voice.tokens; // older reference prep, but no encoder on this device
      } else {
        toast(`"${voice.name}" isn't set up on this device (voice encoder unavailable). Generating with a default voice instead.`, { type: 'error', duration: 6000 });
        ref.refText = null;
      }
      if (ref.refTokens) ref.refInfo = voice.refInfo ?? null; // the worker checks its key
    } else if (voiceLocked && sessionChainRef) {
      ref.refTokens = sessionChainRef.tokens;
      ref.refText = sessionChainRef.text;
      ref.refRateTokens = sessionChainRef.rateTokens ?? null;
    } else {
      ref.instruct = buildInstruct();
    }
  } catch (err) {
    toast(err.code === 'encoder-declined'
      ? `"${voice.name}" needs the voice encoder download first.`
      : 'Could not prepare the voice: ' + err.message, { type: 'error' });
    finishStream();
    return;
  }
  if (!stream || stream.cancelRequested) { finishStream(); return; }

  if (backendIsCpu && !cpuHintShown) {
    cpuHintShown = true;
    toast('CPU mode is much slower. The first audio can take several minutes.', { duration: 8000 });
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
    tempo: s.tempo,
    seed: null,
    instruct: null,
    normalize: false,
    // Natural silence kept around the speech: more at the start and end of the whole
    // text (100 ms there sounded cut off), joins get joinPauseSeconds on top
    leadMs: i === 0 ? EDGE_LEAD_MS : 100,
    trailMs: i === s.chunks.length - 1 ? EDGE_TRAIL_MS : 100,
  };
  if (ref.refTokens) {
    // Cloned / locked voice: same reference for every chunk
    msg.refTokens = new Int32Array(ref.refTokens); // copy — cached buffer must survive transfer
    msg.refText = ref.refText;
    msg.refRateTokens = ref.refRateTokens; // locked voice: chunk 1 count without headroom
    msg.refInfo = ref.refInfo ?? null;
  } else if (i === 0) {
    // Design mode, first chunk: let the model pick a voice, capture its tokens
    msg.instruct = ref.instruct;
    msg.returnTokens = true;
  } else if (s.chainRef) {
    // Design mode, later chunks: chain chunk 1's voice
    msg.refTokens = new Int32Array(s.chainRef.tokens);
    msg.refText = s.chainRef.text;
    msg.refRateTokens = s.chainRef.rateTokens ?? null;
  } else {
    msg.instruct = ref.instruct;
  }
  // No <|denoise|> with a reference (OmniVoice turns it on by default): it lets
  // the model produce a cleaned-up recording instead of copying the reference's,
  // and in 7 of 10 seeds that came out as a different-sounding voice.
  if (msg.refTokens) msg.denoise = false;
  return msg;
}

// Silence between chunks. Pauses after a full stop inside generated chunks
// measure ~0.4-0.75 s (each chunk also keeps up to 100 ms of its own
// lead/tail); a chunk split mid-sentence only gets a comma-length pause.
// Drawn from a range: people don't pause the same every time.
function joinPauseSeconds(prevText) {
  const [lo, hi] = /[.!?…。！？]["'”’)\]]*$/.test(prevText.trim()) ? [0.3, 0.6] : [0.1, 0.25];
  return lo + Math.random() * (hi - lo);
}
const EDGE_LEAD_MS = 200, EDGE_TRAIL_MS = 400; // silence kept before / after the whole text

function onChunkAudio(i, res) {
  const s = stream;
  const pcm = res.pcm; // transferred — ours to mutate

  // Uniform loudness: gain derived from chunk 1's peak, clip-guarded per chunk
  if (s.gain == null) s.gain = res.peak > 1e-6 ? 0.5 / res.peak : 1;
  let g = s.gain;
  if (res.peak > 1e-6 && res.peak * g > 0.98) g = 0.98 / res.peak;
  if (g !== 1) for (let k = 0; k < pcm.length; k++) pcm[k] *= g;

  if (i === 0 && res.tokens) {
    // rateTokens: without the worker's headroom, for the following chunks' estimate
    s.chainRef = { tokens: res.tokens, tokenCount: res.tokenCount, rateTokens: res.rateTokens, text: s.chunks[0].text };
  }
  s.results.push({ text: s.chunks[i].text, pcm });
  s.doneTokens += s.chunks[i].estTokens;
}

function finishStream() {
  const s = stream;
  if (s && s.hardKillTimer) clearTimeout(s.hardKillTimer);

  if (s && s.results.length > 0) {
    const partial = s.cancelRequested && s.results.length < s.chunks.length;
    // Every part is generated before anything plays (streaming the parts as they came felt choppy).
    // [pause] tags (text-chunker.js) ask for exact silences, less what the parts keep at their edges.
    const silence = (sec) => new Float32Array(Math.round(Math.max(0, sec) * 24000));
    const gap = (c) => (c.pauseAfter != null ? c.pauseAfter - 0.2 : joinPauseSeconds(c.text));
    const parts = [];
    if (s.chunks[0].pauseBefore) parts.push(silence(s.chunks[0].pauseBefore - EDGE_LEAD_MS / 1000));
    s.results.forEach((r, i) => { if (i > 0) parts.push(silence(gap(s.chunks[i - 1]))); parts.push(r.pcm); });
    const lastChunk = s.chunks[s.chunks.length - 1];
    if (!partial && lastChunk.pauseAfter) parts.push(silence(lastChunk.pauseAfter - EDGE_TRAIL_MS / 1000));
    const combined = new Float32Array(parts.reduce((n, p) => n + p.length, 0));
    parts.reduce((o, p) => { combined.set(p, o); return o + p.length; }, 0);
    if (currentView !== 'studio') switchView('studio');
    player.undim();
    player.loadStatic(combined);
    if (!s.cancelRequested) player.play(0);
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
      ? `Stopped: kept ${s.results.length} of ${s.chunks.length} parts`
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
  activityLabel = 'Generating'; // also after the model was loaded for this generation
  if (!stream) {
    setStatus(`Generating step ${msg.step}/${msg.numStep}`);
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
    setStatus(`Generating part ${idx + 1} of ${s.chunks.length} · step ${msg.step}/${msg.numStep}${etaStr}`);
  } else {
    setStatus(`Generating step ${msg.step}/${msg.numStep}${etaStr}`);
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
  setStatus('Cancelled, restarting engine…');
  initWorker();
}

generateBtn.addEventListener('click', generate);

// Enter generates (Shift+Enter = newline, Ctrl/Cmd+Enter also generates).
// Touch screens: Return is a new line like in any app (phone keyboards have
// no Shift+Enter), the Generate button sits in thumb reach instead.
const TOUCH_KEYBOARD = matchMedia('(pointer: coarse)');
textEl.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.isComposing && (e.ctrlKey || e.metaKey || (!e.shiftKey && !TOUCH_KEYBOARD.matches))) {
    e.preventDefault();
    generate();
  }
});

// ─── Voice test generation (wizard + voice cards) ───────────────────────────

async function testVoice(voice, btn) {
  if (!isReady || isGenerating || !voice) return;
  player.getAudioCtx(); // unlock audio within the tap: iOS plays nothing that starts after the wait
  const text = TEST_SENTENCES[voice.lang] || TEST_SENTENCES.en;
  isGenerating = true;
  setGenerating(true);
  const orig = btn ? btn.textContent : null;
  if (btn) { btn.disabled = true; btn.classList.add('btn-busy'); busyBtn = btn; }
  setStatus('Testing voice…');
  setActivity('Starting');
  showProgress('indeterminate');

  const msg = {
    type: 'synthesize',
    jobId: 'test-' + Date.now(),
    text,
    lang: null,
    numStep: 16, // "Good": fewer steps often missed the voice
    ...GEN_PARAMS,
    speed: 1.0,
    tempo: currentSpeed(),
    instruct: null,
    seed: null,
    refText: voice.refText,
    normalize: true,
    denoise: false, // see buildChunkMsg
    leadMs: EDGE_LEAD_MS, trailMs: EDGE_TRAIL_MS,
  };
  // Phones encode in a worker of their own (encodeAlone): the speech model's worker has no room for it
  if (!voice.tokens && LOW_MEMORY && encoderAvailable) await encodeVoiceShared(voice).catch((e) => console.warn('Voice encode failed:', e.message));
  if (voice.tokens) { msg.refTokens = new Int32Array(voice.tokens); msg.refInfo = voice.refInfo ?? null; }
  else msg.refAudio = new Float32Array(voice.refAudio);

  const res = await sendJob(msg);

  isGenerating = false;
  setGenerating(false);
  busyBtn = null;
  hideProgress();
  if (btn) { btn.disabled = false; btn.classList.remove('btn-busy'); btn.style.removeProperty('--pct'); btn.textContent = orig; }
  if (res.kind === 'audio') {
    player.playOneShot(res.pcm, res.sampleRate);
    setStatus(`Voice test: ${(res.pcm.length / res.sampleRate).toFixed(1)}s`);
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
    if (nameSpan.parentNode !== header) return; // already renaming (a second tap)
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
  step: null,           // 'method' | 'record' | 'upload' | 'review' | 'saving' | 'done'
  method: null,
  source: null,         // 'recorded' | 'uploaded' | 'generated'
  lang: null,
  deviceId: null,
  stream: null, micCtx: null, srcNode: null, analyser: null, raf: 0,
  recorder: null, chunks: [], recStartTs: 0, recording: false, counting: false,
  stopLocked: false,    // recording, but shorter than MIN_REF_S
  note: null,           // why the last take was not used
  levelRing: new Float32Array(100),
  sourcePcm: null,      // the full recording / upload / generation
  sel: null,            // { start, end } in samples — the part used as the voice
  pcm: null, duration: 0, // the selected part
  presetTokens: null, presetTokenCount: 0, presetSel: null, // tokens only valid for presetSel
  savedVoiceId: null,
};

const MAX_SOURCE_S = 1800; // longer files: only the first 30 minutes are kept
const MIN_REF_S = 3;      // shortest usable reference; shorter ones don't carry the voice
const MAX_REF_S = 30;     // longest selectable reference: it runs through every generation step,
                          // so longer references make generation slower
const DEFAULT_REF_S = 15; // the automatic selection keeps the faster length

const trimEditor = wizReviewWave ? new TrimEditor({
  canvas: wizReviewWave,
  overviewCanvas: wizTrimOverview,
  sampleRate: 24000,
  minS: MIN_REF_S,
  maxS: MAX_REF_S,
  onChange: (sel, { final }) => onTrimChange(sel, final),
}) : null;

function fmtClock(s) {
  return `${Math.floor(s / 60)}:${(s % 60).toFixed(1).padStart(4, '0')}`;
}

// New source audio for the review step; the selection defaults to the best
// <= DEFAULT_REF_S of speech (or `presetSel`, e.g. chunk 1 of a generation;
// or `sel`, a default selection computed already).
function wizSetSource(pcm, presetSel = null, sel = null) {
  const cap = MAX_SOURCE_S * 24000;
  if (pcm.length > cap) {
    pcm = pcm.slice(0, cap);
    toast('This file is longer than 30 minutes. Only the first 30 minutes were loaded.');
  }
  wiz.sourcePcm = pcm;
  wiz.presetSel = presetSel;
  wiz.sel = presetSel || sel || defaultSelection(pcm, 24000, { minS: MIN_REF_S, maxS: DEFAULT_REF_S });
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
  if (final) { updateTrimInfo(); updateAsrButton(); }
}

function updateTrimInfo() {
  if (!wizTrimInfo || !wiz.sel || !wiz.sourcePcm) return;
  const total = wiz.sourcePcm.length / 24000;
  const range = `${fmtClock(wiz.sel.start / 24000)} – ${fmtClock(wiz.sel.end / 24000)} of ${fmtClock(total)}`;
  wizTrimInfo.textContent = `${range} · Drag the green handles or the window to choose the part to clone (${MIN_REF_S}–${MAX_REF_S} s)`;
  updateTranscriptHint();
}

function updateTranscriptHint() {
  if (!wizTranscriptHint) return;
  const atPreset = wiz.presetSel && wiz.sel && wiz.sel.start === wiz.presetSel.start && wiz.sel.end === wiz.presetSel.end;
  if (asr.error) {
    wizTranscriptHint.textContent = asr.error;
  } else if (asrStale()) {
    wizTranscriptHint.textContent = 'The selection changed since it was transcribed. Transcribe it again or edit the text to match.';
  } else if (asr.autoText != null && wizTranscript.value === asr.autoText) {
    const lang = languageName(asr.lang);
    wizTranscriptHint.textContent = `Transcribed automatically${lang ? ` (${lang})` : ''}. Check that it matches the selected part word for word.`;
  } else if (trimEditor && wiz.sourcePcm && !atPreset && trimEditor.cutsSpeech()) {
    wizTranscriptHint.textContent = 'You trimmed the clip: the transcript must contain exactly the words spoken in the selected part.';
  } else {
    wizTranscriptHint.textContent = defaultTranscriptHint();
  }
}

function defaultTranscriptHint() {
  return wiz.source === 'generated'
    ? 'Pre-filled with the generated text.'
    : 'Type the exact words spoken in the selected part, or transcribe them automatically. An exact transcript noticeably improves cloning.';
}


// ── Automatic transcription ──
// Whisper large-v3-turbo (workers/asr-worker.js) — what OmniVoice itself uses
// to transcribe references. The worker starts on first use and is stopped
// when the wizard closes, so its GPU memory is only held while it's useful.

const ASR_CACHED_KEY = 'vocoloco-asr-engine-cached'; // model downloaded before
const ASR_SIZE = '0.9 GB';
const asr = {
  worker: null, nextId: 0, pending: new Map(),
  busy: false, phase: null, pct: null,
  autoText: null, sel: null, lang: null, error: null, // sel: the selection autoText is for
};

function asrWorker() {
  if (asr.worker) return asr.worker;
  const w = new Worker('workers/asr-worker.js?v=5', { type: 'module' });
  w.onmessage = (e) => {
    const m = e.data;
    if (m.type === 'progress') {
      asr.phase = m.phase;
      asr.pct = m.totalBytes && m.loadedBytes < m.totalBytes ? Math.floor((m.loadedBytes / m.totalBytes) * 100) : null;
      updateAsrButton();
    } else if (m.type === 'ready') {
      localStorage.setItem(ASR_CACHED_KEY, '1');
    } else if (m.type === 'result' || m.type === 'error') {
      if (m.type === 'result') localStorage.setItem(ASR_CACHED_KEY, '1');
      const p = asr.pending.get(m.id);
      if (!p) return;
      asr.pending.delete(m.id);
      if (m.type === 'result') p.resolve(m);
      else p.reject(Object.assign(new Error(m.message), { code: m.code }));
    }
  };
  w.onerror = (e) => {
    console.error('ASR worker error:', e);
    stopAsr({ code: 'failed', message: e.message || 'the speech recognizer could not start' });
  };
  asr.worker = w;
  return w;
}

// Pending requests fail with `reason` (default: stopped on purpose, no message shown)
function stopAsr(reason = { code: 'stopped', message: 'stopped' }) {
  if (asr.worker) { asr.worker.terminate(); asr.worker = null; }
  for (const p of asr.pending.values()) p.reject(Object.assign(new Error(reason.message), { code: reason.code }));
  asr.pending.clear();
}

async function resample16k(pcm24) {
  const ctx = new OfflineAudioContext(1, Math.ceil(pcm24.length * 16000 / 24000), 16000);
  const buf = ctx.createBuffer(1, pcm24.length, 24000);
  buf.copyToChannel(pcm24, 0);
  const src = ctx.createBufferSource();
  src.buffer = buf;
  src.connect(ctx.destination);
  src.start();
  return new Float32Array((await ctx.startRendering()).getChannelData(0));
}

// Phones: Whisper runs alone, the speech model's worker is stopped meanwhile (parkSpeech), and
// Whisper's is ended right after, before anything else needs the memory. Elsewhere its weights
// stay on the GPU while the wizard is open.
async function requestTranscript(pcm24) {
  const [base, pcm] = await Promise.all([resolveModelBaseUrl(), resample16k(pcm24)]);
  const run = () => {
    const id = ++asr.nextId;
    setBusy('transcribe', true);
    setActivity('Transcribing');
    showProgress('indeterminate');
    return new Promise((resolve, reject) => {
      asr.pending.set(id, { resolve, reject });
      asrWorker().postMessage({ type: 'transcribe', id, pcm, base, resident: !LOW_MEMORY }, [pcm.buffer]);
    }).finally(() => {
      setBusy('transcribe', false);
      if (!isGenerating) hideProgress();
      if (LOW_MEMORY) stopAsr();
    });
  };
  return LOW_MEMORY ? onGpu(() => parkSpeech().then(run)) : run();
}

function languageName(code) {
  if (!code) return null;
  try { return new Intl.DisplayNames(['en'], { type: 'language' }).of(code); } catch { return code; }
}

function asrOff() { return LOW_MEMORY && localStorage.getItem(ASR_OFF_KEY) === '1'; } // see closedWhile
function asrSupported() { return !!navigator.gpu && !asrOff(); }

const ASR_BUSY_LABELS = ['Downloading 100%', 'Loading…', 'Transcribing…', 'Transcribe again'];

function updateAsrButton() {
  if (!wizAsrBtn) return;
  const cached = localStorage.getItem(ASR_CACHED_KEY) === '1';
  const idle = !asrSupported() ? (asrOff() ? 'Off: low memory' : 'Needs WebGPU')
    : cached ? 'Transcribe' : `Transcribe · ${ASR_SIZE}`;
  let label;
  if (!asr.busy) label = asrSupported() && asrStale() ? 'Transcribe again' : idle;
  else if (asr.phase === 'download' && asr.pct != null) label = `Downloading ${asr.pct}%`;
  else if (asr.phase === 'download' || asr.phase === 'compile') label = 'Loading…';
  else label = 'Transcribing…';
  wizAsrLabel.textContent = label;
  // Every label it can show stays reserved (invisible), so the button keeps one
  // size through download, transcription and back. Reset per review visit.
  if (wizAsrReserve && !wizAsrReserve.textContent) wizAsrReserve.textContent = [idle, ...ASR_BUSY_LABELS].join('\n');
  wizAsrBtn.disabled = !asrSupported() || asr.busy || !wiz.pcm;
  wizAsrBtn.classList.toggle('busy', asr.busy);
  // the download's progress fills the button from the left
  wizAsrBtn.style.setProperty('--asr-pct', asr.busy && asr.phase === 'download' && asr.pct != null ? `${asr.pct}%` : '0%');
  wizAsrBtn.title = cached ? '' : `Downloads the Whisper speech recognition model once (${ASR_SIZE}); it runs on your device`;
}

// The automatic transcript no longer matches the selection (moved since)
function asrStale() {
  return asr.autoText != null && wizTranscript.value === asr.autoText && !!asr.sel && !!wiz.sel
    && (asr.sel.start !== wiz.sel.start || asr.sel.end !== wiz.sel.end);
}

// Only on request: the user picks the part first, then transcribes it
async function runAutoTranscribe() {
  if (!asrSupported() || !wiz.pcm || asr.busy) return;
  keepStorage();
  asr.busy = true;
  asr.error = null;
  asr.phase = asr.worker ? 'transcribe' : 'download';
  updateAsrButton();
  const source = wiz.sourcePcm;
  const sel = { ...wiz.sel };
  try {
    const r = await requestTranscript(wiz.pcm);
    if (wiz.sourcePcm !== source || wiz.step !== 'review') return; // the wizard moved on
    wizTranscript.value = r.text;
    asr.autoText = r.text;
    asr.sel = sel;
    asr.lang = r.lang;
    if (!wiz.lang && r.lang) wiz.lang = r.lang;
  } catch (e) {
    if (e.code === 'stopped') return;
    asr.error = e.code === 'no-webgpu'
      ? 'Automatic transcription needs WebGPU. Type the transcript instead.'
      : `Automatic transcription failed (${e.message}). Type the transcript instead.`;
  } finally {
    asr.busy = false;
    asr.phase = null;
    updateAsrButton();
    if (wiz.step === 'review') updateTranscriptHint();
  }
}

if (wizAsrBtn) wizAsrBtn.addEventListener('click', runAutoTranscribe);
if (wizTranscript) wizTranscript.addEventListener('input', () => { asr.error = null; updateTranscriptHint(); updateAsrButton(); });

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
  record: 'Record your voice',
  upload: 'Upload audio',
  review: 'Review & save',
  saving: 'Saving…',
  done: 'Done',
};

function wizReset() {
  asr.autoText = null;
  asr.sel = null;
  asr.lang = null;
  asr.error = null;
  wiz.step = null;
  wiz.method = null;
  wiz.source = null;
  wiz.lang = null;
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
  showUploadError('');
  showSaveBlocker(null);
  if (wizSaveBtn) wizSaveBtn.disabled = false;
}

async function openVoiceWizard(startStep = 'method', preset = null) {
  if (!await ensureCloning()) return; // its models first, all at once
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
  // ONNX Runtime (no engine): fetch its voice encoder while the user records or picks audio. A new
  // voice asks for the download right away; a generation saved as a voice may reuse its tokens, so
  // it only asks when saving needs the encoder.
  if (isReady && encoderAvailable && !engineBackend()) {
    const prefetch = () => ttsWorker.postMessage({ type: 'prefetch-encoder' });
    if (canEncodeSilently()) prefetch();
    else if (!preset) confirmEncoderDownload().then((ok) => { if (ok) prefetch(); });
  }
}

function closeVoiceWizard() {
  wizStopMedia();
  stopAsr(); // frees its GPU memory
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

  if (step === 'record') startMicCheck(wiz.deviceId || undefined);
  if (step === 'upload') showUploadError('');
  if (step === 'review') {
    requestAnimationFrame(() => {
      if (wiz.sourcePcm && trimEditor) trimEditor.load(wiz.sourcePcm, wiz.sel);
    });
    updateReviewVerdict();
    wizReviewBack.textContent = wiz.source === 'recorded' ? '← Re-record'
      : wiz.source === 'uploaded' ? '← Choose another file'
      : 'Cancel';
    updateTranscriptHint();
    if (wizAsrReserve) wizAsrReserve.textContent = ''; // sized again for this visit
    updateAsrButton();
    // Not on touch screens: the keyboard would pop up over the trim editor
    if (!wizVoiceName.value && !TOUCH_KEYBOARD.matches) wizVoiceName.focus({ preventScroll: true });
  } else {
    stopReviewPreview();
  }
}

function wizBack() {
  switch (wiz.step) {
    case 'method': closeVoiceWizard(); break;
    case 'record': wizGoto('method'); break;
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
onActivate(wizMethodRecord, () => { wiz.method = 'record'; wizGoto('record'); });
onActivate(wizMethodUpload, () => { wiz.method = 'upload'; wizGoto('upload'); });

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

// Short enough for one line on a phone: the row never changes height
const WIZ_ZONE_IDLE_LABEL = `At least ${MIN_REF_S} s · 5 to 12 s is best`;

function recZoneLabel(elapsed) {
  if (elapsed < MIN_REF_S) return `Keep talking, at least ${MIN_REF_S} s`;
  if (elapsed < 5) return 'Usable, a bit longer is better';
  if (elapsed < 12) return 'Good length, stop anytime';
  return `Long enough, stops at ${MAX_REF_S} s`;
}

// Counts up to the minimum first ("1.2 / 3s"), so the requirement stays visible
function recElapsedText(elapsed) {
  const t = (Math.floor(Math.max(0, elapsed) * 10) / 10).toFixed(1);
  return elapsed < MIN_REF_S ? `${t} / ${MIN_REF_S}s` : `${t}s`;
}

// Stop only works from MIN_REF_S on, so a take can't end too short. aria-disabled
// rather than disabled: the button keeps keyboard focus while it waits.
function setRecordStopLocked(locked) {
  wiz.stopLocked = locked;
  if (!wizRecordBtn) return;
  if (locked) wizRecordBtn.setAttribute('aria-disabled', 'true');
  else wizRecordBtn.removeAttribute('aria-disabled');
  if (wiz.recording && wizRecordLabel) {
    wizRecordLabel.textContent = locked ? `Recording · you can stop at ${MIN_REF_S} s` : 'Recording · tap to stop';
  }
}

// The take had too little speech: stay here, keep its length on the timer, say
// why in the reserved status line and offer to record again
function rejectWizTake() {
  wiz.note = `Not enough speech. Talk for at least ${MIN_REF_S} s.`;
  if (wizSilenceHint) wizSilenceHint.textContent = wiz.note;
  if (wizRecZone) wizRecZone.textContent = WIZ_ZONE_IDLE_LABEL;
  if (wizRecordLabel) wizRecordLabel.textContent = 'Tap to record again';
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
    // One reserved status line (no layout jump): a rejected take's note wins over
    // the silence warning. Written only on change, so it is announced once.
    if (wizSilenceHint) {
      const text = wiz.note || (!allQuiet ? ''
        : wiz.recording ? "Can't hear you. Check the mic or speak up."
        : 'No signal. Try another microphone above.');
      if (wizSilenceHint.textContent !== text) wizSilenceHint.textContent = text;
    }

    // Recording timeline
    if (wiz.recording) {
      const elapsed = (now - wiz.recStartTs) / 1000;
      if (wizRecElapsed) wizRecElapsed.textContent = recElapsedText(elapsed);
      if (wizRecFill) wizRecFill.style.width = Math.min(100, (elapsed / 15) * 100) + '%';
      if (wizRecZone) wizRecZone.textContent = recZoneLabel(elapsed);
      if (wiz.stopLocked && elapsed >= MIN_REF_S) setRecordStopLocked(false);
      if (elapsed >= MAX_REF_S) stopWizRecording();
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
    if (wiz.recording) {
      // Too early to stop (the button shows it): keep recording
      if (performance.now() - wiz.recStartTs >= MIN_REF_S * 1000) stopWizRecording();
      return;
    }
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
  wiz.note = null; // a new take replaces the last one
  if (wizSilenceHint) wizSilenceHint.textContent = '';
  if (wizRecElapsed) wizRecElapsed.textContent = recElapsedText(0);
  if (wizRecFill) wizRecFill.style.width = '0%';
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
  setRecordStopLocked(true);
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
  setRecordStopLocked(false);
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
  if (wiz.step !== 'record') return; // left the step while it was decoding
  if (!wizLoadAudio(pcm, 'recorded')) rejectWizTake();
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
  wiz.note = null;
  setRecordStopLocked(false);
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
    if (wizRecElapsed) wizRecElapsed.textContent = recElapsedText(0);
    if (wizRecFill) wizRecFill.style.width = '0%';
    if (wizRecZone) wizRecZone.textContent = WIZ_ZONE_IDLE_LABEL;
    if (wizSilenceHint) wizSilenceHint.textContent = '';
    if (wizLevelFill) { wizLevelFill.style.width = '0%'; wizLevelFill.classList.remove('hot'); }
  }
  setMicStatus(false);
}

window.addEventListener('beforeunload', wizStopMedia);

// ── Upload step ──

// The element stays in the page (empty when there's nothing to say), so the
// live region announces reliably; '' clears it
function showUploadError(text) {
  if (wizUploadError) wizUploadError.textContent = text;
}

async function handleWizFile(file) {
  if (!file) return;
  showUploadError('');
  let pcm;
  try {
    if (wizDropzone) wizDropzone.classList.add('loading');
    pcm = await decodeRefAudio(file);
  } catch (e) {
    console.warn('Audio decode failed:', e);
    showUploadError(file.size > 200 * 1024 * 1024
      ? "Couldn't decode this file. It may be too long for the browser to decode at once. Try a shorter excerpt or a compressed format (MP3, M4A, OGG)."
      : "Couldn't read this file. Use MP3, WAV, M4A, OGG, or a video file with an audio track.");
    return;
  } finally {
    if (wizDropzone) wizDropzone.classList.remove('loading');
    if (wizFileInput) wizFileInput.value = ''; // the same file can be picked again
  }
  if (wiz.step !== 'upload') return; // left the step while it was decoding
  if (!wizLoadAudio(pcm, 'uploaded')) showUploadError(`Too short. Use a clip with at least ${MIN_REF_S} s of speech.`);
}

// Uploads and recordings continue the same way: trim editor, transcript, save.
// A clip with less than MIN_REF_S of speech doesn't get there: returns false
// and changes nothing, the caller explains it on its own step.
function wizLoadAudio(pcm, source) {
  const sel = defaultSelection(pcm.subarray(0, MAX_SOURCE_S * 24000), 24000, { minS: MIN_REF_S, maxS: DEFAULT_REF_S });
  if (sel.end - sel.start < MIN_REF_S * 24000) return false;
  wiz.source = source;
  wiz.presetTokens = null;
  wizSetSource(pcm, null, sel);
  wiz.lang = null;
  wizTranscript.value = '';
  asr.autoText = null;
  asr.sel = null;
  asr.error = null;
  wizGoto('review');
  return true;
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
  if (d < MIN_REF_S) {
    // Recordings and uploads never get here (wizLoadAudio) and the trim editor
    // keeps selections >= MIN_REF_S; only a short preset or clip can
    cls = 'bad'; label = 'Too short';
    blocker = wiz.sourcePcm && wiz.sourcePcm.length >= MIN_REF_S * 24000
      ? `Select at least ${MIN_REF_S} s to save.`
      : `Too short to clone. It needs at least ${MIN_REF_S} s of speech.`;
  }
  else if (d < 5) { cls = 'warn'; label = 'Usable'; }
  else if (d <= 12) { cls = 'good'; label = 'Good length'; }
  else { cls = 'warn'; label = 'Long · slower'; }
  if (wizReviewVerdict) {
    wizReviewVerdict.className = 'verdict-chip ' + cls;
    wizReviewVerdict.textContent = label;
  }
  if (wizSaveBtn) wizSaveBtn.disabled = d < MIN_REF_S;
  showSaveBlocker(blocker);
}

// Hint under the review fields, also the Save button's description
// (aria-describedby), so it's emptied when hidden
function showSaveBlocker(text) {
  if (!wizSaveBlocker) return;
  wizSaveBlocker.textContent = text || '';
  wizSaveBlocker.classList.toggle('hidden', !text);
}

if (wizReviewPlay) {
  wizReviewPlay.addEventListener('click', () => {
    if (!wiz.pcm) return;
    togglePcmPreview(wiz.pcm, wizReviewPlay);
    if (trimEditor && previewBtn === wizReviewPlay && player.isOneShotPlaying()) startReviewPlayhead();
  });
}

// ── Save flow (persist first, encode after — a voice can never be lost) ──

// Saving step: the status line only changes with the phase (it's a live region,
// so no announcement per progress tick); the progress sits on its own reserved,
// tabular line below, so nothing reflows while it counts
function setWizSavingStatus(text, progress = null) {
  if (wizSavingStatus && wizSavingStatus.textContent !== text) wizSavingStatus.textContent = text;
  if (wizSavingPct) {
    wizSavingPct.textContent = progress || '';
    wizSavingPct.style.visibility = progress ? 'visible' : 'hidden';
  }
}

async function wizSave() {
  // Length first: Enter in the name field reaches here even while Save is disabled
  if (!wiz.pcm || wiz.duration < MIN_REF_S) { updateReviewVerdict(); return; }
  const name = wizVoiceName.value.trim();
  if (!name) {
    showSaveBlocker('Give the voice a name.');
    wizVoiceName.focus();
    return;
  }
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
    lang: wiz.lang,
    source: wiz.source || 'recorded',
  };

  setWizSavingStatus('Saving…');
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
        setWizSavingStatus('Analyzing voice… (one-time setup)');
        await encodeVoiceShared(record); // persists via merge, shares in-flight work
      } catch (e) {
        encodeNote = e.code === 'encoder-unavailable'
          ? 'This device could not run the voice encoder. The recording is saved and will finish setting up on a device with more memory.'
          : e.code === 'encoder-declined'
            ? 'Saved. Analyzed once you allow the encoder download.'
            : 'Voice analysis failed. It will be retried automatically before the next use.';
      }
    } else if (!isReady) {
      encodeNote = 'Models are still loading. The voice will finish setting up automatically.';
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
  const modelsSize = await getCacheAPISize('omnivoice-models-v1') + await diskUsage();
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

// TTS models, the speech recognizer (+ the tokenizer files earlier versions kept via transformers.js)
async function clearModelCaches() {
  stopAsr();
  // The engine keeps its weight files open: stop it, so they can be deleted
  rejectPendingEncodes('Engine restarted');
  try { ttsWorker.terminate(); } catch { /* already dead */ }
  isReady = false;
  await Promise.all(['omnivoice-models-v1', 'vocoloco-tokenizer-v1', 'vocoloco-asr-v1', 'vocoloco-asr-v2', 'transformers-cache'].map((n) => caches.delete(n)));
  await clearDisk(); // the engine's weights (model-files.js, openDiskFile)
  localStorage.removeItem(ASR_CACHED_KEY);
  localStorage.removeItem(CLONING_KEY); // the next download asks about voice cloning again
  encoderCached = false;
  Object.assign(cloning, { total: 0, stored: 0, loaded: 0 });
}

const clearCacheBtn = $('clear-cache-btn');
if (clearCacheBtn) {
  clearCacheBtn.addEventListener('click', async () => {
    const ok = await confirmDialog({
      title: 'Clear cached models?',
      body: 'The voice model (~1.2 GB) will download again.',
      confirmLabel: 'Clear models',
      danger: true,
    });
    if (!ok) return;
    await clearModelCaches();
    await clearDB('omnivoice-cache'); // legacy cache cleanup
    calculateStorage();
    toast('Cached models cleared', { type: 'success' });
    initWorker(); // asks before downloading again
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
      body: 'Models, history, and voices will be deleted. Models (~2.4 GB) will re-download on next use.',
      confirmLabel: 'Clear everything',
      danger: true,
    });
    if (!ok) return;
    await clearModelCaches();
    await clearDB('omnivoice-cache');
    await clearDB(HISTORY_DB);
    await clearDB(VOICE_DB);
    location.reload();
  });
}

// ─── Init ───────────────────────────────────────────────────────────────────

// Offline use (sw.js): the app's files and the models' descriptions are kept as they load. Online
// everything still comes from the network first, so updates arrive as before
navigator.serviceWorker?.register('sw.js', { updateViaCache: 'none' }).catch((e) => console.warn('[offline]', e.message));

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
