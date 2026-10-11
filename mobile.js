/**
 * mobile.js — native-app behaviour for the phone layout in mobile.css.
 *  - <html data-view>: the visible view (CSS shows the About footer in Settings)
 *  - per-tab scroll positions; tapping the active tab scrolls to the top
 *  - keyboard tracking (visualViewport): .kb-open / .kb-text and --vv-top /
 *    --vv-h / --kb, so the tag row docks on top of the keyboard on iOS and Android
 *  - --m-sab-max: the largest bottom safe area so far, so the fixed bars hold still
 *  - a Done button in that row (iOS keyboards have no key to close them)
 *  - drag down to close the voice wizard sheet and confirm dialogs
 * Desktop (outside the phone media query) only gets data-view.
 */

const PHONE_QUERY = '(max-width: 768px), (pointer: coarse) and (max-height: 520px)';
const phone = matchMedia(PHONE_QUERY);
const coarse = matchMedia('(pointer: coarse)');
const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
const root = document.documentElement;
const $ = (id) => document.getElementById(id);

// ─── Current view, per-tab scroll positions ─────────────────────────────────

const views = [...document.querySelectorAll('.app-view')];
const wizard = $('voice-wizard');
const scrollPos = {};
let view = null;

function syncState() {
  const shown = views.find((v) => v.style.display !== 'none');
  const name = shown ? shown.id.replace('view-', '') : 'studio';
  if (name !== view) {
    const first = view === null;
    view = name;
    root.dataset.view = name;
    if (!first && phone.matches) document.body.scrollTo(0, scrollPos[name] || 0);
  }
  // getClientRects is empty while the wizard or its view is display:none
  root.classList.toggle('sheet-open', phone.matches && !!wizard && wizard.getClientRects().length > 0);
}

const styleObserver = new MutationObserver(syncState);
for (const el of [...views, wizard]) if (el) styleObserver.observe(el, { attributes: true, attributeFilter: ['style'] });
phone.addEventListener('change', syncState);
syncState();

// On the phone the page scrolls inside body, not the document (mobile.css)
document.body.addEventListener('scroll', () => { if (view) scrollPos[view] = document.body.scrollTop; }, { passive: true });

// Tapping the tab you're on scrolls back to the top, like a native tab bar
document.addEventListener('click', (e) => {
  const tab = e.target.closest && e.target.closest('.mobile-tab');
  if (!tab || !phone.matches || tab.dataset.view !== view) return;
  document.body.scrollTo({ top: 0, behavior: reducedMotion.matches ? 'auto' : 'smooth' });
}, true);

// ─── Keyboard (visualViewport) ──────────────────────────────────────────────
// iOS never resizes the layout viewport for the keyboard, Android Chrome
// doesn't either by default (interactive-widget=resizes-visual), so the
// keyboard is the gap between the visual viewport and the window bottom.

const vv = window.visualViewport;
const NOT_TYPING = new Set(['range', 'checkbox', 'radio', 'button', 'submit', 'reset', 'file', 'color']);
let raf = 0;

// A custom property on <html> restyles the whole page: only write the ones that changed
const written = {};
function setVar(name, value) {
  if (written[name] === value) return;
  written[name] = value;
  root.style.setProperty(name, value);
}

function isTyping(el) {
  if (!el) return false;
  if (el.tagName === 'TEXTAREA' || el.isContentEditable) return true;
  return el.tagName === 'INPUT' && !NOT_TYPING.has(el.type);
}

function updateViewport() {
  raf = 0;
  if (!vv) return;
  const zoomed = vv.scale > 1.05; // pinch zoom also shrinks the visual viewport
  const kb = zoomed ? 0 : Math.max(0, Math.round(window.innerHeight - vv.height - vv.offsetTop));
  const active = document.activeElement;
  const open = phone.matches && kb > 80 && isTyping(active);
  setVar('--vv-top', `${Math.round(vv.offsetTop)}px`);
  setVar('--kb', `${kb}px`);
  // Only the keyboard's tag row uses the height, and it changes on every frame while the
  // browser's toolbar slides: writing it then would restyle the page on every frame
  if (open) setVar('--vv-h', `${Math.round(vv.height)}px`);
  root.classList.toggle('kb-open', open);
  root.classList.toggle('kb-text', open && active.id === 'text-input');
}

function scheduleViewport() { if (!raf) raf = requestAnimationFrame(updateViewport); }

if (vv) {
  vv.addEventListener('resize', scheduleViewport);
  vv.addEventListener('scroll', scheduleViewport);
  document.addEventListener('focusin', scheduleViewport);
  // focus moves to <body> before the next field gets it: wait a moment
  document.addEventListener('focusout', () => setTimeout(scheduleViewport, 60));
  updateViewport();
}

// ─── Bottom safe area that holds still ──────────────────────────────────────
// With Safari's Top tab layout (and in other iOS browsers) the bottom inset grows from
// 0 to 34 px while the toolbar slides away and shrinks again when it comes back. Safari
// moves the fixed bars along with the viewport edge before the page has re-laid out the
// new inset, so the tab bar and the Generate dock shook on every frame of that slide.
// mobile.css pads them with the largest inset seen in this orientation instead.

const insetProbe = document.createElement('div');
insetProbe.style.cssText = 'position:absolute;top:0;left:0;width:0;height:env(safe-area-inset-bottom,0px);visibility:hidden;pointer-events:none';
document.body.appendChild(insetProbe);
const landscape = matchMedia('(orientation: landscape)');
const insetMax = { portrait: 0, landscape: 0 };

function updateInset() {
  const side = landscape.matches ? 'landscape' : 'portrait';
  insetMax[side] = Math.max(insetMax[side], Math.round(insetProbe.getBoundingClientRect().height));
  setVar('--m-sab-max', `${insetMax[side]}px`);
}

if (window.ResizeObserver) new ResizeObserver(updateInset).observe(insetProbe);
landscape.addEventListener('change', updateInset);
updateInset();

// Done: closes the keyboard (iOS has no key for it in a multi-line field).
// The tag bar's pointerdown handler (app.js) keeps the focus until the click.
const tagBar = $('tag-bar');
const textInput = $('text-input');
if (tagBar && textInput && coarse.matches) {
  const done = document.createElement('button');
  done.type = 'button';
  done.className = 'm-kb-done';
  done.textContent = 'Done';
  done.setAttribute('aria-label', 'Close the keyboard');
  done.addEventListener('click', () => textInput.blur());
  tagBar.appendChild(done);
}

// ─── Bottom sheets: drag down to close ──────────────────────────────────────

function dragToClose(sheet, handle, { canClose, close }) {
  let startY = 0, dy = 0, t0 = 0, pointer = null;
  handle.addEventListener('pointerdown', (e) => {
    if (!phone.matches || e.button > 0 || pointer !== null) return;
    if (e.target.closest('button, a, input, select, textarea') || !canClose()) return;
    pointer = e.pointerId;
    startY = e.clientY;
    dy = 0;
    t0 = performance.now();
    handle.setPointerCapture(pointer);
    sheet.style.transition = 'none';
  });
  handle.addEventListener('pointermove', (e) => {
    if (e.pointerId !== pointer) return;
    dy = Math.max(0, e.clientY - startY);
    sheet.style.transform = dy ? `translateY(${dy}px)` : '';
  });
  const end = (e) => {
    if (e.pointerId !== pointer) return;
    pointer = null;
    const speed = dy / Math.max(1, performance.now() - t0); // px per ms
    const height = sheet.getBoundingClientRect().height;
    const closing = dy > Math.min(160, height * 0.3) || (speed > 0.5 && dy > 40);
    if (!dy) { sheet.style.transition = ''; return; }
    sheet.style.transition = `transform ${reducedMotion.matches ? 0 : 0.26}s cubic-bezier(0.32, 0.72, 0, 1)`;
    sheet.style.transform = closing ? `translateY(${height}px)` : '';
    setTimeout(() => {
      if (closing) close();
      sheet.style.transition = '';
      sheet.style.transform = '';
    }, reducedMotion.matches ? 0 : 260);
  };
  handle.addEventListener('pointerup', end);
  handle.addEventListener('pointercancel', end);
}

// Voice wizard: drag its header (not while saving, when it can't be closed)
const wizHead = wizard && wizard.querySelector('.wiz-head');
const wizClose = $('wiz-close-btn');
if (wizHead && wizClose) {
  dragToClose(wizard, wizHead, {
    canClose: () => wizClose.style.visibility !== 'hidden',
    close: () => wizClose.click(),
  });
}

// Confirm dialogs (ui-dialogs.js): dragging down = Cancel
new MutationObserver((records) => {
  for (const r of records) {
    for (const node of r.addedNodes) {
      if (!(node instanceof HTMLElement) || !node.classList.contains('vl-overlay')) continue;
      const dialog = node.querySelector('.vl-dialog');
      if (!dialog) continue;
      dragToClose(dialog, dialog, {
        canClose: () => true,
        close: () => { const c = dialog.querySelector('.vl-btn-cancel'); if (c) c.click(); },
      });
    }
  }
}).observe(document.body, { childList: true });
