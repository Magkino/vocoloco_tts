/**
 * ui-dialogs.js — styled toasts and confirm dialogs replacing native
 * alert/confirm/prompt. Matches the app's glass/neon design language
 * (CSS classes .vl-toast / .vl-overlay / .vl-dialog live in index.html).
 */

// ─── Toasts ─────────────────────────────────────────────────────────────────

let toastStack = null;

function getToastStack() {
  if (!toastStack || !toastStack.isConnected) {
    toastStack = document.createElement('div');
    toastStack.id = 'toast-stack';
    toastStack.setAttribute('role', 'status');
    toastStack.setAttribute('aria-live', 'polite');
    document.body.appendChild(toastStack);
  }
  return toastStack;
}

export function toast(message, { type = 'info', duration = 3500 } = {}) {
  const el = document.createElement('div');
  el.className = 'vl-toast' + (type === 'error' ? ' error' : type === 'success' ? ' success' : '');
  el.textContent = message;
  const dismiss = () => {
    if (!el.isConnected) return;
    el.style.opacity = '0';
    el.style.transform = 'translateY(-6px)';
    setTimeout(() => el.remove(), 250);
  };
  el.addEventListener('click', dismiss);
  getToastStack().appendChild(el);
  if (duration > 0) setTimeout(dismiss, duration);
  return dismiss;
}

// ─── Confirm dialog ─────────────────────────────────────────────────────────

// Serialize dialogs: a second confirmDialog() while one is open waits its turn.
let dialogChain = Promise.resolve();

export function confirmDialog({ title, body = '', confirmLabel = 'Confirm', cancelLabel = 'Cancel', danger = false } = {}) {
  const run = () => new Promise((resolve) => {
    const previouslyFocused = document.activeElement;

    const overlay = document.createElement('div');
    overlay.className = 'vl-overlay';

    const dialog = document.createElement('div');
    dialog.className = 'vl-dialog';
    dialog.setAttribute('role', 'alertdialog');
    dialog.setAttribute('aria-modal', 'true');

    const titleEl = document.createElement('h3');
    titleEl.id = 'vl-dialog-title-' + Date.now();
    titleEl.textContent = title || '';
    dialog.setAttribute('aria-labelledby', titleEl.id);

    const bodyEl = document.createElement('p');
    bodyEl.textContent = body;

    const actions = document.createElement('div');
    actions.className = 'vl-actions';

    const cancelBtn = document.createElement('button');
    cancelBtn.className = 'vl-btn vl-btn-cancel';
    cancelBtn.textContent = cancelLabel;

    const confirmBtn = document.createElement('button');
    confirmBtn.className = 'vl-btn ' + (danger ? 'vl-btn-danger' : 'vl-btn-confirm');
    confirmBtn.textContent = confirmLabel;

    actions.append(cancelBtn, confirmBtn);
    dialog.append(titleEl, bodyEl, actions);
    overlay.appendChild(dialog);

    const close = (result) => {
      document.removeEventListener('keydown', onKeydown, true);
      overlay.remove();
      if (previouslyFocused && previouslyFocused.focus) {
        try { previouslyFocused.focus({ preventScroll: true }); } catch { /* detached */ }
      }
      resolve(result);
    };

    const onKeydown = (e) => {
      if (e.key === 'Escape') {
        e.preventDefault();
        e.stopPropagation();
        close(false);
      } else if (e.key === 'Tab') {
        // Focus trap between the two buttons
        e.preventDefault();
        (document.activeElement === confirmBtn ? cancelBtn : confirmBtn).focus();
      }
    };

    cancelBtn.addEventListener('click', () => close(false));
    confirmBtn.addEventListener('click', () => close(true));
    overlay.addEventListener('click', (e) => { if (e.target === overlay) close(false); });
    document.addEventListener('keydown', onKeydown, true);

    document.body.appendChild(overlay);
    confirmBtn.focus();
  });

  const result = dialogChain.then(run);
  dialogChain = result.catch(() => {});
  return result;
}

/** True while a .vl-overlay dialog is open (used by the global Esc handler). */
export function isDialogOpen() {
  return !!document.querySelector('.vl-overlay');
}
