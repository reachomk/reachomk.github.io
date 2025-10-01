/************ CONFIG ************/
const SHEETS_WEBAPP_URL = 'https://script.google.com/macros/s/AKfycbyP3jSW8aeXnG2zxFKxKlhdIgTIZLBOIup0ZWusXmq3bfLqKsXU5qOA54uorg75439i/exec';
const TOKEN = ''; // must match Apps Script TOKEN (or '' to disable)
const HEARTBEAT_MS = 25000;       // < TTL to keep the lock alive
/********************************/

// Elements
const imgAEl    = document.getElementById('imgA');
const imgBEl    = document.getElementById('imgB');
const workerEl  = document.getElementById('workerId'); // optional personal ID; user_id is the shared group id
const selectEl  = document.getElementById('pair-category');
const submitBtn = document.getElementById('submitBtn');
const skipBtn   = document.getElementById('skipBtn');
const exportBtn = document.getElementById('exportBtn');
const progressEl= document.getElementById('progress');
const statusEl  = document.getElementById('status');

const qs        = new URLSearchParams(location.search);
const sessionId = crypto.randomUUID();
let userId      = qs.get('uid') || '';   // shared among many people in your study
if (workerEl) workerEl.value = userId;   // prefill if provided

// Local backup
const localKey   = 'annotations_local_backup_v1';
const localBackup= JSON.parse(localStorage.getItem(localKey) || '[]');

// State
let current = null;  // { pair_id, A_url, B_url }
let hbTimer = null;
let t0 = 0;
let completedCount = 0;

// Utilities
function pushLocal(row) {
  localBackup.push(row);
  localStorage.setItem(localKey, JSON.stringify(localBackup));
}
async function post(action, payload) {
  const body = JSON.stringify({ token: TOKEN || undefined, action, ...payload });
  try {
    const res = await fetch(SHEETS_WEBAPP_URL, {
      method: 'POST',
      mode: 'cors',
      headers: { 'Content-Type': 'application/json' },
      body
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  } catch (e) {
    // For submit/heartbeat, fallback to no-cors (fire-and-forget)
    if (action === 'submit' || action === 'heartbeat') {
      await fetch(SHEETS_WEBAPP_URL, { method:'POST', mode:'no-cors', headers:{'Content-Type':'application/json'}, body });
      return { ok:true, offline:true };
    }
    throw e;
  }
}
function requireUserId() {
  userId = (workerEl && workerEl.value.trim()) || userId || '';
  if (!userId) {
    statusEl.textContent = 'Enter a User ID (shared group id) before starting.';
    return false;
  }
  return true;
}

// Heartbeat
async function startHeartbeat() {
  stopHeartbeat();
  hbTimer = setInterval(() => {
    if (current?.pair_id) {
      post('heartbeat', { user_id: userId, session_id: sessionId, pair_id: current.pair_id }).catch(()=>{});
    }
  }, HEARTBEAT_MS);
}
function stopHeartbeat() {
  if (hbTimer) { clearInterval(hbTimer); hbTimer = null; }
}

// Claim next pair
async function claimNext() {
  if (!requireUserId()) return;
  statusEl.textContent = 'Claiming next pair...';
  try {
    const resp = await post('claim', { user_id: userId, session_id: sessionId });
    if (!resp?.ok) { statusEl.textContent = resp?.error || 'Claim failed'; return; }
    if (resp.done) {
      document.querySelector('main').innerHTML = `
        <h2>All done for now 🎉</h2>
        <p>No available pairs (all locked or completed).</p>`;
      return;
    }
    current = resp.pair; // { pair_id, A_url, B_url }
    imgAEl.src = current.A_url;
    imgBEl.src = current.B_url;
    selectEl.value = '';
    statusEl.textContent = '';
    progressEl.textContent = `Completed: ${completedCount}`;
    t0 = performance.now();
    startHeartbeat();
  } catch (e) {
    console.error(e);
    statusEl.textContent = 'Claim failed (network).';
  }
}

// Submit / Skip
async function submitChoice(choice) {
  if (!current?.pair_id) return;
  const rt = Math.round(performance.now() - t0);
  const worker_id = null; // keep separate if you want a per-person id

  const row = {
    session_id: sessionId,
    user_id: userId,
    worker_id,
    pair_id: current.pair_id,
    choice, rt_ms: rt,
    A_url: current.A_url, B_url: current.B_url,
    meta: { ua: navigator.userAgent }
  };

  // Local backup first
  pushLocal({ created_at: new Date().toISOString(), ...row });

  // Send to server (releases lock)
  try {
    await post('submit', row);
  } catch (e) {
    console.error(e);
    statusEl.textContent = 'Saved locally; upload will retry on next submit.';
  }

  stopHeartbeat();
  completedCount += (choice !== 'skip') ? 1 : 0;
  current = null;
  claimNext();
}

// UI events
submitBtn.addEventListener('click', () => {
  if (!requireUserId()) return;
  const val = selectEl.value;
  if (!val) { statusEl.textContent = 'Please choose a letter (A–Z).'; return; }
  submitChoice(val);
});
skipBtn.addEventListener('click', () => {
  if (!requireUserId()) return;
  submitChoice('skip');
});
exportBtn.addEventListener('click', () => {
  const rows = JSON.parse(localStorage.getItem(localKey) || '[]');
  if (!rows.length) { alert('No local rows to export.'); return; }
  const headers = Object.keys(rows[0]);
  const csv = [
    headers.join(','),
    ...rows.map(r => headers.map(h => JSON.stringify(r[h] ?? '')).join(','))
  ].join('\n');
  const blob = new Blob([csv], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const a = Object.assign(document.createElement('a'), { href:url, download:'annotations_backup.csv' });
  a.click(); URL.revokeObjectURL(url);
});

// Keyboard niceties (Enter = submit, K = skip)
window.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') submitBtn.click();
  if (e.key.toLowerCase() === 'k') skipBtn.click();
});

// Start
claimNext();
