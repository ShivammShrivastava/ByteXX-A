// Firebase configuration and services for ByteX SatQuery AI
// Note: Firebase Storage is NOT used (requires paid Blaze plan).
// Images are base64-encoded and stored directly in the free Realtime Database.
import { initializeApp, getApps, getApp } from 'firebase/app';
import {
  getAuth,
  signInWithEmailAndPassword,
  createUserWithEmailAndPassword,
  signInWithPopup,
  GoogleAuthProvider,
  signOut as fbSignOut,
  onAuthStateChanged,
  updateProfile
} from 'firebase/auth';
import {
  getDatabase,
  ref as dbRef,
  set,
  onValue,
  off,
  remove,
  push,
  get,
} from 'firebase/database';

// ✅ Real Firebase configuration — satellite-efa0a (ByteX project)
export const firebaseConfig = {
  apiKey: import.meta.env.VITE_FIREBASE_API_KEY || "AIzaSyDJtjpV4DpD-Ev0BeRJDsfZV4k5U63dpW4",
  authDomain: "satellite-efa0a.firebaseapp.com",
  databaseURL: "https://satellite-efa0a-default-rtdb.firebaseio.com",
  projectId: "satellite-efa0a",
  storageBucket: "satellite-efa0a.firebasestorage.app",
  messagingSenderId: "504899672780",
  appId: "1:504899672780:web:4f61a1b212930752cdc069",
  measurementId: "G-TY956H2RJF"
};

let app, auth, db;

try {
  // Reuse existing app instance if already initialized (handles Vite HMR)
  app = getApps().length > 0 ? getApp() : initializeApp(firebaseConfig);
  auth = getAuth(app);
  db = getDatabase(app);
  console.log("✅ Firebase initialized for project: satellite-efa0a");
} catch (error) {
  console.warn("Firebase init error:", error.message);
}

const googleProvider = new GoogleAuthProvider();
googleProvider.setCustomParameters({ prompt: 'select_account' });

// ─── User Data ─────────────────────────────────────────────────────────────

export async function saveUserData(user, extraData = {}) {
  if (!user?.uid) return null;

  const userData = {
    uid: user.uid,
    email: user.email || '',
    displayName: user.displayName || extraData.displayName || user.email?.split('@')[0] || 'Analyst',
    photoURL: user.photoURL || `https://api.dicebear.com/7.x/bottts/svg?seed=${user.uid}`,
    lastLogin: new Date().toISOString(),
    createdAt: extraData.createdAt || new Date().toISOString(),
    role: 'Satellite Analyst',
    ...extraData
  };

  // localStorage (instant, offline)
  try {
    localStorage.setItem('bytex_active_user', JSON.stringify(userData));
    localStorage.setItem(`bytex_user_${user.uid}`, JSON.stringify(userData));
  } catch (e) {}

  // Firebase RTDB at /users/{uid}/profile
  if (db) {
    try {
      await set(dbRef(db, `users/${user.uid}/profile`), userData);
      console.log(`✅ User saved to RTDB: /users/${user.uid}/profile`);
    } catch (e) {
      console.warn("RTDB write error:", e.message);
    }
  }

  return userData;
}

// ─── Recents ───────────────────────────────────────────────────────────────

export async function saveRecentQuery(uid, queryItem) {
  if (!uid) return null;

  const item = {
    ...queryItem,
    id: queryItem.id || `q_${Date.now()}`,
    timestamp: queryItem.timestamp || Date.now()
  };

  // localStorage cache
  try {
    const key = `bytex_recents_${uid}`;
    const existing = JSON.parse(localStorage.getItem(key) || '[]');
    const updated = [item, ...existing.filter(i => i.id !== item.id)].slice(0, 50);
    localStorage.setItem(key, JSON.stringify(updated));
  } catch (e) {}

  // RTDB: /users/{uid}/recents/{id}
  if (db) {
    try {
      await set(dbRef(db, `users/${uid}/recents/${item.id}`), item);
      console.log(`✅ Recent saved to RTDB: /users/${uid}/recents/${item.id}`);
    } catch (e) {
      console.warn("RTDB recents write:", e.message);
    }
  }

  return item;
}

export function subscribeToUserRecents(uid, onUpdate) {
  if (!uid) return () => {};

  // Seed from localStorage immediately
  try {
    const cached = JSON.parse(localStorage.getItem(`bytex_recents_${uid}`) || '[]');
    if (cached.length > 0) onUpdate(cached);
  } catch (e) {}

  if (!db) return () => {};

  try {
    const recentsRef = dbRef(db, `users/${uid}/recents`);
    const unsub = onValue(recentsRef, (snap) => {
      if (snap.exists()) {
        const items = Object.values(snap.val())
          .sort((a, b) => (b.timestamp || 0) - (a.timestamp || 0));
        localStorage.setItem(`bytex_recents_${uid}`, JSON.stringify(items));
        onUpdate(items);
      }
    }, (err) => console.warn("RTDB listener:", err.message));
    return unsub;
  } catch (e) {
    return () => {};
  }
}

export async function deleteRecentQuery(uid, queryId) {
  if (!uid || !queryId) return;
  try {
    const key = `bytex_recents_${uid}`;
    const existing = JSON.parse(localStorage.getItem(key) || '[]');
    localStorage.setItem(key, JSON.stringify(existing.filter(i => i.id !== queryId)));
  } catch (e) {}
  if (db) {
    try {
      await remove(dbRef(db, `users/${uid}/recents/${queryId}`));
    } catch (e) {}
  }
}

// ─── Image → Base64 helper ──────────────────────────────────────────────────

/**
 * Convert a File to a base64 string, resizing to max 1024px first
 * to keep the RTDB payload small (< 1 MB for most images).
 * Returns a plain base64 string (no data: prefix).
 */
export function fileToBase64(file, maxDimension = 1024) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      const img = new Image();
      img.onload = () => {
        const canvas = document.createElement('canvas');
        let { width, height } = img;
        if (width > maxDimension || height > maxDimension) {
          if (width > height) { height = Math.round((height * maxDimension) / width); width = maxDimension; }
          else                { width  = Math.round((width  * maxDimension) / height); height = maxDimension; }
        }
        canvas.width  = width;
        canvas.height = height;
        canvas.getContext('2d').drawImage(img, 0, 0, width, height);
        // JPEG at 85% quality keeps satellite detail while staying small
        const dataUrl = canvas.toDataURL('image/jpeg', 0.85);
        resolve(dataUrl.split(',')[1]); // strip "data:image/jpeg;base64," prefix
      };
      img.onerror = reject;
      img.src = e.target.result;
    };
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

// ─── Satellite Query Pipeline ───────────────────────────────────────────────

/**
 * Submit a VQA / caption / refer query to the backend via Firebase RTDB.
 * The image is stored as base64 directly in RTDB (no Storage needed — free plan).
 *
 * @param {string} uid          - Firebase user ID
 * @param {string} imageBase64  - Plain base64 string of the image (no data: prefix)
 * @param {string} question     - User question
 * @param {string} task         - "vqa" | "caption" | "refer"
 * @returns {Promise<string>} queryId
 */
export async function submitQueryToFirebase(uid, imageBase64, question, task = 'vqa') {
  if (!db) throw new Error("Firebase Database not initialized");

  const queriesRef = dbRef(db, 'queries');
  const newQuery = {
    uid,
    image_base64: imageBase64,   // backend reads this field directly
    question,
    task,
    status: 'pending',
    timestamp: Date.now(),
    source: 'frontend_web',
  };

  const newRef = await push(queriesRef, newQuery);
  const queryId = newRef.key;
  console.log(`✅ Query submitted to RTDB (base64 image): /queries/${queryId}`);
  return queryId;
}

/**
 * Listen in real-time to /results/{queryId}.
 * Calls onResult(data) as soon as the backend writes an answer.
 * Calls onStatusChange(status) when query status updates.
 *
 * Returns an unsubscribe function — call it to stop listening.
 */
export function listenForQueryResult(queryId, { onResult, onStatusChange, onHeartbeat, onError }) {
  if (!db || !queryId) return () => {};

  const resultRef  = dbRef(db, `results/${queryId}`);
  const queryRef   = dbRef(db, `queries/${queryId}`);

  // Listen for result written to /results/{queryId}
  const unsubResult = onValue(resultRef, (snap) => {
    if (snap.exists()) {
      const data = snap.val();
      console.log(`Result received for query ${queryId}:`, data);
      onResult && onResult(data);
    }
  }, (err) => {
    console.warn('Result listener error:', err.message);
    onError && onError(err.message);
  });

  // Listen for status + heartbeat changes on /queries/{queryId}
  const unsubQuery = onValue(queryRef, (snap) => {
    if (snap.exists()) {
      const val = snap.val();
      onStatusChange && onStatusChange(val?.status);
      // Backend writes heartbeat: elapsed seconds every 30s during long inference
      if (val?.heartbeat) onHeartbeat && onHeartbeat(val.heartbeat);
    }
  }, () => {});

  return () => { off(resultRef); off(queryRef); };
}

/**
 * Get current query status from RTDB (one-time read).
 */
export async function getQueryStatus(queryId) {
  if (!db || !queryId) return null;
  try {
    const snap = await get(dbRef(db, `queries/${queryId}`));
    return snap.exists() ? snap.val() : null;
  } catch (e) {
    return null;
  }
}

// ─── Auth Actions ──────────────────────────────────────────────────────────

export async function loginWithEmail(email, password) {
  if (!auth) throw new Error("Firebase Auth not initialized");
  const cred = await signInWithEmailAndPassword(auth, email, password);
  await saveUserData(cred.user);
  return cred.user;
}

export async function registerWithEmail(email, password, displayName) {
  if (!auth) throw new Error("Firebase Auth not initialized");
  const cred = await createUserWithEmailAndPassword(auth, email, password);
  if (displayName) {
    try { await updateProfile(cred.user, { displayName }); } catch (_) {}
  }
  await saveUserData({ ...cred.user, displayName }, { displayName, createdAt: new Date().toISOString() });
  return cred.user;
}

export async function loginWithGoogle() {
  if (!auth) throw new Error("Firebase Auth not initialized");
  const cred = await signInWithPopup(auth, googleProvider);
  await saveUserData(cred.user);
  return cred.user;
}

export async function logoutUser() {
  try {
    localStorage.removeItem('bytex_active_user');
    if (auth) await fbSignOut(auth);
  } catch (e) {}
}

export function onAuthChange(callback) {
  if (auth) {
    return onAuthStateChanged(auth, async (firebaseUser) => {
      if (firebaseUser) {
        callback(firebaseUser);
      } else {
        try {
          const cached = localStorage.getItem('bytex_active_user');
          callback(cached ? JSON.parse(cached) : null);
        } catch (_) {
          callback(null);
        }
      }
    });
  }
  try {
    const cached = localStorage.getItem('bytex_active_user');
    callback(cached ? JSON.parse(cached) : null);
  } catch (_) {
    callback(null);
  }
  return () => {};
}

// ─── AI Agent Query Pipeline ─────────────────────────────────────────────────

/**
 * Submit an AI Agent query to Firebase RTDB.
 * The backend listener picks this up, runs the full 5-stage agent pipeline,
 * and writes the structured report to /agent_results/{queryId}.
 *
 * @param {string} uid          - Firebase user ID
 * @param {string} imageBase64  - Plain base64 string of the image (no data: prefix)
 * @param {string} question     - User's natural-language question
 * @param {number} confThreshold - YOLO confidence threshold (default 0.25)
 * @returns {Promise<string>}   queryId
 */
export async function submitAgentQueryToFirebase(uid, imageBase64, question, confThreshold = 0.25) {
  if (!db) throw new Error("Firebase Database not initialized");

  const queriesRef = dbRef(db, 'queries');
  const newQuery = {
    uid,
    image_base64: imageBase64,
    question,
    task: 'agent',                 // ← listener routes to AI Agent pipeline
    conf_threshold: confThreshold,
    status: 'pending',
    timestamp: Date.now(),
    source: 'frontend_web',
  };

  const newRef = await push(queriesRef, newQuery);
  const queryId = newRef.key;
  console.log(`✅ Agent query submitted to RTDB: /queries/${queryId}`);
  return queryId;
}

/**
 * Submit a CNN-only query to Firebase RTDB.
 * The backend runs YOLOv8 detection and/or ResNet/FCN segmentation
 * and writes the result to /cnn_results/{queryId}.
 *
 * @param {string}   uid           - Firebase user ID
 * @param {string}   imageBase64   - Plain base64 string (no data: prefix)
 * @param {string[]} tasks         - ["detect", "segment"] or just one of them
 * @param {number}   confThreshold - YOLO confidence threshold
 * @returns {Promise<string>}      queryId
 */
export async function submitCNNQueryToFirebase(uid, imageBase64, tasks = ['detect', 'segment'], confThreshold = 0.25) {
  if (!db) throw new Error("Firebase Database not initialized");

  const queriesRef = dbRef(db, 'queries');
  const newQuery = {
    uid,
    image_base64: imageBase64,
    task: 'cnn',                   // ← listener routes to CNN pipeline
    cnn_tasks: tasks,
    conf_threshold: confThreshold,
    status: 'pending',
    timestamp: Date.now(),
    source: 'frontend_web',
  };

  const newRef = await push(queriesRef, newQuery);
  const queryId = newRef.key;
  console.log(`✅ CNN query submitted to RTDB: /queries/${queryId}`);
  return queryId;
}

/**
 * Listen in real-time for an AI Agent result at /agent_results/{queryId}.
 * Calls onResult(data) as soon as the backend writes the report.
 *
 * Returns an unsubscribe function.
 */
export function listenForAgentResult(queryId, { onResult, onStatusChange, onHeartbeat, onError }) {
  if (!db || !queryId) return () => {};

  const resultRef = dbRef(db, `agent_results/${queryId}`);
  const queryRef  = dbRef(db, `queries/${queryId}`);

  const unsubResult = onValue(resultRef, (snap) => {
    if (snap.exists()) {
      const data = snap.val();
      console.log(`Agent result received for query ${queryId}`);
      onResult && onResult(data);
    }
  }, (err) => {
    console.warn('Agent result listener error:', err.message);
    onError && onError(err.message);
  });

  const unsubQuery = onValue(queryRef, (snap) => {
    if (snap.exists()) {
      const val = snap.val();
      onStatusChange && onStatusChange(val?.status);
      if (val?.heartbeat) onHeartbeat && onHeartbeat(val.heartbeat);
    }
  }, () => {});

  return () => { off(resultRef); off(queryRef); };
}

/**
 * Listen in real-time for a CNN result at /cnn_results/{queryId}.
 * Returns an unsubscribe function.
 */
export function listenForCNNResult(queryId, { onResult, onStatusChange, onError }) {
  if (!db || !queryId) return () => {};

  const resultRef = dbRef(db, `cnn_results/${queryId}`);
  const queryRef  = dbRef(db, `queries/${queryId}`);

  const unsubResult = onValue(resultRef, (snap) => {
    if (snap.exists()) {
      const data = snap.val();
      console.log(`✅ CNN result received for query ${queryId}`);
      onResult && onResult(data);
    }
  }, (err) => {
    console.warn("CNN result listener error:", err.message);
    onError && onError(err.message);
  });

  const unsubQuery = onValue(queryRef, (snap) => {
    if (snap.exists()) {
      const status = snap.val()?.status;
      onStatusChange && onStatusChange(status);
    }
  }, () => {});

  return () => { off(resultRef); off(queryRef); };
}

/**
 * One-time read of /agent_results/{queryId} (for rehydrating saved sessions).
 */
export async function getAgentResult(queryId) {
  if (!db || !queryId) return null;
  try {
    const snap = await get(dbRef(db, `agent_results/${queryId}`));
    return snap.exists() ? snap.val() : null;
  } catch (e) { return null; }
}

/**
 * One-time read of /cnn_results/{queryId} (for rehydrating saved sessions).
 */
export async function getCNNResult(queryId) {
  if (!db || !queryId) return null;
  try {
    const snap = await get(dbRef(db, `cnn_results/${queryId}`));
    return snap.exists() ? snap.val() : null;
  } catch (e) { return null; }
}

export { auth, db };
