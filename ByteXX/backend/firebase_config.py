"""
Firebase Configuration for SatQuery AI Backend
================================================
Uses Firebase Admin SDK (with dynamic service-account key discovery)
AND direct RTDB REST helpers as a fallback to avoid JWT clock-skew errors.

Project:  byte-xx
RTDB URL: https://byte-xx-default-rtdb.firebaseio.com
"""

import glob
import json
import os
import urllib.error
import urllib.request

import firebase_admin
from firebase_admin import credentials, db, storage

# ── Firebase client-side config (matches web app in Firebase Console) ─────────
FIREBASE_CONFIG = {

}

# ── RTDB base URL (for REST fallback) ─────────────────────────────────────────
_RTDB_BASE = FIREBASE_CONFIG["databaseURL"]

# ── Service-account key — auto-discover any adminsdk JSON in backend dir ───────
_BACKEND_DIR = os.path.dirname(__file__)

_SA_CANDIDATES = [
    # Auto-glob any Firebase Admin SDK key in backend/
    *glob.glob(os.path.join(_BACKEND_DIR, "*firebase-adminsdk*.json")),
    os.path.join(_BACKEND_DIR, "serviceAccountKey.json"),
]

SERVICE_ACCOUNT_PATH: str = next(
    (p for p in _SA_CANDIDATES if os.path.exists(p)),
    os.path.join(_BACKEND_DIR, "serviceAccountKey.json"),   # default placeholder
)

# ── Realtime DB paths ──────────────────────────────────────────────────────────
DB_QUERIES_PATH    = "queries"        # /queries/{id}/
DB_RESULTS_PATH    = "results"        # /results/{id}/
DB_AGENT_PATH      = "agent_results"  # /agent_results/{id}/
DB_CNN_PATH        = "cnn_results"    # /cnn_results/{id}/
DB_USERS_PATH      = "users"          # /users/{uid}/

# ── Storage ───────────────────────────────────────────────────────────────────
STORAGE_IMAGES_FOLDER = "uploads"

_initialized = False
_use_rest_fallback = False   # True when Admin SDK JWT fails


# ─────────────────────────────────────────────
# Admin SDK initialisation
# ─────────────────────────────────────────────
def init_firebase() -> bool:
    """
    Initialize Firebase Admin SDK (call once at startup).
    Returns True if successfully initialized, False otherwise.
    Falls back to direct REST reads/writes if the SDK JWT is broken.
    """
    global _initialized, _use_rest_fallback

    if _initialized:
        return True

    if not os.path.exists(SERVICE_ACCOUNT_PATH):
        print("  [WARN] Firebase service account key NOT found at:")
        for p in _SA_CANDIDATES:
            print(f"         {p}")
        print("  Falling back to direct RTDB REST API (public read/write).")
        _use_rest_fallback = True
        _initialized = True   # mark as "initialized" so listener starts
        return True

    try:
        # Avoid double-init when running in --reload mode
        if firebase_admin._apps:
            _initialized = True
            return True

        cred = credentials.Certificate(SERVICE_ACCOUNT_PATH)
        firebase_admin.initialize_app(cred, {
            "databaseURL":   FIREBASE_CONFIG["databaseURL"],
            "storageBucket": FIREBASE_CONFIG["storageBucket"],
        })
        _initialized = True
        sa_name = os.path.basename(SERVICE_ACCOUNT_PATH)
        print(f"  [OK] Firebase Admin SDK initialized — key: {sa_name}")
        print(f"       RTDB: {FIREBASE_CONFIG['databaseURL']}")
        return True
    except Exception as e:
        print(f"  [WARN] Firebase Admin SDK init failed: {e}")
        print("  Falling back to direct RTDB REST API.")
        _use_rest_fallback = True
        _initialized = True
        return True


def is_initialized() -> bool:
    """Return True if Firebase backend is ready (SDK or REST fallback)."""
    return _initialized


def uses_rest_fallback() -> bool:
    return _use_rest_fallback


def get_db() -> db.Reference:
    """Get Realtime Database root reference (Admin SDK path)."""
    return db.reference("/")


def get_bucket():
    """Get Firebase Storage bucket."""
    return storage.bucket()


# ─────────────────────────────────────────────
# Direct RTDB REST helpers
# (used when Admin SDK JWT fails, or as primary when no SA key)
# ─────────────────────────────────────────────
def _rtdb_url(path: str) -> str:
    path = path.strip("/")
    return f"{_RTDB_BASE}/{path}.json"


def rtdb_get(path: str):
    """Read a node from RTDB via REST. Returns parsed JSON or None."""
    try:
        url = _rtdb_url(path)
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = resp.read().decode("utf-8")
            return json.loads(data)
    except Exception as e:
        print(f"  [RTDB-REST] GET {path} failed: {e}")
        return None


def rtdb_set(path: str, value) -> bool:
    """Write (PUT) a value to a RTDB path via REST. Returns True on success."""
    try:
        url = _rtdb_url(path)
        data = json.dumps(value).encode("utf-8")
        req = urllib.request.Request(
            url, data=data,
            headers={"Content-Type": "application/json"},
            method="PUT",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            resp.read()
        return True
    except Exception as e:
        print(f"  [RTDB-REST] SET {path} failed: {e}")
        return False


def rtdb_update(path: str, value: dict) -> bool:
    """PATCH (merge-update) a RTDB node via REST. Returns True on success."""
    try:
        url = _rtdb_url(path)
        data = json.dumps(value).encode("utf-8")
        req = urllib.request.Request(
            url, data=data,
            headers={"Content-Type": "application/json"},
            method="PATCH",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            resp.read()
        return True
    except Exception as e:
        print(f"  [RTDB-REST] UPDATE {path} failed: {e}")
        return False


def rtdb_delete(path: str) -> bool:
    """DELETE a RTDB node via REST. Returns True on success."""
    try:
        url = _rtdb_url(path)
        req = urllib.request.Request(url, method="DELETE")
        with urllib.request.urlopen(req, timeout=10) as resp:
            resp.read()
        return True
    except Exception as e:
        print(f"  [RTDB-REST] DELETE {path} failed: {e}")
        return False


# ─────────────────────────────────────────────
# Unified helpers (auto-select SDK vs REST)
# ─────────────────────────────────────────────
def db_get(path: str):
    """Read from RTDB — SDK first, REST fallback."""
    if not _use_rest_fallback:
        try:
            return db.reference(path).get()
        except Exception as e:
            err = str(e)
            if "invalid_grant" in err or "JWT" in err:
                print(f"  [RTDB] JWT error on GET {path} — switching to REST")
                _set_rest_fallback()
            else:
                raise
    return rtdb_get(path)


def db_set(path: str, value) -> bool:
    """Write to RTDB — SDK first, REST fallback."""
    if not _use_rest_fallback:
        try:
            db.reference(path).set(value)
            return True
        except Exception as e:
            err = str(e)
            if "invalid_grant" in err or "JWT" in err:
                print(f"  [RTDB] JWT error on SET {path} — switching to REST")
                _set_rest_fallback()
            else:
                raise
    return rtdb_set(path, value)


def db_update(path: str, value: dict) -> bool:
    """Patch-update RTDB — SDK first, REST fallback."""
    if not _use_rest_fallback:
        try:
            db.reference(path).update(value)
            return True
        except Exception as e:
            err = str(e)
            if "invalid_grant" in err or "JWT" in err:
                print(f"  [RTDB] JWT error on UPDATE {path} — switching to REST")
                _set_rest_fallback()
            else:
                raise
    return rtdb_update(path, value)


def db_delete(path: str) -> bool:
    """Delete from RTDB — SDK first, REST fallback."""
    if not _use_rest_fallback:
        try:
            db.reference(path).delete()
            return True
        except Exception as e:
            err = str(e)
            if "invalid_grant" in err or "JWT" in err:
                print(f"  [RTDB] JWT error on DELETE {path} — switching to REST")
                _set_rest_fallback()
            else:
                raise
    return rtdb_delete(path)


def _set_rest_fallback():
    global _use_rest_fallback
    _use_rest_fallback = True
    print("  [RTDB] Permanently switched to direct REST fallback for this session.")
