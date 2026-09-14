"""
Firebase Setup & Connection Test
==================================
Run this script ONCE to verify your Firebase connection
and initialize the database structure.

Usage:
    python backend/firebase_setup.py

Prerequisites:
    1. Download your service account key from Firebase Console:
       Firebase Console -> Project Settings -> Service Accounts -> Generate new private key
    2. Save the downloaded JSON as: backend/serviceAccountKey.json
"""

import sys
import os
import json
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Locate service account key (try new name, then legacy) ─────────────────────
_BACKEND_DIR = os.path.dirname(__file__)
_SA_CANDIDATES = [
    os.path.join(_BACKEND_DIR, "satellite-efa0a-firebase-adminsdk-fbsvc-075fca07b4.json"),
    os.path.join(_BACKEND_DIR, "serviceAccountKey.json"),
]
KEY_PATH = next((p for p in _SA_CANDIDATES if os.path.exists(p)), None)

if not KEY_PATH:
    print("=" * 60)
    print("  SETUP REQUIRED: Service Account Key Missing")
    print("=" * 60)
    print()
    print("  Expected one of:")
    for p in _SA_CANDIDATES:
        print(f"    {p}")
    print()
    print("  Download from Firebase Console -> Project Settings -> Service Accounts")
    print("  -> Generate new private key -> save to backend/ folder")
    sys.exit(1)

# ── Initialize Firebase ──
print("=" * 60)
print("  SatQuery AI -- Firebase Setup & Test")
print("=" * 60)

import firebase_admin
from firebase_admin import credentials, db, storage

print("\n[1/4] Initializing Firebase Admin SDK...")
cred = credentials.Certificate(KEY_PATH)
firebase_admin.initialize_app(cred, {
    "databaseURL": "https://satellite-efa0a-default-rtdb.firebaseio.com",
    "storageBucket": "satellite-efa0a.firebasestorage.app",
})
print("  OK - Firebase Admin initialized")

# ── Test Realtime DB ──
print("\n[2/4] Testing Realtime Database connection...")
ref = db.reference("/")
ref.child("_health_check").set({
    "status": "ok",
    "timestamp": int(time.time() * 1000),
    "message": "SatQuery AI backend connected"
})
val = ref.child("_health_check").get()
print(f"  OK - DB write/read successful: {val}")

# ── Initialize DB schema (all paths used by backend) ─────────────────────────
print("\n[3/4] Initializing database schema...")
ref.child("_schema").set({
    "version":      "2.0",
    "description":  "ByteX SatQuery AI -- Full Schema (VQA + Agent + CNN)",
    "project":      "satellite-efa0a",
    "app_id":       "1:504899672780:web:4f61a1b212930752cdc069",
    "paths": {
        "queries":        "/queries/{queryId}     -- all incoming jobs (status: pending->processing->done|error)",
        "results":        "/results/{queryId}     -- VQA / caption / refer answers from Qwen2.5-VL",
        "agent_results":  "/agent_results/{queryId} -- full AI Agent structured reports",
        "cnn_results":    "/cnn_results/{queryId}   -- CNN detection + segmentation results",
        "users":          "/users/{uid}/profile    -- user profile",
        "users_recents":  "/users/{uid}/recents/{id} -- query history per user",
    },
    "task_types": {
        "vqa":     "Qwen2.5-VL Visual Question Answering -> /results/",
        "caption": "Qwen2.5-VL Image Captioning        -> /results/",
        "refer":   "Qwen2.5-VL Referring Expression    -> /results/",
        "agent":   "AI Agent (Route+Plan+Execute+Verify+Merge) -> /agent_results/",
        "cnn":     "YOLOv8 + ResNet/FCN CNN pipeline   -> /cnn_results/",
    },
    "last_updated": int(time.time() * 1000),
})

# Initialize placeholder nodes so the paths appear in the Firebase console
for path in ["queries", "results", "agent_results", "cnn_results", "users"]:
    existing = ref.child(path).get()
    if not existing:
        ref.child(path).child("_placeholder").set({"initialized": True})
        ref.child(path).child("_placeholder").delete()

print("  [OK] Schema initialized (v2.0)")

# ── Test Storage ───────────────────────────────────────────────────────────────
print("\n[4/4] Testing Firebase Storage connection...")
try:
    bucket = storage.bucket()
    print(f"  [OK] Storage bucket: {bucket.name}")
except Exception as e:
    print(f"  [WARN]  Storage test skipped: {e}")

# ── Done ───────────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("  SETUP COMPLETE -- satellite-efa0a")
print("=" * 60)
print()
print("  RTDB URL:  https://satellite-efa0a-default-rtdb.firebaseio.com")
print("  Storage:   satellite-efa0a.firebasestorage.app")
print("  App ID:    1:504899672780:web:4f61a1b212930752cdc069")
print()
print("  Database paths:")
print("    /queries/{id}        -- incoming jobs")
print("    /results/{id}        -- VQA / caption / refer answers")
print("    /agent_results/{id}  -- AI Agent full reports")
print("    /cnn_results/{id}    -- CNN detection + segmentation")
print("    /users/{uid}/        -- user profiles + history")
print()
print("  Start backend:")
print("    uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload")
