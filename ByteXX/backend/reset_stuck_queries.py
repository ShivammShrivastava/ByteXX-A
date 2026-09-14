"""
Reset Stuck Firebase Queries
==============================
One-off utility: find all queries currently stuck in 'processing'
(i.e. not in-flight on this backend) and reset them to 'error' so
the frontend unblocks and the user can resubmit.

Usage:
    python backend/reset_stuck_queries.py

Optional — only reset queries older than N minutes:
    python backend/reset_stuck_queries.py --min-age 5
"""

import sys
import os
import time
import argparse

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_BACKEND_DIR = os.path.dirname(__file__)
_SA_CANDIDATES = [
    os.path.join(_BACKEND_DIR, "satellite-efa0a-firebase-adminsdk-fbsvc-075fca07b4.json"),
    os.path.join(_BACKEND_DIR, "serviceAccountKey.json"),
]
KEY_PATH = next((p for p in _SA_CANDIDATES if os.path.exists(p)), None)

if not KEY_PATH:
    print("ERROR: No Firebase service account key found in backend/")
    print("Expected one of:")
    for p in _SA_CANDIDATES:
        print(f"  {p}")
    sys.exit(1)

import firebase_admin
from firebase_admin import credentials, db

parser = argparse.ArgumentParser(description="Reset stuck Firebase queries to 'error'.")
parser.add_argument(
    "--min-age", type=int, default=0,
    help="Only reset queries older than this many minutes (default: 0 = all processing queries)"
)
parser.add_argument(
    "--dry-run", action="store_true",
    help="Print what would be reset without actually writing to Firebase"
)
args = parser.parse_args()

print("=" * 60)
print("  SatQuery AI — Reset Stuck Queries")
print("=" * 60)

cred = credentials.Certificate(KEY_PATH)
firebase_admin.initialize_app(cred, {
    "databaseURL": "https://satellite-efa0a-default-rtdb.firebaseio.com",
})
print("  [OK] Firebase initialized")

ref = db.reference("/queries")
all_queries = ref.get() or {}
now_ms = int(time.time() * 1000)
min_age_ms = args.min_age * 60 * 1000

stuck = []
for qid, qdata in all_queries.items():
    if not isinstance(qdata, dict):
        continue
    if qdata.get("status") != "processing":
        continue
    ts = qdata.get("timestamp", now_ms)
    age_ms = now_ms - ts
    if age_ms >= min_age_ms:
        stuck.append((qid, qdata, age_ms))

if not stuck:
    print(f"\n  No stuck queries found (min_age={args.min_age}m).")
    sys.exit(0)

print(f"\n  Found {len(stuck)} stuck query/queries:\n")
for qid, qdata, age_ms in stuck:
    age_s = age_ms // 1000
    age_str = f"{age_s // 60}m {age_s % 60}s" if age_s >= 60 else f"{age_s}s"
    q_preview = str(qdata.get("question", ""))[:60]
    print(f"    {qid[:16]}...  age={age_str}  question='{q_preview}'")

if args.dry_run:
    print("\n  DRY RUN -- no changes written.")
    sys.exit(0)

print()
reset_count = 0
for qid, _, _ in stuck:
    try:
        ref.child(qid).update({
            "status": "error",
            "error": "Manually reset by admin: query was stuck in processing state.",
        })
        print(f"  [RESET] {qid[:16]}...")
        reset_count += 1
    except Exception as e:
        print(f"  [FAIL]  {qid[:16]}... -- {e}")

print(f"\n  Done. {reset_count}/{len(stuck)} queries reset to 'error'.")
print("  Users can now resubmit their queries from the frontend.")
