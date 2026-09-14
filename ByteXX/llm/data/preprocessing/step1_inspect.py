"""
Step 1 — Extraction & Inspection
==================================
- Extract Annotations_val.zip (images are already extracted)
- Load all JSON annotation files
- Print structure, sample entries, and counts per task per split
- Compare per-image annotations (from zip) vs standalone eval JSONs
"""

import json
import os
import sys
import zipfile
from collections import Counter
from pprint import pformat

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from data.preprocessing.config import (
    DATASET_DIR,
    TRAIN_JSON,
    EVAL_CAP_JSON,
    EVAL_REFER_JSON,
    EVAL_VQA_JSON,
    ANNOTATIONS_VAL_ZIP,
    ANNOTATIONS_TRAIN_ZIP,
    TASK_TAG_CAPTION,
    TASK_TAG_REFER,
    TASK_TAG_VQA,
)


def extract_annotations():
    """Extract annotation zips if not already extracted."""
    print("─── Extraction ───")

    # Annotations_val.zip
    if os.path.exists(ANNOTATIONS_VAL_ZIP):
        extract_dir = os.path.join(DATASET_DIR, "Annotations_val")
        if os.path.isdir(extract_dir) and len(os.listdir(extract_dir)) > 0:
            print(f"  ✓ Annotations_val/ already extracted ({len(os.listdir(extract_dir))} files)")
        else:
            print(f"  Extracting {ANNOTATIONS_VAL_ZIP}...")
            with zipfile.ZipFile(ANNOTATIONS_VAL_ZIP, "r") as z:
                # Filter out __MACOSX junk
                members = [m for m in z.namelist() if "__MACOSX" not in m]
                z.extractall(DATASET_DIR, members)
            n_files = len([f for f in os.listdir(extract_dir) if f.endswith(".json")])
            print(f"  ✓ Extracted {n_files} annotation files to Annotations_val/")
    else:
        print(f"  ⚠ Annotations_val.zip not found at {ANNOTATIONS_VAL_ZIP}")

    # Annotations_train.zip — expected to be missing
    if os.path.exists(ANNOTATIONS_TRAIN_ZIP):
        print(f"  Found Annotations_train.zip — extracting...")
        with zipfile.ZipFile(ANNOTATIONS_TRAIN_ZIP, "r") as z:
            members = [m for m in z.namelist() if "__MACOSX" not in m]
            z.extractall(DATASET_DIR, members)
        print(f"  ✓ Extracted Annotations_train/")
    else:
        print(f"  ℹ Annotations_train.zip not present (train data is in VRSBench_train.json)")


def load_json(path: str) -> list | dict:
    """Load a JSON file with UTF-8 encoding."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def inspect_train_json():
    """Load and inspect VRSBench_train.json — the unified training annotations."""
    print("\n─── VRSBench_train.json ───")

    if not os.path.exists(TRAIN_JSON):
        print(f"  ✗ NOT FOUND: {TRAIN_JSON}")
        return None

    data = load_json(TRAIN_JSON)
    print(f"  Type: {type(data).__name__}")
    print(f"  Total entries: {len(data):,}")
    print(f"  Entry keys: {list(data[0].keys())}")

    # Classify by task tag
    task_counts = Counter()
    for entry in data:
        q_text = entry["conversations"][0]["value"].lower()
        if TASK_TAG_VQA in q_text:
            task_counts["vqa"] += 1
        elif TASK_TAG_CAPTION in q_text:
            task_counts["caption"] += 1
        elif TASK_TAG_REFER in q_text:
            task_counts["refer"] += 1
        else:
            task_counts["unknown"] += 1

    print(f"\n  Task breakdown (train):")
    for task, count in task_counts.most_common():
        print(f"    {task:>10s}: {count:>7,}")

    # Show one sample per task type
    shown = set()
    for entry in data:
        q_text = entry["conversations"][0]["value"].lower()
        for tag_name, tag in [("caption", TASK_TAG_CAPTION), ("refer", TASK_TAG_REFER), ("vqa", TASK_TAG_VQA)]:
            if tag in q_text and tag_name not in shown:
                shown.add(tag_name)
                print(f"\n  Sample [{tag_name}]:")
                print(f"    image: {entry['image']}")
                print(f"    human: {entry['conversations'][0]['value'][:120]}...")
                print(f"    gpt:   {entry['conversations'][1]['value'][:120]}")
        if len(shown) == 3:
            break

    return data


def inspect_eval_json(path: str, label: str):
    """Load and inspect one eval JSON file."""
    print(f"\n─── {label} ───")

    if not os.path.exists(path):
        print(f"  ✗ NOT FOUND: {path}")
        return None

    data = load_json(path)
    print(f"  Type: {type(data).__name__}")
    print(f"  Total entries: {len(data):,}")
    print(f"  Entry keys: {list(data[0].keys())}")

    # Show 1 sample
    print(f"\n  Sample entry:")
    sample = data[0]
    for k, v in sample.items():
        v_str = str(v)
        if len(v_str) > 100:
            v_str = v_str[:100] + "..."
        print(f"    {k}: {v_str}")

    # Count unique images
    img_key = "image_id" if "image_id" in data[0] else "image"
    unique_images = set(d[img_key] for d in data)
    print(f"\n  Unique images: {len(unique_images):,}")

    # Type distribution if available
    if "type" in data[0]:
        type_counts = Counter(d["type"] for d in data)
        print(f"  Type distribution:")
        for t, c in type_counts.most_common():
            print(f"    {t:>25s}: {c:>6,}")

    return data


def compare_annotations_zip_vs_eval():
    """
    Compare per-image annotations from Annotations_val/ with the
    standalone eval JSONs to understand overlap/differences.
    """
    print("\n─── Annotations_val/ vs Standalone Eval JSONs ───")

    ann_dir = os.path.join(DATASET_DIR, "Annotations_val")
    if not os.path.isdir(ann_dir):
        print("  ⚠ Annotations_val/ directory not found — skipping comparison")
        return

    # Load one per-image annotation
    json_files = [f for f in os.listdir(ann_dir) if f.endswith(".json")]
    print(f"  Per-image annotation files: {len(json_files):,}")

    if json_files:
        sample_path = os.path.join(ann_dir, json_files[0])
        with open(sample_path, "r", encoding="utf-8") as f:
            sample = json.load(f)
        print(f"\n  Sample per-image annotation ({json_files[0]}):")
        print(f"    Top-level keys: {list(sample.keys())}")
        if "objects" in sample and sample["objects"]:
            print(f"    objects[0] keys: {list(sample['objects'][0].keys())}")
            print(f"    objects[0].obj_coord: {sample['objects'][0].get('obj_coord')}")
            print(f"    objects[0].obj_corner: {sample['objects'][0].get('obj_corner')}")
        if "qa_pairs" in sample and sample["qa_pairs"]:
            print(f"    qa_pairs[0] keys: {list(sample['qa_pairs'][0].keys())}")
        if "caption" in sample:
            cap = sample["caption"]
            print(f"    caption: {cap[:100]}...")

    # Compare: each per-image JSON ↔ the eval JSONs are just flattened versions
    print(f"\n  Relationship:")
    print(f"    - Annotations_val/*.json: Rich per-image data (caption + objects + qa_pairs)")
    print(f"    - VRSBench_EVAL_Cap.json: Flattened captions from the same images")
    print(f"    - VRSBench_EVAL_referring.json: Flattened referring expressions with bbox")
    print(f"    - VRSBench_EVAL_vqa.json: Flattened QA pairs")
    print(f"    → The standalone JSONs are derived from the per-image annotations.")
    print(f"    → For task-specific loading, the standalone JSONs are simpler to use.")


def run_inspection():
    """Run the full inspection pipeline."""
    print("=" * 70)
    print("  Step 1: Extraction & Inspection")
    print("=" * 70)

    extract_annotations()
    inspect_train_json()
    inspect_eval_json(EVAL_CAP_JSON, "VRSBench_EVAL_Cap.json (Captioning)")
    inspect_eval_json(EVAL_REFER_JSON, "VRSBench_EVAL_referring.json (Referring)")
    inspect_eval_json(EVAL_VQA_JSON, "VRSBench_EVAL_vqa.json (VQA)")
    compare_annotations_zip_vs_eval()

    print("\n  ✓ Inspection complete.\n")


if __name__ == "__main__":
    run_inspection()
