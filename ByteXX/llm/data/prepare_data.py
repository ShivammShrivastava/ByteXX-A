"""
SatQuery AI — VRSBench Data Preparation & Verification
=======================================================
Run this locally (CPU only, no GPU needed) to:
  1. Validate the VRSBench dataset integrity
  2. Preview sample image–question–answer triplets
  3. Report dataset statistics (question types, answer lengths)
  4. Verify all referenced images exist on disk

Usage:
    python data/prepare_data.py --dataset-dir "C:/Users/Suryansh/OneDrive/Desktop/dataset"
"""

import json
import os
import sys
import argparse
from pathlib import Path
from collections import Counter

# Fix Windows console encoding for Unicode output
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
DEFAULT_DATASET_DIR = r"C:\Users\Suryansh\OneDrive\Desktop\dataset"

TASK_TAGS = {
    "vqa": "[vqa]",
    "caption": "[caption]",
    "refer": "[refer]",
}


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def load_json(path: str) -> list | dict:
    """Load a JSON file and return its contents."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_train_image_path(dataset_dir: str, image_name: str) -> str:
    """
    Resolve training image path.
    VRSBench has a double-nested structure: Images_train/Images_train/<name>.
    We check both the nested and flat paths.
    """
    # Try double-nested first (observed in this dataset)
    nested = os.path.join(dataset_dir, "Images_train", "Images_train", image_name)
    if os.path.exists(nested):
        return nested
    # Try flat
    flat = os.path.join(dataset_dir, "Images_train", image_name)
    if os.path.exists(flat):
        return flat
    return None


def resolve_val_image_path(dataset_dir: str, image_name: str) -> str:
    """Resolve validation image path."""
    path = os.path.join(dataset_dir, "Images_val", image_name)
    if os.path.exists(path):
        return path
    return None


def classify_sample(conversation_text: str) -> str:
    """Classify a training sample by its task tag."""
    text_lower = conversation_text.lower()
    for task, tag in TASK_TAGS.items():
        if tag in text_lower:
            return task
    return "unknown"


def clean_question(text: str) -> str:
    """Remove task tags and <image> tokens from question text."""
    cleaned = text.replace("<image>", "").strip()
    for tag in TASK_TAGS.values():
        cleaned = cleaned.replace(tag, "").replace(tag.upper(), "")
    # Also handle case-insensitive
    import re
    cleaned = re.sub(r"\[(vqa|caption|refer)\]\s*", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


# ─────────────────────────────────────────────
# Main Verification
# ─────────────────────────────────────────────
def verify_dataset(dataset_dir: str):
    """Run full dataset verification and print report."""

    print("=" * 70)
    print("  SatQuery AI — VRSBench Dataset Verification")
    print("=" * 70)
    print(f"\nDataset directory: {dataset_dir}\n")

    # ── Check required files ──
    required_files = [
        "VRSBench_train.json",
        "VRSBench_EVAL_vqa.json",
        "VRSBench_EVAL_Cap.json",
        "VRSBench_EVAL_referring.json",
    ]
    required_dirs = ["Images_train", "Images_val"]

    print("─── File Checks ───")
    all_ok = True
    for f in required_files:
        path = os.path.join(dataset_dir, f)
        exists = os.path.exists(path)
        status = "✓" if exists else "✗ MISSING"
        size_str = ""
        if exists:
            size_mb = os.path.getsize(path) / (1024 * 1024)
            size_str = f" ({size_mb:.1f} MB)"
        print(f"  {status} {f}{size_str}")
        if not exists:
            all_ok = False

    for d in required_dirs:
        path = os.path.join(dataset_dir, d)
        exists = os.path.isdir(path)
        status = "✓" if exists else "✗ MISSING"
        print(f"  {status} {d}/")
        if not exists:
            all_ok = False

    if not all_ok:
        print("\n⚠ Some required files/dirs are missing. Continuing with available data...\n")

    # ── Load training data ──
    print("\n─── Training Data Analysis ───")
    train_path = os.path.join(dataset_dir, "VRSBench_train.json")
    if not os.path.exists(train_path):
        print("  ✗ Cannot load training data — VRSBench_train.json not found")
        return

    train_data = load_json(train_path)
    print(f"  Total training samples: {len(train_data):,}")

    # Classify by task
    task_counts = Counter()
    vqa_samples = []
    caption_samples = []

    for sample in train_data:
        conv = sample["conversations"]
        if len(conv) < 2:
            continue
        question_text = conv[0].get("value", "")
        task = classify_sample(question_text)
        task_counts[task] += 1

        if task == "vqa":
            vqa_samples.append(sample)
        elif task == "caption":
            caption_samples.append(sample)

    print(f"\n  Task breakdown:")
    for task, count in task_counts.most_common():
        print(f"    {task:>10s}: {count:>6,} samples")

    # Unique images
    unique_images = set(s["image"] for s in train_data)
    print(f"\n  Unique training images: {len(unique_images):,}")

    # ── Validate training image paths ──
    print(f"\n─── Image Path Validation (Training) ───")
    missing_count = 0
    found_count = 0
    checked = set()
    for img_name in unique_images:
        if img_name in checked:
            continue
        checked.add(img_name)
        path = resolve_train_image_path(dataset_dir, img_name)
        if path:
            found_count += 1
        else:
            missing_count += 1
            if missing_count <= 5:
                print(f"  ✗ Missing: {img_name}")

    print(f"  Found: {found_count:,} / {len(unique_images):,} images")
    if missing_count > 0:
        print(f"  ⚠ Missing: {missing_count:,} images")
    else:
        print(f"  ✓ All training images found!")

    # ── VQA Analysis ──
    print(f"\n─── VQA Samples Analysis ───")
    print(f"  Total VQA training samples: {len(vqa_samples):,}")

    # Answer length distribution
    answer_lengths = []
    for s in vqa_samples:
        answer = s["conversations"][1]["value"]
        answer_lengths.append(len(answer.split()))

    if answer_lengths:
        avg_len = sum(answer_lengths) / len(answer_lengths)
        max_len = max(answer_lengths)
        min_len = min(answer_lengths)
        print(f"  Answer length (words): min={min_len}, avg={avg_len:.1f}, max={max_len}")

    # Common answers
    answer_counter = Counter(s["conversations"][1]["value"].lower() for s in vqa_samples)
    print(f"  Unique answers: {len(answer_counter):,}")
    print(f"  Top 10 most common answers:")
    for ans, count in answer_counter.most_common(10):
        print(f"    '{ans}': {count:,}")

    # ── Caption Analysis ──
    print(f"\n─── Caption Samples Analysis ───")
    print(f"  Total caption training samples: {len(caption_samples):,}")

    if caption_samples:
        cap_lengths = [len(s["conversations"][1]["value"].split()) for s in caption_samples]
        avg_cap = sum(cap_lengths) / len(cap_lengths)
        print(f"  Caption length (words): min={min(cap_lengths)}, avg={avg_cap:.1f}, max={max(cap_lengths)}")

    # ── Eval Data ──
    print(f"\n─── Evaluation Data Analysis ───")
    eval_vqa_path = os.path.join(dataset_dir, "VRSBench_EVAL_vqa.json")
    if os.path.exists(eval_vqa_path):
        eval_data = load_json(eval_vqa_path)
        print(f"  VQA eval samples: {len(eval_data):,}")

        # Question types
        type_counts = Counter(d.get("type", "unknown") for d in eval_data)
        print(f"  Question type distribution:")
        for qtype, count in type_counts.most_common():
            print(f"    {qtype:>20s}: {count:>5,}")

        # Unique eval images
        eval_images = set(d["image_id"] for d in eval_data)
        print(f"  Unique eval images: {len(eval_images):,}")

        # Validate eval image paths
        eval_missing = 0
        for img_name in eval_images:
            path = resolve_val_image_path(dataset_dir, img_name)
            if not path:
                eval_missing += 1
        if eval_missing > 0:
            print(f"  ⚠ Missing eval images: {eval_missing:,}")
        else:
            print(f"  ✓ All eval images found!")

    # ── Preview Samples ──
    print(f"\n─── Sample VQA Pairs (Training) ───")
    for i, sample in enumerate(vqa_samples[:5]):
        question = clean_question(sample["conversations"][0]["value"])
        answer = sample["conversations"][1]["value"]
        image = sample["image"]
        img_path = resolve_train_image_path(dataset_dir, image)
        img_status = "✓" if img_path else "✗"
        print(f"\n  Sample {i+1}:")
        print(f"    Image: {image} [{img_status}]")
        print(f"    Q: {question}")
        print(f"    A: {answer}")

    print(f"\n─── Sample Caption Pairs (Training) ───")
    for i, sample in enumerate(caption_samples[:3]):
        question = clean_question(sample["conversations"][0]["value"])
        answer = sample["conversations"][1]["value"]
        image = sample["image"]
        print(f"\n  Sample {i+1}:")
        print(f"    Image: {image}")
        print(f"    Prompt: {question}")
        print(f"    Caption: {answer[:150]}...")

    # ── Summary ──
    print(f"\n{'=' * 70}")
    print(f"  SUMMARY")
    print(f"{'=' * 70}")
    print(f"  VQA training samples:     {len(vqa_samples):>7,}")
    print(f"  Caption training samples: {len(caption_samples):>7,}")
    print(f"  Combined for fine-tuning: {len(vqa_samples) + len(caption_samples):>7,}")
    print(f"  VQA eval samples:         {len(eval_data) if os.path.exists(eval_vqa_path) else 'N/A':>7,}")
    print(f"  Training images:          {found_count:>7,} / {len(unique_images):,}")
    print(f"\n  ✓ Dataset ready for Colab fine-tuning!")
    print(f"  Next: Open training/SatQuery_AI_VQA_FineTune.py in Google Colab")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SatQuery AI — VRSBench Data Verification")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=DEFAULT_DATASET_DIR,
        help=f"Path to VRSBench dataset directory (default: {DEFAULT_DATASET_DIR})",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.dataset_dir):
        print(f"ERROR: Dataset directory not found: {args.dataset_dir}")
        sys.exit(1)

    verify_dataset(args.dataset_dir)
