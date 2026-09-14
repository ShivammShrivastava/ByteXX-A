"""
SatQuery AI — Preprocess VRSBench into Qwen2.5-VL Training Format
==================================================================
Converts raw VRSBench annotations into JSONL files ready for
Qwen2.5-VL fine-tuning on Google Colab.

Output:
  - train_vqa.jsonl       (VQA samples in chat format)
  - train_caption.jsonl   (Caption samples in chat format)
  - train_combined.jsonl  (VQA + Caption combined)
  - eval_vqa.jsonl        (Eval samples for testing)
  - dataset_stats.json    (Statistics summary)
"""

import json
import os
import sys
import re
from pathlib import Path
from collections import Counter

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
DATASET_DIR = r"C:\Users\Suryansh\OneDrive\Desktop\dataset"
OUTPUT_DIR = r"C:\Users\Suryansh\OneDrive\Desktop\Satellite\data\processed"

RS_SYSTEM_PROMPT = (
    "You are SatQuery AI, a remote sensing image analysis expert specializing in "
    "satellite and aerial imagery interpretation. You can identify land cover types, "
    "objects, spatial relationships, and scene characteristics in remote sensing images. "
    "Provide accurate, concise answers based on visual evidence in the image."
)


def clean_question(text: str) -> str:
    """Remove task tags and <image> tokens, clean up the question."""
    cleaned = text.replace("<image>", "").strip()
    cleaned = re.sub(r"\[(vqa|caption|refer)\]\s*", "", cleaned, flags=re.IGNORECASE)
    # Remove leading/trailing whitespace and newlines
    cleaned = cleaned.strip().strip("\n")
    return cleaned


def resolve_image_path(image_name: str, split: str = "train") -> str:
    """Resolve the full image path."""
    if split == "train":
        # Double nested structure
        path = os.path.join(DATASET_DIR, "Images_train", "Images_train", image_name)
        if os.path.exists(path):
            return path
        path = os.path.join(DATASET_DIR, "Images_train", image_name)
        if os.path.exists(path):
            return path
    else:
        path = os.path.join(DATASET_DIR, "Images_val", image_name)
        if os.path.exists(path):
            return path
    return None


def convert_vqa_sample(annotation: dict, image_path: str) -> dict:
    """Convert a VQA annotation to Qwen2.5-VL chat format."""
    conv = annotation["conversations"]
    question = clean_question(conv[0]["value"])
    answer = conv[1]["value"].strip()

    return {
        "messages": [
            {
                "role": "system",
                "content": RS_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": f"<image>{image_path}</image>\n{question}"
            },
            {
                "role": "assistant",
                "content": answer
            }
        ],
        "image": annotation["image"],
        "image_path": image_path,
        "task": "vqa",
        "question": question,
        "answer": answer
    }


def convert_caption_sample(annotation: dict, image_path: str) -> dict:
    """Convert a caption annotation to Qwen2.5-VL chat format."""
    conv = annotation["conversations"]
    prompt = clean_question(conv[0]["value"])
    caption = conv[1]["value"].strip()

    return {
        "messages": [
            {
                "role": "system",
                "content": RS_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": f"<image>{image_path}</image>\n{prompt}"
            },
            {
                "role": "assistant",
                "content": caption
            }
        ],
        "image": annotation["image"],
        "image_path": image_path,
        "task": "caption",
        "question": prompt,
        "answer": caption
    }


def convert_eval_sample(sample: dict, image_path: str) -> dict:
    """Convert an eval annotation to inference format."""
    return {
        "messages": [
            {
                "role": "system",
                "content": RS_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": f"<image>{image_path}</image>\n{sample['question']}"
            }
        ],
        "image": sample["image_id"],
        "image_path": image_path,
        "question_id": sample.get("question_id", -1),
        "question": sample["question"],
        "ground_truth": sample["ground_truth"],
        "type": sample.get("type", "unknown")
    }


def save_jsonl(data: list, filepath: str):
    """Save list of dicts as JSONL."""
    with open(filepath, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main():
    print("=" * 70)
    print("  SatQuery AI — Data Preprocessing")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Load training annotations ──
    print("\n[1/5] Loading training annotations...")
    train_path = os.path.join(DATASET_DIR, "VRSBench_train.json")
    with open(train_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    print(f"  Loaded {len(train_data):,} raw annotations")

    # ── Process VQA samples ──
    print("\n[2/5] Processing VQA samples...")
    vqa_samples = []
    vqa_skipped = 0

    for ann in train_data:
        conv_text = ann["conversations"][0]["value"].lower()
        if "[vqa]" not in conv_text:
            continue

        image_path = resolve_image_path(ann["image"], "train")
        if image_path is None:
            vqa_skipped += 1
            continue

        sample = convert_vqa_sample(ann, image_path)
        vqa_samples.append(sample)

    vqa_path = os.path.join(OUTPUT_DIR, "train_vqa.jsonl")
    save_jsonl(vqa_samples, vqa_path)
    print(f"  VQA samples: {len(vqa_samples):,} (skipped {vqa_skipped})")
    print(f"  Saved: {vqa_path}")

    # ── Process Caption samples ──
    print("\n[3/5] Processing Caption samples...")
    caption_samples = []
    cap_skipped = 0

    for ann in train_data:
        conv_text = ann["conversations"][0]["value"].lower()
        if "[caption]" not in conv_text:
            continue

        image_path = resolve_image_path(ann["image"], "train")
        if image_path is None:
            cap_skipped += 1
            continue

        sample = convert_caption_sample(ann, image_path)
        caption_samples.append(sample)

    caption_path = os.path.join(OUTPUT_DIR, "train_caption.jsonl")
    save_jsonl(caption_samples, caption_path)
    print(f"  Caption samples: {len(caption_samples):,} (skipped {cap_skipped})")
    print(f"  Saved: {caption_path}")

    # ── Combined dataset ──
    print("\n[4/5] Creating combined dataset...")
    combined = vqa_samples + caption_samples
    # Shuffle for better training
    import random
    random.seed(42)
    random.shuffle(combined)

    combined_path = os.path.join(OUTPUT_DIR, "train_combined.jsonl")
    save_jsonl(combined, combined_path)
    print(f"  Combined samples: {len(combined):,}")
    print(f"  Saved: {combined_path}")

    # ── Process Eval VQA samples ──
    print("\n[5/5] Processing Eval VQA samples...")
    eval_path = os.path.join(DATASET_DIR, "VRSBench_EVAL_vqa.json")
    with open(eval_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    eval_samples = []
    eval_skipped = 0

    for sample in eval_data:
        image_path = resolve_image_path(sample["image_id"], "val")
        if image_path is None:
            eval_skipped += 1
            continue
        eval_samples.append(convert_eval_sample(sample, image_path))

    eval_out_path = os.path.join(OUTPUT_DIR, "eval_vqa.jsonl")
    save_jsonl(eval_samples, eval_out_path)
    print(f"  Eval samples: {len(eval_samples):,} (skipped {eval_skipped})")
    print(f"  Saved: {eval_out_path}")

    # ── Statistics ──
    stats = {
        "train_vqa_count": len(vqa_samples),
        "train_caption_count": len(caption_samples),
        "train_combined_count": len(combined),
        "eval_vqa_count": len(eval_samples),
        "vqa_skipped": vqa_skipped,
        "caption_skipped": cap_skipped,
        "eval_skipped": eval_skipped,
        "unique_train_images": len(set(s["image"] for s in combined)),
        "unique_eval_images": len(set(s["image"] for s in eval_samples)),
        "vqa_answer_distribution": dict(Counter(
            s["answer"].lower() for s in vqa_samples
        ).most_common(20)),
        "eval_type_distribution": dict(Counter(
            s["type"] for s in eval_samples
        )),
    }

    stats_path = os.path.join(OUTPUT_DIR, "dataset_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    # ── Print sample ──
    print("\n" + "=" * 70)
    print("  SAMPLE OUTPUT (first VQA sample)")
    print("=" * 70)
    sample = vqa_samples[0]
    print(f"  System: {sample['messages'][0]['content'][:80]}...")
    print(f"  User:   {sample['question']}")
    print(f"  Answer: {sample['answer']}")
    print(f"  Image:  {sample['image']}")

    print("\n" + "=" * 70)
    print("  SAMPLE OUTPUT (first Caption sample)")
    print("=" * 70)
    sample = caption_samples[0]
    print(f"  Prompt:  {sample['question'][:80]}...")
    print(f"  Caption: {sample['answer'][:120]}...")

    # ── File sizes ──
    print("\n" + "=" * 70)
    print("  OUTPUT FILES")
    print("=" * 70)
    for fname in ["train_vqa.jsonl", "train_caption.jsonl", "train_combined.jsonl",
                   "eval_vqa.jsonl", "dataset_stats.json"]:
        fpath = os.path.join(OUTPUT_DIR, fname)
        size_mb = os.path.getsize(fpath) / (1024 * 1024)
        print(f"  {fname:<30s} {size_mb:>8.1f} MB")

    total_size = sum(
        os.path.getsize(os.path.join(OUTPUT_DIR, f))
        for f in os.listdir(OUTPUT_DIR)
    ) / (1024 * 1024)
    print(f"  {'TOTAL':<30s} {total_size:>8.1f} MB")

    print(f"\n  All processed files saved to: {OUTPUT_DIR}")
    print(f"\n  NEXT STEP: Upload 'train_combined.jsonl' to Google Colab")
    print(f"  or upload to HuggingFace as a dataset for cloud access.")
    print("=" * 70)


if __name__ == "__main__":
    main()
