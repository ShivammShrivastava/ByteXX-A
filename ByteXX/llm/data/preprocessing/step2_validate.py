"""
Step 2 — Image-Annotation Matching Validation
===============================================
For each JSON annotation file, verify that every referenced image
actually exists on disk. Report missing files explicitly.
"""

import json
import os
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from data.preprocessing.config import (
    TRAIN_JSON,
    EVAL_CAP_JSON,
    EVAL_REFER_JSON,
    EVAL_VQA_JSON,
    TASK_TAG_CAPTION,
    TASK_TAG_REFER,
    TASK_TAG_VQA,
    resolve_train_image_path,
    resolve_val_image_path,
)


def _load_json(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_train_images(max_report: int = 10):
    """
    Check that every image referenced in VRSBench_train.json exists.
    Reports per-task and overall counts.
    """
    print("─── Train Image Validation ───")

    if not os.path.exists(TRAIN_JSON):
        print(f"  ✗ {TRAIN_JSON} not found")
        return

    data = _load_json(TRAIN_JSON)

    # Collect unique image names per task
    task_images = {"caption": set(), "refer": set(), "vqa": set(), "unknown": set()}
    for entry in data:
        q = entry["conversations"][0]["value"].lower()
        img = entry["image"]
        if TASK_TAG_CAPTION in q:
            task_images["caption"].add(img)
        elif TASK_TAG_REFER in q:
            task_images["refer"].add(img)
        elif TASK_TAG_VQA in q:
            task_images["vqa"].add(img)
        else:
            task_images["unknown"].add(img)

    all_images = set()
    for imgs in task_images.values():
        all_images.update(imgs)

    print(f"  Unique images referenced: {len(all_images):,}")
    for task, imgs in task_images.items():
        if imgs:
            print(f"    {task:>10s}: {len(imgs):,} unique images")

    # Check existence
    missing = []
    found = 0
    for img_name in sorted(all_images):
        path = resolve_train_image_path(img_name)
        if path is None:
            missing.append(img_name)
        else:
            found += 1

    print(f"\n  Result: {found:,} found, {len(missing):,} missing")
    if missing:
        print(f"  ⚠ Missing images (showing first {min(len(missing), max_report)}):")
        for img in missing[:max_report]:
            print(f"    ✗ {img}")
        if len(missing) > max_report:
            print(f"    ... and {len(missing) - max_report} more")
    else:
        print(f"  ✓ All training images found!")

    return {"found": found, "missing": len(missing), "missing_files": missing}


def validate_eval_images(json_path: str, label: str, max_report: int = 10):
    """
    Check that every image referenced in an eval JSON exists in Images_val/.
    """
    print(f"\n─── {label} Image Validation ───")

    if not os.path.exists(json_path):
        print(f"  ✗ {json_path} not found")
        return

    data = _load_json(json_path)
    img_key = "image_id" if "image_id" in data[0] else "image"
    unique_images = set(d[img_key] for d in data)

    print(f"  Unique images referenced: {len(unique_images):,}")

    missing = []
    found = 0
    for img_name in sorted(unique_images):
        path = resolve_val_image_path(img_name)
        if path is None:
            missing.append(img_name)
        else:
            found += 1

    print(f"  Result: {found:,} found, {len(missing):,} missing")
    if missing:
        print(f"  ⚠ Missing images (showing first {min(len(missing), max_report)}):")
        for img in missing[:max_report]:
            print(f"    ✗ {img}")
        if len(missing) > max_report:
            print(f"    ... and {len(missing) - max_report} more")
    else:
        print(f"  ✓ All images found!")

    return {"found": found, "missing": len(missing), "missing_files": missing}


def run_validation():
    """Run full image-annotation validation for all splits and tasks."""
    print("=" * 70)
    print("  Step 2: Image-Annotation Matching Validation")
    print("=" * 70)

    results = {}
    results["train"] = validate_train_images()
    results["eval_cap"] = validate_eval_images(EVAL_CAP_JSON, "Eval Captioning")
    results["eval_refer"] = validate_eval_images(EVAL_REFER_JSON, "Eval Referring")
    results["eval_vqa"] = validate_eval_images(EVAL_VQA_JSON, "Eval VQA")

    # Summary
    total_missing = sum(
        r["missing"] for r in results.values() if r is not None
    )
    print(f"\n─── Summary ───")
    if total_missing == 0:
        print(f"  ✓ All image references validated — zero missing files across all splits!")
    else:
        print(f"  ⚠ Total missing images across all splits: {total_missing}")

    print()
    return results


if __name__ == "__main__":
    run_validation()
