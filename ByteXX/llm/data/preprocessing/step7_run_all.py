"""
Step 7 — Run All: End-to-End Pipeline Test
=============================================
Orchestrates all preprocessing steps on a small subset:
  1. Inspect dataset structure
  2. Validate image-annotation matching
  3. Create datasets (subset)
  4. Compute channel statistics
  5. Build transforms (train + inference)
  6. Create DataLoaders and test iteration
  7. Visualize samples

Usage:
    python -m data.preprocessing.step7_run_all
    python -m data.preprocessing.step7_run_all --subset 500
    python -m data.preprocessing.step7_run_all --split val
"""

import argparse
import os
import sys
import time

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
from torch.utils.data import DataLoader

from data.preprocessing.config import (
    SUBSET_SIZE,
    BATCH_SIZE,
    NUM_WORKERS,
    ensure_output_dirs,
)
from data.preprocessing.step1_inspect import run_inspection
from data.preprocessing.step2_validate import run_validation
from data.preprocessing.step3_datasets import (
    CaptioningDataset,
    ReferringDataset,
    VQADataset,
)
from data.preprocessing.step4_transforms import (
    compute_channel_stats,
    get_train_transform_captioning,
    get_train_transform_vqa,
    get_train_transform_referring,
)
from data.preprocessing.step5_inference_transforms import get_inference_transform
from data.preprocessing.step6_visualize import (
    visualize_captioning,
    visualize_referring,
    visualize_vqa,
)


def collate_fn(batch: list[dict]) -> dict:
    """
    Custom collate that handles mixed types (tensors + strings + lists).
    Groups same-key items into batches.
    """
    collated = {}
    keys = batch[0].keys()

    for key in keys:
        values = [item[key] for item in batch]

        if isinstance(values[0], torch.Tensor):
            # Stack tensors into a batch
            collated[key] = torch.stack(values)
        elif isinstance(values[0], list):
            # Keep lists of lists (e.g., bbox coords)
            collated[key] = values
        else:
            # Strings and other types: keep as list
            collated[key] = values

    return collated


def test_dataloader(dataset, name: str, batch_size: int = BATCH_SIZE):
    """Test a DataLoader by iterating through one full pass."""
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=False,  # CPU-only for preprocessing test
    )

    print(f"\n  Testing DataLoader for {name}...")
    print(f"    Dataset size: {len(dataset)}")
    print(f"    Batch size: {batch_size}")
    print(f"    Num batches: {len(loader)}")

    t0 = time.time()
    for batch_idx, batch in enumerate(loader):
        if batch_idx == 0:
            # Print first batch info
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    print(f"    Batch['{k}']: shape={v.shape}, dtype={v.dtype}")
                elif isinstance(v, list):
                    print(f"    Batch['{k}']: list of {len(v)} items, type={type(v[0]).__name__}")
    elapsed = time.time() - t0
    print(f"    ✓ Full pass: {elapsed:.1f}s ({len(dataset) / max(elapsed, 0.01):.0f} samples/s)")


def run_all(subset_size: int = SUBSET_SIZE, split: str = "train"):
    """Run the complete preprocessing pipeline end-to-end."""
    print("=" * 70)
    print("  VRSBench Preprocessing Pipeline — Full Test")
    print(f"  Subset size: {subset_size} | Split: {split}")
    print("=" * 70)
    ensure_output_dirs()

    # ── Step 1: Inspection ──
    print("\n" + "▓" * 70)
    run_inspection()

    # ── Step 2: Validation ──
    print("\n" + "▓" * 70)
    run_validation()

    # ── Step 3: Create Datasets (no transforms yet) ──
    print("\n" + "▓" * 70)
    print("=" * 70)
    print("  Step 3: Creating Datasets")
    print("=" * 70)

    cap_ds_raw = CaptioningDataset(split=split, max_samples=subset_size)
    ref_ds_raw = ReferringDataset(split=split, max_samples=subset_size)
    vqa_ds_raw = VQADataset(split=split, max_samples=subset_size)

    print(f"  CaptioningDataset ({split}): {len(cap_ds_raw)} samples")
    print(f"  ReferringDataset ({split}):  {len(ref_ds_raw)} samples")
    print(f"  VQADataset ({split}):        {len(vqa_ds_raw)} samples")

    # ── Step 4: Compute Stats ──
    print("\n" + "▓" * 70)
    print("=" * 70)
    print("  Step 4: Computing Channel Statistics")
    print("=" * 70)

    stats = compute_channel_stats(
        cap_ds_raw,
        num_samples=min(200, len(cap_ds_raw)),
    )
    mean = stats["mean"]
    std = stats["std"]

    # ── Step 6: Visualize RAW samples (before transforms) ──
    print("\n" + "▓" * 70)
    print("=" * 70)
    print("  Step 6a: Visualizing RAW Samples (no transforms)")
    print("=" * 70)

    visualize_captioning(cap_ds_raw, num_samples=5, mean=mean, std=std,
                         save_path=os.path.join(
                             os.path.dirname(os.path.abspath(__file__)),
                             "output", "viz", "captioning_raw.png"))
    visualize_referring(ref_ds_raw, num_samples=5, mean=mean, std=std,
                        save_path=os.path.join(
                            os.path.dirname(os.path.abspath(__file__)),
                            "output", "viz", "referring_raw.png"))
    visualize_vqa(vqa_ds_raw, num_samples=5, mean=mean, std=std,
                  save_path=os.path.join(
                      os.path.dirname(os.path.abspath(__file__)),
                      "output", "viz", "vqa_raw.png"))

    # ── Create Datasets WITH transforms ──
    print("\n" + "▓" * 70)
    print("=" * 70)
    print("  Step 4b: Creating Datasets with Training Transforms")
    print("=" * 70)

    cap_transform = get_train_transform_captioning(mean=mean, std=std)
    vqa_transform = get_train_transform_vqa(mean=mean, std=std)
    ref_transform = get_train_transform_referring(mean=mean, std=std, augment=True)

    cap_ds = CaptioningDataset(split=split, transform=cap_transform, max_samples=subset_size)
    ref_ds = ReferringDataset(split=split, transform=ref_transform, max_samples=subset_size)
    vqa_ds = VQADataset(split=split, transform=vqa_transform, max_samples=subset_size)

    print(f"  Transformed CaptioningDataset: {len(cap_ds)} samples")
    print(f"  Transformed ReferringDataset:  {len(ref_ds)} samples")
    print(f"  Transformed VQADataset:        {len(vqa_ds)} samples")

    # Test single sample
    sample = cap_ds[0]
    print(f"\n  Sample captioning output:")
    print(f"    image: tensor {sample['image'].shape}, range [{sample['image'].min():.3f}, {sample['image'].max():.3f}]")
    print(f"    caption: {sample['caption'][:80]}...")

    sample = ref_ds[0]
    print(f"\n  Sample referring output:")
    print(f"    image: tensor {sample['image'].shape}")
    print(f"    bbox: {sample['bbox']}")
    print(f"    text: {sample['referring_text'][:80]}...")

    sample = vqa_ds[0]
    print(f"\n  Sample VQA output:")
    print(f"    image: tensor {sample['image'].shape}")
    print(f"    Q: {sample['question'][:80]}...")
    print(f"    A: {sample['answer']}")

    # ── Step 6b: Visualize TRANSFORMED samples ──
    print("\n" + "▓" * 70)
    print("=" * 70)
    print("  Step 6b: Visualizing TRANSFORMED Samples")
    print("=" * 70)

    visualize_captioning(cap_ds, num_samples=5, mean=mean, std=std)
    visualize_referring(ref_ds, num_samples=5, mean=mean, std=std)
    visualize_vqa(vqa_ds, num_samples=5, mean=mean, std=std)

    # ── Step 5: Test Inference Transform ──
    print("\n" + "▓" * 70)
    print("=" * 70)
    print("  Step 5: Inference Transform Test")
    print("=" * 70)

    inf_transform = get_inference_transform(mean=mean, std=std)
    inf_ds = VQADataset(split="val", transform=inf_transform, max_samples=10)
    if len(inf_ds) > 0:
        sample = inf_ds[0]
        print(f"  Inference sample: tensor {sample['image'].shape}")
        print(f"  Q: {sample['question'][:80]}")
        print(f"  A: {sample['answer']}")

    # ── Step 7: DataLoader Test ──
    print("\n" + "▓" * 70)
    print("=" * 70)
    print("  Step 7: DataLoader Iteration Test")
    print("=" * 70)

    test_dataloader(cap_ds, "Captioning")
    test_dataloader(ref_ds, "Referring")
    test_dataloader(vqa_ds, "VQA")

    # ── Summary ──
    print("\n" + "=" * 70)
    print("  ✓ PIPELINE TEST COMPLETE")
    print("=" * 70)
    print(f"  Captioning: {len(cap_ds)} samples processed")
    print(f"  Referring:  {len(ref_ds)} samples processed")
    print(f"  VQA:        {len(vqa_ds)} samples processed")
    print(f"  Channel stats: mean={mean}, std={std}")
    print(f"  Visualizations: data/preprocessing/output/viz/")
    print(f"\n  To run on full dataset, increase --subset or remove the limit.")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="VRSBench Preprocessing — Full Pipeline Test"
    )
    parser.add_argument(
        "--subset", type=int, default=SUBSET_SIZE,
        help=f"Number of samples per task (default: {SUBSET_SIZE})"
    )
    parser.add_argument(
        "--split", type=str, default="train", choices=["train", "val"],
        help="Which split to test (default: train)"
    )
    args = parser.parse_args()

    run_all(subset_size=args.subset, split=args.split)
