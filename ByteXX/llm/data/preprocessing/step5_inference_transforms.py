"""
Step 5 — Inference Preprocessing Transforms
=============================================
Same resize + normalize as training, but NO augmentation.
Used for live user-uploaded images at inference time.

Loads saved normalization stats from Step 4 (or falls back to ImageNet).
"""

import os
import sys
from typing import Optional

import torchvision.transforms as T

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from data.preprocessing.config import (
    TARGET_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    STATS_OUTPUT_DIR,
)
from data.preprocessing.step4_transforms import load_channel_stats


def get_inference_transform(
    target_size: tuple = TARGET_SIZE,
    mean: Optional[list] = None,
    std: Optional[list] = None,
    use_imagenet_stats: bool = False,
    stats_path: Optional[str] = None,
) -> T.Compose:
    """
    Inference-time transform: Resize → ToTensor → Normalize.
    
    No augmentation — deterministic pipeline for reproducible predictions.
    
    Stats resolution order:
        1. Explicit mean/std arguments
        2. use_imagenet_stats=True → ImageNet stats
        3. Auto-load from saved dataset stats (computed in Step 4)
        4. Fallback to ImageNet stats with a warning
    
    Args:
        target_size: (H, W) resize target — must match training
        mean: Explicit per-channel mean
        std: Explicit per-channel std
        use_imagenet_stats: Force ImageNet stats
        stats_path: Path to saved channel_stats.json
    
    Returns:
        torchvision.transforms.Compose
    """
    if mean is not None and std is not None:
        pass  # Use provided values
    elif use_imagenet_stats:
        mean = IMAGENET_MEAN
        std = IMAGENET_STD
    else:
        try:
            stats = load_channel_stats(stats_path)
            mean = stats["mean"]
            std = stats["std"]
            print(f"  ℹ Using dataset-computed stats: mean={mean}, std={std}")
        except FileNotFoundError:
            print("  ⚠ No saved stats found. Using ImageNet stats for inference.")
            print("    To use dataset-specific stats, run Step 4 first.")
            mean = IMAGENET_MEAN
            std = IMAGENET_STD

    return T.Compose([
        T.Resize(target_size),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])


def get_inference_transform_with_inverse(
    target_size: tuple = TARGET_SIZE,
    mean: Optional[list] = None,
    std: Optional[list] = None,
    use_imagenet_stats: bool = False,
) -> tuple[T.Compose, T.Compose]:
    """
    Get both the inference transform and its inverse (for visualization).
    
    Returns:
        (forward_transform, inverse_normalize)
    """
    if mean is None or std is None:
        try:
            stats = load_channel_stats()
            mean = stats["mean"]
            std = stats["std"]
        except FileNotFoundError:
            mean = IMAGENET_MEAN
            std = IMAGENET_STD

    forward = T.Compose([
        T.Resize(target_size),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])

    # Inverse normalization: x_original = x_normalized * std + mean
    inv_mean = [-m / s for m, s in zip(mean, std)]
    inv_std = [1.0 / s for s in std]
    inverse = T.Normalize(mean=inv_mean, std=inv_std)

    return forward, inverse


if __name__ == "__main__":
    print("=" * 70)
    print("  Step 5: Inference Transform — Test")
    print("=" * 70)

    transform = get_inference_transform()
    print(f"  Transform: {transform}")

    # Test on a sample image
    from data.preprocessing.step3_datasets import VQADataset
    ds = VQADataset(split="val", max_samples=3)
    if len(ds) > 0:
        sample = ds[0]
        img = sample["image"]  # PIL Image
        transformed = transform(img)
        print(f"\n  Input: PIL {img.size}")
        print(f"  Output: tensor {transformed.shape}, dtype={transformed.dtype}")
        print(f"  Range: [{transformed.min():.3f}, {transformed.max():.3f}]")
    print()
