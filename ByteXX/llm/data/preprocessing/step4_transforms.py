"""
Step 4 — Training Preprocessing Transforms
=============================================
- Compute per-channel RGB mean/std from this dataset (not assuming ImageNet)
- Build task-specific transform pipelines:
    - Captioning & VQA: resize + flip/rotate augmentation + normalize
    - Referring: resize + normalize, with optional bbox-aware augmentation
    
Normalization strategy:
    We compute actual dataset stats AND store ImageNet stats.
    Default: dataset-computed stats (more accurate for satellite imagery).
    Flag: use_imagenet_stats=True when using ImageNet-pretrained backbone.
"""

import json
import os
import sys
import random
from typing import Optional

from PIL import Image
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import numpy as np
from tqdm import tqdm

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from data.preprocessing.config import (
    TARGET_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    STATS_SAMPLE_COUNT,
    STATS_OUTPUT_DIR,
    ensure_output_dirs,
)


# ─────────────────────────────────────────────
# Channel Statistics Computation
# ─────────────────────────────────────────────

def compute_channel_stats(
    dataset,
    num_samples: int = STATS_SAMPLE_COUNT,
    save_path: Optional[str] = None,
) -> dict:
    """
    Compute per-channel mean and std from a dataset of RGB images.
    
    Samples `num_samples` images randomly (or all if dataset is smaller),
    resizes to TARGET_SIZE, and computes running statistics.
    
    This is more accurate than assuming ImageNet stats, especially for
    satellite/aerial imagery which has different color distributions
    (more green/brown, less indoor scenes).
    
    Args:
        dataset: A Dataset that returns dicts with "image" key (PIL Image)
        num_samples: How many images to sample for stats
        save_path: Optional path to save stats JSON
    
    Returns:
        dict with keys: "mean", "std", "num_samples", "imagenet_mean", "imagenet_std"
    """
    print(f"\n─── Computing Channel Statistics ───")
    print(f"  Sampling {min(num_samples, len(dataset))} images from dataset ({len(dataset)} total)...")

    # Determine indices to sample
    n = min(num_samples, len(dataset))
    indices = random.sample(range(len(dataset)), n) if n < len(dataset) else list(range(n))

    resize = T.Resize(TARGET_SIZE)
    to_tensor = T.ToTensor()  # Converts [0,255] → [0,1] float

    # Running sum for mean/std computation (Welford's method)
    channel_sum = torch.zeros(3)
    channel_sq_sum = torch.zeros(3)
    pixel_count = 0

    for i in tqdm(indices, desc="  Computing stats", ncols=80):
        sample = dataset[i]
        img = sample["image"]

        # If already a tensor (transform applied), skip resize
        if isinstance(img, Image.Image):
            img = resize(img)
            img_tensor = to_tensor(img)  # Shape: (3, H, W), range [0, 1]
        elif isinstance(img, torch.Tensor):
            img_tensor = img
        else:
            continue

        # Accumulate per-channel stats
        channel_sum += img_tensor.sum(dim=[1, 2])
        channel_sq_sum += (img_tensor ** 2).sum(dim=[1, 2])
        pixel_count += img_tensor.shape[1] * img_tensor.shape[2]

    # Compute mean and std
    mean = (channel_sum / pixel_count).tolist()
    std = (torch.sqrt(channel_sq_sum / pixel_count - torch.tensor(mean) ** 2)).tolist()

    stats = {
        "mean": mean,
        "std": std,
        "num_samples": n,
        "imagenet_mean": IMAGENET_MEAN,
        "imagenet_std": IMAGENET_STD,
        "note": (
            "Dataset-specific stats computed from VRSBench satellite imagery. "
            "ImageNet stats also stored for use with pretrained backbones. "
            "For satellite imagery, dataset stats may differ significantly from "
            "ImageNet due to different scene types (aerial vs natural photos)."
        ),
    }

    print(f"  Dataset mean: [{mean[0]:.4f}, {mean[1]:.4f}, {mean[2]:.4f}]")
    print(f"  Dataset std:  [{std[0]:.4f}, {std[1]:.4f}, {std[2]:.4f}]")
    print(f"  ImageNet mean: {IMAGENET_MEAN}")
    print(f"  ImageNet std:  {IMAGENET_STD}")

    # Save to disk
    if save_path is None:
        ensure_output_dirs()
        save_path = os.path.join(STATS_OUTPUT_DIR, "channel_stats.json")

    with open(save_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Saved to: {save_path}")

    return stats


def load_channel_stats(stats_path: Optional[str] = None) -> dict:
    """Load previously computed channel stats from JSON."""
    if stats_path is None:
        stats_path = os.path.join(STATS_OUTPUT_DIR, "channel_stats.json")

    if not os.path.exists(stats_path):
        raise FileNotFoundError(
            f"Channel stats not found at {stats_path}. "
            f"Run compute_channel_stats() first."
        )

    with open(stats_path, "r") as f:
        return json.load(f)


# ─────────────────────────────────────────────
# Training Transforms — Captioning & VQA
# ─────────────────────────────────────────────

def get_train_transform_captioning(
    target_size: tuple = TARGET_SIZE,
    mean: Optional[list] = None,
    std: Optional[list] = None,
    use_imagenet_stats: bool = False,
) -> T.Compose:
    """
    Training transform for captioning task.
    
    Pipeline: Resize → RandomHorizontalFlip → RandomVerticalFlip →
              RandomRotation(90°) → ToTensor → Normalize
    
    Augmentation is safe here because captions describe the whole image
    and are invariant to flips/rotations.
    
    Args:
        target_size: (H, W) resize target
        mean: Per-channel mean for normalization (auto-loads if None)
        std: Per-channel std for normalization (auto-loads if None)
        use_imagenet_stats: If True, use ImageNet stats instead of dataset stats
    """
    mean, std = _resolve_stats(mean, std, use_imagenet_stats)

    return T.Compose([
        T.Resize(target_size),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomApply([T.RandomRotation(degrees=[90, 90])], p=0.25),
        T.RandomApply([T.RandomRotation(degrees=[180, 180])], p=0.25),
        T.RandomApply([T.RandomRotation(degrees=[270, 270])], p=0.25),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])


def get_train_transform_vqa(
    target_size: tuple = TARGET_SIZE,
    mean: Optional[list] = None,
    std: Optional[list] = None,
    use_imagenet_stats: bool = False,
) -> T.Compose:
    """
    Training transform for VQA task.
    
    Same as captioning: Resize + flip/rotate augmentation + normalize.
    VQA questions are generally invariant to geometric transforms
    (e.g., "How many buildings?" doesn't change with a flip).
    """
    mean, std = _resolve_stats(mean, std, use_imagenet_stats)

    return T.Compose([
        T.Resize(target_size),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomApply([T.RandomRotation(degrees=[90, 90])], p=0.25),
        T.RandomApply([T.RandomRotation(degrees=[180, 180])], p=0.25),
        T.RandomApply([T.RandomRotation(degrees=[270, 270])], p=0.25),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])


# ─────────────────────────────────────────────
# Training Transforms — Referring (bbox-aware)
# ─────────────────────────────────────────────

def get_train_transform_referring(
    target_size: tuple = TARGET_SIZE,
    mean: Optional[list] = None,
    std: Optional[list] = None,
    use_imagenet_stats: bool = False,
    augment: bool = True,
) -> "ReferringAugmentation":
    """
    Training transform for referring/grounding task.
    
    Returns a ReferringAugmentation object that jointly transforms
    the image AND bounding box coordinates.
    
    If augment=False, only resize + normalize (safe for all cases).
    If augment=True, applies flips with corresponding bbox transforms.
    
    IMPORTANT: Unlike captioning/VQA, we MUST co-transform the bbox
    when flipping or rotating the image. Stale box coords = wrong grounding.
    """
    mean, std = _resolve_stats(mean, std, use_imagenet_stats)
    return ReferringAugmentation(
        target_size=target_size,
        mean=mean,
        std=std,
        augment=augment,
    )


class ReferringAugmentation:
    """
    Joint image + bounding box augmentation for the referring/grounding task.
    
    Supports:
        - Resize (always applied)
        - Horizontal flip (with bbox x-coord mirroring)
        - Vertical flip (with bbox y-coord mirroring)
        - Normalize (always applied)
    
    Bounding box format: [x1, y1, x2, y2] normalized to [0, 1].
    After horizontal flip: x_new = 1 - x_old, swap x1 and x2.
    After vertical flip:   y_new = 1 - y_old, swap y1 and y2.
    """

    def __init__(
        self,
        target_size: tuple = TARGET_SIZE,
        mean: list = IMAGENET_MEAN,
        std: list = IMAGENET_STD,
        augment: bool = True,
        hflip_prob: float = 0.5,
        vflip_prob: float = 0.5,
    ):
        self.target_size = target_size
        self.mean = mean
        self.std = std
        self.augment = augment
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob

        # Non-augmentation transforms
        self.resize = T.Resize(target_size)
        self.to_tensor = T.ToTensor()
        self.normalize = T.Normalize(mean=mean, std=std)

    def __call__(self, image):
        """Standard transform interface (without bbox). Used by Dataset.__getitem__."""
        image = self.resize(image)
        image = self.to_tensor(image)
        image = self.normalize(image)
        return image

    def transform_with_bbox(
        self, image: Image.Image, bbox: list[float]
    ) -> tuple[torch.Tensor, list[float]]:
        """
        Joint transform: image + bbox are transformed together.
        
        Args:
            image: PIL Image
            bbox: [x1, y1, x2, y2] normalized to [0, 1]
        
        Returns:
            (transformed_image_tensor, transformed_bbox)
        """
        x1, y1, x2, y2 = bbox

        # Step 1: Resize (bbox stays the same since coords are normalized)
        image = self.resize(image)

        # Step 2: Augmentation (only during training)
        if self.augment:
            # Horizontal flip
            if random.random() < self.hflip_prob:
                image = TF.hflip(image)
                # Mirror x-coordinates: x_new = 1 - x_old, then swap x1/x2
                x1_new = 1.0 - x2
                x2_new = 1.0 - x1
                x1, x2 = x1_new, x2_new

            # Vertical flip
            if random.random() < self.vflip_prob:
                image = TF.vflip(image)
                # Mirror y-coordinates: y_new = 1 - y_old, then swap y1/y2
                y1_new = 1.0 - y2
                y2_new = 1.0 - y1
                y1, y2 = y1_new, y2_new

        # Step 3: To tensor + normalize
        image = self.to_tensor(image)
        image = self.normalize(image)

        # Clamp bbox to [0, 1] after transforms
        bbox_out = [
            max(0.0, min(1.0, x1)),
            max(0.0, min(1.0, y1)),
            max(0.0, min(1.0, x2)),
            max(0.0, min(1.0, y2)),
        ]

        return image, bbox_out


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _resolve_stats(
    mean: Optional[list],
    std: Optional[list],
    use_imagenet_stats: bool,
) -> tuple[list, list]:
    """Resolve normalization stats: explicit > imagenet > dataset-computed."""
    if mean is not None and std is not None:
        return mean, std

    if use_imagenet_stats:
        return IMAGENET_MEAN, IMAGENET_STD

    # Try to load dataset-computed stats
    try:
        stats = load_channel_stats()
        return stats["mean"], stats["std"]
    except FileNotFoundError:
        print("  ⚠ Dataset stats not computed yet. Using ImageNet stats as fallback.")
        print("    Run compute_channel_stats() first for dataset-specific normalization.")
        return IMAGENET_MEAN, IMAGENET_STD


if __name__ == "__main__":
    print("=" * 70)
    print("  Step 4: Training Transforms — Quick Test")
    print("=" * 70)

    from data.preprocessing.step3_datasets import CaptioningDataset

    # Compute stats from a small subset
    ds = CaptioningDataset(split="train", max_samples=50)
    stats = compute_channel_stats(ds, num_samples=50)

    # Build transforms
    cap_tf = get_train_transform_captioning(mean=stats["mean"], std=stats["std"])
    vqa_tf = get_train_transform_vqa(mean=stats["mean"], std=stats["std"])
    ref_tf = get_train_transform_referring(mean=stats["mean"], std=stats["std"])

    # Test captioning transform
    sample_img = ds[0]["image"]
    transformed = cap_tf(sample_img)
    print(f"\n  Captioning transform output: {transformed.shape}, dtype={transformed.dtype}")
    print(f"    Range: [{transformed.min():.3f}, {transformed.max():.3f}]")

    # Test referring transform with bbox
    test_bbox = [0.3, 0.4, 0.7, 0.8]
    transformed_img, transformed_bbox = ref_tf.transform_with_bbox(sample_img, test_bbox)
    print(f"\n  Referring transform output: {transformed_img.shape}")
    print(f"    Input bbox:  {test_bbox}")
    print(f"    Output bbox: {transformed_bbox}")
    print()
