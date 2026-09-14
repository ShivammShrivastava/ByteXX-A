"""
Step 6 — Sanity-Check Visualization
=====================================
Display processed samples per task:
  - Captioning: image + caption text
  - Referring:  image + bounding box drawn + referring text
  - VQA:        image + question/answer text

Saves output to data/preprocessing/output/viz/ as PNG files.
"""

import os
import sys
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend (no display needed)
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from data.preprocessing.config import (
    VIZ_OUTPUT_DIR,
    IMAGENET_MEAN,
    IMAGENET_STD,
    ensure_output_dirs,
)


def _tensor_to_numpy(
    img_tensor: torch.Tensor,
    mean: list = IMAGENET_MEAN,
    std: list = IMAGENET_STD,
) -> np.ndarray:
    """
    Convert a normalized tensor back to displayable numpy array.
    Reverses Normalize(mean, std) and converts to [0, 255] uint8.
    """
    img = img_tensor.clone()

    # Reverse normalization: x = x * std + mean
    for c in range(3):
        img[c] = img[c] * std[c] + mean[c]

    # Clamp and convert to uint8
    img = torch.clamp(img, 0, 1)
    img = img.permute(1, 2, 0).numpy()  # (C, H, W) → (H, W, C)
    img = (img * 255).astype(np.uint8)
    return img


def _get_image_for_display(sample: dict, mean: list, std: list) -> np.ndarray:
    """Get displayable image from a dataset sample."""
    img = sample["image"]
    if isinstance(img, torch.Tensor):
        return _tensor_to_numpy(img, mean, std)
    else:
        # PIL Image — convert directly
        return np.array(img)


def visualize_captioning(
    dataset,
    num_samples: int = 5,
    mean: Optional[list] = None,
    std: Optional[list] = None,
    save_path: Optional[str] = None,
):
    """
    Visualize captioning samples: images with captions as titles.
    """
    mean = mean or IMAGENET_MEAN
    std = std or IMAGENET_STD
    ensure_output_dirs()

    n = min(num_samples, len(dataset))
    if n == 0:
        print("  ⚠ No captioning samples to visualize")
        return

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 5))
    if n == 1:
        axes = [axes]

    for i in range(n):
        sample = dataset[i]
        img = _get_image_for_display(sample, mean, std)
        caption = sample["caption"]

        axes[i].imshow(img)
        axes[i].set_title(
            caption[:80] + ("..." if len(caption) > 80 else ""),
            fontsize=7,
            wrap=True,
        )
        axes[i].axis("off")

    fig.suptitle("Captioning Samples", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(VIZ_OUTPUT_DIR, "captioning_samples.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved captioning visualization: {save_path}")


def visualize_referring(
    dataset,
    num_samples: int = 5,
    mean: Optional[list] = None,
    std: Optional[list] = None,
    save_path: Optional[str] = None,
):
    """
    Visualize referring samples: images with bounding boxes and referring text.
    """
    mean = mean or IMAGENET_MEAN
    std = std or IMAGENET_STD
    ensure_output_dirs()

    n = min(num_samples, len(dataset))
    if n == 0:
        print("  ⚠ No referring samples to visualize")
        return

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 5))
    if n == 1:
        axes = [axes]

    for i in range(n):
        sample = dataset[i]
        img = _get_image_for_display(sample, mean, std)
        bbox = sample["bbox"]
        ref_text = sample["referring_text"]
        obj_cls = sample.get("obj_cls", "")

        h, w = img.shape[:2]

        axes[i].imshow(img)

        # Draw bounding box (coords are normalized 0-1)
        x1, y1, x2, y2 = bbox
        rect_x = x1 * w
        rect_y = y1 * h
        rect_w = (x2 - x1) * w
        rect_h = (y2 - y1) * h

        rect = patches.Rectangle(
            (rect_x, rect_y), rect_w, rect_h,
            linewidth=2, edgecolor="lime", facecolor="none",
        )
        axes[i].add_patch(rect)

        # Label
        title = ref_text[:60] + ("..." if len(ref_text) > 60 else "")
        if obj_cls and obj_cls != "unknown":
            title = f"[{obj_cls}] {title}"
        axes[i].set_title(title, fontsize=7, wrap=True)
        axes[i].axis("off")

        # Bbox text
        axes[i].text(
            rect_x, rect_y - 3,
            f"({x1:.2f},{y1:.2f},{x2:.2f},{y2:.2f})",
            fontsize=5, color="lime",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.7),
        )

    fig.suptitle("Referring Expression Samples (with bboxes)", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(VIZ_OUTPUT_DIR, "referring_samples.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved referring visualization: {save_path}")


def visualize_vqa(
    dataset,
    num_samples: int = 5,
    mean: Optional[list] = None,
    std: Optional[list] = None,
    save_path: Optional[str] = None,
):
    """
    Visualize VQA samples: images with question/answer pairs as text.
    """
    mean = mean or IMAGENET_MEAN
    std = std or IMAGENET_STD
    ensure_output_dirs()

    n = min(num_samples, len(dataset))
    if n == 0:
        print("  ⚠ No VQA samples to visualize")
        return

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 6))
    if n == 1:
        axes = [axes]

    for i in range(n):
        sample = dataset[i]
        img = _get_image_for_display(sample, mean, std)
        question = sample["question"]
        answer = sample["answer"]
        q_type = sample.get("question_type", "")

        axes[i].imshow(img)
        title_lines = [
            f"Q: {question[:70]}{'...' if len(question) > 70 else ''}",
            f"A: {answer[:50]}{'...' if len(answer) > 50 else ''}",
        ]
        if q_type and q_type != "unknown":
            title_lines.insert(0, f"[{q_type}]")

        axes[i].set_title("\n".join(title_lines), fontsize=7, wrap=True)
        axes[i].axis("off")

    fig.suptitle("VQA Samples", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(VIZ_OUTPUT_DIR, "vqa_samples.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved VQA visualization: {save_path}")


def run_visualization(
    cap_dataset=None,
    ref_dataset=None,
    vqa_dataset=None,
    num_samples: int = 5,
    mean: Optional[list] = None,
    std: Optional[list] = None,
):
    """Run all visualizations."""
    print("=" * 70)
    print("  Step 6: Sanity-Check Visualization")
    print("=" * 70)

    if cap_dataset:
        visualize_captioning(cap_dataset, num_samples, mean, std)
    if ref_dataset:
        visualize_referring(ref_dataset, num_samples, mean, std)
    if vqa_dataset:
        visualize_vqa(vqa_dataset, num_samples, mean, std)

    print(f"\n  Visualizations saved to: {VIZ_OUTPUT_DIR}")
    print()


if __name__ == "__main__":
    from data.preprocessing.step3_datasets import (
        CaptioningDataset, ReferringDataset, VQADataset,
    )

    run_visualization(
        cap_dataset=CaptioningDataset(split="train", max_samples=5),
        ref_dataset=ReferringDataset(split="train", max_samples=5),
        vqa_dataset=VQADataset(split="train", max_samples=5),
        num_samples=5,
    )
