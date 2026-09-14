"""
Step 3 — Task-Specific PyTorch Dataset Classes
================================================
Three lazy-loading Dataset classes that read images from disk on demand:
  - CaptioningDataset: (image, caption) pairs
  - ReferringDataset:  (image, referring_text, bbox) triples
  - VQADataset:        (image, question, answer) triples

All datasets support:
  - max_samples: limit for quick testing on subsets
  - transform: optional image transform (from step 4/5)
  - split="train" or "val"
"""

import json
import os
import re
import sys
from typing import Optional, Callable

from PIL import Image
import torch
from torch.utils.data import Dataset

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


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _load_json(path: str) -> list:
    """Load a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _clean_text(text: str) -> str:
    """Remove <image>, task tags, and <p>...</p> markers from conversation text."""
    cleaned = text.replace("<image>", "").strip()
    cleaned = re.sub(r"\[(vqa|caption|refer)\]\s*", "", cleaned, flags=re.IGNORECASE)
    # Remove <p>...</p> tags but keep content
    cleaned = re.sub(r"</?p>", "", cleaned)
    return cleaned.strip()


def _parse_quantized_bbox(bbox_str: str) -> list[float]:
    """
    Parse VRSBench quantized bounding box format: {<x1><y1><x2><y2>}
    
    Values are quantized coordinates on a scale where ~100 = full image.
    Observed range: 0-160 (objects can extend slightly beyond image bounds).
    
    We normalize to [0, 1] by dividing by 100, then clamp to [0, 1].
    This matches the obj_coord format in Annotations_val/*.json which
    uses 0-1 normalized coordinates.
    
    Returns: [x1, y1, x2, y2] normalized to [0, 1]
    """
    nums = re.findall(r"<(\d+)>", bbox_str)
    if len(nums) != 4:
        return [0.0, 0.0, 1.0, 1.0]  # Fallback: full image

    coords = [int(n) / 100.0 for n in nums]
    # Clamp to [0, 1]
    coords = [max(0.0, min(1.0, c)) for c in coords]
    return coords


# ─────────────────────────────────────────────
# CaptioningDataset
# ─────────────────────────────────────────────

class CaptioningDataset(Dataset):
    """
    Loads (image, caption) pairs for image captioning.
    
    Train split: Filters VRSBench_train.json for [caption] entries.
    Val split:   Uses VRSBench_EVAL_Cap.json directly.
    
    Lazy-loads images from disk — does NOT hold images in memory.
    """

    def __init__(
        self,
        split: str = "train",
        transform: Optional[Callable] = None,
        max_samples: Optional[int] = None,
    ):
        """
        Args:
            split: "train" or "val"
            transform: Optional torchvision transform for the image
            max_samples: Limit number of samples (for quick testing)
        """
        self.split = split
        self.transform = transform
        self.samples = []  # List of (image_path, caption_text)

        if split == "train":
            self._load_train(max_samples)
        elif split == "val":
            self._load_val(max_samples)
        else:
            raise ValueError(f"Unknown split: {split}. Use 'train' or 'val'.")

    def _load_train(self, max_samples: Optional[int]):
        """Parse train JSON for caption entries."""
        data = _load_json(TRAIN_JSON)
        count = 0
        for entry in data:
            q_text = entry["conversations"][0]["value"].lower()
            if TASK_TAG_CAPTION not in q_text:
                continue

            img_path = resolve_train_image_path(entry["image"])
            if img_path is None:
                continue  # Skip missing images (reported in step 2)

            caption = entry["conversations"][1]["value"].strip()
            self.samples.append((img_path, caption))
            count += 1

            if max_samples and count >= max_samples:
                break

    def _load_val(self, max_samples: Optional[int]):
        """Parse eval captioning JSON."""
        data = _load_json(EVAL_CAP_JSON)
        count = 0
        for entry in data:
            img_path = resolve_val_image_path(entry["image_id"])
            if img_path is None:
                continue

            caption = entry["ground_truth"].strip()
            self.samples.append((img_path, caption))
            count += 1

            if max_samples and count >= max_samples:
                break

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        """
        Returns:
            dict with keys:
                - "image": PIL Image or transformed tensor
                - "caption": str
                - "image_path": str (for debugging)
        """
        img_path, caption = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return {
            "image": image,
            "caption": caption,
            "image_path": img_path,
        }


# ─────────────────────────────────────────────
# ReferringDataset
# ─────────────────────────────────────────────

class ReferringDataset(Dataset):
    """
    Loads (image, referring_text, bbox) triples for referring expression grounding.
    
    Train split: Filters VRSBench_train.json for [refer] entries.
        - Referring text extracted from <p>...</p> in the question.
        - Bbox parsed from the GPT response: {<x1><y1><x2><y2>}
        - Coords normalized to [0, 1] by dividing quantized values by 100.
    
    Val split: Uses VRSBench_EVAL_referring.json.
        - Referring text from "question" field.
        - Bbox from "ground_truth" (same quantized format) and/or "obj_corner".
    
    Bounding box format: [x1, y1, x2, y2] normalized to [0, 1].
    """

    def __init__(
        self,
        split: str = "train",
        transform: Optional[Callable] = None,
        max_samples: Optional[int] = None,
    ):
        """
        Args:
            split: "train" or "val"
            transform: Optional transform. For referring, if augmentation is
                       needed, use ReferringAugmentation from step4 which
                       jointly transforms image + bbox.
            max_samples: Limit number of samples
        """
        self.split = split
        self.transform = transform
        self.samples = []  # List of (image_path, referring_text, bbox, obj_cls)

        if split == "train":
            self._load_train(max_samples)
        elif split == "val":
            self._load_val(max_samples)
        else:
            raise ValueError(f"Unknown split: {split}. Use 'train' or 'val'.")

    def _load_train(self, max_samples: Optional[int]):
        """Parse train JSON for referring entries."""
        data = _load_json(TRAIN_JSON)
        count = 0
        for entry in data:
            q_text = entry["conversations"][0]["value"]
            if TASK_TAG_REFER not in q_text.lower():
                continue

            img_path = resolve_train_image_path(entry["image"])
            if img_path is None:
                continue

            # Extract referring text from <p>...</p> tags
            p_match = re.search(r"<p>(.*?)</p>", q_text, re.DOTALL)
            if p_match:
                ref_text = p_match.group(1).strip()
            else:
                ref_text = _clean_text(q_text)

            # Parse bbox from GPT response
            bbox_str = entry["conversations"][1]["value"]
            bbox = _parse_quantized_bbox(bbox_str)

            self.samples.append((img_path, ref_text, bbox, "unknown"))
            count += 1

            if max_samples and count >= max_samples:
                break

    def _load_val(self, max_samples: Optional[int]):
        """Parse eval referring JSON."""
        data = _load_json(EVAL_REFER_JSON)
        count = 0
        for entry in data:
            img_path = resolve_val_image_path(entry["image_id"])
            if img_path is None:
                continue

            ref_text = entry["question"].strip()
            bbox = _parse_quantized_bbox(entry["ground_truth"])
            obj_cls = entry.get("obj_cls", "unknown")

            self.samples.append((img_path, ref_text, bbox, obj_cls))
            count += 1

            if max_samples and count >= max_samples:
                break

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        """
        Returns:
            dict with keys:
                - "image": PIL Image or transformed tensor
                - "referring_text": str
                - "bbox": list of 4 floats [x1, y1, x2, y2] normalized [0,1]
                - "obj_cls": str (object class, "unknown" for train)
                - "image_path": str
        """
        img_path, ref_text, bbox, obj_cls = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            # If transform is a ReferringAugmentation, it handles bbox too
            if hasattr(self.transform, "transform_with_bbox"):
                image, bbox = self.transform.transform_with_bbox(image, bbox)
            else:
                image = self.transform(image)

        return {
            "image": image,
            "referring_text": ref_text,
            "bbox": bbox,
            "obj_cls": obj_cls,
            "image_path": img_path,
        }


# ─────────────────────────────────────────────
# VQADataset
# ─────────────────────────────────────────────

class VQADataset(Dataset):
    """
    Loads (image, question, answer) triples for visual question answering.
    
    Train split: Filters VRSBench_train.json for [vqa] entries.
    Val split:   Uses VRSBench_EVAL_vqa.json directly.
    """

    def __init__(
        self,
        split: str = "train",
        transform: Optional[Callable] = None,
        max_samples: Optional[int] = None,
    ):
        self.split = split
        self.transform = transform
        self.samples = []  # List of (image_path, question, answer, q_type)

        if split == "train":
            self._load_train(max_samples)
        elif split == "val":
            self._load_val(max_samples)
        else:
            raise ValueError(f"Unknown split: {split}. Use 'train' or 'val'.")

    def _load_train(self, max_samples: Optional[int]):
        """Parse train JSON for VQA entries."""
        data = _load_json(TRAIN_JSON)
        count = 0
        for entry in data:
            q_text = entry["conversations"][0]["value"]
            if TASK_TAG_VQA not in q_text.lower():
                continue

            img_path = resolve_train_image_path(entry["image"])
            if img_path is None:
                continue

            question = _clean_text(q_text)
            answer = entry["conversations"][1]["value"].strip()

            self.samples.append((img_path, question, answer, "unknown"))
            count += 1

            if max_samples and count >= max_samples:
                break

    def _load_val(self, max_samples: Optional[int]):
        """Parse eval VQA JSON."""
        data = _load_json(EVAL_VQA_JSON)
        count = 0
        for entry in data:
            img_path = resolve_val_image_path(entry["image_id"])
            if img_path is None:
                continue

            question = entry["question"].strip()
            answer = entry["ground_truth"].strip()
            q_type = entry.get("type", "unknown")

            self.samples.append((img_path, question, answer, q_type))
            count += 1

            if max_samples and count >= max_samples:
                break

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        """
        Returns:
            dict with keys:
                - "image": PIL Image or transformed tensor
                - "question": str
                - "answer": str
                - "question_type": str
                - "image_path": str
        """
        img_path, question, answer, q_type = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return {
            "image": image,
            "question": question,
            "answer": answer,
            "question_type": q_type,
            "image_path": img_path,
        }


if __name__ == "__main__":
    # Quick smoke test with small subsets
    print("=" * 70)
    print("  Step 3: Dataset Classes — Smoke Test")
    print("=" * 70)

    for DatasetClass, name in [
        (CaptioningDataset, "Captioning"),
        (ReferringDataset, "Referring"),
        (VQADataset, "VQA"),
    ]:
        print(f"\n─── {name}Dataset ───")
        for split in ["train", "val"]:
            ds = DatasetClass(split=split, max_samples=5)
            print(f"  {split}: {len(ds)} samples")
            if len(ds) > 0:
                sample = ds[0]
                print(f"    Keys: {list(sample.keys())}")
                for k, v in sample.items():
                    if k == "image":
                        print(f"    {k}: PIL Image {v.size}")
                    elif isinstance(v, str) and len(v) > 80:
                        print(f"    {k}: {v[:80]}...")
                    else:
                        print(f"    {k}: {v}")
    print()
