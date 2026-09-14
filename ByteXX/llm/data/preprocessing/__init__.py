"""
VRSBench Data Preprocessing Pipeline
=====================================
Modular preprocessing for the VRSBench dataset covering three tasks:
  - Image Captioning
  - Referring Expression / Grounding (with bounding boxes)
  - Visual Question Answering (VQA)

Usage:
    python -m data.preprocessing.step7_run_all
"""

from data.preprocessing.config import (
    DATASET_DIR,
    IMAGES_TRAIN_DIR,
    IMAGES_VAL_DIR,
    TARGET_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    SUBSET_SIZE,
)

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
    ReferringAugmentation,
)

from data.preprocessing.step5_inference_transforms import (
    get_inference_transform,
)
