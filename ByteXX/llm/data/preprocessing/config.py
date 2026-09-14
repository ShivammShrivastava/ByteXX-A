"""
VRSBench Preprocessing — Central Configuration
================================================
All paths, constants, and defaults in one place.
"""

import os

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
DATASET_DIR = r"C:\Users\Suryansh\OneDrive\Desktop\dataset"

# Images_train has a double-nested structure from the zip extraction
IMAGES_TRAIN_DIR = os.path.join(DATASET_DIR, "Images_train", "Images_train")
IMAGES_VAL_DIR = os.path.join(DATASET_DIR, "Images_val")

# JSON annotation files
TRAIN_JSON = os.path.join(DATASET_DIR, "VRSBench_train.json")
EVAL_CAP_JSON = os.path.join(DATASET_DIR, "VRSBench_EVAL_Cap.json")
EVAL_REFER_JSON = os.path.join(DATASET_DIR, "VRSBench_EVAL_referring.json")
EVAL_VQA_JSON = os.path.join(DATASET_DIR, "VRSBench_EVAL_vqa.json")

# Annotations zip (only val exists; train annotations are in VRSBench_train.json)
ANNOTATIONS_VAL_ZIP = os.path.join(DATASET_DIR, "Annotations_val.zip")
ANNOTATIONS_TRAIN_ZIP = os.path.join(DATASET_DIR, "Annotations_train.zip")

# Output directories
PREPROCESSING_OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "output"
)
VIZ_OUTPUT_DIR = os.path.join(PREPROCESSING_OUTPUT_DIR, "viz")
STATS_OUTPUT_DIR = os.path.join(PREPROCESSING_OUTPUT_DIR, "stats")

# ─────────────────────────────────────────────
# Image Processing
# ─────────────────────────────────────────────
TARGET_SIZE = (512, 512)  # (H, W) — configurable resize target

# ImageNet normalization stats (common baseline for pretrained backbones)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ─────────────────────────────────────────────
# Dataset Subset Defaults
# ─────────────────────────────────────────────
SUBSET_SIZE = 200  # Default number of samples for quick testing

# Number of samples to use when computing per-channel mean/std
STATS_SAMPLE_COUNT = 500

# DataLoader defaults (tuned for 16GB RAM + RTX 3050 8GB)
BATCH_SIZE = 8
NUM_WORKERS = 2  # Conservative for Windows; increase on Linux

# ─────────────────────────────────────────────
# Task tags in VRSBench_train.json conversations
# ─────────────────────────────────────────────
TASK_TAG_CAPTION = "[caption]"
TASK_TAG_REFER = "[refer]"
TASK_TAG_VQA = "[vqa]"


def ensure_output_dirs():
    """Create output directories if they don't exist."""
    os.makedirs(PREPROCESSING_OUTPUT_DIR, exist_ok=True)
    os.makedirs(VIZ_OUTPUT_DIR, exist_ok=True)
    os.makedirs(STATS_OUTPUT_DIR, exist_ok=True)


def resolve_train_image_path(image_name: str) -> str | None:
    """
    Resolve a training image filename to its full path.
    Handles the double-nested Images_train/Images_train/ structure.
    Returns None if the image doesn't exist.
    """
    # Primary: double-nested path
    path = os.path.join(IMAGES_TRAIN_DIR, image_name)
    if os.path.exists(path):
        return path
    # Fallback: single-level
    path = os.path.join(DATASET_DIR, "Images_train", image_name)
    if os.path.exists(path):
        return path
    return None


def resolve_val_image_path(image_name: str) -> str | None:
    """Resolve a validation image filename to its full path."""
    path = os.path.join(IMAGES_VAL_DIR, image_name)
    if os.path.exists(path):
        return path
    return None
