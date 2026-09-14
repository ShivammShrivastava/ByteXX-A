"""
SatQuery AI — VQA Inference Module (Agentic Tool API)
=====================================================
This module wraps the fine-tuned Qwen2.5-VL-7B model as a callable tool
for the SatQuery AI agentic controller.

The model is loaded from HuggingFace Hub (or local path) in 4-bit
quantization, fitting comfortably in ~5 GB VRAM on an RTX 3050.

Usage as standalone:
    python inference/infer_vqa.py --image path/to/image.png --question "What objects are visible?"

Usage as importable module:
    from inference.infer_vqa import SatQueryVQA
    vqa = SatQueryVQA()
    result = vqa.answer_vqa("path/to/image.png", "What land cover is dominant?")
    print(result)
    # {'answer': 'forest', 'confidence': 0.87, 'reasoning_trace': '...', ...}
"""

import os
import sys
import math
import argparse
from typing import Optional
from pathlib import Path

try:
    import torch
    from transformers import (
        Qwen2_5_VLForConditionalGeneration,
        AutoProcessor,
        BitsAndBytesConfig,
    )
    from peft import PeftModel
    from PIL import Image
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Optional: GeoTIFF support
try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
DEFAULT_LORA_PATH = "YOUR_HF_USERNAME/satquery-ai-vqa-lora"  # ← CHANGE THIS after training
BASE_MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"

RS_SYSTEM_PROMPT = (
    "You are SatQuery AI, a remote sensing image analysis expert specializing in "
    "satellite and aerial imagery interpretation. You can identify land cover types, "
    "objects, spatial relationships, and scene characteristics in remote sensing images. "
    "Provide accurate, concise answers based on visual evidence in the image."
)


# ─────────────────────────────────────────────
# Image Loading (supports PNG, JPEG, TIFF, GeoTIFF)
# ─────────────────────────────────────────────
def load_image(image_path: str) -> Image.Image:
    """
    Load an image from disk. Supports:
      - PNG, JPEG (via Pillow)
      - TIFF (via Pillow or rasterio)
      - GeoTIFF (via rasterio — selects RGB bands)

    For GeoTIFF with more than 3 bands, we select bands 4,3,2 (NIR,R,G)
    or 3,2,1 (R,G,B) depending on band count.
    """
    path = Path(image_path)
    suffix = path.suffix.lower()

    if suffix in (".tif", ".tiff") and HAS_RASTERIO:
        return _load_geotiff(image_path)
    else:
        return Image.open(image_path).convert("RGB")


def _load_geotiff(path: str) -> Image.Image:
    """Load GeoTIFF and convert to RGB PIL Image."""
    import numpy as np

    with rasterio.open(path) as src:
        band_count = src.count

        if band_count >= 4:
            # Assume bands 4,3,2 = NIR, Red, Green → use R,G,B = 3,2,1
            # Or if it's a standard RGB+NIR, use 1,2,3
            r = src.read(1).astype(np.float32)
            g = src.read(2).astype(np.float32)
            b = src.read(3).astype(np.float32)
        elif band_count == 3:
            r = src.read(1).astype(np.float32)
            g = src.read(2).astype(np.float32)
            b = src.read(3).astype(np.float32)
        elif band_count == 1:
            # Grayscale — replicate to 3 channels
            gray = src.read(1).astype(np.float32)
            r = g = b = gray
        else:
            # Use first 3 bands
            r = src.read(1).astype(np.float32)
            g = src.read(2).astype(np.float32) if band_count > 1 else r
            b = src.read(3).astype(np.float32) if band_count > 2 else r

    # Normalize to 0-255
    import numpy as np
    rgb = np.stack([r, g, b], axis=-1)

    # Percentile-based stretch for better visualization
    p2, p98 = np.percentile(rgb[rgb > 0], [2, 98]) if rgb.max() > 0 else (0, 1)
    if p98 > p2:
        rgb = np.clip((rgb - p2) / (p98 - p2) * 255, 0, 255).astype(np.uint8)
    else:
        rgb = np.zeros_like(rgb, dtype=np.uint8)

    return Image.fromarray(rgb, "RGB")


# ─────────────────────────────────────────────
# Model Singleton
# ─────────────────────────────────────────────
class SatQueryVQA:
    """
    Singleton VQA inference module for SatQuery AI.
    Loads the model once and reuses it for all queries.
    """

    _instance = None
    _model = None
    _processor = None

    def __new__(cls, model_path: str = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, model_path: str = None):
        if self._model is not None:
            return  # Already initialized

        self._model_path = model_path or DEFAULT_LORA_PATH
        self._load_model()

    def _load_model(self):
        """Load model with 4-bit quantization."""
        print(f"SatQuery VQA: Loading model from {self._model_path}...")

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        # Check if it's a LoRA adapter path
        is_lora = False
        if os.path.isdir(self._model_path):
            is_lora = os.path.exists(os.path.join(self._model_path, "adapter_config.json"))
        else:
            # Assume HF Hub path — try loading as LoRA
            is_lora = True

        if is_lora:
            print(f"  Loading base: {BASE_MODEL_NAME}")
            base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                BASE_MODEL_NAME,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )
            print(f"  Loading LoRA adapter: {self._model_path}")
            self._model = PeftModel.from_pretrained(base_model, self._model_path)
            self._model = self._model.merge_and_unload()
            self._processor = AutoProcessor.from_pretrained(BASE_MODEL_NAME)
        else:
            self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self._model_path,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )
            self._processor = AutoProcessor.from_pretrained(self._model_path)

        self._model.eval()

        vram = torch.cuda.memory_allocated() / 1024**3
        print(f"  ✓ Model loaded. VRAM used: {vram:.1f} GB")

    def answer_vqa(self, image_path: str, question: str) -> dict:
        """
        Answer a visual question about a remote sensing image.

        Args:
            image_path: Path to the image file (PNG, JPEG, TIFF, GeoTIFF)
            question: Natural language question about the image

        Returns:
            dict with keys:
                - answer (str): The model's answer
                - confidence (float): Confidence score 0.0–1.0
                - reasoning_trace (str): Brief justification
                - model (str): Model identifier
                - task (str): "single_image_vqa"
        """
        # ── Validate inputs ──
        if not os.path.exists(image_path):
            return {
                "answer": "ERROR",
                "confidence": 0.0,
                "reasoning_trace": f"Image file not found: {image_path}",
                "model": "satquery-ai-vqa",
                "task": "single_image_vqa",
            }

        # ── Load image ──
        try:
            image = load_image(image_path)
        except Exception as e:
            return {
                "answer": "ERROR",
                "confidence": 0.0,
                "reasoning_trace": f"Failed to load image: {str(e)}",
                "model": "satquery-ai-vqa",
                "task": "single_image_vqa",
            }

        # ── Build prompt ──
        messages = [
            {"role": "system", "content": RS_SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ]},
        ]

        prompt = self._processor.apply_chat_template(
            messages, add_generation_prompt=True
        )
        inputs = self._processor(
            text=[prompt],
            images=[image],
            padding=True,
            return_tensors="pt",
        ).to(self._model.device)

        # ── Generate with log-probabilities ──
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=0.1,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
            )

        # ── Decode answer ──
        generated_ids = outputs.sequences[:, inputs["input_ids"].shape[1]:]
        answer = self._processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0].strip()

        # ── Compute confidence from token log-probabilities ──
        confidence = self._compute_confidence(outputs.scores, generated_ids[0])

        # ── Build reasoning trace ──
        img_name = os.path.basename(image_path)
        reasoning = (
            f"Analyzed remote sensing image '{img_name}' "
            f"({image.size[0]}x{image.size[1]} px) "
            f"using SatQuery AI fine-tuned VQA model. "
            f"Question: '{question}'. "
            f"Answer confidence: {confidence:.2f}."
        )

        return {
            "answer": answer,
            "confidence": round(confidence, 4),
            "reasoning_trace": reasoning,
            "model": "satquery-ai-vqa",
            "task": "single_image_vqa",
        }

    def _compute_confidence(self, scores: tuple, generated_ids: torch.Tensor) -> float:
        """
        Compute confidence from generation log-probabilities.
        Uses the mean probability of the top-1 token at each generation step.
        """
        if not scores or len(scores) == 0:
            return 0.5  # Default if no scores available

        token_probs = []
        for i, score in enumerate(scores):
            if i >= len(generated_ids):
                break
            # Get probability of the actually generated token
            probs = torch.softmax(score[0], dim=-1)
            token_id = generated_ids[i].item()
            token_prob = probs[token_id].item()
            token_probs.append(token_prob)

        if not token_probs:
            return 0.5

        # Geometric mean of token probabilities (better than arithmetic for sequences)
        log_probs = [math.log(p + 1e-10) for p in token_probs]
        mean_log_prob = sum(log_probs) / len(log_probs)
        confidence = math.exp(mean_log_prob)

        # Clamp to [0, 1]
        return max(0.0, min(1.0, confidence))

    def answer_caption(self, image_path: str) -> dict:
        """
        Generate a caption/description for a remote sensing image.
        (Second single-image task for SatQuery AI)

        Args:
            image_path: Path to the image file

        Returns:
            dict with keys: caption, confidence, reasoning_trace, model, task
        """
        question = "Describe the contents of this remote sensing image in detail."
        result = self.answer_vqa(image_path, question)
        result["task"] = "single_image_caption"
        result["caption"] = result.pop("answer")
        return result


# ─────────────────────────────────────────────
# Convenience function (for agentic controller)
# ─────────────────────────────────────────────
_vqa_instance: Optional[SatQueryVQA] = None


def answer_vqa(image_path: str, question: str, model_path: str = None) -> dict:
    """
    Stateless convenience function for the agentic controller.
    Initializes the model on first call, reuses on subsequent calls.

    Args:
        image_path: Path to the image file
        question: Natural language question
        model_path: Optional path to model (uses default if None)

    Returns:
        dict: {answer, confidence, reasoning_trace, model, task}
    """
    global _vqa_instance
    if _vqa_instance is None:
        _vqa_instance = SatQueryVQA(model_path)
    return _vqa_instance.answer_vqa(image_path, question)


def answer_caption(image_path: str, model_path: str = None) -> dict:
    """Generate a caption for a remote sensing image."""
    global _vqa_instance
    if _vqa_instance is None:
        _vqa_instance = SatQueryVQA(model_path)
    return _vqa_instance.answer_caption(image_path)


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="SatQuery AI — Remote Sensing VQA Inference"
    )
    parser.add_argument("--image", type=str, required=True,
                        help="Path to image file (PNG, JPEG, TIFF, GeoTIFF)")
    parser.add_argument("--question", type=str, default=None,
                        help="Question about the image (omit for captioning)")
    parser.add_argument("--model-path", type=str, default=None,
                        help=f"Model path (default: {DEFAULT_LORA_PATH})")
    parser.add_argument("--caption", action="store_true",
                        help="Generate a caption instead of answering a question")
    args = parser.parse_args()

    if not HAS_TORCH:
        print("ERROR: PyTorch and transformers are required.")
        print("Install with: pip install -r requirements_local.txt")
        sys.exit(1)

    vqa = SatQueryVQA(args.model_path)

    if args.caption or args.question is None:
        result = vqa.answer_caption(args.image)
        print(f"\n  Image:   {args.image}")
        print(f"  Caption: {result['caption']}")
        print(f"  Confidence: {result['confidence']:.2f}")
    else:
        result = vqa.answer_vqa(args.image, args.question)
        print(f"\n  Image:    {args.image}")
        print(f"  Question: {args.question}")
        print(f"  Answer:   {result['answer']}")
        print(f"  Confidence: {result['confidence']:.2f}")

    print(f"\n  Full result: {result}")


if __name__ == "__main__":
    main()
