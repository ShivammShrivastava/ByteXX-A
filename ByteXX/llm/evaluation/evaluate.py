"""
SatQuery AI — Evaluate Fine-Tuned Model on VRSBench VQA Test Split
===================================================================
Can run on:
  - Google Colab (after training, in same session)
  - Locally (with 4-bit model, RTX 3050 8GB is sufficient)

Usage (local):
    python evaluation/evaluate.py \
        --model-path "YOUR_HF_USERNAME/satquery-ai-vqa-lora" \
        --dataset-dir "C:\Users\Suryansh\OneDrive\Desktop\dataset" \
        --output-dir "outputs/results"

Usage (Colab, after training):
    python evaluation/evaluate.py \
        --model-path "./satquery-ai-vqa-lora" \
        --dataset-dir "/tmp/vrsbench" \
        --output-dir "./results"
"""

import json
import os
import sys
import argparse
import time
from pathlib import Path
from collections import Counter, defaultdict

try:
    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from peft import PeftModel
    from PIL import Image
    from tqdm import tqdm
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("WARNING: torch/transformers not installed. Install with:")
    print("  pip install -r requirements_local.txt")


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
DEFAULT_DATASET_DIR = r"C:\Users\Suryansh\OneDrive\Desktop\dataset"
DEFAULT_MODEL_PATH = "YOUR_HF_USERNAME/satquery-ai-vqa-lora"
BASE_MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"

RS_SYSTEM_PROMPT = (
    "You are SatQuery AI, a remote sensing image analysis expert specializing in "
    "satellite and aerial imagery interpretation. You can identify land cover types, "
    "objects, spatial relationships, and scene characteristics in remote sensing images. "
    "Provide accurate, concise answers based on visual evidence in the image."
)


# ─────────────────────────────────────────────
# Model Loading
# ─────────────────────────────────────────────
def load_model(model_path: str, load_in_4bit: bool = True):
    """Load the fine-tuned model (base + LoRA adapter)."""
    from transformers import BitsAndBytesConfig

    print(f"Loading model from: {model_path}")

    quantization_config = None
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    # Check if it's a LoRA adapter or merged model
    adapter_config_path = os.path.join(model_path, "adapter_config.json")
    is_lora = os.path.exists(adapter_config_path)

    if is_lora or (not os.path.exists(model_path) and "/" in model_path):
        # Load base model + LoRA
        print(f"  Loading base model: {BASE_MODEL_NAME}")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            BASE_MODEL_NAME,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        print(f"  Loading LoRA adapter: {model_path}")
        model = PeftModel.from_pretrained(model, model_path)
        model = model.merge_and_unload()
    else:
        # Load merged model directly
        print(f"  Loading merged model: {model_path}")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16,
        )

    processor = AutoProcessor.from_pretrained(
        model_path if not is_lora else BASE_MODEL_NAME
    )

    model.eval()
    print(f"  ✓ Model loaded. VRAM: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")
    return model, processor


# ─────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────
def run_inference(model, processor, image: Image.Image, question: str) -> str:
    """Run VQA inference on a single image-question pair."""
    messages = [
        {"role": "system", "content": RS_SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": question},
        ]},
    ]

    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(
        text=[prompt],
        images=[image],
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=64,
            temperature=0.1,
            do_sample=False,
        )

    # Decode only the generated tokens
    generated = processor.batch_decode(
        output_ids[:, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )[0]

    return generated.strip()


# ─────────────────────────────────────────────
# Evaluation Metrics
# ─────────────────────────────────────────────
def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    ans = answer.strip().lower()
    # Remove trailing punctuation
    ans = ans.rstrip(".,!?;:")
    # Common normalizations
    replacements = {
        "yes, ": "yes",
        "no, ": "no",
        "it is ": "",
        "there is ": "",
        "there are ": "",
    }
    for old, new in replacements.items():
        if ans.startswith(old):
            ans = new + ans[len(old):]
    return ans.strip()


def compute_accuracy(predictions: list, ground_truths: list) -> float:
    """Compute exact match accuracy with normalization."""
    if not predictions:
        return 0.0
    correct = sum(
        1 for pred, gt in zip(predictions, ground_truths)
        if normalize_answer(pred) == normalize_answer(gt)
    )
    return correct / len(predictions)


# ─────────────────────────────────────────────
# Main Evaluation
# ─────────────────────────────────────────────
def evaluate(model_path: str, dataset_dir: str, output_dir: str,
             max_samples: int = None, load_in_4bit: bool = True):
    """Run full evaluation on VRSBench VQA test split."""

    print("=" * 70)
    print("  SatQuery AI — VRSBench VQA Evaluation")
    print("=" * 70)

    # ── Load evaluation data ──
    eval_path = os.path.join(dataset_dir, "VRSBench_EVAL_vqa.json")
    if not os.path.exists(eval_path):
        print(f"ERROR: Eval file not found: {eval_path}")
        return

    with open(eval_path, "r") as f:
        eval_data = json.load(f)

    if max_samples:
        eval_data = eval_data[:max_samples]

    print(f"\n  Eval samples: {len(eval_data):,}")

    # ── Load model ──
    model, processor = load_model(model_path, load_in_4bit)

    # ── Run inference ──
    print(f"\n  Running inference...")
    results = []
    type_preds = defaultdict(list)
    type_gts = defaultdict(list)
    all_preds = []
    all_gts = []

    start_time = time.time()

    for i, sample in enumerate(tqdm(eval_data, desc="Evaluating")):
        image_name = sample["image_id"]
        question = sample["question"]
        ground_truth = sample["ground_truth"]
        q_type = sample.get("type", "unknown")

        # Load image
        img_path = os.path.join(dataset_dir, "Images_val", image_name)
        if not os.path.exists(img_path):
            continue

        try:
            image = Image.open(img_path).convert("RGB")
            prediction = run_inference(model, processor, image, question)
        except Exception as e:
            prediction = f"ERROR: {str(e)}"

        results.append({
            "question_id": sample.get("question_id", i),
            "image_id": image_name,
            "question": question,
            "ground_truth": ground_truth,
            "prediction": prediction,
            "type": q_type,
            "correct": normalize_answer(prediction) == normalize_answer(ground_truth),
        })

        type_preds[q_type].append(prediction)
        type_gts[q_type].append(ground_truth)
        all_preds.append(prediction)
        all_gts.append(ground_truth)

        # Progress update every 500 samples
        if (i + 1) % 500 == 0:
            current_acc = compute_accuracy(all_preds, all_gts)
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta = (len(eval_data) - i - 1) / rate
            print(f"  [{i+1}/{len(eval_data)}] Acc: {current_acc:.3f} | "
                  f"Rate: {rate:.1f} samples/s | ETA: {eta/60:.0f} min")

    total_time = time.time() - start_time

    # ── Compute metrics ──
    print(f"\n{'=' * 70}")
    print(f"  RESULTS")
    print(f"{'=' * 70}")

    overall_acc = compute_accuracy(all_preds, all_gts)
    print(f"\n  Overall Accuracy: {overall_acc:.4f} ({overall_acc*100:.1f}%)")
    print(f"  Total samples evaluated: {len(results):,}")
    print(f"  Total time: {total_time/60:.1f} minutes")
    print(f"  Speed: {len(results)/total_time:.1f} samples/second")

    # Per-type accuracy
    print(f"\n  {'Question Type':<22s} {'Count':>6s}  {'Accuracy':>10s}")
    print(f"  {'─' * 42}")

    type_results = {}
    for q_type in sorted(type_preds.keys()):
        acc = compute_accuracy(type_preds[q_type], type_gts[q_type])
        count = len(type_preds[q_type])
        print(f"  {q_type:<22s} {count:>6,}  {acc:>9.1%}")
        type_results[q_type] = {"accuracy": acc, "count": count}

    print(f"  {'─' * 42}")
    print(f"  {'OVERALL':<22s} {len(all_preds):>6,}  {overall_acc:>9.1%}")

    # ── Save results ──
    os.makedirs(output_dir, exist_ok=True)

    # Full predictions
    results_path = os.path.join(output_dir, "vqa_predictions.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # Metrics summary
    metrics = {
        "model": model_path,
        "dataset": "VRSBench_EVAL_vqa",
        "total_samples": len(results),
        "overall_accuracy": overall_acc,
        "per_type_accuracy": type_results,
        "total_time_seconds": total_time,
        "samples_per_second": len(results) / total_time,
    }
    metrics_path = os.path.join(output_dir, "vqa_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Markdown report
    report_lines = [
        "# SatQuery AI — VQA Evaluation Results\n",
        f"**Model**: `{model_path}`\n",
        f"**Dataset**: VRSBench VQA Test Split ({len(results):,} samples)\n",
        f"**Overall Accuracy**: **{overall_acc:.1%}**\n",
        f"**Evaluation Time**: {total_time/60:.1f} minutes\n",
        "\n## Per-Type Accuracy\n",
        "| Question Type | Count | Accuracy |",
        "|:---|---:|---:|",
    ]
    for q_type in sorted(type_results.keys()):
        r = type_results[q_type]
        report_lines.append(f"| {q_type} | {r['count']:,} | {r['accuracy']:.1%} |")
    report_lines.append(f"| **OVERALL** | **{len(results):,}** | **{overall_acc:.1%}** |")

    report_path = os.path.join(output_dir, "vqa_report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))

    print(f"\n  Results saved to: {output_dir}/")
    print(f"    - vqa_predictions.json  (full predictions)")
    print(f"    - vqa_metrics.json      (metrics summary)")
    print(f"    - vqa_report.md         (markdown report)")


if __name__ == "__main__":
    if not HAS_TORCH:
        print("ERROR: PyTorch and transformers are required.")
        print("Install with: pip install -r requirements_local.txt")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="SatQuery AI — VQA Evaluation")
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH,
                        help="Path to fine-tuned model (HF repo or local path)")
    parser.add_argument("--dataset-dir", type=str, default=DEFAULT_DATASET_DIR,
                        help="Path to VRSBench dataset directory")
    parser.add_argument("--output-dir", type=str, default="outputs/results",
                        help="Directory to save evaluation results")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max samples to evaluate (None = all)")
    parser.add_argument("--no-4bit", action="store_true",
                        help="Disable 4-bit quantization (uses more VRAM)")
    args = parser.parse_args()

    evaluate(
        model_path=args.model_path,
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        load_in_4bit=not args.no_4bit,
    )
