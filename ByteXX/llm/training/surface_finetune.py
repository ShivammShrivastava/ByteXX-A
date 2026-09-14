"""
SatQuery AI — Surface-Level Fine-Tuning Script
================================================
Uses standard transformers + PEFT (Python 3.13 compatible, no Unsloth).

"Surface level" settings:
  - LoRA rank r=4  (minimum — just teaches model the domain)
  - Only q_proj + v_proj (attention only — fastest)
  - 1000 VQA samples (small subset)
  - 1 epoch
  - ~1-2 hours on RTX 3050 8GB

Run:
    python training/surface_finetune.py
"""

import sys
import os
import json
import re
import time
import math
import gc

# Force UTF-8 output on Windows
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
DATASET_DIR   = r"C:\Users\Suryansh\OneDrive\Desktop\dataset"
TRAIN_JSON    = os.path.join(DATASET_DIR, "VRSBench_train.json")
TRAIN_IMG_DIR = os.path.join(DATASET_DIR, "Images_train", "Images_train")
OUTPUT_DIR    = r"C:\Users\Suryansh\OneDrive\Desktop\satquery-ai-vqa-lora"

MAX_SAMPLES   = 1000    # Surface-level: just 1000 samples
NUM_EPOCHS    = 1
BATCH_SIZE    = 1
GRAD_ACCUM    = 8       # Effective batch = 8
LEARNING_RATE = 3e-4
MAX_SEQ_LEN   = 256     # Short sequences — saves VRAM
LORA_RANK     = 4       # Minimum rank — surface level
LORA_ALPHA    = 8
LORA_TARGET   = ["q_proj", "v_proj"]   # Attention only

RS_SYSTEM_PROMPT = (
    "You are SatQuery AI, a remote sensing image analysis expert specializing in "
    "satellite and aerial imagery interpretation. You can identify land cover types, "
    "objects, spatial relationships, and scene characteristics in remote sensing images. "
    "Provide accurate, concise answers based on visual evidence in the image."
)

print("=" * 60)
print("  SatQuery AI — Surface-Level Fine-Tuning")
print("=" * 60)
print(f"  Samples   : {MAX_SAMPLES}")
print(f"  Epochs    : {NUM_EPOCHS}")
print(f"  LoRA rank : {LORA_RANK}")
print(f"  Targets   : {LORA_TARGET}")
print(f"  Output    : {OUTPUT_DIR}")
print("=" * 60)

# ─────────────────────────────────────────────
# Step 1: Check GPU
# ─────────────────────────────────────────────
import torch

assert torch.cuda.is_available(), "No GPU found! Make sure CUDA is available."
gpu_name = torch.cuda.get_device_name(0)
vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1024**3
print(f"\n[GPU] {gpu_name} | {vram_gb:.1f} GB VRAM")

# ─────────────────────────────────────────────
# Step 2: Load model in 4-bit
# ─────────────────────────────────────────────
print("\n[1/5] Loading Qwen2.5-VL-3B-Instruct in 4-bit...")
print("      (First run downloads ~6 GB — needs internet)")

from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct",
    quantization_config=quant_config,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True,
)

processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct",
    trust_remote_code=True,
)

vram_after_load = torch.cuda.memory_allocated() / 1024**3
print(f"  Model loaded! VRAM: {vram_after_load:.1f} GB")

# ─────────────────────────────────────────────
# Step 3: Add LoRA (surface level)
# ─────────────────────────────────────────────
print("\n[2/5] Applying surface-level LoRA (r=4, attention only)...")

from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training

model = prepare_model_for_kbit_training(
    model, use_gradient_checkpointing=True
)

lora_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    target_modules=LORA_TARGET,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.3f}%)")

# ─────────────────────────────────────────────
# Step 4: Load dataset
# ─────────────────────────────────────────────
print(f"\n[3/5] Loading {MAX_SAMPLES} VQA samples from local dataset...")

assert os.path.exists(TRAIN_JSON),    f"Not found: {TRAIN_JSON}"
assert os.path.isdir(TRAIN_IMG_DIR),  f"Not found: {TRAIN_IMG_DIR}"

def clean_text(text):
    text = text.replace("<image>", "").strip()
    text = re.sub(r"\[(vqa|caption|refer)\]\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"</?p>", "", text)
    return text.strip()

with open(TRAIN_JSON, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

samples = []
for item in raw_data:
    if len(samples) >= MAX_SAMPLES:
        break
    convs      = item.get("conversations", [])
    image_name = item.get("image", "")
    if not convs or len(convs) < 2 or not image_name:
        continue
    tag = convs[0].get("value", "").lower()
    if "[vqa]" not in tag:
        continue
    img_path = os.path.join(TRAIN_IMG_DIR, image_name)
    if not os.path.exists(img_path):
        continue
    samples.append({
        "image_path": img_path,
        "question":   clean_text(convs[0]["value"]),
        "answer":     convs[1]["value"].strip(),
    })

print(f"  Loaded {len(samples)} VQA samples")
print(f"  Sample: Q={samples[0]['question'][:60]}  A={samples[0]['answer']}")

# ─────────────────────────────────────────────
# Step 5: Train
# ─────────────────────────────────────────────
print(f"\n[4/5] Training ({len(samples)} samples x {NUM_EPOCHS} epoch)...")
print(f"      Estimated time: ~1-2 hours on {gpu_name}")
print(f"      Progress saved every 200 steps to: {OUTPUT_DIR}_checkpoints/\n")

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm

class VQADataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        image = Image.open(s["image_path"]).convert("RGB")
        return image, s["question"], s["answer"]

def collate_fn(batch):
    images, questions, answers = zip(*batch)

    # Build chat messages
    all_messages = []
    for q, a in zip(questions, answers):
        messages = [
            {"role": "system",    "content": RS_SYSTEM_PROMPT},
            {"role": "user",      "content": [
                {"type": "image"},
                {"type": "text", "text": q},
            ]},
            {"role": "assistant", "content": a},
        ]
        all_messages.append(messages)

    # Apply chat template to each message set
    texts = [
        processor.apply_chat_template(m, add_generation_prompt=False)
        for m in all_messages
    ]

    # Tokenize
    batch_inputs = processor(
        text=texts,
        images=list(images),
        padding=True,
        truncation=True,
        max_length=MAX_SEQ_LEN,
        return_tensors="pt",
    )

    # Labels = input_ids (causal LM)
    batch_inputs["labels"] = batch_inputs["input_ids"].clone()
    return batch_inputs

dataset    = VQADataset(samples)
dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=0,
)

# Optimizer
optimizer = AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=LEARNING_RATE,
    weight_decay=0.01,
)
total_steps = math.ceil(len(dataloader) / GRAD_ACCUM) * NUM_EPOCHS
scheduler   = CosineAnnealingLR(optimizer, T_max=total_steps)

model.train()
device    = next(model.parameters()).device
step      = 0
best_loss = float("inf")
log_every = 50
save_every = 200

os.makedirs(f"{OUTPUT_DIR}_checkpoints", exist_ok=True)

start_time = time.time()
print(f"{'Step':>6} | {'Loss':>8} | {'LR':>10} | {'VRAM':>6}")
print("-" * 40)

for epoch in range(NUM_EPOCHS):
    accum_loss = 0
    optimizer.zero_grad()

    for i, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}")):
        # Move to GPU
        batch = {k: v.to(device) if hasattr(v, "to") else v
                 for k, v in batch.items()}

        outputs = model(**batch)
        loss    = outputs.loss / GRAD_ACCUM
        loss.backward()
        accum_loss += loss.item()

        if (i + 1) % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step += 1

            if step % log_every == 0:
                vram = torch.cuda.memory_allocated() / 1024**3
                lr   = scheduler.get_last_lr()[0]
                print(f"{step:>6} | {accum_loss:>8.4f} | {lr:>10.2e} | {vram:>5.1f}G")
                accum_loss = 0

            # Save checkpoint
            if step % save_every == 0:
                ckpt_path = f"{OUTPUT_DIR}_checkpoints/step-{step}"
                model.save_pretrained(ckpt_path)
                print(f"  Checkpoint saved: {ckpt_path}")

elapsed = time.time() - start_time
print(f"\nTraining complete! Time: {elapsed/3600:.1f} hours")

# ─────────────────────────────────────────────
# Step 6: Save final adapter
# ─────────────────────────────────────────────
print(f"\n[5/5] Saving LoRA adapter to: {OUTPUT_DIR}")
os.makedirs(OUTPUT_DIR, exist_ok=True)
model.save_pretrained(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)

size_mb = sum(
    os.path.getsize(os.path.join(OUTPUT_DIR, f))
    for f in os.listdir(OUTPUT_DIR)
    if os.path.isfile(os.path.join(OUTPUT_DIR, f))
) / 1024**2

print(f"  Adapter saved! Size: {size_mb:.1f} MB")
print(f"  Files: {os.listdir(OUTPUT_DIR)}")
print()
print("=" * 60)
print("  DONE! Backend will auto-load this adapter.")
print(f"  Start backend: uvicorn backend.main:app --port 8000")
print("=" * 60)
