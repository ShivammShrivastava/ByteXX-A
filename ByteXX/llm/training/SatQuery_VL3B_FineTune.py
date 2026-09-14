# ╔══════════════════════════════════════════════════════════════════════╗
# ║  SatQuery AI — Qwen2.5-VL-3B Fine-Tuning on VRSBench              ║
# ║  Google Colab Free Tier (T4 GPU, 16 GB VRAM)                       ║
# ║                                                                    ║
# ║  MODEL:  Qwen2.5-VL-3B-Instruct (4-bit via Unsloth)               ║
# ║  TASKS:  VQA + Captioning + Referring/Grounding                    ║
# ║  DATA:   VRSBench from HuggingFace (xiang709/VRSBench)            ║
# ║                                                                    ║
# ║  INSTRUCTIONS:                                                     ║
# ║  1. Open Google Colab → Runtime → Change runtime type → T4 GPU     ║
# ║  2. Copy each "# === CELL N ===" section into a separate cell      ║
# ║  3. Run cells 1-9 sequentially (Shift+Enter)                       ║
# ║  4. Total training time: ~2-3 hours on T4                          ║
# ║                                                                    ║
# ║  HF USERNAME: shivamw                                              ║
# ╚══════════════════════════════════════════════════════════════════════╝


# === CELL 1: Install Dependencies (~3-5 min) =============================
# Copy everything below this line into Colab Cell 1

%%capture
import os, re
# Colab-specific Unsloth install (handles xformers version matching)
if "COLAB_" not in "".join(os.environ.keys()):
    !pip install unsloth
else:
    import torch
    major_version = re.match(r'[\d]{1,}\.[\d]{1,}', str(torch.__version__)).group(0)
    xformers = 'xformers==' + {
        '2.10': '0.0.34', '2.9': '0.0.33.post1',
        '2.8': '0.0.32.post2', '2.7': '0.0.31.post2',
    }.get(major_version, "0.0.34")
    !pip install sentencepiece protobuf "datasets>=4.0.0" "huggingface_hub>=0.34.0" hf_transfer
    !pip install --no-deps unsloth_zoo bitsandbytes accelerate {xformers} peft trl triton unsloth
    !pip install --no-deps --upgrade "torchao>=0.16.0"

# Pin compatible versions
!pip install transformers==4.56.2
!pip install --no-deps trl==0.22.2
!pip install qwen-vl-utils

print("✅ All dependencies installed!")


# === CELL 2: Load Qwen2.5-VL-3B (~3 min, downloads ~3 GB) ===============
# Copy everything below this line into Colab Cell 2

from unsloth import FastVisionModel
import torch

model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/Qwen2.5-VL-3B-Instruct-bnb-4bit",
    load_in_4bit=True,
    use_gradient_checkpointing="unsloth",
)

vram_gb = torch.cuda.memory_allocated() / 1024**3
print(f"✅ Qwen2.5-VL-3B loaded! VRAM: {vram_gb:.1f} GB")
print(f"   GPU: {torch.cuda.get_device_name(0)}")


# === CELL 3: Add LoRA Adapters (instant) =================================
# Copy everything below this line into Colab Cell 3

model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=True,      # Fine-tune vision encoder
    finetune_language_layers=True,     # Fine-tune language model
    finetune_attention_modules=True,   # Fine-tune attention layers
    finetune_mlp_modules=True,         # Fine-tune MLP layers
    r=16,                              # LoRA rank
    lora_alpha=16,                     # LoRA alpha (= rank for stable training)
    lora_dropout=0,                    # No dropout (Unsloth recommendation)
    bias="none",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"✅ LoRA adapters added!")
print(f"   Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
print(f"   VRAM: {torch.cuda.memory_allocated()/1024**3:.1f} GB")


# === CELL 4: Load Dataset from Google Drive (~2-5 min) ====================
# Copy everything below this line into Colab Cell 4
#
# PREREQUISITE: Upload your dataset folder to Google Drive:
#   Google Drive/dataset/
#     ├── VRSBench_train.json
#     └── Images_train/
#         └── Images_train/     (nested — this is how VRSBench is structured)
#             ├── 00002_0000.png
#             ├── ...

from google.colab import drive
from datasets import Dataset
from PIL import Image
import json
import re
import os

# ── Mount Google Drive ──
drive.mount('/content/drive')

# ── Configure paths ──
# Change this if your dataset folder has a different name/location in Drive
DRIVE_DATASET_DIR = "/content/drive/MyDrive/dataset"
TRAIN_JSON = os.path.join(DRIVE_DATASET_DIR, "VRSBench_train.json")
TRAIN_IMAGES_DIR = os.path.join(DRIVE_DATASET_DIR, "Images_train", "Images_train")

# Verify paths
assert os.path.exists(TRAIN_JSON), f"Not found: {TRAIN_JSON}"
assert os.path.isdir(TRAIN_IMAGES_DIR), f"Not found: {TRAIN_IMAGES_DIR}"
print(f"✅ Dataset found on Google Drive!")
print(f"   JSON: {TRAIN_JSON}")
print(f"   Images: {TRAIN_IMAGES_DIR}")

# ── System prompt ──
RS_SYSTEM_PROMPT = (
    "You are SatQuery AI, a remote sensing image analysis expert specializing in "
    "satellite and aerial imagery interpretation. You can identify land cover types, "
    "objects, spatial relationships, and scene characteristics in remote sensing images. "
    "Provide accurate, concise answers based on visual evidence in the image."
)

def clean_text(text):
    """Remove task tags and <image> tokens."""
    cleaned = text.replace("<image>", "").strip()
    cleaned = re.sub(r"\[(vqa|caption|refer)\]\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"</?p>", "", cleaned)
    return cleaned.strip()

# ── Load JSON ──
print("\n📥 Loading VRSBench_train.json...")
with open(TRAIN_JSON, "r", encoding="utf-8") as f:
    raw_data = json.load(f)
print(f"   Total entries: {len(raw_data):,}")

# ── Convert to Qwen2.5-VL chat format ──
print("\n🔄 Converting to training format...")
converted = []
task_counts = {"vqa": 0, "caption": 0, "refer": 0, "skipped": 0, "img_missing": 0}

for i, sample in enumerate(raw_data):
    convs = sample.get("conversations", [])
    image_name = sample.get("image", "")

    if not convs or len(convs) < 2 or not image_name:
        task_counts["skipped"] += 1
        continue

    # Build image path
    img_path = os.path.join(TRAIN_IMAGES_DIR, image_name)
    if not os.path.exists(img_path):
        task_counts["img_missing"] += 1
        continue

    tag = convs[0].get("value", "").lower()
    human_text = clean_text(convs[0]["value"])
    gpt_text = convs[1]["value"].strip()

    # Classify task
    if "[vqa]" in tag:
        task_counts["vqa"] += 1
    elif "[caption]" in tag:
        task_counts["caption"] += 1
    elif "[refer]" in tag:
        task_counts["refer"] += 1
    else:
        task_counts["skipped"] += 1
        continue

    # Load image as PIL (lazy — will be read during training)
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as e:
        task_counts["skipped"] += 1
        continue

    # Build Qwen2.5-VL chat format
    converted.append({
        "messages": [
            {
                "role": "system",
                "content": [{"type": "text", "text": RS_SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": human_text},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": gpt_text}],
            },
        ]
    })

    # Progress update
    if (i + 1) % 20000 == 0:
        print(f"   {i+1:,}/{len(raw_data):,} processed | Converted: {len(converted):,}")

dataset = Dataset.from_list(converted)
print(f"\n✅ Dataset ready: {len(dataset):,} training samples")
print(f"   VQA:       {task_counts['vqa']:,}")
print(f"   Caption:   {task_counts['caption']:,}")
print(f"   Referring: {task_counts['refer']:,}")
print(f"   Skipped:   {task_counts['skipped']:,}")
if task_counts["img_missing"] > 0:
    print(f"   ⚠ Missing images: {task_counts['img_missing']:,}")


# === CELL 4-ALT: Resume/Quick Dataset (use if Cell 4 fails or restarts) ==
# Only use this cell if Colab disconnected and you need to reload.
# This loads a smaller subset for faster recovery.

# from datasets import load_dataset, Dataset
# import re
#
# RS_SYSTEM_PROMPT = (
#     "You are SatQuery AI, a remote sensing image analysis expert specializing in "
#     "satellite and aerial imagery interpretation. You can identify land cover types, "
#     "objects, spatial relationships, and scene characteristics in remote sensing images. "
#     "Provide accurate, concise answers based on visual evidence in the image."
# )
#
# def clean_text(text):
#     cleaned = text.replace("<image>", "").strip()
#     cleaned = re.sub(r"\[(vqa|caption|refer)\]\s*", "", cleaned, flags=re.IGNORECASE)
#     cleaned = re.sub(r"</?p>", "", cleaned)
#     return cleaned.strip()
#
# print("📥 Loading VRSBench (streaming, max 50K samples)...")
# hf_dataset = load_dataset("xiang709/VRSBench", name="VRSBench", split="train", streaming=True)
# converted = []
# for i, sample in enumerate(hf_dataset):
#     if i >= 50000:
#         break
#     convs = sample.get("conversations", [])
#     image = sample.get("image", None)
#     if not convs or len(convs) < 2 or image is None:
#         continue
#     tag = convs[0].get("value", "").lower()
#     if not any(t in tag for t in ["[vqa]", "[caption]", "[refer]"]):
#         continue
#     human_text = clean_text(convs[0]["value"])
#     gpt_text = convs[1]["value"].strip()
#     converted.append({
#         "messages": [
#             {"role": "system", "content": [{"type": "text", "text": RS_SYSTEM_PROMPT}]},
#             {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": human_text}]},
#             {"role": "assistant", "content": [{"type": "text", "text": gpt_text}]},
#         ]
#     })
#     if (i + 1) % 10000 == 0:
#         print(f"   {i+1:,} streamed | Converted: {len(converted):,}")
#
# dataset = Dataset.from_list(converted)
# print(f"✅ Quick dataset: {len(dataset):,} samples")


# === CELL 5: Train (~2-3 hours on T4) ====================================
# Copy everything below this line into Colab Cell 5

from trl import SFTTrainer, SFTConfig
from unsloth import is_bfloat16_supported, UnslothVisionDataCollator
import torch

use_bf16 = is_bfloat16_supported()
print(f"🔧 Precision: {'bf16' if use_bf16 else 'fp16'}")
print(f"🔧 Dataset: {len(dataset):,} samples")
print(f"🔧 Effective batch size: 16 (batch=2 × grad_accum=8)")
print(f"🔧 Epochs: 2")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    data_collator=UnslothVisionDataCollator(model, tokenizer),
    train_dataset=dataset,
    args=SFTConfig(
        # ── Batch size ──
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,  # Effective batch = 2 × 8 = 16

        # ── Training schedule ──
        num_train_epochs=2,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.01,
        max_grad_norm=1.0,

        # ── Precision ──
        fp16=not use_bf16,
        bf16=use_bf16,

        # ── Logging & saving ──
        logging_steps=25,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        output_dir="satquery_checkpoints",

        # ── Optimizer ──
        optim="adamw_8bit",

        # ── Sequence length ──
        max_seq_length=1024,

        # ── Misc ──
        seed=3407,
        report_to="none",
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
    ),
)

print("🚀 Training started! Keep this tab active.")
print("   Estimated time: ~2-3 hours on T4")
print("   Checkpoints saved every 500 steps to satquery_checkpoints/")
print()

stats = trainer.train()

print(f"\n✅ Training complete!")
print(f"   Total time: {stats.metrics['train_runtime']/3600:.1f} hours")
print(f"   Final loss: {stats.metrics['train_loss']:.4f}")
print(f"   VRAM peak: {torch.cuda.max_memory_allocated()/1024**3:.1f} GB")


# === CELL 5-ALT: Resume Training (if Colab disconnected) =================
# Use this if your session was interrupted. Re-run Cells 1-4 first,
# then use this cell instead of Cell 5.

# trainer = SFTTrainer(
#     model=model,
#     tokenizer=tokenizer,
#     data_collator=UnslothVisionDataCollator(model, tokenizer),
#     train_dataset=dataset,
#     args=SFTConfig(
#         per_device_train_batch_size=2,
#         gradient_accumulation_steps=8,
#         num_train_epochs=2,
#         learning_rate=2e-4,
#         lr_scheduler_type="cosine",
#         warmup_ratio=0.03,
#         weight_decay=0.01,
#         max_grad_norm=1.0,
#         fp16=not use_bf16,
#         bf16=use_bf16,
#         logging_steps=25,
#         save_strategy="steps",
#         save_steps=500,
#         save_total_limit=3,
#         output_dir="satquery_checkpoints",
#         optim="adamw_8bit",
#         max_seq_length=1024,
#         seed=3407,
#         report_to="none",
#         remove_unused_columns=False,
#         dataset_text_field="",
#         dataset_kwargs={"skip_prepare_dataset": True},
#     ),
# )
# print("🔄 Resuming from last checkpoint...")
# stats = trainer.train(resume_from_checkpoint=True)
# print(f"✅ Done! Loss: {stats.metrics['train_loss']:.4f}")


# === CELL 6: Quick Inference Test ========================================
# Copy everything below this line into Colab Cell 6

from unsloth import FastVisionModel
from qwen_vl_utils import process_vision_info

# Switch to inference mode
FastVisionModel.for_inference(model)

# Test on first training sample
test_sample = dataset[0]
test_img = test_sample["messages"][1]["content"][0]["image"]
test_q = test_sample["messages"][1]["content"][1]["text"]
expected_ans = test_sample["messages"][2]["content"][0]["text"]

# Build inference prompt
messages = [
    {"role": "system", "content": RS_SYSTEM_PROMPT},
    {"role": "user", "content": [
        {"type": "image", "image": test_img},
        {"type": "text", "text": test_q},
    ]},
]

text_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)
inputs = tokenizer(
    text=[text_prompt], images=image_inputs, videos=video_inputs,
    padding=True, return_tensors="pt"
).to("cuda")

with torch.no_grad():
    output_ids = model.generate(
        **inputs, max_new_tokens=128,
        temperature=0.1, do_sample=False,
    )

generated = tokenizer.batch_decode(
    output_ids[:, inputs["input_ids"].shape[1]:],
    skip_special_tokens=True,
)[0].strip()

print("=" * 60)
print("  Quick Inference Test")
print("=" * 60)
print(f"  Q: {test_q}")
print(f"  Expected: {expected_ans}")
print(f"  Generated: {generated}")
print(f"  Match: {'✅' if generated.lower().strip() == expected_ans.lower().strip() else '❌'}")
print("=" * 60)

# Test a second sample
test2 = dataset[min(100, len(dataset)-1)]
test2_q = test2["messages"][1]["content"][1]["text"]
test2_img = test2["messages"][1]["content"][0]["image"]
test2_expected = test2["messages"][2]["content"][0]["text"]

messages2 = [
    {"role": "system", "content": RS_SYSTEM_PROMPT},
    {"role": "user", "content": [
        {"type": "image", "image": test2_img},
        {"type": "text", "text": test2_q},
    ]},
]
text2 = tokenizer.apply_chat_template(messages2, add_generation_prompt=True)
imgs2, vids2 = process_vision_info(messages2)
inp2 = tokenizer(text=[text2], images=imgs2, videos=vids2, padding=True, return_tensors="pt").to("cuda")
with torch.no_grad():
    out2 = model.generate(**inp2, max_new_tokens=128, temperature=0.1, do_sample=False)
gen2 = tokenizer.batch_decode(out2[:, inp2["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
print(f"\n  Q: {test2_q}")
print(f"  Expected: {test2_expected}")
print(f"  Generated: {gen2}")


# === CELL 7: Save LoRA Adapter Locally ===================================
# Copy everything below this line into Colab Cell 7

ADAPTER_NAME = "satquery-ai-vqa-lora"

model.save_pretrained(ADAPTER_NAME)
tokenizer.save_pretrained(ADAPTER_NAME)

import os
adapter_size = sum(
    os.path.getsize(os.path.join(ADAPTER_NAME, f))
    for f in os.listdir(ADAPTER_NAME)
    if os.path.isfile(os.path.join(ADAPTER_NAME, f))
) / (1024 * 1024)
print(f"✅ Adapter saved to '{ADAPTER_NAME}/' ({adapter_size:.1f} MB)")
print(f"   Files: {os.listdir(ADAPTER_NAME)}")


# === CELL 8: Push to HuggingFace Hub =====================================
# Copy everything below this line into Colab Cell 8

from huggingface_hub import login

# Login to HuggingFace (paste your token when prompted)
# Get token from: https://huggingface.co/settings/tokens (Write access)
login()

HF_REPO = "shivamw/satquery-ai-vqa-lora"

print(f"📤 Pushing to https://huggingface.co/{HF_REPO} ...")
model.push_to_hub(HF_REPO, tokenizer=tokenizer)
print(f"✅ Pushed! Model available at: https://huggingface.co/{HF_REPO}")
print(f"   Use this path in your FastAPI backend config.")


# === CELL 9: Download Adapter as ZIP (Optional) ==========================
# Copy everything below this line into Colab Cell 9

!zip -r satquery-ai-vqa-lora.zip satquery-ai-vqa-lora/
from google.colab import files
files.download("satquery-ai-vqa-lora.zip")
print("✅ Download started! Save the ZIP for local inference.")
