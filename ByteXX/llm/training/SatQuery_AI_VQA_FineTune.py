# ╔══════════════════════════════════════════════════════════════════════╗
# ║  SatQuery AI — Qwen2.5-VL-7B Fine-Tuning for Remote Sensing VQA   ║
# ║  Run on Google Colab Free Tier (T4 GPU)                            ║
# ║                                                                    ║
# ║  INSTRUCTIONS:                                                     ║
# ║  1. Open Google Colab → Runtime → Change runtime type → T4 GPU     ║
# ║  2. Upload train_combined.jsonl (from data/processed/)             ║
# ║  3. Copy each "# === CELL N ===" section into a separate cell      ║
# ║  4. Run cells 1-8 sequentially                                     ║
# ╚══════════════════════════════════════════════════════════════════════╝


# === CELL 1: Install Dependencies (~3-5 min) ===========================

# %%capture
# import os, re
# if "COLAB_" not in "".join(os.environ.keys()):
#     !pip install unsloth
# else:
#     import torch
#     v = re.match(r'[\d]{1,}\.[\d]{1,}', str(torch.__version__)).group(0)
#     xformers = 'xformers==' + {'2.10':'0.0.34','2.9':'0.0.33.post1','2.8':'0.0.32.post2'}.get(v, "0.0.34")
#     !pip install sentencepiece protobuf "datasets==4.3.0" "huggingface_hub>=0.34.0" hf_transfer
#     !pip install --no-deps unsloth_zoo bitsandbytes accelerate {xformers} peft trl triton unsloth
#     !pip install --no-deps --upgrade "torchao>=0.16.0"
# !pip install transformers==4.56.2
# !pip install --no-deps trl==0.22.2
# print("✅ All dependencies installed!")


# === CELL 2: Load Model (~5 min, downloads ~6 GB) ======================

# from unsloth import FastVisionModel
# import torch
#
# model, tokenizer = FastVisionModel.from_pretrained(
#     "unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit",
#     load_in_4bit=True,
#     use_gradient_checkpointing="unsloth",
# )
# print(f"✅ Model loaded! GPU: {torch.cuda.memory_allocated()/1024**3:.1f} GB")


# === CELL 3: Add LoRA Adapters (instant) ================================

# model = FastVisionModel.get_peft_model(
#     model,
#     finetune_vision_layers=True,
#     finetune_language_layers=True,
#     finetune_attention_modules=True,
#     finetune_mlp_modules=True,
#     r=16, lora_alpha=16, lora_dropout=0,
#     bias="none", random_state=3407,
#     use_rslora=False, loftq_config=None,
# )
# trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
# total = sum(p.numel() for p in model.parameters())
# print(f"✅ LoRA added! Trainable: {trainable:,}/{total:,} ({100*trainable/total:.2f}%)")


# === CELL 4: Load Dataset from HuggingFace (~10-15 min) =================

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
# print("Loading VRSBench from HuggingFace...")
# hf_dataset = load_dataset("xiang709/VRSBench", name="VRSBench", split="train")
# print(f"Raw: {len(hf_dataset)} samples | Columns: {hf_dataset.column_names}")
#
# def clean_text(text):
#     return re.sub(r"\[(?:vqa|caption|refer)\]\s*", "", text, flags=re.IGNORECASE).replace("<image>","").strip()
#
# converted = []
# for i, sample in enumerate(hf_dataset):
#     convs = sample.get("conversations", [])
#     image = sample.get("image", None)
#     if not convs or len(convs) < 2 or image is None:
#         continue
#     tag = convs[0].get("value", "").lower()
#     if "[vqa]" not in tag and "[caption]" not in tag:
#         continue
#     converted.append({
#         "messages": [
#             {"role":"system","content":[{"type":"text","text":RS_SYSTEM_PROMPT}]},
#             {"role":"user","content":[{"type":"image","image":image},{"type":"text","text":clean_text(convs[0]["value"])}]},
#             {"role":"assistant","content":[{"type":"text","text":convs[1]["value"].strip()}]},
#         ]
#     })
#     if (i+1) % 20000 == 0:
#         print(f"  {i+1:,}/{len(hf_dataset):,} | Converted: {len(converted):,}")
#
# dataset = Dataset.from_list(converted)
# print(f"✅ Dataset ready: {len(dataset):,} samples")


# === CELL 5: Train (~4-6 hours) =========================================

# from trl import SFTTrainer, SFTConfig
# from unsloth import is_bfloat16_supported, UnslothVisionDataCollator
# import torch
#
# use_bf16 = is_bfloat16_supported()
# print(f"Precision: {'bf16' if use_bf16 else 'fp16'}")
#
# trainer = SFTTrainer(
#     model=model, tokenizer=tokenizer,
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
#         fp16=not use_bf16,
#         bf16=use_bf16,
#         logging_steps=25,
#         save_strategy="steps",
#         save_steps=500,
#         save_total_limit=3,
#         output_dir="satquery_checkpoints",
#         optim="adamw_8bit",
#         max_seq_length=1024,
#         max_grad_norm=1.0,
#         seed=3407,
#         report_to="none",
#         remove_unused_columns=False,
#         dataset_text_field="",
#         dataset_kwargs={"skip_prepare_dataset": True},
#     ),
# )
#
# print("🚀 Training started... Keep this tab active!")
# stats = trainer.train()
# print(f"✅ Done! Time: {stats.metrics['train_runtime']/3600:.1f}h | Loss: {stats.metrics['train_loss']:.4f}")


# === CELL 6: Quick Test =================================================

# from unsloth import FastVisionModel
# from qwen_vl_utils import process_vision_info
# FastVisionModel.for_inference(model)
#
# test = dataset[0]
# img = test["messages"][1]["content"][0]["image"]
# q = test["messages"][1]["content"][1]["text"]
# expected = test["messages"][2]["content"][0]["text"]
#
# msgs = [{"role":"system","content":RS_SYSTEM_PROMPT},
#         {"role":"user","content":[{"type":"image","image":img},{"type":"text","text":q}]}]
# txt = tokenizer.apply_chat_template(msgs, add_generation_prompt=True)
# img_in, vid_in = process_vision_info(msgs)
# inp = tokenizer(text=[txt], images=img_in, videos=vid_in, padding=True, return_tensors="pt").to("cuda")
# out = model.generate(**inp, max_new_tokens=128, temperature=0.1, do_sample=False)
# gen = tokenizer.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0]
# print(f"Q: {q}\nExpected: {expected}\nGenerated: {gen}")


# === CELL 7: Save Adapter ===============================================

# model.save_pretrained("satquery-ai-vqa-lora")
# tokenizer.save_pretrained("satquery-ai-vqa-lora")
# print("✅ Saved locally!")


# === CELL 8: Push to HuggingFace Hub ====================================

# from huggingface_hub import login
# login()  # Paste your token from https://huggingface.co/settings/tokens
#
# HF_REPO = "YOUR_HF_USERNAME/satquery-ai-vqa-lora"  # ← CHANGE THIS!
# model.push_to_hub(HF_REPO, tokenizer=tokenizer)
# print(f"✅ Pushed to https://huggingface.co/{HF_REPO}")


# === CELL 9 (OPTIONAL): Download as ZIP =================================

# !zip -r satquery-ai-vqa-lora.zip satquery-ai-vqa-lora/
# from google.colab import files
# files.download("satquery-ai-vqa-lora.zip")
