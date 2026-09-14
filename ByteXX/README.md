# React + Vite

This template provides a minimal setup to get React working in Vite with HMR and some Oxlint rules.

Currently, two official plugins are available:

- [@vitejs/plugin-react](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react) uses [Oxc](https://oxc.rs)
- [@vitejs/plugin-react-swc](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react-swc) uses [SWC](https://swc.rs/)

## React Compiler

The React Compiler is not enabled on this template because of its impact on dev & build performances. To add it, see [this documentation](https://react.dev/learn/react-compiler/installation).

## Expanding the Oxlint configuration

If you are developing a production application, we recommend using TypeScript with type-aware lint rules enabled. Check out the [TS template](https://github.com/vitejs/vite/tree/main/packages/create-vite/template-react-ts) for information on how to integrate TypeScript and Oxlint's TypeScript related rules in your project.

---

# SatQuery AI — Single-Image VQA Specialist

Fine-tuned **Qwen2.5-VL-7B-Instruct** for remote sensing Visual Question Answering (VQA) and image captioning using the **VRSBench** dataset.

This is the mandatory VQA baseline component for the SatQuery AI agentic pipeline.

## Quick Start (3 Steps)

### Step 1: Verify Dataset (Local — CPU only)
```bash
cd Satellite
python data/prepare_data.py --dataset-dir "C:\Users\IPS\Desktop\dataset"
```

### Step 2: Fine-Tune on Google Colab (Free T4 GPU)
1. Open [Google Colab](https://colab.research.google.com)
2. Create a new notebook → Runtime → Change runtime type → **T4 GPU**
3. Open `training/SatQuery_AI_VQA_FineTune.py`
4. Copy each `CELL` section into separate Colab cells
5. Run cells sequentially (Shift+Enter)
6. After training, the LoRA adapter is pushed to your HuggingFace Hub

### Step 3: Run Inference Locally (RTX 3050 — 4-bit)
```bash
pip install -r requirements_local.txt

# VQA
python inference/infer_vqa.py \
    --image "path/to/satellite_image.png" \
    --question "What land cover is dominant in this image?" \
    --model-path "YOUR_HF_USERNAME/satquery-ai-vqa-lora"

# Captioning
python inference/infer_vqa.py \
    --image "path/to/satellite_image.png" \
    --caption \
    --model-path "YOUR_HF_USERNAME/satquery-ai-vqa-lora"
```

---

## Project Structure

```
Satellite/
├── README.md                                # This file
├── requirements_local.txt                   # Local inference dependencies
│
├── data/
│   └── prepare_data.py                      # Dataset verification (CPU)
│
├── training/
│   └── SatQuery_AI_VQA_FineTune.py          # Colab training script
│
├── evaluation/
│   └── evaluate.py                          # VRSBench VQA eval
│
└── inference/
    └── infer_vqa.py                         # Inference API (agentic tool)
```

---

## Training Details

| Parameter | Value |
|-----------|-------|
| Base Model | `Qwen/Qwen2.5-VL-7B-Instruct` |
| Quantization | 4-bit NF4 (via Unsloth) |
| Method | QLoRA |
| LoRA Rank | 16 |
| LoRA Alpha | 16 |
| Learning Rate | 2e-4 |
| Scheduler | Cosine with 3% warmup |
| Epochs | 2 |
| Effective Batch Size | 16 (batch=2 × grad_accum=8) |
| Training Samples | ~106K (85K VQA + 20K caption) |
| Precision | FP16 (T4 GPU) |

### Dataset: VRSBench

| Split | Samples | Images |
|-------|---------|--------|
| Train (VQA) | 85,838 | 20,264 |
| Train (Caption) | 20,264 | 20,264 |
| Eval (VQA) | 37,409 | 9,349 |

**VQA Question Types**: object color, quantity, existence, position, category, direction, size, shape, scene type, reasoning, image, rural/urban.

### Hardware Requirements

| Task | Platform | GPU | VRAM |
|------|----------|-----|------|
| Data prep | Local | None | — |
| Training | Colab Free | T4 | ~10-12 GB |
| Inference | Local | RTX 3050 | ~5 GB |

---

## Evaluation

Run evaluation on VRSBench VQA test split:

```bash
# On Colab (after training)
python evaluation/evaluate.py \
    --model-path "./satquery-ai-vqa-lora" \
    --dataset-dir "/tmp/vrsbench"

# Locally
python evaluation/evaluate.py \
    --model-path "YOUR_HF_USERNAME/satquery-ai-vqa-lora" \
    --dataset-dir "C:\Users\IPS\Desktop\dataset"

# Quick test (100 samples)
python evaluation/evaluate.py \
    --model-path "YOUR_HF_USERNAME/satquery-ai-vqa-lora" \
    --dataset-dir "C:\Users\IPS\Desktop\dataset" \
    --max-samples 100
```

Output includes per-type accuracy breakdown and a markdown report.

---

## Inference API (for Agentic Controller)

The inference module is designed to be called by the SatQuery AI agentic controller:

```python
from inference.infer_vqa import answer_vqa, answer_caption

# VQA
result = answer_vqa("path/to/image.tiff", "What objects are visible?")
# {
#     "answer": "buildings and roads",
#     "confidence": 0.87,
#     "reasoning_trace": "Analyzed remote sensing image...",
#     "model": "satquery-ai-vqa",
#     "task": "single_image_vqa"
# }

# Captioning
result = answer_caption("path/to/image.png")
# {
#     "caption": "The image shows an urban area with...",
#     "confidence": 0.82,
#     "reasoning_trace": "...",
#     "model": "satquery-ai-vqa",
#     "task": "single_image_caption"
# }
```

### Supported Input Formats
- **PNG, JPEG** — Direct loading
- **TIFF** — Via Pillow
- **GeoTIFF** — Via rasterio (auto RGB band selection)

---

## Colab Training — Step by Step

1. **Go to** https://colab.research.google.com
2. **New notebook** → Runtime → Change runtime type → **T4 GPU**
3. **Open** `training/SatQuery_AI_VQA_FineTune.py` in a text editor
4. **Copy Cell 1** (Install) into first Colab cell → Run (takes ~3-5 min)
5. **Copy Cell 2** (Load model) → Run (downloads ~6 GB, takes ~5 min)
6. **Copy Cell 3** (LoRA adapters) → Run (instant)
7. **Copy Cell 4** or **Cell 4-ALT** (Load dataset) → Run (takes ~10-15 min)
8. **Copy Cell 5** (Train) → Run (takes ~4-6 hours)
9. **Copy Cell 6** (Quick test) → Run (verify it works)
10. **Copy Cell 7** (Save & push to HF Hub) → Run

### If Session Disconnects
- Your checkpoints are saved every 500 steps
- Reconnect, re-run Cells 1-4, then use **Cell 5-ALT** to resume

### Kaggle Alternative
Same script works on Kaggle — just enable GPU and Internet in notebook settings. Kaggle gives 30 hours/week of free GPU.

---

## HuggingFace Setup

1. Create free account at https://huggingface.co
2. Go to https://huggingface.co/settings/tokens
3. Create a new token with "Write" access
4. In Colab Cell 7, you'll be prompted to enter this token
5. Replace `YOUR_HF_USERNAME` in the script with your username

---

## Architecture

```
┌──────────────────────────────────────────────────┐
│                 SatQuery AI Agent                 │
│  ┌────────────┐  ┌───────────┐  ┌─────────────┐ │
│  │  VQA Tool  │  │ Caption   │  │ Change Det. │ │
│  │ (this repo)│  │   Tool    │  │    Tool     │ │
│  └─────┬──────┘  └─────┬─────┘  └──────┬──────┘ │
│        │               │               │        │
│        └───────────┬────┘               │        │
│                    │                    │        │
│        ┌───────────▼────────────┐       │        │
│        │  Qwen2.5-VL-7B + LoRA │       │        │
│        │  (Fine-tuned on        │       │        │
│        │   VRSBench VQA+Cap)    │       │        │
│        └────────────────────────┘       │        │
└──────────────────────────────────────────────────┘
```

The VQA and Caption tools share the same fine-tuned model. Other specialist tools (change detection, optical-SAR fusion) will use separate models and are handled by other team members.

---

## License

Dataset: VRSBench — CC BY-NC 4.0
Base Model: Qwen2.5-VL — Apache 2.0
