"""
SatQuery AI — Backend Configuration
=====================================
Now powered by Google Gemma 4 31B via OpenRouter (cloud inference).
No local GPU/VRAM required — queries are sent to OpenRouter's free API.

All values can be overridden via environment variables.
"""
import os

# ─────────────────────────────────────────────
# OpenRouter / Gemma 4
# ─────────────────────────────────────────────
OPENROUTER_API_KEY = os.environ.get(
    "OPENROUTER_API_KEY",
    "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"
)

# Primary model: Google Gemma 4 31B Instruct (free tier on OpenRouter)
OPENROUTER_MODEL = os.environ.get(
    "OPENROUTER_MODEL",
    "google/gemma-4-31b-it:free"
)

# Fallback models tried in order if primary is rate-limited (429).
# List fetched from OpenRouter /models endpoint — only confirmed :free + vision models.
OPENROUTER_FALLBACK_MODELS = [
    m.strip() for m in os.environ.get(
        "OPENROUTER_FALLBACK_MODELS",
        ",".join([
            "google/gemma-4-26b-a4b-it:free",        # Google Gemma 4 — confirmed free+vision
            "inclusionai/ling-3.0-flash-vl:free",    # Ling VL — confirmed free+vision
            "thinkingmachines/inkling:free",          # Inkling — confirmed free+vision
        ])
    ).split(",") if m.strip()
]

OPENROUTER_BASE_URL = os.environ.get(
    "OPENROUTER_BASE_URL",
    "https://openrouter.ai/api/v1"
)

# HTTP-Referer and site title sent to OpenRouter (good practice)
OPENROUTER_REFERER  = os.environ.get("OPENROUTER_REFERER",  "https://bytex.satquery.ai")
OPENROUTER_SITENAME = os.environ.get("OPENROUTER_SITENAME", "ByteX SatQuery AI")

# Max new tokens for each task type (1024 avoids finish_reason=length truncation)
MAX_NEW_TOKENS_VQA     = int(os.environ.get("SATQUERY_MAX_TOKENS_VQA",     "1024"))
MAX_NEW_TOKENS_CAPTION = int(os.environ.get("SATQUERY_MAX_TOKENS_CAPTION", "1024"))
MAX_NEW_TOKENS_REFER   = int(os.environ.get("SATQUERY_MAX_TOKENS_REFER",   "512"))

# Max retries on 429 / transient errors (keep low to avoid 2+ minute waits)
OPENROUTER_MAX_RETRIES = int(os.environ.get("OPENROUTER_MAX_RETRIES", "2"))

# ─────────────────────────────────────────────
# System Prompt
# ─────────────────────────────────────────────
RS_SYSTEM_PROMPT = (
    "You are SatQuery AI, an expert remote sensing and geospatial intelligence analyst. "
    "You analyse satellite and aerial imagery with precision. "
    "Your answers are structured, evidence-based, and concise. "
    "When provided an image, inspect it thoroughly before answering. "
    "Use remote sensing terminology where appropriate. "
    "Always include: (1) a direct answer, (2) key visual observations, "
    "(3) spatial context, and (4) a confidence note."
)

# ─────────────────────────────────────────────
# Server
# ─────────────────────────────────────────────
API_HOST          = os.environ.get("SATQUERY_HOST", "0.0.0.0")
API_PORT          = int(os.environ.get("SATQUERY_PORT", "8000"))
CORS_ORIGINS      = os.environ.get("SATQUERY_CORS_ORIGINS", "*").split(",")
MAX_IMAGE_SIZE_MB = int(os.environ.get("SATQUERY_MAX_IMAGE_MB", "10"))
