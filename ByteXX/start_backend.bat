@echo off
title ByteX AI Backend
cd /d "C:\Users\Suryansh\ByteXX"
echo.
echo ============================================================
echo   ByteX SatQuery AI  ^|  Backend Server
echo   VLM: Qwen2.5-VL-3B  ^|  CNN: YOLOv8 + ResNet/FCN
echo   AI Agent: Route -^> Plan -^> Execute -^> Verify -^> Merge
echo   Firebase: satellite-efa0a-default-rtdb.firebaseio.com
echo ============================================================
echo.

:: ── Activate virtual environment if present ──────────────────────────────────
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
    echo [OK] Virtual environment activated (.venv)
) else if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
    echo [OK] Virtual environment activated (venv)
) else (
    echo [--] No venv found, using system Python
)

:: ── HuggingFace model cache ───────────────────────────────────────────────────
set HF_HOME=C:\Users\Suryansh\OneDrive\Desktop\huggingface
set TRANSFORMERS_CACHE=%HF_HOME%\hub
set HUGGINGFACE_HUB_CACHE=%HF_HOME%\hub

:: ── Model settings ────────────────────────────────────────────────────────────
:: Set to "stub" to skip loading the VLM (faster startup, CNN + Agent still work)
set SATQUERY_MODEL_MODE=real
set SATQUERY_ADAPTER_PATH=C:\Users\Suryansh\OneDrive\Desktop\satquery-ai-vqa-lora

:: ── Server settings ───────────────────────────────────────────────────────────
set SATQUERY_HOST=0.0.0.0
set SATQUERY_PORT=8000
set SATQUERY_CORS_ORIGINS=*

echo [OK] Environment configured
echo [..] Starting server on http://localhost:8000
echo      API docs: http://localhost:8000/docs
echo.
echo  Press Ctrl+C to stop the server
echo ============================================================
echo.

python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

pause
