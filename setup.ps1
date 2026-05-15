# CameraLM - one-shot setup for Windows + Python 3.11 + CUDA 12.1
#
# Run from the repo root in PowerShell:
#   ./setup.ps1
#
# Re-running is safe - it reuses the existing .venv.
#
# After this finishes, install Ollama from https://ollama.com/download and run:
#   ollama pull qwen3-vl:2b

$ErrorActionPreference = "Stop"

$pyVersion = & py -3.11 --version 2>$null
if (-not $pyVersion) {
    Write-Error "Python 3.11 not found. Install it from python.org or the Microsoft Store, then re-run."
    exit 1
}
Write-Host "Using $pyVersion"

if (-not (Test-Path ".\.venv")) {
    Write-Host "Creating virtual environment in .venv ..."
    py -3.11 -m venv .venv
}

. .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip wheel setuptools

Write-Host "Installing CUDA 12.1 PyTorch wheels ..."
pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.5.1+cu121 torchvision==0.20.1+cu121

Write-Host "Installing project dependencies ..."
pip install -r requirements.txt

Write-Host ""
Write-Host "Verifying installation ..."
python -c "import torch; print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available(), '| device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
python -c "import cv2; print('opencv:', cv2.__version__)"
python -c "import ultralytics; print('ultralytics:', ultralytics.__version__)"
python -c "import insightface; print('insightface:', insightface.__version__)"
python -c "import boxmot; print('boxmot:', boxmot.__version__)"
python -c "import faiss; print('faiss-cpu: ok')"
python -c "import flask, waitress; print('flask + waitress: ok')"

Write-Host ""
Write-Host "Python side ready. Next:"
Write-Host "  1. Install Ollama from https://ollama.com/download"
Write-Host "  2. ollama pull qwen3-vl:2b"
Write-Host "  3. .\.venv\Scripts\Activate.ps1"
Write-Host "  4. python -m cameralm.main"
