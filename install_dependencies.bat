@echo off
REM Standalone dependency installer for STAGELIGHT.
REM Pip-installs Python packages. Does NOT download model files.
REM Safe to run repeatedly.

title STAGELIGHT - Install Dependencies
echo.
echo ============================================================
echo  STAGELIGHT - Python Dependency Installer
echo ============================================================
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not on your PATH.
    echo Install Python 3.10+ from https://www.python.org
    pause
    exit /b 1
)

echo Found Python:
python --version
echo.
echo Installing dependencies...
echo.

echo [1/5] PySide6 (installer GUI)
python -m pip install --upgrade PySide6 --quiet

echo [2/5] rembg (subject mask)
python -m pip install --upgrade rembg --quiet

echo [3/5] transformers + timm (depth/normal models)
python -m pip install --upgrade transformers timm --quiet

echo [4/5] Pillow + numpy + scipy (image processing)
python -m pip install --upgrade Pillow numpy scipy --quiet

echo [5/5] onnxruntime (rembg acceleration)
python -m pip install --upgrade onnxruntime --quiet

echo.
echo ============================================================
echo  Python packages done.
echo ============================================================
echo.
echo NOW DOWNLOAD THESE MODEL FILES (one time, ~3.7GB total):
echo.
echo 1. SD 1.5 photoreal checkpoint (~2GB)
echo    Save to: ComfyUI/models/checkpoints/
echo    Juggernaut Reborn (recommended):
echo    https://huggingface.co/HyperX-Sentience/Juggernaut-Reborn/
echo        resolve/main/juggernaut_reborn.safetensors
echo.
echo 2. IC-Light fc model (~1.7GB)
echo    Save to: ComfyUI/models/unet/
echo    https://huggingface.co/lllyasviel/ic-light/
echo        resolve/main/iclight_sd15_fc.safetensors
echo.
echo 3. kijai's ComfyUI-IC-Light extension
echo    Install via ComfyUI Manager - Custom Nodes Manager
echo    Search "IC-Light"
echo.
echo Next: run bootstrap.bat to launch the STAGELIGHT installer.
echo After installing: hard-refresh browser (Ctrl+Shift+R).
echo.
pause
