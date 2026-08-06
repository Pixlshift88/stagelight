@echo off
title STAGELIGHT v0.9.5
python --version >nul 2>&1
if errorlevel 1 (
    echo Python not found. Install from https://www.python.org
    pause & exit /b 1
)
python -c "import PySide6" >nul 2>&1
if errorlevel 1 (
    echo Installing PySide6...
    pip install PySide6 --quiet
)
python installer.py
