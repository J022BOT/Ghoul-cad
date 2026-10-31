@echo off
setlocal EnableExtensions
chcp 65001 >nul
title J022 Ghoul PET
cd /d "%~dp0"

set "PYTHONHOME="
set "PYTHONPATH="
set "OLLAMA_MODELS=%~dp0ollama\models"
set "J022_CAD_DIR=%~dp0CAD"
set "J022_LIBRECAD=%~dp0LibreCAD\LibreCAD.exe"
set "J022_OLLAMA_EXE=%~dp0ollama\ollama.exe"

if not exist "%~dp0python\python.exe" (
  echo [ERROR] Missing: python\python.exe
  pause
  exit /b 1
)

echo Starting desktop pet only ...
"%~dp0python\python.exe" "%~dp0app\launch_pet.py"

echo.
echo Pet stopped. Press any key to close.
pause >nul
