@echo off
setlocal EnableExtensions
chcp 65001 >nul
title J022 Ghoul
cd /d "%~dp0"

REM ---------- clear env vars that would break the bundled Python ----------
set "PYTHONHOME="
set "PYTHONPATH="
set "PYTHONSTARTUP="

REM ---------- everything points inside this folder (any drive / any PC) ----------
set "OLLAMA_MODELS=%~dp0ollama\models"
set "J022_CAD_DIR=%~dp0CAD"
set "J022_LIBRECAD=%~dp0LibreCAD\LibreCAD.exe"
set "J022_OLLAMA_EXE=%~dp0ollama\ollama.exe"

if not exist "%~dp0python\python.exe" (
  echo [ERROR] Missing: python\python.exe
  echo Please re-copy the whole folder.
  pause
  exit /b 1
)
if not exist "%~dp0app\launch.py" (
  echo [ERROR] Missing: app\launch.py
  pause
  exit /b 1
)

echo ============================================================
echo   J022 Ghoul  -  portable edition
echo   python : %~dp0python\python.exe
echo   models : %~dp0ollama\models
echo ============================================================
echo.
echo First start may take 1-3 minutes (model loading).
echo Do NOT close this window: it IS the web service.
echo.

"%~dp0python\python.exe" "%~dp0app\launch.py"

echo.
echo J022 Ghoul stopped. Press any key to close.
pause >nul
