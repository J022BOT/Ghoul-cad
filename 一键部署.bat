@echo off
setlocal EnableExtensions
chcp 65001 >nul
title ghoul-cad setup
cd /d "%~dp0"

echo ============================================================
echo   ghoul-cad  runtime setup
echo   downloads Python / Node / Ollama / LibreCAD / LibreDWG
echo ============================================================
echo.
echo This can take a long time (about 12 GB if models included).
echo Keep the window open. Ctrl+C to abort.
echo.

if not exist "%~dp0scripts\setup_runtime.ps1" (
  echo [ERROR] Missing: scripts\setup_runtime.ps1
  echo Please make sure you are running this from the repository root.
  pause
  exit /b 1
)

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\setup_runtime.ps1" %*

echo.
echo Setup finished. Press any key to close.
pause >nul
