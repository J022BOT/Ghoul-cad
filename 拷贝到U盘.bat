@echo off
setlocal EnableExtensions
chcp 65001 >nul
cd /d "%~dp0"
title Copy J022 Ghoul to USB drive

echo ============================================================
echo   Copy this package to a USB drive
echo   (target drive MUST be NTFS or exFAT, NOT FAT32)
echo ============================================================
echo.
echo Existing drives:
wmic logicaldisk get caption,volumename,filesystem 2>nul
echo.
set /p DRIVE=Enter target drive letter (e.g. E):

if "%DRIVE%"=="" goto :eof
set "TARGET=%DRIVE%:\J022-Ghoul-Portable"

if /i "%DRIVE%"=="%~d0" (
  echo [ERROR] Target is the same drive as this package.
  pause
  exit /b 1
)

echo.
echo Copying to %TARGET% ... (about 11.8 GB, please wait)
if not exist "%DRIVE%:\" (
  echo [ERROR] Drive %DRIVE%: not found.
  pause
  exit /b 1
)
robocopy "%~dp0." "%TARGET%" /E /MT:8 /R:2 /W:2 /NP
echo.
echo Done. Check the summary above: "Failed" should be 0.
pause
