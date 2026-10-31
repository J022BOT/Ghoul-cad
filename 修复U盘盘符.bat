@echo off
setlocal EnableExtensions
chcp 65001 >nul
title Repair USB drive letter

REM =====================================================================
REM  Repair the drive letter of a USB stick.
REM
REM  Use this when the stick is plugged in but nothing happens: no icon in
REM  Explorer and no Eject entry in the tray. That means the volume lost its
REM  mount point, which is what
REM      diskpart  remove all dismount
REM  does - "remove all" deletes the drive letter itself.
REM
REM  This file is self-contained (the PS payload is embedded as base64), so
REM  it works even when the stick cannot be read at all. Keep a copy on the
REM  local disk or the desktop: while the stick has no letter you cannot
REM  start anything from the stick.
REM
REM  It only ADDS a letter. It never formats, never deletes, never writes
REM  data. Safe to run at any time.
REM =====================================================================

fltmc >nul 2>&1
if errorlevel 1 (
  echo Requesting administrator rights - please click Yes on the UAC prompt ...
  powershell -NoProfile -ExecutionPolicy Bypass -Command "Start-Process -FilePath '%~f0' -Verb RunAs"
  exit /b 0
)

echo ============================================================
echo   Repair the drive letter of a USB stick
echo ============================================================
echo.
set "PS=%TEMP%\j022_repair_letter.ps1"
powershell -NoProfile -ExecutionPolicy Bypass -Command "$b='JEVycm9yQWN0aW9uUHJlZmVyZW5jZSA9ICdTdG9wJwokUHJlZmVyID0gJ0UnCmZ1bmN0aW9uIFNheSgkbSkgeyBXcml0ZS1Ib3N0ICgiICAiICsgJG0pIH0KClNheSAiU2Nhbm5pbmcgcmVtb3ZhYmxlIGRpc2tzIC4uLiIKCiRjYW5kcyA9IEAoKQpmb3JlYWNoICgkZCBpbiAoR2V0LURpc2sgfCBXaGVyZS1PYmplY3QgeyAkXy5CdXNUeXBlIC1pbiBAKCdVU0InLCdTRCcpIH0pKSB7CiAgICBpZiAoJGQuSXNPZmZsaW5lKSB7IHRyeSB7IFNldC1EaXNrIC1OdW1iZXIgJGQuTnVtYmVyIC1Jc09mZmxpbmUgJGZhbHNlIH0gY2F0Y2ggeyB9IH0KICAgIGlmICgkZC5Jc1JlYWRPbmx5KSB7IHRyeSB7IFNldC1EaXNrIC1OdW1iZXIgJGQuTnVtYmVyIC1Jc1JlYWRPbmx5ICRmYWxzZSB9IGNhdGNoIHsgfSB9CiAgICBmb3JlYWNoICgkcCBpbiAoR2V0LVBhcnRpdGlvbiAtRGlza051bWJlciAkZC5OdW1iZXIgLUVycm9yQWN0aW9uIFNpbGVudGx5Q29udGludWUpKSB7CiAgICAgICAgaWYgKCRwLkRyaXZlTGV0dGVyKSB7IGNvbnRpbnVlIH0KICAgICAgICBpZiAoJHAuVHlwZSAtZXEgJ1Jlc2VydmVkJykgeyBjb250aW51ZSB9CiAgICAgICAgaWYgKCRwLlNpemUgLWx0IDEwME1CKSB7IGNvbnRpbnVlIH0KICAgICAgICAkY2FuZHMgKz0gW3BzY3VzdG9tb2JqZWN0XUB7IERpc2sgPSAkZC5OdW1iZXI7IFBhcnQgPSAkcC5QYXJ0aXRpb25OdW1iZXI7IFNpemUgPSAkcC5TaXplOyBNb2RlbCA9ICRkLkZyaWVuZGx5TmFtZSB9CiAgICB9Cn0KCmlmICgtbm90ICRjYW5kcykgewogICAgU2F5ICJOb3RoaW5nIHRvIHJlcGFpcjogZXZlcnkgcmVtb3ZhYmxlIHBhcnRpdGlvbiBhbHJlYWR5IGhhcyBhIGRyaXZlIGxldHRlci4iCiAgICBTYXkgIklmIHRoZSBzdGljayBpcyBzdGlsbCBpbnZpc2libGUsIHVucGx1ZyBpdCBhbmQgdHJ5IGFub3RoZXIgVVNCIHBvcnQuIgogICAgZXhpdCAwCn0KCiR1c2VkID0gQChHZXQtVm9sdW1lIHwgV2hlcmUtT2JqZWN0IHsgJF8uRHJpdmVMZXR0ZXIgfSB8IEZvckVhY2gtT2JqZWN0IHsgJF8uRHJpdmVMZXR0ZXIgfSkKCmZ1bmN0aW9uIEdldC1GcmVlTGV0dGVyIHsKICAgIHBhcmFtKFtzdHJpbmddJFdhbnQpCiAgICBpZiAoJFdhbnQgLWFuZCAoJHVzZWQgLW5vdGNvbnRhaW5zICRXYW50KSkgeyByZXR1cm4gJFdhbnQgfQogICAgZm9yZWFjaCAoJGMgaW4gQCgnRScsJ0YnLCdHJywnSCcsJ0knLCdKJywnSycsJ0wnLCdNJywnTicsJ08nLCdQJywnUScsJ1InLCdTJywnVCcsJ1UnLCdWJywnVycsJ1gnLCdZJywnWicpKSB7CiAgICAgICAgaWYgKCR1c2VkIC1ub3Rjb250YWlucyAkYykgeyByZXR1cm4gJGMgfQogICAgfQogICAgcmV0dXJuICRudWxsCn0KCiRmYWlsID0gMApmb3JlYWNoICgkYyBpbiAkY2FuZHMpIHsKICAgICRsZXR0ZXIgPSBHZXQtRnJlZUxldHRlciAkUHJlZmVyCiAgICBpZiAoLW5vdCAkbGV0dGVyKSB7IFNheSAiTm8gZnJlZSBkcml2ZSBsZXR0ZXIgbGVmdC4iOyAkZmFpbCA9IDE7IGJyZWFrIH0KICAgIHRyeSB7CiAgICAgICAgU2V0LVBhcnRpdGlvbiAtRGlza051bWJlciAkYy5EaXNrIC1QYXJ0aXRpb25OdW1iZXIgJGMuUGFydCAtTmV3RHJpdmVMZXR0ZXIgJGxldHRlciAtRXJyb3JBY3Rpb24gU3RvcAogICAgICAgICR1c2VkICs9ICRsZXR0ZXIKICAgICAgICBTYXkgKCJbdl0gZGlzayAiICsgJGMuRGlzayArICIgcGFydGl0aW9uICIgKyAkYy5QYXJ0ICsgIiAoIiArICRjLk1vZGVsICsgIiwgIiArIFttYXRoXTo6Um91bmQoJGMuU2l6ZS8xR0IsMSkgKyAiIEdCKSAtPiAiICsgJGxldHRlciArICI6IikKICAgIH0gY2F0Y2ggewogICAgICAgIFNheSAoIlshXSBmYWlsZWQ6ICIgKyAkXy5FeGNlcHRpb24uTWVzc2FnZSkKICAgICAgICAkZmFpbCA9IDEKICAgIH0KfQoKU3RhcnQtU2xlZXAgLVNlY29uZHMgMgpXcml0ZS1Ib3N0ICIiClNheSAiQ3VycmVudCB2b2x1bWVzOiIKR2V0LVZvbHVtZSB8IFdoZXJlLU9iamVjdCB7ICRfLkRyaXZlVHlwZSAtZXEgJ1JlbW92YWJsZScgfSB8IEZvckVhY2gtT2JqZWN0IHsKICAgICRsID0gaWYgKCRfLkRyaXZlTGV0dGVyKSB7ICRfLkRyaXZlTGV0dGVyICsgJzonIH0gZWxzZSB7ICcobm8gbGV0dGVyKScgfQogICAgV3JpdGUtSG9zdCAoIiAgICAgICIgKyAkbCArICIgICIgKyAkXy5GaWxlU3lzdGVtTGFiZWwgKyAiICAiICsgJF8uRmlsZVN5c3RlbSArICIgICIgKyBbbWF0aF06OlJvdW5kKCRfLlNpemUvMUdCLDEpICsgIiBHQiAgIiArICRfLkhlYWx0aFN0YXR1cykKfQoKaWYgKCRmYWlsIC1lcSAwKSB7IGV4aXQgMCB9IGVsc2UgeyBleGl0IDEgfQo='; [IO.File]::WriteAllText('%PS%', [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($b)), (New-Object Text.UTF8Encoding($false)))"
if not exist "%PS%" (
  echo [ERROR] cannot unpack the helper script
  pause
  exit /b 1
)
powershell -NoProfile -ExecutionPolicy Bypass -File "%PS%"
set "RC=%ERRORLEVEL%"
del /q "%PS%" >nul 2>&1
echo.
if "%RC%"=="0" (
  echo Done. Open "This PC" - the stick should be back with its letter.
) else (
  echo Something went wrong. Open Disk Management ^(diskmgmt.msc^) and
  echo assign a letter to the removable volume by hand.
)
echo.
pause
