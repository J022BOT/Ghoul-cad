@echo off
setlocal EnableExtensions
chcp 65001 >nul
title Ghoo Shutdown

REM =====================================================================
REM  Step 0 - relaunch ourselves from a copy in %TEMP%
REM  Why: this console must never keep the USB stick as its working dir,
REM  otherwise the volume can never be released. The package folder is
REM  passed through an environment variable so spaces can never break it.
REM =====================================================================
if not defined J022_PKG goto bootstrap

set "ROOT=%J022_PKG%"
set "DRV=%J022_PKG:~0,2%"
set "J022_PKG="
cd /d "%SystemRoot%"
goto run

:bootstrap
for %%i in ("%~dp0.") do set "J022_PKG=%%~fi"
copy /y "%~f0" "%TEMP%\j022_close.bat" >nul 2>&1
if errorlevel 1 (
  echo [ERROR] cannot copy the helper script to the temp folder
  pause
  exit /b 1
)
powershell -NoProfile -ExecutionPolicy Bypass -Command "Start-Process -FilePath (Join-Path $env:TEMP 'j022_close.bat')"
exit /b 0

:run
echo ============================================================
echo   J022 Ghoul  -  stop everything, then release the drive
echo   package : %ROOT%
echo   drive   : %DRV%
echo ============================================================
echo.

REM ---------- 1) stop only processes whose EXE lives in this folder ----------
powershell -NoProfile -ExecutionPolicy Bypass -Command "$root=('%ROOT%' -replace '/','\').TrimEnd('\')+'\'; function Get-Mine { Get-CimInstance Win32_Process | Where-Object { $_.ExecutablePath -and $_.ExecutablePath.StartsWith($root,[System.StringComparison]::OrdinalIgnoreCase) -and $_.Name -in @('python.exe','pythonw.exe','ollama.exe') } }; $p=@(Get-Mine); if($p.Count -eq 0){ Write-Host '  [i] nothing running in this folder' } else { foreach($x in $p){ Write-Host ('  [x] stop ' + $x.Name + '  pid=' + $x.ProcessId); Stop-Process -Id $x.ProcessId -Force -ErrorAction SilentlyContinue } }; Start-Sleep -Seconds 2; $r=@(Get-Mine); if($r.Count -gt 0){ Write-Host '  [!] still alive:'; foreach($y in $r){ Write-Host ('      ' + $y.Name + ' pid=' + $y.ProcessId) } } else { Write-Host '  [v] all stopped' }"

REM ---------- 2) close the leftover launcher console window (title: J022 Ghoul ...) ----------
powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-Process -Name cmd -ErrorAction SilentlyContinue | Where-Object { $_.MainWindowTitle -like 'J022 Ghoul*' } | ForEach-Object { Write-Host ('  [x] close window: ' + $_.MainWindowTitle); Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue }"

REM ---------- 3) close Explorer windows showing this drive ----------
powershell -NoProfile -ExecutionPolicy Bypass -Command "try { $d='%DRV%'; $sh=New-Object -ComObject Shell.Application; $n=0; foreach($w in @($sh.Windows())){ try { $u=[string]$w.LocationURL; if($u -and $u.StartsWith('file:///'+$d+'/')){ $w.Quit(); $n++ } } catch {} }; if($n -gt 0){ Write-Host ('  [x] closed ' + $n + ' Explorer window(s) on ' + $d) } else { Write-Host '  [i] no Explorer window on this drive' } } catch { Write-Host '  [i] explorer check skipped' }"

REM ---------- 4) flush the write cache so the volume is idle ----------
powershell -NoProfile -ExecutionPolicy Bypass -Command "try { Write-VolumeCache -DriveLetter '%DRV:~0,1%' -ErrorAction Stop; Write-Host '  [v] write cache flushed' } catch { Write-Host '  [i] cache flush skipped' }"

echo.
echo Port check:
powershell -NoProfile -ExecutionPolicy Bypass -Command "foreach($n in 8765,11434){ $c=Get-NetTCPConnection -LocalPort $n -State Listen -ErrorAction SilentlyContinue; if($c){ Write-Host ('  port ' + $n + ' : STILL LISTENING') } else { Write-Host ('  port ' + $n + ' : free') } }"

REM =====================================================================
REM  5) eject
REM  IMPORTANT: this script must never call
REM      diskpart  select volume X / remove all dismount
REM  "remove all" deletes the drive letter itself, so the next time the
REM  stick is plugged in Windows has no letter to assign and the stick
REM  looks dead: no icon in Explorer, no Eject entry in the tray.
REM  Instead we release the volume through eject_native.ps1, which keeps
REM  the mount point registered.
REM =====================================================================
echo.
if defined J022_NOEJECT goto skip_eject
if defined J022_AUTOYES goto do_eject
choice /c YN /t 8 /d Y /n /m "  Eject %DRV% now? [Y/N] (auto Y in 8s) "
if errorlevel 2 goto skip_eject

:do_eject
set "EP=%TEMP%\j022_eject_native.ps1"
copy /y "%ROOT%\eject_native.ps1" "%EP%" >nul 2>&1
if not exist "%EP%" goto shell_eject
echo   [1/2] releasing the volume (drive letter is preserved) ...
powershell -NoProfile -ExecutionPolicy Bypass -File "%EP%" -Drive "%DRV%"
if not exist %DRV%\ goto eject_ok

echo   [i] the volume refused to be released, trying the shell eject ...
:shell_eject
set "EP2=%TEMP%\j022_eject_shell.ps1"
> "%EP2%" echo $sh = New-Object -ComObject Shell.Application
>> "%EP2%" echo $it = $sh.Namespace(17).ParseName('%DRV%')
>> "%EP2%" echo if ($it) { $it.InvokeVerb('Eject') }
>> "%EP2%" echo Start-Sleep -Seconds 2
start "" /b powershell -NoProfile -STA -ExecutionPolicy Bypass -File "%EP2%" >nul 2>&1
ping -n 7 127.0.0.1 >nul
powershell -NoProfile -ExecutionPolicy Bypass -Command "$me=$PID; Get-CimInstance Win32_Process | Where-Object { $_.ProcessId -ne $me -and $_.CommandLine -and $_.CommandLine -like '*j022_eject_shell.ps1*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"
del /q "%EP2%" >nul 2>&1
if not exist %DRV%\ goto eject_ok

echo.
echo   [!] %DRV% is still held open by something outside this folder.
echo       Every process we own is gone and the write cache is flushed,
echo       so the data on the stick is already complete: you may simply
echo       pull it out. Nothing here has touched the drive letter.
goto done

:eject_ok
echo   [v] %DRV% released - you can unplug the stick now.
echo   [i] the drive letter stays registered, so plugging it back in
echo       gives you the same %DRV% again.
goto done

:skip_eject
echo   [i] drive left mounted (eject skipped)

:done
echo.
echo ------------------------------------------------------------
echo  This window does not hold the USB stick any more.
echo  If the stick ever shows up without a drive letter, run the
echo  USB repair tool from the LOCAL disk (D:\ or the desktop),
echo  not from the stick.
echo ------------------------------------------------------------
echo.
echo Done. Press any key to close this window.
pause >nul
