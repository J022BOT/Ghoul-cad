@echo off
setlocal EnableExtensions
chcp 65001 >nul
cd /d "%~dp0"
set "PYTHONHOME="
set "PYTHONPATH="

if not exist "%~dp0python\python.exe" (
  echo [ERROR] Missing: python\python.exe
  pause
  exit /b 1
)

echo Creating desktop shortcut ...
REM Locate the launcher with a wildcard so this bat stays ASCII-only
REM (GBK/UTF-8 codepage mangles Chinese literals inside bat files).
REM Match on 'gho' case-insensitively so either launcher name is found
"%~dp0python\python.exe" -c "import os,glob;base=os.path.abspath(r'%~dp0');cands=[p for p in glob.glob(os.path.join(base,'*.bat')) if 'gho' in os.path.basename(p).lower()];tgt=cands[0] if cands else os.path.join(base,'Start-Ghoul.bat');d=os.path.join(os.path.expanduser('~'),'Desktop');p=os.path.join(d,'J022 Ghoul.lnk');from win32com.client import Dispatch;s=Dispatch('WScript.Shell').CreateShortCut(p);s.TargetPath=tgt;s.WorkingDirectory=base;s.IconLocation=os.path.join(base,'app','j_cad_killer.ico');s.Description='J022 Ghoul portable';s.save();print('target :',tgt);print('created:',p)"
echo.
echo Done. Press any key to close.
pause >nul
