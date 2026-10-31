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
REM 用通配符定位启动脚本，避免 bat 里出现中文字面量（GBK/UTF-8 代码页会拼错文件名）
"%~dp0python\python.exe" -c "import os,glob;base=os.path.abspath(r'%~dp0');cands=[p for p in glob.glob(os.path.join(base,'*.bat')) if 'Ghoo' in os.path.basename(p)];tgt=cands[0] if cands else os.path.join(base,'Start-Ghoo.bat');d=os.path.join(os.path.expanduser('~'),'Desktop');p=os.path.join(d,'J022 Ghoul.lnk');from win32com.client import Dispatch;s=Dispatch('WScript.Shell').CreateShortCut(p);s.TargetPath=tgt;s.WorkingDirectory=base;s.IconLocation=os.path.join(base,'app','j_cad_killer.ico');s.Description='J022 Ghoul portable';s.save();print('target :',tgt);print('created:',p)"
echo.
echo Done. Press any key to close.
pause >nul
