@echo off
setlocal EnableExtensions
chcp 65001 >nul
cd /d "%~dp0"

set "PYTHONHOME="
set "PYTHONPATH="

REM ---------- same package-local env as the launcher ----------
REM Without these, a stale user-level OLLAMA_MODELS on the host machine
REM would show up in this report and look like a broken package.
set "OLLAMA_MODELS=%~dp0ollama\models"
set "J022_CAD_DIR=%~dp0CAD"
set "J022_LIBRECAD=%~dp0LibreCAD\LibreCAD.exe"
set "J022_OLLAMA_EXE=%~dp0ollama\ollama.exe"

echo ================ environment self-check ================
echo package root : %~dp0
echo.
"%~dp0python\python.exe" -c "import sys;print('python  :', sys.version.split()[0], '|', sys.executable)"
"%~dp0python\python.exe" -c "import numpy,PIL,cv2,edge_tts,pyttsx3,comtypes;print('deps    : numpy',numpy.__version__,'| PIL',PIL.__version__,'| cv2',cv2.__version__)"
"%~dp0python\python.exe" -c "import sys;sys.path.insert(0,r'%~dp0app');import portable;print('paths   :');print('  ',portable.summary())"
"%~dp0python\python.exe" -c "import sys;sys.path.insert(0,r'%~dp0app');import runtime_guard as rg;print('vision  :', 'deps OK' if not rg.missing_deps() else 'MISSING '+str(rg.missing_deps()), '| bundled', rg.bundled_python())"
"%~dp0python\python.exe" -c "import sys;sys.path.insert(0,r'%~dp0app');import dwg_export;print('dwg     :',dwg_export.status()['note'])"
"%~dp0python\python.exe" -c "import sys;sys.path.insert(0,r'%~dp0app');import cad_agent;print('lc draft:',cad_agent.librecad_draft_state(),'(0=text shows, 1=text becomes boxes)')"
"%~dp0python\python.exe" -c "import sys;sys.path.insert(0,r'%~dp0app');import portable;print('librecad:',portable.librecad_build(),'(2.2 = Qt6 build, 2.1 = old Qt5)');print('  path  :',portable.LIBRECAD)"
"%~dp0python\python.exe" -c "import sys,os;sys.path.insert(0,r'%~dp0app');import homotope_bridge as hb;print('homotope:', hb.available(), '| node:', hb.find_node() or hb.reason_unavailable())"
"%~dp0python\python.exe" -c "import sys,os;sys.path.insert(0,r'%~dp0app');import portable;print('lc exe  :', 'OK  ' if os.path.isfile(portable.LIBRECAD) else 'MISS', portable.LIBRECAD)"
"%~dp0python\python.exe" -c "import sys;sys.path.insert(0,r'%~dp0app');import dxf_reader;print('dxf read:', 'builtin OK (import needs no ezdxf)')"
echo.
echo ---- key files ----
for %%F in ("python\python.exe" "app\launch.py" "ollama\ollama.exe" "ollama\models\manifests" "tools\libredwg\dxf2dwg.exe" "tools\libredwg\libredwg-0.dll" "tools\libredwg\libiconv-2.dll" "tools\node\node.exe" "tools\homotope\homotope_cli.mjs" "tools\homotope\node_modules\@homotope\engine") do (
  if exist "%~dp0%%~F" (echo   OK    %%~F) else (echo   MISS  %%~F)
)
echo.
echo ---- ollama models (from manifests) ----
dir /b "%~dp0ollama\models\manifests\registry.ollama.ai\library" 2>nul
echo.
echo Done. Press any key to close.
pause >nul
