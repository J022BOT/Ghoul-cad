"""launch.py — J022 Ghoul 一键启动逻辑（Unicode 安全）。

由 launch_cad.bat 调用。负责：
  1) 确保 Ollama 本地推理服务在跑（未起则后台拉起）
  2) 启动桌宠 J022 Ghoul（独立进程，AI 运行时常驻桌面；--no-pet 关闭）
  3) 启动 Web 界面（本进程内前台运行 web_app.main，即“服务窗口”）
  4) 端口就绪后自动打开浏览器 http://127.0.0.1:8765

设计要点：
  - 全程用 Python，彻底规避 .bat 的 GBK/UTF-8 编码问题。
  - 若 Web 已在运行，则直接打开浏览器后返回（不重复起服务）。
  - 关闭本窗口即停止 Web 服务（服务在本进程内，无孤儿进程）；
    桌宠是独立进程，会继续运行（右键菜单 → 退出 可关闭）。
"""
from __future__ import annotations
import os
import sys
import time
import ctypes
import threading
import subprocess
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

OLLAMA = "http://localhost:11434"
WEB = "http://127.0.0.1:8765"

# ollama 常见安装位置（安装器不一定写 PATH，逐个兜底）
OLLAMA_CANDIDATES = [
    r"C:\Users\ASUS\AppData\Local\Programs\Ollama\ollama.exe",
    r"C:\Program Files\Ollama\ollama.exe",
]


def _find_ollama():
    """返回可用的 ollama 可执行文件路径；找不到返回 None。

    便携版优先用包内 ollama\\ollama.exe（口径见 portable.py）。
    """
    try:
        from portable import find_ollama
        exe = find_ollama()
    except Exception:
        exe = ""
    if exe:
        return exe
    import shutil
    found = shutil.which("ollama")
    if found:
        return found
    for c in OLLAMA_CANDIDATES:
        if os.path.exists(c):
            return c
    return None


def _set_title():
    try:
        ctypes.windll.kernel32.SetConsoleTitleW("J022 Ghoul · Web 服务")
    except Exception:
        pass


def http_ok(url: str, timeout: float = 3.0) -> bool:
    try:
        urllib.request.urlopen(url, timeout=timeout)
        return True
    except Exception:
        return False


def _ollama_models():
    """返回 Ollama 已加载的模型名列表；服务不通返回 None。"""
    try:
        with urllib.request.urlopen(OLLAMA + "/api/tags", timeout=3.0) as r:
            import json
            data = json.loads(r.read().decode("utf-8", "ignore"))
            return [m.get("name", "") for m in data.get("models", [])]
    except Exception:
        return None


def _kill_ollama():
    """杀掉 ollama 全部进程（含看门狗 'ollama app.exe'——它会在 serve 挂掉时
    自动重启一个新 serve，但重启的进程不带 OLLAMA_MODELS，导致模型列表为空）。"""
    for name in ("ollama.exe", "ollama app.exe"):
        try:
            subprocess.run(["taskkill", "/F", "/IM", name],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass
    for _ in range(10):
        if _ollama_models() is None:
            return True
        time.sleep(1)
    return False


def _start_ollama():
    """带 OLLAMA_MODELS 启动 ollama serve，等待就绪。"""
    exe = _find_ollama()
    if not exe:
        print("    （未找到 ollama 可执行文件）")
        print("    请确认已安装 Ollama，或手动运行: ollama serve")
        time.sleep(3)
        return
    env = os.environ.copy()
    # 模型库目录：便携版为包内 ollama\models，开发态为 D:\Local AI\ollama\models
    try:
        from portable import find_ollama_models
        models_dir = find_ollama_models()
    except Exception:
        models_dir = ""
    if models_dir and not env.get("OLLAMA_MODELS"):
        env["OLLAMA_MODELS"] = models_dir
    si = subprocess.STARTUPINFO()
    si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    si.wShowWindow = 6  # SW_MINIMIZE，尽量不打扰
    try:
        subprocess.Popen([exe, "serve"], startupinfo=si, env=env)
        print(f"    已启动: {exe}（模型目录 {models_dir}）")
    except Exception as e:
        print("    （自动启动 ollama serve 失败：%s）" % e)
        print("    请手动运行: ollama serve")
    for i in range(20):
        time.sleep(1)
        if _ollama_models() is not None:
            print("    Ollama 服务已就绪")
            return
    print("    Ollama 服务未就绪，Web 界面仍会启动（AI 功能可能暂不可用）")


def ensure_ollama():
    models = _ollama_models()
    if models is None:                       # 服务没起 → 直接拉
        print("[1/3] 正在后台启动 Ollama 本地推理服务...")
        _start_ollama()
        return
    if models:                               # 服务正常
        print(f"[1/3] Ollama 服务已在运行（已加载 {len(models)} 个模型：{', '.join(models)}）")
        return
    # 服务在跑但模型列表为空（看门狗重启的 serve 丢了 OLLAMA_MODELS）→ 自愈
    print("[1/3] Ollama 在运行但模型列表为空，正在自动修复（重启并挂载模型目录）...")
    if _kill_ollama():
        _start_ollama()
    else:
        print("    未能停止旧进程，请手动重启电脑或运行: ollama serve")


def open_browser_when_ready():
    for _ in range(30):
        time.sleep(1)
        if http_ok(WEB + "/", 2.0):
            try:
                import webbrowser
                webbrowser.open(WEB)
            except Exception:
                pass
            return


def start_pet():
    """后台启动桌宠（独立进程：主窗口关闭后桌宠继续运行）。"""
    pet_py = os.path.join(HERE, "desktop_pet.py")
    if not os.path.exists(pet_py):
        return
    try:
        flags = 0
        if os.name == "nt":
            flags = subprocess.CREATE_NEW_PROCESS_GROUP
        subprocess.Popen([sys.executable, pet_py], cwd=HERE,
                         creationflags=flags)
        print("[+] 桌宠 J022 Ghoul 已启动（左键单击对话 / 拖拽移动 / 右键菜单）")
    except Exception as e:  # noqa: BLE001
        print(f"    （桌宠启动失败：{e}）")


def main():
    # 自检：用错解释器（比如系统自带的清爽 Python）启动时，缺 numpy/cv2
    # 只在点「生成/导入」时才暴露。这里直接换成包内自带的 Python 重启自己。
    try:
        import runtime_guard
        runtime_guard.ensure()
    except SystemExit:
        raise
    except Exception:                                  # noqa: BLE001
        pass
    _set_title()
    print("=" * 60)
    print("  J022 Ghoul   ·   纯本地运行 · 断网可用")
    print("=" * 60)
    try:
        from portable import summary
        print("  路径自检：" + summary())
    except Exception:
        pass

    ensure_ollama()

    # 桌宠：AI 模型运行时桌宠常驻桌面（--no-pet 可关闭）
    if "--no-pet" not in sys.argv:
        start_pet()

    if http_ok(WEB + "/"):
        print("[2/3] Web 界面已在运行")
        print("[3/3] 正在打开浏览器 -> %s" % WEB)
        try:
            import webbrowser
            webbrowser.open(WEB)
        except Exception:
            pass
        print("\n已在浏览器打开。可关闭本窗口（服务继续在后台运行）。")
        return

    # 端口就绪后自动开浏览器（后台线程）
    threading.Thread(target=open_browser_when_ready, daemon=True).start()

    print("[2/3] 正在启动 Web 界面（本窗口即服务窗口，关闭即停止服务）...")
    print("[3/3] 浏览器将自动打开 -> %s" % WEB)
    print("-" * 60)
    print("  在浏览器里输入自然语言或上传草图，即可生成 DXF 并用 LibreCAD 打开。")
    print("-" * 60)

    import web_app
    web_app.main()  # 前台阻塞运行，直至 Ctrl+C 或关闭窗口


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n已停止。")
    except Exception:
        import traceback
        traceback.print_exc()
        try:
            input("\n发生错误，按回车键关闭本窗口...")
        except Exception:
            pass
