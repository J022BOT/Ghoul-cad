# -*- coding: utf-8 -*-
"""portable.py — 可移植路径解析（桌宠 / 网页 / CAD 共用）。

设计原则
  1. 一切资源**优先在「包根目录」内**查找，找不到才回退到历史硬编码路径；
  2. 包根目录 = 本文件所在目录；若本文件位于 ``<包根>\\app\\`` 下，
     则包根取上一级（便携版布局：``J022-Ghoul-Portable\\app\\portable.py``）；
  3. 需要写入的目录（CAD 输出）按需自动创建，绝不让程序因缺目录而崩。

于是同一份代码既能跑在开发态 ``D:\\Local AI\\cad-agent``，
也能跑在便携态 ``<任意盘>:\\J022-Ghoul-Portable\\app``，
不依赖任何固定盘符。
"""
from __future__ import annotations

import os
import shutil

APP_DIR = os.path.dirname(os.path.abspath(__file__))

if os.path.basename(APP_DIR).lower() == "app":
    PKG_ROOT = os.path.dirname(APP_DIR)          # 便携版：<包根>\app\portable.py
else:
    PKG_ROOT = APP_DIR                           # 开发态：代码目录即包根


def _env(name):
    v = os.environ.get(name, "").strip()
    return v or None


def _first_dir(*cands):
    for c in cands:
        if c and os.path.isdir(c):
            return c
    return None


def _first_file(*cands):
    for c in cands:
        if c and os.path.isfile(c):
            return c
    return None


def ensure_dir(path):
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        pass
    return path


# ---------------------------------------------------------------- CAD 输出目录
def _resolve_cad_dir():
    env = _env("J022_CAD_DIR")
    if env:
        return ensure_dir(env)
    # 便携版：包内 CAD 目录（发布时随包提供，保证可写且不污染系统盘）
    pkg_cad = os.path.join(PKG_ROOT, "CAD")
    if os.path.isdir(pkg_cad):
        return pkg_cad
    # 开发态：沿用原有的 D:\CAD
    legacy = "D:/CAD"
    if os.path.isdir(legacy):
        return legacy
    return ensure_dir(pkg_cad)


CAD_DIR = _resolve_cad_dir()


# ---------------------------------------------------------------- LibreCAD
def librecad_build(path=""):
    """判断这只 LibreCAD 是 2.2.x 还是 2.1.x（给诊断用）。

    2.2.x 是 Qt6 构建，主程序约 32MB；2.1.x 是 Qt5 构建，只有 7.7MB。
    注意：Windows 文件名不区分大小写，``LibreCAD.exe`` 与 ``librecad.exe``
    是**同一个文件**，所以没法靠"哪个名字存在"来区分版本，只能看体积/依赖。
    """
    exe = path or LIBRECAD
    if not exe or not os.path.isfile(exe):
        return "missing"
    folder = os.path.dirname(exe)
    if os.path.isfile(os.path.join(folder, "Qt6Core.dll")):
        return "2.2"
    if os.path.isfile(os.path.join(folder, "Qt5Core.dll")):
        return "2.1"
    return "2.2" if os.path.getsize(exe) > 10 * 1024 * 1024 else "2.1"


def find_librecad():
    """返回 LibreCAD.exe 路径；找不到返回空串（调用处应跳过自动打开）。"""
    env = _env("J022_LIBRECAD")
    if env and os.path.isfile(env):
        return env
    pf = os.environ.get("ProgramFiles", r"C:\Program Files")
    pf86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
    found = _first_file(
        os.path.join(PKG_ROOT, "LibreCAD", "librecad.exe"),   # 便携版自带（2.2.x 小写名）
        os.path.join(PKG_ROOT, "LibreCAD", "LibreCAD.exe"),   # 同目录的不同大小写写法
        "D:/CAD/LibreCAD/LibreCAD.exe",                        # 历史位置
        "D:/Local AI/LibreCAD/LibreCAD.exe",                   # 现有 D 盘安装
        os.path.join(pf, "LibreCAD", "LibreCAD.exe"),
        os.path.join(pf86, "LibreCAD", "LibreCAD.exe"),
    )
    if found:
        return found
    return shutil.which("LibreCAD.exe") or shutil.which("LibreCAD") or ""


LIBRECAD = find_librecad()


# ---------------------------------------------------------------- Ollama
def find_ollama():
    """返回 ollama 可执行文件路径；找不到返回空串。"""
    env = _env("J022_OLLAMA_EXE")
    if env and os.path.isfile(env):
        return env
    found = _first_file(
        os.path.join(PKG_ROOT, "ollama", "ollama.exe"),        # 便携版自带
        r"C:\Users\ASUS\AppData\Local\Programs\Ollama\ollama.exe",
        r"C:\Program Files\Ollama\ollama.exe",
    )
    if found:
        return found
    return shutil.which("ollama") or ""


def find_ollama_models():
    """返回 ollama 模型库目录；找不到返回空串。"""
    env = _env("OLLAMA_MODELS")
    if env and os.path.isdir(env):
        return env
    return _first_dir(
        os.path.join(PKG_ROOT, "ollama", "models"),            # 便携版自带
        "D:/Local AI/ollama/models",                           # 现有 D 盘模型库
    ) or ""


OLLAMA_EXE = find_ollama()
OLLAMA_MODELS = find_ollama_models()


def summary():
    """启动时打印一行自检信息，便于排障。"""
    return ("包根=%s | CAD=%s | LibreCAD=%s | Ollama=%s | 模型库=%s"
            % (PKG_ROOT, CAD_DIR, LIBRECAD or "(未找到)",
               OLLAMA_EXE or "(未找到)", OLLAMA_MODELS or "(未找到)"))


if __name__ == "__main__":
    print(summary())
