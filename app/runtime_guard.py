# -*- coding: utf-8 -*-
"""启动自检 / 自动纠正解释器。

为什么需要它（真实踩坑）：
    web_app 的重依赖全是**懒加载**的（numpy / cv2 / ezdxf 都用时才 import），
    所以随便拿一个"干净"的 Python（比如系统自带的 3.13）启动网页服务，
    界面照样能打开、编辑器照样能用 —— 只有真的点「生成」或「导入 DXF」
    时才会炸。用户看到的就是一句莫名其妙的「未安装 ezdxf」/「图片识别不可用」。

做法：
    启动时先探一次硬依赖（numpy / cv2）。缺了，就去找一个**真的装了这些库**
    的解释器把当前进程重启一遍（设 ``J022_RELAUNCHED=1`` 防循环）。全程只提示
    两行，用户无需操作。

找解释器的顺序（第一个能 import numpy+cv2 的胜出）：
    1. 环境变量 ``J022_PYTHON``（用户显式指定，最优先）；
    2. 包内自带 ``<包根>\\python\\python.exe``（便携版布局）；
    3. 各盘根目录下的 ``J022-Ghoul-Portable\\python\\python.exe``
       —— 覆盖"代码被放在别处（开发态目录）但机器上有便携包"的情况；
    4. PATH 里的 python.exe（全部，不只看第一个）；
    5. ``%LOCALAPPDATA%\\Programs\\Python\\Python3xx\\python.exe`` 等常见安装位置。

    实际踩坑：在 ``D:\\Local AI\\cad-agent``（开发态，代码目录即包根）里启动，
    而便携包在 ``D:\\J022-Ghoul-Portable``，旧版本只认"自己旁边"，于是直接
    放弃并提示"包内没有自带 python"——用户看到的就是那句。
"""

from __future__ import annotations

import os
import subprocess
import sys

#: 图片识别必需的第三方库（缺了网页能开，但一点生成就失败）
HARD_DEPS = ("numpy", "cv2")
#: 可选依赖：缺了也不影响，DXF 导入有内置的纯 Python 读取器兜底
SOFT_DEPS = ("ezdxf",)

_ENV_FLAG = "J022_RELAUNCHED"
_ENV_PYTHON = "J022_PYTHON"
#: 便携包目录名（各盘根目录下找这些名字）
_PORTABLE_NAMES = ("J022-Ghoul-Portable",)

#: 探测结果缓存：同一解释器只跑一次
_PROBE = {}
#: 最多探测多少个候选（每个都是一次 subprocess，别无限扫）
_MAX_PROBE = 12


# --------------------------------------------------------------------- 基础
def _importable(name: str) -> bool:
    try:
        __import__(name)
        return True
    except Exception:                                  # noqa: BLE001
        return False


def missing_deps(mods=HARD_DEPS):
    return [m for m in mods if not _importable(m)]


def _clean_env():
    """去掉会污染子进程解释器的变量。"""
    env = dict(os.environ)
    for k in ("PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP"):
        env.pop(k, None)
    env.pop(_ENV_FLAG, None)
    return env


def _norm(p: str) -> str:
    try:
        return os.path.normcase(os.path.abspath(p))
    except Exception:                                  # noqa: BLE001
        return (p or "").lower()


def _which_all(name: str):
    """PATH 里所有匹配的可执行文件（shutil.which 只返回第一个）。"""
    out = []
    for d in (os.environ.get("PATH") or "").split(os.pathsep):
        d = d.strip('"').strip()
        if not d:
            continue
        p = os.path.join(d, name)
        if os.path.isfile(p):
            out.append(p)
    return out


def _drives(prefer=None):
    """所有存在的盘符根目录，**prefer 所在盘优先**（其次字母序）。

    prefer 传脚本自身路径而不是 cwd：用户在别的盘开命令行、跑 E 盘上的包时，
    "本盘优先"才符合直觉，否则会跑去用 D 盘那份。
    """
    ref = os.path.abspath(prefer) if prefer else os.getcwd()
    try:
        cur = os.path.splitdrive(ref)[0].upper()
    except Exception:                                  # noqa: BLE001
        cur = ""
    roots = []
    try:
        roots = [d for d in os.listdrives()]           # Python 3.12+
    except Exception:                                  # noqa: BLE001
        roots = []
    if not roots:
        for c in "CDEFGHIJKLMNOPQRSTUVWXYZ":
            p = c + ":\\"
            if os.path.isdir(p):
                roots.append(p)
    roots.sort(key=lambda r: (r[:2].upper() != cur, r))
    return roots


# --------------------------------------------------------------------- 候选
def bundled_python():
    """包内自带的解释器（``<包根>\\python\\python.exe``），没有则 None。

    同时覆盖"同盘其它位置的便携包"，因为开发态目录旁边通常没有 python\\。
    """
    here = os.path.dirname(os.path.abspath(__file__))
    pkg = os.path.dirname(here)
    for cand in (os.path.join(pkg, "python", "python.exe"),
                 os.path.join(here, "python", "python.exe")):
        if os.path.isfile(cand):
            return os.path.normpath(cand)
    for root in _drives(here):
        for name in _PORTABLE_NAMES:
            cand = os.path.join(root, name, "python", "python.exe")
            if os.path.isfile(cand):
                return os.path.normpath(cand)
    return None


def candidates():
    """按可信度列出所有可能的解释器路径（去重、保序）。"""
    here = os.path.dirname(os.path.abspath(__file__))
    pkg = os.path.dirname(here)
    out, seen = [], set()

    def add(p):
        if not p:
            return
        p = os.path.abspath(p)
        if not os.path.isfile(p):
            return
        n = _norm(p)
        if n in seen:
            return
        seen.add(n)
        out.append(p)

    # 1) 用户显式指定
    add((os.environ.get(_ENV_PYTHON) or "").strip() or None)
    # 2) 包内（本目录上一级 / 本目录）
    add(os.path.join(pkg, "python", "python.exe"))
    add(os.path.join(here, "python", "python.exe"))
    # 3) 各盘根目录下的便携包
    for root in _drives(here):
        for name in _PORTABLE_NAMES:
            add(os.path.join(root, name, "python", "python.exe"))
    # 4) PATH（python.exe / python3.exe 全部）
    for exe in ("python.exe", "python3.exe"):
        for p in _which_all(exe):
            add(p)
    # 5) 官方安装目录
    local = os.environ.get("LOCALAPPDATA") or ""
    bases = [os.path.join(local, "Programs", "Python")] if local else []
    bases.append(os.environ.get("ProgramFiles", r"C:\Program Files") + r"\Python")
    for base in bases:
        try:
            subs = sorted(os.listdir(base), reverse=True)
        except Exception:                              # noqa: BLE001
            continue
        for sub in subs:
            add(os.path.join(base, sub, "python.exe"))
    return out


def _python_has(exe: str, mods=HARD_DEPS) -> bool:
    """目标解释器是否真的装了这些库（避免重启到一个同样缺依赖的环境）。

    超时给两档：U 盘冷启动时 numpy+cv2 可能要 20s 以上，第一次超时后
    再给一次更长的机会，避免误判成"没装全"。
    """
    key = (_norm(exe), tuple(mods))
    if key in _PROBE:
        return _PROBE[key]
    code = "import " + ", ".join(mods)
    env = _clean_env()
    ok = False
    for timeout in (30, 90):
        try:
            r = subprocess.run([exe, "-c", code], timeout=timeout, env=env,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.TimeoutExpired:
            continue                                   # 慢介质再给一次机会
        except Exception:                              # noqa: BLE001
            break
        ok = r.returncode == 0
        break
    _PROBE[key] = ok
    return ok


def find_usable_python(mods=HARD_DEPS):
    """返回第一个**真的装了**这些库的解释器；没有则 None。"""
    cur = _norm(sys.executable)
    tried = 0
    for exe in candidates():
        if _norm(exe) == cur:
            continue
        if tried >= _MAX_PROBE:
            break
        tried += 1
        if _python_has(exe, mods):
            return exe
    return None


def describe(probe: bool = False):
    """当前运行环境摘要（写日志 / 排查用）。

    ``probe=True`` 时才会去探测"有没有别的解释器能顶替"（要跑子进程，
    最坏十几秒）；网页接口 /api/env 用默认值，只报事实不做探测。
    """
    miss = missing_deps()
    return {
        "python": sys.executable,
        "version": sys.version.split()[0],
        "missing": miss,
        "soft_missing": missing_deps(SOFT_DEPS),
        "bundled_python": bundled_python(),
        # 只有显式 probe 才去扫盘（要跑子进程）；没探测时给 None，别谎报"可用"
        "usable_python": find_usable_python() if (probe and miss) else None,
        "relaunched": os.environ.get(_ENV_FLAG) == "1",
    }


# --------------------------------------------------------------------- 重启
def _restart_target(argv):
    """从 argv 里取出要重跑的脚本路径（拿不到就返回 None）。"""
    args = list(argv if argv is not None else sys.argv)
    if not args:
        return None
    first = args[0]
    if first.startswith("-"):                          # python -c / -m
        return None
    return os.path.abspath(first)


def ensure(argv=None, quiet: bool = False) -> bool:
    """必要时换一个装齐依赖的解释器重启当前脚本。

    返回 True 表示**已经重启**（调用方应立刻 ``sys.exit(0)`` 结束自己）；
    返回 False 表示环境没问题（或无法自救），照常继续。
    """
    if os.environ.get(_ENV_FLAG) == "1":               # 防死循环
        return False
    miss = missing_deps()
    if not miss:
        return False

    script = _restart_target(argv)
    if not script or not os.path.isfile(script):
        if not quiet:
            print("[提示] 当前解释器缺少 %s，且定位不到要重启的脚本，图片识别功能将不可用。"
                  % "、".join(miss))
        return False

    target = find_usable_python()
    if not target:
        if not quiet:
            print("[提示] 当前解释器缺少 %s（%s）。" % ("、".join(miss), sys.executable))
            print("[提示] 没找到装齐这些库的 Python，图片识别功能将不可用。")
            print("       可安装依赖：  python -m pip install numpy opencv-python")
            print("       或设置环境变量 J022_PYTHON 指向便携包内的 python.exe")
        return False
    if _norm(target) == _norm(sys.executable):
        return False

    env = _clean_env()
    env[_ENV_FLAG] = "1"
    args = [target] + list(argv if argv is not None else sys.argv)
    workdir = os.path.dirname(script)
    if not quiet:
        print("[环境] 当前解释器缺 %s：%s" % ("、".join(miss), sys.executable))
        print("[环境] 自动切换到可用解释器并重启：%s" % target)
    try:
        rc = subprocess.call(args, env=env, cwd=workdir or None)
    except Exception as e:                             # noqa: BLE001
        if not quiet:
            print("[提示] 自动重启失败（%s），继续用当前解释器运行。" % e)
        return False
    sys.exit(rc if isinstance(rc, int) and rc >= 0 else 0)


if __name__ == "__main__":
    import json
    print("候选解释器：")
    for c in candidates():
        print("  %s  -> %s" % (c, "OK" if _python_has(c) else "缺依赖"))
    print(json.dumps(describe(), ensure_ascii=False, indent=2))
