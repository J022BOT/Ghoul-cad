# -*- coding: utf-8 -*-
"""runtime_guard 回归：解释器定位 / 跨目录查找 / 探测真实性。

真实踩坑背景：
    在 ``D:\\Local AI\\cad-agent``（开发态：代码目录即包根，旁边没有 ``python\\``）
    里启动，而便携包在 ``D:\\J022-Ghoul-Portable``。旧实现只认"自己旁边"，
    于是直接放弃并打印"包内没有自带 python\\python.exe"——用户只在控制台看到
    一行字，图片识别就静默不可用了。
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
sys.path.insert(0, APP)

FAILS = []


def check(name, ok, detail=""):
    print(("  OK   " if ok else "  FAIL ") + name + (("  " + str(detail)) if detail else ""))
    if not ok:
        FAILS.append(name)


def t_basic():
    print("\n[1] 基本定位")
    import runtime_guard as rg
    bp = rg.bundled_python()
    check("能定位包内 python", bool(bp) and os.path.isfile(bp), bp)
    cands = rg.candidates()
    check("候选非空", len(cands) > 0, len(cands))
    check("候选都真实存在", all(os.path.isfile(c) for c in cands))
    check("候选无重复", len({os.path.normcase(c) for c in cands}) == len(cands), len(cands))


def t_probe():
    print("\n[2] 探测结果可信")
    import runtime_guard as rg
    exe = rg.find_usable_python()
    if exe is None:
        check("本机确实没有装齐依赖的解释器（跳过探测断言）", True)
        return
    check("给出的解释器真的能 import numpy+cv2", rg._python_has(exe), exe)


def t_cross_dir():
    """复现踩坑：代码放在没有 python\\ 兄弟目录的地方，也要能找回便携包。"""
    print("\n[3] 跨目录查找（开发态布局）")
    tmp = tempfile.mkdtemp(prefix="rg_")
    try:
        shutil.copy2(os.path.join(APP, "runtime_guard.py"),
                     os.path.join(tmp, "runtime_guard.py"))
        code = ("import json, runtime_guard as rg;"
                "print(json.dumps({'bp': rg.bundled_python(), 'n': len(rg.candidates())}))")
        r = subprocess.run([sys.executable, "-c", code], cwd=tmp,
                           capture_output=True, text=True, timeout=240)
        ok = r.returncode == 0
        check("子进程能运行", ok, (r.stderr or "").strip().splitlines()[-1:])
        if ok:
            d = json.loads(r.stdout.strip().splitlines()[-1])
            check("临时目录（无兄弟 python）仍能找到包内解释器",
                  bool(d["bp"]) and os.path.isfile(d["bp"]), d["bp"])
            check("候选列表跨盘扩展", d["n"] >= 2, d["n"])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def t_env_flag():
    print("\n[4] 重启防循环")
    import runtime_guard as rg
    old = os.environ.get("J022_RELAUNCHED")
    os.environ["J022_RELAUNCHED"] = "1"
    try:
        check("J022_RELAUNCHED=1 时不再重启", rg.ensure(quiet=True) is False)
    finally:
        if old is None:
            os.environ.pop("J022_RELAUNCHED", None)
        else:
            os.environ["J022_RELAUNCHED"] = old


def main():
    print("=" * 56)
    print("  runtime_guard 回归测试")
    print("=" * 56)
    t_basic()
    t_probe()
    t_cross_dir()
    t_env_flag()
    print("\n=== 结果：%d 项失败 ===" % len(FAILS))
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
