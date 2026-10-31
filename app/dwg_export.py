# -*- coding: utf-8 -*-
"""dwg_export.py — DXF → DWG 导出（后端探测 + 多方案回退）。

为什么需要这一层
  DWG 是 Autodesk 的**二进制私有格式**，本项目自己只写 DXF（dxf_writer.py）。
  所以 DWG 一律由外部转换器生成，按优先级探测：

    1. ODA File Converter   ← 工业标准，能出 R2000~R2018，质量最好
    2. LibreDWG dxf2dwg     ← 随包携带（tools/libredwg/），离线可用，出 R2000
    3. AutoCAD accoreconsole ← 本机装了 AutoCAD 时可用
    4. QCAD 命令行          ← 装了 QCAD 时可用

  一个都找不到就**明确报错**，绝不把 DXF 改个后缀冒充 DWG —— 那种文件
  用户拿去出图/发给客户会当场打不开，比没有 DWG 更糟。

硬约束
  · 生成的 DWG 必须以 b"AC10" 开头（DWG 文件签名），否则判失败并删掉半成品；
  · 转换失败不吞异常，把转换器的原始输出一起带回去，方便排障；
  · 中间 DXF 永远保留（LibreCAD 预览、二次编辑都要用）。
"""
from __future__ import annotations

import math
import os
import shutil
import subprocess
import tempfile

import portable

# ---------------------------------------------------------------- 版本表
# DWG 文件头前 6 字节就是版本号（"AC1015" 等），这是唯一可靠的判定依据。
DWG_VERSIONS = {
    "AC1009": "R12",
    "AC1012": "R13",
    "AC1014": "R14",
    "AC1015": "R2000",
    "AC1018": "R2004",
    "AC1021": "R2007",
    "AC1024": "R2010",
    "AC1027": "R2013",
    "AC1032": "R2018",
}

# 用户能选的版本 → 各后端各自的写法
_ODA_TOKEN = {
    "R2000": "ACAD2000", "R2004": "ACAD2004", "R2007": "ACAD2007",
    "R2010": "ACAD2010", "R2013": "ACAD2013", "R2018": "ACAD2018",
}
_LIBREDWG_MAX = "R2000"          # LibreDWG 的写出器只支持到 R2000

DEFAULT_VERSION = "R2000"        # 兼容性最好；和我们的 DXF（AC1015）对齐


def _first_file(*cands):
    for c in cands:
        if c and os.path.isfile(c):
            return c
    return None


def _first_dir(*cands):
    for c in cands:
        if c and os.path.isdir(c):
            return c
    return None


# 上一次 rebuild 里被跳过的图元（类型, 原因），供日志与排障用
_SKIPS = []


# ---------------------------------------------------------------- 干净 DXF
def build_clean_dxf(entities, path: str, unit: str = "mm") -> int:
    """用 ezdxf 从零重建一份"规范" DXF（供转换器吃）。

    为什么不直接把 dxf_writer 的产物丢给转换器：我们那份 DXF 是手写的，追求的是
    "LibreCAD 能开、中文不乱码"，TABLES 里没有 LTYPE/BLOCK_RECORD、句柄也不完整。
    LibreDWG 一读就报
        Unknown LAYER.ltype CONTINUOUS
        dwg.block_control and HEADER.BLOCK_CONTROL_OBJECT missing
        HANDSEED 94 already exists
    这些不影响 LibreCAD，但会让转出来的 DWG 结构残缺。ezdxf 生成的 DXF 表结构完整，
    同样的几何转出来报错少一个数量级。

    返回成功写入的实体数；ezdxf 不可用时抛 ImportError（调用处退回原始 DXF）。
    """
    import ezdxf
    from dxf_writer import (_TYPE_ALIASES, _pt2, _num, _bool, DXF,
                            TEXT_STYLE, TEXT_STYLE_FONT,
                            dash_ltype_name, parse_dash_name, DEFAULT_DASH,
                            arrow_triangles)
    from curves import spline_definition

    _SKIPS.clear()
    doc = ezdxf.new("R2000", setup=True)
    doc.header["$INSUNITS"] = {"mm": 4, "cm": 5, "m": 6, "inch": 1}.get(
        str(unit or "mm").lower(), 4)
    msp = doc.modelspace()
    layers = {}
    # 文字样式：必须和 dxf_writer 用同一个名字。LibreCAD 把样式名当字体名，
    # 这里若退回 Standard，DWG 里的 Φ / 中文同样会缺字形。
    # 注意 ezdxf 的签名是 add(name, *, font="", dxfattribs=None)：font 是
    # keyword-only 必填参数，写成 dxfattribs={"font":...} 会抛 TypeError，
    # 之前被 except 吞掉 → 实体指向一个不存在的样式，DWG 里静默退回 Standard。
    try:
        if TEXT_STYLE not in doc.styles:
            doc.styles.add(TEXT_STYLE, font=TEXT_STYLE_FONT)
    except TypeError:                             # 老版 ezdxf
        try:
            doc.styles.add(TEXT_STYLE, dxfattribs={"font": TEXT_STYLE_FONT})
        except Exception:                         # noqa: BLE001
            pass
    except Exception:                             # noqa: BLE001
        pass

    # 虚线线型：标准 DASHED 是英寸级（dash 1.27 / gap 0.254），在毫米图纸上几乎
    # 连成实线；重定义为毫米级图案。带参图案（DASHED10_4 …）在实体循环里按需登记。
    added_lts = set()

    def ensure_linetype(name):
        if name in added_lts:
            return
        added_lts.add(name)
        pat = parse_dash_name(name) or DEFAULT_DASH
        d, g = float(pat[0]), float(pat[1])
        try:
            if name in doc.linetypes:
                doc.linetypes.remove(name)
            doc.linetypes.add(name, pattern=[d + g, d, -g], description="Dashed")
        except Exception:                         # noqa: BLE001
            pass

    try:                                          # 默认 DASHED 预登记，保持旧行为
        ensure_linetype("DASHED")
    except Exception:                             # noqa: BLE001
        pass

    def layer_aci(name):
        """图层颜色（ACI）：数字层名按 dxf_writer 的调色板取色，其余给 7（白/黑）。

        不能直接调 DXF._layer_color(None, name) —— 它是实例方法，第一个参数是 self。
        """
        try:
            n = int(str(name))
        except (TypeError, ValueError):
            return 7
        if n < 1:
            return 7
        pal = getattr(DXF, "_LAYER_COLORS", None) or [1, 2, 3, 4, 5, 6, 30, 210]
        return pal[(n - 1) % len(pal)]

    def lay(name):
        name = str(name or "0")
        if name not in layers:
            try:
                layers[name] = (doc.layers.get(name) if name in doc.layers
                                else doc.layers.add(name, color=layer_aci(name)))
            except Exception:                     # noqa: BLE001
                layers[name] = doc.layers.get("0")
                return "0"        # 层加不进表就别让实体指向一个不存在的图层
        return name

    n = 0
    for ent in entities or []:
        if not isinstance(ent, dict):
            continue
        t = str(ent.get("type") or ent.get("kind") or "").strip().lower()
        t = _TYPE_ALIASES.get(t, t)
        layer = lay(ent.get("layer"))
        at = {"layer": layer}
        if _bool(ent.get("dash")):
            d, g = _num(ent.get("dashLen"), 5.0), _num(ent.get("dashGap"), 3.0)
            name = str(ent.get("linetype") or dash_ltype_name(d, g))
            ensure_linetype(name)
            at["linetype"] = name
        elif ent.get("linetype"):
            name = str(ent.get("linetype"))
            ensure_linetype(name)
            at["linetype"] = name
        try:
            if t == "line":
                msp.add_line(_pt2(ent.get("p1", ent.get("start", ent.get("from")))),
                             _pt2(ent.get("p2", ent.get("end", ent.get("to")))), dxfattribs=at)
            elif t == "circle":
                r = ent.get("radius", ent.get("r"))
                if r is None and ent.get("diameter") is not None:
                    r = _num(ent.get("diameter")) / 2.0
                msp.add_circle(_pt2(ent.get("center", ent.get("c"))), _num(r), dxfattribs=at)
            elif t == "arc":
                r = ent.get("radius", ent.get("r"))
                s = ent.get("start", ent.get("start_angle", ent.get("begin")))
                e = ent.get("end", ent.get("end_angle", ent.get("finish")))
                if r is None or s is None or e is None:
                    raise KeyError("arc 参数不全")
                s, e = _num(s), _num(e)
                while e <= s:
                    e += 360.0
                msp.add_arc(_pt2(ent.get("center", ent.get("c"))), _num(r), s, e,
                            dxfattribs=at)
            elif t == "rect":
                if all(k in ent for k in ("x", "y", "w", "h")):
                    x, y = _num(ent["x"]), _num(ent["y"])
                    w, h = _num(ent["w"]), _num(ent["h"])
                elif "p1" in ent and "p2" in ent:
                    (x1, y1), (x2, y2) = _pt2(ent["p1"]), _pt2(ent["p2"])
                    x, y, w, h = min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1)
                else:
                    x, y = _num(ent.get("x1")), _num(ent.get("y1"))
                    w = _num(ent.get("x2")) - x
                    h = _num(ent.get("y2")) - y
                msp.add_lwpolyline([(x, y), (x + w, y), (x + w, y + h), (x, y + h)],
                                   close=True, dxfattribs=at)
            elif t == "polyline":
                pts = [_pt2(p) for p in (ent.get("points") or ent.get("vertices") or [])
                       if isinstance(p, (list, tuple, dict))]
                if len(pts) < 2:
                    raise ValueError("polyline 顶点不足")
                msp.add_lwpolyline(pts, close=_bool(ent.get("closed", ent.get("close"))),
                                   dxfattribs=at)
            elif t == "spline":
                pts = [_pt2(p) for p in (ent.get("points") or ent.get("fit_points")
                                         or ent.get("control_points") or [])
                       if isinstance(p, (list, tuple, dict))]
                closed = _bool(ent.get("closed", ent.get("close")))
                if len(pts) < 3:
                    raise ValueError("spline 控制点不足")
                fit = spline_definition({"points": [list(p) for p in pts],
                                         "closed": closed})
                ctrl, knots = fit.get("ctrl") or [], fit.get("knots") or []
                if len(ctrl) >= 2 and len(knots) == len(ctrl) + 4:
                    sp = msp.add_spline(dxfattribs=at)
                    sp.dxf.degree = 3
                    # ezdxf 1.x 把 set_control_points()/set_knots() 换成了同名属性；
                    # 两种写法都认，免得换台机器就静默丢样条。
                    if hasattr(sp, "set_control_points"):
                        sp.set_control_points([(p[0], p[1], 0) for p in ctrl])
                        sp.set_knots(knots)
                    else:
                        sp.control_points = [(p[0], p[1], 0) for p in ctrl]
                        sp.knots = knots
                    sp.closed = bool(closed)
                else:                              # 兜底：退化成多段线
                    msp.add_lwpolyline(pts, close=closed, dxfattribs=at)
            elif t == "ellipse":
                rx = _num(ent.get("rx", ent.get("a", ent.get("radius"))))
                ry = _num(ent.get("ry", ent.get("b", ent.get("radius2", rx))))
                rot = math.radians(_num(ent.get("rotation", ent.get("angle")), 0.0))
                s = math.radians(_num(ent.get("start"), 0.0))
                e = math.radians(_num(ent.get("end"), 360.0))
                if rx <= 0 or ry <= 0:
                    raise ValueError("ellipse 半径非法")
                if ry > rx:                        # DXF 要求 ratio=短/长 ≤ 1
                    rx, ry, rot = ry, rx, rot + math.pi / 2
                msp.add_ellipse(_pt2(ent.get("center", ent.get("c"))),
                                major_axis=(rx * math.cos(rot), rx * math.sin(rot)),
                                ratio=max(1e-6, min(1.0, ry / rx)),
                                start_param=s, end_param=e, dxfattribs=at)
            elif t == "text":
                value = ent.get("value", ent.get("text",
                                ent.get("content", ent.get("label", ""))))
                tb = msp.add_text(str(value), dxfattribs={
                    "height": _num(ent.get("height"), 5.0),
                    "rotation": _num(ent.get("rotation", ent.get("angle")), 0.0),
                    "style": TEXT_STYLE,
                    "layer": layer})
                tb.set_placement(_pt2(ent.get("p", ent.get("position", ent.get("at")))))
            elif t == "point":
                msp.add_point(_pt2(ent.get("p", ent.get("position", ent.get("at")))),
                              dxfattribs=at)
            else:
                continue
            n += 1
            for tri in arrow_triangles(ent):       # 箭头：实心三角形（SOLID）
                try:
                    msp.add_solid([(p[0], p[1]) for p in tri],
                                  dxfattribs={"layer": layer})
                except Exception:                  # noqa: BLE001
                    pass
        except Exception as e:                      # noqa: BLE001  单图元坏，不毁整图
            _SKIPS.append((t, "%s: %s" % (type(e).__name__, e)))
            continue
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    doc.saveas(path)
    if _SKIPS:
        # 静默丢图元最坑：DXF 里有、DWG 里没有，用户还以为导出成功。
        print("[DWG] 规范 DXF 重建跳过 %d 个图元：%s"
              % (len(_SKIPS), "；".join("%s(%s)" % (t, m) for t, m in _SKIPS[:3])))
    return n


# ---------------------------------------------------------------- 后端探测
def _find_odafc():
    """ODA File Converter：装在 Program Files 或包内 tools/odafc/。"""
    env = os.environ.get("J022_ODAFC", "").strip()
    if env and os.path.isfile(env):
        return env
    pf = os.environ.get("ProgramFiles", r"C:\Program Files")
    pf86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
    names = ("ODAFileConverter.exe", "ODAFileConverter")
    cands = []
    for base in (os.path.join(portable.PKG_ROOT, "tools", "odafc"),
                 os.path.join(portable.PKG_ROOT, "tools"),
                 os.path.join(pf, "ODA"),
                 os.path.join(pf86, "ODA")):
        if not os.path.isdir(base):
            continue
        for n in names:                       # 直接放外面
            cands.append(os.path.join(base, n))
        try:                                  # 各版本子目录：ODAFileConverter 25.x/
            for sub in sorted(os.listdir(base), reverse=True):
                for n in names:
                    cands.append(os.path.join(base, sub, n))
        except OSError:
            pass
    found = _first_file(*cands)
    return found or (shutil.which("ODAFileConverter") or "")


def _find_libredwg():
    """LibreDWG 的 dxf2dwg.exe：包内 tools/libredwg/ 优先。"""
    env = os.environ.get("J022_LIBREDWG", "").strip()
    if env:
        if os.path.isfile(env):
            return env
        p = os.path.join(env, "dxf2dwg.exe")
        if os.path.isfile(p):
            return p
    for d in (os.path.join(portable.PKG_ROOT, "tools", "libredwg"),
              os.path.join(portable.APP_DIR, "tools", "libredwg")):
        p = _first_file(os.path.join(d, "dxf2dwg.exe"))
        if p:
            return p
        if os.path.isdir(d):                 # 解压后可能多一层目录
            try:
                for sub in sorted(os.listdir(d), reverse=True):
                    p = _first_file(os.path.join(d, sub, "dxf2dwg.exe"))
                    if p:
                        return p
            except OSError:
                pass
    return shutil.which("dxf2dwg") or ""


def _resolve_dll(name, local_dir):
    """在 exe 同目录 / System32 / PATH 里找一个 DLL。"""
    cands = [os.path.join(local_dir, name),
             os.path.join(os.environ.get("SystemRoot", r"C:\Windows"),
                          "System32", name)]
    for d in os.environ.get("PATH", "").split(os.pathsep):
        d = d.strip().strip('"')
        if d:
            cands.append(os.path.join(d, name))
    for c in cands:
        if os.path.isfile(c):
            return c
    return ""


# dxf2dwg.exe 自己只导入 libredwg-0.dll，真正的坑在下一层：官方 GA 构建的
# libredwg-0.dll 还导入 libiconv-2.dll。缺了它 Windows 会弹一个模态的
# "系统错误"对话框把进程挂住 —— 在网页后台转换里就是永久卡死，用户只看到
# 导出转圈。所以执行前必须先查依赖，缺了就明确报错，绝不让它自己去弹窗。
_LIBREDWG_DLLS = ("libredwg-0.dll", "libiconv-2.dll")


def _libredwg_missing(exe):
    """返回缺失的依赖 DLL 名列表；空列表表示可以跑。"""
    if not exe:
        return list(_LIBREDWG_DLLS)
    local = os.path.dirname(os.path.abspath(exe))
    return [n for n in _LIBREDWG_DLLS if not _resolve_dll(n, local)]


def _find_accoreconsole():
    pf = os.environ.get("ProgramFiles", r"C:\Program Files")
    cands = []
    ad = os.path.join(pf, "Autodesk")
    if os.path.isdir(ad):
        try:
            for sub in sorted(os.listdir(ad), reverse=True):
                cands.append(os.path.join(ad, sub, "accoreconsole.exe"))
        except OSError:
            pass
    return _first_file(*cands) or (shutil.which("accoreconsole") or "")


def _find_qcad():
    pf = os.environ.get("ProgramFiles", r"C:\Program Files")
    return _first_file(os.path.join(pf, "QCAD", "qcad.exe")) or ""


def probe(refresh: bool = False) -> dict:
    """返回各后端可用情况（顺序即优先级）。

    {"ok": bool, "best": "odafc"|"libredwg"|..., "backends": [{...}, ...],
     "versions": ["R2000", ...], "note": "..."}
    """
    global _CACHE
    if _CACHE is not None and not refresh:
        return _CACHE
    backends = []
    oda = _find_odafc()
    if oda:
        backends.append({"id": "odafc", "name": "ODA File Converter",
                         "path": oda, "versions": list(_ODA_TOKEN),
                         "writes": True})
    lw = _find_libredwg()
    lw_missing = _libredwg_missing(lw) if lw else []
    if lw and not lw_missing:
        backends.append({"id": "libredwg", "name": "LibreDWG dxf2dwg",
                         "path": lw, "versions": ["R2000"], "writes": True})
    acc = _find_accoreconsole()
    if acc:
        backends.append({"id": "accoreconsole", "name": "AutoCAD 内核",
                         "path": acc, "versions": ["R2018", "R2013", "R2010"],
                         "writes": True})
    qc = _find_qcad()
    if qc:
        backends.append({"id": "qcad", "name": "QCAD", "path": qc,
                         "versions": ["R2000", "R2013", "R2018"], "writes": True})
    best = backends[0]["id"] if backends else None
    versions = backends[0]["versions"] if backends else []
    note = ("可用：%s" % backends[0]["name"]) if backends else (
        "未找到 DWG 转换器。把 ODA File Converter 装到 Program Files\\ODA，"
        "或把 LibreDWG 的 dxf2dwg.exe 放进包内 tools\\libredwg\\ 即可启用 DWG 输出")
    if not backends and lw and lw_missing:
        # 转换器在、依赖不在：这才是"点了导出 DWG 却弹系统错误"的真因
        note = ("LibreDWG 缺少依赖文件 %s，请把它们放回 %s（与 dxf2dwg.exe 同目录）"
                % ("、".join(lw_missing), os.path.dirname(lw)))
    _CACHE = {"ok": bool(backends), "best": best, "backends": backends,
              "versions": versions, "note": note,
              "libredwg_missing": lw_missing}
    return _CACHE


_CACHE = None


# ---------------------------------------------------------------- 结果校验
def dwg_version_of(path: str) -> str:
    """读 DWG 文件头判版本；不是 DWG 返回空串。"""
    try:
        with open(path, "rb") as f:
            head = f.read(6)
    except OSError:
        return ""
    tag = head.decode("ascii", "ignore")
    if not tag.startswith("AC10"):
        return ""
    return DWG_VERSIONS.get(tag, tag)


# ---------------------------------------------------------------- 各后端转换
def _run(cmd, cwd=None, timeout=180):
    """跑转换器，返回 (rc, 合并输出)。不抛异常，编码按 GBK→UTF-8 兜。"""
    try:
        p = subprocess.run(cmd, cwd=cwd, timeout=timeout,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except FileNotFoundError:
        return 127, "转换器不存在：%s" % cmd[0]
    except subprocess.TimeoutExpired:
        return 124, "转换超时（>%ds）" % timeout
    except Exception as e:                    # noqa: BLE001
        return 125, "%s: %s" % (type(e).__name__, e)
    out = p.stdout or b""
    for enc in ("utf-8", "gbk"):
        try:
            return p.returncode, out.decode(enc, "replace")
        except Exception:                     # noqa: BLE001
            pass
    return p.returncode, out.decode("latin-1", "replace")


def _convert_odafc(exe, dxf_path, dwg_path, version):
    """ODA CLI：ODAFileConverter <inDir> <outDir> <版本> DWG <递归> <审核> [过滤]"""
    in_dir = tempfile.mkdtemp(prefix="j022_oda_in_")
    out_dir = tempfile.mkdtemp(prefix="j022_oda_out_")
    try:
        staged = os.path.join(in_dir, "input.dxf")
        shutil.copyfile(dxf_path, staged)
        tok = _ODA_TOKEN.get(version, "ACAD2000")
        cmd = [exe, in_dir, out_dir, tok, "DWG", "0", "1", "input.dxf"]
        rc, log = _run(cmd)
        produced = os.path.join(out_dir, "input.dwg")
        if rc != 0 or not os.path.isfile(produced):
            # 有些版本不接受末尾的过滤参数，去掉再试一次
            rc2, log2 = _run([exe, in_dir, out_dir, tok, "DWG", "0", "1"])
            log = (log + "\n" + log2).strip()
            if rc2 != 0 or not os.path.isfile(produced):
                return False, log
        shutil.move(produced, dwg_path)
        return True, log
    finally:
        shutil.rmtree(in_dir, ignore_errors=True)
        shutil.rmtree(out_dir, ignore_errors=True)


def _convert_libredwg(exe, dxf_path, dwg_path, version):
    """LibreDWG：dxf2dwg -y -o <out.dwg> <in.dxf>"""
    miss = _libredwg_missing(exe)
    if miss:                    # 缺 DLL 时进程会先弹模态框，不能交给它去失败
        return False, ("缺少依赖 DLL：%s（应放在 %s 同目录）"
                       % ("、".join(miss), os.path.dirname(exe)))
    if version and version != _LIBREDWG_MAX:
        # 它的写出器最高到 R2000，实测指定更高版本会直接报错退出
        pass
    for cmd in ([exe, "-y", "-o", dwg_path, dxf_path],
                [exe, "-o", dwg_path, dxf_path],
                [exe, dxf_path]):                 # 最后一种：输出同名 .dwg
        rc, log = _run(cmd, cwd=os.path.dirname(dxf_path) or None)
        if rc == 0 and os.path.isfile(dwg_path):
            return True, log
        alt = os.path.splitext(dxf_path)[0] + ".dwg"
        if rc == 0 and os.path.isfile(alt):
            if os.path.abspath(alt) != os.path.abspath(dwg_path):
                shutil.move(alt, dwg_path)
            return True, log
    return False, log


def _convert_accoreconsole(exe, dxf_path, dwg_path, version):
    """AutoCAD 内核：脚本 _OPEN 后 _SAVEAS。"""
    scr = os.path.join(tempfile.mkdtemp(prefix="j022_acad_"), "run.scr")
    try:
        with open(scr, "w", encoding="ascii", errors="ignore") as f:
            f.write('_.OPEN\n"%s"\n_.SAVEAS\n2018\n"%s"\n_.QUIT\n'
                    % (dxf_path, dwg_path))
        rc, log = _run([exe, "/i", dxf_path, "/s", scr])
        if os.path.isfile(dwg_path):
            return True, log
        return False, log
    finally:
        shutil.rmtree(os.path.dirname(scr), ignore_errors=True)


def _convert_qcad(exe, dxf_path, dwg_path, version):
    """QCAD：-autostart 脚本（需要 QCAD 3+）。"""
    d = tempfile.mkdtemp(prefix="j022_qcad_")
    js = os.path.join(d, "conv.js")
    try:
        with open(js, "w", encoding="utf-8") as f:
            f.write('var di = getDocumentInterface();\n'
                    'di.openAndDecide("%s");\n'
                    'di.saveFile("%s");\n'
                    'di.quit();\n'
                    % (dxf_path.replace("\\", "/"), dwg_path.replace("\\", "/")))
        rc, log = _run([exe, "-platform", "offscreen", "-autostart", js])
        return os.path.isfile(dwg_path), log
    finally:
        shutil.rmtree(d, ignore_errors=True)


_HANDLERS = {
    "odafc": _convert_odafc,
    "libredwg": _convert_libredwg,
    "accoreconsole": _convert_accoreconsole,
    "qcad": _convert_qcad,
}


# ---------------------------------------------------------------- 对外接口
def convert(dxf_path: str, dwg_path: str = None, version: str = None,
            backend: str = None, timeout: int = 180,
            entities=None, unit: str = "mm") -> dict:
    """把 DXF 转成 DWG。

    entities 给了的话，会先用 ezdxf 重建一份"规范 DXF"再转（转换器更吃得下），
    重建失败就退回原始 DXF —— 转换这件事不能因为优化步骤失败而整体失败。

    返回 {"ok","path","backend","backend_name","version","note","log"}；
    失败时 ok=False + note 说明原因，且不会留下半成品文件。
    """
    dxf_path = os.path.abspath(dxf_path or "")
    if not os.path.isfile(dxf_path):
        return {"ok": False, "path": None, "note": "DXF 源文件不存在：%s" % dxf_path}
    dwg_path = os.path.abspath(dwg_path or os.path.splitext(dxf_path)[0] + ".dwg")
    info = probe()
    if not info["ok"]:
        return {"ok": False, "path": None, "note": info["note"],
                "backends": [], "backend": None}

    logs, tmpdir = [], None
    src_dxf = dxf_path
    if entities:
        try:
            tmpdir = tempfile.mkdtemp(prefix="j022_clean_")
            clean = os.path.join(tmpdir, "clean.dxf")
            n = build_clean_dxf(entities, clean, unit)
            if n:
                src_dxf = clean
                logs.append("（已用 ezdxf 重建规范 DXF：%d 个图元）" % n)
        except Exception as e:                # noqa: BLE001
            logs.append("（规范 DXF 重建失败，改用原始 DXF：%s）" % e)

    cands = info["backends"]
    if backend:
        cands = [b for b in cands if b["id"] == backend] or cands
    version = (version or DEFAULT_VERSION).upper().replace("ACAD", "R")
    if not version.startswith("R"):
        version = "R" + version.lstrip("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
    tried = []
    try:
        for b in cands:
            ver = version
            if ver not in b["versions"]:
                ver = b["versions"][-1] if b["versions"] else version
            tried.append(b["name"])
            # 上次残留会让"文件存在就算成功"误判，先清掉
            for p in (dwg_path, os.path.splitext(src_dxf)[0] + ".dwg",
                      os.path.splitext(dxf_path)[0] + ".dwg"):
                if os.path.isfile(p):
                    try:
                        os.remove(p)
                    except OSError:
                        pass
            try:
                ok, log = _HANDLERS[b["id"]](b["path"], src_dxf, dwg_path, ver)
            except Exception as e:            # noqa: BLE001
                ok, log = False, "%s: %s" % (type(e).__name__, e)
            logs.append("[%s] %s" % (b["name"], (log or "").strip()[-1500:]))
            if not ok or not os.path.isfile(dwg_path):
                continue
            real = dwg_version_of(dwg_path)
            if not real:
                # 转换器"成功"了但文件头不是 AC10xx —— 假 DWG，宁可报错
                try:
                    os.remove(dwg_path)
                except OSError:
                    pass
                logs.append("[%s] 产物不是合法 DWG（文件头缺 AC10xx 签名）" % b["name"])
                continue
            if os.path.getsize(dwg_path) < 128:
                try:
                    os.remove(dwg_path)
                except OSError:
                    pass
                logs.append("[%s] 产物过小，判失败" % b["name"])
                continue
            note = "%s 生成 %s" % (b["name"], real)
            if real != version:
                note += "（请求 %s，该后端只支持到 %s）" % (version, real)
            return {"ok": True, "path": dwg_path, "backend": b["id"],
                    "backend_name": b["name"], "version": real, "note": note,
                    "log": "\n".join(logs)}
    finally:
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)
    return {"ok": False, "path": None, "backend": None,
            "note": "DWG 转换失败（试过：%s）" % ("、".join(tried) or "无可用后端"),
            "log": "\n".join(logs)}


def status() -> dict:
    """供 Web/CLI 展示的一句话状态。"""
    info = probe()
    if info["ok"]:
        b = info["backends"][0]
        return {"ok": True, "name": b["name"],
                "versions": list(info["versions"] or []),
                "note": "DWG 输出可用（%s，支持 %s）"
                        % (b["name"], "/".join(info["versions"] or []))}
    return {"ok": False, "name": None, "versions": [],
            "note": info["note"]}


if __name__ == "__main__":
    import json
    print(json.dumps(probe(), ensure_ascii=False, indent=2))
