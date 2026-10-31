# -*- coding: utf-8 -*-
"""mathtext.py — 把模型偶尔吐出的 LaTeX 数学/化学式转成可读 Unicode 纯文本。

桌宠气泡和网页对话都只渲染纯文本，模型若输出 \\[ ... \\] / \\( ... \\) / \\text{} /
\\frac / \\rightarrow 之类的 LaTeX，会原样显示成"代码"。本模块在显示前做一次清洗，
把常见 LaTeX 转成人类可读的等价文本。
"""

import re

# LaTeX 命令 -> Unicode（未列出的命令在最后一步会被整体删掉）
_CMD = {
    "rightarrow": "→", "to": "→", "gets": "←", "leftarrow": "←",
    "Rightarrow": "⇒", "Leftarrow": "⇐", "Leftrightarrow": "⇔",
    "times": "×", "cdot": "·", "cdotp": "·", "div": "÷", "pm": "±",
    "mp": "∓", "geq": "≥", "ge": "≥", "leq": "≤", "le": "≤",
    "neq": "≠", "ne": "≠", "approx": "≈", "equiv": "≡", "sim": "∼",
    "propto": "∝", "infty": "∞", "pi": "π", "alpha": "α", "beta": "β",
    "gamma": "γ", "delta": "δ", "theta": "θ", "lambda": "λ", "mu": "μ",
    "sigma": "σ", "omega": "ω", "phi": "φ", "psi": "ψ", "tau": "τ",
    "Delta": "Δ", "Sigma": "Σ", "Omega": "Ω", "Theta": "Θ", "Lambda": "Λ",
    "Phi": "Φ", "Gamma": "Γ", "Pi": "Π", "partial": "∂", "nabla": "∇",
    "in": "∈", "notin": "∉", "subset": "⊂", "supset": "⊃", "subseteq": "⊆",
    "supseteq": "⊇", "cup": "∪", "cap": "∩", "forall": "∀", "exists": "∃",
    "emptyset": "∅", "angle": "∠", "perp": "⊥", "parallel": "∥",
    "sum": "Σ", "prod": "Π", "int": "∫", "oint": "∮", "oint": "∮",
    "sqrt": "√", "log": "log", "ln": "ln", "lg": "lg", "exp": "exp",
    "cos": "cos", "sin": "sin", "tan": "tan", "lim": "lim",
    "deg": "°", "circ": "°", "bullet": "•", "ast": "*", "star": "★",
    "oplus": "⊕", "otimes": "⊗", "nabla": "∇", "aleph": "ℵ",
    "quad": " ", "qquad": "  ", "cdotp": "·", "dots": "…", "ldots": "…",
    "cdot": "·", "rangle": "⟩", "langle": "⟨", "mid": "|", "vert": "|",
    # 分组/格式类：直接删掉，保留内部文字
    "text": "", "textbf": "", "textit": "", "mathrm": "", "mathit": "",
    "mathbf": "", "mathsf": "", "mathcal": "", "rm": "", "bf": "",
    "em": "", "left": "", "right": "", "big": "", "Big": "", "bigg": "",
    "Bigg": "", "left.": "", "right.": "", "displaystyle": "",
    "scriptstyle": "", "textstyle": "", "limits": "", "nolimits": "",
}

_SUBS = str.maketrans("0123456789+-=()", "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎")
_SUPS = str.maketrans("0123456789+-=()", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾")


def sanitize_math(text):
    """把 LaTeX 片段转成可读纯文本；非 LaTeX 文本基本原样返回。"""
    if not text:
        return text

    # 1) 去掉显示/行内公式定界符
    text = text.replace("\\[", " ").replace("\\]", " ")
    text = text.replace("\\(", " ").replace("\\)", " ")
    text = re.sub(r"\$+", " ", text)

    # 2) \frac{a}{b} / \dfrac / \tfrac -> a/b
    text = re.sub(r"\\(?:frac|dfrac|tfrac)\s*\{([^{}]*)\}\s*\{([^{}]*)\}",
                  lambda m: m.group(1) + "/" + m.group(2), text)

    # 3) \sqrt{x} -> √x ; \sqrt[n]{x} -> ⁿ√x
    text = re.sub(r"\\sqrt\s*(?:\[([^{}]*)\])?\s*\{([^{}]*)\}",
                  lambda m: (("[" + m.group(1) + "]") if m.group(1) else "") + "√" + m.group(2),
                  text)

    # 4) 下标 _{...} 或 _X
    text = re.sub(r"_\{([^{}]*)\}", lambda m: m.group(1).translate(_SUBS), text)
    text = re.sub(r"_([A-Za-z0-9])", lambda m: m.group(1).translate(_SUBS), text)

    # 5) 上标 ^{...} 或 ^X
    text = re.sub(r"\^\{([^{}]*)\}", lambda m: m.group(1).translate(_SUPS), text)
    text = re.sub(r"\^([A-Za-z0-9])", lambda m: m.group(1).translate(_SUPS), text)

    # 6) 其它 \命令：已知命令替换，未知命令连同反斜杠一起删掉
    text = re.sub(r"\\([a-zA-Z]+)", lambda m: _CMD.get(m.group(1), ""), text)

    # 7) 清掉残留的反斜杠、分组花括号，压缩空白
    text = text.replace("\\", "")
    text = text.replace("{", "").replace("}", "")
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


# ---------------------------------------------------------------- 表情 -> 颜文字
# 桌宠气泡 / 网页对话只渲染纯文本，而气泡字体对彩色 emoji 支持很差
# （常显示为方块或墨块）。用户要求：桌宠说话一律用**颜文字**（(◕ᴗ◕) 这种
# 由符号拼出来的表情），不要 emoji。这里把常见 emoji 映射成颜文字，
# 未收录的未知 emoji 直接删除。
_EMOJI_KAOMOJI = {
    # 笑/开心
    "😂": "(≧▽≦)", "🤣": "(≧▽≦)", "😅": "(;^_^A", "😆": "(≧▽≦)",
    "😄": "(´▽`)", "😃": "(´▽`)", "😀": "(´▽`)", "😁": "(´▽`)",
    "😊": "(´∀`)", "😇": "(´∀`)", "🙂": "(´∀`)", "😌": "(´∀`)",
    "😉": "(｡･ω･｡)ﾉ", "😜": "(｡･ω･｡)ﾉ", "😛": "(｡･ω･｡)ﾉ",
    "🤗": "ヽ(・∀・)ﾉ", "🤭": "(*´艸｀)", "😏": "(￣ー￣)",
    "😎": "(●´ω｀●)", "🤓": "(･∀･)", "😈": "(｀・ω・´)",
    # 喜欢
    "😍": "(´>⋏<`)", "🥰": "(´>⋏<`)", "😘": "(´>⋏<`)",
    "😳": "(///ω///)", "🥺": "(´；ω；｀)",
    # 平静/思考/困
    "🤔": "(´･_･`)", "😐": "(・_・)", "😑": "(・_・)", "😔": "(´･_･`)",
    "😴": "(－ω－) zzZ", "🥱": "(－ω－) zzZ", "😒": "(￣ー￣)",
    "🙄": "(￣ー￣)", "🫠": "( ꒪⌓꒪)",
    # 难过/生气/惊吓
    "😭": "(；ω；｀)", "😢": "(；ω；｀)", "😤": "(`へ´)",
    "😡": "(`ε´)", "😠": "(`ε´)", "🤬": "(`ε´)",
    "😱": "(；ﾟДﾟ)", "😨": "(；ﾟДﾟ)", "😰": "(；ﾟДﾟ)", "🤯": "(；ﾟДﾟ)",
    "😵": "(；ﾟДﾟ)",
    # 手势/肯定
    "👍": "(･ω･)b", "👌": "(･ω･)b", "🙏": "m(_ _)m", "👏": "ヽ(´▽`)/",
    "🤝": "( ･∀･)つ", "🫡": "( ･∀･)ﾉ", "💪": "(`・ω・´)",
    # 物品/符号
    "🎉": "ヽ(´▽`)/", "✨": "☆*:.。.", "💡": "(´･ω･`)!?",
    "🔥": "(´▽`ʃ♡ƪ)", "🌸": "✿", "🍵": "(´∀`)", "☕": "(´∀`)",
    "🍰": "(´∀`)", "🤖": "(´･ω･`)", "👻": "(｀・ω・´)", "🐟": "(・∀・)",
    "❤️": "♡", "🧡": "♡", "💛": "♡", "💚": "♡", "💙": "♡",
    "💜": "♡", "🤍": "♡", "🖤": "♡", "🤎": "♡",
    "❤": "♡", "💕": "♡", "💖": "♡", "💗": "♡", "♥": "♡",
    "✅": "✓", "☑": "✓", "❌": "✗", "❎": "✗",
    "⚠": "!", "⚠️": "!", "⚡": "!", "⛔": "✗", "‼": "!",
    "❗": "!", "❓": "?", "⁉": "!?",
}

# 真正的 emoji 区（1F000-1FAFF）+ 变体选择符 + ZWJ。
# 刻意**不**清理 U+2600-26FF / U+2700-27BF 整体区间：那里面混有颜文字常用字符
# （☆ ★ ✿ ♡ 等），整段删会把正常颜文字也弄坏；少数常用符号已在上方字典映射。
_EMOJI_STRIP = re.compile(
    "[\U0001F000-\U0001FAFF"      # emoji 主体
    "\U0000FE00-\U0000FE0F"       # 变体选择符
    "\U0000200D]"                 # 零宽连接符(ZWJ)
)


_EMOJI_KEYS = sorted(_EMOJI_KAOMOJI, key=len, reverse=True)


def demojify(text):
    """把 emoji 换成颜文字（未知 emoji 直接删），并清理残留空格。

    用逐字符扫描而非 str.replace：这样能在"相邻两个 emoji"之间补一个空格，
    避免得到 (´∀`)(･ω･)b 这种挤在一起的结果；同时绝不会去动普通文本里
    本来就存在的 ")("（例如代码/数学里的 (a)(b)），因为空格只在发生
    emoji 替换的那个接缝处插入。
    """
    if not text:
        return text
    out, i, n = [], 0, len(text)
    while i < n:
        hit = None
        for k in _EMOJI_KEYS:
            if text.startswith(k, i):
                hit = k
                break
        if hit is not None:
            kao = _EMOJI_KAOMOJI[hit]
            # 接缝处补空格：前一个以 ')' 收尾、且新颜文字以 '(' 开头
            if (out and kao and out[-1].endswith(")")
                    and kao.startswith("(")):
                out.append(" ")
            out.append(kao)
            i += len(hit)
            continue
        ch = text[i]
        if not _EMOJI_STRIP.match(ch):     # 未知 emoji / 变体符 / ZWJ → 丢弃
            out.append(ch)
        i += 1
    text = "".join(out)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s+([，。！？、；：）】」》])", r"\1", text)
    text = re.sub(r"([（【「《])\s+", r"\1", text)
    return text.strip()


if __name__ == "__main__":
    samples = [
        r"\[\text{2HF + Sb} \rightarrow \text{SbF}_3 + \text{H}_2\]",
        r"\(E = mc^2\)",
        r"质量守恒：\frac{m_1}{V_1} = \rho",
        r"解集为 x \geq 3 且 x \leq 10",
        r"这是普通文本，没有公式。",
        r"反应：\begin{equation} a+b=c \end{equation}",
    ]
    for s in samples:
        print(repr(s), "=>", sanitize_math(s))

    print("\n-- 表情转颜文字 --")
    for t in ["好的，我来处理 😊👍", "完成啦！🎉✨",
              "你又摸鱼了 ⚠", "姐姐喜欢你 ❤️",
              "没问题 🙏", "我在思考 🤔",
              "(◕ᴗ◕) 这是颜文字，不应被改动",
              "调用 foo(a)(b) 计算… 🎯",
              "未收录的 emoji 🤹🪅 应当被删除"]:
        print(repr(t), "=>", demojify(t))
