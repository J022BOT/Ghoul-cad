"""ollama_stream.py — 对本地 Ollama /api/chat 的流式、带时限保护的请求封装。

为什么单独抽出来：桌宠与网页端都要和 Ollama 对话，且都遇到过同样的稳定性问题——
Ollama 卡慢/冷加载时，旧的「一次性读全量 + 固定 socket 超时」会让调用方长时间挂起
（桌宠停在「……」被误认卡死、网页请求转圈数分钟）。

本模块关键点：
  * 直接基于 socket 增量读取响应体（不再用 urllib 的 read(n)，后者会一直阻塞到
    凑满 n 字节或 EOF，在 Ollama 停滞时无法及时超时）；
  * 自己处理 chunked 分块与 NDJSON 逐行解析；
  * 三道时限（首字 / 停滞 / 总时长）+ 动态 socket 超时：即使服务端长时间不发数据
    或卡死，也只在 total 秒内返回，绝不无限挂起；
  * 连接不上 / 模型未拉取 / 超时都有清晰中文报错，而不是吞掉或假死。
仅依赖标准库。
"""

from __future__ import annotations

import json
import socket
import ssl
import time
from urllib.parse import urlparse

# 默认三道时限（秒）
DEFAULT_FIRST_TOKEN = 90.0   # 允许模型冷加载 / 出首个字的最长时间
DEFAULT_STALL = 30.0         # 出字后，超过该秒数无新内容判为停滞
DEFAULT_TOTAL = 240.0        # 单轮绝对上限
DEFAULT_SOCKET_TIMEOUT = 65.0  # 建连 / 读响应头的最长阻塞（兜底）
DEFAULT_KEEP_ALIVE = "30m"   # 生成后把模型在显存里留 30 分钟

# 读响应头的总时限：Ollama 有时会先回一行再「滴」数据，逐次 recv 的 5 秒
# 超时永远等不到头，旧实现会一直循环（表现就是"卡住"）。这里加总闸。
HEADER_TOTAL = 20.0


def _decode_chunks(buf):
    """解码 chunked-transfer 字节。返回 (已解码字节, 未处理剩余字节, 是否结束)。

    buf 可能含不完整的分块（跨多次 recv），此时只解码完整部分，剩余原样退回。
    """
    out = b""
    finished = False
    i = 0
    n = len(buf)
    while True:
        crlf = buf.find(b"\r\n", i)
        if crlf == -1:
            break  # 长度行不完整
        size_line = buf[i:crlf].split(b";")[0].strip()
        try:
            size = int(size_line, 16)
        except ValueError:
            break
        i = crlf + 2  # 跳过 size 行后的 CRLF
        if size == 0:
            finished = True
            break
        if n - i < size + 2:  # 数据 + 结尾 CRLF 还不完整
            break
        out += buf[i:i + size]
        i = i + size + 2  # 跳过 data + CRLF
    return out, buf[i:], finished


def stream_ollama_chat(messages, model="qwen2.5-coder:7b", host="http://localhost:11434",
                       temperature=0.7, num_predict=None, on_token=None,
                       first_token=DEFAULT_FIRST_TOKEN, stall=DEFAULT_STALL,
                       total=DEFAULT_TOTAL, socket_timeout=DEFAULT_SOCKET_TIMEOUT,
                       keep_alive=DEFAULT_KEEP_ALIVE, num_ctx=None):
    """流式请求 Ollama，逐 token 读取并做时限保护。

    返回 (text, err)：err 非空即失败（text 可能为已收到的部分内容）。
    on_token(text_so_far): 每收到新片段回调一次，用于实时出字（可传 None）。
    keep_alive: 生成结束后模型保留多久（"30m"/-1 常驻），避免反复卸载→冷加载。
    num_ctx: 上下文窗口，默认用模型自带值；显式给小值可显著降低显存抖动。
    """
    options = {"temperature": temperature}
    if num_predict:
        options["num_predict"] = num_predict
    if num_ctx:
        options["num_ctx"] = int(num_ctx)
    payload_obj = {
        "model": model,
        "messages": messages,
        "stream": True,
        "options": options,
    }
    if keep_alive is not None:
        payload_obj["keep_alive"] = keep_alive
    payload = json.dumps(payload_obj).encode("utf-8")

    parsed = urlparse(host)
    host_ = parsed.hostname or host
    port_ = parsed.port or (443 if parsed.scheme == "https" else 80)
    path_ = parsed.path or "/api/chat"
    use_tls = (parsed.scheme == "https")

    # 手工拼 HTTP 请求并强制 Connection: close，避免 keep-alive + chunked 的歧义，
    # 同时绕过任何系统代理（直连本地 Ollama）。
    header_lines = [
        "POST %s HTTP/1.1" % path_,
        "Host: %s" % (parsed.netloc or host_),
        "Content-Type: application/json",
        "Content-Length: %d" % len(payload),
        "Connection: close",
        "Accept: application/x-ndjson",
        "",
        "",
    ]
    req_bytes = ("\r\n".join(header_lines)).encode("utf-8") + payload

    t0 = time.time()
    last_data = t0
    chunks = []
    done_seen = False
    chunked = False
    raw_pending = b""     # 来自 recv 的（可能含 chunk 帧的）原始字节
    line_pending = ""     # 已解码但尚未切成 JSON 对象的文本
    _decoder = json.JSONDecoder()

    try:
        conn = socket.create_connection((host_, port_), timeout=socket_timeout)
        if use_tls:
            conn = ssl.create_default_context().wrap_socket(conn, server_hostname=host_)
        sock = conn
    except OSError as e:
        return "", _conn_err(e, host)

    try:
        sock.sendall(req_bytes)
        # ---- 读响应头（带总闸，防止「一滴滴」数据把循环拖死）----
        header_buf = b""
        header_t0 = time.time()
        while b"\r\n\r\n" not in header_buf:
            if time.time() - header_t0 > HEADER_TOTAL or len(header_buf) > 65536:
                return "", ("Ollama 迟迟不回响应头（>%.0f秒），可能是模型卡住了——"
                            "用「🔧 修复模型卡住」按钮清一下，或重启 Ollama"
                            % HEADER_TOTAL)
            try:
                sock.settimeout(5.0)
                piece = sock.recv(4096)
            except socket.timeout:
                continue          # 由上面的总闸决定何时放弃
            if not piece:
                return "", "Ollama 无响应（连接被关闭）"
            header_buf += piece
        header_blob, _, rest = header_buf.partition(b"\r\n\r\n")
        head_lines = header_blob.split(b"\r\n")
        try:
            status = int(head_lines[0].decode("utf-8", "ignore").split(" ")[1])
        except Exception:
            status = 0
        hdr = {}
        for ln in head_lines[1:]:
            if b":" in ln:
                k, _, v = ln.partition(b":")
                hdr[k.strip().lower().decode("utf-8", "ignore")] = v.strip().decode("utf-8", "ignore")
        if status in (400, 404):
            return "", "模型 %s 还没拉取，先 `ollama pull %s` 一下~" % (model, model)
        if status != 200:
            return "", "Ollama 返回错误 %d" % status
        chunked = hdr.get("transfer-encoding", "").lower() == "chunked"
        raw_pending = rest

        # ---- 读响应体（增量 + 时限保护）----
        while True:
            now = time.time()
            if not chunks:
                allowed = min(5.0, max(0.2, first_token - (now - t0)))
            else:
                allowed = stall
            try:
                sock.settimeout(allowed)
                more = sock.recv(8192)
            except socket.timeout:
                elapsed = time.time() - t0
                if not chunks and elapsed > first_token:
                    return "".join(chunks), "模型加载/出首字太慢（>%.0f秒），先重试一次~" % first_token
                if chunks and (time.time() - last_data) > stall:
                    return "".join(chunks), "生成中断（%.0f秒无新内容）" % stall
                if elapsed > total:
                    return "".join(chunks), "响应总超时（%.0f秒）" % total
                continue
            if not more:
                # 连接关闭：close-delimited 结束（非 chunked 时）
                break
            last_data = time.time()
            raw_pending += more

            # 解码：chunked 先去分块帧，否则已是纯 body
            if chunked:
                decoded, raw_pending, finished = _decode_chunks(raw_pending)
                line_pending += decoded.decode("utf-8", "ignore")
                if finished:
                    done_seen = True
            else:
                line_pending += raw_pending.decode("utf-8", "ignore")
                raw_pending = b""

            # 逐段解析 JSON 对象（用 raw_decode，兼容「逐行 NDJSON」与
            # 「chunked 拼接、无换行分隔」两种情形；跨 recv 的不完整对象留到下次）。
            while True:
                line_pending = line_pending.lstrip()
                if not line_pending:
                    break
                try:
                    obj, idx = _decoder.raw_decode(line_pending)
                except ValueError:
                    break  # 数据还不完整，等更多
                line_pending = line_pending[idx:]
                if obj.get("done"):
                    done_seen = True
                    break
                piece = (obj.get("message") or {}).get("content", "")
                if piece:
                    chunks.append(piece)
                    if on_token:
                        try:
                            on_token("".join(chunks))
                        except Exception:  # noqa: BLE001
                            pass
            if done_seen:
                break
            if (now - t0) > total:
                return "".join(chunks), "响应总超时（%.0f秒）" % total
    except (socket.timeout, TimeoutError):
        elapsed = time.time() - t0
        if not chunks and elapsed > first_token:
            return "".join(chunks), "模型加载/出首字太慢（>%.0f秒），先重试一次~" % first_token
        if chunks and (time.time() - last_data) > stall:
            return "".join(chunks), "生成中断（%.0f秒无新内容）" % stall
        return "".join(chunks), "响应总超时（%.0f秒）" % total
    except OSError as e:
        return "", _conn_err(e, host)
    finally:
        try:
            sock.close()
        except Exception:  # noqa: BLE001
            pass

    full = "".join(chunks).strip()
    if not full:
        return "", "Ollama 返回了空响应"
    return full, None


def _conn_err(e, host):
    reason = str(getattr(e, "reason", e))
    if "refused" in reason.lower() or "10061" in reason or "failed to connect" in reason.lower():
        return "Ollama 没启动（连不上 %s），先把它打开再聊~" % host
    if "timed out" in reason.lower():
        return "连接 Ollama 超时，可能它卡住了"
    return "连不上 Ollama：%s" % reason[:80]


# ---------------------------------------------------------------------------
# 模型状态管理：预热 / 查询 / 卸载
#
# 为什么需要：Ollama 默认 5 分钟不用就把模型从显存里卸掉，下次请求又要冷加载
# （7B 视觉模型在机械盘上能到一两分钟），用户看到的就是「卡住」。另外如果上一
# 次请求还挂在服务端（客户端已经放弃），新请求会排在它后面一直等。这里提供
# 「看现在谁在跑 / 直接卸载 / 预热」三件工具，给 web 端做自愈按钮用。
# ---------------------------------------------------------------------------
def _host_base(host: str) -> str:
    h = (host or "http://localhost:11434").rstrip("/")
    if h.endswith("/api"):
        h = h[:-4]
    return h


def _http_json(host: str, path: str, payload=None, timeout: float = 5.0) -> dict:
    """极简 JSON 客户端（绕过系统代理，直连本地 Ollama）。失败返回 {}。"""
    import urllib.request

    url = _host_base(host) + path
    try:
        body = None
        headers = {}
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=body, headers=headers,
                                     method="POST" if body else "GET")
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", "ignore")
        return json.loads(raw) if raw.strip() else {}
    except Exception:  # noqa: BLE001
        return {}


def ps_models(host: str = "http://localhost:11434", timeout: float = 3.0):
    """当前已加载（占着显存）的模型名列表；Ollama 没起来时返回 []。"""
    data = _http_json(host, "/api/ps", timeout=timeout)
    out = []
    for m in (data.get("models") or []):
        name = m.get("name") or m.get("model")
        if name:
            out.append(str(name))
    return out


def installed_models(host: str = "http://localhost:11434", timeout: float = 3.0):
    """本机已拉取的模型名列表。"""
    data = _http_json(host, "/api/tags", timeout=timeout)
    return [str(m.get("name")) for m in (data.get("models") or []) if m.get("name")]


def is_loaded(host: str, model: str) -> bool:
    """模型是否已在显存里（已加载 → 首字很快，可以用紧一点的超时）。"""
    if not model:
        return False
    base = model.split(":")[0]
    for n in ps_models(host):
        if n == model or n.split(":")[0] == base:
            return True
    return False


def unload_model(host: str, model: str, timeout: float = 6.0) -> bool:
    """把模型从显存卸载（keep_alive=0）。用于解救卡住/占满显存的模型。"""
    data = _http_json(host, "/api/generate",
                      {"model": model, "prompt": "", "keep_alive": 0,
                       "stream": False}, timeout=timeout)
    return bool(data)


def warmup_model(host: str, model: str, keep_alive=DEFAULT_KEEP_ALIVE,
                 timeout: float = 180.0) -> bool:
    """把模型预加载进显存并常驻一段时间，避免首次对话/识图冷加载卡顿。"""
    data = _http_json(host, "/api/generate",
                      {"model": model, "prompt": "", "stream": False,
                       "keep_alive": keep_alive},
                      timeout=timeout)
    return bool(data.get("done", True)) and bool(data)


def repair(host: str, models=(), warm: str = "", keep_alive=DEFAULT_KEEP_ALIVE):
    """一键自愈：卸载所有占着显存的模型 → （可选）预热指定模型。

    返回 {"ok","before","after","warmed","note","error"}。
    Ollama 卡得连 /api/ps 都不回时 ok=False，提示用户重启它。
    """
    info = {"ok": False, "before": [], "after": [], "warmed": False,
            "note": "", "error": ""}
    probe = _http_json(host, "/api/ps", timeout=3.0)
    if not probe:
        info["error"] = "连不上 Ollama（可能没启动），先双击「启动 Ghoo.bat」把服务拉起来"
        return info
    info["before"] = [str(m.get("name") or m.get("model"))
                      for m in (probe.get("models") or [])
                      if (m.get("name") or m.get("model"))]

    targets = list(info["before"])
    for m in models:
        if m and m not in targets:
            targets.append(m)
    cleared = 0
    for m in targets:
        if unload_model(host, m):
            cleared += 1
    info["after"] = ps_models(host)
    info["ok"] = True

    if warm:
        info["warmed"] = warmup_model(host, warm, keep_alive=keep_alive,
                                      timeout=180.0)
    notes = []
    if info["before"]:
        notes.append("已卸载 %d 个占显存的模型（%s）" %
                     (len(info["before"]), "、".join(info["before"])))
    else:
        notes.append("当前没有加载中的模型")
    if warm:
        notes.append(("已预热 %s，模型常驻 30 分钟" % warm) if info["warmed"]
                     else ("预热 %s 没成功（模型可能还没拉取）" % warm))
    info["note"] = "；".join(notes)
    return info
