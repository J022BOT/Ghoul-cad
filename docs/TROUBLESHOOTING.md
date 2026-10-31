# 故障排查

按症状查。每条都是**真实遇到过**的问题，不是假设。

---

## 启动

### 双击没反应 / 黑窗口闪一下就没了

在 `cmd` 里手动跑一遍看报错：

```bat
cd /d <仓库路径>
启动 Ghoo.bat
```

最常见的原因是没跑 `一键部署.bat`，`python\python.exe` 不存在。

### `[ERROR] Missing: python\python.exe`

运行时还没装。跑一次 `一键部署.bat`（或只装 Python 部分）：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_runtime.ps1 -Only python
```

### `[提示] 当前解释器缺少 numpy、cv2`

用错解释器了（比如自己在命令行敲了 `python app\web_app.py`，用的是系统那个干净的 Python）。

正常情况会自动纠正，日志里会出现：

```
[环境] 当前解释器缺少 numpy、cv2：C:\...\python.exe
[环境] 自动切换到可用解释器并重启：D:\...\python\python.exe
```

如果没纠正成功（提示里带"没找到装齐这些库的 Python"），三选一：

1. 跑 `一键部署.bat -Only python` 装依赖；
2. 装到系统 Python：`python -m pip install numpy opencv-python pillow ezdxf`；
3. 显式指定：
   ```bat
   set J022_PYTHON=D:\ghoul-cad\python\python.exe
   启动 Ghoo.bat
   ```

网页顶部也会出现红色横幅说明当前用的是哪个解释器。

### 提示 8765 端口被占用

上一次的窗口没关干净：

```powershell
netstat -ano | findstr :8765
taskkill /PID <上面查到的PID> /F
```

或者改端口：编辑 `app\web_app.py` 顶部的 `PORT = 8765`。

### 提示 11434 端口被占用 / 模型列表为空

`ollama serve` 在跑但丢了 `OLLAMA_MODELS`（看门狗重启时会这样，模型列表就空了）。
直接点网页上的「🔧 模型卡住？」，或关掉所有 ollama 进程后重新启动：

```powershell
taskkill /F /IM ollama.exe
taskkill /F /IM "ollama app.exe"
启动 Ghoo.bat
```

---

## 识图

### 上传图片后一条线都没识别到

**多半是颜色过滤和黑白化的顺序问题**（旧版本的 bug，已修）。
判定顺序必须是：先按颜色筛墨迹，再做黑白化 —— 反过来做会把整张图变成纯白背景。

如果你用的是自己改过的版本，确认 `cad_agent` 里"有颜色过滤指令时跳过 bw_photo"这段还在。

### 识别出一堆乱七八糟的短线

说明这次走的是"保留全部墨迹"路线，把扫描噪点、纸张纹理都算进去了。
建议：

- 加颜色条件（「只要黑色线条」/「只要红框」）；
- 或者把图片裁到只留图形区域再传。

### 「Homotope 辅助」显示不可用

引擎要 Node ≥ 22。检查：

```powershell
tools\node\node.exe --version
```

不到 22 或文件不存在 → `一键部署.bat` 里只装 node + homotope 两部分。
另外 `tools\homotope\node_modules` 缺失时也会不可用（`npm install` 没跑成功）。

> 它只是"辅助"：OpenCV 能出合格结果时不会调用它。

---

## 图纸

### LibreCAD 里中文/符号变成一个个小方框

两件事同时满足才行：

1. **字体**：DXF 的 STYLE 名必须等于字体文件名（LibreCAD 拿样式名去 `resources/fonts/`
   找 `<名>.lff`），本仓库用的是 `noto-sans-sc`，随包附字体。
2. **草稿模式**：LibreCAD 的"草稿模式"会把所有文字画成方框（标题栏会带 `[草稿模式]`）。
   打开图纸前程序会自动关掉它（写注册表
   `HKCU\Software\LibreCAD\LibreCAD\Appearance\DraftMode`，2.2.x 另有一份在
   `...\LibreCAD-2\Appearance`，两个都写）。

手动关：LibreCAD → 选项 → 应用首选项 → 外观 → 取消「草稿模式」。

### 导出 DWG 弹「dxf2dwg.exe - 系统错误 找不到 libiconv-2.dll」

`tools\libredwg\` 缺依赖。这个 exe 依赖链是
`dxf2dwg.exe → libredwg-0.dll → libiconv-2.dll`，缺了不会给正常错误，
而是弹模态框把后台转换永久挂住（所以程序会在启动转换器之前就拦住）。

重装这部分：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_runtime.ps1 -Only libredwg -Force
```

### DWG 导出失败 / 提示转换器不可用

`一键部署.bat` 里 LibreDWG 那步失败了（GitHub 直连不通最常见）。
脚本会自动试 `ghfast.top` 镜像，仍失败就手动下
[libredwg-0.14-win64.zip](https://github.com/LibreDWG/libredwg/releases/download/0.14/libredwg-0.14-win64.zip)，
把 `dxf2dwg.exe`、`libredwg-0.dll`、`libiconv-2.dll` 放进 `tools\libredwg\`。

**不影响 DXF**：转换失败时 DXF 照常交付，只是没有 DWG。

### LibreCAD 打不开生成的 DWG

已知限制：LibreCAD 2.2.2（官方目前只有 alpha 连续构建）**读不了 LibreDWG 生成的 DWG**。
程序遇到 `.dwg` 会改用资源管理器定位文件，请用 AutoCAD / 浩辰 / BricsCAD 等打开，
或者直接用同时生成的 DXF。

### 「未安装 ezdxf，无法解析 DXF」

新版本不会再有这个提示：导入 DXF 有内置的纯 Python 解析器兜底
（`app\dxf_reader.py`，零依赖）。只有**二进制 DXF** 会明确提示另存为 ASCII。

---

## 模型

### 模型一直转圈 / 不回话

点网页上的「🔧 模型卡住？」：会卸掉显存里的旧模型并重新预热。

### 第一次回答特别慢（几十秒到两分钟）

冷加载模型。7B 模型在机械盘 / U 盘上实测能到 157 秒，SSD 上约 4.4 秒。
程序把首字时限放到 200 秒，并用 `keep_alive=30m` 常驻 —— 第二次开始就快了。

想手动预热：

```powershell
ollama run qwen2.5-coder:7b "hi"
```

---

## 桌宠

### 桌宠不见了

`只启动桌宠.bat` 单独起；或者看任务管理器里有没有 `pythonw.exe` / `python.exe`。

它的位置记在 `app\pet_*.json` 之类的状态文件里，被拖到屏幕外时删掉状态文件即可复位。

### 点一下桌宠就闪退（事件查看器显示 0xc000041d）

Win32 WNDPROC 回调被 GC 回收了。修复方式是把回调对象存进进程常驻字典
（`_CLASS_PROCS` / `_EDIT_PROCS`）—— 本仓库的代码已经是这样写的，
如果你改过 `desktop_pet.py` 请保留这个约定。

---

## 编辑器

### 选中了但拖不动

内嵌 JS 在 `mousemove` 里抛异常了（Python 侧完全看不到）。按 F12 看 Console。
最常见原因：改过 `moveEntity()` 的签名，而调用方还在传图元对象。

跑一遍前端测试能直接抓到：

```powershell
python app\_tests\test_web_editor.py
```

### 吸附太粘 / 不生效

工具栏「🧲 吸附」可以调两个强度：距离（px，默认 8）和角度（°，默认 6）。
调 0 即关闭对应吸附。

---

## U 盘 / 便携

### 插上 U 盘后完全没反应（没有盘符、没有弹出项）

**不要用 `diskpart` 的 `remove all dismount`** —— `remove all` 会把卷的所有挂载点一起删掉，
包括盘符本身，下次插入就彻底认不到。

用随包的 `修复U盘盘符.bat`（桌面 / 仓库根目录各一份，需要管理员）。

### U 盘上跑得特别慢

模型和 Python 都在 U 盘上，冷启动 IO 是瓶颈。建议：

- 把模型放本机 SSD，用 `set OLLAMA_MODELS=D:\ollama\models` 指过去；
- 或者干脆只把仓库放 U 盘、运行时装在本机。

### 弹出失败（"设备正在使用中"）

用 `关闭运行.bat`，它会：结束本包内的 python/ollama 进程 → 关残留窗口 →
关停在本盘上的资源管理器 → 刷新写缓存 → 先走系统级"释放卷 + 弹出介质"，
被拒绝再回退到资源管理器弹出。
