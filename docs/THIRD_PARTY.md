# 第三方组件与许可

本仓库（`ghoul-cad`）自身以 **MIT** 许可发布（见根目录 `LICENSE`）。

运行时需要的第三方组件**不随仓库分发**，由 `scripts/setup_runtime.ps1` 在部署时下载。
它们各自的许可如下，使用时请遵守各自条款。

---

## 随仓库分发的第三方文件

| 文件 | 来源 | 许可 | 说明 |
|---|---|---|---|
| `app/assets/` 下的所有 PNG / JPG / ICO | 本项目生成 | 同本项目 | 桌宠帧图、看板徽章、图标 |
| `tools/libredwg/README-libredwg.txt`<br>`tools/libredwg/NOTICE.txt` | GNU LibreDWG | GPL-3.0-or-later | 仅许可文本，不含二进制 |

---

## 部署时下载的组件

| 组件 | 版本 | 许可 | 用途 | 是否必需 |
|---|---|---|---|---|
| [Python](https://www.python.org/) | 3.13.x | PSF-2.0 | 运行时 | 必需 |
| [NumPy](https://numpy.org/) | 2.5.x | BSD-3-Clause | 数组与几何计算 | 必需（识图） |
| [OpenCV (headless)](https://opencv.org/) | 4.11 | Apache-2.0 | 提线、霍夫、形态学 | 必需（识图） |
| [Pillow](https://python-pillow.org/) | 13.x | MIT-CMU (HPND) | 图片读写 | 必需 |
| [ezdxf](https://ezdxf.mozman.at/) | 1.4.x | MIT | DXF 读写（可选，内置读取器兜底） | 可选 |
| [Node.js](https://nodejs.org/) | 22.x | MIT | 跑 Homotope 引擎 | 可选（辅助识图） |
| [@homotope/engine](https://www.npmjs.com/package/@homotope/engine) | 0.2.x | MIT | 笔迹矢量化（centerline） | 可选 |
| [Ollama](https://ollama.com/) | 0.40.x | MIT | 本地大模型运行时 | 必需（AI 功能） |
| 模型 `qwen2.5-coder:7b` | — | [Qwen License](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct/blob/main/LICENSE) | 指令 → 几何 JSON | 必需 |
| 模型 `qwen2.5vl:7b` | — | [Qwen License](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/blob/main/LICENSE) | 看图兜底 / 读文字 | 可选 |
| [LibreCAD](https://librecad.org/) | 2.2.2 alpha | GPL-2.0 | 打开 / 校验 / 二次编辑图纸 | 可选 |
| [LibreDWG](https://www.gnu.org/software/libredwg/) | 0.14 | GPL-3.0-or-later | DXF → DWG 转换 | 可选 |

> 许可类型以各项目官方仓库为准；上表用于快速判断能不能用、怎么用。

---

## 关于 GPL 组件的边界

LibreCAD 与 LibreDWG 都是 **GPL 系**，本项目通过**独立进程 + 命令行**调用它们：

- LibreCAD：`subprocess` 启动 `librecad.exe` 打开一个文件；
- LibreDWG：`subprocess` 启动 `dxf2dwg.exe dxf dwg` 转换文件。

两者之间只交换**文件**，不链接、不改动其源码，属于"聚合"而非"衍生"，
因此本项目的 MIT 许可不受影响，你也不必因为使用本项目而公开自己的代码。

**但如果你要再分发**：

- 分发 LibreCAD / LibreDWG 的二进制，必须一并提供它们的完整源码或书面获取方式；
- 本项目**不**分发这两个二进制，只提供下载脚本 —— 脚本下载后由你自行使用，不再分发。

---

## 模型权重

`qwen2.5-coder:7b`（约 4.7 GB）与 `qwen2.5vl:7b`（约 6.0 GB）由 Ollama 从官方仓库拉取，
使用需遵守 Qwen 的许可条款。本项目不打包、不再分发模型权重。

---

## 参考与致谢

- DXF 格式细节来自 Autodesk 公开的 DXF Reference；
- LibreCAD 的"样式名必须等于字体文件名"、"草稿模式把文字画成方框"两条结论，
  来自实际调试与 LibreCAD 源码行为；
- Homotope 引擎的 `centerline` 预设设计用于笔迹单中线提取，正好对上 CAD 的中心线需求。
