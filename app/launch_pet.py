"""launch_pet.py — 桌宠独立启动入口。

职责：
  1) 确保 Ollama 本地推理服务在跑（复用 launch.py 的自愈逻辑）
  2) 前台启动桌宠（desktop_pet.main 的消息循环）

用法：python launch_pet.py [--no-chat]
"""
from __future__ import annotations
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def main():
    print("=" * 56)
    print("  J022 Ghoul PET   ·   桌面宠物   ·   纯本地运行")
    print("=" * 56)

    # 1) 确保 Ollama 在跑（AI 对话需要；失败也不挡住桌宠启动）
    try:
        import launch
        launch.ensure_ollama()
    except Exception as e:  # noqa: BLE001
        print(f"    （Ollama 检查失败：{e}）")

    # 2) 启动桌宠
    import desktop_pet
    sys.argv = [sys.argv[0]] + (["--no-chat"] if "--no-chat" in sys.argv else [])
    desktop_pet.main()


if __name__ == "__main__":
    main()
