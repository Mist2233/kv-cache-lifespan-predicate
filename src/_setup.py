"""路径引导：把 src/ 及其所有子目录加入 sys.path，使各脚本可跨目录 import。

所有脚本（无论位于哪个子目录）都通过：
    sys.path.insert(0, <src 的父目录>)
    from _setup import add_src_paths; add_src_paths()
统一完成路径初始化，因此目录调整后无需改动各脚本的 import。
"""
from __future__ import annotations

import os
import sys

SRC = os.path.dirname(os.path.abspath(__file__))


def add_src_paths() -> str:
    if SRC not in sys.path:
        sys.path.insert(0, SRC)
    for name in sorted(os.listdir(SRC)):
        d = os.path.join(SRC, name)
        if os.path.isdir(d) and not name.startswith(("__", ".")) and d not in sys.path:
            sys.path.insert(0, d)
    return SRC
