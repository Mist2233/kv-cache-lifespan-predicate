"""绘图统一样式（中文字体 + 配色）。"""

from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _setup import add_src_paths  # noqa: E402
add_src_paths()

import os

import matplotlib

matplotlib.use("Agg")

CJK_FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc",
]

# 语义配色（全文统一）
C_HOT = "#C44E52"      # 热
C_WARM = "#DD8452"     # 温
C_COLD = "#4C72B0"     # 冷
C_GRAY = "#8C8C8C"
C_GREEN = "#55A868"
C_PURPLE = "#8172B3"


def setup():
    import matplotlib.font_manager as fm
    from matplotlib import rcParams

    for p in CJK_FONT_CANDIDATES:
        if os.path.exists(p):
            fm.fontManager.addfont(p)
            name = fm.FontProperties(fname=p).get_name()
            rcParams["font.family"] = "sans-serif"
            rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            break
    rcParams["axes.unicode_minus"] = False
    rcParams["figure.dpi"] = 110
    rcParams["savefig.dpi"] = 160
    rcParams["axes.grid"] = True
    rcParams["grid.alpha"] = 0.22
    rcParams["axes.axisbelow"] = True
