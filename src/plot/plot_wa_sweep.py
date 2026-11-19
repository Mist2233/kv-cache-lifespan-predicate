"""把不同占用率下的 WA 结果合成一张敏感性图。"""

from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _setup import add_src_paths  # noqa: E402
add_src_paths()

import glob
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import plotstyle  # noqa: E402

plotstyle.setup()
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    files = sorted(glob.glob(os.path.join(ROOT, "data", "sim_strategies_trace_v1_u*.csv")))
    if not files:
        print("没有 sweep 结果")
        return
    rows = []
    for f in files:
        util = float(f.split("_u")[-1].replace(".csv", ""))
        df = pd.read_csv(f)
        df["util"] = util
        rows.append(df)
    df = pd.concat(rows)

    style = {
        "s0 不分层": ("#8C8C8C", "o--"),
        "s1 命中三层": ("#DD8452", "s-"),
        "km16 聚类": ("#4C72B0", "^-"),
        "or16 真值": ("#55A868", "D-"),
        "or64 真值": ("#8172B3", "v-"),
    }

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    for name, g in df.groupby("strategy"):
        c, m = style.get(name, ("#000", "o-"))
        g = g.sort_values("util")
        ax.plot(g["util"], g["wa_steady"], m, color=c, label=name, lw=1.8, ms=6)
    ax.axhline(1.1, color="k", ls="--", lw=1.2)
    ax.text(0.752, 1.105, "目标 1.1", fontsize=9)
    ax.set_xlabel("盘稳态占用率（越紧越像真实满盘缓存）")
    ax.set_ylabel("稳态写放大 WA")
    ax.set_title("分组策略 → 写放大：占用率敏感性（trace_v1，RU=128MB）", fontsize=11)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.25)
    out = os.path.join(ROOT, "figs", "fig06_wa_vs_util.png")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"[fig] {out}")

    piv = df.pivot_table(index="util", columns="strategy", values="wa_steady")
    print(piv.round(3).to_string())


if __name__ == "__main__":
    main()
