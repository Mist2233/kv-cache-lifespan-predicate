"""用图解释 M3：它到底是什么、怎么读。

三个子图：
  (a) 一个桶里，每个块相对桶均值的偏差是怎么算出来的（箭头 = |L−μ|/μ）；
  (b) 所有块的偏差分布（ECDF），M3 就是在这条曲线上读 50% 和 90% 两个点；
  (c) 箱线图式的汇总：箱体 = 25%~75%，须 = 90%，横线 = 中位。
"""

from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _setup import add_src_paths  # noqa: E402
add_src_paths()

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  # noqa: E402
import plotstyle  # noqa: E402
from cluster_lifetime import FEATURES, add_kind, attach, design_matrix, load  # noqa: E402

plotstyle.setup()
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def deviations(lab, y):
    lab = np.asarray(lab)
    out = np.zeros(len(y))
    for b in np.unique(lab):
        m = lab == b
        mu = y[m].mean()
        out[m] = np.abs(y[m] - mu) / mu
    return out


def main():
    td = os.path.join(ROOT, "data/trace_docvqa36")
    reqs, blocks = load(td)
    rows, _ = attach(reqs, blocks)
    bs = int(blocks[0].get("block_size", 528))
    from mm_input import MMInput
    rows = add_kind(rows, reqs, MMInput(os.path.join(ROOT, "models", "Qwen3.5-9B")),
                    bs, max(len(r["prompt_token_ids"]) for r in reqs))
    df = pd.DataFrame(rows)
    se = {}
    for r in reqs:
        se[r["sid"]] = max(se.get(r["sid"], 0.0), r["t_end"])
    df = df[df["t_death"].notna()].copy()
    df["L_A"] = df["sid"].map(se) - df["t_death"]
    df = df[df["L_A"] > 0].reset_index(drop=True)
    y = df["L_A"].values
    K = 16

    cls = np.searchsorted(np.quantile(y, np.linspace(0, 1, K + 1)[1:-1]), y, side="right")
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_predict, GroupKFold
    X = design_matrix(df, FEATURES)
    rf = cross_val_predict(
        RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=0),
        X, cls, cv=GroupKFold(4), groups=df["sid"].values)
    s0 = np.zeros(len(df), dtype=int)

    dev_or = deviations(cls, y)
    dev_rf = deviations(rf, y)
    dev_s0 = deviations(s0, y)

    fig = plt.figure(figsize=(15, 4.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.1, 1.0], wspace=0.28)

    # ---------- (a) 一个桶里的偏差怎么算 ----------
    ax = fig.add_subplot(gs[0, 0])
    b = 5
    m = cls == b
    L = y[m]
    order = np.argsort(L)
    L = L[order]
    mu = L.mean()
    x = np.arange(len(L))
    ax.scatter(x, L, s=14, color="#4C72B0", label="桶内每个块")
    ax.axhline(mu, color="#C44E52", lw=1.6, label=f"桶均值 μ = {mu:.1f}s")
    ax.fill_between([-1, len(L)], mu * 0.95, mu * 1.05, color="#C44E52", alpha=0.15,
                    label="±5% 区间")
    for i in [0, len(L) // 3, 2 * len(L) // 3, len(L) - 1]:
        ax.annotate("", xy=(x[i], mu), xytext=(x[i], L[i]),
                    arrowprops=dict(arrowstyle="<->", color="#8C8C8C", lw=1))
        ax.text(x[i] + 0.8, (mu + L[i]) / 2,
                f"{abs(L[i]-mu)/mu*100:.0f}%", fontsize=8, color="#555555")
    ax.set_xlabel("桶内的块（按寿命排序）")
    ax.set_ylabel("寿命 (s)")
    ax.set_title("(a) 每个块的偏差 = |寿命 − 桶均值| / 桶均值", fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.set_xlim(-1, len(L))

    # ---------- (b) 偏差分布：M3 就是在这条曲线上读两点 ----------
    ax2 = fig.add_subplot(gs[0, 1])
    for name, d, c in (("不分层", dev_s0, "#8C8C8C"),
                       ("RF16（我们的）", dev_rf, "#4C72B0"),
                       ("oracle16（完美）", dev_or, "#55A868")):
        xs = np.sort(d)
        ys = np.arange(1, len(xs) + 1) / len(xs)
        ax2.plot(xs * 100, ys * 100, lw=1.8, color=c, label=name)
    ax2.axhline(50, color="k", ls=":", lw=0.9)
    ax2.axhline(90, color="k", ls="--", lw=0.9)
    ax2.text(1, 51, "50% 的块", fontsize=8)
    ax2.text(1, 91, "90% 的块", fontsize=8)
    for d, c in ((dev_or, "#55A868"), (dev_rf, "#4C72B0")):
        ax2.plot([np.median(d) * 100] * 2, [0, 50], color=c, lw=0.8, ls=":")
        ax2.plot([np.percentile(d, 90) * 100] * 2, [0, 90], color=c, lw=0.8, ls=":")
    ax2.set_xscale("log")
    ax2.set_xlim(0.5, 300)
    ax2.set_xlabel("相对偏差 (%)")
    ax2.set_ylabel("累积占比 (%)")
    ax2.set_title("(b) 偏差分布：曲线与 50%/90% 的交点就是 M3", fontsize=10)
    ax2.legend(fontsize=8, frameon=False, loc="lower right")

    # ---------- (c) 箱线图式汇总 ----------
    ax3 = fig.add_subplot(gs[0, 2])
    data = [("不分层", dev_s0, "#8C8C8C"), ("RF16", dev_rf, "#4C72B0"),
            ("oracle16", dev_or, "#55A868")]
    for i, (name, d, c) in enumerate(data):
        q25, q50, q75, q90 = np.percentile(d, [25, 50, 75, 90])
        ax3.plot([q25 * 100, q75 * 100], [i, i], lw=14, color=c, alpha=0.45,
                 solid_capstyle="butt")
        ax3.plot([q75 * 100, q90 * 100], [i, i], lw=1.6, color=c)
        ax3.plot([q50 * 100] * 2, [i - 0.22, i + 0.22], lw=2.4, color="k")
        ax3.text(q50 * 100, i + 0.28, f"中位 {q50*100:.1f}%", ha="center", fontsize=8)
        ax3.text(q90 * 100, i - 0.30, f"p90 {q90*100:.1f}%", ha="center", fontsize=7,
                 color="#555555")
    ax3.set_yticks(range(len(data)))
    ax3.set_yticklabels([d[0] for d in data])
    ax3.set_xscale("log")
    ax3.set_xlim(0.5, 300)
    ax3.set_xlabel("相对偏差 (%)，对数轴")
    ax3.set_title("(c) 箱线图式读法：箱=25~75%，须=90%，黑线=中位", fontsize=10)
    ax3.invert_yaxis()

    fig.suptitle("M3 是什么：每个块离自己桶均值有多远，再看这个'多远'的分布", fontsize=12)
    out = os.path.join(ROOT, "figs", "fig14_M3_explained.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[fig] {out}")


if __name__ == "__main__":
    main()
