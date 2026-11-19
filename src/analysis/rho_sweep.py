"""ρ 扫描：把"寿命 / 盘重写周期"这个无量纲量定义清楚，并测出分组有效的区间。

定义（本文件为准）
------------------
令
  C  = 盘的可写容量（host-visible，= 模拟里 Device 的 capacity）
  W  = 稳态下的主机写入速率（字节/秒，取放大后 trace 的 总字节 / 时间跨度）
  T  = C / W            ← 盘被整体重写一遍所需时间（device residence time）

  L  = 块寿命的**字节加权中位数**（按 chunk 大小加权；本实验所有 chunk 等大，
       即普通中位数）
  ρ  = L / T            ← 主指标：数据寿命相对于盘重写周期的位置
  R  = (L90 − L10) / T  ← 辅助指标：寿命分布的跨度相当于几个"盘世代"，
                           它决定分组能把寿命分开多少

为什么这样定义
--------------
* 分组收益来自"同一 GC 时刻，有的数据已死、有的还活着"。
  这个差别只有在**寿命跨度与 T 可比**时才存在：
    - 全部寿命 ≪ T：都在 GC 前死光 → 分不分组 δ 都=0 → WA=1
    - 全部寿命 ≫ T：都还活着 → 分不分组 δ 都≈利用率
    - 寿命跨度 ~ T：分组才把 δ 从"按比例混合"变成"按类分开"
* 所以 ρ 决定"工作点"，R 决定"可分性"。单看达标率或单看 WA 都不够。

用法：python src/rho_sweep.py data/trace_docvqa36
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
from cluster_lifetime import (  # noqa: E402
    FEATURES, add_kind, attach, design_matrix, load, within_frac,
)
from gc_sim import Chunk, Device, max_live_bytes  # noqa: E402

plotstyle.setup()
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

KB_PER_TOKEN = 32
RU_MB = 6144.0
OP = 0.07
PASSES = 120


def build_chunks(df, lab, size, passes=PASSES, seed=0, jitter=0.10):
    rng = np.random.default_rng(seed)
    base = [Chunk(float(tw), float(tw) + max(float(l), 0.05), size, int(s))
            for tw, l, s in zip(df["t_death"].values, df["L_A"].values, lab)]
    span = float(df["t_death"].max())
    out = []
    for _ in range(passes):
        off = rng.uniform(0, span * 4)
        for c in base:
            life = (c.t_death - c.t_write) * (1 + rng.uniform(-jitter, jitter))
            out.append(Chunk(c.t_write + off, c.t_write + off + max(life, 0.05),
                             c.size, c.stream))
    return out


def main():
    trace = sys.argv[1] if len(sys.argv) > 1 else "data/trace_docvqa36"
    td = trace if os.path.isabs(trace) else os.path.join(ROOT, trace)
    tag = os.path.basename(td)

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
    size = int(bs * KB_PER_TOKEN * 1024)
    y = df["L_A"].values

    # ---- 分组方案 ----
    groups = {}
    groups["s0 不分层"] = np.zeros(len(df), dtype=int)
    groups["s1 命中三层"] = df["cached_blocks"].map(
        lambda v: 2 if v >= 3 else (1 if v >= 1 else 0)).values
    from sklearn.cluster import KMeans
    groups["km16"] = KMeans(n_clusters=16, n_init=10, random_state=0).fit_predict(
        design_matrix(df, FEATURES))
    groups["oracle16"] = np.searchsorted(
        np.quantile(y, np.linspace(0, 1, 17)[1:-1]), y, side="right")
    groups = {k: pd.factorize(v)[0] for k, v in groups.items()}

    # ---- 固定的物理量 ----
    chunks0 = build_chunks(df, groups["s0 不分层"], size)
    live = max_live_bytes(chunks0)
    ws = [c.t_write for c in chunks0]
    span = max(c.t_death for c in chunks0) - min(ws)
    W = sum(c.size for c in chunks0) / span          # 字节/秒
    L50 = float(np.median(y))
    L10, L90 = np.percentile(y, [10, 90])
    print(f"[{tag}] 寿命 中位 {L50:.1f}s  p10 {L10:.1f}s  p90 {L90:.1f}s")
    print(f"       活跃数据 {live/(1<<30):.0f}GB  写入速率 W={W/1e9:.2f} GB/s")

    results = []
    # 用"活跃数据/容量"扫 ρ：容量越大 ρ 越小
    for util in (0.50, 0.55, 0.60, 0.65, 0.70, 0.74, 0.80, 0.85):
        C = live / util
        T = C / W
        rho = L50 / T
        R = (L90 - L10) / T
        row = dict(util=util, cap_gb=C / (1 << 30), T_dev=T, rho=rho, R=R)
        ok = True
        for name, lab in groups.items():
            chunks = build_chunks(df, lab, size)
            try:
                d = Device(chunks, capacity_gb=C / (1 << 30), ru_mb=RU_MB, op_frac=OP)
                r = d.run()
                row[name] = r["wa_steady"]
                row[name + "_delta"] = r["delta"]
            except RuntimeError:
                row[name] = float("nan")
                ok = False
        results.append(row)
        print(f"  util={util:.2f} C={row['cap_gb']:6.0f}GB T={T:6.0f}s ρ={rho:.2f} R={R:.2f} | "
              + "  ".join(f"{k.split()[0]}={row[k]:.3f}" if not np.isnan(row[k]) else f"{k.split()[0]}=FAIL"
                          for k in groups))

    res = pd.DataFrame(results)
    res.to_csv(os.path.join(ROOT, "data", f"rho_sweep_{tag}.csv"), index=False)
    plot(res, tag)


def plot(res, tag):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

    ax = axes[0]
    styles = {"s0 不分层": ("#8C8C8C", "o--"), "s1 命中三层": ("#DD8452", "s--"),
              "km16": ("#4C72B0", "^-"), "oracle16": ("#55A868", "D-")}
    for k, (c, m) in styles.items():
        ax.plot(res["rho"], res[k], m, color=c, label=k, lw=1.8, ms=5)
    ax.axhline(1.1, color="k", ls=":", lw=1)
    ax.set_xlabel("ρ = 寿命中位 / 盘重写周期 T_dev")
    ax.set_ylabel("稳态 WA")
    ax.set_title("(a) 工作点 ρ 对 WA 的影响", fontsize=10)
    ax.legend(fontsize=8, frameon=False)

    ax2 = axes[1]
    gap = res["s0 不分层"] - res["oracle16"]
    ax2.plot(res["rho"], gap, "o-", color="#C44E52", lw=1.8)
    ax2.set_xlabel("ρ")
    ax2.set_ylabel("分组收益 = WA(不分层) − WA(真值分组)")
    ax2.set_title("(b) 分组到底在哪个 ρ 区间有用", fontsize=10)
    ax2.grid(alpha=0.25)
    # 标出"有意义"的区间（收益 > 0.05）
    good = res[gap > 0.05]
    if len(good):
        ax2.axvspan(good["rho"].min(), good["rho"].max(), color="#55A868", alpha=0.15)
        ax2.text(good["rho"].min(), gap.max() * 0.9,
                 f" 有意义区间 ρ∈[{good['rho'].min():.2f},{good['rho'].max():.2f}]",
                 fontsize=8, color="#2E6B45")

    fig.suptitle(f"ρ（寿命/盘重写周期）扫描（{tag}）", fontsize=12)
    out = os.path.join(ROOT, "figs", f"fig12_rho_{tag}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[fig] {out}")


if __name__ == "__main__":
    main()
