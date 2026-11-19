"""把工作点固定在指定 ρ 上，重跑所有分组方法（论文主表）。

ρ = L / T_dev，T_dev = C / W  →  C = L · W / ρ
给定期望的 ρ（默认 0.41，对齐题面 baseline ≈1.28），反推盘容量，然后比较各分组。

用法：python src/compare_at_rho.py data/trace_docvqa36 --rho 0.41
"""

from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _setup import add_src_paths  # noqa: E402
add_src_paths()

import argparse
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
    ap = argparse.ArgumentParser()
    ap.add_argument("trace", nargs="?", default="data/trace_docvqa36")
    ap.add_argument("--rho", type=float, default=0.41)
    a = ap.parse_args()
    td = a.trace if os.path.isabs(a.trace) else os.path.join(ROOT, a.trace)
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

    # 固定物理量
    zero = np.zeros(len(df), dtype=int)
    c0 = build_chunks(df, zero, size)
    W = sum(c.size for c in c0) / (max(c.t_death for c in c0) - min(c.t_write for c in c0))
    L50 = float(np.median(y))
    cap_gb = L50 * W / a.rho / (1 << 30)
    live = max_live_bytes(c0)
    print(f"[{tag}] L50={L50:.1f}s W={W/1e9:.2f}GB/s → 目标 ρ={a.rho} ⇒ 盘容量 {cap_gb:.0f}GB")
    print(f"       活跃数据 {live/(1<<30):.0f}GB，占用率 {live/(cap_gb*(1<<30)):.2f}，"
          f"RU 数 {int(cap_gb*1024/RU_MB)}")

    # 分组
    from sklearn.cluster import KMeans
    X = design_matrix(df, FEATURES)
    groups = {"s0 不分层": zero,
              "s1 命中三层": df["cached_blocks"].map(
                  lambda v: 2 if v >= 3 else (1 if v >= 1 else 0)).values}
    for k in (4, 8, 16):
        groups[f"km{k}"] = KMeans(n_clusters=k, n_init=10,
                                  random_state=0).fit_predict(X)
        groups[f"oracle{k}"] = np.searchsorted(
            np.quantile(y, np.linspace(0, 1, k + 1)[1:-1]), y, side="right")

    out = []
    for name, lab in groups.items():
        lab = pd.factorize(lab)[0]
        f5, p95, _ = within_frac(lab, y)
        chunks = build_chunks(df, lab, size)
        try:
            d = Device(chunks, capacity_gb=cap_gb, ru_mb=RU_MB, op_frac=OP)
            r = d.run()
            wa, delta = r["wa_steady"], r["delta"]
        except RuntimeError as e:
            wa, delta = float("nan"), float("nan")
        out.append(dict(method=name, streams=int(lab.max()) + 1, pass5=f5,
                        p95=p95, wa=wa, delta=delta,
                        feasible=not np.isnan(wa)))
        print(f"  {name:<12} streams={out[-1]['streams']:3d} ±5%={f5*100:5.1f}% "
              f"p95={p95*100:4.0f}% WA={wa if np.isnan(wa) else f'{wa:.3f}'}")

    res = pd.DataFrame(out)
    res.to_csv(os.path.join(ROOT, "data", f"compare_rho{a.rho}_{tag}.csv"), index=False)
    plot(res, tag, a.rho, cap_gb)


def plot(res, tag, rho, cap_gb):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(res))
    ok = res["feasible"]
    ax.bar(x[ok], res.loc[ok, "wa"], color="#4C72B0")
    for i, v in zip(x[ok], res.loc[ok, "wa"]):
        ax.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=8)
    for i in x[~ok]:
        ax.text(i, 1.02, "不可行\n(RU 不够)", ha="center", fontsize=7, color="#C44E52")
    ax.axhline(1.1, color="k", ls="--", lw=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{m}\n({n} stream, ±5%={p*100:.0f}%)"
                        for m, n, p in zip(res["method"], res["streams"], res["pass5"])],
                       fontsize=7)
    ax.set_ylabel("稳态写放大 WA")
    ax.set_ylim(0.95, max(1.35, res["wa"].max() * 1.1))
    ax.set_title(f"固定 ρ={rho}（盘 {cap_gb:.0f}GB，RU 6GB）下的分组对比（{tag}）",
                 fontsize=10)
    out = os.path.join(ROOT, "figs", f"fig13_compare_rho{rho}_{tag}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[fig] {out}")


if __name__ == "__main__":
    main()
