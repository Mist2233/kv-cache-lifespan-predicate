"""把不同分组策略换算成写放大 WA（调用 gc_sim）。

策略：
  s0     不分层（全部一个 stream）—— 题面红线那类
  s1     按前缀命中次数分温/冷/热三层（题面描述的 baseline）
  km_k   KMeans 聚类（只用可观测特征）
  or_k   按真实寿命分位（oracle，天花板）

产出：
  data/sim_strategies_<tag>.csv
  figs/fig05_wa.png

用法：python src/sim_strategies.py data/trace_v1
"""

from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _setup import add_src_paths  # noqa: E402
add_src_paths()

import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  # noqa: E402
import plotstyle  # noqa: E402
from cluster_lifetime import FEATURES, add_kind, attach, design_matrix, load  # noqa: E402
from gc_sim import Chunk, amplify_random, max_live_bytes, run  # noqa: E402

plotstyle.setup()
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

KB_PER_TOKEN = 32          # Qwen3.5-9B：8 层 full attention × 4 KV head × 256 dim × 2B × 2(K/V)
UTIL = 0.75                # 稳态占用率：调到让"混放"≈1.3，对齐题面绿色曲线的量级
RU_MB = 128.0
OP = 0.07
K_PASSES = 200             # 复制份数（配合 WINDOW_MULT 让活跃数据≈0.85×盘容量）
WINDOW_MULT = 4.0          # 分布窗口 = WINDOW_MULT × trace 时长 → 平均约 15 份重叠


def build_chunks(df, labels, bs, trace_end):
    size = int(bs * KB_PER_TOKEN * 1024)
    out = []
    for i, (t_w, t_d, lab) in enumerate(zip(df["t_create"], df["t_death"],
                                            labels)):
        death = float(t_d) if t_d and t_d > t_w else trace_end + 1.0
        out.append(Chunk(float(t_w), death, size, int(lab)))
    return out


def main():
    global UTIL, RU_MB, K_PASSES
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("trace", nargs="?", default="data/trace_v1")
    ap.add_argument("--util", type=float, default=UTIL)
    ap.add_argument("--ru-mb", type=float, default=RU_MB)
    ap.add_argument("--tag-suffix", default="")
    ap.add_argument("--passes", type=int, default=K_PASSES)
    a = ap.parse_args()
    UTIL, RU_MB, K_PASSES = a.util, a.ru_mb, a.passes
    td = a.trace
    td = td if os.path.isabs(td) else os.path.join(ROOT, td)
    tag = os.path.basename(td)

    reqs, blocks = load(td)
    rows, _ = attach(reqs, blocks)
    bs = int(blocks[0].get("block_size", 528))
    from mm_input import MMInput
    rows = add_kind(rows, reqs, MMInput(os.path.join(ROOT, "models", "Qwen3.5-9B")),
                    bs, max(len(r["prompt_token_ids"]) for r in reqs))
    df = pd.DataFrame(rows).reset_index(drop=True)
    trace_end = max(r["t_end"] for r in reqs)
    print(f"[{tag}] 可用块 {len(df)}，每块 {bs} token = {bs*KB_PER_TOKEN/1024:.1f} MB")

    from sklearn.cluster import KMeans
    strategies: dict[str, np.ndarray] = {
        "s0 不分层": np.zeros(len(df), dtype=int),
        "s1 命中三层": df["cached_blocks"].map(
            lambda v: 2 if v >= 3 else (1 if v >= 1 else 0)).values,
    }
    # 结构分组：一个请求一个桶（不需要预测，桶内寿命天然相近）
    strategies["s2 请求级"] = pd.factorize(
        df["sid"].astype(str) + "_" + df["turn"].astype(str))[0]

    X = design_matrix(df, FEATURES)
    for k in (16,):
        strategies[f"km{k} 聚类"] = KMeans(n_clusters=k, n_init=10,
                                          random_state=0).fit_predict(X)
    for k in (16,):
        y = df["L_phys"].values
        edges = np.quantile(y, np.linspace(0, 1, k + 1)[1:-1])
        strategies[f"or{k} 真值"] = np.searchsorted(edges, y, side="right")

    out_rows = []
    for name, lab in strategies.items():
        lab = np.asarray(lab)
        # 重编号成 0..n-1
        _, lab = np.unique(lab, return_inverse=True)
        base = build_chunks(df, df["t_death"].values, bs, trace_end)
        for c, l in zip(base, lab):
            c.stream = int(l)
        chunks = amplify_random(base, K_PASSES, window=WINDOW_MULT * trace_end)

        live = max_live_bytes(chunks)
        cap_gb = live / UTIL / (1 << 30)
        try:
            r = run(chunks, capacity_gb=cap_gb, ru_mb=RU_MB, op_frac=OP)
        except RuntimeError as e:
            print(f"  {name:<12} 仿真失败：{e}")
            continue
        out_rows.append(dict(strategy=name, n_streams=int(lab.max()) + 1,
                             live_gb=live / (1 << 30), cap_gb=cap_gb,
                             host_gb=r["host_gb"], wa=r["wa"],
                             wa_steady=r["wa_steady"], delta=r["delta"],
                             gc=r["gc_count"], copies_gb=r["copies_gb"]))
        print(f"  {name:<12} streams={int(lab.max())+1:3d}  cap={cap_gb:6.1f}GB  "
              f"WA={r['wa']:.3f} steady={r['wa_steady']:.3f} δ={r['delta']:.3f} "
              f"GC={r['gc_count']:6d}")

    res = pd.DataFrame(out_rows)
    sfx = a.tag_suffix
    res.to_csv(os.path.join(ROOT, "data", f"sim_strategies_{tag}{sfx}.csv"), index=False)
    plot(res, f"{tag}{sfx}", td)


def plot(res, tag, td):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))

    ax = axes[0]
    order = list(res["strategy"])
    colors = ["#8C8C8C" if s.startswith("s0") else
              "#DD8452" if s.startswith("s1") else
              "#4C72B0" if s.startswith("km") else "#55A868" for s in order]
    ax.barh(order, res["wa_steady"], color=colors)
    for i, (w, ns) in enumerate(zip(res["wa_steady"], res["n_streams"])):
        ax.text(w + 0.01, i, f"{w:.3f}  ({ns} streams)", va="center", fontsize=8)
    ax.axvline(1.1, color="k", ls="--", lw=1.2)
    ax.text(1.11, -0.6, "目标 1.1", fontsize=9)
    ax.set_xlabel("稳态写放大 WA")
    ax.set_title("(a) 不同分组策略的稳态写放大", fontsize=11)
    ax.invert_yaxis()
    ax.set_xlim(1.0, max(res["wa_steady"]) * 1.25)

    ax2 = axes[1]
    ax2.plot(res["n_streams"], res["wa_steady"], "o-", color="#C44E52")
    for _, r in res.iterrows():
        ax2.annotate(r["strategy"], (r["n_streams"], r["wa_steady"]),
                     fontsize=7, alpha=0.75,
                     xytext=(3, 3), textcoords="offset points")
    ax2.axhline(1.1, color="k", ls="--", lw=1.2)
    ax2.set_xlabel("stream（桶）数")
    ax2.set_ylabel("稳态 WA")
    ax2.set_title("(b) 桶数 vs 写放大（同一张盘，只改分组）", fontsize=11)
    ax2.set_xscale("log", base=2)

    fig.suptitle(f"分组策略 → 盘侧写放大（{tag}，KU={K_PASSES}，"
                 f"RU={RU_MB:.0f}MB，占用率{UTIL:.0%}）", fontsize=12)
    out = os.path.join(ROOT, "figs", f"fig05_wa_{tag}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[fig] {out}")


if __name__ == "__main__":
    main()
