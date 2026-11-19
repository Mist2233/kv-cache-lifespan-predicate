"""桶数扫描：在 N 个 stream 的约束下，怎么把"请求"合并到桶里。

背景：同一个请求内的块，会话剩余寿命天然一致（相对标准差 ~1.3%），
所以"一请求一桶"完全不需要预测。但 FDP 的 stream 数有限（8~200），
当 N < 并发请求数时，必须把寿命相近的请求合并。

三种合并方式：
  1. random    —— 随机把请求分到 N 个桶（下限）
  2. oracle    —— 按真实剩余寿命排序后切 N 段（上限，不可实现）
  3. predict   —— 用可观测特征预测剩余寿命，排序后切 N 段（我们的方法）

对每个 (N, 方式) 同时量：
  * 题面口径：桶内相对偏差 ≤ ±5% 的块占比
  * 物理口径：稳态写放大 WA（gc_sim）

用法：python src/bucket_sweep.py data/trace_docvqa36
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
from gc_sim import Chunk, amplify_random, max_live_bytes, run  # noqa: E402

plotstyle.setup()
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

KB_PER_TOKEN = 32
UTIL = 0.80
RU_MB = 128.0
OP = 0.07
PASSES = 80


def main():
    td = sys.argv[1] if len(sys.argv) > 1 else "data/trace_docvqa36"
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
    size = int(bs * KB_PER_TOKEN * 1024)

    # ---- 请求级聚合 ----
    key = df["sid"].astype(str) + "_" + df["turn"].astype(str)
    df["req"] = pd.factorize(key)[0]
    req = df.groupby("req").agg(sid=("sid", "first"), turn=("turn", "first"),
                                L_sess=("L_sess", "mean"),
                                L_phys=("L_phys", "mean")).reset_index()
    n_req = len(req)
    print(f"[{tag}] 块 {len(df)} 个，请求 {n_req} 个")

    # ---- 请求级预测（块级模型 + 按请求平均，GroupKFold 防泄漏）----
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import GroupKFold

    X = design_matrix(df, FEATURES)
    y = df["L_sess"].values
    pred = np.zeros(len(df))
    for tr, te in GroupKFold(n_splits=4).split(X, y, df["sid"].values):
        m = HistGradientBoostingRegressor(max_depth=6, learning_rate=0.08,
                                          max_iter=300, random_state=0)
        m.fit(X[tr], np.log1p(y[tr]))
        pred[te] = np.expm1(m.predict(X[te]))
    df["pred"] = pred
    req["pred"] = df.groupby("req")["pred"].mean().values

    rng = np.random.default_rng(0)
    rows_out = []
    for N in (8, 16, 32, 64, 128):
        if N > n_req:
            continue
        for mode in ("random", "oracle", "predict"):
            if mode == "random":
                score = rng.random(n_req)
            elif mode == "oracle":
                score = req["L_sess"].values
            else:
                score = req["pred"].values
            order = np.argsort(score)
            bucket = np.empty(n_req, dtype=int)
            bucket[order] = np.floor(np.arange(n_req) * N / n_req).astype(int)
            lab = bucket[df["req"].values]

            f5, p95, _ = within_frac(lab, df["L_sess"].values)

            chunks = []
            for t_w, t_d, s in zip(df["t_create"].values, df["t_death"].values, lab):
                death = float(t_d) if t_d and t_d > t_w else trace_end + 1.0
                chunks.append(Chunk(float(t_w), death, size, int(s)))
            big = amplify_random(chunks, PASSES, window=4 * trace_end)
            live = max_live_bytes(big)
            # 盘容量：按占用率定，但至少给每个 stream 留 3 个 RU，避免高 N 时的活锁
            cap_gb = max(live / UTIL, 3 * N * RU_MB * (1 << 20)) / (1 << 30)
            try:
                r = run(big, capacity_gb=cap_gb, ru_mb=RU_MB, op_frac=OP)
                wa = r["wa_steady"]
            except RuntimeError as e:
                wa = float("nan")
                print(f"   N={N} {mode}: 仿真失败 {e}")
            rows_out.append(dict(N=N, mode=mode, pass5=f5, p95=p95, wa=wa))
            print(f"  N={N:3d} {mode:<8} ±5%={f5*100:5.1f}%  p95={p95*100:4.0f}%  WA={wa:.3f}")

    res = pd.DataFrame(rows_out)
    res.to_csv(os.path.join(ROOT, "data", f"bucket_sweep_{tag}.csv"), index=False)
    plot(res, tag)


def plot(res, tag):
    style = {"random": ("#8C8C8C", "o--", "随机合并"),
             "oracle": ("#55A868", "D-", "按真值合并（上限）"),
             "predict": ("#4C72B0", "^-", "按预测合并（我们的）")}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

    ax = axes[0]
    for mode, (c, m, lb) in style.items():
        g = res[res["mode"] == mode].sort_values("N")
        ax.plot(g["N"], g["pass5"] * 100, m, color=c, label=lb, lw=1.8, ms=6)
    ax.axhline(48, color="k", ls="--", lw=1.1)
    ax.text(9, 50, "WA≤1.1 所需 ~48%", fontsize=8)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("stream 数 N")
    ax.set_ylabel("±5% 达标率 (%)")
    ax.set_title("(a) 分组精度 vs 桶数", fontsize=10)
    ax.legend(fontsize=8, frameon=False)

    ax2 = axes[1]
    for mode, (c, m, lb) in style.items():
        g = res[res["mode"] == mode].sort_values("N")
        ax2.plot(g["N"], g["wa"], m, color=c, label=lb, lw=1.8, ms=6)
    ax2.axhline(1.1, color="k", ls="--", lw=1.1)
    ax2.set_xscale("log", base=2)
    ax2.set_xlabel("stream 数 N")
    ax2.set_ylabel("稳态写放大 WA")
    ax2.set_title("(b) 写放大 vs 桶数", fontsize=10)
    ax2.legend(fontsize=8, frameon=False)

    fig.suptitle(f"有限 stream 下的请求合并（{tag}）", fontsize=12)
    out = os.path.join(ROOT, "figs", f"fig09_bucket_sweep_{tag}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[fig] {out}")


if __name__ == "__main__":
    main()
