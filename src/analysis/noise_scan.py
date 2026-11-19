"""噪声扫描：分组精度 → 写放大，找出 WA ≤ 1.1 需要多高的精度。

做法：
  1. 用真值把块分成 16 桶（oracle，WA≈1.0）；
  2. 以概率 p 把一个块随机改判到别的桶（模拟"预测错了"）；
  3. 对每个 p，同时量两个指标：
       * 题面口径：桶内相对偏差 ≤ ±5% 的块占比
       * 物理口径：稳态写放大 WA
  4. 画出 WA 随"±5% 达标率"变化的曲线，标出目标 1.1 与我们当前水平。

用法：python src/noise_scan.py data/trace_v1
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
from cluster_lifetime import add_kind, attach, load, within_frac  # noqa: E402
from gc_sim import Chunk, amplify_random, max_live_bytes, run  # noqa: E402

plotstyle.setup()
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

KB_PER_TOKEN = 32
UTIL = 0.80          # 用与 baseline 标定一致的点（baseline≈1.29，贴近题面 1.25–1.4）
RU_MB = 128.0
OP = 0.07
K_PASSES = 120
N_BUCKETS = 16


def main():
    td = sys.argv[1] if len(sys.argv) > 1 else "data/trace_v1"
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

    y = df["L_phys"].values
    base_lab = np.searchsorted(np.quantile(y, np.linspace(0, 1, N_BUCKETS + 1)[1:-1]),
                               y, side="right")
    size = int(bs * KB_PER_TOKEN * 1024)

    rng = np.random.default_rng(0)
    out = []
    for p in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        lab = base_lab.copy()
        if p > 0:
            flip = rng.random(len(lab)) < p
            lab[flip] = rng.integers(0, N_BUCKETS, flip.sum())
        pass5, p95, _ = within_frac(lab, y)

        chunks = []
        for t_w, t_d, s in zip(df["t_create"].values, df["t_death"].values, lab):
            death = float(t_d) if t_d and t_d > t_w else trace_end + 1.0
            chunks.append(Chunk(float(t_w), death, size, int(s)))
        big = amplify_random(chunks, K_PASSES, window=4 * trace_end)
        live = max_live_bytes(big)
        r = run(big, capacity_gb=live / UTIL / (1 << 30), ru_mb=RU_MB, op_frac=OP)
        out.append(dict(noise=p, pass5=pass5, p95=p95, wa=r["wa"], wa_steady=r["wa_steady"],
                        delta=r["delta"], gc=r["gc_count"]))
        print(f"p={p:.1f}  ±5%达标={pass5*100:5.1f}%  WA={r['wa']:.3f}  "
              f"稳态WA={r['wa_steady']:.3f}  δ={r['delta']:.3f}", flush=True)

    res = pd.DataFrame(out)
    res.to_csv(os.path.join(ROOT, "data", f"noise_scan_{tag}.csv"), index=False)

    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    ax.plot(res["pass5"] * 100, res["wa_steady"], "o-", color="#4C72B0", lw=2)
    ax.axhline(1.1, color="k", ls="--", lw=1.2)
    ax.text(3, 1.105, "目标 1.1", fontsize=9)
    ax.axvline(21, color="#C44E52", ls=":", lw=1.4)
    ax.text(22, res["wa_steady"].max() * 0.98, "我们当前 21%", color="#C44E52", fontsize=9)

    # 找到 WA 首次 ≤ 1.1 的精度
    ok = res[res["wa_steady"] <= 1.1]
    if len(ok):
        need = ok.sort_values("pass5").iloc[0]
        ax.axvline(need["pass5"] * 100, color="#55A868", ls=":", lw=1.4)
        ax.text(need["pass5"] * 100 + 1, 1.16,
                f"需要 {need['pass5']*100:.0f}%", color="#55A868", fontsize=9)
    ax.set_xlabel("分组精度：桶内相对偏差 ≤ ±5% 的块占比 (%)")
    ax.set_ylabel("稳态写放大 WA")
    ax.set_title(f"要多准才能把 WA 压到 1.1？（{tag}，{N_BUCKETS} 桶，占用率 {UTIL:.0%}）",
                 fontsize=11)
    ax.grid(alpha=0.25)
    out_fig = os.path.join(ROOT, "figs", f"fig07_noise_scan_{tag}.png")
    fig.savefig(out_fig, dpi=160, bbox_inches="tight")
    print(f"[fig] {out_fig}")
    print(res.round(3).to_string())


if __name__ == "__main__":
    main()
