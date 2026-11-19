"""大规模 trace 分析：标准协议（固定利用率 U=0.74，K ≈ RU/7）。

对任意 trace：
  1. 构造特征、分桶、仿真；
  2. 比较 不分层 / A3 排序 / oracle；
  3. K 扫描找最优（受 RU 数约束）。

用法：python src/analyze_big.py data/trace_big
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
import classify_buckets as cb  # noqa: E402
from cluster_lifetime import add_kind, attach, load  # noqa: E402
from classify_buckets import build  # noqa: E402
from features_v2 import FEATURES_V2, add_features_v2  # noqa: E402
from fusion_v3 import cut, eta2_log, oof_rank, wa_of  # noqa: E402
from gc_sim import Device, max_live_bytes  # noqa: E402
from metrics import summarize  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
KB_PER_TOKEN = 32
RU_GIB = 6.0
UTIL = 0.74
EVENT_BUDGET = 200_000     # 控制仿真规模：块数 × passes ≈ 常数


def main():
    trace = sys.argv[1] if len(sys.argv) > 1 else "data/trace_big"
    td = trace if os.path.isabs(trace) else os.path.join(ROOT, trace)
    tag = os.path.basename(td.rstrip("/"))

    reqs, blocks = load(td)
    bs = int(blocks[0].get("block_size", 528))
    from mm_input import MMInput
    rows, _ = attach(reqs, blocks)
    rows = add_kind(rows, reqs, MMInput(os.path.join(ROOT, "models", "Qwen3.5-9B")),
                    bs, max(len(r["prompt_token_ids"]) for r in reqs))
    df = pd.DataFrame(rows)
    se = {}
    for r in reqs:
        se[r["sid"]] = max(se.get(r["sid"], 0.0), r["t_end"])
    df = df[df["t_death"].notna()].copy()
    df["L_A"] = df["sid"].map(se) - df["t_death"]
    df = df[df["L_A"] > 0].reset_index(drop=True)
    df = add_features_v2(df, reqs)
    size = int(bs * KB_PER_TOKEN * 1024)
    y = df["L_A"].values
    logy = np.log(np.maximum(y, 1e-3))

    n = len(df)
    cb.PASSES = max(12, EVENT_BUDGET // n)
    print(f"[{tag}] 块 {n}，会话 {df['sid'].nunique()}，"
          f"寿命 中位 {np.median(y):.0f}s / p10 {np.percentile(y,10):.0f}s / "
          f"p90 {np.percentile(y,90):.0f}s")
    print(f"        放大倍数 passes={cb.PASSES}（事件总量≈{n*cb.PASSES//1000}k）")

    c0 = build(df, np.zeros(n, dtype=int), size)
    live = max_live_bytes(c0)
    cap = live / UTIL
    n_ru = int(cap / (RU_GIB * (1 << 30)))
    K_rule = int(np.clip(n_ru / 7, 3, 16))
    print(f"        活跃 {live/(1<<30):.0f}GiB → 盘 {cap/(1<<30):.0f}GiB，"
          f"RU {n_ru}，规则 K={K_rule}")

    # ---------- 3. 方法对比 ----------
    w0 = Device(c0, capacity_gb=cap / (1 << 30), ru_mb=RU_GIB * 1024,
                op_frac=0.07).run()["wa_steady"]
    print(f"\n{'方法':<16}{'K':>3}{'η²':>8}{'IBRD50':>9}{'WA':>8}")
    print(f"{'不分层':<16}{1:>3}{0.0:>8.3f}{summarize(np.zeros(n,int),y)['ibrd50']*100:>8.1f}%{w0:>8.3f}")

    best = (None, 1e9, None)
    for K in [k for k in (4, 8, 12, 16, 20, 24, 32) if k <= max(2, n_ru // 3)]:
        try:
            lab = cut(oof_rank(df, FEATURES_V2, K), K)
            w = Device(build(df, lab, size), capacity_gb=cap / (1 << 30),
                       ru_mb=RU_GIB * 1024, op_frac=0.07).run()["wa_steady"]
        except RuntimeError:
            print(f"{'A3 排序':<16}{K:>3}   失败（RU 不足）")
            continue
        m = summarize(lab, y)
        print(f"{'A3 排序':<16}{K:>3}{eta2_log(lab,y):>8.3f}{m['ibrd50']*100:>8.1f}%{w:>8.3f}")
        if w < best[1]:
            best = (K, w, lab)

    # oracle
    if best[0]:
        K = best[0]
        orc = pd.factorize(np.searchsorted(
            np.quantile(logy, np.linspace(0, 1, K + 1)[1:-1]), logy, side="right"))[0]
        try:
            wo = Device(build(df, orc, size), capacity_gb=cap / (1 << 30),
                        ru_mb=RU_GIB * 1024, op_frac=0.07).run()["wa_steady"]
        except RuntimeError:
            wo = float("nan")
        print(f"{'oracle':<16}{K:>3}{eta2_log(orc,y):>8.3f}"
              f"{summarize(orc,y)['ibrd50']*100:>8.1f}%{wo:>8.3f}")

    if best[0]:
        print(f"\n最佳 K={best[0]}，A3 WA={best[1]:.3f}；"
              f"相对不分层降幅 {w0-best[1]:.3f}")


if __name__ == "__main__":
    main()
