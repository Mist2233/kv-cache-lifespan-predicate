"""固定利用率下的 K 扫描（修正版）。

之前不同数据集/子集之间 WA 差异很大，根因不是 K，而是**设备利用率没有被控制住**：
ρ = L/T_dev 的校准只固定了"寿命/盘重写周期"的比值，没有固定 live_bytes/capacity。
实测子采样时利用率在 0.656~0.887 之间漂移，WA 从 1.07 变到 1.89。

本脚本改为：capacity = live_bytes / 0.74（固定利用率），再扫 K。

用法：python src/k_sweep_util.py
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
from ablate_v2v3 import prep  # noqa: E402
from classify_buckets import build  # noqa: E402
from features_v2 import FEATURES_V2  # noqa: E402
from fusion_v3 import cut, eta2_log, oof_rank, wa_of  # noqa: E402
from gc_sim import max_live_bytes  # noqa: E402
from metrics import summarize  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RU_FRAC = 6.144e9
UTIL = 0.74


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("trace", nargs="?", default=None,
                    help="只分析指定 trace；留空则跑默认三个数据集")
    a = ap.parse_args()
    if a.trace:
        traces = [(a.trace, os.path.basename(a.trace.rstrip("/")))]
    else:
        traces = [("data/trace_docvqa36", "docvqa36"),
                  ("data/trace_v1", "trace_v1"),
                  ("data/trace_docvqa", "docvqa12")]
    out = []
    for t, tag in traces:
        p = os.path.join(ROOT, t)
        if not os.path.isdir(p):
            continue
        df, size, _, _, _, _ = prep(p, tag)
        y = df["L_A"].values
        c0 = build(df, np.zeros(len(df), dtype=int), size)
        live = max_live_bytes(c0)
        cap = live / UTIL
        n_ru = int(cap / RU_FRAC)
        k_max = max(2, n_ru // 3)
        ks = [k for k in (2, 3, 4, 6, 8, 10, 12, 16, 20) if k <= k_max]
        print(f"\n[{tag}] 块 {len(df)}，活跃 {live/1e9:.0f}GB，固定利用率 {UTIL} "
              f"→ 盘 {cap/1e9:.0f}GB，RU {n_ru}，K 上限 {k_max}")
        print(f"  {'K':>3}{'η²':>8}{'IBRD50':>9}{'WA':>8}")
        for K in ks:
            sc = oof_rank(df, FEATURES_V2, K)
            lab = cut(sc, K)
            m = summarize(lab, y)
            wa = wa_of(df, lab, size, cap / (1 << 30))
            out.append(dict(trace=tag, K=K, n_ru=n_ru, eta2=eta2_log(lab, y),
                            ibrd50=m["ibrd50"], wa=wa))
            print(f"  {K:>3}{out[-1]['eta2']:>8.3f}{m['ibrd50']*100:>8.1f}%{wa:>8.3f}")
    d = pd.DataFrame(out)
    d.to_csv(os.path.join(ROOT, "data", "k_sweep_util.csv"), index=False)
    if len(d):
        print("\n" + "=" * 60)
        print(d.pivot_table(index="K", columns="trace", values="wa").round(3).to_string())
        print("\n各数据集最佳 K：")
        for tag, sub in d.groupby("trace"):
            if sub.wa.notna().any():
                b = sub.loc[sub.wa.idxmin()]
                print(f"  {tag:<10} K={int(b.K):2d}  WA={b.wa:.3f}")


if __name__ == "__main__":
    main()
