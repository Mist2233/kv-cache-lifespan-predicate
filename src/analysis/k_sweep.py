"""K 扫描：桶（stream）数取多少最划算。

对每个数据集，让 K 在可行范围内变化，用同一个方法（v2 特征 + A3 排序）
重新训练并仿真，记录 η²、IBRD、WA。

可行上界由盘上 RU 数决定：每个 stream 至少要 3 个 RU（1 个 open + 迁移余量）。

用法：python src/k_sweep.py
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
from features_v2 import FEATURES_V2  # noqa: E402
from fusion_v3 import cut, eta2_log, oof_rank, wa_of  # noqa: E402
from metrics import summarize  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    traces = [("data/trace_docvqa36", "docvqa36"),
              ("data/trace_v1", "trace_v1"),
              ("data/trace_docvqa", "docvqa12")]
    out = []
    for t, tag in traces:
        p = os.path.join(ROOT, t)
        if not os.path.isdir(p):
            continue
        df, size, cap, n_ru, _, _ = prep(p, tag)
        y = df["L_A"].values
        k_max = max(2, n_ru // 3)
        ks = [k for k in (2, 3, 4, 6, 8, 10, 12, 16, 20, 24) if k <= k_max]
        print(f"\n[{tag}] 块 {len(df)}，盘 {cap:.0f}GB，RU {n_ru}，K 上限 {k_max}")
        print(f"  {'K':>3}{'η²':>8}{'IBRD50':>9}{'WA':>8}")
        for K in ks:
            score = oof_rank(df, FEATURES_V2, K)
            lab = cut(score, K)
            m = summarize(lab, y)
            wa = wa_of(df, lab, size, cap)
            out.append(dict(trace=tag, K=K, n_ru=n_ru, eta2=eta2_log(lab, y),
                            ibrd50=m["ibrd50"], wa=wa))
            print(f"  {K:>3}{out[-1]['eta2']:>8.3f}{m['ibrd50']*100:>8.1f}%{wa:>8.3f}")
    d = pd.DataFrame(out)
    d.to_csv(os.path.join(ROOT, "data", "k_sweep.csv"), index=False)
    if len(d):
        print("\n" + "=" * 60)
        print("WA 汇总")
        print(d.pivot_table(index="K", columns="trace", values="wa").round(3).to_string())
        print("\n各数据集的最佳 K：")
        for tag, sub in d.groupby("trace"):
            best = sub.loc[sub.wa.idxmin()]
            print(f"  {tag:<10} 最佳 K={int(best.K):2d}  WA={best.wa:.3f}")


if __name__ == "__main__":
    main()
