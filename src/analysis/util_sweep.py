"""固定放大协议下的利用率扫描（与 analyze_big 完全一致的设备/放大设置）。

用途：给出"盘利用率 -> 写放大"的曲线，用于确定方法真正有效的工作点区间。
与 rho_sweep.py 的区别：本脚本沿用 analyze_big 的 EVENT_BUDGET 放大协议，
因此 74% 那一行与主结果表完全一致。

用法：python src/analysis/util_sweep.py data/trace_docvqa36
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _setup import add_src_paths  # noqa: E402
add_src_paths()

import classify_buckets as cb  # noqa: E402
from classify_buckets import build  # noqa: E402
from cluster_lifetime import attach, load  # noqa: E402
from features_v2 import FEATURES_V2, add_features_v2  # noqa: E402
from fusion_v3 import cut, oof_rank  # noqa: E402
from gc_sim import Device, max_live_bytes  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RU_GIB = 6.0
EVENT_BUDGET = 200_000
UTILS = (0.50, 0.55, 0.60, 0.65, 0.70, 0.74, 0.80, 0.85)


def main():
    trace = sys.argv[1] if len(sys.argv) > 1 else "data/trace_docvqa36"
    td = trace if os.path.isabs(trace) else os.path.join(ROOT, trace)
    reqs, blocks = load(td)
    bs = int(blocks[0].get("block_size", 528))
    rows, _ = attach(reqs, blocks)
    df = pd.DataFrame(rows)
    se = {}
    for r in reqs:
        se[r["sid"]] = max(se.get(r["sid"], 0), r["t_end"])
    df = df[df.t_death.notna()].copy()
    df["L_A"] = df.sid.map(se) - df.t_death
    df = df[df.L_A > 0].reset_index(drop=True)
    df = add_features_v2(df, reqs)
    n = len(df)
    cb.PASSES = max(12, EVENT_BUDGET // n)
    size = int(bs * 32 * 1024)

    y = np.log(np.maximum(df.L_A.values, 1e-3))
    c0 = build(df, np.zeros(n, dtype=int), size)
    live = max_live_bytes(c0)
    print(f"[{os.path.basename(td)}] 块 {n}，passes={cb.PASSES}，活跃 {live/(1<<30):.0f}GiB")
    print(f"\n{'利用率':>6}{'容量GiB':>9}{'RU':>5}{'K':>4}{'不分层':>9}{'我们的方法':>11}{'完美分桶':>9}")
    for u in UTILS:
        cap = live / u / (1 << 30)
        n_ru = int(cap / RU_GIB)
        K = int(np.clip(n_ru / 7, 3, 16))
        try:
            w0 = Device(build(df, np.zeros(n, dtype=int), size),
                        capacity_gb=cap, ru_mb=RU_GIB * 1024, op_frac=0.07).run()["wa_steady"]
        except RuntimeError:
            w0 = float("nan")
        K = min(K, n_ru // 3)
        try:
            lab = cut(oof_rank(df, FEATURES_V2, K), K)
            wa = Device(build(df, lab, size),
                        capacity_gb=cap, ru_mb=RU_GIB * 1024, op_frac=0.07).run()["wa_steady"]
        except RuntimeError:
            wa = float("nan")
        # 完美分桶
        yo = y
        orc = pd.factorize(np.searchsorted(
            np.quantile(yo, np.linspace(0, 1, K + 1)[1:-1]), yo, side="right"))[0]
        try:
            wo = Device(build(df, orc, size),
                        capacity_gb=cap, ru_mb=RU_GIB * 1024, op_frac=0.07).run()["wa_steady"]
        except RuntimeError:
            wo = float("nan")
        print(f"{u*100:>5.0f}%{cap:>9.0f}{n_ru:>5}{K:>4}{w0:>9.3f}{wa:>11.3f}{wo:>9.3f}")


if __name__ == "__main__":
    main()
