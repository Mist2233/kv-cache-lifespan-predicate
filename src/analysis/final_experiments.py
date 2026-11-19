"""收尾实验：
1) trace_big 上的方法对比（s0 / A3 / A4 / RF 多分类，K=4）；
2) 匹配 ρ 的跨数据集比较：把 docvqa36 的盘放大到 ρ=0.29（与 trace_big 同 ρ），
   再看两个数据集在**同一世代比**下的降幅。
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
from cluster_lifetime import add_kind, attach, design_matrix, load  # noqa: E402
from classify_buckets import build  # noqa: E402
from features_v2 import FEATURES, FEATURES_V2, add_features_v2  # noqa: E402
from fusion_v3 import cut, eta2_log, oof_rank, oof_reg, wa_of  # noqa: E402
from gc_sim import Device, max_live_bytes  # noqa: E402
from metrics import summarize  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
KB_PER_TOKEN = 32
RU_BYTES = 6 * (1 << 30)
GIB = 1 << 30


def load_df(trace, tag, budget=200_000):
    reqs, blocks = load(trace)
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
    cb.PASSES = max(12, budget // len(df))
    size = int(bs * KB_PER_TOKEN * 1024)
    return df, size


def main():
    print("=" * 74)
    print("实验 1：trace_big 上的方法对比（K=4）")
    df, size = load_df(os.path.join(ROOT, "data/trace_big"), "trace_big")
    y = df["L_A"].values
    n = len(df)
    c0 = build(df, np.zeros(n, dtype=int), size)
    live = max_live_bytes(c0)
    cap = live / 0.74
    W = sum(c.size for c in c0) / (max(c.t_death for c in c0) - min(c.t_write for c in c0))
    print(f"块 {n}，盘 {cap/GIB:.0f}GiB，RU {int(cap/RU_BYTES)}，T_dev={cap/W:.0f}s")

    K = 4
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import cross_val_predict, GroupKFold
    logy = np.log(np.maximum(y, 1e-3))
    cls = pd.factorize(np.searchsorted(
        np.quantile(logy, np.linspace(0, 1, K + 1)[1:-1]), logy, side="right"))[0]
    X = design_matrix(df, FEATURES)
    rf = cross_val_predict(
        HistGradientBoostingClassifier(max_depth=5, learning_rate=0.08, max_iter=300,
                                       random_state=0),
        X, cls, cv=GroupKFold(4), groups=df["sid"].values)

    methods = {
        "不分层": np.zeros(n, dtype=int),
        "A3 排序": cut(oof_rank(df, FEATURES_V2, K), K),
        "A4 回归log": cut(oof_reg(df, FEATURES_V2, K), K),
        "多分类(HGB)": pd.factorize(rf)[0],
        "oracle": cls,
    }
    print(f"{'方法':<14}{'η²':>8}{'IBRD50':>9}{'WA':>8}")
    for name, lab in methods.items():
        m = summarize(lab, y)
        try:
            w = wa_of(df, lab, size, cap / GIB)
        except RuntimeError:
            w = float("nan")
        print(f"{name:<14}{eta2_log(lab,y):>8.3f}{m['ibrd50']*100:>8.1f}%{w:>8.3f}")

    print("\n" + "=" * 74)
    print("实验 2：匹配 ρ 的跨数据集比较（都取 ρ≈0.29）")
    for t, tag in [("data/trace_docvqa36", "docvqa36"), ("data/trace_big", "trace_big")]:
        df2, size2 = load_df(os.path.join(ROOT, t), tag)
        y2 = df2["L_A"].values
        n2 = len(df2)
        c02 = build(df2, np.zeros(n2, dtype=int), size2)
        live2 = max_live_bytes(c02)
        W2 = sum(c.size for c in c02) / (max(c.t_death for c in c02) - min(c.t_write for c in c02))
        L2 = float(np.median(y2))
        t_dev = L2 / 0.29
        cap2 = W2 * t_dev
        U2 = live2 / cap2
        n_ru = int(cap2 / RU_BYTES)
        K2 = int(np.clip(n_ru / 7, 3, 16))
        if cap2 <= live2:
            print(f"[{tag}] 不可行：ρ=0.29 需要容量 {cap2/GIB:.0f}GiB < 活跃 {live2/GIB:.0f}GiB")
            continue
        w0 = wa_of(df2, np.zeros(n2, dtype=int), size2, cap2 / GIB)
        best = (None, 1e9)
        for Kk in (4, 8, 12, 16):
            if Kk > n_ru // 3:
                continue
            try:
                lab = cut(oof_rank(df2, FEATURES_V2, Kk), Kk)
                w = wa_of(df2, lab, size2, cap2 / GIB)
            except RuntimeError:
                continue
            if w < best[1]:
                best = (Kk, w)
        print(f"[{tag}] 盘 {cap2/GIB:.0f}GiB  U={U2:.2f}  RU {n_ru}  中位寿命 {L2:.0f}s  "
              f"T_dev={t_dev:.0f}s")
        print(f"   不分层 WA={w0:.3f}  A3 最优 K={best[0]} WA={best[1]:.3f}  "
              f"降幅 {w0-best[1]:.3f}")


if __name__ == "__main__":
    main()
