"""信息天花板诊断：寿命 = 剩余会话时长 R − 淘汰延迟 D。

L_A = (session_end − t_create) − (t_death − t_create)
    = R_remain − D_evict

两部分分别建模：
  1. 只用现有特征，分别预测 R 和 D，看哪一部分可预测性差；
  2. 把**真实**的 R 或 D 作为额外特征喂进去（作弊上界），看 IBRD/WA 能改善多少
     —— 这直接回答"缺的是哪一半信息"。
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
from features_v2 import FEATURES_V2, add_features_v2  # noqa: E402
from fusion_v3 import cut, eta2_log, oof_rank, wa_of  # noqa: E402
from gc_sim import Device, max_live_bytes  # noqa: E402
from metrics import summarize  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
KB_PER_TOKEN = 32
RU_BYTES = 6 * (1 << 30)
GIB = 1 << 30


def load_df(trace, budget=200_000):
    reqs, blocks = load(trace)
    bs = int(blocks[0].get("block_size", 528))
    from mm_input import MMInput
    rows, _ = attach(reqs, blocks)
    rows = add_kind(rows, reqs, MMInput(os.path.join(ROOT, "models", "Qwen3.5-9B")),
                    bs, max(len(r["prompt_token_ids"]) for r in reqs))
    df = pd.DataFrame(rows)
    se, n_turns = {}, {}
    for r in reqs:
        se[r["sid"]] = max(se.get(r["sid"], 0.0), r["t_end"])
        n_turns[r["sid"]] = max(n_turns.get(r["sid"], 0), r["turn"] + 1)
    df = df[df["t_death"].notna()].copy()
    df["L_A"] = df["sid"].map(se) - df["t_death"]
    df = df[df["L_A"] > 0].reset_index(drop=True)
    df = add_features_v2(df, reqs)

    # 两个分解量
    df["R_remain"] = df["sid"].map(se) - df["t_create"]      # 剩余会话时长
    df["D_evict"] = df["t_death"] - df["t_create"]           # 淘汰延迟
    df["remaining_turns"] = df["sid"].map(n_turns) - df["turn"] - 1
    cb.PASSES = max(12, budget // len(df))
    size = int(bs * KB_PER_TOKEN * 1024)
    return df, size


def reg_oof(df, feats, target):
    """在 log 空间做 OOF 回归，返回 (R²_log, Spearman)。"""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import GroupKFold
    from scipy.stats import spearmanr
    X = design_matrix(df, feats)
    t = np.log(np.maximum(df[target].values, 1e-3))
    pred = np.zeros(len(df))
    g = df["sid"].values
    for tr, te in GroupKFold(4).split(X, cut(t, 8), groups=g):
        m = HistGradientBoostingRegressor(max_depth=5, learning_rate=0.08,
                                          max_iter=300, random_state=0)
        m.fit(X[tr], t[tr])
        pred[te] = m.predict(X[te])
    ss_res = ((t - pred) ** 2).sum()
    ss_tot = ((t - t.mean()) ** 2).sum()
    return 1 - ss_res / ss_tot, spearmanr(pred, t).statistic


def main():
    for t, tag in [("data/trace_docvqa36", "docvqa36"),
                   ("data/trace_big", "trace_big")]:
        p = os.path.join(ROOT, t)
        if not os.path.isdir(p):
            continue
        df, size = load_df(p)
        y = df["L_A"].values
        logy = np.log(np.maximum(y, 1e-3))
        n = len(df)
        c0 = build(df, np.zeros(n, dtype=int), size)
        live = max_live_bytes(c0)
        cap = live / 0.74
        n_ru = int(cap / RU_BYTES)
        K = int(np.clip(n_ru / 7, 3, 16))

        print(f"\n{'='*66}\n[{tag}] 块 {n}，RU {n_ru}，K={K}")
        # 分解自洽性
        recon = df["R_remain"] - df["D_evict"]
        print(f"  分解自洽性: max|L_A − (R−D)| = {np.abs(recon - y).max():.2e}")
        print(f"  R_remain: 中位 {np.median(df.R_remain):.0f}s   "
              f"D_evict: 中位 {np.median(df.D_evict):.0f}s   L_A: 中位 {np.median(y):.0f}s")

        # 1) 两部分各自的可预测性
        print("\n  【可预测性】只用现有特征（OOF，log 空间）")
        for target in ("L_A", "R_remain", "D_evict"):
            r2, rho = reg_oof(df, FEATURES_V2, target)
            print(f"    {target:<10} R²={r2:+.3f}   Spearman={rho:+.3f}")

        # 2) 作弊上界：把真实的 R 或 D 作为额外特征
        print("\n  【信息天花板】把真实分量当特征喂进去（A3 排序，同一 K）")
        base = cut(oof_rank(df, FEATURES_V2, K), K)
        cases = {
            "现有特征（基线）": base,
            "+ 真实剩余会话时长 R": None,
            "+ 真实淘汰延迟 D": None,
            "+ 真实剩余轮数": None,
        }
        # 构造带 oracle 列的 df
        df2 = df.copy()
        df2["oracle_R"] = np.log(np.maximum(df["R_remain"].values, 1e-3))
        df2["oracle_D"] = np.log(np.maximum(df["D_evict"].values, 1e-3))
        df2["oracle_turns"] = df["remaining_turns"].values
        cases["+ 真实剩余会话时长 R"] = cut(
            oof_rank(df2, FEATURES_V2 + ["oracle_R"], K), K)
        cases["+ 真实淘汰延迟 D"] = cut(
            oof_rank(df2, FEATURES_V2 + ["oracle_D"], K), K)
        cases["+ 真实剩余轮数"] = cut(
            oof_rank(df2, FEATURES_V2 + ["oracle_turns"], K), K)

        # oracle 参照
        orc = pd.factorize(np.searchsorted(
            np.quantile(logy, np.linspace(0, 1, K + 1)[1:-1]), logy, side="right"))[0]
        cases["oracle 完美分桶"] = orc

        print(f"    {'方案':<22}{'η²':>8}{'IBRD50':>9}{'WA':>8}")
        for name, lab in cases.items():
            m = summarize(lab, y)
            try:
                w = wa_of(df, lab, size, cap / GIB)
            except RuntimeError:
                w = float("nan")
            print(f"    {name:<22}{eta2_log(lab,y):>8.3f}{m['ibrd50']*100:>8.1f}%{w:>8.3f}")


if __name__ == "__main__":
    main()
