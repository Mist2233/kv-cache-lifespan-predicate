"""A3（排序学习）+ 新特征消融。

A3：只学"谁比谁活得久"。构造样本对 (i, j)，标签 1[L_i > L_j]，
在特征差 (x_i − x_j) 上训练二分类器；推理时用 w·x 作为寿命得分，
再按得分分位切成 K 个桶。

同时做 A4（log 回归）的**逐特征消融**：每次去掉一个新特征，看 η²/WA 变化。

用法：python src/followup_a3.py data/trace_docvqa36
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
from cluster_lifetime import add_kind, attach, design_matrix, load  # noqa: E402
from classify_buckets import build  # noqa: E402
from features_v2 import FEATURES_V2, NEW_FEATURES, add_features_v2  # noqa: E402
from gc_sim import Device  # noqa: E402
from metrics import summarize  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
KB_PER_TOKEN = 32
RU_MB = 6144.0
OP = 0.07
K = 16


def eta2_log(labels, y):
    x = np.log(np.maximum(np.asarray(y, float), 1e-3))
    labels = np.asarray(labels)
    mu = x.mean()
    ss_tot = ((x - mu) ** 2).sum()
    ss_w = sum(((x[labels == b] - x[labels == b].mean()) ** 2).sum()
               for b in np.unique(labels))
    return 1.0 - ss_w / ss_tot


def wa_of(df, lab, size, cap):
    try:
        return Device(build(df, np.asarray(lab), size), capacity_gb=cap,
                      ru_mb=RU_MB, op_frac=OP).run()["wa_steady"]
    except RuntimeError:
        return float("nan")


def load_all(trace):
    reqs, blocks = load(trace)
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
    return add_features_v2(df, reqs), bs


def cut(labels_scores, K=K):
    edges = np.quantile(labels_scores, np.linspace(0, 1, K + 1)[1:-1])
    return pd.factorize(np.searchsorted(edges, labels_scores, side="right"))[0]


def oof_rank(df, feats, seed=0, n_pairs=30000):
    """A3：pairwise 排序，返回每个块的得分。"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    X = design_matrix(df, feats)
    y = np.log(np.maximum(df["L_A"].values, 1e-3))
    g = df["sid"].values
    score = np.zeros(len(df))
    rng = np.random.default_rng(seed)
    for tr, te in GroupKFold(n_splits=4).split(X, y, groups=g):
        i = rng.integers(0, len(tr), n_pairs)
        j = rng.integers(0, len(tr), n_pairs)
        mask = y[tr][i] != y[tr][j]
        i, j = i[mask], j[mask]
        d = X[tr][i] - X[tr][j]
        lab = (y[tr][i] > y[tr][j]).astype(int)
        m = LogisticRegression(max_iter=2000, C=1.0).fit(d, lab)
        score[te] = X[te] @ m.coef_[0]
    return score


def oof_reg(df, feats):
    """A4：log 寿命回归，返回预测值。"""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import GroupKFold
    X = design_matrix(df, feats)
    y = np.log(np.maximum(df["L_A"].values, 1e-3))
    pred = np.zeros(len(df))
    g = df["sid"].values
    cls = cut(y)
    for tr, te in GroupKFold(n_splits=4).split(X, cls, groups=g):
        m = HistGradientBoostingRegressor(max_depth=5, learning_rate=0.08,
                                          max_iter=300, random_state=0)
        m.fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    return pred


def main():
    trace = sys.argv[1] if len(sys.argv) > 1 else "data/trace_docvqa36"
    td = trace if os.path.isabs(trace) else os.path.join(ROOT, trace)
    tag = os.path.basename(td)
    df, bs = load_all(td)
    size = int(bs * KB_PER_TOKEN * 1024)
    y = df["L_A"].values
    c0 = build(df, np.zeros(len(df), dtype=int), size)
    W = sum(c.size for c in c0) / (max(c.t_death for c in c0) - min(c.t_write for c in c0))
    cap = float(np.median(y)) * W / 0.41 / (1 << 30)
    logy = np.log(np.maximum(y, 1e-3))
    truth = pd.factorize(cut(logy))[0]
    print(f"[{tag}] 块 {len(df)}，盘 {cap:.0f}GB，K={K}")

    out = []

    def report(name, lab, note=""):
        m = summarize(lab, y)
        row = dict(method=name, note=note, eta2=eta2_log(lab, y),
                   ibrd50=m["ibrd50"], ibrd90=m["ibrd90"], pass5=m["pass5"],
                   wa=wa_of(df, lab, size, cap),
                   acc=float((np.asarray(lab) == truth).mean()))
        out.append(row)
        print(f"  {name:<26} η²={row['eta2']:.3f}  IBRD50={row['ibrd50']*100:5.1f}%  "
              f"WA={row['wa']:.3f}  {note}")

    # --- A4 基准 ---
    report("A4 回归log（全特征）", cut(oof_reg(df, FEATURES_V2)))

    # --- A3 排序 ---
    report("A3 pairwise 排序", cut(oof_rank(df, FEATURES_V2)))

    # --- 新特征逐项消融（A4）---
    print("\n逐特征消融（A4 回归log，去掉某个特征后）：")
    for f in NEW_FEATURES:
        sub = [x for x in FEATURES_V2 if x != f]
        report(f"A4 去掉 {f}", cut(oof_reg(df, sub)), "-1 特征")

    res = pd.DataFrame(out)
    res.to_csv(os.path.join(ROOT, "data", f"followup_{tag}.csv"), index=False)
    print(f"\n-> data/followup_{tag}.csv")


if __name__ == "__main__":
    main()
