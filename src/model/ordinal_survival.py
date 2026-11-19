"""A1/A4：序数回归（离散时间生存模型），对数寿命目标。

方法对照（同一工作点、同一份 OOF 划分）：
  mc        多分类（当前做法）：直接分 K 个分位桶
  reg_log   A4：回归 log 寿命，再按预测值切分位桶
  ord       A1：K-1 个二分类器 P(log L > t_j)，得分 = Σ_j P_j，再切分位桶
            （这就是离散时间生存模型：每个阈值是一个时间区间，
              P(log L > t_j) 即"存活过该区间"的概率）

每个方法都跑两套特征：旧特征 FEATURES、新特征 FEATURES_V2，做消融。

用法：python src/ordinal_survival.py data/trace_docvqa36 [--K 16] [--rho 0.41]
"""

from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _setup import add_src_paths  # noqa: E402
add_src_paths()

import argparse
import os
import sys

import numpy as np
import pandas as pd
from cluster_lifetime import add_kind, attach, design_matrix, load  # noqa: E402
from classify_buckets import build  # noqa: E402
from features_v2 import FEATURES_V2, add_features_v2  # noqa: E402
from gc_sim import Device  # noqa: E402
from metrics import summarize  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
KB_PER_TOKEN = 32
RU_MB = 6144.0
OP = 0.07
PASSES = 120


def eta2_log(labels, y):
    x = np.log(np.maximum(np.asarray(y, float), 1e-3))
    labels = np.asarray(labels)
    mu = x.mean()
    ss_tot = ((x - mu) ** 2).sum()
    ss_w = 0.0
    for b in np.unique(labels):
        m = labels == b
        ss_w += ((x[m] - x[m].mean()) ** 2).sum()
    return 1.0 - ss_w / ss_tot if ss_tot > 0 else 0.0


def wa_of(df, lab, size, cap_gb):
    try:
        return Device(build(df, np.asarray(lab), size), capacity_gb=cap_gb,
                      ru_mb=RU_MB, op_frac=OP).run()["wa_steady"]
    except RuntimeError:
        return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace", nargs="?", default="data/trace_docvqa36")
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--rho", type=float, default=0.41)
    a = ap.parse_args()
    td = a.trace if os.path.isabs(a.trace) else os.path.join(ROOT, a.trace)
    tag = os.path.basename(td)
    K = a.K

    reqs, blocks = load(td)
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
    df = add_features_v2(df, reqs)
    size = int(bs * KB_PER_TOKEN * 1024)
    y = df["L_A"].values
    logy = np.log(np.maximum(y, 1e-3))
    g = df["sid"].values

    c0 = build(df, np.zeros(len(df), dtype=int), size)
    W = sum(c.size for c in c0) / (max(c.t_death for c in c0) - min(c.t_write for c in c0))
    cap = float(np.median(y)) * W / a.rho / (1 << 30)
    print(f"[{tag}] 块 {len(df)}，L50={np.median(y):.0f}s，ρ={a.rho} → 盘 {cap:.0f}GB，K={K}")

    # 真值分位桶（在 log 上等价，分位数对单调变换不变）
    edges = np.quantile(logy, np.linspace(0, 1, K + 1)[1:-1])
    cls = np.searchsorted(edges, logy, side="right")

    from sklearn.ensemble import (HistGradientBoostingClassifier,
                                  HistGradientBoostingRegressor)
    from sklearn.model_selection import GroupKFold

    def oof(feats, mode):
        X = design_matrix(df, feats)
        pred = np.zeros(len(df), dtype=float if mode != "mc" else int)
        for tr, te in GroupKFold(n_splits=4).split(X, cls, groups=g):
            if mode == "mc":
                m = HistGradientBoostingClassifier(max_depth=5, learning_rate=0.08,
                                                   max_iter=300, random_state=0)
                m.fit(X[tr], cls[tr])
                pred[te] = m.predict(X[te])
            elif mode == "reg_log":
                m = HistGradientBoostingRegressor(max_depth=5, learning_rate=0.08,
                                                  max_iter=300, random_state=0)
                m.fit(X[tr], logy[tr])
                pred[te] = m.predict(X[te])
            else:  # ord：K-1 个二分类器
                score = np.zeros(len(te))
                for j, t in enumerate(edges):
                    lab = (logy > t).astype(int)
                    m = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.10,
                                                       max_iter=200, random_state=0)
                    m.fit(X[tr], lab[tr])
                    score += m.predict_proba(X[te])[:, 1]
                pred[te] = score
        # 把连续得分/回归值切成 K 个等频桶
        if mode != "mc":
            pred = np.searchsorted(np.quantile(pred, np.linspace(0, 1, K + 1)[1:-1]),
                                   pred, side="right")
        return pd.factorize(pred)[0]

    out = []
    for fname, feats in (("旧特征", None), ("新特征", FEATURES_V2)):
        f = feats if feats is not None else (
            ["turn", "block_idx", "pos_frac", "prompt_len", "n_images",
             "cached_blocks", "session_age", "idle_before", "prev_gen",
             "n_active", "idle_ratio", "turn_frac", "blocks_in_req",
             "prompt_growth", "pos_from_img"])
        for mode, label in (("mc", "多分类(基线)"), ("reg_log", "A4 回归log"),
                            ("ord", "A1 序数/生存")):
            lab = oof(f, mode)
            m = summarize(lab, y)
            row = dict(features=fname, method=label, K=len(np.unique(lab)),
                       eta2=eta2_log(lab, y), ibrd50=m["ibrd50"],
                       ibrd90=m["ibrd90"], pass5=m["pass5"],
                       wa=wa_of(df, lab, size, cap),
                       acc=float((lab == pd.factorize(cls)[0]).mean()))
            out.append(row)
            print(f"  {fname} {label:<12} η²={row['eta2']:.3f}  "
                  f"IBRD50={row['ibrd50']*100:5.1f}%  acc={row['acc']*100:5.1f}%  "
                  f"WA={row['wa']:.3f}")

    # 对照
    for name, lab in (("不分层", np.zeros(len(df), dtype=int)),
                      ("oracle", pd.factorize(cls)[0])):
        m = summarize(lab, y)
        out.append(dict(features="—", method=name, K=len(np.unique(lab)),
                        eta2=eta2_log(lab, y), ibrd50=m["ibrd50"],
                        ibrd90=m["ibrd90"], pass5=m["pass5"],
                        wa=wa_of(df, lab, size, cap),
                        acc=float((lab == pd.factorize(cls)[0]).mean())))
        print(f"  {name:<12} η²={out[-1]['eta2']:.3f}  WA={out[-1]['wa']:.3f}")

    res = pd.DataFrame(out)
    path = os.path.join(ROOT, "data", f"ordinal_{tag}.csv")
    res.to_csv(path, index=False)
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()
