"""A3 + A4 融合，并在多个数据集上验证。

方法：
  A4     = log 寿命回归（HGB）→ 按预测值分位切桶
  A3     = pairwise 排序（线性）→ 按得分分位切桶
  fusion = z(score_A3)·w + z(score_A4)·(1−w) → 分位切桶

在 docvqa36 / trace_v1（合成）/ trace_docvqa（12 会话）三个数据集上跑同一套流程。

用法：python src/fusion_v3.py
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
from features_v2 import FEATURES_V3, add_features_v3  # noqa: E402
from gc_sim import Device  # noqa: E402
from metrics import summarize  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
KB_PER_TOKEN = 32
RU_MB = 6144.0
OP = 0.07


def eta2_log(lab, y):
    x = np.log(np.maximum(np.asarray(y, float), 1e-3))
    lab = np.asarray(lab)
    mu = x.mean()
    ss_tot = ((x - mu) ** 2).sum()
    ss_w = sum(((x[lab == b] - x[lab == b].mean()) ** 2).sum() for b in np.unique(lab))
    return 1.0 - ss_w / ss_tot


def cut(scores, k):
    e = np.quantile(scores, np.linspace(0, 1, k + 1)[1:-1])
    return pd.factorize(np.searchsorted(e, scores, side="right"))[0]


def wa_of(df, lab, size, cap, seed=0):
    try:
        return Device(build(df, np.asarray(lab), size, seed=seed), capacity_gb=cap,
                      ru_mb=RU_MB, op_frac=OP).run()["wa_steady"]
    except RuntimeError:
        return float("nan")


def oof_reg(df, feats, K, seed=0):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import GroupKFold
    X = design_matrix(df, feats)
    logy = np.log(np.maximum(df["L_A"].values, 1e-3))
    pred, g = np.zeros(len(df)), df["sid"].values
    for tr, te in GroupKFold(4).split(X, cut(logy, K), groups=g):
        m = HistGradientBoostingRegressor(max_depth=5, learning_rate=0.08,
                                          max_iter=300, random_state=seed)
        m.fit(X[tr], logy[tr])
        pred[te] = m.predict(X[te])
    return pred


def oof_rank(df, feats, K, seed=0, n_pairs=30000):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    X = design_matrix(df, feats)
    logy = np.log(np.maximum(df["L_A"].values, 1e-3))
    g = df["sid"].values
    score = np.zeros(len(df))
    rng = np.random.default_rng(seed)
    for tr, te in GroupKFold(4).split(X, cut(logy, K), groups=g):
        i = rng.integers(0, len(tr), n_pairs)
        j = rng.integers(0, len(tr), n_pairs)
        m = logy[tr][i] != logy[tr][j]
        i, j = i[m], j[m]
        lm = LogisticRegression(max_iter=2000).fit(X[tr][i] - X[tr][j],
                                                   (logy[tr][i] > logy[tr][j]).astype(int))
        score[te] = X[te] @ lm.coef_[0]
    return score


def z(a):
    a = np.asarray(a, float)
    s = a.std()
    return (a - a.mean()) / (s if s > 1e-9 else 1.0)


def run_trace(trace, tag):
    reqs, blocks = load(trace)
    if not reqs:
        print(f"[{tag}] 空，跳过")
        return []
    rows, _ = attach(reqs, blocks)
    if not rows:
        print(f"[{tag}] 没有可归属块，跳过")
        return []
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
    if len(df) < 80:
        print(f"[{tag}] 块数太少({len(df)})，跳过")
        return []
    df = add_features_v3(df, reqs)
    size = int(bs * KB_PER_TOKEN * 1024)
    y = df["L_A"].values

    c0 = build(df, np.zeros(len(df), dtype=int), size)
    W = sum(c.size for c in c0) / (max(c.t_death for c in c0) - min(c.t_write for c in c0))
    cap = float(np.median(y)) * W / 0.41 / (1 << 30)
    n_ru = int(cap * 1024 / RU_MB)
    K = int(min(16, max(4, n_ru // 4)))
    print(f"\n[{tag}] 块 {len(df)}，盘 {cap:.0f}GB，RU {n_ru}，K={K}")

    s_a4 = oof_reg(df, FEATURES_V3, K)
    s_a3 = oof_rank(df, FEATURES_V3, K)

    out = []
    cands = {
        "A4 回归log": s_a4,
        "A3 排序": s_a3,
        "融合 w=0.3": 0.3 * z(s_a3) + 0.7 * z(s_a4),
        "融合 w=0.5": 0.5 * z(s_a3) + 0.5 * z(s_a4),
        "融合 w=0.7": 0.7 * z(s_a3) + 0.3 * z(s_a4),
    }
    for name, sc in cands.items():
        lab = cut(sc, K)
        m = summarize(lab, y)
        wa = wa_of(df, lab, size, cap)
        out.append(dict(trace=tag, method=name, K=K, eta2=eta2_log(lab, y),
                        ibrd50=m["ibrd50"], ibrd90=m["ibrd90"], wa=wa))
        print(f"   {name:<12} η²={out[-1]['eta2']:.3f}  IBRD50={m['ibrd50']*100:5.1f}%  WA={wa:.3f}")
    return out


def main():
    traces = [
        ("data/trace_docvqa36", "docvqa36"),
        ("data/trace_v1", "trace_v1"),
        ("data/trace_docvqa", "docvqa12"),
    ]
    allrows = []
    for t, tag in traces:
        p = os.path.join(ROOT, t)
        if os.path.isdir(p):
            allrows += run_trace(p, tag)
    if allrows:
        d = pd.DataFrame(allrows)
        d.to_csv(os.path.join(ROOT, "data", "fusion_crossdata.csv"), index=False)
        print("\n" + "=" * 70)
        print(d.pivot_table(index="method", columns="trace", values="wa").round(3).to_string())


if __name__ == "__main__":
    main()
