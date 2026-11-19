"""验证："η²（桶间分离度）只能粗筛，IBRD（桶内偏差）才能细比" 是否成立。

实验设计
--------
两组配置，故意让 η² 的覆盖范围不同：

  【粗筛组】质量谱：12 组合成分组（按概率 q 使用真值分位桶，否则随机），
            K ∈ {4, 8, 16}。η² 覆盖 0 → 0.99。
            → 用来检验"跨度很大时 η² 能不能排序 WA"

  【细比组】真实方法族：A4 回归 log（5 个模型种子）、A3 pairwise（5 个 pair 采样种子）、
            A1 序数（2 个种子）、以及 A4 的 7 个特征消融。
            全部落在同一性能带（η²≈0.35–0.50）。
            → 用来检验"方法之间细比时，谁更准"

指标
----
  η²      = 1 − SS_within(log L)/SS_total(log L)      （桶间分离度）
  IBRD₅₀  = median |L−μ_b|/μ_b                        （桶内相对偏差）
  WDM     = Σ_b (n_b/N)·(Q3_b−Q1_b)/T_dev             （桶内死亡时刻跨度 / 盘重写周期）
            —— 新提出的候选：绝对时间跨度，而非相对偏差

分析
----
  * 各组内 η²、IBRD、WDM 与 WA 的 Spearman 相关（bootstrap 95% CI）
  * 全样本偏相关 / 双变量回归：WA ~ a·η² + b·IBRD
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
from cluster_lifetime import (  # noqa: E402
    FEATURES, add_kind, attach, design_matrix, load,
)
from classify_buckets import build  # noqa: E402
from features_v2 import FEATURES_V2, NEW_FEATURES, add_features_v2  # noqa: E402
from gc_sim import Device  # noqa: E402
from metrics import deviations  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
KB_PER_TOKEN = 32
RU_MB = 6144.0
OP = 0.07
K = 16


# ------------------------------------------------------------------ 指标
def eta2_log(lab, y):
    x = np.log(np.maximum(np.asarray(y, float), 1e-3))
    lab = np.asarray(lab)
    mu = x.mean()
    ss_tot = ((x - mu) ** 2).sum()
    ss_w = sum(((x[lab == b] - x[lab == b].mean()) ** 2).sum()
               for b in np.unique(lab))
    return 1.0 - ss_w / ss_tot


def ibrd50(lab, y):
    return float(np.median(deviations(np.asarray(lab), y)))


def wdm(lab, y, t_dev):
    """桶内死亡时刻跨度 (IQR) 按桶大小加权，再除以 T_dev。"""
    lab = np.asarray(lab)
    y = np.asarray(y, float)
    tot = 0.0
    for b in np.unique(lab):
        v = y[lab == b]
        if len(v) < 2:
            continue
        q1, q3 = np.percentile(v, [25, 75])
        tot += len(v) / len(y) * (q3 - q1)
    return tot / t_dev


def wa_of(df, lab, size, cap, seed=0):
    try:
        return Device(build(df, np.asarray(lab), size, seed=seed), capacity_gb=cap,
                      ru_mb=RU_MB, op_frac=OP).run()["wa_steady"]
    except RuntimeError:
        return float("nan")


def cut(scores, k=K):
    e = np.quantile(scores, np.linspace(0, 1, k + 1)[1:-1])
    return pd.factorize(np.searchsorted(e, scores, side="right"))[0]


# ------------------------------------------------------------------ 模型
def oof_reg(df, feats, seed=0):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import GroupKFold
    X = design_matrix(df, feats)
    y = np.log(np.maximum(df["L_A"].values, 1e-3))
    pred, g = np.zeros(len(df)), df["sid"].values
    for tr, te in GroupKFold(4).split(X, cut(y), groups=g):
        m = HistGradientBoostingRegressor(max_depth=5, learning_rate=0.08,
                                          max_iter=300, random_state=seed)
        m.fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    return pred


def oof_rank(df, feats, seed=0, n_pairs=30000):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    X = design_matrix(df, feats)
    y = np.log(np.maximum(df["L_A"].values, 1e-3))
    g = df["sid"].values
    score = np.zeros(len(df))
    rng = np.random.default_rng(seed)
    for tr, te in GroupKFold(4).split(X, y, groups=g):
        i = rng.integers(0, len(tr), n_pairs)
        j = rng.integers(0, len(tr), n_pairs)
        m = y[tr][i] != y[tr][j]
        i, j = i[m], j[m]
        lm = LogisticRegression(max_iter=2000).fit(X[tr][i] - X[tr][j],
                                                   (y[tr][i] > y[tr][j]).astype(int))
        score[te] = X[te] @ lm.coef_[0]
    return score


def oof_ord(df, feats, seed=0):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import GroupKFold
    X = design_matrix(df, feats)
    logy = np.log(np.maximum(df["L_A"].values, 1e-3))
    edges = np.quantile(logy, np.linspace(0, 1, K + 1)[1:-1])
    g = df["sid"].values
    score = np.zeros(len(df))
    for tr, te in GroupKFold(4).split(X, cut(logy), groups=g):
        s = np.zeros(len(te))
        for t in edges:
            lab = (logy > t).astype(int)
            m = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.10,
                                               max_iter=200, random_state=seed)
            m.fit(X[tr], lab[tr])
            s += m.predict_proba(X[te])[:, 1]
        score[te] = s
    return score


def bootstrap_spearman(x, y, n=2000, seed=0):
    from scipy.stats import spearmanr
    rng = np.random.default_rng(seed)
    x, y = np.asarray(x), np.asarray(y)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(x), len(x))
        if len(np.unique(x[idx])) < 3:
            continue
        vals.append(spearmanr(x[idx], y[idx]).statistic)
    vals = np.array(vals)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main():
    trace = sys.argv[1] if len(sys.argv) > 1 else "data/trace_docvqa36"
    td = trace if os.path.isabs(trace) else os.path.join(ROOT, trace)
    tag = os.path.basename(td)

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

    c0 = build(df, np.zeros(len(df), dtype=int), size)
    W = sum(c.size for c in c0) / (max(c.t_death for c in c0) - min(c.t_write for c in c0))
    cap = float(np.median(y)) * W / 0.41 / (1 << 30)
    T_dev = cap * 1e9 / W
    print(f"[{tag}] 块 {len(df)}，盘 {cap:.0f}GB，T_dev={T_dev:.0f}s")

    recs = []

    def add(group, name, lab):
        lab = pd.factorize(np.asarray(lab))[0]
        rec = dict(group=group, name=name, K=len(np.unique(lab)),
                   eta2=eta2_log(lab, y), ibrd50=ibrd50(lab, y),
                   wdm=wdm(lab, y, T_dev), wa=wa_of(df, lab, size, cap))
        recs.append(rec)
        print(f"  [{group}] {name:<28} η²={rec['eta2']:.3f}  IBRD={rec['ibrd50']*100:5.1f}%  "
              f"WDM={rec['wdm']:.3f}  WA={rec['wa']:.3f}")

    # ---------- 粗筛组：质量谱 ----------
    rng = np.random.default_rng(0)
    print("\n== 粗筛组（质量谱，η² 覆盖 0~1）==")
    for k in (4, 8, 16):
        truth = pd.factorize(np.searchsorted(
            np.quantile(logy, np.linspace(0, 1, k + 1)[1:-1]), logy, side="right"))[0]
        for q in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
            lab = truth.copy()
            flip = rng.random(len(lab)) > q
            lab[flip] = rng.integers(0, k, flip.sum())
            add("粗筛", f"K={k}, q={q:.1f}", lab)

    # ---------- 细比组：真实方法族 ----------
    print("\n== 细比组（真实方法，全部落在 η² 0.35~0.50）==")
    for s in range(5):
        add("细比", f"A4 回归log seed={s}", cut(oof_reg(df, FEATURES_V2, seed=s)))
    for s in range(5):
        add("细比", f"A3 排序 seed={s}", cut(oof_rank(df, FEATURES_V2, seed=s)))
    for s in range(2):
        add("细比", f"A1 序数 seed={s}", cut(oof_ord(df, FEATURES_V2, seed=s)))
    for f in NEW_FEATURES:
        sub = [x for x in FEATURES_V2 if x != f]
        add("细比", f"A4 去掉 {f}", cut(oof_reg(df, sub)))

    d = pd.DataFrame(recs).dropna()
    d.to_csv(os.path.join(ROOT, "data", f"metric_validation_{tag}.csv"), index=False)

    from scipy.stats import spearmanr
    print("\n" + "=" * 78)
    print("相关性分析".center(74))
    print("=" * 78)
    for g in ("粗筛", "细比"):
        sub = d[d.group == g]
        print(f"\n【{g}组】n={len(sub)}，η² 范围 {sub.eta2.min():.3f}~{sub.eta2.max():.3f}")
        for col in ("eta2", "ibrd50", "wdm"):
            r = spearmanr(sub[col], sub.wa).statistic
            lo, hi = bootstrap_spearman(sub[col], sub.wa)
            print(f"   {col:<8} Spearman={r:+.3f}   95%CI=[{lo:+.2f}, {hi:+.2f}]")

    print(f"\n【全部】n={len(d)}")
    for col in ("eta2", "ibrd50", "wdm"):
        r = spearmanr(d[col], d.wa).statistic
        lo, hi = bootstrap_spearman(d[col], d.wa)
        print(f"   {col:<8} Spearman={r:+.3f}   95%CI=[{lo:+.2f}, {hi:+.2f}]")

    # 双变量回归（全部点）
    from numpy.linalg import lstsq
    A = np.column_stack([np.ones(len(d)), d.eta2, d.ibrd50])
    coef, *_ = lstsq(A, d.wa.values, rcond=None)
    pred = A @ coef
    r2 = 1 - ((d.wa - pred) ** 2).sum() / ((d.wa - d.wa.mean()) ** 2).sum()
    print(f"\n回归 WA ~ a·η² + b·IBRD + c: a={coef[1]:+.3f}, b={coef[2]:+.3f}, R²={r2:.3f}")


if __name__ == "__main__":
    main()
