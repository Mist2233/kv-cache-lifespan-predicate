"""实验 A2：预测总轮次需要多准才有收益？（噪声注入敏感性分析）"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _setup import add_src_paths  # noqa: E402
add_src_paths()
import sys, os, numpy as np, pandas as pd
import classify_buckets as cb
from cluster_lifetime import load, attach
from classify_buckets import build
from features_v2 import FEATURES_V2, add_features_v2
from fusion_v3 import cut, eta2_log
from gc_sim import Device, max_live_bytes
from metrics import summarize
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
import scipy.stats as ss

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GIB = 1 << 30

def main():
    td = sys.argv[1] if len(sys.argv) > 1 else "data/trace_openomni_v3"
    reqs, blocks = load(td if os.path.isabs(td) else os.path.join(ROOT, td))
    bs = int(blocks[0].get("block_size", 512))
    rows, _ = attach(reqs, blocks)
    df = pd.DataFrame(rows)
    se = {}
    for r in reqs: se[r["sid"]] = max(se.get(r["sid"], 0), r["t_end"])
    df = df[df.t_death.notna()].copy()
    df["L_A"] = df.sid.map(se) - df.t_death
    df = df[df.L_A > 0].reset_index(drop=True)
    df = add_features_v2(df, reqs)
    total = (df.groupby("sid")["turn"].max() + 1).astype(float)
    df["turns_true"] = df.sid.map(total)
    base = [f for f in FEATURES_V2]
    y = np.log(np.maximum(df.L_A.values, 1e-3)); g = df.sid.values
    K = 12
    size = int(bs * 32 * 1024)

    def run(feats, tag):
        X = df[feats].astype(float).replace([np.inf, -np.inf], 0).fillna(0)
        X = ((X - X.mean()) / X.std().replace(0, 1)).values
        score = np.zeros(len(X)); rng = np.random.default_rng(0)
        for tr, te in GroupKFold(4).split(X, y, groups=g):
            i = rng.integers(0, len(tr), 30000); j = rng.integers(0, len(tr), 30000)
            m = y[tr][i] != y[tr][j]; i, j = i[m], j[m]
            lm = LogisticRegression(max_iter=2000).fit(X[tr][i]-X[tr][j], (y[tr][i]>y[tr][j]).astype(int))
            score[te] = X[te] @ lm.coef_[0]
        lab = cut(score, K); m = summarize(lab, df.L_A.values)
        for p in (1,): cb.PASSES = 32
        ch = build(df, lab, size); live = max_live_bytes(ch); cap = live/0.74/GIB
        w = Device(ch, capacity_gb=cap, ru_mb=6144, op_frac=0.07).run()["wa_steady"]
        print(f"  {tag:<28} η²={eta2_log(lab, df.L_A.values):.3f} IBRD50={m['ibrd50']*100:5.1f}% WA={w:.3f}")
        return w

    print(f"块 {len(df)} 会话 {df.sid.nunique()}  K={K}")
    run(base, "基线")
    # 完美知道总轮次
    df["turns_pred"] = df["turns_true"]
    run(base + ["turns_pred"], "完美总轮次 (ρ=1.0)")
    # 注入噪声，构造不同 Spearman
    rng = np.random.default_rng(0)
    for target in (0.3, 0.5, 0.7, 0.9):
        lo, hi = 0.0, 3.0
        for _ in range(40):
            sig = (lo + hi) / 2
            p = df["turns_true"].values * np.exp(rng.normal(0, sig, len(df)))
            rho = ss.spearmanr(p, df["turns_true"].values)[0]
            if rho > target: lo = sig
            else: hi = sig
        df["turns_pred"] = p
        run(base + ["turns_pred"], f"总轮次 Spearman≈{target:.1f}")

if __name__ == "__main__":
    main()
