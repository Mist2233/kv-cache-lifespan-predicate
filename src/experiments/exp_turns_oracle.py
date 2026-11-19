"""实验 A：如果知道"会话总轮次"，预测能好多少？（oracle 上界分析）

在 trace_openomni_v3 上：
  1. 用现有特征训练 pairwise 排序（基线）；
  2. 加入"真实会话总轮次"（oracle，写块时未知）以及 turns_left = 总轮次 - 当前轮次；
  3. 对比 η² / IBRD₅₀ / WA。
如果 oracle 能大幅提升，说明"预测会话总轮次"是最值得投入的方向。
"""

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
from sklearn.ensemble import GradientBoostingRegressor
import scipy.stats as ss

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GIB = 1 << 30
PASSES = 32

def prep(td):
    reqs, blocks = load(td)
    bs = int(blocks[0].get("block_size", 512))
    rows, _ = attach(reqs, blocks)
    df = pd.DataFrame(rows)
    se = {}
    for r in reqs:
        se[r["sid"]] = max(se.get(r["sid"], 0), r["t_end"])
    df = df[df.t_death.notna()].copy()
    df["L_A"] = df.sid.map(se) - df.t_death
    df = df[df.L_A > 0].reset_index(drop=True)
    df = add_features_v2(df, reqs)
    # oracle：会话总轮次
    total = df.groupby("sid")["turn"].max() + 1
    df["turns_total_oracle"] = df.sid.map(total).astype(float)
    df["turns_left_oracle"] = (df.turns_total_oracle - df.turn).clip(lower=0)
    return df, bs

def kind_onehot(df):
    for k in ("image", "audio", "text"):
        df[f"kind_{k}"] = (df.get("kind") == k).astype(float) if "kind" in df else 0.0
    return df

def run_pairwise(X, y, g, K):
    score = np.zeros(len(X)); rng = np.random.default_rng(0)
    for tr, te in GroupKFold(4).split(X, y, groups=g):
        i = rng.integers(0, len(tr), 30000); j = rng.integers(0, len(tr), 30000)
        m = y[tr][i] != y[tr][j]; i, j = i[m], j[m]
        lm = LogisticRegression(max_iter=2000).fit(X[tr][i] - X[tr][j],
                                                   (y[tr][i] > y[tr][j]).astype(int))
        score[te] = X[te] @ lm.coef_[0]
    return cut(score, K)

def wa(df, lab, bs, passes):
    cb.PASSES = passes
    size = int(bs * 32 * 1024)
    ch = build(df, lab, size)
    live = max_live_bytes(ch); cap = live / 0.74 / GIB
    return Device(ch, capacity_gb=cap, ru_mb=6144, op_frac=0.07).run()["wa_steady"], cap

def main():
    td = sys.argv[1] if len(sys.argv) > 1 else "data/trace_openomni_v3"
    df, bs = prep(td if os.path.isabs(td) else os.path.join(ROOT, td))
    df = kind_onehot(df)
    base = [f for f in FEATURES_V2] + ["kind_image", "kind_audio", "kind_text"]
    extra = ["turns_total_oracle", "turns_left_oracle"]
    y = np.log(np.maximum(df.L_A.values, 1e-3)); g = df.sid.values
    K = 12
    print(f"[{os.path.basename(td)}] 块 {len(df)} 会话 {df.sid.nunique()}")

    def eval_case(name, feats):
        X = df[feats].astype(float).replace([np.inf, -np.inf], 0).fillna(0)
        X = ((X - X.mean()) / X.std().replace(0, 1)).values
        lab = run_pairwise(X, y, g, K)
        m = summarize(lab, df.L_A.values)
        w, cap = wa(df, lab, bs, PASSES)
        print(f"  {name:<26} η²={eta2_log(lab, df.L_A.values):.3f} "
              f"IBRD50={m['ibrd50']*100:5.1f}% IBRD90={m['ibrd90']*100:5.1f}% WA={w:.3f} (RU {int(cap/(6))})")
        return eta2_log(lab, df.L_A.values), m["ibrd50"], w

    eval_case("基线（现有特征）", base)
    eval_case("+ 真实总轮次（oracle）", base + extra)

    # 总轮次本身可预测吗？用会话第 1 轮的观测量预测总轮次
    first = df[df.turn == 0].groupby("sid").first()
    if len(first) > 50:
        feats = [f for f in ["prompt_len", "gen_len", "n_images", "prompt_growth"] if f in first]
        X = first[feats].astype(float).fillna(0).values
        t = first["turns_total_oracle"].values
        pred = np.zeros(len(X))
        for tr, te in GroupKFold(4).split(X, t, groups=first.index.values):
            mdl = GradientBoostingRegressor(random_state=0).fit(X[tr], np.log(t[tr]))
            pred[te] = np.exp(mdl.predict(X[te]))
        rho = ss.spearmanr(pred, t)[0]
        print(f"\n  用第 1 轮特征预测总轮次：Spearman={rho:.2f}，"
              f"MAE={np.mean(np.abs(pred-t)):.1f} 轮（真实中位 {np.median(t):.0f}）")

if __name__ == "__main__":
    main()
