"""验证：桶间分离度 η² 与写放大 WA 到底有多相关。

做法：造一批"质量可控"的分组（按概率 q 把块分到真值分位桶，否则随机），
q 从 0 扫到 1，得到 η² 从 ~0 到 ~1 的一串分组；再加上真实方法
（不分层、命中三层、KMeans、RF、会话年龄规则、oracle）。

对每个分组同时算：
  * η²（桶间分离度）
  * IBRD50 / IBRD90（桶内相对偏差）
  * 原口径 ±5% 达标率
  * 分类准确率（对真值分位桶）
  * 稳态 WA（仿真）

然后算各指标与 WA 的 Spearman 相关，并把 WA 对 η² 做拟合。

用法：python src/validate_eta2.py data/trace_docvqa36 --rho 0.41
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
import matplotlib.pyplot as plt  # noqa: E402
import plotstyle  # noqa: E402
from cluster_lifetime import (  # noqa: E402
    FEATURES, add_kind, attach, design_matrix, load,
)
from classify_buckets import build  # noqa: E402
from gc_sim import Device  # noqa: E402
from metrics import summarize  # noqa: E402

plotstyle.setup()
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
KB_PER_TOKEN = 32
RU_MB = 6144.0
OP = 0.07
PASSES = 120
K = 16


def eta2(labels, values):
    """桶间方差 / 总方差。"""
    values = np.asarray(values, float)
    labels = np.asarray(labels)
    mu = values.mean()
    ss_tot = ((values - mu) ** 2).sum()
    ss_within = 0.0
    for b in np.unique(labels):
        m = labels == b
        ss_within += ((values[m] - values[m].mean()) ** 2).sum()
    return 1.0 - ss_within / ss_tot if ss_tot > 0 else 0.0


def wa_of(df, lab, size, cap_gb):
    try:
        return Device(build(df, np.asarray(lab), size), capacity_gb=cap_gb,
                      ru_mb=RU_MB, op_frac=OP).run()["wa_steady"]
    except RuntimeError:
        return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace", nargs="?", default="data/trace_docvqa36")
    ap.add_argument("--rho", type=float, default=0.41)
    a = ap.parse_args()
    td = a.trace if os.path.isabs(a.trace) else os.path.join(ROOT, a.trace)
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
    size = int(bs * KB_PER_TOKEN * 1024)
    y = df["L_A"].values

    c0 = build(df, np.zeros(len(df), dtype=int), size)
    W = sum(c.size for c in c0) / (max(c.t_death for c in c0) - min(c.t_write for c in c0))
    cap = float(np.median(y)) * W / a.rho / (1 << 30)
    print(f"[{tag}] ρ={a.rho}, 盘 {cap:.0f}GB, K={K}")

    true_bucket = np.searchsorted(np.quantile(y, np.linspace(0, 1, K + 1)[1:-1]),
                                  y, side="right")
    rng = np.random.default_rng(0)

    groups = []
    # 1) 质量可控的合成分组：以概率 q 用真值桶，否则随机
    for q in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        lab = true_bucket.copy()
        flip = rng.random(len(lab)) > q
        lab[flip] = rng.integers(0, K, flip.sum())
        groups.append((f"合成 q={q:.1f}", lab))
    # 2) 真实方法
    from sklearn.cluster import KMeans
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_predict, GroupKFold
    X = design_matrix(df, FEATURES)
    groups.append(("不分层", np.zeros(len(df), dtype=int)))
    groups.append(("命中三层", df["cached_blocks"].map(
        lambda v: 2 if v >= 3 else (1 if v >= 1 else 0)).values))
    groups.append(("KMeans16", KMeans(n_clusters=K, n_init=10, random_state=0)
                   .fit_predict(X)))
    groups.append(("RF16", cross_val_predict(
        RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                               random_state=0, n_jobs=-1),
        X, true_bucket, cv=GroupKFold(4), groups=df["sid"].values)))
    groups.append(("会话年龄规则", (pd.qcut(df["session_age"], 8, labels=False,
                                     duplicates="drop").astype(int) * 2
                              + (df["turn"] > 2).astype(int))))
    groups.append(("oracle16", true_bucket))

    out = []
    for name, lab in groups:
        lab = pd.factorize(np.asarray(lab))[0]
        m = summarize(lab, y)
        acc = float((np.asarray(lab) == pd.factorize(true_bucket)[0]).mean())
        wa = wa_of(df, lab, size, cap)
        out.append(dict(method=name, eta2=eta2(lab, y), ibrd50=m["ibrd50"],
                        ibrd90=m["ibrd90"], pass5=m["pass5"], acc=acc,
                        wa=wa, k=len(np.unique(lab))))
        print(f"  {name:<14} η²={out[-1]['eta2']:.3f}  IBRD50={m['ibrd50']*100:5.1f}%  "
              f"acc={acc*100:5.1f}%  WA={wa:.3f}")

    res = pd.DataFrame(out).dropna()
    from scipy.stats import spearmanr, pearsonr
    print("\n=== 各指标与 WA 的相关性 ===")
    for col in ("eta2", "ibrd50", "ibrd90", "pass5", "acc"):
        rho_s = spearmanr(res[col], res["wa"]).statistic
        r_p = pearsonr(res[col], res["wa"])[0]
        print(f"  {col:<8} Spearman={rho_s:+.3f}  Pearson={r_p:+.3f}")

    # 拟合 WA ~ f(η²)
    z = 1.0 / (1.0 - res["eta2"].clip(upper=0.999))
    coef = np.polyfit(res["eta2"], z, 1)
    pred = coef[0] * res["eta2"] + coef[1]
    ss = 1 - ((z - pred) ** 2).sum() / ((z - z.mean()) ** 2).sum()
    print(f"\n拟合 1/(1-η²) = a·η² + b: a={coef[0]:.2f}, b={coef[1]:.2f}, R²={ss:.3f}")

    res.to_csv(os.path.join(ROOT, "data", f"validate_eta2_{tag}.csv"), index=False)
    plot(res, tag, a.rho)


def plot(res, tag, rho):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    ax = axes[0]
    synth = res[res["method"].str.startswith("合成")]
    real = res[~res["method"].str.startswith("合成")]
    ax.scatter(synth["eta2"], synth["wa"], s=45, color="#8C8C8C",
               label="合成分组（扫质量）")
    ax.scatter(real["eta2"], real["wa"], s=80, color="#4C72B0", label="真实方法",
               zorder=3)
    for _, r in real.iterrows():
        ax.annotate(r["method"], (r["eta2"], r["wa"]), fontsize=7,
                    xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("桶间分离度 η²")
    ax.set_ylabel("稳态 WA")
    ax.set_title("(a) η² 与 WA 的关系", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.25)

    ax2 = axes[1]
    for col, c, lb in (("eta2", "#4C72B0", "η²"), ("ibrd50", "#DD8452", "IBRD50"),
                       ("pass5", "#55A868", "±5% 达标率"),
                       ("acc", "#8172B3", "分类准确率")):
        from scipy.stats import spearmanr
        rho_s = spearmanr(res[col], res["wa"]).statistic
        ax2.scatter(res[col], res["wa"], s=25, color=c, alpha=0.75,
                    label=f"{lb} (ρ={rho_s:+.2f})")
    ax2.set_xlabel("指标值")
    ax2.set_ylabel("稳态 WA")
    ax2.set_title("(b) 哪个指标最能预测 WA（Spearman ρ）", fontsize=10)
    ax2.legend(fontsize=7, frameon=False)
    ax2.grid(alpha=0.25)

    fig.suptitle(f"η² 与 WA 的相关性验证（{tag}，ρ={rho}）", fontsize=12)
    out = os.path.join(ROOT, "figs", f"fig16_validate_eta2_{tag}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[fig] {out}")


if __name__ == "__main__":
    main()
