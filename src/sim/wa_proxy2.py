"""修正版 WA 代理 + 学习型代理。

修正点
------
上一版用 Δ_b = RU_bytes / W_b 当作驻留时间，严重高估 δ。
稳态下每个 stream 的 RU 周转时间其实是**一个盘周期**：

    T_res ≈ U · T_dev        （U=利用率，T_dev=C/W）

所以单个块在 GC 时的存活概率 ≈ min(L, T_res)/T_res，δ 再按桶加权。

学习型代理
----------
把 WA 当作 (η², IBRD, RU/stream, U, ρ, δ̂…) 的函数，用随机森林拟合，
并用"留一数据集"验证（用 docvqa36 训练、trace_big 测试，反之亦然）。
这是检验"能否用便宜的特征替代仿真"的直接办法。
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
from wa_proxy import prep  # noqa: E402  （复用数据准备）
from features_v2 import FEATURES_V2  # noqa: E402
from fusion_v3 import cut, eta2_log, oof_rank  # noqa: E402
from gc_sim import Device  # noqa: E402
from metrics import summarize  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RU_BYTES = 6 * (1 << 30)
UTIL = 0.74
GIB = 1 << 30


def delta_res(labels, lifetimes, t_res):
    """按 T_res 估计 GC 存活比例 δ。"""
    labels = np.asarray(labels)
    L = np.asarray(lifetimes, float)
    n = len(L)
    delta = 0.0
    for b in np.unique(labels):
        m = labels == b
        w = m.sum() / n
        live = np.minimum(L[m], t_res) / t_res
        delta += w * live.mean()
    return float(np.clip(delta, 0.0, 0.999))


def main():
    rows = []
    for t, tag in [("data/trace_docvqa36", "docvqa36"), ("data/trace_big", "trace_big")]:
        p = os.path.join(ROOT, t)
        if not os.path.isdir(p):
            continue
        df, size, cap, W, n = prep(p, tag)
        y = df["L_A"].values
        logy = np.log(np.maximum(y, 1e-3))
        n_ru = int(cap / RU_BYTES)
        t_dev = cap / W
        t_res = UTIL * t_dev
        rho = float(np.median(y)) / t_dev
        print(f"\n[{tag}] 块 {n}，RU {n_ru}，T_dev={t_dev:.0f}s，T_res={t_res:.0f}s，ρ={rho:.2f}")

        def rec(name, lab):
            lab = pd.factorize(np.asarray(lab))[0]
            K = len(np.unique(lab))
            m = summarize(lab, y)
            d = delta_res(lab, y, t_res)
            try:
                wa = Device(build_lab(df, lab, size), capacity_gb=cap / GIB,
                            ru_mb=6 * 1024, op_frac=0.07).run()["wa_steady"]
            except RuntimeError:
                return
            rows.append(dict(trace=tag, method=name, K=K, ru_per_stream=n_ru / K,
                             eta2=eta2_log(lab, y), ibrd50=m["ibrd50"],
                             delta_res=d, wa_hat=1 / (1 - d), rho=rho, U=UTIL,
                             t_dev=t_dev, n_ru=n_ru, wa=wa))
            print(f"  {name:<14} K={K:>2} RU/stream={n_ru/K:5.1f}  η²={rows[-1]['eta2']:.3f}  "
                  f"IBRD={m['ibrd50']*100:5.1f}%  δ̂={d:.3f}  ŴA={1/(1-d):6.2f}  WA={wa:.3f}")

        from classify_buckets import build as build_lab
        rec("不分层", np.zeros(n, dtype=int))
        for K in (2, 4, 6, 8, 12, 16, 20, 24, 32):
            if K > n_ru // 3:
                continue
            rec(f"A3 K={K}", cut(oof_rank(df, FEATURES_V2, K), K))
        Kb = 8 if n > 5000 else 16
        orc = pd.factorize(np.searchsorted(
            np.quantile(logy, np.linspace(0, 1, Kb + 1)[1:-1]), logy, side="right"))[0]
        rec(f"oracle K={Kb}", orc)

    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(ROOT, "data", "wa_proxy2.csv"), index=False)
    from scipy.stats import spearmanr
    print("\n" + "=" * 70)
    for tag, sub in d.groupby("trace"):
        print(f"[{tag}] n={len(sub)}")
        for c in ("eta2", "ibrd50", "delta_res", "wa_hat"):
            print(f"   {c:<10} {spearmanr(sub[c], sub.wa).statistic:+.3f}")
    print(f"[全部] n={len(d)}")
    for c in ("eta2", "ibrd50", "delta_res", "wa_hat"):
        print(f"   {c:<10} {spearmanr(d[c], d.wa).statistic:+.3f}")

    # ---------- 学习型代理：留一数据集验证 ----------
    from sklearn.ensemble import RandomForestRegressor
    feats = ["eta2", "ibrd50", "delta_res", "ru_per_stream", "K", "rho", "t_dev"]
    print("\n" + "=" * 70)
    print("学习型代理（留一数据集验证）")
    for train_tag, test_tag in [("docvqa36", "trace_big"), ("trace_big", "docvqa36")]:
        tr = d[d.trace == train_tag]
        te = d[d.trace == test_tag]
        if len(tr) < 5 or len(te) < 3:
            continue
        m = RandomForestRegressor(n_estimators=300, min_samples_leaf=1,
                                  random_state=0, n_jobs=-1).fit(tr[feats], tr.wa)
        pred = m.predict(te[feats])
        err = np.abs(pred - te.wa)
        print(f"  训练 {train_tag} → 测试 {test_tag}: "
              f"MAE={err.mean():.3f}，最大误差={err.max():.3f}")
        for i, (_, r) in enumerate(te.iterrows()):
            print(f"     {r.method:<12} 实际 {r.wa:.3f}  预测 {pred[i]:.3f}")


if __name__ == "__main__":
    main()
