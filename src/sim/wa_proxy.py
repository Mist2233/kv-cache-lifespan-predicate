"""WA 的解析代理指标：把"设备约束"纳入指标，而不只是看分组紧不紧。

动机
----
实验发现：K 从 4 增到 32 时，η² 和 IBRD 都"变好"，WA 却变差。
原因是它们只衡量"桶内寿命是否接近"，**看不到每个 stream 分到多少 RU**。

本脚本提出并检验一个新指标 **δ̂（预测的 GC 存活比例）**：

  对第 b 个桶：
    w_b  = n_b / N                      桶的写入份额
    W_b  = W · w_b                      该 stream 的写入速率
    Δ_b  = RU_bytes / W_b               写满一个 RU 所需时间
    live_i = min(L_i, Δ_b) / Δ_b        该块在 RU 被回收时的存活概率
    δ_b  = mean_i live_i

  δ̂ = Σ_b w_b · δ_b
  ŴA = 1 / (1 − δ̂)

直觉：
  * Δ_b ≪ L  → RU 很快写满，数据还活着 → 大量搬运 → WA 高
  * Δ_b ≫ L  → RU 写满时数据早死了   → 几乎不用搬 → WA 低

这个式子天然包含两件事：
  1. 桶内寿命分布（越好 → δ_b 越小）
  2. 每个 stream 的写入份额（K 越大 → 份额越小 → Δ_b 越大 → δ_b 越大）

用法：python src/wa_proxy.py
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
from cluster_lifetime import add_kind, attach, load  # noqa: E402
from classify_buckets import build  # noqa: E402
from features_v2 import FEATURES_V2, add_features_v2  # noqa: E402
from fusion_v3 import cut, eta2_log, oof_rank, wa_of  # noqa: E402
from gc_sim import Device, max_live_bytes  # noqa: E402
from metrics import summarize  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
KB_PER_TOKEN = 32
RU_BYTES = 6 * (1 << 30)
UTIL = 0.74
GIB = 1 << 30


def delta_hat(labels, lifetimes, w_rate, ru_bytes=RU_BYTES):
    """预测 GC 时的存活字节比例 δ̂。"""
    labels = np.asarray(labels)
    L = np.asarray(lifetimes, float)
    n = len(L)
    delta = 0.0
    for b in np.unique(labels):
        m = labels == b
        w_b = m.sum() / n
        if w_b <= 0:
            continue
        w_b_rate = w_rate * w_b               # 该 stream 的写入速率 (B/s)
        if w_b_rate <= 0:
            continue
        delta_b = ru_bytes / w_b_rate          # 写满一个 RU 的时间
        live = np.minimum(L[m], delta_b) / delta_b
        delta += w_b * live.mean()
    return float(np.clip(delta, 0.0, 0.999))


def prep(trace, tag, event_budget=200_000):
    import classify_buckets as cb
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
    size = int(bs * KB_PER_TOKEN * 1024)
    cb.PASSES = max(12, event_budget // len(df))
    c0 = build(df, np.zeros(len(df), dtype=int), size)
    live = max_live_bytes(c0)
    cap = live / UTIL
    W = sum(c.size for c in c0) / (max(c.t_death for c in c0) - min(c.t_write for c in c0))
    return df, size, cap, W, len(df)


def main():
    traces = [("data/trace_docvqa36", "docvqa36"), ("data/trace_big", "trace_big")]
    rows = []
    for t, tag in traces:
        p = os.path.join(ROOT, t)
        if not os.path.isdir(p):
            continue
        df, size, cap, W, n = prep(p, tag)
        y = df["L_A"].values
        logy = np.log(np.maximum(y, 1e-3))
        n_ru = int(cap / RU_BYTES)
        print(f"\n[{tag}] 块 {n}，RU {n_ru}，W={W/1e9:.2f} GB/s，"
              f"T_dev={cap/W:.0f}s，中位寿命 {np.median(y):.0f}s")

        def rec(name, lab):
            lab = pd.factorize(np.asarray(lab))[0]
            m = summarize(lab, y)
            d = delta_hat(lab, y, W)
            try:
                wa = Device(build(df, lab, size), capacity_gb=cap / GIB,
                            ru_mb=6 * 1024, op_frac=0.07).run()["wa_steady"]
            except RuntimeError:
                wa = float("nan")
            rows.append(dict(trace=tag, method=name, K=len(np.unique(lab)),
                             eta2=eta2_log(lab, y), ibrd50=m["ibrd50"],
                             delta_hat=d, wa_hat=1 / (1 - d), wa=wa))
            print(f"  {name:<14} K={len(np.unique(lab)):>2}  η²={rows[-1]['eta2']:.3f}  "
                  f"IBRD={m['ibrd50']*100:5.1f}%  δ̂={d:.3f}  ŴA={1/(1-d):.2f}  WA={wa:.3f}")

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
    d.to_csv(os.path.join(ROOT, "data", "wa_proxy.csv"), index=False)
    from scipy.stats import spearmanr
    print("\n" + "=" * 70)
    print("各指标与仿真 WA 的相关性（Spearman）")
    for tag, sub in d.groupby("trace"):
        print(f"\n[{tag}] n={len(sub)}")
        for c in ("eta2", "ibrd50", "delta_hat", "wa_hat"):
            print(f"   {c:<10} {spearmanr(sub[c], sub.wa).statistic:+.3f}")
    print(f"\n[全部] n={len(d)}")
    for c in ("eta2", "ibrd50", "delta_hat", "wa_hat"):
        print(f"   {c:<10} {spearmanr(d[c], d.wa).statistic:+.3f}")


if __name__ == "__main__":
    main()
