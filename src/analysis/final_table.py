"""统一主表：所有分组方法在同一工作点（固定 ρ）下的 M3 与 WA。

列：M3 中位偏差 / p90 偏差（题面口径 M1 也一并给出）、稳态 WA。

用法：python src/final_table.py data/trace_docvqa36 [--rho 0.41]
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
from cluster_lifetime import (  # noqa: E402
    FEATURES, add_kind, attach, design_matrix, load,
)
from classify_buckets import add_occupancy, build  # noqa: E402
from gc_sim import Device  # noqa: E402
from metrics import summarize  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
KB_PER_TOKEN = 32
RU_MB = 6144.0
OP = 0.07
PASSES = 120
EVENT_BUDGET = 200_000   # 与 analyze_big 一致：块数 × passes ≈ 常数


def wa_of(df, lab, size, cap_gb):
    ch = build(df, np.asarray(lab), size)
    try:
        return Device(ch, capacity_gb=cap_gb, ru_mb=RU_MB, op_frac=OP).run()["wa_steady"]
    except RuntimeError:
        return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace", nargs="?", default="data/trace_docvqa36")
    ap.add_argument("--rho", type=float, default=0.41)
    ap.add_argument("--util", type=float, default=None, help="固定盘利用率（与 analyze_big 一致）")
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
    import classify_buckets as _cb
    _cb.PASSES = max(12, EVENT_BUDGET // len(df))
    df = add_occupancy(df, size)
    y = df["L_A"].values

    c0 = build(df, np.zeros(len(df), dtype=int), size)
    W = sum(c.size for c in c0) / (max(c.t_death for c in c0) - min(c.t_write for c in c0))
    from gc_sim import max_live_bytes as _mlb
    if a.util is not None:
        cap_gb = _mlb(c0) / a.util / (1 << 30)
    else:
        cap_gb = float(np.median(y)) * W / a.rho / (1 << 30)
    print(f"[{tag}] 块 {len(df)}，L50={np.median(y):.0f}s，ρ={a.rho} → 盘 {cap_gb:.0f}GB，"
          f"RU 数 {int(cap_gb*1024/RU_MB)}")

    from sklearn.cluster import KMeans
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_predict, GroupKFold

    X_content = design_matrix(df, FEATURES)
    g = df["sid"].values
    methods = {}
    methods["s0 不分层"] = np.zeros(len(df), dtype=int)
    methods["s1 命中三层"] = df["cached_blocks"].map(
        lambda v: 2 if v >= 3 else (1 if v >= 1 else 0)).values
    for k in (4, 8, 16):
        methods[f"km{k}"] = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(X_content)
        methods[f"oracle{k}"] = np.searchsorted(
            np.quantile(y, np.linspace(0, 1, k + 1)[1:-1]), y, side="right")
    # RF16：直接分类到 16 个真值桶（只用内容/会话特征）
    cls16 = methods["oracle16"]
    methods["RF16"] = cross_val_predict(
        RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=0,
                               n_jobs=-1),
        X_content, cls16, cv=GroupKFold(4), groups=g)

    out = []
    for name, lab in methods.items():
        lab = pd.factorize(np.asarray(lab))[0]
        m = summarize(lab, y)
        wa = wa_of(df, lab, size, cap_gb)
        out.append(dict(method=name, streams=int(lab.max()) + 1,
                        ibrd50=m["ibrd50"], ibrd90=m["ibrd90"],
                        pass5=m["pass5"], wa=wa))
        print(f"  {name:<12} streams={out[-1]['streams']:3d}  "
              f"IBRD50={m['ibrd50']*100:5.1f}%  IBRD90={m['ibrd90']*100:5.1f}%  "
              f"M1(±5%)={m['pass5']*100:5.1f}%  WA={wa:.3f}")

    res = pd.DataFrame(out)
    path = os.path.join(ROOT, "data", f"final_table_{tag}.csv")
    res.to_csv(path, index=False)
    print(f"\n-> {path}")

    # markdown
    print("\n| 方法 | streams | IBRD50 | IBRD90 | 原口径(±5%) | WA |")
    print("|---|---|---|---|---|---|")
    for _, r in res.iterrows():
        print(f"| {r['method']} | {r['streams']} | {r['ibrd50']*100:.1f}% | "
              f"{r['ibrd90']*100:.1f}% | {r['pass5']*100:.1f}% | {r['wa']:.3f} |")


if __name__ == "__main__":
    main()
