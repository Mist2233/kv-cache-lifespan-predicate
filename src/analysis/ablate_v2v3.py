"""特征集 v2 vs v3 × 方法 × 数据集 的严格消融（固定 K=8，保证三个数据集都可行）。

v2 特征 = 旧特征 + 7 个新特征（其中 age_ratio/turn_ratio/prompt_len_ratio 无效）
v3 特征 = 旧特征 + 4 个有效特征 + 8 个交互/非线性特征

方法：A4（log 回归）、A3（pairwise 排序）、融合 w=0.5
数据集：docvqa36 / trace_v1 / docvqa12

用法：python src/ablate_v2v3.py
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
from cluster_lifetime import add_kind, load, attach  # noqa: E402
from classify_buckets import build  # noqa: E402
from features_v2 import FEATURES_V2, FEATURES_V3, add_features_v2, add_features_v3  # noqa: E402
from fusion_v3 import cut, eta2_log, oof_rank, oof_reg, wa_of, z  # noqa: E402
from metrics import summarize  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
KB_PER_TOKEN = 32
K_FIX = 8


def prep(trace, tag):
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
    df = add_features_v3(df, reqs)      # v3 包含 v2 的所有列
    size = int(bs * KB_PER_TOKEN * 1024)
    y = df["L_A"].values
    c0 = build(df, np.zeros(len(df), dtype=int), size)
    W = sum(c.size for c in c0) / (max(c.t_death for c in c0) - min(c.t_write for c in c0))
    cap = float(np.median(y)) * W / 0.41 / (1 << 30)
    n_ru = int(cap * 1024 / 6144)
    K = min(K_FIX, max(4, n_ru // 4))
    return df, size, cap, n_ru, K, tag


def main():
    traces = [("data/trace_docvqa36", "docvqa36"),
              ("data/trace_v1", "trace_v1"),
              ("data/trace_docvqa", "docvqa12")]
    out = []
    for t, tag in traces:
        p = os.path.join(ROOT, t)
        if not os.path.isdir(p):
            continue
        df, size, cap, n_ru, K, tag = prep(p, tag)
        y = df["L_A"].values
        print(f"\n[{tag}] 块 {len(df)}，RU {n_ru}，K={K}")
        for fname, feats in (("v2", FEATURES_V2), ("v3", FEATURES_V3)):
            s4 = oof_reg(df, feats, K)
            s3 = oof_rank(df, feats, K)
            cands = {"A4": s4, "A3": s3, "融合0.5": 0.5 * z(s3) + 0.5 * z(s4)}
            for mname, sc in cands.items():
                lab = cut(sc, K)
                m = summarize(lab, y)
                wa = wa_of(df, lab, size, cap)
                out.append(dict(trace=tag, feats=fname, method=mname, K=K,
                                eta2=eta2_log(lab, y), ibrd50=m["ibrd50"], wa=wa))
                print(f"  {fname} {mname:<8} η²={out[-1]['eta2']:.3f}  "
                      f"IBRD50={m['ibrd50']*100:5.1f}%  WA={wa:.3f}")
    d = pd.DataFrame(out)
    d.to_csv(os.path.join(ROOT, "data", "ablate_v2v3.csv"), index=False)
    print("\n" + "=" * 72)
    print(d.pivot_table(index=["feats", "method"], columns="trace", values="wa").round(3).to_string())


if __name__ == "__main__":
    main()
