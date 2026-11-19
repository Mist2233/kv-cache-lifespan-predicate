"""OpenOmni（图像+语音）trace 的寿命分析与仿真。

与 `analyze_big.py` 的区别：
  * 模态 kind 由记录里的 image_spans / audio_spans 直接给出（含音频模态）；
  * 特征矩阵额外加入 image/audio/text 三个独热（原 design_matrix 只认 system/image/assistant/user）。

用法：python src/analyze_openomni.py data/trace_openomni
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
import classify_buckets as cb  # noqa: E402
from cluster_lifetime import attach, load  # noqa: E402
from classify_buckets import build  # noqa: E402
from features_v2 import FEATURES_V2, add_features_v2  # noqa: E402
from fusion_v3 import cut, eta2_log, oof_rank, wa_of  # noqa: E402
from gc_sim import Device, max_live_bytes  # noqa: E402
from metrics import summarize  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
KB_PER_TOKEN = 32
GIB = 1 << 30
UTIL = 0.74


def assign_kind(df, reqs):
    """按 pos0 落在哪个 span 里，给块标 image / audio / text。"""
    spans = {}
    for r in reqs:
        spans[(r["sid"], r["turn"])] = (
            [(a, b) for a, b, *_ in r.get("image_spans") or []],
            [(a, b) for a, b in r.get("audio_spans") or []],
        )
    kinds = []
    for r in df.itertuples():
        img, aud = spans.get((r.sid, r.turn), ([], []))
        p = int(r.pos0)
        if any(a <= p < b for a, b in img):
            kinds.append("image")
        elif any(a <= p < b for a, b in aud):
            kinds.append("audio")
        else:
            kinds.append("text")
    return np.array(kinds)


def design(df, feats):
    X = df[feats].astype(float).copy()
    for k in ("image", "audio", "text"):
        X[f"kind_{k}"] = (df["kind"] == k).astype(float)
    X = X.replace([np.inf, -np.inf], 0).fillna(0)
    mu, sd = X.mean(), X.std().replace(0, 1)
    return ((X - mu) / sd).values


def main():
    trace = sys.argv[1] if len(sys.argv) > 1 else "data/trace_openomni"
    td = trace if os.path.isabs(trace) else os.path.join(ROOT, trace)
    tag = os.path.basename(td.rstrip("/"))

    reqs, blocks = load(td)
    bs = int(blocks[0].get("block_size", 528))
    rows, _ = attach(reqs, blocks)
    df = pd.DataFrame(rows)
    se = {}
    for r in reqs:
        se[r["sid"]] = max(se.get(r["sid"], 0.0), r["t_end"])
    df = df[df["t_death"].notna()].copy()
    df["L_A"] = df["sid"].map(se) - df["t_death"]
    df = df[df["L_A"] > 0].reset_index(drop=True)
    df["kind"] = assign_kind(df, reqs)
    df = add_features_v2(df, reqs)
    n = len(df)
    import os as _os
    cb.PASSES = int(_os.environ.get('PASSES', max(2, 150_000 // n)))
    size = int(bs * KB_PER_TOKEN * 1024)
    y = df["L_A"].values
    logy = np.log(np.maximum(y, 1e-3))

    c0 = build(df, np.zeros(n, dtype=int), size)
    live = max_live_bytes(c0)
    cap = live / UTIL
    # 小 trace 用 1 GiB 才能有足够 RU；大 trace 用标准 6 GiB（与其他实验一致）
    RU_MB = 1024.0 if live / (1 << 30) < 60 else 6144.0
    n_ru = int(cap / (RU_MB * (1 << 20)))
    K = int(np.clip(n_ru / 7, 3, 16))
    print(f"[{tag}] 块 {n}，会话 {df['sid'].nunique()}，passes={cb.PASSES}")
    print(f"  寿命 中位 {np.median(y):.0f}s / p10 {np.percentile(y,10):.0f}s / "
          f"p90 {np.percentile(y,90):.0f}s")
    print(f"  模态分布: {df['kind'].value_counts().to_dict()}")
    print(f"  活跃 {live/GIB:.0f}GiB → 盘 {cap/GIB:.0f}GiB，RU {n_ru}，K={K}")

    X = design(df, FEATURES_V2)
    from sklearn.model_selection import GroupKFold
    score = np.zeros(n)
    rng = np.random.default_rng(0)
    g = df["sid"].values
    for tr, te in GroupKFold(4).split(X, cut(logy, K), groups=g):
        from sklearn.linear_model import LogisticRegression
        i = rng.integers(0, len(tr), 30000)
        j = rng.integers(0, len(tr), 30000)
        m = logy[tr][i] != logy[tr][j]
        i, j = i[m], j[m]
        lm = LogisticRegression(max_iter=2000).fit(
            X[tr][i] - X[tr][j], (logy[tr][i] > logy[tr][j]).astype(int))
        score[te] = X[te] @ lm.coef_[0]

    lab_a3 = cut(score, K)
    lab_a3_8 = cut(score, 8)
    lab_a3_16 = cut(score, 16)
    orc = pd.factorize(np.searchsorted(
        np.quantile(logy, np.linspace(0, 1, K + 1)[1:-1]), logy, side="right"))[0]

    print(f"\n{'方案':<22}{'η²':>8}{'IBRD50':>9}{'WA':>8}")
    cases = [("不分层", np.zeros(n, dtype=int))]
    for kk in (3, 4, 6, 8, 9, 12, 16):
        if kk <= n_ru // 3:
            cases.append((f"A3 排序 (K={kk})", cut(score, kk)))
    kb = min(16, n_ru // 3)
    cases.append((f"oracle (K={kb})", pd.factorize(np.searchsorted(
        np.quantile(logy, np.linspace(0, 1, kb + 1)[1:-1]), logy, side="right"))[0]))
    for name, lab in cases:
        m = summarize(lab, y)
        try:
            w = wa_of(df, lab, size, cap / GIB) if False else Device(build(df, lab, size), capacity_gb=cap / GIB, ru_mb=RU_MB, op_frac=0.07).run()['wa_steady']
        except RuntimeError:
            w = float("nan")
        print(f"{name:<22}{eta2_log(lab,y):>8.3f}{m['ibrd50']*100:>8.1f}%{w:>8.3f}")

    # 多盘（2 盘 × 8 流，桶绑定盘、均衡映射）
    try:
        from multidisk_sim import run_multidisk
        for D, S in ((2, 8),):
            lab = cut(score, D * S)
            ch = build(df, lab, size)
            r = run_multidisk(ch, D, S, cap / GIB, ru_mb=RU_MB, mode="bucket", strategy="balanced")
            print(f"{f'多盘 {D}x{S} (bal)':<22}{eta2_log(lab,y):>8.3f}"
                  f"{'':>9}{(r['wa_steady'] if r['failed']==0 else float('nan')):>8.3f}")
    except Exception as e:  # noqa: BLE001
        print("多盘实验失败:", e)


if __name__ == "__main__":
    main()
