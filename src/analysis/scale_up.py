"""规模扩展实验：把真实 trace 按时间平移复制 N 份，验证"设备充足"假设。

复制方式：每个副本里的会话获得新的 sid，但保留 orig_sid 用于交叉验证分组
（避免同一会话的副本同时出现在训练集和测试集）。

关键：活跃数据随 N 线性增长 → 容量 = live/0.74 也线性增长 → RU 数线性增长。
因此这个实验能干净地隔离"设备规模"这一个变量。

用法：python src/scale_up.py data/trace_docvqa36
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
from features_v2 import FEATURES_V2, add_features_v2  # noqa: E402
from fusion_v3 import cut, eta2_log, wa_of  # noqa: E402
import classify_buckets as cb  # noqa: E402


def oof_rank_g(df, feats, K, groups, seed=0, n_pairs=30000):
    """按给定分组做 OOF 排序（复制出来的副本要和原会话同组，防止泄漏）。"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    X = design_matrix(df, feats)
    logy = np.log(np.maximum(df["L_A"].values, 1e-3))
    score = np.zeros(len(df))
    rng = np.random.default_rng(seed)
    n_splits = min(4, len(np.unique(groups)))
    for tr, te in GroupKFold(n_splits=n_splits).split(X, cut(logy, K), groups=groups):
        i = rng.integers(0, len(tr), n_pairs)
        j = rng.integers(0, len(tr), n_pairs)
        m = logy[tr][i] != logy[tr][j]
        i, j = i[m], j[m]
        lm = LogisticRegression(max_iter=2000).fit(
            X[tr][i] - X[tr][j], (logy[tr][i] > logy[tr][j]).astype(int))
        score[te] = X[te] @ lm.coef_[0]
    return score
from gc_sim import Device, max_live_bytes  # noqa: E402
from metrics import summarize  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
KB_PER_TOKEN = 32
RU_GIB = 6.0


def replicate(reqs, blocks, n):
    """把整条 trace 复制 n 份，第 k 份整体平移 k×总时长。"""
    if n == 1:
        return reqs, blocks
    new_r, new_b = [], []
    for k in range(n):
        # 微小平移：让第 k 份副本与原份在时间上几乎重合（模拟"更多并发用户"），
        # 同时又足够区分，避免 attach 把块错配到别的副本。
        off = k * 0.005
        for r in reqs:
            r2 = dict(r)
            r2["sid"] = r["sid"] + k * 100000
            r2["orig_sid"] = r["sid"]
            r2["t_start"] = r["t_start"] + off
            r2["t_end"] = r["t_end"] + off
            new_r.append(r2)
        for b in blocks:
            b2 = dict(b)
            b2["t_create"] = b["t_create"] + off
            b2["t_death"] = (b["t_death"] + off) if b.get("t_death") else b.get("t_death")
            b2["t_last_access"] = b.get("t_last_access", 0) + off
            new_b.append(b2)
    return new_r, new_b


def main():
    trace = sys.argv[1] if len(sys.argv) > 1 else "data/trace_docvqa36"
    td = trace if os.path.isabs(trace) else os.path.join(ROOT, trace)
    tag = os.path.basename(td)

    reqs0, blocks0 = load(td)
    orig_span = max(r["t_end"] for r in reqs0) - min(r["t_start"] for r in reqs0) + 1.0
    bs = int(blocks0[0].get("block_size", 528))
    from mm_input import MMInput
    MM = MMInput(os.path.join(ROOT, "models", "Qwen3.5-9B"))

    print(f"{'副本':>4}{'块':>7}{'活跃GiB':>9}{'容量GiB':>9}{'RU数':>7}{'K':>4}"
          f"{'不分层':>8}{'A3':>7}{'oracle':>8}")
    rows = []
    for n in (1, 2, 3):
        reqs, blocks = replicate(reqs0, blocks0, n)
        rows_all, _ = attach(reqs, blocks)
        rows_all = add_kind(rows_all, reqs, MM, bs,
                            max(len(r["prompt_token_ids"]) for r in reqs))
        df = pd.DataFrame(rows_all)
        se = {}
        for r in reqs:
            se[r["sid"]] = max(se.get(r["sid"], 0.0), r["t_end"])
        df = df[df["t_death"].notna()].copy()
        df["L_A"] = df["sid"].map(se) - df["t_death"]
        df = df[df["L_A"] > 0].reset_index(drop=True)
        df = add_features_v2(df, reqs)
        df["group"] = df["sid"].map(lambda s: s % 100000)   # 用原始会话分组，防泄漏
        size = int(bs * KB_PER_TOKEN * 1024)
        y = df["L_A"].values
        logy = np.log(np.maximum(y, 1e-3))

        passes = 120        # 更多用户 = 更高写入速率，passes 保持不变
        cb.PASSES = passes
        c0 = build(df, np.zeros(len(df), dtype=int), size, window=orig_span)
        live = max_live_bytes(c0)
        cap = live / 0.74
        n_ru = int(cap / (RU_GIB * (1 << 30)))
        K = int(np.clip(n_ru / 7, 3, 16))

        s0 = np.zeros(len(df), dtype=int)
        w0 = Device(c0, capacity_gb=cap / (1 << 30), ru_mb=RU_GIB * 1024,
                    op_frac=0.07).run()["wa_steady"]
        try:
            a3 = cut(oof_rank_g(df, FEATURES_V2, K, df["group"].values), K)
            w3 = Device(build(df, a3, size, window=orig_span), capacity_gb=cap / (1 << 30),
                        ru_mb=RU_GIB * 1024, op_frac=0.07).run()["wa_steady"]
        except RuntimeError:
            w3 = float("nan")
        try:
            orc = pd.factorize(np.searchsorted(
                np.quantile(logy, np.linspace(0, 1, K + 1)[1:-1]), logy, side="right"))[0]
            wo = Device(build(df, orc, size, window=orig_span), capacity_gb=cap / (1 << 30),
                        ru_mb=RU_GIB * 1024, op_frac=0.07).run()["wa_steady"]
        except RuntimeError:
            wo = float("nan")
        print(f"{n:>4}{len(df):>7}{live/(1<<30):>9.0f}{cap/(1<<30):>9.0f}{n_ru:>7}{K:>4}"
              f"{w0:>8.3f}{w3:>7.3f}{wo:>8.3f}")
        rows.append(dict(n=n, blocks=len(df), live_gib=live / (1 << 30),
                         cap_gib=cap / (1 << 30), n_ru=n_ru, K=K,
                         wa_s0=w0, wa_a3=w3, wa_oracle=wo))
    pd.DataFrame(rows).to_csv(os.path.join(ROOT, "data", "scale_up.csv"), index=False)


if __name__ == "__main__":
    main()
