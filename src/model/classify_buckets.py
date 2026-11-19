"""直接分类到寿命桶：找出 km4 到底错在哪，并尝试改进。

思路：题面要的是"分桶"，那就把"寿命四分位"当作分类标签，用可观测特征直接预测
桶号，而不是先回归寿命再切分位。

对比四种：
  km4        —— 无监督 KMeans（当前方法）
  reg+cut    —— 回归寿命，再按预测值四分位切
  clf        —— 直接四分类（本脚本的重点）
  oracle4    —— 用真值分桶（上限）

指标：分类准确率、混淆矩阵、题面 ±5%、稳态 WA。

用法：python src/classify_buckets.py data/trace_docvqa36
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
    FEATURES, add_kind, attach, design_matrix, load, within_frac,
)

FEATURES_OCC = FEATURES + ["live_gb", "util_at_create"]


def add_occupancy(df, size):
    """给每个块加上"它创建那一刻，全盘/全集群的活跃 KV 占用"。

    这是部署时**可观测**的量（推理服务自己知道当前 KV 用了多少），
    但属于"系统状态"而不是"内容特征"——是否允许用需要确认。
    """
    import heapq

    ev = sorted(zip(df["t_create"].values,
                    np.ones(len(df), dtype=int),
                    (df["t_death"].values)), key=lambda x: x[0])
    # 用 (death_time, size) 的小顶堆模拟"当前活跃"
    live, heap = 0.0, []
    out = []
    # 先按创建排序，逐个插入，同时弹出已死的
    order = np.argsort(df["t_create"].values)
    deaths = df["t_death"].values
    sizes = np.full(len(df), size, dtype=float)
    res = np.zeros(len(df))
    for idx in order:
        t = df["t_create"].values[idx]
        while heap and heap[0][0] <= t:
            _, sz = heapq.heappop(heap)
            live -= sz
        live += sizes[idx]
        heapq.heappush(heap, (deaths[idx], sizes[idx]))
        res[idx] = live / (1 << 30)
    df["live_gb"] = res
    return df
from gc_sim import Chunk, Device  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
KB_PER_TOKEN = 32
RU_MB = 6144.0
OP = 0.07
PASSES = 120
RHO = 0.41


def build(df, lab, size, seed=0, jitter=0.10, window=None):
    """把每条块记录放大成"稳态写入流"。

    window: 放大时间窗。默认取 df 的最大死亡时刻；
            对复制出来的长 trace 应传入**原始**时长，否则数据密度会被稀释。
    """
    rng = np.random.default_rng(seed)
    base = [Chunk(float(tw), float(tw) + max(float(l), 0.05), size, int(s))
            for tw, l, s in zip(df["t_death"].values, df["L_A"].values, lab)]
    span = float(window) if window is not None else float(df["t_death"].max())
    out = []
    for _ in range(PASSES):
        off = rng.uniform(0, span * 4)
        for c in base:
            life = (c.t_death - c.t_write) * (1 + rng.uniform(-jitter, jitter))
            out.append(Chunk(c.t_write + off, c.t_write + off + max(life, 0.05),
                             c.size, c.stream))
    return out


def wa_of(df, lab, size, cap_gb):
    ch = build(df, lab, size)
    try:
        d = Device(ch, capacity_gb=cap_gb, ru_mb=RU_MB, op_frac=OP)
        return d.run()["wa_steady"]
    except RuntimeError:
        return float("nan")


def main():
    td = sys.argv[1] if len(sys.argv) > 1 else "data/trace_docvqa36"
    td = td if os.path.isabs(td) else os.path.join(ROOT, td)
    tag = os.path.basename(td)
    K = 4

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
    df = add_occupancy(df, size)
    y = df["L_A"].values

    # 容量（固定 ρ）
    c0 = build(df, np.zeros(len(df), dtype=int), size)
    W = sum(c.size for c in c0) / (max(c.t_death for c in c0) - min(c.t_write for c in c0))
    cap_gb = float(np.median(y)) * W / RHO / (1 << 30)
    print(f"[{tag}] 块 {len(df)}，L50={np.median(y):.1f}s，ρ={RHO} → 盘 {cap_gb:.0f}GB")

    # 标签：寿命四分位
    edges = np.quantile(y, np.linspace(0, 1, K + 1)[1:-1])
    cls = np.searchsorted(edges, y, side="right")     # 0..K-1
    print("各桶寿命区间:", [f"{edges[i-1] if i>0 else 0:.0f}-{edges[i] if i<K-1 else y.max():.0f}s"
                        for i in range(K)])

    df["util_at_create"] = df["live_gb"] / cap_gb
    X = design_matrix(df, FEATURES_OCC)
    groups = df["sid"].values

    from sklearn.cluster import KMeans
    from sklearn.ensemble import (HistGradientBoostingClassifier,
                                  HistGradientBoostingRegressor)
    from sklearn.model_selection import GroupKFold

    out = {}

    # 1) 无监督 KMeans
    out["km4"] = KMeans(n_clusters=K, n_init=10, random_state=0).fit_predict(X)

    # 2) 回归 + 分位切
    pred_reg = np.zeros(len(df))
    gkf = GroupKFold(n_splits=4)
    for tr, te in gkf.split(X, y, groups):
        m = HistGradientBoostingRegressor(max_depth=6, learning_rate=0.08,
                                          max_iter=300, random_state=0)
        m.fit(X[tr], np.log1p(y[tr]))
        pred_reg[te] = np.expm1(m.predict(X[te]))
    out["reg+cut"] = np.searchsorted(np.quantile(pred_reg, np.linspace(0, 1, K + 1)[1:-1]),
                                     pred_reg, side="right")

    # 3) 直接分类
    pred_cls = np.zeros(len(df), dtype=int)
    proba = None
    for tr, te in gkf.split(X, cls, groups):
        m = HistGradientBoostingClassifier(max_depth=5, learning_rate=0.08,
                                           max_iter=300, random_state=0)
        m.fit(X[tr], cls[tr])
        pred_cls[te] = m.predict(X[te])
    out["clf4"] = pred_cls

    # 4) oracle
    out["oracle4"] = cls

    # 评估
    res = []
    for name, lab in out.items():
        acc = float((np.asarray(lab) == cls).mean())
        f5, p95, _ = within_frac(np.asarray(lab), y)
        wa = wa_of(df, np.asarray(lab), size, cap_gb)
        res.append(dict(method=name, acc_vs_oracle=acc, pass5=f5, p95=p95, wa=wa))
        print(f"  {name:<10} 与真值桶一致率={acc*100:5.1f}%  ±5%={f5*100:5.1f}%  "
              f"p95={p95*100:4.0f}%  WA={wa:.3f}")

    # 混淆矩阵（clf4 vs 真值）
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(cls, out["clf4"], labels=list(range(K)))
    print("\n混淆矩阵（行=真值桶, 列=预测桶）:")
    print(pd.DataFrame(cm, index=[f"真{i}" for i in range(K)],
                       columns=[f"预{i}" for i in range(K)]).to_string())

    # 相邻错 vs 跨桶错
    err = out["clf4"] != cls
    dist = np.abs(out["clf4"] - cls)
    print(f"\n总错误率 {err.mean()*100:.1f}%；其中邻桶错 {((dist==1)).mean()*100:.1f}%，"
          f"跨两桶以上 {(dist>=2).mean()*100:.1f}%")

    # 特征重要性（置换）
    from sklearn.inspection import permutation_importance
    m = HistGradientBoostingClassifier(max_depth=5, learning_rate=0.08,
                                       max_iter=300, random_state=0)
    m.fit(X, cls)
    imp = permutation_importance(m, X, cls, n_repeats=5, random_state=0,
                                 scoring="accuracy")
    names = FEATURES_OCC + [f"kind_{k}" for k in ("system", "image", "assistant_text", "user_text")]
    order = np.argsort(-imp.importances_mean)[:10]
    print("\n最重要的特征:")
    for i in order:
        print(f"   {names[i]:<18} {imp.importances_mean[i]:+.4f}")

    pd.DataFrame(res).to_csv(os.path.join(ROOT, "data", f"classify_{tag}.csv"), index=False)


if __name__ == "__main__":
    main()
