"""系统模型 A 下的统一指标：SSD 写入 = 显存淘汰时刻，SSD 失效 = 会话结束。

为什么必须统一：
  之前 ±5% 用的是 L_sess（会话结束 − 创建），而 WA 仿真用的是 t_death（显存淘汰），
  两个指标算的不是同一个寿命，结论不可比。

模型 A（卸载式）：
  * KV 在显存里被淘汰时才写到 SSD → 写入时刻 = t_death
  * SSD 上的副本在"会话不再需要它"时失效 → 失效时刻 = 该会话结束时刻
  * 因此 SSD 寿命 L_A = session_end − t_death

两个指标都用 L_A：
  * 题面口径：桶内相对偏差 ≤ ±5% 的块占比
  * 物理口径：稳态 WA（chunks 的写入/失效时刻就是 t_death / session_end）

另外给放大的副本加**寿命抖动**（±10%），避免"同一批请求复制 N 份"导致寿命精确重复。

用法：python src/modelA_metrics.py data/trace_docvqa36
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
import matplotlib.pyplot as plt  # noqa: E402
import plotstyle  # noqa: E402
from cluster_lifetime import (  # noqa: E402
    FEATURES, add_kind, attach, design_matrix, load, within_frac,
)
from gc_sim import Chunk, max_live_bytes, run  # noqa: E402

plotstyle.setup()
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

KB_PER_TOKEN = 32
UTIL = 0.80
RU_MB = 128.0
OP = 0.07
PASSES = 80
JITTER = 0.10          # 寿命抖动 ±10%，去掉"精确复制"伪影


def amplify_jitter(chunks, passes, window, seed=0, jitter=JITTER):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(passes):
        off = rng.uniform(0, window)
        for c in chunks:
            life = c.t_death - c.t_write
            life2 = max(life * (1.0 + rng.uniform(-jitter, jitter)), 0.05)
            out.append(Chunk(c.t_write + off, c.t_write + off + life2,
                             c.size, c.stream))
    return out


def main():
    td = sys.argv[1] if len(sys.argv) > 1 else "data/trace_docvqa36"
    td = td if os.path.isabs(td) else os.path.join(ROOT, td)
    tag = os.path.basename(td)

    reqs, blocks = load(td)
    rows, _ = attach(reqs, blocks)
    bs = int(blocks[0].get("block_size", 528))
    from mm_input import MMInput
    rows = add_kind(rows, reqs, MMInput(os.path.join(ROOT, "models", "Qwen3.5-9B")),
                    bs, max(len(r["prompt_token_ids"]) for r in reqs))
    df = pd.DataFrame(rows).reset_index(drop=True)

    # ---- 模型 A 的寿命 ----
    session_end = {}
    for r in reqs:
        session_end[r["sid"]] = max(session_end.get(r["sid"], 0.0), r["t_end"])
    df = df[df["t_death"].notna()].copy()
    df["L_A"] = df["sid"].map(session_end) - df["t_death"]
    df = df[df["L_A"] > 0].reset_index(drop=True)
    print(f"[{tag}] 可用于模型 A 的块 {len(df)} 个（有淘汰时刻且会话在其之后结束）")
    print(f"    L_A: 中位 {df.L_A.median():.1f}s  p10 {df.L_A.quantile(.1):.1f}s  "
          f"p90 {df.L_A.quantile(.9):.1f}s")

    y = df["L_A"].values
    size = int(bs * KB_PER_TOKEN * 1024)
    req_key = df["sid"].astype(str) + "_" + df["turn"].astype(str)
    df["req"] = pd.factorize(req_key)[0]

    # ---- 各种分组 ----
    groups: dict[str, np.ndarray] = {}
    groups["s1 命中三层"] = df["cached_blocks"].map(
        lambda v: 2 if v >= 3 else (1 if v >= 1 else 0)).values

    from sklearn.cluster import KMeans
    groups["km16 特征聚类"] = KMeans(n_clusters=16, n_init=10,
                                   random_state=0).fit_predict(
        design_matrix(df, FEATURES))

    groups["oracle16 真值"] = np.searchsorted(
        np.quantile(y, np.linspace(0, 1, 17)[1:-1]), y, side="right")

    # 请求级：一请求一桶
    groups["请求级(105桶)"] = df["req"].values

    # 请求合并到 32 桶（按真实 L_A 排序 → 这是可实现的，因为排序键是预测值；
    # 这里先用真值给出上限，再用预测值给实际结果）
    req_l = df.groupby("req")["L_A"].mean()
    n_req = len(req_l)
    order = np.argsort(req_l.values)
    bucket = np.empty(n_req, dtype=int)
    bucket[order] = np.floor(np.arange(n_req) * 32 / n_req).astype(int)
    groups["请求合并32(真值)"] = bucket[df["req"].values]

    # 预测（块级模型按会话分组交叉验证，再按请求平均）
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import GroupKFold
    X = design_matrix(df, FEATURES)
    pred = np.zeros(len(df))
    for tr, te in GroupKFold(n_splits=4).split(X, y, df["sid"].values):
        m = HistGradientBoostingRegressor(max_depth=6, learning_rate=0.08,
                                          max_iter=300, random_state=0)
        m.fit(X[tr], np.log1p(y[tr]))
        pred[te] = np.expm1(m.predict(X[te]))
    df["pred"] = pred
    req_p = df.groupby("req")["pred"].mean()
    order_p = np.argsort(req_p.values)
    bucket_p = np.empty(n_req, dtype=int)
    bucket_p[order_p] = np.floor(np.arange(n_req) * 32 / n_req).astype(int)
    groups["请求合并32(预测)"] = bucket_p[df["req"].values]
    acc = np.abs(pred - y).sum() / y.sum()
    from scipy.stats import spearmanr
    print(f"[predict] ρ={spearmanr(pred, y).statistic:.3f}  累积偏差={acc*100:.1f}%")

    # ---- 两条指标 ----
    rows_out = []
    for name, lab in groups.items():
        lab = pd.factorize(lab)[0]
        f5, p95, _ = within_frac(lab, y)
        chunks = [Chunk(float(tw), float(tw) + max(float(l), 0.05), size, int(s))
                  for tw, l, s in zip(df["t_death"].values, y, lab)]
        big = amplify_jitter(chunks, PASSES, window=4 * max(r["t_end"] for r in reqs))
        live = max_live_bytes(big)
        try:
            r = run(big, capacity_gb=live / UTIL / (1 << 30), ru_mb=RU_MB, op_frac=OP)
            wa = r["wa_steady"]
        except RuntimeError as e:
            wa = float("nan")
            print("   仿真失败:", e)
        rows_out.append(dict(strategy=name, n=len(np.unique(lab)), pass5=f5,
                             p95=p95, wa=wa))
        print(f"  {name:<18} 桶={len(np.unique(lab)):3d}  ±5%={f5*100:5.1f}%  "
              f"p95={p95*100:4.0f}%  WA={wa:.3f}")

    res = pd.DataFrame(rows_out)
    res.to_csv(os.path.join(ROOT, "data", f"modelA_{tag}.csv"), index=False)
    plot(res, tag)


def plot(res, tag):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    x = np.arange(len(res))
    colors = ["#DD8452" if "命中" in s else "#4C72B0" if ("km" in s or "预测" in s)
              else "#8C8C8C" if "请求级" in s else "#55A868" for s in res["strategy"]]

    ax = axes[0]
    ax.bar(x, res["pass5"] * 100, color=colors)
    for i, v in enumerate(res["pass5"] * 100):
        ax.text(i, v + 1, f"{v:.0f}%", ha="center", fontsize=8)
    ax.axhline(48, color="k", ls="--", lw=1)
    ax.text(-0.4, 50, "经验阈值 ~48%", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}\n({n} 桶)" for s, n in
                        zip(res["strategy"], res["n"])], fontsize=7.5)
    ax.set_ylabel("±5% 达标率 (%)")
    ax.set_title("(a) 题面口径：桶内 ±5% 的块占比", fontsize=10)
    ax.set_ylim(0, 80)

    ax2 = axes[1]
    ax2.plot(x, res["wa"], "o-", color="#4C72B0", lw=1.8, ms=7)
    for i, v in enumerate(res["wa"]):
        ax2.text(i, v + 0.002, f"{v:.3f}", ha="center", fontsize=8)
    ax2.axhline(1.1, color="k", ls="--", lw=1)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{s}\n({n} 桶)" for s, n in
                         zip(res["strategy"], res["n"])], fontsize=7.5)
    ax2.set_ylabel("稳态写放大 WA")
    ax2.set_title("(b) 物理口径：稳态写放大（纵轴从 0.98 起）", fontsize=10)
    ax2.set_ylim(0.98, 1.09)

    fig.suptitle(f"模型 A（写入=淘汰时刻，失效=会话结束）下的两个指标（{tag}）", fontsize=12)
    out = os.path.join(ROOT, "figs", f"fig10_modelA_{tag}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[fig] {out}")


if __name__ == "__main__":
    main()
