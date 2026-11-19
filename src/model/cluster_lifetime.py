"""核心实验：用"可外部观测、不依赖模型/框架"的特征把 chunk 分到 k 个桶里，
看桶内寿命偏差能不能压到 ±5%。

三种分组策略对比：
  1. baseline   —— 题面描述的"命中概率分温冷热三层"（k 固定 3）
  2. oracle     —— 直接按真实寿命分位分 k 桶（任何算法的天花板）
  3. ours       —— 只用可观测特征：
       (a) KMeans 聚类（无监督）
       (b) 用特征回归寿命，再按预测值分位分桶（有监督，需要训练集）

两种寿命定义都跑：
  L_phys = 淘汰时刻 - 写入时刻（HBM 里的驻留时间）
  L_sess = 会话结束时刻 - 写入时刻（这个用户还会用多久）

指标（题面口径）：桶内任意 chunk 相对桶均值的偏差 ≤ ±5% 的占比。

用法：python src/cluster_lifetime.py data/trace_v1
"""

from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _setup import add_src_paths  # noqa: E402
add_src_paths()

import json
import os
import sys

import numpy as np
import pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------- 数据

def load(trace_dir: str):
    reqs = [json.loads(l) for l in open(os.path.join(trace_dir, "requests.jsonl"))]
    blocks = [json.loads(l) for l in open(os.path.join(trace_dir, "blocks.jsonl"))]
    return reqs, blocks


def attach(reqs, blocks):
    """块 -> 创建它的请求 + 该请求可观测的上下文特征。"""
    by_sid: dict[int, list[dict]] = {}
    for r in reqs:
        by_sid.setdefault(r["sid"], []).append(r)
    for rs in by_sid.values():
        rs.sort(key=lambda r: r["turn"])
    session_end = {sid: max(r["t_end"] for r in rs) for sid, rs in by_sid.items()}
    session_start = {sid: min(r["t_start"] for r in rs) for sid, rs in by_sid.items()}

    # 每条请求：前一轮的结束时刻 / 生成长度
    prev_end, prev_gen = {}, {}
    for sid, rs in by_sid.items():
        for i, r in enumerate(rs):
            prev_end[(sid, r["turn"])] = rs[i - 1]["t_end"] if i > 0 else r["t_start"]
            prev_gen[(sid, r["turn"])] = rs[i - 1]["gen_len"] if i > 0 else 0

    # 每条会话里"到目前为止的平均思考时间"，用来判断用户现在聊得比平时快还是慢
    seen_idle: dict[int, list[float]] = {}
    prev_prompt: dict[tuple, int] = {}
    for sid, rs in by_sid.items():
        for i, r in enumerate(rs):
            prev_prompt[(sid, r["turn"])] = len(rs[i - 1]["prompt_token_ids"]) if i > 0 else 0

    starts = sorted(r["t_start"] for r in reqs)

    def n_active(t):
        # 该时刻正在跑的请求数（粗粒度：t_start <= t <= t_end）
        return sum(1 for r in reqs if r["t_start"] <= t <= r["t_end"])

    rows = []
    for b in blocks:
        bs = b.get("block_size", 528)
        pos0 = b.get("pos0", -1)
        if pos0 < 0 or not b.get("t_death"):
            continue
        t_c = b["t_create"]
        cand = [r for r in reqs
                if r["t_start"] - 0.05 <= t_c <= r["t_end"]
                and pos0 + bs <= len(r["prompt_token_ids"])]
        if not cand:
            continue
        # 优先用 token 精确匹配：前缀缓存会让多个请求共享同一段 KV，
        # 仅按时间匹配会把块归属到错误的请求/会话，污染寿命标签。
        toks = b.get("tokens") or []
        matched = None
        if toks:
            for r in cand:
                if r["prompt_token_ids"][pos0:pos0 + bs] == toks[:bs]:
                    matched = r
                    break
        if matched is not None:
            r = matched
        else:
            r = min(cand, key=lambda x: abs(t_c - x["t_start"]))
        sid, turn = r["sid"], r["turn"]
        L_phys = b["t_death"] - t_c
        L_sess = session_end[sid] - t_c
        # 会话行为特征（写入时刻就能拿到）
        past = seen_idle.setdefault(sid, [])
        prev_idle = prev_end[(sid, turn)] - (by_sid[sid][turn - 1]["t_start"]
                                             if turn > 0 else t_c)
        idle_mean = float(np.mean(past)) if past else 0.0
        past.append(max(prev_idle, 0.0))
        img_ends = [sp[1] for sp in (r.get("image_spans") or [])]
        last_img_end = max(img_ends) if img_ends else 0

        rows.append({
            "key": b.get("key"),
            "sid": sid, "turn": turn, "pos0": pos0,
            "n_active": n_active(t_c),
            "idle_ratio": (max(prev_idle, 0.0) / (idle_mean + 0.1)) if past else 1.0,
            "turn_frac": turn / 5.0,
            "blocks_in_req": len(r["prompt_token_ids"]) // bs,
            "prompt_growth": len(r["prompt_token_ids"]) - prev_prompt[(sid, turn)],
            "pos_from_img": (pos0 - last_img_end) if last_img_end else -1.0,
            "block_idx": pos0 // bs,
            "pos_frac": pos0 / max(len(r["prompt_token_ids"]), 1),
            "prompt_len": len(r["prompt_token_ids"]),
            "n_images": r.get("n_images", 0),
            "cached_blocks": int(r.get("num_cached_tokens") or 0) // bs,
            "session_age": t_c - session_start[sid],
            "idle_before": t_c - prev_end[(sid, turn)],
            "prev_gen": prev_gen[(sid, turn)],
            "n_active": n_active(t_c),
            "kind": None,          # 下面用位置投票补
            "t_create": t_c,
            "t_death": b.get("t_death"),
            "L_phys": L_phys,
            "L_sess": L_sess,
        })
    return rows, by_sid


def add_kind(rows, reqs, mm, bs, max_len):
    from collections import Counter
    votes: dict[int, Counter] = {}
    for r in reqs:
        ids = r["prompt_token_ids"]
        segs = mm.segment(ids, n_turns=r["turn"] + 1)
        kinds = ["other"] * len(ids)
        for sg in segs:
            for p in range(sg.start, min(sg.end, len(ids))):
                kinds[p] = sg.kind
        for pos0 in range(0, len(ids), bs):
            if pos0 >= len(ids):
                continue
            c = Counter(k for k in kinds[pos0:pos0 + bs] if k != "other")
            if c:
                votes.setdefault(pos0, Counter()).update(c)
    pos_kind = {p: c.most_common(1)[0][0] for p, c in votes.items()}
    for r in rows:
        r["kind"] = pos_kind.get(r["pos0"], "other")
    return rows


# ---------------------------------------------------------------- 指标

def within_frac(labels, lifetime, thresh=0.05, min_size=10):
    """按 labels 分桶，返回 (加权±5%内占比, p95相对偏差, 平均桶大小)。"""
    lifetime = np.asarray(lifetime, dtype=float)
    labels = np.asarray(labels)
    covered, total, devs, sizes = 0, 0, [], []
    for b in np.unique(labels):
        m = labels == b
        n = m.sum()
        if n < min_size:
            continue                      # 太小的桶不计（防止"每块一个桶"作弊）
        g = lifetime[m]
        center = g.mean()
        if center <= 0:
            continue
        d = np.abs(g - center) / center
        covered += (d <= thresh).sum()
        total += n
        devs.extend(d.tolist())
        sizes.append(n)
    if total == 0:
        return 0.0, float("nan"), 0.0
    return covered / total, float(np.percentile(devs, 95)), float(np.mean(sizes))


def oracle_labels(values, k):
    edges = np.quantile(values, np.linspace(0, 1, k + 1)[1:-1])
    return np.searchsorted(edges, values, side="right")


# ---------------------------------------------------------------- 主流程

FEATURES = ["turn", "block_idx", "pos_frac", "prompt_len", "n_images",
            "cached_blocks", "session_age", "idle_before", "prev_gen",
            # 新增：负载 / 会话行为
            "n_active", "idle_ratio", "turn_frac", "blocks_in_req",
            "prompt_growth", "pos_from_img"]
FEATURES_STRUCT = ["turn", "block_idx", "pos_frac", "prompt_len", "cached_blocks",
                   "session_age", "idle_before"]


def design_matrix(df, feats):
    X = df[feats].astype(float).copy()
    # 类别型：kind 独热
    for k in ("system", "image", "assistant_text", "user_text"):
        X[f"kind_{k}"] = (df["kind"] == k).astype(float)
    X = X.replace([np.inf, -np.inf], 0).fillna(0)
    mu, sd = X.mean(), X.std().replace(0, 1)
    return ((X - mu) / sd).values


def evaluate(df, tag, out_rows):
    from sklearn.cluster import KMeans
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import GroupKFold

    for target in ("L_phys", "L_sess"):
        y = df[target].values

        # 1) baseline：按命中次数分三层（题面描述的方法）
        lab = df["cached_blocks"].map(lambda v: 2 if v >= 3 else (1 if v >= 1 else 0))
        f, p95, sz = within_frac(lab.values, y)
        out_rows.append(dict(trace=tag, target=target, method="baseline(命中3层)",
                             k=3, within5=f, p95=p95, avg_size=sz))

        # 2) oracle：按真值分位
        for k in (3, 8, 16, 32, 64):
            f, p95, sz = within_frac(oracle_labels(y, k), y)
            out_rows.append(dict(trace=tag, target=target, method="oracle", k=k,
                                 within5=f, p95=p95, avg_size=sz))

        # 2.5) 规则：按"轮次"分桶 / 按"轮次 × 会话已用时长"分桶（都是可观测的）
        for name, key in (("rule(turn)", df["turn"].astype(int)),
                          ("rule(turn×age4)", df["turn"].astype(int) * 4
                           + pd.qcut(df["session_age"], 4, labels=False).astype(int)),
                          ("rule(turn×age8)", df["turn"].astype(int) * 8
                           + pd.qcut(df["session_age"], 8, labels=False).astype(int))):
            f, p95, sz = within_frac(key.values, y)
            out_rows.append(dict(trace=tag, target=target, method=name,
                                 k=int(key.nunique()), within5=f, p95=p95, avg_size=sz))

        # 3a) KMeans（无监督，只用特征）
        for name, feats in (("kmeans(all)", FEATURES), ("kmeans(struct)", FEATURES_STRUCT)):
            X = design_matrix(df, feats)
            for k in (3, 8, 16, 32, 64):
                lab = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(X)
                f, p95, sz = within_frac(lab, y)
                out_rows.append(dict(trace=tag, target=target, method=name, k=k,
                                     within5=f, p95=p95, avg_size=sz))

        # 3b) 特征 -> 回归寿命 -> 按预测值分位分桶（按会话分组交叉验证，防泄漏）
        X = design_matrix(df, FEATURES)
        groups = df["sid"].values
        pred = np.zeros(len(df))
        gkf = GroupKFold(n_splits=4)
        for tr, te in gkf.split(X, y, groups):
            m = HistGradientBoostingRegressor(max_depth=6, learning_rate=0.08,
                                              max_iter=300, random_state=0)
            m.fit(X[tr], np.log1p(y[tr]))
            pred[te] = np.expm1(m.predict(X[te]))
        pred = np.clip(pred, 0, None)
        # 诊断：特征到底能解释多少寿命？
        from scipy.stats import spearmanr
        rho = spearmanr(pred, y).statistic
        ss_res = ((np.log1p(y) - np.log1p(pred)) ** 2).sum()
        ss_tot = ((np.log1p(y) - np.log1p(y).mean()) ** 2).sum()
        acc = np.abs(pred - y).sum() / y.sum()
        print(f"[{tag}/{target}] 预测质量: Spearman ρ={rho:.3f}, "
              f"log 空间 R²={1 - ss_res / ss_tot:.3f}, 累积相对偏差(Σ|pred-true|/Σtrue)={acc*100:.1f}%")
        out_rows.append(dict(trace=tag, target=target, method="predict(累积偏差)",
                             k=-1, within5=acc, p95=float("nan"), avg_size=len(df)))
        for k in (3, 8, 16, 32, 64):
            f, p95, sz = within_frac(oracle_labels(pred, k), y)
            out_rows.append(dict(trace=tag, target=target, method="predict+桶", k=k,
                                 within5=f, p95=p95, avg_size=sz))
        print(f"[{tag}/{target}] predict+桶: " +
              ", ".join(f"k={r['k']}:{r['within5']*100:.0f}%"
                        for r in out_rows if r['trace'] == tag and r['target'] == target
                        and r['method'] == 'predict+桶'))


def main():
    from mm_input import MMInput

    td = sys.argv[1] if len(sys.argv) > 1 else "data/trace_v1"
    td = td if os.path.isabs(td) else os.path.join(ROOT, td)
    tag = os.path.basename(td)

    reqs, blocks = load(td)
    rows, _ = attach(reqs, blocks)
    bs = int(blocks[0].get("block_size", 528))
    mm = MMInput(os.path.join(ROOT, "models", "Qwen3.5-9B"))
    rows = add_kind(rows, reqs, mm, bs, max(len(r["prompt_token_ids"]) for r in reqs))

    df = pd.DataFrame(rows)
    print(f"[{tag}] 可归属块 {len(df)}，会话 {df.sid.nunique()}")
    print("特征分布抽样:")
    print(df[["turn", "block_idx", "n_images", "cached_blocks", "session_age",
              "idle_before", "L_phys", "L_sess"]].describe().round(1).to_string())

    out_rows: list[dict] = []
    evaluate(df, tag, out_rows)
    res = pd.DataFrame(out_rows)
    out = os.path.join(ROOT, "data", f"cluster_eval_{tag}.csv")
    res.to_csv(out, index=False)
    print(f"\n=== {tag} 结果 ===")
    print(res.pivot_table(index=["target", "method"], columns="k",
                          values="within5").round(3).to_string())
    print(f"-> {out}")


if __name__ == "__main__":
    main()
