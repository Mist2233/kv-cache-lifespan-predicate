"""对比两种"寿命"定义的可聚类性（只用 CPU，几分钟）。

定义 P（物理）：chunk 在 HBM 前缀缓存里从写入到被淘汰的时间。
定义 B（会话）：chunk 从写入到它所属会话结束的时间——即"这个用户还会用它多久"。

对每种定义，做**oracle 分桶**：直接按真实寿命值分 k 桶（这是任何预测算法的
天花板——预测再准也不会比"直接知道真值"更好），再看：
    同一桶内，任意 chunk 相对桶均值的偏差 ≤ ±5% 的占比
这个指标就是题面的验收口径（聚类偏差）。

如果连 oracle 都做不到，说明这个定义下"k 个桶"本身就不够；如果 oracle 很高，
说明问题只是"怎么把 chunk 分到正确的桶"，那是可以做的。

    python src/compare_definitions.py data/trace_v1
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

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load(trace_dir):
    reqs = [json.loads(l) for l in open(os.path.join(trace_dir, "requests.jsonl"))]
    blocks = [json.loads(l) for l in open(os.path.join(trace_dir, "blocks.jsonl"))]
    return reqs, blocks


def attach(reqs, blocks):
    """把每个块归给它所属的请求（用创建时刻落在哪条请求的生命周期里）。"""
    session_end: dict[int, float] = {}
    for r in reqs:
        session_end[r["sid"]] = max(session_end.get(r["sid"], 0.0), r["t_end"])

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
        creator = min(cand, key=lambda r: abs(t_c - r["t_start"]))
        L_phys = b["t_death"] - t_c
        L_sess = session_end[creator["sid"]] - t_c
        rows.append({"sid": creator["sid"], "turn": creator["turn"],
                     "pos0": pos0, "L_phys": L_phys, "L_sess": L_sess})
    return rows, session_end


def oracle_bucket(values, k):
    """按真值分位数分 k 桶，返回 (落在±5%内的占比, 桶数)。"""
    v = np.asarray(values, dtype=float)
    edges = np.quantile(v, np.linspace(0, 1, k + 1)[1:-1])
    idx = np.searchsorted(edges, v, side="right")
    inside = []
    for b in np.unique(idx):
        g = v[idx == b]
        center = g.mean()
        if center <= 0:
            continue
        dev = np.abs(g - center) / center
        inside.append((dev <= 0.05).mean())
    return float(np.average(inside, weights=[(idx == b).sum() for b in np.unique(idx)])), len(np.unique(idx))


def main():
    td = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "data/trace_v1")
    if not os.path.isabs(td):
        td = os.path.join(ROOT, td)
    reqs, blocks = load(td)
    rows, session_end = attach(reqs, blocks)
    n = len(rows)
    print(f"可归属的块: {n} / {len(blocks)}；会话数 {len(session_end)}")

    for key in ("L_phys", "L_sess"):
        v = np.array([r[key] for r in rows])
        print(f"\n=== 定义 {key}: 中位 {np.median(v):.1f}s, p10 {np.percentile(v,10):.1f}s, "
              f"p90 {np.percentile(v,90):.1f}s, CV {v.std()/v.mean():.2f}")
        for k in (3, 8, 16, 32):
            frac, nb = oracle_bucket(v, k)
            print(f"   oracle {k:2d} 桶 -> ±5% 内 {frac*100:5.1f}%  (实际桶数 {nb})")

    # 同一请求内的块，会话寿命自然完全相同 —— 这是定义 B 的"免费"结构
    from collections import defaultdict
    grp = defaultdict(list)
    for r in rows:
        grp[(r["sid"], r["turn"])].append(r["L_sess"])
    same_req = [np.std(v) / max(np.mean(v), 1e-9) for v in grp.values() if len(v) > 1]
    print(f"\n同一请求内 L_sess 的相对标准差: {np.mean(same_req):.4f} "
          f"(n={len(same_req)} 组) —— 定义 B 下同请求的块寿命天然一致")

    # 不同会话/轮次之间的跨度：决定"需要几个桶"
    vals = sorted({round(r["L_sess"], 1) for r in rows})
    print(f"L_sess 的不同取值有 {len(vals)} 个，范围 {vals[0]:.1f}–{vals[-1]:.1f}s")
    for thresh in (0.05, 0.10):
        # 用"贪心合并相邻值"估计：要多少桶才能让每个桶内相对偏差 ≤ thresh
        buckets, cur = [], []
        for x in vals:
            cur.append(x)
            c = np.mean(cur)
            if (max(cur) - min(cur)) / c > 2 * thresh:
                cur.pop()
                buckets.append(cur)
                cur = [x]
        if cur:
            buckets.append(cur)
        print(f"   若要求桶内 max|L-mean|/mean ≤ {thresh:.0%}，至少需要 {len(buckets)} 个桶")


if __name__ == "__main__":
    main()
