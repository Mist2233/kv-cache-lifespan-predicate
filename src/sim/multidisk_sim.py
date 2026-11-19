"""多盘 + 多流（FDP）仿真 —— 两种"盘-流"关系。

题干"多盘模拟多流"没有规定数据如何在盘间分布，因此我们把它具体化为两种模型：

**模型 A：桶绑定盘（bucket-partitioned）**
  - 每块盘最多支持 S 个 FDP stream，D 块盘共可支持 K = D×S 个桶；
  - 每个桶固定落在某一块盘上（一个桶的数据只写这一块盘）；
  - 作用：**多盘可以支持更多桶**（细分组），代价是每块盘容量只有 1/D、RU 变少，
    必须满足 R_d ≥ 3S，否则该盘 GC 会死锁。

**模型 B：块轮转分盘（chunk-partitioned）**
  - 每块盘都承载全部 K 个桶（每盘 K 个 stream），一块 KV 数据按块轮转落到某块盘；
  - 多盘不增加桶数，只分担写入量；
  - 此时每盘利用率 = (活跃数据/D) / 盘容量；若每盘容量取总容量的 1/D，
    则多盘是"线性扩展"，写放大与单盘完全一致（尺度不变）。

写放大统一为全局口径：WA = Σ盘拷贝字节 / Σ盘主机写入字节。
"""

from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _setup import add_src_paths  # noqa: E402
add_src_paths()

import numpy as np

from gc_sim import Chunk, Device, max_live_bytes


def _bucket_bytes(chunks):
    d: dict[int, float] = {}
    for c in chunks:
        d[c.stream] = d.get(c.stream, 0.0) + c.size
    return d


def _assign_bucket_to_disk(chunks, n_disks: int, streams_per_disk: int, strategy: str):
    streams = sorted({c.stream for c in chunks})
    if n_disks * streams_per_disk < len(streams):
        raise ValueError(f"盘数×每盘流数({n_disks}×{streams_per_disk}) < 桶数({len(streams)})")
    if strategy == "consecutive":
        return {b: b // streams_per_disk for b in streams}
    bsz = _bucket_bytes(chunks)
    load, cnt, out = [0.0] * n_disks, [0] * n_disks, {}
    for b in sorted(streams, key=lambda x: -bsz.get(x, 0.0)):
        cand = [d for d in range(n_disks) if cnt[d] < streams_per_disk]
        d = min(cand, key=lambda x: load[x])
        out[b] = d
        load[d] += bsz.get(b, 0.0)
        cnt[d] += 1
    return out


def run_multidisk(chunks, n_disks: int, streams_per_disk: int, total_capacity_gb: float,
                  ru_mb: float = 6144.0, op_frac: float = 0.07,
                  mode: str = "bucket", strategy: str = "balanced") -> dict:
    """mode="bucket"（模型 A）或 "chunk"（模型 B）。"""
    per_disk: list[list[Chunk]] = [[] for _ in range(n_disks)]
    if mode == "bucket":
        mapping = _assign_bucket_to_disk(chunks, n_disks, streams_per_disk, strategy)
        for c in chunks:
            per_disk[mapping[c.stream]].append(c)
    elif mode == "chunk":
        if streams_per_disk < (max(c.stream for c in chunks) + 1):
            raise ValueError("模型 B 要求每盘 Stream 数 ≥ 桶数")
        for i, c in enumerate(chunks):
            per_disk[i % n_disks].append(c)
    else:
        raise ValueError(mode)

    cap_d = total_capacity_gb / n_disks
    host_tot = copies_tot = 0.0
    steady_num = steady_den = 0.0
    disks = []
    for d in range(n_disks):
        ch = per_disk[d]
        live = max_live_bytes(ch) if ch else 0.0
        util = live / (cap_d * (1 << 30)) if cap_d > 0 else float("inf")
        if not ch:
            disks.append(dict(disk=d, host_gb=0.0, wa=float("nan"), steady=float("nan"),
                              live_gb=0.0, util=0.0, n_streams=0, n_rus=0))
            continue
        try:
            rep = Device(ch, capacity_gb=cap_d, ru_mb=ru_mb, op_frac=op_frac).run()
        except RuntimeError as e:
            disks.append(dict(disk=d, error=str(e), host_gb=0.0, wa=float("nan"),
                              steady=float("nan"), live_gb=live / (1 << 30),
                              util=util, n_streams=len({c.stream for c in ch}), n_rus=0))
            continue
        host_tot += rep["host_gb"]
        copies_tot += rep["copies_gb"]
        steady_num += rep["wa_steady"] * rep["host_gb"]
        steady_den += rep["host_gb"]
        disks.append(dict(disk=d, host_gb=rep["host_gb"], wa=rep["wa"],
                          steady=rep["wa_steady"], live_gb=live / (1 << 30),
                          util=util, n_streams=len({c.stream for c in ch}),
                          n_rus=rep["n_rus"], copies_gb=rep["copies_gb"]))

    failed = sum(1 for d in disks if "error" in d)
    wa = float("nan") if failed else (
        (host_tot + copies_tot) / host_tot if host_tot > 0 else float("nan"))
    steady = float("nan") if failed else (
        steady_num / steady_den if steady_den > 0 else float("nan"))
    return dict(wa=wa, wa_steady=steady, mode=mode, n_disks=n_disks,
                streams_per_disk=streams_per_disk, strategy=strategy,
                per_disk=disks, failed=failed,
                max_util=max((d["util"] for d in disks), default=float("nan")))


def main():
    import os
    import sys
    import pandas as pd
    import classify_buckets as cb
    from cluster_lifetime import add_kind, attach, load
    from classify_buckets import build
    from features_v2 import FEATURES_V2, add_features_v2
    from fusion_v3 import cut, oof_rank
    from gc_sim import max_live_bytes as mlb

    ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    arg = sys.argv[1] if len(sys.argv) > 1 else "data/trace_docvqa36"
    td = arg if os.path.isabs(arg) else os.path.join(ROOT, arg)

    reqs, blocks = load(td)
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
    cb.PASSES = 120
    size = int(bs * 32 * 1024)
    n = len(df)

    ch0 = build(df, np.zeros(n, dtype=int), size)
    live = mlb(ch0)
    U = 0.74
    total_cap = live / U / (1 << 30)
    print(f"[{os.path.basename(td)}] 块 {n}，活跃 {live/(1<<30):.0f}GiB，"
          f"总容量 {total_cap:.0f}GiB，利用率 {U}")
    print(f"\n{'方案':<44}{'桶数':>5}{'全局WA':>8}{'最大盘利用':>10}{'失败盘':>6}")

    def show(name, r, K):
        w = f"{r['wa_steady']:.3f}" if r.get("failed", 0) == 0 else "失败"
        print(f"{name:<44}{K:>5}{w:>8}{r.get('max_util', float('nan')):>10.3f}"
              f"{r.get('failed', 0):>6}")

    # 单盘基线
    try:
        rep = Device(ch0, capacity_gb=total_cap, ru_mb=6144, op_frac=0.07).run()
        show("不分层（单盘）", dict(wa_steady=rep["wa_steady"],
                                    max_util=live / (total_cap * (1 << 30)), failed=0), 1)
    except RuntimeError:
        show("不分层（单盘）", dict(failed=1, max_util=float("nan")), 1)
    for K in (8, 16):
        lab = cut(oof_rank(df, FEATURES_V2, K), K)
        ch = build(df, lab, size)
        try:
            rep = Device(ch, capacity_gb=total_cap, ru_mb=6144, op_frac=0.07).run()
            show(f"A3 单盘 {K} 流", dict(wa_steady=rep["wa_steady"],
                                         max_util=mlb(ch) / (total_cap * (1 << 30)), failed=0), K)
        except RuntimeError:
            show(f"A3 单盘 {K} 流", dict(failed=1, max_util=float("nan")), K)

    # 模型 B：多盘线性扩展（每盘都承载全部桶，容量按 1/D 分）
    for D in (2, 4):
        for K in (8, 16):
            lab = cut(oof_rank(df, FEATURES_V2, K), K)
            ch = build(df, lab, size)
            r = run_multidisk(ch, D, K, total_cap, mode="chunk")
            show(f"A3 模型B（{D} 盘 × {K} 流，线性扩展）", r, K)

    # 模型 A：桶绑定盘（多盘换更多桶）
    for D, S in ((2, 8), (4, 4), (4, 8)):
        K = D * S
        lab = cut(oof_rank(df, FEATURES_V2, K), K)
        ch = build(df, lab, size)
        for strat in ("consecutive", "balanced"):
            r = run_multidisk(ch, D, S, total_cap, mode="bucket", strategy=strat)
            show(f"A3 模型A（{D} 盘 × {S} 流，{strat[:3]}）", r, K)
            if strat == "balanced" and r["failed"] == 0:
                for d in r["per_disk"]:
                    print(f"      盘{d['disk']}: {d['n_streams']} 流  "
                          f"利用 {d['util']:.2f}  WA {d['wa']:.3f}  RU {d['n_rus']}")


if __name__ == "__main__":
    main()
