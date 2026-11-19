"""把 vLLM 的 KV 事件（zmq 订阅落盘）还原成 block 生命周期表。

vLLM 的事件语义（v1）：
    BlockStored   —— 块被写入 GPU 前缀缓存。若该 hash 之前已存在，这次属于
                     "命中复用"（emit_cached_block_events），等价于一次访问。
    BlockRemoved  —— 块被淘汰/释放，之后该 hash 的数据就是垃圾了。

于是每个块：
    t_create      = 第一次 BlockStored 的 ts
    t_last_access = 最后一次 BlockStored（含复用）的 ts
    t_death       = 第一次 BlockRemoved 的 ts（没有则是删失）

注意：混合架构（Qwen3.5 每 4 层 1 层 full attention，其余是线性注意力）里，
线性注意力层保存的是 **请求级的常量状态**，不是逐 token 的 KV。所以这里只保留
kv_cache_spec_kind 为 full_attention / 未标注 的事件。
"""

from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _setup import add_src_paths  # noqa: E402
add_src_paths()

import argparse
import json
import os
import sys
from collections import defaultdict
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace_dir")
    ap.add_argument("--out", default="blocks.jsonl")
    ap.add_argument("--keep-mamba", action="store_true")
    args = ap.parse_args()

    td = args.trace_dir if os.path.isabs(args.trace_dir) else os.path.join(ROOT, args.trace_dir)
    ev_path = os.path.join(td, "kv_events.jsonl")
    if not os.path.exists(ev_path):
        print(f"[skip] {ev_path} 不存在")
        return

    # 每个 hash 维护"当前这一段生命"：被淘汰后再次写入 = 新的一段生命。
    # 不这样切，会把"淘汰后重建"的块算成一个超长寿命（active > resident 的 bug 就来自这里）。
    open_life: dict[str, dict] = {}
    rows: list[dict] = []
    skipped_kind = 0

    def close(h: str, t: float):
        life = open_life.pop(h, None)
        if life is None:
            return
        life["t_death"] = t
        life["censored"] = False
        rows.append(life)

    with open(ev_path) as f:
        for line in f:
            e = json.loads(line)
            kind = e.get("kv_cache_spec_kind")
            if e["type"] == "BlockStored" and kind and kind != "full_attention" and not args.keep_mamba:
                skipped_kind += 1
                continue
            hs = e.get("block_hashes") or []
            if not hs:
                continue
            if e["type"] == "BlockStored":
                tks = e.get("token_ids") or []
                bs = e.get("block_size") or 16
                for i, h in enumerate(hs):
                    toks = tks[i * bs:(i + 1) * bs] if len(tks) == len(hs) * bs else tks
                    life = open_life.get(h)
                    if life is None:
                        # vLLM 的 BlockStored 事件里 parent_block_hash 只描述**批内第一个块**；
                        # 批内后续块的父块是前一个 hash。之前统一用 event 的 parent 会导致
                        # 位置（pos0 = 深度 × block_size）全部算错。
                        parent = (e.get("parent_block_hash") if i == 0 else hs[i - 1])
                        open_life[h] = {
                            "key": h, "tokens": toks, "block_size": bs,
                            "t_create": e["ts"], "t_last_access": e["ts"],
                            "t_death": None, "censored": True, "n_use": 1,
                            "spec_kind": kind or "",
                            "parent": parent,
                        }
                    else:
                        life["t_last_access"] = max(life["t_last_access"], e["ts"])
                        life["n_use"] += 1
            elif e["type"] == "BlockRemoved":
                t = e["ts"]
                for h in hs:
                    close(h, t)

    for h, life in list(open_life.items()):
        rows.append(life)

    # ---- 用 parent 链还原块在请求里的位置（= 链深度 × block_size）----
    # 为什么不能靠 token 内容配对：图像 token 全是同一个 <|image_pad|> id，
    # 528 长的纯图像块在不同请求里 token 完全相同，token 配对必然撞车。
    parent_of = {r["key"]: r.get("parent") for r in rows}
    depth_cache: dict[str, int] = {}

    def depth_of(h: str, guard: int = 0) -> int:
        if h in depth_cache:
            return depth_cache[h]
        if guard > 256:
            return -1
        p = parent_of.get(h)
        if p is None or p not in parent_of:
            depth_cache[h] = 0
            return 0
        depth_cache[h] = depth_of(p, guard + 1) + 1
        return depth_cache[h]

    for r in rows:
        d = depth_of(r["key"])
        r["depth"] = d
        r["pos0"] = d * r["block_size"] if d >= 0 else -1
        r.pop("parent", None)

    out = os.path.join(td, args.out)
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    rows.sort(key=lambda r: r["t_create"])
    n_cens = sum(1 for r in rows if r["censored"])
    print(f"lives={len(rows)} evicted={len(rows)-n_cens} censored={n_cens} "
          f"skipped_non_full_attention_events={skipped_kind} -> {out}")


if __name__ == "__main__":
    main()
