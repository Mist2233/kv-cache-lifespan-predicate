"""块级 prefix-cache 模拟器。

用途：在没有/不信任 vLLM 内部事件时，用**可外部观测的信息**（请求到达、
prompt token 序列、生成长度、结束时刻）重建每个 KV block 的生命周期。
它同时是后续 FDP/GC 写放大仿真的底座。

语义对齐 vLLM v1：
  * block_size = 16 tokens；
  * 前缀缓存以"链式哈希"（父块哈希 + 本块 token）为键，块边界从位置 0 起每 16 个 token；
  * 容量用满时按 LRU（最近未被访问）淘汰；
  * 请求运行期间每个 decode step 都会读自己的所有块 -> 请求结束时刻即最后一次访问；
  * 请求结束后块仍留在缓存里（可被后续请求命中），直到被淘汰。

与真实实现的两处已知差异（记录在 README）：
  1. 多模态图像的 "mm salt" 没有计入哈希（同一位置、同一占位符序列但不同图片，
     模拟器会当成同一个块）。实验里用 image_fp 做了修正：哈希里带上图片指纹。
  2. 不做抢占（preemption）与 swap，容量压力只体现为淘汰。
"""

from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _setup import add_src_paths  # noqa: E402
add_src_paths()

import hashlib
import json
import os
from collections import OrderedDict
from dataclasses import dataclass, field

BLOCK_SIZE = 16


@dataclass
class Block:
    key: str
    pos0: int
    tokens: list[int]
    t_create: float
    t_last_access: float
    t_death: float | None = None
    censored: bool = True
    n_use: int = 0


class PrefixCacheSim:
    def __init__(self, cap_blocks: int, block_size: int = BLOCK_SIZE):
        self.cap = cap_blocks
        self.bs = block_size
        self.table: dict[str, Block] = {}
        self.lru: OrderedDict[str, None] = OrderedDict()
        self._evicted: list[Block] = []

    def _key(self, parent: str, toks: list[int], salt: str = "") -> str:
        h = hashlib.blake2b(digest_size=16)
        h.update(parent.encode())
        h.update(salt.encode())
        h.update(json.dumps(toks).encode())
        return h.hexdigest()

    def _evict_one(self, t: float):
        while self.lru:
            k, _ = self.lru.popitem(last=False)
            b = self.table.pop(k, None)
            if b is not None:
                b.t_death = t
                b.censored = False
                self._evicted.append(b)
                return

    def _touch(self, k: str, t: float):
        self.lru.pop(k, None)
        self.lru[k] = None
        self.table[k].t_last_access = t

    def run(self, requests: list[dict]) -> list[Block]:
        """requests: 需要 sid/t_start/t_end/prompt_token_ids/gen_len，按 t_start 排序。"""
        reqs = sorted(requests, key=lambda r: r["t_start"])
        for r in reqs:
            self._process(r)

        # 实验结束时仍存活的块（右删失）
        blocks = list(self.table.values()) + self._evicted
        for b in list(self.table.values()):
            b.censored = True
            b.t_death = None
        return blocks

    def _process(self, r: dict):
        ids = r["prompt_token_ids"]
        gen_len = r.get("gen_len", 0)
        t_start = r["t_start"]
        t_end = r["t_end"]
        n_prompt = len(ids)
        seq = ids + [0] * gen_len
        n_blocks = len(seq) // self.bs

        # ---- 1) 先算出这条请求每个块的 (key, 就绪时刻, 是否含图像) ----
        entries = []
        parent = ""
        for d in range(n_blocks):
            pos0 = d * self.bs
            in_prompt = pos0 < n_prompt
            if in_prompt:
                toks = ids[pos0:min(pos0 + self.bs, n_prompt)]
                salt = ""
                for (s0, s1, fp) in r.get("image_spans", []) or []:
                    if s0 < pos0 + self.bs and s1 > pos0:  # 与图像区间有交叠
                        salt = fp
                        break
                key = self._key(parent, toks, salt)
                t_ready = t_start
            else:
                # 生成阶段的块：身份由 (sid,turn,d) 唯一确定，写完时刻按解码进度插值
                key = self._key(parent, [d], f"gen:{r['sid']}:{r['turn']}")
                n_gen_in_block = (d + 1) * self.bs - n_prompt
                frac = min(1.0, max(n_gen_in_block, 1) / max(gen_len, 1))
                t_ready = t_start + frac * (t_end - t_start)
            entries.append((key, pos0, toks if in_prompt else [], t_ready))
            parent = key

        # ---- 2) 前缀命中 / 新建（含 LRU 淘汰） ----
        for key, pos0, toks, t_ready in entries:
            b = self.table.get(key)
            if b is not None:
                self._touch(key, t_start)
                continue
            while len(self.table) >= self.cap:
                self._evict_one(t_ready)
            self.table[key] = Block(key, pos0, toks, t_ready, t_start)
            self.lru[key] = None

        # ---- 3) 请求结束：所有块刚被访问过；顺手记一次"被这条请求使用" ----
        for key, _, _, _ in entries:
            if key in self.table:
                self._touch(key, t_end)
                self.table[key].n_use += 1

def blocks_to_jsonl(blocks: list[Block], path: str, extra: dict | None = None):
    with open(path, "w") as f:
        for b in blocks:
            d = {
                "key": b.key,
                "pos0": b.pos0,
                "tokens": b.tokens,
                "t_create": b.t_create,
                "t_last_access": b.t_last_access,
                "t_death": b.t_death,
                "censored": b.censored,
                "n_use": b.n_use,
            }
            if extra:
                d.update(extra)
            f.write(json.dumps(d) + "\n")


def sim_from_trace(trace_dir: str, cap_blocks: int, bs: int = BLOCK_SIZE):
    reqs = [json.loads(l) for l in open(os.path.join(trace_dir, "requests.jsonl"))]
    sim = PrefixCacheSim(cap_blocks, block_size=bs)
    blocks = sim.run(reqs)
    out = os.path.join(trace_dir, "blocks_sim.jsonl")
    blocks_to_jsonl(blocks, out)
    return blocks, out


if __name__ == "__main__":
    import sys

    td = sys.argv[1]
    cap = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
    bs = int(sys.argv[3]) if len(sys.argv) > 3 else BLOCK_SIZE
    bl, out = sim_from_trace(td, cap, bs)
    n_c = sum(1 for b in bl if not b.censored)
    print(f"blocks={len(bl)} evicted={n_c} bs={bs} cap={cap} -> {out}")
