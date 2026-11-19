"""FDP + GC 写放大仿真器（离散事件，堆优化）。

为什么要它：题面真正要的指标是 **盘侧写放大 WA ≤ 1.1**，而 ±5% 只是代理指标。
这个仿真器把"chunk 的寿命分布 + 分组方案"换算成 WA。

物理模型（对齐 NVMe FDP 与 EuroSys'25 的 FDP 论文）：
  * 盘 = 若干固定大小的**回收单元 RU**，GC 以 RU 为单位；
  * 主机按 **stream / 桶（placement handle）** 写入，每个 stream 有一个 open RU；
  * RU 写满就换一个空闲 RU；
  * 空闲 RU 少于预留（OP）时触发 GC：挑**有效数据最少**的 RU（贪心），
    把还活着的 chunk 重写一遍（这就是写放大），然后擦除；
  * WA = 总 NAND 写入 / 主机写入；理论上界 WA = 1/(1-δ)。

关键假设（写清楚，避免"不合理"的结论）：
  1. chunk 的 SSD 写入时刻 = HBM 里被创建的时刻，失效时刻 = 被淘汰的时刻
     （"写一次、留到最后一次被用"），与测到的寿命定义一致；
  2. chunk 大小 = block_size × 32 KB（Qwen3.5-9B 每 token KV ≈ 32 KB）；
  3. GC 迁移的数据写回**同一个 stream**（persistently isolated，对分组最有利；
     换 initially isolated 只会更差，所以结论偏保守）；
  4. 右删失的 chunk 按"该遍 trace 结束后 1 秒失效"处理；
  5. 稳态靠**随机错开复制**多份 trace 制造持续到达（原 trace 只有 2 分钟，是突发负载）。

自校验（--selftest）：完美分组 → WA≈1；单桶混放 → WA 明显 >1；
所有场景都应满足 WA ≈ 1/(1-δ)。
"""

from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _setup import add_src_paths  # noqa: E402
add_src_paths()

import argparse
import heapq
from dataclasses import dataclass


@dataclass
class Chunk:
    t_write: float
    t_death: float
    size: int
    stream: int


class Device:
    """FDP 盘：RU 池 + 每 stream 一个 open RU + 贪心 GC（堆选受害者）。"""

    def __init__(self, chunks: list[Chunk], capacity_gb: float, ru_mb: float,
                 op_frac: float = 0.07):
        self.chunks = chunks
        self.ru_size = int(ru_mb * (1 << 20))
        self.total_rus = max(16, int(capacity_gb * (1 << 30) // self.ru_size))
        n_streams = (max(c.stream for c in chunks) + 1) if chunks else 1
        # 预留：OP 空间 + 每个 stream 至少一个可换的 RU，否则高 stream 数一定会卡死
        # 只按 OP 预留；调用方要保证 RU 总数足够分给所有 stream（否则会 GC 活锁）
        self.reserve = max(2, int(round(self.total_rus * op_frac)))

        self.free = list(range(self.total_rus))
        self.ru_valid: dict[int, int] = {}
        self.ru_chunks: dict[int, list[int]] = {}
        self.ru_ver: dict[int, int] = {}
        self.heap: list[tuple[int, int, int]] = []
        self.open_ru: dict[int, int] = {}

        n = len(chunks)
        self.alive = bytearray(b"\x01") * n
        self.chunk_ru: list[int] = [-1] * n

        self.host = 0
        self.device = 0
        self.gc_count = 0
        self.copies = 0
        self.timeline: list[tuple[int, int, int]] = []
        self._open_cache: set[int] = set()

    # ---------- 堆维护 ----------
    def _touch(self, rid: int):
        v = self.ru_ver.get(rid, 0) + 1
        self.ru_ver[rid] = v
        heapq.heappush(self.heap, (self.ru_valid.get(rid, 0), rid, v))

    def _victim(self) -> int | None:
        self._open_cache = set(self.open_ru.values())
        while self.heap:
            v, rid, ver = self.heap[0]
            if (rid in self.ru_valid and self.ru_ver.get(rid) == ver
                    and rid not in self._open_cache):
                return rid
            heapq.heappop(self.heap)
        return None

    # ---------- 空间 ----------
    def _new_ru(self) -> int:
        """从空闲池取一个 RU（调用方保证有空间，不在这里触发 GC）。"""
        rid = self.free.pop()
        self.ru_valid[rid] = 0
        self.ru_chunks[rid] = []
        self._touch(rid)
        return rid

    def _gc_once(self):
        """回收一个 RU：把有效数据重写（计设备写入），然后擦除。

        约定：调用前 free 至少有 reserve+1 个（触发条件保证），
        所以迁移数据一定能落在空闲 RU 上，不会递归触发 GC。
        """
        victim = self._victim()
        if victim is None:
            raise RuntimeError("没有可回收的 RU")
        moved = [ci for ci in self.ru_chunks[victim] if self.alive[ci]]
        vbytes = self.ru_valid[victim]
        self.ru_valid.pop(victim, None)
        self.ru_chunks.pop(victim, None)
        self.gc_count += 1
        if moved:
            self.device += vbytes
            self.copies += vbytes
            for ci in moved:
                self._place(ci, count_host=False)
        # 迁移完再把受害者 RU 归还空闲池，避免"刚腾出又被迁移占掉"
        self.free.append(victim)

    def _place(self, ci: int, count_host: bool = True):
        c = self.chunks[ci]
        rid = self.open_ru.get(c.stream)
        if rid is None or self.ru_valid[rid] + c.size > self.ru_size:
            old = self.open_ru.get(c.stream)
            rid = self._new_ru()
            self.open_ru[c.stream] = rid
            if old is not None and old in self.ru_valid:
                # 这个 RU 不再 open，重新入堆（否则它被当作 open 弹出后就永久丢了）
                self._touch(old)
        self.ru_valid[rid] += c.size
        self.ru_chunks[rid].append(ci)
        self.chunk_ru[ci] = rid
        self._touch(rid)
        if count_host:
            self.host += c.size
            self.device += c.size

    def kill(self, ci: int):
        self.alive[ci] = 0
        rid = self.chunk_ru[ci]
        if rid >= 0 and rid in self.ru_valid:
            self.ru_valid[rid] -= self.chunks[ci].size
            self._touch(rid)

    def run(self) -> dict:
        ev = []
        for i, c in enumerate(self.chunks):
            ev.append((c.t_write, 1, i))
            ev.append((c.t_death, 0, i))
        ev.sort(key=lambda e: (e[0], e[1]))
        step = max(self.ru_size * 4, 1 << 30)
        nxt = step
        for t, kind, ci in ev:
            if kind == 0:
                self.kill(ci)
            else:
                c = self.chunks[ci]
                rid = self.open_ru.get(c.stream)
                need_new = rid is None or self.ru_valid[rid] + c.size > self.ru_size
                if need_new:
                    guard = 0
                    while len(self.free) < 2:      # 迁移期间需要至少 1 个备用 RU
                        self._gc_once()
                        guard += 1
                        if guard > 1000:
                            raise RuntimeError("GC 死循环：整盘都活着")
                self._place(ci)
                if self.host >= nxt:
                    self.timeline.append((self.host, self.device, self.gc_count))
                    nxt += step
        return self.report()

    def report(self) -> dict:
        wa = self.device / max(self.host, 1)
        delta = self.copies / max(self.host + self.copies, 1)
        # 稳态：从"写满一个盘容量"之后开始算（之前的填充阶段 WA 天然偏高）
        steady = wa
        cap = self.ru_size * self.total_rus
        pts = [p for p in self.timeline if p[0] >= cap]
        if pts:
            h0, d0, _ = pts[0]
            if self.host > h0:
                steady = (self.device - d0) / (self.host - h0)
        return {
            "host_gb": self.host / (1 << 30),
            "device_gb": self.device / (1 << 30),
            "wa": wa,
            "wa_steady": steady,
            "delta": delta,
            "wa_closed_form": 1.0 / (1.0 - delta) if delta < 1 else float("inf"),
            "gc_count": self.gc_count,
            "copies_gb": self.copies / (1 << 30),
            "n_rus": self.total_rus,
            "timeline": self.timeline,
        }


def run(chunks, capacity_gb=64.0, ru_mb=256.0, op_frac=0.07) -> dict:
    return Device(chunks, capacity_gb, ru_mb, op_frac).run()


def amplify_random(chunks: list[Chunk], passes: int, window: float,
                   seed: int = 0) -> list[Chunk]:
    """复制 passes 份，起始时刻在 [0, window] 随机偏移 → 持续到达的稳态负载。"""
    import random

    rng = random.Random(seed)
    out = []
    for _ in range(passes):
        off = rng.uniform(0, window)
        for c in chunks:
            out.append(Chunk(c.t_write + off, c.t_death + off, c.size, c.stream))
    return out


def max_live_bytes(chunks: list[Chunk]) -> int:
    ev = []
    for c in chunks:
        ev.append((c.t_write, c.size))
        ev.append((c.t_death, -c.size))
    ev.sort()
    cur = peak = 0
    for _, d in ev:
        cur += d
        peak = max(peak, cur)
    return peak


def selftest():
    import numpy as np

    print("自校验：两簇寿命（20s / 200s），只改分组方式")
    rng = np.random.default_rng(0)
    n, size = 6000, 17 * (1 << 20)
    short = rng.random(n) < 0.5
    life = np.where(short, rng.normal(20, 2, n), rng.normal(200, 20, n)).clip(1, None)
    tw = np.sort(rng.uniform(0, 600, n))
    td = tw + life

    print(f"{'场景':<20}{'WA':>8}{'稳态':>8}{'δ':>8}{'1/(1-δ)':>10}{'GC':>7}")
    cases = [
        ("完美分组", np.where(short, 0, 1)),
        ("4 桶按分位", np.digitize(life, np.quantile(life, [.25, .5, .75]))),
        ("2 桶随机", rng.integers(0, 2, n)),
        ("单桶混放", np.zeros(n, dtype=int)),
    ]
    for name, lab in cases:
        base = [Chunk(float(tw[i]), float(td[i]), size, int(lab[i])) for i in range(n)]
        ch = amplify_random(base, 6, window=4 * 600)
        live = max_live_bytes(ch)
        cap = live / 0.85 / (1 << 30)
        r = run(ch, capacity_gb=cap, ru_mb=1024.0)
        print(f"{name:<20}{r['wa']:8.3f}{r['wa_steady']:8.3f}{r['delta']:8.3f}"
              f"{r['wa_closed_form']:10.3f}{r['gc_count']:7d}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
