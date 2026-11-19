"""调试：把真实 KV 事件原样打出来，看字段长什么样。"""

from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _setup import add_src_paths  # noqa: E402
add_src_paths()

import asyncio
import json
import os
import sys
import time
from mm_input import MMInput  # noqa: E402
from run_trace import prepare_request  # noqa: E402
from workload import WorkloadConfig, build_sessions  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(ROOT, "models", "Qwen3.5-9B")


def main():
    from engines import AsyncVLLMEngine, KVEventSubscriber

    mm = MMInput(MODEL_DIR)
    from run_trace import pick_free_port

    port = pick_free_port()
    print(f"[events] endpoint = tcp://*:{port}")
    sub = KVEventSubscriber(f"tcp://127.0.0.1:{port}")
    sub.start()

    eng = AsyncVLLMEngine(MODEL_DIR, kv_cache_memory_bytes=int(1 * (1 << 30)),
                          max_model_len=16384, gpu_memory_utilization=0.70, max_num_seqs=8,
                          event_endpoint=f"tcp://*:{port}")

    cfg = WorkloadConfig(n_sessions=1, turns_per_session=1, arrival_span_s=0,
                         think_mean_s=0, seed=1)
    s = build_sessions(cfg)[0]
    _, ids, imgs, _, _ = prepare_request(mm, s, 0, [])

    async def go():
        await eng.generate(ids, imgs, max_tokens=8, request_id="d1")
        await eng.generate(ids, imgs, max_tokens=8, request_id="d2")
        await eng.shutdown()

    asyncio.run(go())
    time.sleep(1.5)
    sub.stop()

    with open(os.path.join(ROOT, "data", "debug_events.json"), "w") as f:
        json.dump([{"ts": ts, "ev": (ev if isinstance(ev, dict) else str(ev))}
                   for ts, ev in sub.records], f, ensure_ascii=False, indent=1, default=str)
    print("n events:", len(sub.records))
    seen = set()
    for ts, ev in sub.records:
        if not isinstance(ev, dict):
            print("non-dict:", type(ev)); continue
        key = (ev.get("type"), ev.get("kv_cache_spec_kind"), ev.get("block_size"))
        if key in seen:
            continue
        seen.add(key)
        print("=" * 70)
        print("keys:", sorted(ev.keys()))
        for k, v in ev.items():
            sv = str(v)
            print(f"  {k}: {sv[:120]}")


if __name__ == "__main__":
    main()
