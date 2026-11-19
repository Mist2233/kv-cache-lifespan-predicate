"""冒烟测试：模型能不能加载、多模态能不能跑、block_size 是多少、KV 事件有没有。

先跑这个，再跑完整 trace——省得一个 20 分钟的实验挂在第一步。

    python src/smoke_test.py
"""

from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _setup import add_src_paths  # noqa: E402
add_src_paths()

import os
import sys
import time
from mm_input import MMInput  # noqa: E402
from workload import WorkloadConfig, build_sessions, build_messages, image_fingerprint  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(ROOT, "models", "Qwen3.5-9B")


def main():
    import asyncio

    from engines import AsyncVLLMEngine, KVEventSubscriber

    mm = MMInput(MODEL_DIR)
    from run_trace import pick_free_port

    port = pick_free_port()
    print(f"[events] endpoint = tcp://*:{port}")
    sub = KVEventSubscriber(f"tcp://127.0.0.1:{port}")
    sub.start()
    time.sleep(0.5)

    t0 = time.time()
    eng = AsyncVLLMEngine(MODEL_DIR, kv_cache_memory_bytes=int(2 * (1 << 30)),
                          max_model_len=16384, gpu_memory_utilization=0.75,
                          max_num_seqs=8, event_endpoint=f"tcp://*:{port}")
    print(f"[load] {time.time()-t0:.1f}s")

    cfg = WorkloadConfig(n_sessions=2, turns_per_session=2, arrival_span_s=0,
                         think_mean_s=0, seed=1)
    sessions = build_sessions(cfg)
    s = sessions[0]
    msgs, ids, imgs, spans, new_img = None, None, None, None, None
    from run_trace import prepare_request

    msgs, ids, imgs, spans, new_img = prepare_request(mm, s, 0, [])
    print(f"[input] prompt={len(ids)} tokens, images={len(imgs)}, image_spans={spans[:1]}")

    async def go():
        r = await eng.generate(ids, imgs, max_tokens=48, request_id="smoke1")
        print(f"[gen] {r.gen_ids[:8]}... len={len(r.gen_ids)} cached={r.num_cached_tokens} "
              f"dt={r.t_end-r.t_start:.2f}s")
        print("[text]", r.text[:200].replace("\n", " "))
        # 第二次相同前缀，检查 prefix cache 是否命中
        r2 = await eng.generate(ids, imgs, max_tokens=16, request_id="smoke2")
        print(f"[gen2-same-prefix] cached={r2.num_cached_tokens} / prompt={len(ids)}")
        await eng.shutdown()

    asyncio.run(go())
    time.sleep(1.5)
    sub.stop()
    kinds: dict[str, int] = {}
    for ts, ev in sub.records:
        if isinstance(ev, dict):
            k = f"{ev.get('type')}/kind={ev.get('kv_cache_spec_kind')}/bs={ev.get('block_size')}"
        else:
            k = str(type(ev).__name__) + "/" + str(getattr(ev, "kv_cache_spec_kind", None))
        kinds[k] = kinds.get(k, 0) + 1
    print("[kv events]", kinds)
    for ts, ev in sub.records[:3]:
        if isinstance(ev, dict):
            print("[event]", ev.get("type"), "bs=", ev.get("block_size"),
                  "n_hashes=", len(ev.get("block_hashes") or []),
                  "n_tokens=", len(ev.get("token_ids") or []),
                  "kind=", ev.get("kv_cache_spec_kind"), "group=", ev.get("group_idx"))


if __name__ == "__main__":
    main()
