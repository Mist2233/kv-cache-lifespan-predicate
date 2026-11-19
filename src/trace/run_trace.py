"""跑真实多轮多模态推理，同时采集 KV 事件（block 级生命周期 trace）。

用法：
    python src/run_trace.py --engine stub --out data/trace_stub
    python src/run_trace.py --engine vllm --out data/trace_v1 --kv-gb 2

输出：
    <out>/requests.jsonl   每条请求：轮次、起止时刻、prompt token 序列、生成长度、前缀命中
    <out>/kv_events.jsonl  每个 KV 事件：时间戳、BlockStored/BlockRemoved、block hash、token_ids
    <out>/meta.json        引擎/模型/容量等运行配置
"""

from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _setup import add_src_paths  # noqa: E402
add_src_paths()

import argparse
import asyncio
import json
import os
import sys
import time
from mm_input import MMInput            # noqa: E402
from workload import WorkloadConfig, build_sessions, build_messages, image_fingerprint  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(ROOT, "models", "Qwen3.5-9B")

MAX_GEN_TOKENS = 160


def pick_free_port(start: int = 5557, tries: int = 40) -> int:
    """挑一个空闲端口做 KV 事件通道（残留的 EngineCore 会占着旧端口）。"""
    import socket

    for port in range(start, start + tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("0.0.0.0", port))
            except OSError:
                continue
            return port
    raise RuntimeError("找不到空闲端口")


def prepare_request(mm, s, turn, history):
    if getattr(s, "raw", None) is not None:
        tinfo = s.raw["turns"][turn]
        if s.raw.get("real_len"):
            from workload_docvqa_real import _pil, build_messages_docvqa_real
            msgs = build_messages_docvqa_real(s.raw, turn, history)
            new_img = _pil(tinfo["doc_id"]) if tinfo.get("img_bytes") else None
        else:
            from workload_docvqa import build_messages_docvqa
            msgs = build_messages_docvqa(s.raw, turn, history)
            new_img = tinfo["images"][0] if tinfo["images"] else None
    else:
        msgs, new_img = build_messages(s, turn, history)
    tk = mm.tokenize(msgs, add_generation_prompt=True)
    ids = tk["input_ids"][0]
    all_imgs = []
    for m in msgs:
        if isinstance(m.get("content"), list):
            for c in m["content"]:
                if isinstance(c, dict) and c.get("type") == "image":
                    all_imgs.append(c["image"])
    segs = mm.segment(ids, n_turns=turn + 1)
    img_segs = [sg for sg in segs if sg.kind == "image"]
    image_spans = [[sg.start, sg.end, image_fingerprint(img)]
                   for sg, img in zip(img_segs, all_imgs)]
    return msgs, ids, all_imgs, image_spans, new_img


async def run(args):
    mm = MMInput(args.model)
    cfg = WorkloadConfig(
        n_sessions=args.sessions,
        turns_per_session=args.turns,
        arrival_span_s=args.arrival_span,
        think_mean_s=args.think_mean,
        seed=args.seed,
    )
    if args.dataset in ("docvqa", "docvqa_real"):
        from types import SimpleNamespace

        if args.dataset == "docvqa_real":
            from workload_docvqa_real import load_sessions, schedule
            raw = load_sessions(n_sessions=args.sessions, seed=args.seed)
            for d in raw:
                d["real_len"] = True
        else:
            from workload_docvqa import load_sessions, schedule
            raw = load_sessions(n_sessions=args.sessions, seed=args.seed)
        schedule(raw, arrival_span=args.arrival_span, think_mean=args.think_mean,
                 seed=args.seed)
        sessions = []
        for d in raw:
            sessions.append(SimpleNamespace(sid=d["sid"], arrive_at=d["arrive_at"],
                                            turns=len(d["turns"]),
                                            think_times=d["think_times"], raw=d))
        cfg = SimpleNamespace(n_sessions=len(sessions), turns_per_session=-1,
                              arrival_span_s=args.arrival_span,
                              think_mean_s=args.think_mean, seed=args.seed)
    else:
        sessions = build_sessions(cfg)
    out_dir = os.path.join(ROOT, args.out)
    os.makedirs(out_dir, exist_ok=True)

    sub = None
    if args.engine == "vllm":
        from engines import AsyncVLLMEngine, KVEventSubscriber

        port = pick_free_port(args.event_port)
        print(f"[events] endpoint = tcp://*:{port}")
        sub = KVEventSubscriber(f"tcp://127.0.0.1:{port}")
        sub.start()
        time.sleep(0.5)
        engine = AsyncVLLMEngine(
            args.model,
            kv_cache_memory_bytes=int(args.kv_gb * (1 << 30)) if args.kv_gb else None,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_util,
            max_num_seqs=args.max_num_seqs,
            enforce_eager=args.enforce_eager,
            event_endpoint=f"tcp://*:{port}",
        )
        engine_name = "vllm"
    else:
        from engines import StubEngine

        engine = StubEngine(mm, reply_tokens=args.stub_reply_tokens)
        engine_name = "stub"

    t_origin = time.time()
    req_f = open(os.path.join(out_dir, "requests.jsonl"), "w")
    errors: list[str] = []
    n_done = 0

    async def run_session(s):
        nonlocal n_done
        history: list[dict] = []
        wait = s.arrive_at - (time.time() - t_origin)
        if wait > 0:
            await asyncio.sleep(wait)
        for turn in range(s.turns):
            try:
                msgs, ids, all_imgs, image_spans, new_img = prepare_request(mm, s, turn, history)
                res = await engine.generate(
                    ids, all_imgs or None, max_tokens=MAX_GEN_TOKENS,
                    request_id=f"s{s.sid}t{turn}")
                reply = (res.text or "").strip()
                if not reply and res.gen_ids:
                    reply = mm.tokenizer.decode(res.gen_ids, skip_special_tokens=True).strip()
                if not reply:
                    reply = "（无内容）"
                rec = {
                    "sid": s.sid,
                    "turn": turn,
                    "t_start": res.t_start - t_origin,
                    "t_end": res.t_end - t_origin,
                    "prompt_len": len(ids),
                    "gen_len": len(res.gen_ids),
                    "num_cached_tokens": res.num_cached_tokens,
                    "has_image": new_img is not None,
                    "n_images": len(all_imgs),
                    "image_fp": image_fingerprint(new_img) if new_img is not None else "",
                    "image_spans": image_spans,
                    "prompt_token_ids": ids,
                }
                req_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                req_f.flush()
                n_done += 1
                history.append({"role": "user", "content": msgs[-1]["content"]})
                history.append({"role": "assistant", "content": reply})
            except Exception as e:  # noqa: BLE001
                import traceback

                errors.append(f"sid={s.sid} turn={turn}: {type(e).__name__}: {e}")
                traceback.print_exc()
                return
            if turn < s.turns - 1:
                await asyncio.sleep(s.think_times[turn])

    t0 = time.time()
    await asyncio.gather(*[run_session(s) for s in sessions])
    wall = time.time() - t0
    req_f.close()

    if sub is not None:
        await asyncio.sleep(1.5)
        sub.stop()
        n_store = n_remove = 0
        with open(os.path.join(out_dir, "kv_events.jsonl"), "w") as f:
            for ts, ev in sub.records:
                if isinstance(ev, dict):
                    d = {"ts": ts - t_origin, "type": ev.get("type", "?")}
                    for k in ("block_hashes", "parent_block_hash", "token_ids", "block_size",
                              "medium", "group_idx", "kv_cache_spec_kind", "session_id"):
                        if k in ev:
                            d[k] = ev[k]
                else:
                    d = {"ts": ts - t_origin, "type": type(ev).__name__}
                    for k in ("block_hashes", "parent_block_hash", "token_ids", "block_size",
                              "medium", "group_idx", "kv_cache_spec_kind", "session_id"):
                        if hasattr(ev, k):
                            d[k] = getattr(ev, k)
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
                if d["type"] == "BlockStored":
                    n_store += 1
                elif d["type"] == "BlockRemoved":
                    n_remove += 1
        print(f"[events] stored={n_store} removed={n_remove}")

    await engine.shutdown()

    meta = {
        "engine": engine_name,
        "model": args.model,
        "workload": getattr(cfg, "__dict__", {}),
        "dataset": args.dataset,
        "kv_gb": args.kv_gb,
        "max_model_len": args.max_model_len,
        "gpu_util": args.gpu_util,
        "max_gen_tokens": MAX_GEN_TOKENS,
        "wall_s": wall,
        "n_requests": n_done,
        "errors": errors,
        "ts_origin_wall": t_origin,
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[done] {n_done} requests, wall={wall:.1f}s, errors={len(errors)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["vllm", "stub"], default="stub")
    ap.add_argument("--model", default=MODEL_DIR)
    ap.add_argument("--out", default="data/trace_stub")
    ap.add_argument("--sessions", type=int, default=16)
    ap.add_argument("--turns", type=int, default=6)
    ap.add_argument("--arrival-span", type=float, default=30.0)
    ap.add_argument("--think-mean", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kv-gb", type=float, default=2.0)
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--gpu-util", type=float, default=0.75)
    ap.add_argument("--max-num-seqs", type=int, default=32)
    ap.add_argument("--stub-reply-tokens", type=int, default=140)
    ap.add_argument("--dataset", choices=["synthetic", "docvqa", "docvqa_real"],
                    default="synthetic")
    ap.add_argument("--enforce-eager", action="store_true")
    ap.add_argument("--event-port", type=int, default=5557)
    args = ap.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
