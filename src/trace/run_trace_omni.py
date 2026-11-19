"""用 Qwen2.5-Omni 跑"图像 + 语音"多轮推理，采集 KV 事件（块级生命周期 trace）。

数据：`data/openomni_300/sessions.json`
  - 每个会话若干轮；每轮 = 一张 COCO 图 + 一段 TTS 合成的语音提问 + 文本回答。

与 `run_trace.py` 的区别：
  - 模型换成 Qwen2.5-Omni（支持音频输入）；
  - 输入为 **chat template 文本 + multi_modal_data（audio/image）**，而不是 token ids；
  - 记录 vLLM 展开后的完整 prompt token 序列，并标注图像/音频 token 的区间，
    供后续 `derive_blocks.py` / `attach()` 使用。

用法：
  python src/run_trace_omni.py --out data/trace_openomni --sessions 100 --kv-gb 4
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

import numpy as np
from engines import AsyncVLLMEngine, KVEventSubscriber  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(ROOT, "models", "Qwen2.5-Omni-7B")
MAX_GEN_TOKENS = 160


def load_wav(path: str, target_sr: int = 16000) -> np.ndarray:
    import wave
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        n = w.getnframes()
        data = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32) / 32768.0
    if sr != target_sr:      # 简单重采样（线性插值）
        idx = np.linspace(0, len(data) - 1, int(len(data) * target_sr / sr))
        data = np.interp(idx, np.arange(len(data)), data).astype(np.float32)
    return data


def image_fp(path: str) -> str:
    import hashlib
    return hashlib.md5(open(path, "rb").read()).hexdigest()[:16]


def pick_free_port(start: int = 5557, tries: int = 40) -> int:
    import socket
    for port in range(start, start + tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("0.0.0.0", port))
            except OSError:
                continue
            return port
    raise RuntimeError("找不到空闲端口")


def build_messages(session, turn, history, system_prompt):
    t = session["turns"][turn]
    content = []
    content.append({"type": "image", "image": t["image"]})
    content.append({"type": "audio", "audio": t["audio"]})
    content.append({"type": "text",
                    "text": "Please answer the question in the audio about this image."})
    return ([{"role": "system", "content": system_prompt}] + history
            + [{"role": "user", "content": content}])


def collect_media(messages):
    """收集整段对话里出现的全部图像与音频（多轮时历史里的媒体也要重新送入）。"""
    imgs, auds = [], []
    for m in messages:
        c = m.get("content")
        if isinstance(c, list):
            for item in c:
                if item.get("type") == "image":
                    imgs.append(item["image"])
                elif item.get("type") == "audio":
                    auds.append(item["audio"])
    return imgs, auds


def find_spans(token_ids, tok, bos_name, eos_name):
    """在展开后的 token 序列里找 [bos, eos] 区间。"""
    bos = tok.convert_tokens_to_ids(bos_name)
    eos = tok.convert_tokens_to_ids(eos_name)
    spans, start = [], None
    for i, t in enumerate(token_ids):
        if t == bos:
            start = i
        elif t == eos and start is not None:
            spans.append([start, i + 1])
            start = None
    return spans


async def run(args):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    from prompts import SYSTEM_PROMPT

    sessions = json.load(open(args.sessions_file))
    sessions = sessions[: args.sessions]
    print(f"[data] {len(sessions)} 个会话，"
          f"共 {sum(len(s['turns']) for s in sessions)} 轮")

    out_dir = os.path.join(ROOT, args.out)
    os.makedirs(out_dir, exist_ok=True)

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
        event_endpoint=f"tcp://*:{port}",
        block_size=args.block_size,
    )

    t_origin = time.time()
    req_f = open(os.path.join(out_dir, "requests.jsonl"), "w")
    errors = []
    n_done = 0

    async def run_session(s):
        nonlocal n_done
        history = []
        wait = s["arrive_at"] - (time.time() - t_origin)
        if wait > 0:
            await asyncio.sleep(wait)
        for turn in range(len(s["turns"])):
            try:
                msgs = build_messages(s, turn, history, SYSTEM_PROMPT)
                prompt = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                                 tokenize=False)
                img_paths, aud_paths = collect_media(msgs)
                from PIL import Image
                imgs = [Image.open(p).convert("RGB") for p in img_paths]
                auds = [load_wav(p) for p in aud_paths]
                res = await engine.generate_mm(
                    prompt, images=imgs, audios=auds,
                    max_tokens=MAX_GEN_TOKENS,
                    request_id=f"s{s['sid']}t{turn}")
                reply = (res.text or "").strip()
                pids = res.prompt_ids or tok(prompt).input_ids
                img_spans = find_spans(pids, tok, "<|vision_bos|>", "<|vision_eos|>")
                aud_spans = find_spans(pids, tok, "<|audio_bos|>", "<|audio_eos|>")
                rec = {
                    "sid": s["sid"], "turn": turn,
                    "t_start": res.t_start - t_origin,
                    "t_end": res.t_end - t_origin,
                    "prompt_len": len(pids),
                    "gen_len": len(res.gen_ids),
                    "num_cached_tokens": res.num_cached_tokens,
                    "has_image": len(img_paths) > 0,
                    "n_images": len(img_paths),
                    "n_audios": len(aud_paths),
                    "image_fp": image_fp(img_paths[-1]) if img_paths else "",
                    "image_spans": [[a, b, image_fp(img_paths[min(k, len(img_paths) - 1)])]
                                    for k, (a, b) in enumerate(img_spans)],
                    "audio_spans": aud_spans,
                    "prompt_token_ids": pids,
                }
                req_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                req_f.flush()
                n_done += 1
                history.append({"role": "user", "content": msgs[-1]["content"]})
                history.append({"role": "assistant", "content": reply or "（无内容）"})
                # 轮间"用户思考时间"：与 run_trace.py 保持一致（之前漏掉了，导致会话过于紧凑）
                if turn < len(s.get("think_times") or []):
                    await asyncio.sleep(s["think_times"][turn])
            except Exception as e:  # noqa: BLE001
                import traceback
                errors.append(f"sid={s['sid']} turn={turn}: {type(e).__name__}: {e}")
                print(f"[error] sid={s['sid']} turn={turn}: {e}")
                traceback.print_exc(limit=2)

    await asyncio.gather(*(run_session(s) for s in sessions))
    req_f.close()
    await engine.shutdown()
    sub.stop()

    with open(os.path.join(out_dir, "kv_events.jsonl"), "w") as f:
        import msgspec
        for ts, batch in sub.records:
            ev = msgspec.to_builtins(batch)
            d = {"ts": ts - t_origin, "type": ev.get("type", "?")}
            for k in ("block_hashes", "parent_block_hash", "token_ids", "block_size",
                      "medium", "group_idx", "kv_cache_spec_kind", "session_id"):
                if k in ev:
                    d[k] = ev[k]
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    meta = {"engine": "vllm-omni", "model": args.model, "n_sessions": len(sessions),
            "n_requests": n_done, "kv_gb": args.kv_gb,
            "max_model_len": args.max_model_len, "errors": errors,
            "wall_s": time.time() - t_origin}
    json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w"),
              ensure_ascii=False, indent=1)
    print(f"[done] {n_done} requests, wall={time.time()-t_origin:.1f}s, errors={len(errors)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_DIR)
    ap.add_argument("--out", default="data/trace_openomni")
    ap.add_argument("--sessions", type=int, default=100)
    ap.add_argument("--sessions-file",
                    default=os.path.join(ROOT, "data/openomni_300/sessions.json"))
    ap.add_argument("--kv-gb", type=float, default=4.0)
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--gpu-util", type=float, default=0.8)
    ap.add_argument("--max-num-seqs", type=int, default=32)
    ap.add_argument("--event-port", type=int, default=5557)
    ap.add_argument("--block-size", type=int, default=None,
                    help="vLLM KV cache 块大小；调大可让块粒度变粗")
    args = ap.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
