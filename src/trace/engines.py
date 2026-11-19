"""推理引擎封装（async）：真实 vLLM / 桩引擎。

为什么用 async：中心推理场景的核心是**并发会话**——多个用户的请求同时在飞，
前缀缓存命中、抢占、淘汰都由这种交错决定。同步 LLM + 线程会退化成串行，
拿不到真实的块生命周期。
"""

from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _setup import add_src_paths  # noqa: E402
add_src_paths()

import asyncio
import threading
import time
from dataclasses import dataclass


@dataclass
class GenResult:
    gen_ids: list[int]
    num_cached_tokens: int
    t_start: float
    t_end: float
    text: str = ""
    prompt_ids: list[int] | None = None   # vLLM 展开后的完整 prompt token 序列


class StubEngine:
    """假引擎：按固定长度返回"回复"，用于没有权重时验证整条分析管线。"""

    def __init__(self, mm, reply_tokens: int = 140):
        self.mm = mm
        self.reply_tokens = reply_tokens
        self._eos = mm.tokenizer.eos_token_id

    async def generate(self, prompt_token_ids, images, max_tokens: int = 256,
                       request_id: str = "") -> GenResult:
        t0 = time.time()
        n = min(self.reply_tokens, max_tokens)
        await asyncio.sleep(0.002 * n)
        return GenResult([self._eos] * n, 0, t0, time.time(), "")

    async def shutdown(self):
        return


class AsyncVLLMEngine:
    """真实 vLLM 异步引擎。"""

    def __init__(self, model_dir: str, *, kv_cache_memory_bytes: int | None = None,
                 max_model_len: int = 16384, gpu_memory_utilization: float = 0.75,
                 enable_prefix_caching: bool = True, event_endpoint: str = "tcp://*:5557",
                 max_num_seqs: int = 32, enforce_eager: bool = False,
                 block_size: int | None = None):
        import os

        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        from vllm.config.kv_events import KVEventsConfig
        from vllm.v1.engine.async_llm import AsyncLLM

        try:
            from vllm.engine.arg_utils import AsyncEngineArgs as EngineArgs
        except Exception:  # noqa: BLE001
            from vllm.engine.arg_utils import EngineArgs

        self.kv_events_cfg = KVEventsConfig(
            enable_kv_cache_events=True,
            publisher="zmq",
            endpoint=event_endpoint,
            topic="",
            hwm=1_000_000,
            max_queue_size=1_000_000,
        )
        args = EngineArgs(
            model=model_dir,
            trust_remote_code=True,
            enable_prefix_caching=enable_prefix_caching,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            kv_cache_memory_bytes=kv_cache_memory_bytes,
            max_num_seqs=max_num_seqs,
            enforce_eager=enforce_eager,
            disable_log_stats=True,
            kv_events_config=self.kv_events_cfg,
            block_size=block_size,
        )
        if not hasattr(args, "enable_log_requests"):
            try:
                args.enable_log_requests = False
            except Exception:  # noqa: BLE001
                pass
        self.engine = AsyncLLM.from_engine_args(args)
        from vllm import SamplingParams

        self.SamplingParams = SamplingParams

    async def generate(self, prompt_token_ids, images, max_tokens: int = 256,
                       request_id: str = "req") -> GenResult:
        req = {"prompt_token_ids": list(prompt_token_ids)}
        if images:
            req["multi_modal_data"] = {"image": images}
        sp = self.SamplingParams(temperature=0.0, max_tokens=max_tokens, min_tokens=16)
        t0 = time.time()
        final = None
        async for out in self.engine.generate(req, sp, request_id):
            final = out
        t1 = time.time()
        if final is None:
            return GenResult([], 0, t0, t1, "")
        gen_ids = list(final.outputs[0].token_ids)
        cached = getattr(final, "num_cached_tokens", None) or 0
        return GenResult(gen_ids, int(cached), t0, t1, final.outputs[0].text)

    async def generate_mm(self, prompt: str, images=None, audios=None,
                          max_tokens: int = 256, request_id: str = "req") -> GenResult:
        """多模态输入（文本 prompt + 图像 + 音频），用于 Qwen2.5-Omni。"""
        req = {"prompt": prompt}
        mmd = {}
        if images:
            mmd["image"] = images
        if audios:
            mmd["audio"] = audios
        if mmd:
            req["multi_modal_data"] = mmd
        sp = self.SamplingParams(temperature=0.0, max_tokens=max_tokens, min_tokens=16)
        t0 = time.time()
        final = None
        async for out in self.engine.generate(req, sp, request_id):
            final = out
        t1 = time.time()
        if final is None:
            return GenResult([], 0, t0, t1, "")
        gen_ids = list(final.outputs[0].token_ids)
        cached = getattr(final, "num_cached_tokens", None) or 0
        return GenResult(gen_ids, int(cached), t0, t1, final.outputs[0].text,
                         list(getattr(final, "prompt_token_ids", None) or []))

    async def shutdown(self):
        try:
            await self.engine.shutdown()
        except Exception:  # noqa: BLE001
            pass


class KVEventSubscriber(threading.Thread):
    """订阅 vLLM 的 ZMQ KV 事件，落成 (ts, event) 列表。"""

    def __init__(self, endpoint: str):
        super().__init__(daemon=True)
        import zmq

        self.ctx = zmq.Context.instance()
        self.sock = self.ctx.socket(zmq.SUB)
        self.sock.setsockopt(zmq.SUBSCRIBE, b"")
        self.sock.setsockopt(zmq.RCVHWM, 0)
        self.sock.connect(endpoint)
        self.endpoint = endpoint
        self.records: list[tuple[float, object]] = []
        self._stop_event = threading.Event()

    def run(self):
        import msgspec
        from vllm.distributed.kv_events import EventBatch

        dec = msgspec.msgpack.Decoder(EventBatch)
        while not self._stop_event.is_set():
            try:
                if not self.sock.poll(150):
                    continue
                parts = self.sock.recv_multipart()
                batch = dec.decode(parts[-1])
                for ev in batch.events:
                    self.records.append((batch.ts, ev))
            except Exception:  # noqa: BLE001
                continue

    def stop(self):
        self._stop_event.set()
        self.join(timeout=2)
        try:
            self.sock.close(linger=0)
        except Exception:  # noqa: BLE001
            pass
