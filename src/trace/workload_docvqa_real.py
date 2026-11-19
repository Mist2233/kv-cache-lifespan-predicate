"""真实会话长度的 DocVQA workload。

与 `workload_docvqa.py` 的区别：
  * 会话轮数从 **WildChat 真实用户分布** 采样（中位 3，p90 8，长尾）；
  * 一个会话可以跨**多份文档**提问（延续：把上一份文档的问题问完，再换下一份）；
  * 图片以字节形式保存、按需转换为 PIL，避免上百张高清文档图同时驻留内存。

会话结构：
  turns[i] = {"user": 问题, "img_bytes": [当前文档的图] 或 [], "assistant": ""}
  同一文档只在"该文档的第一问"时附带图片，后续追问靠上下文。
"""

from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _setup import add_src_paths  # noqa: E402
add_src_paths()

import io
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PARQUET = os.path.join(ROOT, "data/DocVQA/data/train-00000-of-00001-f795527aa19f0f90.parquet")
LENS_NPY = os.path.join(ROOT, "data/WildChat/session_lengths.npy")

DOC_CACHE: dict[int, dict] = {}
_PIL_CACHE: dict[int, object] = {}


def _query(q) -> str:
    if isinstance(q, dict):
        return q.get("en") or q.get("de") or next(iter(q.values()))
    return str(q)


def _docs():
    """{doc_id: {"bytes": ..., "questions": [...]}}，按需加载一次。"""
    if DOC_CACHE:
        return DOC_CACHE
    import pandas as pd

    df = pd.read_parquet(PARQUET)
    groups: dict[int, list[int]] = {}
    for i in range(len(df)):
        groups.setdefault(hash(df.iloc[i]["image"]["bytes"]), []).append(i)
    for did, (h, idx) in enumerate(groups.items()):
        DOC_CACHE[did] = {
            "bytes": df.iloc[idx[0]]["image"]["bytes"],
            "questions": [_query(df.iloc[i]["query"]) for i in idx],
        }
    return DOC_CACHE


def load_session_lengths() -> np.ndarray:
    return np.load(LENS_NPY)


def load_sessions(n_sessions: int = 200, seed: int = 0, cap: int = 12,
                  max_docs: int = 3):
    """按真实会话长度分布生成会话。"""
    docs = _docs()
    n_docs = len(docs)
    lens = load_session_lengths()
    lens = np.clip(lens, 1, cap)
    rng = np.random.default_rng(seed)

    sessions = []
    for sid in range(n_sessions):
        L = int(rng.choice(lens))
        # 依次取文档，直到问题数够 L 轮
        turns, used, n_used_docs = [], 0, 0
        pool = []          # [(问题, doc_id, 是否该文档首次提问)]
        tried = 0
        while used < L and tried < 20 and n_used_docs < max_docs:
            did = int(rng.integers(0, n_docs))
            tried += 1
            d = docs[did]
            for k, q in enumerate(d["questions"][:max(L - used, 0)]):
                pool.append((q, did, k == 0))
            used += min(len(d["questions"]), L - used)
            n_used_docs += 1
        i = 0
        while pool and len(pool) < L:          # 问题不够则循环复用
            q, did, _ = pool[i % len(pool)]
            pool.append((q, did, False))
            i += 1
        for q, did, first in pool[:L]:
            turns.append({"user": q,
                          "img_bytes": [docs[did]["bytes"]] if first else [],
                          "assistant": "", "doc_id": did})
        if not turns:      # 兜底
            d = docs[int(rng.integers(0, n_docs))]
            turns = [{"user": d["questions"][0], "img_bytes": [d["bytes"]],
                      "assistant": "", "doc_id": 0}]
        sessions.append({"sid": sid, "turns": turns, "arrive_at": 0.0,
                         "think_times": [], "n_turns_target": L})
    return sessions


def schedule(sessions, arrival_span: float = 30.0, think_mean: float = 6.0, seed: int = 0):
    rng = np.random.default_rng(seed)
    for s in sessions:
        s["arrive_at"] = float(rng.uniform(0, arrival_span))
        s["think_times"] = list(rng.exponential(think_mean, max(len(s["turns"]) - 1, 0)))
    sessions.sort(key=lambda x: x["arrive_at"])
    return sessions


def _pil(doc_id: int):
    if doc_id not in _PIL_CACHE:
        from PIL import Image
        b = _docs()[doc_id]["bytes"]
        _PIL_CACHE[doc_id] = Image.open(io.BytesIO(b)).convert("RGB")
    return _PIL_CACHE[doc_id]


def build_messages_docvqa_real(session, turn: int, history: list[dict]):
    t = session["turns"][turn]
    content = [{"type": "image", "image": _pil(t["doc_id"])}
               for _ in t["img_bytes"]]
    content.append({"type": "text", "text": t["user"]})
    from prompts import SYSTEM_PROMPT
    return [{"role": "system", "content": SYSTEM_PROMPT}] + history + [
        {"role": "user", "content": content}]


if __name__ == "__main__":
    ss = load_sessions(n_sessions=20, seed=0)
    lens = [len(s["turns"]) for s in ss]
    print(f"20 个会话轮数: {lens}，中位 {np.median(lens):.0f}")
    for s in ss[:3]:
        print(f"  sid={s['sid']} turns={len(s['turns'])} "
              f"q1={s['turns'][0]['user'][:60]}")
