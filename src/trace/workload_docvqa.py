"""DocVQA 文档会话：真实文档图 + 真实问题 + 多轮追问。

为什么用它：
  * 图像是**真实文档扫描件**，中位 1706×2198 → 约 3600 个视觉 token，
    和题面"多模态使输入序列显著增长"的场景一致（对比 MMDU 的 220×177 小图）；
  * 问题是**真实用户问题**（DocVQA 标注），不是模板；
  * 同一份文档上通常有 2–5 个问题，天然构成"一份文档、连续追问"的多轮会话。

会话结构（轮数分布）参考 MMDU 的统计：多数会话 2–5 轮。
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


def _query(q) -> str:
    if isinstance(q, dict):
        return q.get("en") or q.get("de") or next(iter(q.values()))
    return str(q)


def load_sessions(n_sessions: int = 12, min_q: int = 2, max_q: int = 5,
                  seed: int = 0, parquet: str = PARQUET):
    import pandas as pd
    from PIL import Image

    df = pd.read_parquet(parquet)
    groups: dict[int, list[int]] = {}
    for i in range(len(df)):
        b = df.iloc[i]["image"]["bytes"]
        groups.setdefault(hash(b), []).append(i)

    rng = np.random.default_rng(seed)
    cand = [(h, idx) for h, idx in groups.items() if len(idx) >= min_q]
    order = rng.permutation(len(cand))

    sessions = []
    for oi in order:
        h, idx = cand[oi]
        idx = idx[:max_q]
        img = Image.open(io.BytesIO(df.iloc[idx[0]]["image"]["bytes"])).convert("RGB")
        turns = []
        for k, i in enumerate(idx):
            turns.append({
                "user": _query(df.iloc[i]["query"]),
                "images": [img] if k == 0 else [],   # 图只在首轮发，后续靠上下文
                "assistant": "",
            })
        sessions.append({"sid": len(sessions), "turns": turns,
                         "arrive_at": 0.0, "think_times": []})
        if len(sessions) >= n_sessions:
            break
    return sessions


def schedule(sessions, arrival_span: float = 30.0, think_mean: float = 6.0, seed: int = 0):
    rng = np.random.default_rng(seed)
    for s in sessions:
        s["arrive_at"] = float(rng.uniform(0, arrival_span))
        s["think_times"] = list(rng.exponential(think_mean, max(len(s["turns"]) - 1, 0)))
    sessions.sort(key=lambda x: x["arrive_at"])
    return sessions


def build_messages_docvqa(session, turn: int, history: list[dict], image_loader=None):
    t = session["turns"][turn]
    content = [{"type": "image", "image": im} for im in t["images"]]
    content.append({"type": "text", "text": t["user"]})
    from prompts import SYSTEM_PROMPT
    return [{"role": "system", "content": SYSTEM_PROMPT}] + history + [
        {"role": "user", "content": content}]


if __name__ == "__main__":
    ss = load_sessions(n_sessions=5)
    for s in ss:
        print(f"sid={s['sid']} turns={len(s['turns'])} q1={s['turns'][0]['user'][:70]}")
