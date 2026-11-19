"""用真实多轮多模态对话数据构造 workload（MMDU / VisDial 通用）。

与 `workload.py`（合成截图 + 模板问题）的区别：
  * 图像 = 数据集里的**真实图片**；
  * 每轮的提问 = 数据集里的**真实用户问题**；
  * 轮数/图片数 = 数据集本身的结构。

仍然可控/可复现的部分：会话数、到达时刻、思考时间（这些用真实 trace 的统计量标定）。

MMDU 的 JSON 结构（laolao77/MMDU）大致是：
    [{"id": ..., "image": ["xxx.jpg", ...],
      "conversations": [{"from": "human", "value": "<image>请描述..."},
                        {"from": "gpt", "value": "..."}, ...]}, ...]
不同版本的字段名略有差异，这里做了兼容处理。
"""

from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _setup import add_src_paths  # noqa: E402
add_src_paths()

import json
import os
import random
import zipfile
from dataclasses import dataclass

from prompts import SYSTEM_PROMPT

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class RealDialog:
    sid: int
    turns: list[dict]          # [{"user": str, "images": [路径...], "assistant": str}]
    arrive_at: float = 0.0
    think_times: list[float] = None


def _norm_turns(sample: dict) -> list[dict]:
    """把不同格式的对话样本统一成 [{'user','images','assistant'}]。"""
    conv = sample.get("conversations") or sample.get("conversation") or []
    imgs = sample.get("image") or sample.get("images") or []
    if isinstance(imgs, str):
        imgs = [imgs]
    turns, cur = [], None
    img_i = 0
    for m in conv:
        role = m.get("from") or m.get("role", "")
        text = m.get("value") or m.get("content", "")
        text = text.replace("<ImageHere>", "<image>")
        if role in ("human", "user"):
            n_img = text.count("<image>")
            used = []
            for _ in range(max(n_img, 0)):
                if img_i < len(imgs):
                    used.append(imgs[img_i])
                    img_i += 1
            if n_img == 0 and img_i < len(imgs) and not turns:
                used = [imgs[img_i]]
                img_i += 1
            cur = {"user": text.replace("<image>", "").strip(), "images": used,
                   "assistant": ""}
            turns.append(cur)
        elif role in ("gpt", "assistant", "assistant"):
            if cur is not None:
                cur["assistant"] = text.strip()
    return [t for t in turns if t["user"]]


def load_mmdu(json_path: str, image_dir: str, max_sessions: int = 30,
              min_turns: int = 2, max_turns: int = 6, seed: int = 0) -> list[RealDialog]:
    """挑出多轮样本，转成会话列表（图路径已解析成绝对路径）。"""
    data = json.load(open(json_path))
    rng = random.Random(seed)
    rng.shuffle(data)
    out = []
    for s in data:
        turns = _norm_turns(s)
        if not (min_turns <= len(turns) <= max_turns):
            continue
        if not any(t["images"] for t in turns):
            continue
        ok = []
        for t in turns:
            paths = [os.path.join(image_dir, os.path.basename(p)) for p in t["images"]]
            paths = [p for p in paths if os.path.exists(p)]
            ok.append({"user": t["user"], "images": paths, "assistant": t["assistant"]})
        if not any(t["images"] for t in ok):
            continue          # 图不在 zip 里就跳过
        out.append(RealDialog(sid=len(out), turns=ok))
        if len(out) >= max_sessions:
            break
    return out


def extract_images(zip_path: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(out_dir)
    return out_dir


def build_messages_real(session: RealDialog, turn: int, history: list[dict],
                        image_loader):
    """构造第 turn 轮的 messages（含系统词和历史）。image_loader(路径)->PIL.Image"""
    t = session.turns[turn]
    content = []
    for p in t["images"]:
        content.append({"type": "image", "image": image_loader(p)})
    content.append({"type": "text", "text": t["user"]})
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}] + history + [
        {"role": "user", "content": content}]
    return msgs


def schedule(sessions: list[RealDialog], arrival_span: float = 30.0,
             think_mean: float = 6.0, seed: int = 0):
    """给每个会话分配到达时刻和每轮思考时间（分布可由真实 trace 标定）。"""
    rng = random.Random(seed)
    for s in sessions:
        s.arrive_at = rng.uniform(0, arrival_span)
        s.think_times = [rng.expovariate(1.0 / think_mean) for _ in range(len(s.turns) - 1)]
    sessions.sort(key=lambda x: x.arrive_at)
    return sessions


if __name__ == "__main__":
    import sys

    jp = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "data/MMDU/mmdu-45k.json")
    idir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "data/MMDU/pics")
    ss = load_mmdu(jp, idir)
    print(f"loaded {len(ss)} sessions")
    for s in ss[:3]:
        print(f"  sid={s.sid} turns={len(s.turns)} imgs={sum(len(t['images']) for t in s.turns)}")
        print("   ", s.turns[0]["user"][:80])
