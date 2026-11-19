"""Qwen3.5 多模态输入的公共工具：构造对话、tokenize、按模态/轮次切分 token。

设计原则（对应题面"方法不能依赖模型与推理框架"）：
    这里只依赖 *处理器*（chat template + tokenizer + image processor）产出的
    可见输入，即"真正喂给模型的 token 序列"，不依赖任何模型内部对象。
    寿命分析的输入就是这条 token 序列 + 每条请求的到达/结束时刻。
"""

from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _setup import add_src_paths  # noqa: E402
add_src_paths()

from dataclasses import dataclass, field
from typing import Any

# 这些 id 由 tokenizer 在运行时解析，避免硬编码
SPECIAL_TOKENS = [
    "<|im_start|>",
    "<|im_end|>",
    "<|vision_start|>",
    "<|vision_end|>",
    "<|image_pad|>",
    "<|video_pad|>",
]


def _numpyify(d: dict[str, Any]) -> dict[str, Any]:
    """把 tensor / list 统一成 numpy 数组或嵌套 list，方便下游处理。"""
    out: dict[str, Any] = {}
    for k, v in d.items():
        if hasattr(v, "tolist"):
            out[k] = v.tolist()
        else:
            out[k] = v
    return out


@dataclass
class Segment:
    """token 序列中一段连续区间，带语义标签。"""

    start: int          # 含
    end: int            # 不含
    kind: str           # system / user_text / assistant_text / image / video / template
    turn: int = -1      # 属于第几轮对话（从 0 开始），模板/系统段为 -1

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass
class TokenizedConversation:
    input_ids: list[int]
    segments: list[Segment]
    image_grid_thw: list[list[int]] = field(default_factory=list)
    pixel_values_shape: tuple[int, ...] | None = None

    @property
    def length(self) -> int:
        return len(self.input_ids)


class MMInput:
    """对 Qwen3VLProcessor 的薄封装。"""

    def __init__(self, model_dir: str):
        from transformers import AutoProcessor

        self.processor = AutoProcessor.from_pretrained(model_dir)
        self.tokenizer = self.processor.tokenizer
        self.tok = {t: self.tokenizer.convert_tokens_to_ids(t) for t in SPECIAL_TOKENS}
        # 反向表：id -> 特殊 token 名
        self.special_by_id = {v: k for k, v in self.tok.items()}

    # ---------- 基本 tokenize ----------

    def tokenize(self, messages: list[dict[str, Any]], add_generation_prompt: bool = True,
                 enable_thinking: bool = False):
        out = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
        )
        return _numpyify(out)

    def apply_chat_template_ids(self, messages: list[dict[str, Any]],
                                enable_thinking: bool = False) -> list[int]:
        """只要 input_ids（不带 pixel_values），用于快速统计。"""
        return self.tokenize(messages, add_generation_prompt=True,
                             enable_thinking=enable_thinking)["input_ids"][0]

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    # ---------- 切分 ----------

    def segment(self, input_ids: list[int], n_turns: int) -> list[Segment]:
        """把 token 序列切成 system / 每轮 user / assistant / image 等段。

        依据模型自己的模板结构：
            <|im_start|>role\\n ...content... <|im_end|>
            <|vision_start|><|image_pad|>...<|vision_end|>
        """
        ids = input_ids
        tok = self.tok
        segs: list[Segment] = []
        i = 0
        n = len(ids)
        turn = -1
        cur_role: str | None = None
        buf_start: int | None = None
        role = ""

        def flush(upto: int, kind: str):
            nonlocal buf_start
            if buf_start is not None and upto > buf_start:
                segs.append(Segment(buf_start, upto, kind, turn))
            buf_start = None

        while i < n:
            t = ids[i]
            if t == tok["<|im_start|>"]:
                # 关闭上一个残留段
                flush(i, "template")
                # 读 role 名：im_start 后紧跟 "role" 文本 token，直到换行
                j = i + 1
                role_text = ""
                while j < n and ids[j] not in self.special_by_id:
                    s = self.tokenizer.decode([ids[j]])
                    if "\n" in s:
                        role_text += s
                        j += 1
                        break
                    role_text += s
                    j += 1
                role = role_text.replace("\n", "").strip()
                if role == "user":
                    turn += 1
                cur_role = role
                segs.append(Segment(i, j, "template", turn))
                buf_start = j
                i = j
                continue
            if t == tok["<|im_end|>"]:
                kind = {
                    "system": "system",
                    "user": "user_text",
                    "assistant": "assistant_text",
                }.get(cur_role or "", "other")
                flush(i, kind)
                segs.append(Segment(i, i + 1, "template", turn))
                cur_role = None
                i += 1
                continue
            if t == tok["<|vision_start|>"]:
                flush(i, "template")
                segs.append(Segment(i, i + 1, "template", turn))
                buf_start = i + 1
                i += 1
                continue
            if t in (tok["<|image_pad|>"], tok["<|video_pad|>"]):
                kind = "image" if t == tok["<|image_pad|>"] else "video"
                j = i
                while j < n and ids[j] == t:
                    j += 1
                segs.append(Segment(i, j, kind, turn))
                i = j
                buf_start = None
                continue
            if t == tok["<|vision_end|>"]:
                segs.append(Segment(i, i + 1, "template", turn))
                i += 1
                buf_start = None
                continue
            i += 1
        flush(n, "template")
        _ = n_turns
        return segs


def summarize_segments(segs: list[Segment]) -> dict[str, int]:
    out: dict[str, int] = {}
    for s in segs:
        out[s.kind] = out.get(s.kind, 0) + s.length
    return out
