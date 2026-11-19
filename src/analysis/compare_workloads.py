"""对照：合成 workload vs 真实数据集（MMDU）的输入分布。

只做 CPU 侧（tokenize），比较：
  * 每会话轮数、每轮图片数
  * 每次请求的 prompt 长度、图像 token 占比
  * 上下文随轮次的增长

产出：data/workload_compare.json、figs/fig08_workload_compare.png
"""

from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _setup import add_src_paths  # noqa: E402
add_src_paths()

import json
import os
import sys
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt  # noqa: E402
import plotstyle  # noqa: E402
from mm_input import MMInput  # noqa: E402

plotstyle.setup()
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(ROOT, "models", "Qwen3.5-9B")


def stats_for(conversations, mm):
    """conversations: [ [(msgs, images, spans), ... ] ]，返回每请求统计。"""
    rows = []
    for sid, turns in enumerate(conversations):
        for t, (msgs, imgs, spans) in enumerate(turns):
            tk = mm.tokenize(msgs, add_generation_prompt=True)
            ids = tk["input_ids"][0]
            n_img_tok = sum(e - s for s, e, _ in spans)
            rows.append({
                "sid": sid, "turn": t, "prompt_len": len(ids),
                "n_images": len(imgs),
                "img_tokens": n_img_tok,
                "img_share": n_img_tok / max(len(ids), 1),
            })
    return rows


def build_synthetic(mm, n_sessions=12, turns=6, seed=0):
    from workload import WorkloadConfig, build_messages, build_sessions, image_fingerprint

    cfg = WorkloadConfig(n_sessions=n_sessions, turns_per_session=turns,
                         arrival_span_s=30, think_mean_s=6, seed=seed)
    sessions = build_sessions(cfg)
    out = []
    for s in sessions:
        history, conv = [], []
        for t in range(s.turns):
            msgs, new_img = build_messages(s, t, history)
            ids = mm.tokenize(msgs)["input_ids"][0]
            segs = mm.segment(ids, n_turns=t + 1)
            spans = [(sg.start, sg.end, "syn")
                     for sg in segs if sg.kind == "image"]
            imgs = [c["image"] for m in msgs if isinstance(m.get("content"), list)
                    for c in m["content"] if c.get("type") == "image"]
            conv.append((msgs, imgs, spans))
            history.append({"role": "user", "content": msgs[-1]["content"]})
            history.append({"role": "assistant", "content": "好。"})
        out.append(conv)
    return out


def build_real(mm, n_sessions=12, max_turns=6, seed=0):
    from workload_real import build_messages_real, load_mmdu, schedule
    from PIL import Image

    def loader(p):
        return Image.open(p).convert("RGB")

    ss = load_mmdu(os.path.join(ROOT, "data/MMDU/mmdu-45k.json"),
                   os.path.join(ROOT, "data/MMDU/pics/mmdu-45k_pics"),
                   max_sessions=n_sessions, min_turns=3, max_turns=max_turns, seed=seed)
    schedule(ss, arrival_span=30, think_mean=6, seed=seed)
    out = []
    for s in ss:
        history, conv = [], []
        for t in range(len(s.turns)):
            msgs = build_messages_real(s, t, history, loader)
            ids = mm.tokenize(msgs)["input_ids"][0]
            segs = mm.segment(ids, n_turns=t + 1)
            spans = [(sg.start, sg.end, "real") for sg in segs if sg.kind == "image"]
            imgs = [c["image"] for m in msgs if isinstance(m.get("content"), list)
                    for c in m["content"] if c.get("type") == "image"]
            conv.append((msgs, imgs, spans))
            history.append({"role": "user", "content": msgs[-1]["content"]})
            history.append({"role": "assistant", "content": "好。"})
        out.append(conv)
    return out


def summarize(rows, name):
    df = np.array([r["prompt_len"] for r in rows])
    share = np.array([r["img_share"] for r in rows])
    nimg = np.array([r["n_images"] for r in rows])
    return {
        "name": name, "n_requests": len(rows),
        "prompt_len": {"mean": float(df.mean()), "p50": float(np.median(df)),
                       "p90": float(np.percentile(df, 90)), "max": int(df.max())},
        "img_share_mean": float(share.mean()),
        "n_images_mean": float(nimg.mean()),
        "n_images_max": int(nimg.max()),
    }


def main():
    mm = MMInput(MODEL_DIR)
    print("building synthetic ...")
    syn = build_synthetic(mm)
    print("building real (MMDU) ...")
    real = build_real(mm)

    syn_rows = stats_for(syn, mm)
    real_rows = stats_for(real, mm)
    s1, s2 = summarize(syn_rows, "合成(模板+生成截图)"), summarize(real_rows, "MMDU(真实多轮多图)")

    out = {"synthetic": s1, "real": s2,
           "synthetic_rows": syn_rows, "real_rows": real_rows}
    os.makedirs(os.path.join(ROOT, "data"), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, "data", "workload_compare.json"), "w"),
              ensure_ascii=False, indent=2)

    print(json.dumps({"synthetic": s1, "real": s2}, ensure_ascii=False, indent=2))
    plot(syn_rows, real_rows)


def plot(syn_rows, real_rows):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    ax = axes[0]
    ax.hist([r["prompt_len"] for r in syn_rows], bins=20, alpha=0.6,
            label="合成", color="#4C72B0")
    ax.hist([r["prompt_len"] for r in real_rows], bins=20, alpha=0.6,
            label="MMDU", color="#C44E52")
    ax.set_xlabel("每次请求的 prompt 长度 (token)")
    ax.set_ylabel("请求数")
    ax.set_title("(a) 上下文长度分布", fontsize=10)
    ax.legend(fontsize=8, frameon=False)

    ax2 = axes[1]
    ax2.hist([r["img_share"] * 100 for r in syn_rows], bins=20, alpha=0.6,
             label="合成", color="#4C72B0")
    ax2.hist([r["img_share"] * 100 for r in real_rows], bins=20, alpha=0.6,
             label="MMDU", color="#C44E52")
    ax2.set_xlabel("图像 token 占比 (%)")
    ax2.set_ylabel("请求数")
    ax2.set_title("(b) 多模态占比", fontsize=10)
    ax2.legend(fontsize=8, frameon=False)

    ax3 = axes[2]
    for rows, name, c in ((syn_rows, "合成", "#4C72B0"), (real_rows, "MMDU", "#C44E52")):
        turns = sorted({r["turn"] for r in rows})
        med = [np.median([r["prompt_len"] for r in rows if r["turn"] == t]) for t in turns]
        ax3.plot([t + 1 for t in turns], med, "o-", color=c, label=name)
    ax3.set_xlabel("轮次")
    ax3.set_ylabel("prompt 长度中位数")
    ax3.set_title("(c) 上下文随轮次增长", fontsize=10)
    ax3.legend(fontsize=8, frameon=False)

    fig.suptitle("合成 workload vs 真实数据集（MMDU）", fontsize=12)
    out = os.path.join(ROOT, "figs", "fig08_workload_compare.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[fig] {out}")


if __name__ == "__main__":
    main()
