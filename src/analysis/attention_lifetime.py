"""token 级寿命（注意力视角）：一个 token 什么时候"真正被需要"。

块级 trace 里，同一个 16-token 块内的 token 同生共死。但题面的真问题是：
"这个 token 的 KV 以后还会不会被读到？"——在多模态里这是高度不连续的，
模型可能在生成到第 30 个 token 时突然回头看图里的某个区域。

这里用最直接的办法量它：
  * 跑一次真实生成，逐 step 记录**最后一个 query 位置**对全部历史 token 的注意力；
  * 只统计 full attention 层（Qwen3.5 每 4 层 1 层），对 head 取平均；
  * 定义 token 的"注意力寿命" = 最后一次被显著注意（超过该 step 注意力质量的
    阈值占比）的 step 序号 / 时刻。

产出：figs/fig03_attention_lifetime.png + data/attention_trace.npz

    python src/attention_lifetime.py --turns 2 --max-new 48
"""

from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _setup import add_src_paths  # noqa: E402
add_src_paths()

import argparse
import json
import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt  # noqa: E402
import plotstyle  # noqa: E402
from mm_input import MMInput  # noqa: E402
from workload import WorkloadConfig, build_sessions, build_messages  # noqa: E402

plotstyle.setup()

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(ROOT, "models", "Qwen3.5-9B")


def load_model(path):
    import torch

    try:
        from transformers import AutoModelForImageTextToText as Cls
    except Exception:  # noqa: BLE001
        from transformers import Qwen3_5ForConditionalGeneration as Cls

    m = Cls.from_pretrained(
        path, dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="eager", trust_remote_code=True)
    m.eval()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns", type=int, default=2)
    ap.add_argument("--max-new", type=int, default=48)
    ap.add_argument("--eps-frac", type=float, default=0.01,
                    help="注意力质量占该 step 总质量的比例阈值")
    args = ap.parse_args()

    import torch

    mm = MMInput(MODEL_DIR)
    t0 = time.time()
    model = load_model(MODEL_DIR)
    print(f"[load] {time.time()-t0:.1f}s  "
          f"mem={torch.cuda.memory_allocated()/1e9:.1f}GB")

    cfg = WorkloadConfig(n_sessions=1, turns_per_session=args.turns,
                         arrival_span_s=0, think_mean_s=0, seed=7)
    s = build_sessions(cfg)[0]
    history: list[dict] = []

    all_rows = []
    for turn in range(args.turns):
        msgs, new_img = build_messages(s, turn, history)
        enc = mm.processor.apply_chat_template(
            msgs, tokenize=True, return_dict=True, return_tensors="pt",
            add_generation_prompt=True, enable_thinking=False)
        input_ids = enc["input_ids"].to("cuda")
        ids_list = enc["input_ids"][0].tolist()
        segs = mm.segment(ids_list, n_turns=turn + 1)
        kinds = ["other"] * len(ids_list)
        for sg in segs:
            for p in range(sg.start, min(sg.end, len(kinds))):
                kinds[p] = sg.kind
        mminput = {k: v.to("cuda") for k, v in enc.items()
                   if k not in ("input_ids", "attention_mask") and torch.is_tensor(v)}

        seq_len = input_ids.shape[1]
        # 用 generate(return_dict_in_generate=True, output_attentions=True)：
        # 它自己处理 mrope 的 position_ids/rope_deltas，比手写解码循环稳。
        # outputs.attentions[0] 是 prefill（q_len=seq_len），之后每步 q_len=1。
        with torch.inference_mode():
            gen = model.generate(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                max_new_tokens=args.max_new,
                do_sample=False,
                output_attentions=True,
                return_dict_in_generate=True,
                **mminput,
            )
            rows = []
            for step, atts in enumerate(gen.attentions or []):
                vecs = []
                for a in atts:
                    if a is None:
                        continue
                    q = a[0, :, -1 if step == 0 else 0, :]
                    if q.shape[-1] < seq_len:
                        continue
                    vecs.append(q[:, :seq_len].float().mean(dim=0))
                if vecs:
                    rows.append(torch.stack(vecs).mean(dim=0))
                else:
                    rows.append(torch.zeros(seq_len))
            A = torch.stack(rows).float().cpu().numpy() if rows else torch.zeros(1, seq_len).numpy()
            gen_ids = gen.sequences[0, seq_len:].tolist()
            del gen, rows
            torch.cuda.empty_cache()

        text = mm.tokenizer.decode(gen_ids, skip_special_tokens=True)
        print(f"[turn{turn}] prompt={seq_len} gen={len(gen_ids)} steps={A.shape[0]}")
        print("   ", text[:160].replace("\n", " "))
        all_rows.append({"turn": turn, "A": A, "kinds": kinds, "prompt_len": seq_len,
                         "gen_text": text})

        history.append({"role": "user", "content": msgs[-1]["content"]})
        history.append({"role": "assistant", "content": text or "（无内容）"})

    analyze(all_rows, args)


def _attn_last_query(attentions, cur_len):
    """各 full-attention 层、各 head 对历史 token 的注意力，在 head 上取平均。"""
    import torch

    vecs = []
    for a in attentions:
        if a is None:
            continue
        # a: [bsz, heads, q_len, kv_len]
        v = a[0, :, -1, :cur_len].float().mean(dim=0)
        vecs.append(v)
    if not vecs:
        return torch.zeros(cur_len)
    return torch.stack(vecs).mean(dim=0)


def analyze(rows, args):
    fig = plt.figure(figsize=(15, 4.2 * len(rows)))
    for i, r in enumerate(rows):
        A = r["A"]
        kinds = r["kinds"]
        T, L = A.shape

        # 每个 step 的注意力质量归一化，再判断"是否被显著注意"
        mass = A.sum(axis=1, keepdims=True) + 1e-9
        P = A / mass
        sig = P > args.eps_frac

        ax = fig.add_subplot(len(rows), 2, 2 * i + 1)
        im = ax.imshow(np.log10(P + 1e-6), aspect="auto", origin="lower",
                       cmap="magma", vmin=-4, vmax=0)
        # 标出图像区间
        spans, start = [], None
        for p, k in enumerate(kinds):
            if k == "image" and start is None:
                start = p
            if k != "image" and start is not None:
                spans.append((start, p))
                start = None
        if start is not None:
            spans.append((start, len(kinds)))
        for (s0, s1) in spans:
            ax.axvline(s0, color="#33CCFF", lw=1.0, ls="--")
            ax.axvline(s1, color="#33CCFF", lw=1.0, ls="--")
        ax.set_xlabel("历史 token 位置（蓝色虚线之间 = 图像 token）")
        ax.set_ylabel("生成步")
        ax.set_title(f"(turn{r['turn']}) 每步注意力质量分布 log10", fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.03)

        # token 的"注意力寿命"：最后一次被显著注意的 step
        ax2 = fig.add_subplot(len(rows), 2, 2 * i + 2)
        last_step = np.full(L, -1)
        for p in range(L):
            idx = np.where(sig[:, p])[0]
            if len(idx):
                last_step[p] = idx[-1]
        color = {"system": "#4C72B0", "image": "#C44E52", "user_text": "#DD8452",
                 "assistant_text": "#55A868", "template": "#999999", "other": "#BBBBBB"}
        for kind in ["system", "image", "user_text", "assistant_text"]:
            idx = [p for p, k in enumerate(kinds) if k == kind]
            if not idx:
                continue
            ax2.scatter(idx, last_step[idx], s=8, color=color[kind], alpha=0.7,
                        label=kind)
        for (s0, s1) in spans:
            ax2.axvspan(s0, s1, color="#C44E52", alpha=0.08)
        ax2.set_xlabel("token 位置")
        ax2.set_ylabel("最后一次被注意的生成步")
        ax2.set_title(f"(turn{r['turn']}) token 级注意力寿命", fontsize=10)
        ax2.legend(fontsize=8, frameon=False, markerscale=2)

    fig.suptitle("token 级寿命：注意力视角（块级 trace 看不到的组内差异）", fontsize=13)
    out = os.path.join(ROOT, "figs", "fig03_attention_lifetime.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"[fig] {out}")

    np.savez(os.path.join(ROOT, "data", "attention_trace.npz"),
             **{f"turn_{r['turn']}": r["A"] for r in rows})
    with open(os.path.join(ROOT, "data", "attention_meta.json"), "w") as f:
        json.dump([{"turn": r["turn"], "prompt_len": r["prompt_len"],
                    "gen_text": r["gen_text"][:500]} for r in rows], f,
                  ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
