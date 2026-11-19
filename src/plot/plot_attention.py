"""重新绘制 token 级注意力寿命图（只读已保存的 npz，不需要 GPU）。

原始采集在 attention_lifetime.py（需要 GPU）。这里把"每个 token 最后一次被
显著注意的生成步"改成更好读的形式：

  (a) 每步注意力热力图（log10）
  (b) 每步"被显著注意的 token 数"，按模态堆叠 —— 注意力工作集大小
  (c) 各类 token 的"注意力寿命" ECDF —— 最后一次被注意的步数分布
"""

from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _setup import add_src_paths  # noqa: E402
add_src_paths()

import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt  # noqa: E402
import plotstyle  # noqa: E402
from mm_input import MMInput  # noqa: E402
from workload import WorkloadConfig, build_sessions, build_messages  # noqa: E402

plotstyle.setup()
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(ROOT, "models", "Qwen3.5-9B")

COLOR = {"system": "#4C72B0", "image": "#C44E52", "user_text": "#DD8452",
         "assistant_text": "#55A868", "template": "#999999", "other": "#BBBBBB"}
LABEL = {"system": "系统指令词", "image": "图像", "user_text": "用户文本",
         "assistant_text": "历史回复", "template": "模板", "other": "其他"}


def rebuild_kinds(mm, meta):
    """按与采集时完全相同的构造方式重建每条 prompt 的 token 类别。"""
    cfg = WorkloadConfig(n_sessions=1, turns_per_session=len(meta),
                         arrival_span_s=0, think_mean_s=0, seed=7)
    s = build_sessions(cfg)[0]
    history: list[dict] = []
    out = {}
    for rec in meta:
        turn = rec["turn"]
        msgs, _ = build_messages(s, turn, history)
        ids = mm.tokenize(msgs, add_generation_prompt=True)["input_ids"][0]
        segs = mm.segment(ids, n_turns=turn + 1)
        kinds = ["other"] * len(ids)
        for sg in segs:
            for p in range(sg.start, min(sg.end, len(ids))):
                kinds[p] = sg.kind
        out[turn] = (ids, kinds)
        history.append({"role": "user", "content": msgs[-1]["content"]})
        history.append({"role": "assistant", "content": rec["gen_text"]})
    return out


def main(frac: float = 0.01):
    mm = MMInput(MODEL_DIR)
    meta = json.load(open(os.path.join(ROOT, "data", "attention_meta.json")))
    npz = np.load(os.path.join(ROOT, "data", "attention_trace.npz"))
    rebuilt = rebuild_kinds(mm, meta)

    n = len(meta)
    fig = plt.figure(figsize=(15, 3.9 * n))
    gs = fig.add_gridspec(n, 3, width_ratios=[1.19, 1, 1], hspace=0.45, wspace=0.38)

    for i, rec in enumerate(meta):
        turn = rec["turn"]
        A = npz[f"turn_{turn}"]
        ids, kinds = rebuilt[turn]
        L = min(A.shape[1], len(kinds))
        A = A[:, :L]
        kinds = kinds[:L]
        steps = A.shape[0]

        # 归一化到"每个 step 对 prompt 的注意力占比"
        P = A / (A.sum(axis=1, keepdims=True) + 1e-9)

        ax = fig.add_subplot(gs[i, 0])
        im = ax.imshow(np.log10(P + 1e-6), aspect="auto", origin="lower",
                       cmap="magma", vmin=-3.5, vmax=0)
        spans, st = [], None
        for p, k in enumerate(kinds):
            if k == "image" and st is None:
                st = p
            if k != "image" and st is not None:
                spans.append((st, p)); st = None
        if st is not None:
            spans.append((st, L))
        for (s0, s1) in spans:
            ax.axvline(s0, color="#33CCFF", lw=1.0, ls="--")
            ax.axvline(s1, color="#33CCFF", lw=1.0, ls="--")
        ax.set_xlabel("历史 token 位置（蓝虚线之间=图像）")
        ax.set_ylabel("生成步")
        ax.set_title(f"(turn{turn+1}) 每步注意力分布 log10", fontsize=10)
        cb = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)

        # (b) 每步被显著注意的 token 数（按模态堆叠）
        ax2 = fig.add_subplot(gs[i, 1])
        sig = P > frac
        bottom = np.zeros(steps)
        for k in ["system", "image", "assistant_text", "user_text"]:
            idx = [p for p in range(L) if kinds[p] == k]
            if not idx:
                continue
            cnt = sig[:, idx].sum(axis=1)
            ax2.bar(range(steps), cnt, bottom=bottom, color=COLOR[k],
                    label=LABEL[k], width=1.0)
            bottom += cnt
        ax2.set_xlabel("生成步")
        ax2.set_ylabel(f"注意力占比 > {frac:.0%} 的 token 数")
        ax2.set_title(f"(turn{turn+1}) 注意力的\"工作集\"（每步在看多少历史 token）", fontsize=10)
        ax2.legend(fontsize=7, frameon=False, ncol=2)

        # (c) 注意力寿命 ECDF
        ax3 = fig.add_subplot(gs[i, 2])
        last = np.full(L, -1)
        for p in range(L):
            idx = np.where(sig[:, p])[0]
            if len(idx):
                last[p] = idx[-1]
        for k in ["system", "image", "assistant_text", "user_text"]:
            idx = [p for p in range(L) if kinds[p] == k]
            if not idx:
                continue
            v = np.sort(last[idx])
            y = np.arange(1, len(v) + 1) / len(v)
            ax3.plot(np.clip(v, 0, None), y, lw=1.8, color=COLOR[k],
                     label=f"{LABEL[k]} (n={len(idx)})")
        ax3.set_xlabel("最后一次被注意的生成步（注意力寿命）")
        ax3.set_ylabel("累积占比")
        ax3.set_title(f"(turn{turn+1}) 各类 token 的注意力寿命", fontsize=10)
        ax3.legend(fontsize=7, frameon=False)
        ax3.set_xlim(0, steps)

    fig.suptitle("token 级寿命（注意力视角）：528-token 块内部的差异", fontsize=13, y=0.995)
    out = os.path.join(ROOT, "figs", "fig03_attention_lifetime.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"[fig] {out}")


if __name__ == "__main__":
    main()
