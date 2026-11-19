"""输入解剖：多模态情况下 Qwen3.5 到底"看到"多少 token、这些 token 长什么样。

产出：
    data/input_anatomy.json   —— 结构化统计
    figs/fig01_input_layout.png —— token 布局图 + 分辨率-视觉 token 数曲线

用法：
    PYTHONPATH=pylibs python src/analyze_input.py
"""

from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _setup import add_src_paths  # noqa: E402
add_src_paths()

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import plotstyle

plotstyle.setup()
import numpy as np
from matplotlib.patches import Patch

C_HOT = "#C44E52"
from mm_input import MMInput, summarize_segments  # noqa: E402
from prompts import SYSTEM_PROMPT  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(ROOT, "models", "Qwen3.5-9B")
OUT_DATA = os.path.join(ROOT, "data", "input_anatomy.json")
OUT_FIG = os.path.join(ROOT, "figs", "fig01_input_layout.png")

# 颜色：按语义分类
KIND_COLOR = {
    "system": "#4C72B0",
    "user_text": "#DD8452",
    "assistant_text": "#55A868",
    "image": "#C44E52",
    "video": "#8172B3",
    "template": "#BBBBBB",
    "other": "#999999",
}
KIND_LABEL = {
    "system": "系统指令词",
    "user_text": "用户文本",
    "assistant_text": "助手回复",
    "image": "图像 token",
    "video": "视频 token",
    "template": "对话模板/分隔符",
}


def make_image(w: int, h: int, seed: int = 0):
    """造一张"截图感"的图（不同区域有明显结构，便于真实推理时模型有事可做）。"""
    from PIL import Image, ImageDraw

    rng = np.random.default_rng(seed)
    img = Image.new("RGB", (w, h), (250, 250, 248))
    d = ImageDraw.Draw(img)
    # 顶栏
    d.rectangle([0, 0, w, max(6, h // 14)], fill=(32, 48, 80))
    # 文字行
    y = h // 10
    while y < h * 0.55:
        x = w // 20
        while x < w * 0.95:
            line_w = int(rng.integers(w // 40, w // 6))
            d.rectangle([x, y, min(x + line_w, int(w * 0.95)), y + max(2, h // 90)],
                        fill=(70, 70, 70))
            x += line_w + w // 40
        y += max(6, h // 30)
    # 图表区
    box = [int(w * 0.08), int(h * 0.62), int(w * 0.92), int(h * 0.93)]
    d.rectangle(box, outline=(120, 120, 120), width=max(1, w // 400))
    xs = np.linspace(box[0] + w * 0.03, box[2] - w * 0.03, 9)
    for i in range(len(xs) - 1):
        bh = rng.uniform(0.2, 1.0) * (box[3] - box[1]) * 0.75
        d.rectangle([xs[i], box[3] - bh, xs[i + 1] - (xs[1] - xs[0]) * 0.25, box[3]],
                    fill=(90, 140, 200))
    return img


def build_conversation(mm: MMInput, img_tokens_target: str = "screenshot_1280x720"):
    """构造一条 4 轮多模态对话：图 + 文本追问 + 换图 + 追问。"""
    from prompts import SYSTEM_PROMPT as SYS

    w, h = {"screenshot_1280x720": (1280, 720),
            "photo_1920x1080": (1920, 1080)}.get(img_tokens_target, (1280, 720))
    img1 = make_image(w, h, seed=1)
    img2 = make_image(720, 1280, seed=2)  # 手机竖屏截图

    msgs = [
        {"role": "system", "content": SYS},
        {"role": "user", "content": [
            {"type": "image", "image": img1},
            {"type": "text", "text": "帮我看看这张图：这份季度经营看板里，哪一项指标环比下滑最快？"}, ]},
        {"role": "assistant", "content": "图中是第 3 季度经营看板，包含营收、毛利率、活跃用户数、客单价四项指标及柱状图。环比下滑最快的是「活跃用户数」，由上季度 1,284 万降至 1,102 万，环比 -14.2%；其余三项分别为营收 -3.1%、毛利率 -0.8 pct、客单价 +1.6%。"},
        {"role": "user", "content": [
            {"type": "text", "text": "活跃用户下滑主要集中在哪个端？把柱状图里的数字读出来。"}]},
        {"role": "assistant", "content": "柱状图按 App、小程序、Web 三个端给出。App 端 612 万（上季 771 万，-20.6%），小程序端 361 万（上季 402 万，-10.2%），Web 端 129 万（上季 111 万，+16.2%）。下滑主要来自 App 端，贡献了约 83% 的减少量；Web 端是唯一正增长的渠道。"},
        {"role": "user", "content": [
            {"type": "image", "image": img2},
            {"type": "text", "text": "这是 App 端的埋点明细截图，和刚才的比例对得上吗？"}, ]},
        {"role": "assistant", "content": "对得上。截图里 App 端日活按版本拆分：8.2.x 为 431 万、8.1.x 为 148 万、7.x 及以下为 33 万，合计 612 万，与上一张图 App 端数字一致。8.2.x 占比 70.4%，说明流失并非版本迁移造成；结合截图中的启动失败率 4.7%（上季 0.9%），推测下滑主因是 8.2.0 版本的启动崩溃。"},
        {"role": "user", "content": [
            {"type": "text", "text": "那接下来该优先修什么？给三条按优先级排序的行动项。"}]},
    ]
    return msgs, (img1, img2)


def per_turn_prompt_lengths(mm: MMInput, msgs: list[dict]) -> list[dict]:
    """按真实请求方式，计算每轮请求的 prompt 长度（= 到该轮 user 消息为止的全部 token）。"""
    rows = []
    for i, m in enumerate(msgs):
        if m["role"] != "user":
            continue
        prefix = msgs[: i + 1]
        o = mm.tokenize(prefix, add_generation_prompt=True)
        rows.append({"turn": len(rows), "position": i, "prompt_tokens": len(o["input_ids"][0])})
    return rows


def analyze_layout(mm: MMInput):
    msgs, imgs = build_conversation(mm)
    out = mm.tokenize(msgs, add_generation_prompt=True)
    ids = out["input_ids"][0]
    segs = mm.segment(ids, n_turns=4)
    kind_tot = summarize_segments(segs)

    # 每轮的 token 累计（用于展示"轮次越往后上下文越长"）
    per_turn = {}
    for s in segs:
        per_turn.setdefault(s.turn, {}).setdefault(s.kind, 0)
        per_turn[s.turn][s.kind] += s.length

    grid = out["image_grid_thw"]
    turn_rows = per_turn_prompt_lengths(mm, msgs)
    n_img = sum(1 for s in segs if s.kind == "image")
    return {
        "total_tokens": len(ids),
        "kind_tokens": kind_tot,
        "image_tokens_total": kind_tot.get("image", 0),
        "image_grid_thw": grid,
        "n_image_segments": n_img,
        "per_turn": {str(k): v for k, v in sorted(per_turn.items())},
        "pixel_values_shape": list(np.shape(out["pixel_values"])),
        "per_request": turn_rows,
        "segments": [
            {"start": s.start, "end": s.end, "len": s.length, "kind": s.kind, "turn": s.turn}
            for s in segs
        ],
    }, ids, segs, imgs


def sweep_resolutions(mm: MMInput):
    """常见分辨率 -> 视觉 token 数。视觉 token = (H/32) * (W/32)（merge 2 × patch 16）。"""
    from PIL import Image

    common = [
        ("256x256 缩略图", 256, 256),
        ("640x480 VGA 照片", 640, 480),
        ("1024x768 文档扫描", 1024, 768),
        ("1280x720 屏幕截图", 1280, 720),
        ("1920x1080 全屏截图", 1920, 1080),
        ("2560x1440 2K 截图", 2560, 1440),
        ("3840x2160 4K 截图", 3840, 2160),
        ("720x1280 手机竖屏", 720, 1280),
        ("4096x4096 上限", 4096, 4096),
    ]
    rows = []
    for name, w, h in common:
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": Image.new("RGB", (w, h), (128, 128, 128))},
            {"type": "text", "text": "描述这张图。"}]}]
        o = mm.tokenize(msgs)
        ids = o["input_ids"][0]
        segs = mm.segment(ids, n_turns=1)
        n_img = sum(s.length for s in segs if s.kind == "image")
        rows.append({
            "name": name, "w": w, "h": h,
            "image_tokens": n_img,
            "total_tokens": len(ids),
            "grid": o["image_grid_thw"],
            "pixels_per_token": (w * h) / max(n_img, 1),
        })
    return rows


def plot(layout, ids, segs, sweep):
    fig = plt.figure(figsize=(15, 8.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1], width_ratios=[2.1, 1],
                          hspace=0.42, wspace=0.22)

    # ---- (a) token 布局 ----
    ax = fig.add_subplot(gs[0, :])
    rows = ["system", "user_text", "image", "assistant_text", "template"]
    ypos = {k: i for i, k in enumerate(rows)}
    for s in segs:
        if s.kind not in ypos:
            continue
        ax.add_patch(plt.Rectangle((s.start, ypos[s.kind] - 0.34), max(s.length, 1), 0.68,
                                   color=KIND_COLOR[s.kind],
                                   alpha=0.9 if s.kind != "template" else 0.5))
    # 轮次分隔线
    turn_starts = []
    for s in segs:
        if s.kind == "user_text" and s.turn >= 0:
            turn_starts.append(s.start)
    for k, x in enumerate(turn_starts):
        ax.axvline(x, color="#333333", lw=0.8, ls=":")
        ax.text(x + 6, -0.55, f"turn{k+1}", fontsize=8, color="#333333")
    ax.set_yticks([ypos[r] for r in rows])
    ax.set_yticklabels([KIND_LABEL[r] for r in rows])
    ax.set_xlim(0, layout["total_tokens"])
    ax.set_ylim(-0.7, len(rows) - 0.1)
    ax.set_xlabel("token 位置（从 0 开始）")
    ax.set_title(f"(a) 一条 4 轮多模态对话的 token 布局：共 {layout['total_tokens']} tokens，"
                 f"其中图像 {layout['image_tokens_total']} tokens "
                 f"({layout['image_tokens_total']/layout['total_tokens']*100:.0f}%)",
                 fontsize=11)
    ax.grid(axis="x", alpha=0.15)
    handles = [Patch(color=KIND_COLOR[k], label=KIND_LABEL[k]) for k in rows]
    ax.legend(handles=handles, fontsize=8, loc="upper right", ncol=5, frameon=False)

    # ---- (b) 每轮上下文增长 ----
    ax2 = fig.add_subplot(gs[1, 0])
    reqs = layout["per_request"]
    labels = [f"turn{r['turn']+1}" for r in reqs]
    lens = [r["prompt_tokens"] for r in reqs]
    prev = 0
    newtoks = []
    for v in lens:
        newtoks.append(v - prev)
        prev = v
    ax2.bar(labels, newtoks, color="#7F9AC8", label="本轮新增 token")
    ax2.plot(labels, lens, "o-", color=C_HOT, label="该轮请求的 prompt 总长")
    for i, v in enumerate(lens):
        ax2.text(i, v + max(lens) * 0.02, str(v), ha="center", fontsize=8, color=C_HOT)
    ax2.set_ylabel("tokens")
    ax2.set_title("(b) 多轮对话：历史全部留在 KV 里，prompt 逐轮变长", fontsize=10)
    ax2.legend(fontsize=8, frameon=False, loc="upper left")
    ax2.set_ylim(0, max(lens) * 1.18)

    # ---- (c) 分辨率 -> 视觉 token ----
    ax3 = fig.add_subplot(gs[1, 1])
    names = [r["name"] for r in sweep]
    vals = [r["image_tokens"] for r in sweep]
    ax3.barh(range(len(names)), vals, color="#C44E52", alpha=0.85)
    ax3.set_yticks(range(len(names)))
    ax3.set_yticklabels(names, fontsize=8)
    ax3.invert_yaxis()
    for i, v in enumerate(vals):
        ax3.text(v + max(vals) * 0.01, i, f"{v}", va="center", fontsize=8)
    ax3.set_xlabel("单张图片的视觉 token 数")
    ax3.set_xlim(0, max(vals) * 1.22)
    ax3.set_title("(c) 一张图 = 多少 token（patch16/merge2）", fontsize=10)
    ax3.grid(axis="x", alpha=0.2)

    fig.suptitle("Qwen3.5-9B 多模态输入解剖", fontsize=13, y=0.98)
    fig.savefig(OUT_FIG, dpi=160, bbox_inches="tight")
    print(f"[fig] {OUT_FIG}")


def main():
    mm = MMInput(MODEL_DIR)
    layout, ids, segs, _ = analyze_layout(mm)
    sweep = sweep_resolutions(mm)

    os.makedirs(os.path.dirname(OUT_DATA), exist_ok=True)
    with open(OUT_DATA, "w") as f:
        json.dump({"layout": layout, "sweep": sweep,
                   "system_prompt_tokens": mm.count_tokens(SYSTEM_PROMPT)}, f,
                  ensure_ascii=False, indent=2)

    print(f"总 token: {layout['total_tokens']}")
    print("按类别:", layout["kind_tokens"])
    print("图像 grid:", layout["image_grid_thw"])
    print("系统指令词 token:", mm.count_tokens(SYSTEM_PROMPT))
    print("分辨率扫描:")
    for r in sweep:
        print(f"  {r['name']:<24} {r['image_tokens']:>6} 视觉 token  (总 {r['total_tokens']})")

    plot(layout, ids, segs, sweep)


if __name__ == "__main__":
    main()
