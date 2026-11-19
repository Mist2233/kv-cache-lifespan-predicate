"""中心推理场景的多轮多模态 workload 生成。

设计要点（对应题面约束）：
  * 多模态：图像 + 文本两种模态（后续可扩展视频）。
  * 多轮对话：每个会话 6 轮，历史累积。
  * 厂商前缀：所有会话共用同一条系统指令词（最热的一段 KV）。
  * 真实频次：会话并发数、用户思考时间按泊松/指数分布抽样，固定种子可复现。
  * 图像按会话内"文档/截图"语义成组出现：同一会话首轮发主图，后续轮次发局部截图。

产出是"请求规格"，真正的 token 由 Qwen3.5 的 processor 现场生成（见 run_trace.py），
因此这个脚本本身不依赖模型。
"""

from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _setup import add_src_paths  # noqa: E402
add_src_paths()

import json
import os
from dataclasses import dataclass, field

import numpy as np

from prompts import SYSTEM_PROMPT


# ---------------------------------------------------------------- 图像

def _font(size: int):
    from PIL import ImageFont

    path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    try:
        return ImageFont.truetype(path, size)
    except Exception:  # noqa: BLE001
        return ImageFont.load_default()


def make_screenshot(w: int, h: int, seed: int = 0):
    """生成一张"经营看板"截图（有标题、KPI、柱状图、明细表，文字可读）。

    真实实验可换成真实数据集图像（见 README「数据」一节）；这里保证：
      * 内容足够丰富，模型确实需要"看图"才能回答；
      * 图像尺寸分布贴近真实截图/文档扫描；
      * 完全确定性（同一 seed 同一张图），可复现。
    """
    from PIL import Image, ImageDraw

    rng = np.random.default_rng(seed)
    img = Image.new("RGB", (w, h), (250, 250, 248))
    d = ImageDraw.Draw(img)
    sc = w / 1280.0                       # 以 1280 宽为基准缩放字号
    f_title = _font(max(10, int(26 * sc)))
    f_kpi = _font(max(8, int(22 * sc)))
    f_small = _font(max(7, int(15 * sc)))

    # 顶栏
    d.rectangle([0, 0, w, int(52 * sc)], fill=(32, 48, 80))
    d.text((int(24 * sc), int(14 * sc)), "灵犀数据 · 2026 Q3 经营看板",
           font=f_title, fill=(255, 255, 255))

    # KPI 卡片
    kpis = [
        ("营收", f"{rng.integers(8200, 9800)/100:.2f} 亿", f"{(rng.random()-0.55)*8:+.1f}%"),
        ("毛利率", f"{rng.integers(3800, 4600)/100:.1f}%", f"{(rng.random()-0.6)*2:+.1f} pct"),
        ("活跃用户", f"{rng.integers(1000, 1400)} 万", f"{(rng.random()-0.75)*20:+.1f}%"),
        ("客单价", f"{rng.integers(40, 90)} 元", f"{(rng.random()-0.4)*5:+.1f}%"),
    ]
    cw = (w - int(48 * sc)) // 4
    for i, (name, val, chg) in enumerate(kpis):
        x0 = int(24 * sc) + i * cw
        y0 = int(66 * sc)
        d.rectangle([x0, y0, x0 + cw - int(12 * sc), y0 + int(78 * sc)],
                    fill=(238, 242, 248), outline=(200, 208, 220))
        d.text((x0 + int(12 * sc), y0 + int(8 * sc)), name, font=f_small, fill=(70, 80, 100))
        d.text((x0 + int(12 * sc), y0 + int(28 * sc)), val, font=f_kpi, fill=(30, 40, 60))
        d.text((x0 + int(12 * sc), y0 + int(54 * sc)), chg, font=f_small,
               fill=(200, 60, 60) if chg.startswith("-") or "-" in chg else (40, 150, 90))

    # 柱状图
    box = [int(0.05 * w), int(0.40 * h), int(0.55 * w), int(0.90 * h)]
    d.rectangle(box, outline=(150, 150, 150))
    d.text((box[0], box[1] - int(24 * sc)), "分渠道活跃用户（万）",
           font=f_small, fill=(60, 60, 60))
    labels = ["App", "小程序", "Web", "开放平台"]
    vals = [rng.integers(100, 800) for _ in labels]
    bars = len(vals)
    bw = (box[2] - box[0]) / (bars * 2)
    maxv = max(vals)
    for i, (lb, v) in enumerate(zip(labels, vals)):
        x = box[0] + (2 * i + 0.6) * bw
        bh = (box[3] - box[1] - int(76 * sc)) * v / maxv
        d.rectangle([x, box[3] - int(30 * sc) - bh, x + bw, box[3] - int(30 * sc)],
                    fill=(90, 140, 200))
        d.text((x, box[3] - int(26 * sc)), lb, font=f_small, fill=(60, 60, 60))
        d.text((x, box[3] - int(30 * sc) - bh - int(16 * sc)), str(int(v)),
               font=f_small, fill=(40, 40, 40))

    # 明细表
    tx0, ty0 = int(0.58 * w), int(0.40 * h)
    cols = ["版本", "日活(万)", "占比", "启动失败率"]
    rows = 6
    rh = int(26 * sc)
    for j, c in enumerate(cols):
        d.text((tx0 + j * int(0.10 * w), ty0), c, font=f_small, fill=(40, 40, 40))
    for r in range(rows):
        vals_r = [f"8.{r + 5}.x" if r < 3 else f"7.{r}.x",
                  f"{rng.integers(20, 500)}",
                  f"{rng.random()*100:.1f}%",
                  f"{rng.random()*6:.1f}%"]
        for j, v in enumerate(vals_r):
            d.text((tx0 + j * int(0.10 * w), ty0 + (r + 1) * rh), v,
                   font=f_small, fill=(70, 70, 70))
    return img


# 会话主图分辨率分布（贴近真实截图/文档）
MAIN_IMAGE_SIZES = [(1280, 720), (1920, 1080), (1024, 768), (720, 1280)]
FOLLOWUP_SIZES = [(1280, 720), (1024, 768)]

QUESTIONS_IMG = [
    "帮我看看这张图：这份季度经营看板里，哪一项指标环比下滑最快？",
    "这张图的图表部分说明了什么？把关键数字读出来。",
    "请总结这张截图里最重要的三条信息。",
    "图中是否存在异常值？给出你的判断依据。",
]
QUESTIONS_FOLLOW = [
    "和上一张图对比，这次的数据有什么变化？",
    "把刚才那张图里的明细表逐行读出来。",
    "这个页面上标注的失败率是多少？和历史比如何？",
    "如果只能改一处，你建议改哪里？为什么？",
]
QUESTIONS_TEXT = [
    "接着上面，给三条按优先级排序的行动项。",
    "把结论压缩成一句话，给管理层看。",
    "假设你要向产品同学解释这个问题，会怎么说？",
    "列出你判断时用到的不确定信息。",
]


@dataclass
class SessionSpec:
    sid: int
    arrive_at: float          # 会话开始时刻（模拟秒）
    think_times: list[float]  # 每轮之间的用户思考时间
    turns: int
    # 每轮：0=纯文本，1=带一张主图，2=带一张追问图
    turn_modes: list[int]
    image_seeds: list[int] = field(default_factory=list)


@dataclass
class WorkloadConfig:
    n_sessions: int = 24
    turns_per_session: int = 6
    arrival_span_s: float = 60.0     # 会话在 [0, span] 内均匀到达
    think_mean_s: float = 6.0        # 用户思考时间均值（指数分布）
    p_image_turn0: float = 1.0       # 首轮带图概率
    p_image_later: float = 0.45      # 后续轮次带新图概率
    seed: int = 0


def build_sessions(cfg: WorkloadConfig) -> list[SessionSpec]:
    rng = np.random.default_rng(cfg.seed)
    sessions: list[SessionSpec] = []
    for sid in range(cfg.n_sessions):
        arrive = float(rng.uniform(0, cfg.arrival_span_s))
        think = [float(rng.exponential(cfg.think_mean_s))
                 for _ in range(cfg.turns_per_session - 1)]
        modes = []
        for t in range(cfg.turns_per_session):
            if t == 0:
                modes.append(1 if rng.random() < cfg.p_image_turn0 else 0)
            else:
                modes.append(2 if rng.random() < cfg.p_image_later else 0)
        seeds = [int(rng.integers(0, 10**6)) for _ in range(cfg.turns_per_session)]
        sessions.append(SessionSpec(sid, arrive, think, cfg.turns_per_session, modes, seeds))
    sessions.sort(key=lambda s: s.arrive_at)
    return sessions


def build_messages(session: SessionSpec, turn: int, history: list[dict],
                   image_factory=make_screenshot):
    """构造第 turn 轮要发给模型的完整 messages（含历史），返回 (messages, new_image)。

    图像只在本轮生成，历史消息里通过引用保持"同一张图"（image 对象复用）。
    """
    mode = session.turn_modes[turn]
    rng = np.random.default_rng(session.image_seeds[turn])

    new_img = None
    if mode == 1:
        w, h = MAIN_IMAGE_SIZES[int(rng.integers(0, len(MAIN_IMAGE_SIZES)))]
        new_img = image_factory(w, h, seed=session.image_seeds[turn])
        text = QUESTIONS_IMG[int(rng.integers(0, len(QUESTIONS_IMG)))]
        content = [{"type": "image", "image": new_img}, {"type": "text", "text": text}]
    elif mode == 2:
        w, h = FOLLOWUP_SIZES[int(rng.integers(0, len(FOLLOWUP_SIZES)))]
        new_img = image_factory(w, h, seed=session.image_seeds[turn])
        text = QUESTIONS_FOLLOW[int(rng.integers(0, len(QUESTIONS_FOLLOW)))]
        content = [{"type": "image", "image": new_img}, {"type": "text", "text": text}]
    else:
        text = QUESTIONS_TEXT[int(rng.integers(0, len(QUESTIONS_TEXT)))]
        content = [{"type": "text", "text": text}]

    msgs = [{"role": "system", "content": SYSTEM_PROMPT}] + history + [
        {"role": "user", "content": content}]
    return msgs, new_img


def image_fingerprint(img) -> str:
    """图像的稳定指纹（用于多模态 block hash 的等价性检查）。"""
    import hashlib

    return hashlib.sha256(img.tobytes()).hexdigest()[:16]


def dump_workload(cfg: WorkloadConfig, out_path: str):
    sessions = build_sessions(cfg)
    with open(out_path, "w") as f:
        json.dump({
            "config": cfg.__dict__,
            "sessions": [s.__dict__ for s in sessions],
        }, f, ensure_ascii=False, indent=2)
    return sessions


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(os.path.join(root, "data"), exist_ok=True)
    cfg = WorkloadConfig()
    ss = dump_workload(cfg, os.path.join(root, "data", "workload_spec.json"))
    n_img = sum(sum(1 for m in s.turn_modes if m > 0) for s in ss)
    print(f"sessions={len(ss)} turns/session={cfg.turns_per_session} "
          f"requests={len(ss)*cfg.turns_per_session} images={n_img}")
