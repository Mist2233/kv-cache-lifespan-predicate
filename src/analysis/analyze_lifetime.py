"""把 KV block 生命周期 trace 变成"每个 token 的寿命"，并出分布图。

输入：trace 目录（requests.jsonl + blocks*.jsonl）
      blocks.jsonl      —— 来自真实 vLLM 事件（derive_blocks.py）
      blocks_sim.jsonl  —— 来自块级缓存模拟器（cache_sim.py）

寿命定义（单位：秒，相对实验开始）：
    t_create       块被写入（prefill 或生成补齐）
    t_last_access  最后一次被读（请求运行期间每 step 都读；命中前缀也算）
    t_death        被淘汰（离开 HBM/前缀缓存）—— 物理上"数据变成垃圾"的时刻
    resident = t_death - t_create      驻留寿命
    active   = t_last_access - t_create 活跃寿命（第一次写到最后一次读）
    idle     = t_death - t_last_access  死区（写完没人再读，但还占着）

输出：
    data/tokens_lifetime_<tag>.csv
    data/lifetime_summary_<tag>.json
    figs/fig02_lifetime_<tag>.png
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
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt  # noqa: E402
import plotstyle  # noqa: E402
from mm_input import MMInput  # noqa: E402

plotstyle.setup()

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(ROOT, "models", "Qwen3.5-9B")

KIND_ORDER = ["system", "image", "user_text", "assistant_text", "generated"]
KIND_COLOR = {
    "system": "#4C72B0",
    "image": "#C44E52",
    "user_text": "#DD8452",
    "assistant_text": "#55A868",
    "generated": "#8172B3",
}
KIND_LABEL = {
    "system": "系统指令词",
    "image": "图像 token",
    "user_text": "用户文本",
    "assistant_text": "历史回复",
    "generated": "本次生成",
}


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def build_pos_kind_table(reqs, mm, bs: int, n_pos: int) -> dict[int, str]:
    """位置 -> 最多的语义类别（在所有请求上投票）。

    图像 token 的 id 全都一样，靠内容无法区分"这是哪张图"；但"第 pos0 个 token
    属于什么模态"在所有会话里是一致的（系统词/图像/对话），所以用位置投票最稳。
    """
    from collections import Counter

    votes: dict[int, Counter] = {p: Counter() for p in range(0, n_pos + 1, bs)}
    for r in reqs:
        ids = r["prompt_token_ids"]
        segs = mm.segment(ids, n_turns=r["turn"] + 1)
        kinds = ["other"] * len(ids)
        for sg in segs:
            for p in range(sg.start, min(sg.end, len(ids))):
                kinds[p] = sg.kind
        for pos0 in votes:
            if pos0 >= len(ids):
                continue
            chunk = [kinds[p] for p in range(pos0, min(pos0 + bs, len(ids)))]
            non_other = [k for k in chunk if k != "other"]
            if non_other:
                votes[pos0][Counter(non_other).most_common(1)[0][0]] += 1
    return {p: c.most_common(1)[0][0] for p, c in votes.items() if c}


def build_position_index(reqs, mm, bs: int = 16):
    """(token tuple) -> [(pos0, req_idx, kind_list)]，用于把块映射回位置与模态。

    vLLM 的块边界从位置 0 起每 16 个 token，所以直接用切片的 token 元组做索引。
    """
    idx: dict[tuple, list[tuple[int, int]]] = defaultdict(list)
    seg_kind: dict[int, list[str]] = {}
    for ri, r in enumerate(reqs):
        ids = r["prompt_token_ids"]
        segs = mm.segment(ids, n_turns=r["turn"] + 1)
        kinds = ["other"] * len(ids)
        for s in segs:
            for p in range(s.start, min(s.end, len(ids))):
                kinds[p] = s.kind
        seg_kind[ri] = kinds
        for pos0 in range(0, len(ids) + 1, bs):
            toks = tuple(ids[pos0:pos0 + bs])
            if len(toks) == bs:
                idx[toks].append((pos0, ri))
    return idx, seg_kind


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace_dir")
    ap.add_argument("--blocks", default="blocks.jsonl")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--model", default=MODEL_DIR)
    args = ap.parse_args()

    trace_dir = os.path.join(ROOT, args.trace_dir) if not os.path.isabs(args.trace_dir) else args.trace_dir
    blocks_path = os.path.join(trace_dir, args.blocks)
    tag = args.tag or os.path.basename(trace_dir)
    meta_path = os.path.join(trace_dir, "meta.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    wall = meta.get("wall_s") or 0.0

    reqs = load_jsonl(os.path.join(trace_dir, "requests.jsonl"))
    blocks = load_jsonl(blocks_path)
    mm = MMInput(args.model)
    from collections import Counter
    bs_counter = Counter(int(b.get("block_size", 16)) for b in blocks)
    bs_global = bs_counter.most_common(1)[0][0] if blocks else 16
    max_len = max(len(r["prompt_token_ids"]) for r in reqs)
    print(f"[bs] 事件里的 block_size 分布: {dict(bs_counter)} -> 采用 {bs_global}")
    pos_kind = build_pos_kind_table(reqs, mm, bs_global, max_len)

    # 命中概率：request 侧 num_cached_tokens 是权威的前缀命中长度，
    # 块位置 pos0 被命中的次数 = 有多少条请求的前缀覆盖了它。
    cached_pairs = [(int(r.get("num_cached_tokens") or 0), len(r["prompt_token_ids"]))
                    for r in reqs]

    def hits_at(pos0: int) -> int:
        return sum(1 for c, l in cached_pairs if c > pos0 and l > pos0)

    print(f"[hit] 有前缀命中的请求 {sum(1 for c, _ in cached_pairs if c > 0)}/{len(reqs)}，"
          f"命中块数中位 {np.median([c // bs_global for c, _ in cached_pairs]):.0f}")

    t_end_all = max(r["t_end"] for r in reqs)
    n_cens = sum(1 for b in blocks if b.get("censored"))
    print(f"requests={len(reqs)} blocks={len(blocks)} censored={n_cens} "
          f"trace_end={t_end_all:.1f}s")

    # ---- 块 -> (位置, 创建者请求, 模态) ----
    rows = []
    seen = set()
    skipped = 0
    for b in blocks:
        bs = int(b.get("block_size", bs_global))
        toks = b.get("tokens") or []
        if bs != bs_global:
            skipped += 1
            continue
        pos0 = int(b.get("pos0", -1))
        if pos0 < 0:
            skipped += 1
            continue
        kind = pos_kind.get(pos0, "other")
        # 轮次：用"创建时刻落在哪条请求的生命周期里"来归属
        t_create_blk = b["t_create"]
        cand = [r for r in reqs
                if r["t_start"] - 0.05 <= t_create_blk <= r["t_end"]
                and pos0 + bs <= len(r["prompt_token_ids"])]
        if cand:
            creator = min(cand, key=lambda r: abs(t_create_blk - r["t_start"]))
            turn_of = creator["turn"]
        else:
            turn_of = -1
        t_create = b["t_create"]
        t_last = max(b.get("t_last_access", t_create), t_create)
        t_death = b.get("t_death")
        censored = bool(b.get("censored")) or t_death is None
        if censored:
            t_death_eff = t_end_all
        else:
            t_death_eff = t_death
        rows.append({
            "key": b["key"],
            "pos0": pos0,
            "kind": kind,
            "sid": -1,
            "turn": turn_of,
            "t_create": t_create,
            "t_last_access": t_last,
            "t_death": t_death_eff,
            "resident": t_death_eff - t_create,
            "active": max(t_last - t_create, 0.0),
            "idle": max(t_death_eff - t_last, 0.0),
            "censored": censored,
            "n_use": b.get("n_use", 1),
            "n_hit": hits_at(pos0),
            "bs": bs,
        })
        seen.add(b["key"])

    # ---- 展开成 per-token（同块 16 个 token 共享寿命）----
    tok_rows = []
    for r in rows:
        tok_rows.append((r, r["bs"]))
    n_tokens = sum(n for _, n in tok_rows)
    print(f"mapped blocks={len(rows)} tokens={n_tokens} skipped={skipped}")

    import pandas as pd

    df = pd.DataFrame(rows)
    if df.empty:
        print("没有可分析的块（事件为空？）")
        return

    # 复用次数 = 用过它的请求数 - 1（事件里等价于重复 store 的次数）
    df["n_reuse"] = (df["n_use"] - 1).clip(lower=0)
    df["hit"] = df["idle"] > 0

    # ---------- 统计 ----------
    def q(x, p):
        return float(np.percentile(x, p)) if len(x) else float("nan")

    summary = {
        "trace_dir": args.trace_dir,
        "blocks": len(df),
        "tokens": int(n_tokens),
        "censored_frac": float(df["censored"].mean()),
        "trace_end_s": t_end_all,
        "resident": {"mean": float(df["resident"].mean()), "p50": q(df["resident"], 50),
                     "p90": q(df["resident"], 90), "p99": q(df["resident"], 99)},
        "active": {"mean": float(df["active"].mean()), "p50": q(df["active"], 50),
                   "p90": q(df["active"], 90)},
        "idle": {"mean": float(df["idle"].mean()), "p50": q(df["idle"], 50)},
        "by_kind": {},
        "n_image_blocks": int((df["kind"] == "image").sum()),
        "n_system_blocks": int((df["kind"] == "system").sum()),
    }
    for k, g in df.groupby("kind"):
        summary["by_kind"][k] = {
            "blocks": int(len(g)),
            "tokens": int(g["bs"].sum()),
            "resident_mean": float(g["resident"].mean()),
            "resident_p50": q(g["resident"], 50),
            "resident_p90": q(g["resident"], 90),
            "censored_frac": float(g["censored"].mean()),
        }

    os.makedirs(os.path.join(ROOT, "data"), exist_ok=True)
    out_csv = os.path.join(ROOT, "data", f"tokens_lifetime_{tag}.csv")
    df.to_csv(out_csv, index=False)
    out_json = os.path.join(ROOT, "data", f"lifetime_summary_{tag}.json")
    json.dump(summary, open(out_json, "w"), ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    plot(df, tag, t_end_all, meta)
    plot_by_turn(df, tag)


def plot_by_turn(df, tag):
    """按轮次分层：后面的轮次，历史越长、越容易被挤掉，寿命是不是更短？"""
    d = df[df["turn"] >= 0]
    if d.empty:
        return
    turns = sorted(d["turn"].unique())
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    ax = axes[0]
    data = [d[d["turn"] == t]["resident"].values for t in turns]
    bp = ax.boxplot(data, tick_labels=[f"turn{t+1}" for t in turns], showfliers=False,
                    patch_artist=True)
    for patch, t in zip(bp["boxes"], turns):
        patch.set_facecolor(KIND_COLOR["image"] if t % 2 else KIND_COLOR["system"])
        patch.set_alpha(0.55)
    ax.set_ylabel("驻留寿命 (s)")
    ax.set_title("(a) 按轮次：寿命箱线图（无离群点）", fontsize=10)

    ax2 = axes[1]
    med = [np.median(d[d["turn"] == t]["resident"]) for t in turns]
    cen = [d[d["turn"] == t]["censored"].mean() * 100 for t in turns]
    bars = ax2.bar([f"turn{t+1}" for t in turns], med, color="#7F9AC8")
    for i, (v, c) in enumerate(zip(med, cen)):
        ax2.text(i, v + max(med) * 0.03, f"{v:.1f}s\n删失{c:.0f}%",
                 ha="center", va="bottom", fontsize=8)
    ax2.set_ylim(0, max(med) * 1.35)
    ax2.set_ylabel("寿命中位数 (s)")
    ax2.set_title("(b) 各轮寿命中位数（标注右删失占比）", fontsize=10)

    ax3 = axes[2]
    nuse = [d[d["turn"] == t]["n_reuse"].mean() for t in turns]
    nblk = [len(d[d["turn"] == t]) for t in turns]
    ax3.bar([f"turn{t+1}" for t in turns], nuse, color="#55A868")
    for i, (v, n) in enumerate(zip(nuse, nblk)):
        ax3.text(i, v, f"{v:.2f}\n(n={n})", ha="center", va="bottom", fontsize=8)
    ax3.set_ylabel("平均复用次数")
    ax3.set_title("(c) 各轮块的跨请求复用次数", fontsize=10)

    fig.suptitle(f"按轮次分层的 KV 寿命（{tag}）", fontsize=13)
    out = os.path.join(ROOT, "figs", f"fig02b_lifetime_by_turn_{tag}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[fig] {out}")


def plot(df, tag, t_end, meta):
    fig = plt.figure(figsize=(15.5, 9))
    gs = fig.add_gridspec(2, 3, hspace=0.36, wspace=0.28)

    # ---- (a) 寿命 ECDF（按 token 加权）----
    ax = fig.add_subplot(gs[0, 0])
    for kind in KIND_ORDER:
        g = df[df["kind"] == kind]
        if g.empty:
            continue
        w = g["bs"].to_numpy(dtype=float)
        order = np.argsort(g["resident"].to_numpy())
        x = g["resident"].to_numpy()[order]
        y = np.cumsum(w[order]) / w.sum()
        ax.plot(x, y, label=KIND_LABEL.get(kind, kind), color=KIND_COLOR.get(kind), lw=1.8)
    x = np.sort(df["resident"].values)
    y = np.arange(1, len(x) + 1) / len(x)
    ax.plot(x, y, color="k", ls="--", lw=1.2, label="全部 block")
    ax.set_xlabel("驻留寿命 resident = 淘汰时刻 − 写入时刻 (s)")
    ax.set_ylabel("累积占比（按 token 数）")
    ax.set_title("(a) 每个 token 的寿命 ECDF", fontsize=10)
    ax.legend(fontsize=7, frameon=False)
    ax.set_xlim(left=0)

    # ---- (b) 寿命直方图（按类别堆叠）----
    ax2 = fig.add_subplot(gs[0, 1])
    bins = np.linspace(0, max(df["resident"].max(), 1e-3), 40)
    bottom = np.zeros(len(bins) - 1)
    for kind in KIND_ORDER:
        g = df[df["kind"] == kind]
        if g.empty:
            continue
        h, _ = np.histogram(g["resident"], bins=bins, weights=g["bs"].astype(float))
        ax2.bar(bins[:-1], h, width=np.diff(bins), align="edge", bottom=bottom,
                color=KIND_COLOR.get(kind), label=KIND_LABEL.get(kind, kind))
        bottom += h
    ax2.set_xlabel("驻留寿命 (s)")
    ax2.set_ylabel("token 数")
    ax2.set_title("(b) 寿命分布（按 token 归属堆叠）", fontsize=10)
    ax2.legend(fontsize=7, frameon=False)

    # ---- (c) 位置 vs 寿命 ----
    ax3 = fig.add_subplot(gs[0, 2])
    for kind in KIND_ORDER:
        g = df[df["kind"] == kind]
        if g.empty:
            continue
        ax3.scatter(g["pos0"], g["resident"], s=6, alpha=0.55,
                    color=KIND_COLOR.get(kind), label=KIND_LABEL.get(kind, kind))
    ax3.set_xlabel("块在请求中的位置（token 下标）")
    ax3.set_ylabel("驻留寿命 (s)")
    ax3.set_title("(c) 越靠前的块（系统词/先出现的图）活得越久？", fontsize=10)
    ax3.legend(fontsize=7, frameon=False, markerscale=2)

    # ---- (d)(e) 复现华为 baseline：按"命中次数/概率"分温冷热三层 ----
    df2 = df.copy()
    # 复用次数 = 用过它的请求数 - 1
    df2["n_reuse"] = (df2["n_use"] - 1).clip(lower=0)
    # 复现华为 baseline：按"命中概率"分温冷热三层（这里用块位置被命中的次数）
    df2["tier"] = df2["n_hit"].map(lambda v: 2 if v >= 3 else (1 if v >= 1 else 0))
    colors = ["#4C72B0", "#DD8452", "#C44E52"]
    tier_names = ["冷(命中0次)", "温(命中1-2次)", "热(命中≥3次)"]

    ax4 = fig.add_subplot(gs[1, 0])
    devs = {}
    for t in range(3):
        g = df2[df2["tier"] == t]
        if g.empty:
            continue
        center = g["resident"].mean()
        dev = np.abs(g["resident"] - center) / max(center, 1e-9)
        devs[tier_names[t]] = {
            "blocks": int(len(g)), "tokens": int(g["bs"].sum()),
            "center_s": float(center),
            "mean_dev": float(dev.mean()), "p95_dev": float(np.percentile(dev, 95)),
            "frac_within_5pct": float((dev <= 0.05).mean()),
        }
        ax4.scatter(g["pos0"] + np.random.default_rng(t).normal(0, 12, len(g)),
                    g["resident"], s=6, color=colors[t], alpha=0.5,
                    label=f"{tier_names[t]} n={len(g)}")
        ax4.axhline(center, color=colors[t], ls="--", lw=1)
    ax4.set_xlabel("块位置")
    ax4.set_ylabel("驻留寿命 (s)")
    ax4.set_title("(d) 按命中概率分层：层内寿命（虚线=层均值）", fontsize=10)
    ax4.legend(fontsize=7, frameon=False, markerscale=2)

    ax5 = fig.add_subplot(gs[1, 1])
    for t in range(3):
        g = df2[df2["tier"] == t]
        if g.empty:
            continue
        center = g["resident"].mean()
        dev = np.abs(g["resident"] - center) / max(center, 1e-9)
        ax5.hist(np.clip(dev, 0, 2), bins=40, alpha=0.55, color=colors[t],
                 label=f"{tier_names[t]} n={len(g)}")
    ax5.axvline(0.05, color="k", ls="--", lw=1.2)
    ax5.text(0.06, ax5.get_ylim()[1] * 0.88, "±5% 验收线", fontsize=8)
    ax5.set_xlabel("|寿命 − 层均值| / 层均值")
    ax5.set_ylabel("块数")
    ax5.set_title("(e) 相对偏差：命中概率分层离 ±5% 有多远？", fontsize=10)
    ax5.legend(fontsize=8, frameon=False)

    # 参考下界：用真实寿命做 oracle 三等分，偏差都收敛不了就说明"寿命本身"太散
    qs = np.quantile(df2["resident"], [1 / 3, 2 / 3])
    df2["oracle_tier"] = np.digitize(df2["resident"], qs)
    oracle = {}
    for t, nm in enumerate(tier_names):
        g = df2[df2["oracle_tier"] == t]
        if g.empty:
            continue
        center = g["resident"].mean()
        dev = np.abs(g["resident"] - center) / max(center, 1e-9)
        oracle[nm] = {"blocks": int(len(g)), "center_s": float(center),
                      "frac_within_5pct": float((dev <= 0.05).mean()),
                      "p95_dev": float(np.percentile(dev, 95))}
    devs["_oracle_quantile"] = oracle

    # ---- (f) 关键数字 ----
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis("off")
    lines = [
        f"trace: {tag}",
        f"请求数: {meta.get('n_requests', '?')}   会话数: {meta.get('workload', {}).get('n_sessions', '?')}",
        f"块数: {len(df)}   token 数: {int(df['bs'].sum())}",
        f"右删失块占比: {df['censored'].mean()*100:.1f}%",
        f"寿命 p50 / p90: {df['resident'].median():.1f}s / {df['resident'].quantile(0.9):.1f}s",
        "",
    ]
    for k, v in devs.items():
        if k.startswith("_"):
            continue
        lines.append(f"{k}: 均值{v['center_s']:.1f}s  在±5%内 "
                     f"{v['frac_within_5pct']*100:.1f}%  p95偏差{v['p95_dev']*100:.0f}%")
    if "_oracle_quantile" in devs:
        o = devs["_oracle_quantile"]
        inside = ", ".join(f"{k.split('(')[0]}{v['frac_within_5pct']*100:.0f}%"
                           for k, v in o.items())
        lines.append("")
        lines.append(f"oracle 真值三等分（无法预测，仅作下界）: {inside}")
        lines.append("=> 即便按真值分层，层内偏差依然超标，说明「三层」粒度不够")
    ax6.text(0.0, 0.95, "\n".join(lines), va="top", fontsize=8.5)

    fig.suptitle(f"每个 token 的 KV 寿命分布（{tag}）", fontsize=13)
    out = os.path.join(ROOT, "figs", f"fig02_lifetime_{tag}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[fig] {out}")
    json.dump(devs, open(os.path.join(ROOT, "data", f"tier_dev_{tag}.json"), "w"),
              ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
