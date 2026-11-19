"""扩展特征（A 方案第 1 步 / C 部分）。

只加入**写入时刻可观测**的量：
  age_ratio        会话年龄 / 该 trace 的会话时长中位数
  turn_ratio       当前轮次 / 轮数中位数
  prompt_len_ratio 当前 prompt 长度 / 长度中位数
  arrival_rate     最近 30 s 内新到达的请求数（负载强度）
  img_token_ratio  本请求中图像 token 占比
  gen_total        本会话此前累计生成的 token 数
  gen_per_turn     平均每轮生成长度

注：age_ratio / turn_ratio / prompt_len_ratio 里的"中位数"是常数。
对树模型而言，单个特征乘一个正常数不改变分裂点，因此常数是否来自训练折
不影响结果（等价性由单调变换保证）。
"""

from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _setup import add_src_paths  # noqa: E402
add_src_paths()

import numpy as np
import pandas as pd

from cluster_lifetime import FEATURES

NEW_FEATURES = [
    "age_ratio", "turn_ratio", "prompt_len_ratio",
    "arrival_rate", "img_token_ratio", "gen_total", "gen_per_turn",
]
FEATURES_V2 = FEATURES + NEW_FEATURES


# v3：删掉 v2 里无效的 3 个"常数倍"特征（age_ratio/turn_ratio/prompt_len_ratio），
# 换成真正的非线性 / 交互特征。
NEW_FEATURES_V3 = [
    "arrival_rate", "img_token_ratio", "gen_total", "gen_per_turn",
    "idle_over_age", "growth_ratio", "img_pos_mean", "blocks_per_turn",
    "req_rate_session", "n_img_total", "turn_x_age", "last_img_frac",
]
FEATURES_V3 = FEATURES + NEW_FEATURES_V3


def add_features_v3(df: pd.DataFrame, reqs: list[dict]) -> pd.DataFrame:
    """v3：保留 v2 中有效的 4 个，删掉无效的 3 个，新增 8 个交互/非线性特征。"""
    df = add_features_v2(df, reqs)

    sess = {}
    for r in reqs:
        s = sess.setdefault(r["sid"], {"t0": r["t_start"], "t1": r["t_end"], "turns": 0})
        s["t0"] = min(s["t0"], r["t_start"])
        s["t1"] = max(s["t1"], r["t_end"])
        s["turns"] = max(s["turns"], r["turn"] + 1)
    dur_med = float(np.median([s["t1"] - s["t0"] for s in sess.values()]))
    turn_med = float(np.median([s["turns"] for s in sess.values()]))

    by_sid = {}
    for r in reqs:
        by_sid.setdefault(r["sid"], []).append(r)
    n_img_cum, img_center = {}, {}
    for sid, rs in by_sid.items():
        rs.sort(key=lambda r: r["turn"])
        acc = 0
        for r in rs:
            n_img_cum[(sid, r["turn"])] = acc
            acc += int(r.get("n_images") or 0)
            spans = r.get("image_spans") or []
            if spans:
                img_center[(sid, r["turn"])] = float(
                    np.mean([(sg[0] + sg[1]) / 2 for sg in spans]))
            else:
                img_center[(sid, r["turn"])] = -1.0

    df["idle_over_age"] = df["idle_before"] / (df["session_age"] + 1.0)
    df["growth_ratio"] = df["prompt_growth"] / (df["prompt_len"] + 1.0)
    df["img_pos_mean"] = [
        img_center.get((s, t), -1.0) / max(p, 1)
        for s, t, p in zip(df["sid"], df["turn"], df["prompt_len"])]
    df["blocks_per_turn"] = df["blocks_in_req"] / (df["turn"] + 1.0)
    df["req_rate_session"] = df["turn"] / (df["session_age"] + 1.0)
    df["n_img_total"] = [n_img_cum.get((s, t), 0) for s, t in zip(df["sid"], df["turn"])]
    df["turn_x_age"] = (df["turn"] / max(turn_med, 1.0)) * (df["session_age"] / max(dur_med, 1e-9))
    df["last_img_frac"] = [
        (max(p - max(0, int(v)), 0) / max(p, 1) if v >= 0 else -1.0)
        for p, v in zip(df["prompt_len"], df["pos_from_img"])]
    return df


def add_features_v2(df: pd.DataFrame, reqs: list[dict]) -> pd.DataFrame:
    df = df.copy()

    # --- 会话级常数（顺序统计量，树模型不受影响）---
    sess = {}
    for r in reqs:
        s = sess.setdefault(r["sid"], {"t0": r["t_start"], "t1": r["t_end"],
                                       "turns": 0, "plen": []})
        s["t0"] = min(s["t0"], r["t_start"])
        s["t1"] = max(s["t1"], r["t_end"])
        s["turns"] = max(s["turns"], r["turn"] + 1)
        s["plen"].append(len(r["prompt_token_ids"]))
    dur_med = float(np.median([s["t1"] - s["t0"] for s in sess.values()]))
    turn_med = float(np.median([s["turns"] for s in sess.values()]))
    plen_med = float(np.median([p for s in sess.values() for p in s["plen"]]))

    # --- 请求级：累计生成、图像 token ---
    by_sid: dict[int, list[dict]] = {}
    for r in reqs:
        by_sid.setdefault(r["sid"], []).append(r)
    gen_cum, img_tok = {}, {}
    for sid, rs in by_sid.items():
        rs.sort(key=lambda r: r["turn"])
        acc = 0
        for r in rs:
            gen_cum[(sid, r["turn"])] = acc
            acc += int(r.get("gen_len") or 0)
            spans = r.get("image_spans") or []
            img_tok[(sid, r["turn"])] = sum(max(0, sg[1] - sg[0]) for sg in spans)

    starts = np.array(sorted(r["t_start"] for r in reqs))

    def arrival_rate(t: float, win: float = 30.0) -> float:
        return float(np.searchsorted(starts, t + 1e-9) - np.searchsorted(starts, t - win))

    df["age_ratio"] = df["session_age"] / max(dur_med, 1e-9)
    df["turn_ratio"] = df["turn"] / max(turn_med, 1.0)
    df["prompt_len_ratio"] = df["prompt_len"] / max(plen_med, 1.0)
    df["arrival_rate"] = [arrival_rate(t) for t in df["t_create"].values]
    df["gen_total"] = [gen_cum.get((s, t), 0) for s, t in zip(df["sid"], df["turn"])]
    df["gen_per_turn"] = df["gen_total"] / np.maximum(df["turn"].values, 1)
    df["img_token_ratio"] = [
        img_tok.get((s, t), 0) / max(p, 1)
        for s, t, p in zip(df["sid"], df["turn"], df["prompt_len"])
    ]
    return df
