"""会话级 hazard（风险率）模型：预测"会话还能持续几轮/多久"。

问题重述
--------
在块写入 SSD 的时刻（= 显存淘汰时刻），要预测
    L_A = 会话结束时刻 − 当前时刻
即"这个会话还能聊多久"。这是标准的离散时间生存分析问题。

方法
----
1. 在**轮次粒度**上构造生存数据：每个 (会话, 轮次 t) 一条样本，
   标签 = 会话在 t 轮之后是否还有下一轮；
2. 用可观测特征训练 hazard 模型：h(t, x) = P(继续 | 历史)；
3. 由 hazard 推导"期望剩余轮数"：
      E[rem | t] ≈ h(t) · E[rem_turns | reached t+1]
   其中后半部分用训练集的边际 hazard 曲线估计；
4. 乘上平均每轮耗时 → 期望剩余时长 → 减去本轮已过去的时长 → 预测 L_A；
5. 按预测 L_A 分位切桶，测 IBRD / WA。

对照：
  * 轮次先验（只用 E[remaining|t]）
  * 现有 A3 排序模型

用法：python src/session_hazard.py
"""

from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _setup import add_src_paths  # noqa: E402
add_src_paths()

import os
import sys

import numpy as np
import pandas as pd
import classify_buckets as cb  # noqa: E402
from cluster_lifetime import add_kind, attach, load  # noqa: E402
from classify_buckets import build  # noqa: E402
from features_v2 import FEATURES_V2, add_features_v2  # noqa: E402
from fusion_v3 import cut, eta2_log, oof_rank, wa_of  # noqa: E402
from gc_sim import max_live_bytes  # noqa: E402
from metrics import summarize  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
KB_PER_TOKEN = 32
GIB = 1 << 30
UTIL = 0.74


# ----------------------------------------------------------------- 会话级数据
def session_table(reqs):
    """每个 (会话, 轮次) 一行，含可观测行为特征与 hazard 标签。"""
    by = {}
    for r in reqs:
        by.setdefault(r["sid"], []).append(r)
    rows = []
    for sid, rs in by.items():
        rs.sort(key=lambda r: r["turn"])
        T = len(rs)
        t0 = rs[0]["t_start"]
        gaps, gens, imgs = [], [], 0
        for t, r in enumerate(rs):
            imgs += int(r.get("n_images") or 0)
            rows.append({
                "sid": sid, "turn": t, "T": T,
                "continue": int(t < T - 1),
                "elapsed": r["t_start"] - t0,
                "gap_prev": (r["t_start"] - rs[t - 1]["t_end"]) if t > 0 else 0.0,
                "gap_mean": float(np.mean(gaps)) if gaps else 0.0,
                "gen_mean": float(np.mean(gens)) if gens else 0.0,
                "gen_prev": gens[-1] if gens else 0,
                "prompt_len": len(r["prompt_token_ids"]),
                "n_img_cum": imgs,
                "n_img_this": int(r.get("n_images") or 0),
                # 该轮剩余时间（从轮次开始算）
                "rem_time": max(rs[-1]["t_end"] - r["t_start"], 0.0),
                # 该轮时长
                "turn_dur": max(r["t_end"] - r["t_start"], 0.1),
            })
            gaps.append(max(r["t_start"] - (rs[t - 1]["t_end"] if t > 0 else r["t_start"]), 0.0))
            gens.append(int(r.get("gen_len") or 0))
    return pd.DataFrame(rows)


HFEATS = ["turn", "elapsed", "gap_prev", "gap_mean", "gen_mean", "gen_prev",
          "prompt_len", "n_img_cum", "n_img_this"]


def fit_hazard(stab, seed=0):
    """OOF hazard 预测 + 训练集的边际 hazard 曲线。"""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import GroupKFold
    X = stab[HFEATS].astype(float).values
    y = stab["continue"].values
    g = stab["sid"].values
    p = np.zeros(len(stab))
    for tr, te in GroupKFold(4).split(X, y, groups=g):
        m = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.08,
                                           max_iter=200, random_state=seed)
        m.fit(X[tr], y[tr])
        p[te] = m.predict_proba(X[te])[:, 1]
    # 边际：P(继续 | 已到第 t 轮) —— 用全体数据估计
    marg = stab.groupby("turn")["continue"].mean().to_dict()
    return p, marg


def expected_rem_turns(t, T_max, marg):
    """从第 t 轮之后还能持续多少轮（边际曲线积分）。"""
    e, surv = 0.0, 1.0
    for k in range(t + 1, T_max + 1):
        surv *= marg.get(k - 1, 0.0)      # 在第 k-1 轮后继续的概率
        if surv < 1e-6:
            break
        e += surv
    return e


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", nargs="*", default=None)
    a = ap.parse_args()
    if a.traces:
        pairs = [(x, os.path.basename(x.rstrip("/"))) for x in a.traces]
    else:
        pairs = [("data/trace_docvqa36", "docvqa36"),
                 ("data/trace_big", "trace_big")]
    for t, tag in pairs:
        p = os.path.join(ROOT, t)
        if not os.path.isdir(p):
            continue
        reqs, blocks = load(p)
        bs = int(blocks[0].get("block_size", 528))
        from mm_input import MMInput
        rows, _ = attach(reqs, blocks)
        rows = add_kind(rows, reqs, MMInput(os.path.join(ROOT, "models", "Qwen3.5-9B")),
                        bs, max(len(r["prompt_token_ids"]) for r in reqs))
        df = pd.DataFrame(rows)
        se = {}
        for r in reqs:
            se[r["sid"]] = max(se.get(r["sid"], 0.0), r["t_end"])
        df = df[df["t_death"].notna()].copy()
        df["L_A"] = df["sid"].map(se) - df["t_death"]
        df = df[df["L_A"] > 0].reset_index(drop=True)
        df = add_features_v2(df, reqs)
        n = len(df)
        cb.PASSES = max(12, 200_000 // n)
        size = int(bs * KB_PER_TOKEN * 1024)
        y = df["L_A"].values
        c0 = build(df, np.zeros(n, dtype=int), size)
        live = max_live_bytes(c0)
        cap = live / UTIL
        K = int(np.clip(int(cap / (6 * GIB)) / 7, 3, 16))

        # ---- 会话级 hazard ----
        stab = session_table(reqs)
        hp, marg = fit_hazard(stab)
        # 经验条件期望：E[剩余时长 | 当前轮次 t]（用全体数据估计，作为时间换算）
        rem_time_by_turn = stab.groupby("turn")["rem_time"].mean().to_dict()
        stab["h"] = hp
        # 每轮平均耗时（用于轮→秒）
        turn_dur = float(np.median(stab["turn_dur"]))
        # 每个轮次的期望剩余时长（s）
        T_max = int(stab["T"].max())
        rem_cache = {tt: expected_rem_turns(tt, T_max, marg) * turn_dur
                     for tt in range(T_max)}
        # 用 hazard 在当前轮次的预测，去调整"下一轮的期望剩余时长"
        stab["rem_hat"] = [
            hp[i] * rem_time_by_turn.get(int(stab["turn"].iloc[i]) + 1,
                                         rem_time_by_turn.get(int(stab["turn"].iloc[i]), turn_dur))
            for i in range(len(stab))]
        hmap = {(r.sid, r.turn): r.rem_hat for r in stab.itertuples()}

        # 块的预测 L_A：期望剩余时长 − 本轮已过去的时间
        #   本轮已过去时间 ≈ 淘汰时刻 − 该轮开始时刻（用 elapsed 近似）
        req_start = {(r["sid"], r["turn"]): r["t_start"] for r in reqs}
        pred_L = []
        for r in df.itertuples():
            base = hmap.get((r.sid, r.turn), rem_time_by_turn.get(int(r.turn), turn_dur))
            elapsed_in_turn = max(r.t_death - req_start.get((int(r.sid), int(r.turn)),
                                                           r.t_death), 0.0)
            pred_L.append(max(base - elapsed_in_turn, 1e-3))
        pred_L = np.array(pred_L)

        # ---- 会话级指标 ----
        from scipy.stats import spearmanr
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(stab["continue"], stab["h"])
        rho = spearmanr(stab["rem_hat"], stab["rem_time"]).statistic
        print(f"\n{'='*66}\n[{tag}] 块 {n}，会话 {len(stab['sid'].unique())}，K={K}")
        print(f"  会话级：hazard AUC={auc:.3f}，剩余时长 Spearman={rho:+.3f}，"
              f"平均轮时长={turn_dur:.1f}s")

        # ---- 分桶评估 ----
        print(f"  {'方案':<28}{'η²':>8}{'IBRD50':>9}{'WA':>8}")
        turn_prior = df["turn"].map(
            lambda tt: rem_time_by_turn.get(int(tt),
                                            rem_time_by_turn.get(int(T_max - 1), turn_dur))).values
        cases = {
            "轮次先验": cut(turn_prior, K),
            "hazard 生存模型": cut(pred_L, K),
            "A3 当前方法": cut(oof_rank(df, FEATURES_V2, K), K),
        }
        for name, lab in cases.items():
            m = summarize(lab, y)
            try:
                w = wa_of(df, lab, size, cap / GIB)
            except RuntimeError:
                w = float("nan")
            print(f"  {name:<28}{eta2_log(lab,y):>8.3f}{m['ibrd50']*100:>8.1f}%{w:>8.3f}")


if __name__ == "__main__":
    main()
