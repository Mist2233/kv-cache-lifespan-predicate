"""实验 B：共享前缀块的寿命按"最晚结束的会话"计算，能否消除归属歧义？

做法：
  1. 对每个请求的每个块对齐位置，把 token 片段哈希 → 记录"哪些会话用过这个块"；
  2. 块的寿命 = max(用过它的会话的结束时刻) − 淘汰时刻；
  3. 对比新旧寿命分布与预测效果。
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _setup import add_src_paths  # noqa: E402
add_src_paths()
import sys, os, json, hashlib, numpy as np, pandas as pd
import classify_buckets as cb
from cluster_lifetime import load, attach
from classify_buckets import build
from features_v2 import FEATURES_V2, add_features_v2
from fusion_v3 import cut, eta2_log
from gc_sim import Device, max_live_bytes
from metrics import summarize
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GIB = 1 << 30

def h(tokens):
    return hashlib.md5(np.asarray(tokens, dtype=np.int64).tobytes()).hexdigest()[:16]

def main():
    td = sys.argv[1] if len(sys.argv) > 1 else "data/trace_openomni_v3"
    reqs, blocks = load(td if os.path.isabs(td) else os.path.join(ROOT, td))
    bs = int(blocks[0].get("block_size", 512))
    se = {}
    for r in reqs: se[r["sid"]] = max(se.get(r["sid"], 0), r["t_end"])

    # 块 token -> 用过它的会话集合
    users = {}
    for r in reqs:
        ids = r["prompt_token_ids"]; sid = r["sid"]
        for pos in range(0, len(ids) - bs + 1, bs):
            k = h(ids[pos:pos + bs])
            users.setdefault(k, set()).add(sid)

    rows, _ = attach(reqs, blocks)
    df = pd.DataFrame(rows)
    # 给每个块算"最晚会话结束时刻"
    bl_by_key = {b["key"]: b for b in blocks}
    max_end, n_sess = {}, {}
    for b in blocks:
        k = h(b.get("tokens") or [])
        ss_ = users.get(k, set())
        n_sess[b["key"]] = len(ss_)
        max_end[b["key"]] = max((se[s] for s in ss_), default=None)
    # attach 已把 key 丢了？用 t_create+pos0 近似匹配回 key —— 改为在 rows 里补
    # rows 里没有 key，用 (sid,turn,pos0) 反查不到；这里重新按 key 建表再 merge
    df = df[df["key"].notna()].copy()
    df["n_sess_share"] = df["key"].map(lambda k: n_sess.get(k, 1)).fillna(1).astype(int)
    mt = df["key"].map(lambda k: max_end.get(k))
    df["L_A_old"] = df.sid.map(se) - df.t_death
    df["L_A_new"] = [m - td_ if m is not None else np.nan
                     for m, td_ in zip(mt, df.t_death)]
    print(f"能找到共享信息的块 {df.L_A_new.notna().sum()}/{len(df)}")
    df["L_A_new"] = df["L_A_new"].fillna(df["L_A_old"])   # 找不到则退回旧定义
    df = df[df.L_A_old > 0].copy()
    n_shared = int((df.n_sess_share > 1).sum())
    print(f"块 {len(df)}，其中被多个会话共享的 {n_shared} ({100*n_shared/len(df):.1f}%)")
    chg = (df.L_A_new - df.L_A_old) / df.L_A_old
    print(f"共享块寿命变化：中位 {np.median(chg[df.n_sess_share>1])*100:+.0f}%  "
          f"（旧寿命中位 {np.median(df.L_A_old):.0f}s → 新 {np.median(df.L_A_new):.0f}s）")
    df = add_features_v2(df, reqs)
    base = [f for f in FEATURES_V2]
    size = int(bs * 32 * 1024)

    def run(col, tag):
        y = np.log(np.maximum(df[col].values, 1e-3)); g = df.sid.values
        X = df[base].astype(float).replace([np.inf, -np.inf], 0).fillna(0)
        X = ((X - X.mean()) / X.std().replace(0, 1)).values
        score = np.zeros(len(X)); rng = np.random.default_rng(0)
        for tr, te in GroupKFold(4).split(X, y, groups=g):
            i = rng.integers(0, len(tr), 30000); j = rng.integers(0, len(tr), 30000)
            m = y[tr][i] != y[tr][j]; i, j = i[m], j[m]
            lm = LogisticRegression(max_iter=2000).fit(X[tr][i]-X[tr][j], (y[tr][i]>y[tr][j]).astype(int))
            score[te] = X[te] @ lm.coef_[0]
        lab = cut(score, 12); m = summarize(lab, df[col].values)
        df["L_A"] = df[col].values
        cb.PASSES = 32; ch_ = build(df, lab, size)
        live = max_live_bytes(ch_); cap = live/0.74/GIB
        w = Device(ch_, capacity_gb=cap, ru_mb=6144, op_frac=0.07).run()["wa_steady"]
        print(f"  {tag:<30} η²={eta2_log(lab, df[col].values):.3f} IBRD50={m['ibrd50']*100:5.1f}% WA={w:.3f}")

    print()
    for col, tag in (("L_A_old", "寿命=所属会话（旧）"), ("L_A_new", "寿命=最晚会话（新）")):
        df["L_A"] = df[col].values
        cb.PASSES = 32
        ch0 = build(df, np.zeros(len(df), dtype=int), size)
        live0 = max_live_bytes(ch0); cap0 = live0 / 0.74 / GIB
        w0 = Device(ch0, capacity_gb=cap0, ru_mb=6144, op_frac=0.07).run()["wa_steady"]
        print(f"  [{tag}] 不分层 WA={w0:.3f}  (RU {int(cap0/6)})")
        run(col, "  └ 我们的方法")

if __name__ == "__main__":
    main()
