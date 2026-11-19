"""Omni trace 最终分析：共享前缀块按"最晚使用它的会话结束"计算寿命 L_A。

L_A = max(使用该块的会话结束时刻) − 块淘汰时刻
  - 用 token 序列把块映射到所有使用过它的会话（前缀缓存共享）；
  - 找不到共享信息时退回"所属会话"的结束时刻。
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _setup import add_src_paths  # noqa: E402
add_src_paths()
import sys, os, hashlib, numpy as np, pandas as pd
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
RU_MB = 6144.0
UTIL = 0.74

def h(tokens):
    return hashlib.md5(np.asarray(tokens, dtype=np.int64).tobytes()).hexdigest()[:16]

def assign_kind(df, reqs):
    spans = {}
    for r in reqs:
        spans[(r["sid"], r["turn"])] = (
            [(a, b) for a, b, *_ in r.get("image_spans") or []],
            [(a, b) for a, b in r.get("audio_spans") or []])
    kinds = []
    for r in df.itertuples():
        img, aud = spans.get((r.sid, r.turn), ([], []))
        p = int(r.pos0)
        kinds.append("image" if any(a <= p < b for a, b in img)
                     else ("audio" if any(a <= p < b for a, b in aud) else "text"))
    return np.array(kinds)

def main():
    td = sys.argv[1] if len(sys.argv) > 1 else "data/trace_openomni_v3"
    reqs, blocks = load(td if os.path.isabs(td) else os.path.join(ROOT, td))
    bs = int(blocks[0].get("block_size", 512))
    se = {}
    for r in reqs: se[r["sid"]] = max(se.get(r["sid"], 0), r["t_end"])
    # 块 -> 使用过它的会话集合
    users = {}
    for r in reqs:
        ids = r["prompt_token_ids"]
        for pos in range(0, len(ids) - bs + 1, bs):
            users.setdefault(h(ids[pos:pos + bs]), set()).add(r["sid"])
    max_end = {b["key"]: (max((se[s] for s in users.get(h(b.get("tokens") or []), set())),
                             default=None)) for b in blocks}

    rows, _ = attach(reqs, blocks)
    df = pd.DataFrame(rows)
    df = df[df.t_death.notna()].copy()
    fallback = df.sid.map(se)
    mend = [max_end.get(k) if max_end.get(k) is not None else fb
            for k, fb in zip(df.key, fallback)]
    df["L_A"] = np.array(mend) - df.t_death.values
    df = df[df.L_A > 0].reset_index(drop=True)
    df["kind"] = assign_kind(df, reqs)
    df = add_features_v2(df, reqs)
    n = len(df)
    import os as _os
    cb.PASSES = int(_os.environ.get("PASSES", 32))
    size = int(bs * 32 * 1024)
    c0 = build(df, np.zeros(n, dtype=int), size)
    live = max_live_bytes(c0); cap = live / UTIL; n_ru = int(cap / (RU_MB * (1 << 20)))
    print(f"[{os.path.basename(td)}] 块 {n}，会话 {df.sid.nunique()}，passes={cb.PASSES}")
    print(f"  寿命 中位 {np.median(df.L_A):.0f}s / p10 {np.percentile(df.L_A,10):.0f}s / "
          f"p90 {np.percentile(df.L_A,90):.0f}s，模态 {df.kind.value_counts().to_dict()}")
    print(f"  活跃 {live/GIB:.0f}GiB → 盘 {cap/GIB:.0f}GiB，RU {n_ru}")

    X = df[FEATURES_V2].astype(float).copy()
    for k in ("image", "audio", "text"):
        X[f"kind_{k}"] = (df["kind"] == k).astype(float)
    X = X.replace([np.inf, -np.inf], 0).fillna(0)
    X = ((X - X.mean()) / X.std().replace(0, 1)).values
    y = np.log(np.maximum(df.L_A.values, 1e-3)); g = df.sid.values
    score = np.zeros(n); rng = np.random.default_rng(0)
    for tr, te in GroupKFold(4).split(X, y, groups=g):
        i = rng.integers(0, len(tr), 30000); j = rng.integers(0, len(tr), 30000)
        m = y[tr][i] != y[tr][j]; i, j = i[m], j[m]
        lm = LogisticRegression(max_iter=2000).fit(X[tr][i]-X[tr][j], (y[tr][i]>y[tr][j]).astype(int))
        score[te] = X[te] @ lm.coef_[0]

    def sim(lab):
        ch = build(df, lab, size); lv = max_live_bytes(ch); cp = lv / UTIL / GIB
        return Device(ch, capacity_gb=cp, ru_mb=RU_MB, op_frac=0.07).run()["wa_steady"]

    print(f"\n{'方案':<24}{'η²':>7}{'IBRD50':>9}{'WA':>8}")
    w0 = sim(np.zeros(n, dtype=int)); print(f"{'不分层':<24}{0.0:>7.3f}{summarize(np.zeros(n,int), df.L_A.values)['ibrd50']*100:>8.1f}%{w0:>8.3f}")
    for K in (16, 24, 32, 40, 48, 56):
        if K > n_ru // 3: continue
        lab = cut(score, K); m = summarize(lab, df.L_A.values)
        print(f"{f'A3 排序 (K={K})':<24}{eta2_log(lab, df.L_A.values):>7.3f}{m['ibrd50']*100:>8.1f}%{sim(lab):>8.3f}")
    kb = min(32, n_ru // 3)
    orc = pd.factorize(np.searchsorted(np.quantile(y, np.linspace(0,1,kb+1)[1:-1]), y, side="right"))[0]
    m = summarize(orc, df.L_A.values)
    print(f"{f'oracle (K={kb})':<24}{eta2_log(orc, df.L_A.values):>7.3f}{m['ibrd50']*100:>8.1f}%{sim(orc):>8.3f}")

if __name__ == "__main__":
    main()
