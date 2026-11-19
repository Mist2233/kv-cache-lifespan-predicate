"""自适应 K 验证：数据规模变小时，最优 K 是否跟着变小？

做法：对 docvqa36 按会话数做子采样（36/24/18/12/6 个会话），
每个子集上重新做 K 扫描，看 K* 与"块数 / RU 数"的关系。

同时检验经验公式：K ≈ clamp(3, min(RU数/7, 块数/100), 16)

用法：python src/k_adaptive.py
"""

from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _setup import add_src_paths  # noqa: E402
add_src_paths()

import json
import os
import shutil
import sys

import numpy as np
import pandas as pd
from ablate_v2v3 import prep  # noqa: E402
from features_v2 import FEATURES_V2  # noqa: E402
from fusion_v3 import cut, eta2_log, oof_rank, wa_of  # noqa: E402
from metrics import summarize  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "data", "trace_docvqa36")


def make_subset(n_sess: int, out_dir: str):
    """按会话子采样：先用全量数据把每个块归到会话，再只保留选中会话的块。

    （直接拷贝全部 blocks 不行：会话在时间上重叠，attach 会把别的会话的块
      错配给剩下的会话，导致块数几乎不下降。）
    """
    from cluster_lifetime import attach, load
    reqs_all, blocks_all = load(SRC)
    rows_all, _ = attach(reqs_all, blocks_all)
    sids = sorted({r["sid"] for r in reqs_all})[:n_sess]
    own = {(round(r["t_create"], 6), r["pos0"]) for r in rows_all if r["sid"] in sids}
    keep_req = [r for r in reqs_all if r["sid"] in sids]
    keep_blk = [b for b in blocks_all
                if (round(b["t_create"], 6), b.get("pos0", -1)) in own]
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "requests.jsonl"), "w") as f:
        for r in keep_req:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(out_dir, "blocks.jsonl"), "w") as f:
        for b in keep_blk:
            f.write(json.dumps(b) + "\n")
    s = os.path.join(SRC, "meta.json")
    if os.path.exists(s):
        shutil.copy(s, os.path.join(out_dir, "meta.json"))
    return len(keep_req), len(keep_blk)


def k_grid(n_ru: int):
    k_max = max(2, n_ru // 3)
    return [k for k in (2, 3, 4, 6, 8, 10, 12, 16, 20) if k <= k_max]


def main():
    out = []
    for n in (36, 24, 18, 12, 6):
        d = os.path.join(ROOT, "data", f"_sub_{n}")
        nreq, nsid = make_subset(n, d)
        try:
            df, size, cap, n_ru, _, _ = prep(d, f"sub{n}")
        except Exception as e:
            print(f"[{n} 会话] 跳过: {e}")
            continue
        y = df["L_A"].values
        nb = len(df)
        print(f"\n[{n} 会话] 请求 {nreq}，块 {nb}，盘 {cap:.0f}GB，RU {n_ru}")
        best = None
        for K in k_grid(n_ru):
            sc = oof_rank(df, FEATURES_V2, K)
            lab = cut(sc, K)
            m = summarize(lab, y)
            wa = wa_of(df, lab, size, cap)
            out.append(dict(n_sess=n, n_blocks=nb, n_ru=n_ru, K=K,
                            eta2=eta2_log(lab, y), ibrd50=m["ibrd50"], wa=wa))
            print(f"   K={K:2d}  η²={out[-1]['eta2']:.3f}  IBRD={m['ibrd50']*100:5.1f}%  WA={wa:.3f}")
            if not np.isnan(wa) and (best is None or wa < best[1]):
                best = (K, wa)
        if best:
            rule = int(np.clip(min(n_ru / 7, nb / 100), 3, 16))
            print(f"   -> K*={best[0]} (WA={best[1]:.3f})；经验公式给 K={rule}")
            out[-1]["k_star"] = best[0]
            out[-1]["k_rule"] = rule

    d = pd.DataFrame(out)
    d.to_csv(os.path.join(ROOT, "data", "k_adaptive.csv"), index=False)

    print("\n" + "=" * 62)
    print("每个子集的最优 K 与经验公式对比")
    print(f"{'会话数':>6}{'块数':>7}{'RU':>5}{'K*':>5}{'公式K':>7}{'WA@K*':>8}")
    for n, sub in d.groupby("n_sess"):
        i = sub.wa.idxmin()
        nb, ru = sub.loc[i, "n_blocks"], sub.loc[i, "n_ru"]
        rule = int(np.clip(min(ru / 7, nb / 100), 3, 16))
        print(f"{n:>6}{nb:>7}{ru:>5}{int(sub.loc[i,'K']):>5}{rule:>7}{sub.loc[i,'wa']:>8.3f}")

    # 清理临时目录
    for n in (36, 24, 18, 12, 6):
        p = os.path.join(ROOT, "data", f"_sub_{n}")
        if os.path.isdir(p):
            shutil.rmtree(p)


if __name__ == "__main__":
    main()
