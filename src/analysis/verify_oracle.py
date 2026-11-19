"""验证 oracle4 ≈ 1.00 是不是模型假象。

三个维度做稳健性检查：
  1. 随机种子（trace 放大时的偏移与寿命抖动）—— 3 个；
  2. RU 大小 —— 2GB / 6GB / 12GB；
  3. 工作点 ρ —— 0.36 / 0.41 / 0.47。
每个组合跑 s0（不分层）与 oracle4（按真值 4 桶），看 oracle4 是否稳定接近 1.0、
以及它与 s0 的差距是否稳定。

用法：python src/verify_oracle.py data/trace_docvqa36
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
from cluster_lifetime import add_kind, attach, load  # noqa: E402
from gc_sim import Chunk, Device  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
KB_PER_TOKEN = 32
OP = 0.07
PASSES = 120


def build(df, lab, size, seed):
    rng = np.random.default_rng(seed)
    base = [Chunk(float(tw), float(tw) + max(float(l), 0.05), size, int(s))
            for tw, l, s in zip(df["t_death"].values, df["L_A"].values, lab)]
    span = float(df["t_death"].max())
    out = []
    for _ in range(PASSES):
        off = rng.uniform(0, span * 4)
        for c in base:
            life = (c.t_death - c.t_write) * (1 + rng.uniform(-0.1, 0.1))
            out.append(Chunk(c.t_write + off, c.t_write + off + max(life, 0.05),
                             c.size, c.stream))
    return out


def main():
    td = sys.argv[1] if len(sys.argv) > 1 else "data/trace_docvqa36"
    td = td if os.path.isabs(td) else os.path.join(ROOT, td)
    tag = os.path.basename(td)

    reqs, blocks = load(td)
    rows, _ = attach(reqs, blocks)
    bs = int(blocks[0].get("block_size", 528))
    from mm_input import MMInput
    rows = add_kind(rows, reqs, MMInput(os.path.join(ROOT, "models", "Qwen3.5-9B")),
                    bs, max(len(r["prompt_token_ids"]) for r in reqs))
    df = pd.DataFrame(rows)
    se = {}
    for r in reqs:
        se[r["sid"]] = max(se.get(r["sid"], 0.0), r["t_end"])
    df = df[df["t_death"].notna()].copy()
    df["L_A"] = df["sid"].map(se) - df["t_death"]
    df = df[df["L_A"] > 0].reset_index(drop=True)
    size = int(bs * KB_PER_TOKEN * 1024)
    y = df["L_A"].values
    L50 = float(np.median(y))

    lab_s0 = np.zeros(len(df), dtype=int)
    lab_or4 = np.searchsorted(np.quantile(y, [0.25, 0.5, 0.75]), y, side="right")

    out = []
    for seed in (0, 1, 2):
        c0 = build(df, lab_s0, size, seed)
        W = sum(c.size for c in c0) / (max(c.t_death for c in c0)
                                       - min(c.t_write for c in c0))
        for ru in (2048.0, 6144.0, 12288.0):
            for rho in (0.36, 0.41, 0.47):
                cap_gb = L50 * W / rho / (1 << 30)
                row = dict(seed=seed, ru=ru, rho=rho, cap_gb=cap_gb,
                           n_ru=int(cap_gb * 1024 / ru))
                for name, lab in (("s0", lab_s0), ("oracle4", lab_or4)):
                    ch = build(df, lab, size, seed)
                    try:
                        d = Device(ch, capacity_gb=cap_gb, ru_mb=ru, op_frac=OP)
                        r = d.run()
                        row[name] = r["wa_steady"]
                    except RuntimeError:
                        row[name] = float("nan")
                out.append(row)
                print(f"  seed={seed} RU={ru/1024:.0f}GB ρ={rho:.2f} "
                      f"cap={cap_gb:.0f}GB RU数={row['n_ru']:3d}  "
                      f"s0={row['s0']:.3f}  oracle4={row['oracle4']:.3f}")

    res = pd.DataFrame(out)
    res.to_csv(os.path.join(ROOT, "data", f"verify_oracle_{tag}.csv"), index=False)
    print("\n=== 汇总 ===")
    print(res.groupby("rho")[["s0", "oracle4"]].agg(["mean", "std"]).round(3).to_string())
    print("\noracle4 的极值:", round(res["oracle4"].min(), 3), "~",
          round(res["oracle4"].max(), 3))
    print("s0 的极值:", round(res["s0"].min(), 3), "~", round(res["s0"].max(), 3))


if __name__ == "__main__":
    main()
