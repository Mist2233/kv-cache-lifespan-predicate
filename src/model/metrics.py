"""统一的分组质量指标。

- M1（题面原口径）：桶内相对偏差 ≤ ±5% 的块占比
- IBRD（建议口径）：桶内相对偏差的**中位数（IBRD50）**与 **p90（IBRD90）**
  * 中位：一半的块，离自己桶的均值不超过这个比例
  * p90 ：九成的块，离自己桶的均值不超过这个比例（尾部起点）
"""

from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _setup import add_src_paths  # noqa: E402
add_src_paths()

import numpy as np


def deviations(labels, values) -> np.ndarray:
    """每个样本相对"自己所属桶的均值"的相对偏差。"""
    labels = np.asarray(labels)
    values = np.asarray(values, dtype=float)
    dev = np.zeros(len(values))
    for b in np.unique(labels):
        m = labels == b
        mu = values[m].mean()
        dev[m] = np.abs(values[m] - mu) / mu if mu > 0 else np.inf
    return dev


def summarize(labels, values, thresh: float = 0.05) -> dict:
    dev = deviations(labels, values)
    return {
        "pass5": float((dev <= thresh).mean()),        # 原口径 M1
        "ibrd50": float(np.median(dev)),               # IBRD₅₀
        "ibrd90": float(np.percentile(dev, 90)),       # IBRD₉₀
        "pass_ibrd": bool(np.median(dev) <= 0.05 and np.percentile(dev, 90) <= 0.20),
    }
