#!/usr/bin/env bash
# 一键跑通：真实推理 trace -> 块生命周期还原 -> 每 token 寿命分布图
#
# 用法：
#   bash scripts/run_all.sh                      # 默认 16 会话 6 轮，KV 限 2GB
#   bash scripts/run_all.sh 24 6 3.0 data/trace_v2
#
# 注意：本脚本要在 **danger-full-access** 沙箱下跑（需要 GPU）。
set -eu
cd "$(dirname "$0")/.."

SESSIONS="${1:-16}"
TURNS="${2:-6}"
KVGB="${3:-2.0}"
OUT="${4:-data/trace_v1}"

export PATH="$HOME/venvs/vllm/bin:/usr/local/cuda-13.1/bin:$PATH"
export CUDA_HOME=/usr/local/cuda-13.1
export HF_HUB_OFFLINE=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export MPLCONFIGDIR="$PWD/.mplconfig"
export PYTHONPATH="$PWD/pylibs"
unset ALL_PROXY all_proxy || true
export NO_PROXY="localhost,127.0.0.1" no_proxy="localhost,127.0.0.1"

echo "=== [1/3] 推理 + 事件采集 ($OUT) ==="
python src/trace/run_trace.py --engine vllm --sessions "$SESSIONS" --turns "$TURNS" \
    --arrival-span 30 --kv-gb "$KVGB" --gpu-util 0.75 --max-model-len 16384 --out "$OUT"

echo "=== [2/3] 事件 -> 块生命周期 ==="
python src/trace/derive_blocks.py "$OUT"

echo "=== [3/3] 每 token 寿命分布 ==="
TAG="$(basename "$OUT")"
python src/analysis/analyze_lifetime.py "$OUT" --blocks blocks.jsonl --tag "$TAG"

echo "完成：figs/fig02_lifetime_${TAG}.png"
