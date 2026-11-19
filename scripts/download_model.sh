#!/usr/bin/env bash
# 下载 Qwen3.5-9B 到工作区（本机 ~/models 在沙箱下只读）
# 坑：shell 里 ALL_PROXY=socks5://... 会让 httpx 报 socksio 缺失；
#     NO_PROXY 里的 [::1] 会让 httpx 报 Invalid port ':1]'。
set -u
cd "$(dirname "$0")/.."

export PATH="$HOME/venvs/vllm/bin:$PATH"
unset ALL_PROXY all_proxy HF_HUB_OFFLINE
export NO_PROXY="localhost,127.0.0.1" no_proxy="localhost,127.0.0.1"
export HF_HOME="$PWD/.hf"

MODEL="${1:-Qwen/Qwen3.5-9B}"
DEST="${2:-models/Qwen3.5-9B}"

echo "[$(date +%H:%M:%S)] downloading $MODEL -> $DEST"
hf download "$MODEL" --local-dir "$DEST"
echo "[$(date +%H:%M:%S)] DONE $?"
