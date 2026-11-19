#!/usr/bin/env bash
# 通宵实验：一条命令跑完，每步落盘 + 写状态文件。
#   bash scripts/overnight.sh
# 需要 GPU（Full access）。中途失败会跳过该步继续后面的。
set -u
cd "$(dirname "$0")/.."
ROOT="$PWD"
STATUS="$ROOT/data/STATUS.txt"
mkdir -p data figs logs
: > "$STATUS"

export PATH="$HOME/venvs/vllm/bin:/usr/local/cuda-13.1/bin:$PATH"
export CUDA_HOME=/usr/local/cuda-13.1
export HF_HUB_OFFLINE=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export MPLCONFIGDIR="$ROOT/.mplconfig"
export PYTHONPATH="$ROOT/pylibs"
unset ALL_PROXY all_proxy
export NO_PROXY="localhost,127.0.0.1" no_proxy="localhost,127.0.0.1"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$STATUS"; }

kill_stale() {
    for pid in $(ps -eo pid,args | grep -E "EngineCore|run_trace|attention_lifetime" \
                 | grep -v grep | grep -v "$0" | awk '{print $1}'); do
        kill -9 "$pid" 2>/dev/null
    done
    sleep 2
}

stage() {  # stage <name> <logfile> <command...>
    local name="$1"; shift
    local logf="$1"; shift
    log "START $name"
    if "$@" >> "$logf" 2>&1; then
        log "OK    $name"
    else
        log "FAIL  $name (see $logf)"
    fi
}

kill_stale
nvidia-smi --query-gpu=memory.used --format=csv >> "$STATUS" 2>&1

# ---------- 主实验 A：16 会话，KV 2GB（有压力） ----------
stage "trace_v1" logs/trace_v1.log \
    python -u src/trace/run_trace.py --engine vllm --sessions 16 --turns 6 \
        --arrival-span 30 --kv-gb 2.0 --gpu-util 0.75 --max-model-len 16384 \
        --out data/trace_v1
stage "derive_v1" logs/derive_v1.log python src/trace/derive_blocks.py data/trace_v1
stage "analyze_v1" logs/analyze_v1.log \
    python src/analysis/analyze_lifetime.py data/trace_v1 --blocks blocks.jsonl --tag v1

# ---------- 主实验 B：24 会话，KV 3GB（更高并发） ----------
kill_stale
stage "trace_v2" logs/trace_v2.log \
    python -u src/trace/run_trace.py --engine vllm --sessions 24 --turns 6 \
        --arrival-span 30 --kv-gb 3.0 --gpu-util 0.75 --max-model-len 16384 \
        --out data/trace_v2
stage "derive_v2" logs/derive_v2.log python src/trace/derive_blocks.py data/trace_v2
stage "analyze_v2" logs/analyze_v2.log \
    python src/analysis/analyze_lifetime.py data/trace_v2 --blocks blocks.jsonl --tag v2

# ---------- 对照：块级缓存模拟器跑同一批请求 ----------
stage "sim_v1" logs/sim_v1.log bash -c \
    "python src/sim/cache_sim.py data/trace_v1 620 && python src/analysis/analyze_lifetime.py data/trace_v1 --blocks blocks_sim.jsonl --tag v1_sim"

# ---------- token 级：注意力视角 ----------
kill_stale
stage "attention" logs/attention.log \
    python -u src/analysis/attention_lifetime.py --turns 2 --max-new 48

log "ALL DONE"
