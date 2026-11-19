# KV Cache 寿命预测与 FDP 多流分组

面向 **QLC SSD 上 KV Cache 卸载**的研究代码：预测每个 KV block 在盘上的驻留寿命，
按预测寿命把数据写入不同的 **FDP stream**，从而降低 SSD 写放大（Write Amplification, WA）。

研究问题来自华为"难题揭榜"课题。本仓库包含：真实推理 trace 采集、寿命定义与指标、
寿命预测模型、事件驱动的 FDP 仿真器，以及多盘多流扩展。

---

## 研究思路

1. **采集**：在单卡 GPU 上用 vLLM（Qwen3.5-9B / Qwen2.5-Omni-7B）跑多轮多模态对话，
   订阅 KV 事件（`BlockStored` / `BlockRemoved`），还原每个 KV block 的完整生命周期。
2. **定义寿命**：`L_A = 会话结束时刻 − 块被淘汰（写入 SSD）时刻`。
   若块被多个会话共享（前缀缓存），则取**最后一个使用它的会话**的结束时刻。
3. **预测**：在"块写入 SSD 的时刻"采集 26 个可观测特征（不读模型内部状态），
   用 pairwise 排序（LogisticRegression）学习寿命打分，再按分位数切成 K 个桶（对应 K 个 FDP stream）。
4. **仿真**：事件驱动的 QLC FDP 仿真器（RU = 6 GiB、贪心最少有效数据 GC），
   比较"不分层 / 预测分组 / 完美分桶"三者的稳态写放大。

## 主要结果（仿真，盘利用率 74%）

| 数据集 | 块数 | RU 数 | 不分层 WA | 方法 WA | 完美分桶 | 降幅 |
|---|---|---|---|---|---|---|
| trace_docvqa36 | 1,279 | 87 | 1.296 | **1.161**（K=12） | 1.000 | 0.135 |
| trace_big | 9,467 | 112 | 1.125 | **1.043**（K=8） | 1.000 | 0.081 |
| trace_real_len | 12,844 | 81 | 1.242 | **1.097**（K=8） | 1.000 | 0.144 |
| trace_openomni_v3（图像+语音） | 16,718 | 169 | 1.493 | **1.099**（K=56） | 1.000 | 0.394 |

结论要点：

- **机制成立**：按寿命正确分桶可把写放大降到接近 1.00（完美分桶列）；
- **方法有效**：三份 trace 可压到 1.1 以内（1.043 / 1.097 / 1.099），DocVQA 为 1.161；
- **收益由工作点决定**：利用率、RU 数、寿命与盘重写周期之比（ρ）都会显著影响绝对值；
- **预测瓶颈是"会话何时结束"**：显存淘汰延迟不可预测但影响很小；剩余会话时长才是关键。

## 目录结构

```
src/
  _setup.py               # 统一路径引导（各脚本跨目录 import 用）
  trace/                  # 真实推理与 trace 采集
    run_trace.py          #   Qwen3.5，图文多轮
    run_trace_omni.py     #   Qwen2.5-Omni，图像+语音
    engines.py            #   vLLM 封装 + ZMQ KV 事件订阅
    derive_blocks.py      #   KV 事件 -> block 生命周期
    mm_input.py           #   多模态输入切分（图像/语音/文本 token 区间）
    prompts.py            #   厂商风格系统指令词
    workload*.py          #   工作负载构造（DocVQA / WildChat 真实长度）
    openomni_pool.py      #   OpenOmni 素材重建：COCO 图像 + TTS 语音
    openomni_prep.py      #   多轮会话构造
  model/                  # 寿命建模
    cluster_lifetime.py   #   块 -> 请求/会话归属、L_A 计算、特征
    features_v2.py        #   26 个可观测特征
    fusion_v3.py          #   pairwise 排序训练与分桶
    classify_buckets.py   #   放大成稳态写入流
    metrics.py            #   桶内相对偏差 IBRD 等
    metric_validation.py  #   指标 vs 仿真写放大 的相关性
    ordinal_survival.py   #   序数回归 / 离散时间生存模型（对比方法）
    session_hazard.py     #   会话级 hazard 模型（对比方法）
  sim/                    # 盘侧仿真
    gc_sim.py             #   事件驱动 QLC FDP 仿真器（RU + 贪心 GC）
    multidisk_sim.py      #   多盘 + 多流（两种盘-流关系）
    wa_proxy.py           #   写放大代理指标
    cache_sim.py          #   缓存替换模拟
  analysis/               # 实验分析
    analyze_big.py        #   主实验：预测 + 仿真对比
    analyze_openomni_shared.py  # 图像+语音（共享前缀寿命定义）
    k_sweep*.py           #   桶数 / 利用率扫描
    rho_sweep.py          #   世代比扫描
    metric_validation.py  #   指标验证
    compare_*.py, final_*.py, ...
  experiments/            # 探索性实验
    exp_shared_prefix.py  #   共享前缀寿命定义
    exp_turns_*.py        #   会话总轮次可行性
    smoke_test.py         #   管线冒烟测试
  plot/                   # 绘图
    plot_*.py, plotstyle.py

scripts/                  # 下载模型 / 一键跑通 / 夜间批处理
figs/                     # 论文/汇报用图
```

## 环境与运行

```bash
# 1) 依赖（Python 3.10，完整 trace 采集需要 CUDA GPU + vLLM）
pip install -r requirements.txt

# 2) 下载模型（约 22 GB）
bash scripts/download_model.sh

# 3) 采集 trace（示例：Qwen3.5，图文多轮）
python src/trace/run_trace.py --engine vllm --dataset docvqa --out data/trace_docvqa

# 4) 还原块生命周期
python src/trace/derive_blocks.py data/trace_docvqa

# 5) 分析与仿真
python src/analysis/analyze_big.py data/trace_docvqa
```

图像+语音（OpenOmni）流程：

```bash
# 生成 3000 组 COCO 图像 + TTS 语音素材
python src/trace/openomni_pool.py 3000
# 采集 Qwen2.5-Omni 多轮图像+语音 trace
python src/trace/run_trace_omni.py --sessions-file data/openomni_pool/sessions.json \
    --kv-gb 4 --block-size 512 --max-num-seqs 128 --out data/trace_openomni
# 分析（共享前缀寿命定义）
python src/analysis/analyze_openomni_shared.py data/trace_openomni
```

## 说明

- 仓库**不包含** trace 数据、模型权重与中间结果（体积大，可由上述脚本重新生成）；
- 特征只用请求元数据与 tokenizer，**不读取模型内部状态**，因此不依赖具体模型或推理框架；
- 仿真器的 GC 策略偏乐观，写放大绝对值会优于真实盘，主要用于方法之间的相对比较。
