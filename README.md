# Qwen3.8-27B serving benchmark on RTX PRO 6000 Blackwell: vLLM FP8 vs NVFP4 vs llama.cpp Q6_K

vLLM FP8 / NVFP4 / llama.cpp Q6_K 三方对比 · TTFT / Prefill / Decode / 并发 / 长上下文捞针 / MTP
基于 2026-08-15 实机测试，所有数据均可复现（测试脚本附于仓库）

## 📌 核心结论（TL;DR）

| # | 结论 | 依据 |
|---|---|---|
| 1 | **NVFP4+MTP 是短上下文最优配置** | decode 132 t/s、8 并发聚合 654 t/s、prefill 全面领先，质量与 FP8 无可见差距 |
| 2 | **MTP 白捡 2.3 倍 decode（仅短上下文）** | FP8 51→116 t/s，NVFP4 56→132 t/s，接受率 83~100% |
| 3 | **长上下文 decode 别开 MTP** | 200K 档：NVFP4 无 MTP 43.7 > FP8 无 MTP 39.6 > Q6_K 35.7 >> MTP 开 18~19 t/s，接受率随长度衰减 |
| 4 | **并发扩展极佳** | vLLM 8 并发单流仅降 ~10%，聚合近线性（95→654 t/s）；llama.cpp 4 并发聚合 158 t/s |
| 5 | **100K 长上下文一次答对** | 大海捞针（70% 深度）FP8 / NVFP4 / Q6_K 三方案均通过 |
| 6 | **NVFP4 需 3 个手工修复** | 详见部署教程 §3；FP8 官方仓 1 个环境变量即可跑通 |
| 7 | **llama.cpp 在 Blackwell 必须用 CUDA 12.8 编译** | CUDA 13.x 自编译 FA kernel 异常：prefill 慢 5~26 倍、fa=0 直接崩溃（见测试报告 §7） |

## 🖥️ 测试环境

| 项目 | 配置 |
|---|---|
| GPU | NVIDIA RTX PRO 6000 Blackwell 96GB（sm_120，97887 MiB） |
| 系统 | Linux x86_64，驱动 595.84 / CUDA 13.2 |
| vLLM | 0.21.0（Python 3.12 + torch 2.11.0+cu130） |
| llama.cpp | b9692 自编译（**CUDA 12.8**，FA_ALL_QUANTS=ON；⚠️ CUDA 13.x 编译在 sm_120 上 FA kernel 异常，勿用） |
| 模型 | [Qwen/Qwen3.8-27B-FP8](https://modelscope.cn/models/Qwen/Qwen3.8-27B-FP8) · [unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4) · [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF)（UD-Q6_K_XL） |
| 配置 | 262,144 上下文 · MTP(k=2) · gpu_memory_utilization 0.92 |

## 📊 实测数据速览

### ⚡ 单流 decode（短上下文基准，MTP 开/关）

| 配置 | 无 MTP | +MTP(k=2) |
|---|---|---|
| FP8 | 50.9 t/s | 115.8 t/s（2.28×） |
| NVFP4 | 56.4 t/s | **132.3 t/s（2.35×）** |

### 📏 上下文长度衰减（单并发，MTP 已开）

| Prompt | FP8 TTFT / Prefill / Decode | NVFP4 TTFT / Prefill / Decode |
|---|---|---|
| ~1.4K | 0.23 s / 6,265 t/s / 95.6 t/s | 0.16 s / 9,431 t/s / 100.2 t/s |
| ~29.5K | 4.14 s / 7,124 t/s / 53.3 t/s | 2.97 s / 9,915 t/s / 62.1 t/s |
| ~88.9K | 16.5 s / 5,390 t/s / 28.5 t/s | 13.2 s / 6,766 t/s / 34.8 t/s |
| ~177.4K | 45.4 s / 3,906 t/s / 19.1 t/s | 39.5 s / 4,496 t/s / 18.2 t/s |

### 📏 上下文长度衰减（无 MTP vLLM vs llama.cpp Q6_K）

| Prompt | FP8 无 MTP Decode | NVFP4 无 MTP Decode | Q6_K TTFT / Prefill / Decode |
|---|---|---|---|
| ~1.4K | 52.6 t/s | 58.6 t/s | 0.81 s / 1,863 t/s / 55.4 t/s |
| ~29.5K | 49.9 t/s | 55.4 t/s | 9.46 s / 3,117 t/s / 51.6 t/s |
| ~88.9K | 45.6 t/s | 49.1 t/s | 37.6 s / 2,367 t/s / 44.4 t/s |
| ~177.4K | 39.6 t/s | **43.7 t/s** | 105.9 s / 1,676 t/s / 35.7 t/s |

> 200K 档 decode 全场排序：**NVFP4 无 MTP 43.7** > FP8 无 MTP 39.6 > Q6_K 35.7 >> NVFP4+MTP 18.2 ≈ FP8+MTP 19.1。
> MTP 的验证开销在长上下文下超过其收益；llama.cpp Q6_K decode 与 vLLM 无 MTP 接近，但 prefill/TTFT 差距 2~4 倍。

### 🔀 并发扩展（500 tok/流）

| 并发 | FP8+MTP 单流 / 聚合 | NVFP4+MTP 单流 / 聚合 | Q6_K 单流 / 聚合 |
|---|---|---|---|
| 1 | 79.1 / 78.8 t/s | 95.5 / 95.0 t/s | 49.8 / 49.6 t/s |
| 2 | 70.3 / 137.0 t/s | 77.0 / 151.3 t/s | 46.5 / 92.8 t/s |
| 4 | 77.5 / 299.3 t/s | 91.1 / 346.5 t/s | 39.8 / 158.5 t/s |
| 8 | 73.9 / 556.2 t/s | 86.3 / **654.1 t/s** | —（仅 4 槽） |

### 🎯 长上下文真实性

大海捞针：~97K tokens 文本 70% 深度藏入随机密码，FP8 / NVFP4 / Q6_K 三方案均一次答对。

## 🚀 一键启动命令（环境就绪后）

```bash
# NVFP4 + MTP（推荐，需先按教程 §3 做修复）
VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_NVFP4_GEMM_BACKEND=cutlass \
  vllm serve ~/models/Qwen3.8-27B-NVFP4 \
  --served-model-name qwen38-nvfp4-mtp \
  --reasoning-parser qwen3 \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.92 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}' \
  --port 8000

# FP8 + MTP
VLLM_USE_FLASHINFER_SAMPLER=0 vllm serve ~/models/Qwen3.8-27B-FP8 \
  --served-model-name qwen38-fp8-mtp \
  --reasoning-parser qwen3 \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.92 \
  --max-num-seqs 512 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}' \
  --port 8000

# llama.cpp Q6_K（需 CUDA 12.8 自编译，见部署教程 §4）
LD_LIBRARY_PATH=~/cuda-12.8/lib64 llama-server \
  --model ~/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q6_K_XL.gguf \
  --host 127.0.0.1 --port 8000 \
  --ctx-size 1048576 --n-gpu-layers 99 --threads 8 --parallel 4 \
  --flash-attn on --no-mmap --cache-type-k q8_0 --cache-type-v q8_0
```

服务就绪后访问 http://127.0.0.1:8000（OpenAI 兼容 API）。

## 🤖 把本仓库交给代码智能体，一键复现部署（省 Token）

本教程沉淀了 4 个真实踩坑（FlashInfer JIT 不认 sm_120、vLLM 漏传 quant_config ×2、
NVFP4 静态 KV 量化乱码、Mamba cache 超界），Agent 自行试错每一步都在烧 Token。
用法：复制下面这段话，发给任意代码智能体（Kimi Code CLI、Claude Code、Cursor、Codex 等）：

> 我的机器是 NVIDIA RTX PRO 6000 Blackwell（96GB，sm_120，Linux x86_64）。请阅读仓库
> https://github.com/Deep-AI-Evo/qwen3.8-27b-fp8-nvfp4-rtx-pro6000-serving-benchmark
> 中 docs/Qwen3.8-27B-PRO6000-部署教程.md，严格按照其中的版本号与命令在本机部署
> Qwen3.8-27B FP8 / NVFP4 并启动 vLLM 服务；完成后参照 docs/Qwen3.8-27B-PRO6000-测试报告.md
> 中的方法做功能验证并汇报结果。

## 📏 测试方法论

- prefill/TTFT：OpenAI API 流式请求，TTFT = 请求发出 → 首个 token，prefill t/s = prompt_tokens / TTFT；每轮随机前缀杜绝缓存命中
- decode：流式生成 tokens/s（剔除首 token），**必须用 `usage.completion_tokens` 计数**——MTP 投机采样下单个流式 chunk 可携带多个已接受 token，按 chunk 数统计会低估约 2~3 倍（本仓库曾因此误报，已修正）
- 并发 profile：N 线程同时发请求分别计时；聚合吞吐 = Σtokens / 窗口时间
- 所有对比均在 GPU 空闲状态下进行；关键数据 3 轮取均值
- 捞针测试：~97K tokens 填充文本 70% 深度插入随机密码，验证真实长上下文召回

## 📚 完整文档

| 文档 | 内容 |
|---|---|
| 📖 [部署教程](docs/Qwen3.8-27B-PRO6000-部署教程.md) | 环境准备、镜像下载、启动参数详解、API 示例、4 个踩坑实录 |
| 📈 [测试报告](docs/Qwen3.8-27B-PRO6000-测试报告.md) | 功能测试、prefill/decode/并发/捞针全量数据、MTP 对比、使用建议 |
| ⚖️ [跨设备横向对比](docs/Qwen3.8-27B-跨设备横向对比.md) | DGX Spark / RTX PRO 5000 / RTX PRO 6000 三设备同口径数据对比 |

## 🗂 仓库结构

```
├── README.md                       # 本文件
├── docs/                           # 部署教程 + 测试报告
├── tests/                          # 测试脚本
│   ├── run_matrix.sh               # 一键跑完整测试矩阵
│   ├── functional_test.py          # 功能测试（5 用例 + 基准）
│   ├── prefill_test.py             # prefill / TTFT / decode（随机文本防缓存）
│   ├── conc_test.py                # N 并发 decode profile
│   └── needle_test.py              # 长上下文大海捞针
├── patches/                        # vLLM 0.21.0 lm_head quant_config 补丁
└── results/                        # 全部原始测试输出（JSON）
```

## 🔄 复现方法

```bash
# 完整矩阵（服务已在 8000 端口运行）
./tests/run_matrix.sh <label>

# 单项
python3 tests/prefill_test.py http://127.0.0.1:8000 <model> <label> 1024 32768 100000 200000
python3 tests/conc_test.py    http://127.0.0.1:8000 <model> <N> <max_tokens> <label>
python3 tests/needle_test.py  http://127.0.0.1:8000 <model> <label> 100000 0.7
```

## 🔗 同组织相关仓库

- [DGX Spark 部署 NVFP4 教程](https://github.com/Deep-AI-Evo/qwen3.8-27b-nvfp4-dgx-spark-tutorial)
- [RTX PRO 5000 llama.cpp Q6_K vs vLLM FP8](https://github.com/Deep-AI-Evo/qwen3.8-27b-q6k-fp8-rtx-pro5000-serving-benchmark)

---

测试与文档：Deep-AI-Evo · 模型：Qwen3.8-27B (Apache-2.0) · 数据基于单机实测，不同环境可能略有差异
