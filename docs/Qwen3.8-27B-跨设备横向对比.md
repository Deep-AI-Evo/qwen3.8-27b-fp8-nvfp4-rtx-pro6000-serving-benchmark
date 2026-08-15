# Qwen3.8-27B 跨设备横向对比（DGX Spark / RTX PRO 5000 / RTX PRO 6000）

> 数据来源：
> [DGX Spark NVFP4 教程](https://github.com/Deep-AI-Evo/qwen3.8-27b-nvfp4-dgx-spark-tutorial)、
> [RTX PRO 5000 Q6_K vs FP8](https://github.com/Deep-AI-Evo/qwen3.8-27b-q6k-fp8-rtx-pro5000-serving-benchmark)、
> 本仓库（RTX PRO 6000）。
>
> ⚠️ 口径差异提醒：三台设备的 vLLM 版本（0.27.1 / 0.26.0 / 0.21.0）、MTP 设置（×3 / 未开 / ×2）、
> 操作系统（aarch64 Linux / Windows / x86_64 Linux）与 prompt 尺寸档位并不完全相同，
> 下表取最接近的档位对比，数量级参考意义大于精确对比。

## 设备与部署概览

| | DGX Spark | RTX PRO 5000 | RTX PRO 6000（本仓库） |
|---|---|---|---|
| GPU | GB10（128GB 统一内存） | 72GB Blackwell (sm_120) | 96GB Blackwell (sm_120) |
| 系统 | aarch64 Linux | Windows 10 | x86_64 Linux |
| 引擎/格式 | vLLM 0.27.1 · NVFP4 | vLLM 0.26.0 · FP8；llama.cpp Q6_K | vLLM 0.21.0 · FP8 / NVFP4；llama.cpp Q6_K |
| MTP | ×3 | 未开 | ×2（另测无 MTP） |
| 上下文 | 256K | 256K（llama.cpp） | 262K（vLLM）/ 1M（llama.cpp） |

> ⚠️ llama.cpp 特别提醒：Blackwell（sm_120）上自编译 llama.cpp **必须用 CUDA 12.8**。
> 本机实测 CUDA 13.3 构建 FA kernel 异常（prefill 慢 5~26 倍，非 FA 路径直接 CUDA error），
> 换 CUDA 12.8 后恢复正常（Q6_K pp16K：696 → 3,536 t/s）。PRO 5000 数据来自官方预编译
> Windows 二进制（b9692, cuda-13.3），不受影响。

## 单并发 Decode（tok/s，越高越好）

| 上下文 | DGX Spark NVFP4 | PRO 5000 vLLM FP8（无 MTP） | PRO 5000 llama.cpp Q6_K | PRO 6000 FP8+MTP | PRO 6000 NVFP4+MTP | PRO 6000 FP8 无 MTP | PRO 6000 NVFP4 无 MTP | PRO 6000 llama.cpp Q6_K |
|---|---|---|---|---|---|---|---|---|
| 短（~1-3K） | ~21 | 37.3 | 39.7 | 95.6* | 100.2* | 52.6 | 58.6 | 55.4 |
| ~30-40K | — | — | — | 53.3 | 62.1 | 49.9 | 55.4 | 51.6 |
| ~100K | 16.6 | — | — | 28.5 | 34.8 | 45.6 | 49.1 | 44.4 |
| ~200K | 14.2 | 26.4 | 39.9 | 19.1 | 18.2 | 39.6 | **43.7** | 35.7 |

\* MTP 开时短上下文 decode 接近翻倍，但 MTP 接受率随上下文变长而下降，
200K 档 MTP 的验证开销已超过收益（19.1/18.2，反而远低于无 MTP 的 39.6/43.7）。
**结论：短上下文/高并发开 MTP，长上下文生成关 MTP。**

> 勘误：本仓库 decode 列曾因"按流式 chunk 计数"低估约 2~3 倍
> （MTP 下单个 chunk 携带多个已接受 token），已改用 `usage.completion_tokens` 重测修正。

## 单并发 Prefill（tok/s）与 TTFT（秒）

| Prompt | DGX Spark NVFP4 | PRO 5000 vLLM FP8 | PRO 5000 Q6_K | PRO 6000 FP8+MTP | PRO 6000 NVFP4+MTP | PRO 6000 Q6_K |
|---|---|---|---|---|---|---|
| ~1-3K | 1,800 / 0.57s | 1,033 / 2.55s | 749 / 3.60s | 6,265 / 0.23s | 8,941 / 0.17s | 1,863 / 0.81s |
| ~30-40K | — | 3,671 / 10.2s | 1,576 / 23.8s | 7,100 / 4.16s | 9,465 / 3.11s | 3,117 / 9.46s |
| ~100K | 1,230 / 83s | — | — | 5,384 / 16.5s | 6,616 / 13.4s | 2,367 / 37.6s |
| ~200-232K | 840 / 244s | 2,114 / 110s | 768 / 302s | 3,869 / 45.8s | 4,447 / 39.9s | 1,676 / 105.9s |

Prefill/TTFT 维度 vLLM 全面领先 llama.cpp（2~4 倍），RTX PRO 6000 全面领先其余两设备
（agent 体感最关键的指标）。

## 并发扩展（decode 聚合吞吐 tok/s）

| 并发 | DGX Spark NVFP4 | PRO 5000 vLLM FP8 | PRO 6000 FP8+MTP | PRO 6000 NVFP4+MTP | PRO 6000 Q6_K（4 槽） |
|---|---|---|---|---|---|
| 1 | 20.0 | 37.3 | 78.8 | 95.0 | 49.6 |
| 2 | 22.7 | 64.7 | 137.0 | 151.3 | 92.8 |
| 4 | 44.0 | — | 299.3 | 346.5 | 158.5 |
| 8 | 77.7 | — | 556.2 | 654.1 | — |

## 长上下文真实性（大海捞针）

| 设备 | 规模 | 深度 | 结果 |
|---|---|---|---|
| DGX Spark | 255,376 tok | 70% | ✅ 一次答对 |
| PRO 6000 FP8 / NVFP4 | 96,886 tok | 70% | ✅ 均一次答对 |
| PRO 6000 llama.cpp Q6_K | 96,885 tok | 70% | ✅ 一次答对（48.2s） |

## 小结

- **算力档位**：PRO 6000（96GB）在 prefill/TTFT/并发吞吐上约为 DGX Spark 的 4~8 倍，
  短上下文 decode 约为 4~5 倍；与 PRO 5000 同架构（sm_120）但显存更大、带宽更高，各档全面领先
  （Q6_K prefill 各档约为 PRO 5000 的 2~2.5 倍）。
- **超长上下文 decode（200K）正确姿势是关 MTP**：PRO 6000 NVFP4 无 MTP 43.7 t/s 为全场最高，
  FP8 无 MTP 39.6，Q6_K 35.7，均远超 MTP 开（18~19）与 DGX Spark（14.2）。
  MTP 收益随上下文衰减，长文写作/长上下文 RAG 场景应使用无 MTP 配置。
- **llama.cpp 的价值**：单文件部署极简、Q6_K 权重质量最高、短上下文 decode（55.4 t/s）
  与 vLLM 无 MTP（52.6~58.6）相当；但 prefill/并发明显弱于 vLLM，且 Blackwell 上
  **必须用 CUDA 12.8 自编译**（官方 Windows 预编译二进制无此问题）。
- **格式选择**：NVFP4 在 Blackwell 桌面卡上（PRO 6000/5000 均 sm_120）可行且更快，
  但需踩坑修复（见各自仓库）；FP8 是最省心的基线。
