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
| 引擎/格式 | vLLM 0.27.1 · NVFP4 | vLLM 0.26.0 · FP8；llama.cpp Q6_K | vLLM 0.21.0 · FP8 / NVFP4 |
| MTP | ×3 | 未开 | ×2 |
| 上下文 | 256K | 256K（llama.cpp） | 262K |

## 单并发 Decode（tok/s，越高越好）

| 上下文 | DGX Spark NVFP4 | PRO 5000 vLLM FP8 | PRO 5000 llama.cpp Q6_K | PRO 6000 FP8+MTP | PRO 6000 NVFP4+MTP |
|---|---|---|---|---|---|
| 短（~1-3K） | ~21 | 37.3 | 39.7 | 39.5* | 47.3* |
| ~30-40K | — | — | — | 24.5 | 27.0 |
| ~100K | 16.6 | — | — | 13.7 | 14.6 |
| ~200K | 14.2 | 26.4 | 39.9 | 8.1 | 8.7 |

\* 本仓库 decode 在 MTP 开启下测量；注意 MTP 接受率随上下文长度下降（长上下文草稿命中率低），
这是本仓库 200K decode 数字偏低的部分原因。PRO 5000 未开 MTP。
llama.cpp Q6_K 长上下文 decode 不衰减的特性在 200K 档优势明显（39.9 t/s，全场最高）。

## 单并发 Prefill（tok/s）与 TTFT（秒）

| Prompt | DGX Spark NVFP4 | PRO 5000 vLLM FP8 | PRO 6000 FP8+MTP | PRO 6000 NVFP4+MTP |
|---|---|---|---|---|
| ~1-3K | 1,800 / 0.57s | 1,033 / 2.55s | 6,265 / 0.23s | 8,941 / 0.17s |
| ~30-40K | — | 3,671 / 10.2s | 7,100 / 4.16s | 9,465 / 3.11s |
| ~100K | 1,230 / 83s | — | 5,384 / 16.5s | 6,616 / 13.4s |
| ~200-232K | 840 / 244s | 2,114 / 110s | 3,869 / 45.8s | 4,447 / 39.9s |

Prefill/TTFT 维度 RTX PRO 6000 全面领先（agent 体感最关键的指标）。

## 并发扩展（decode 聚合吞吐 tok/s）

| 并发 | DGX Spark NVFP4 | PRO 5000 vLLM FP8 | PRO 6000 FP8+MTP | PRO 6000 NVFP4+MTP |
|---|---|---|---|---|
| 1 | 20.0 | 37.3 | 78.8 | 95.0 |
| 2 | 22.7 | 64.7 | 137.0 | 151.3 |
| 4 | 44.0 | — | 299.3 | 346.5 |
| 8 | 77.7 | — | 556.2 | 654.1 |

## 长上下文真实性（大海捞针）

| 设备 | 规模 | 深度 | 结果 |
|---|---|---|---|
| DGX Spark | 255,376 tok | 70% | ✅ 一次答对 |
| PRO 6000 FP8 / NVFP4 | 96,886 tok | 70% | ✅ 均一次答对 |

## 小结

- **算力档位**：PRO 6000（96GB）在 prefill/TTFT/并发吞吐上约为 DGX Spark 的 4~8 倍，
  与 PRO 5000 同架构（sm_120）但显存更大、带宽更高，各档全面领先。
- **长上下文 decode 是 vLLM+MTP 的弱项**：200K 档 llama.cpp Q6_K（PRO 5000）反而全场最快，
  超长上下文持续生成场景值得单独评估 llama.cpp 路线。
- **格式选择**：NVFP4 在 Blackwell 桌面卡上（PRO 6000/5000 均 sm_120）可行且更快，
  但需踩坑修复（见各自仓库）；FP8 是最省心的基线。
