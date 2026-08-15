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
| 短（~1-3K） | ~21 | 37.3 | 39.7 | 95.6* | 100.2* |
| ~30-40K | — | — | — | 53.3 | 62.1 |
| ~100K | 16.6 | — | — | 28.5 | 34.8 |
| ~200K | 14.2 | 26.4 | 39.9 | 19.1 | 18.2 |

\* 本仓库 decode 在 MTP 开启下测量。MTP 接受率随上下文变长而下降，
200K 档 MTP 的收益已基本被验证开销抵消（PRO 5000 未开 MTP 的 FP8 在 200K 达 26.4 t/s，
超过本仓库 MTP 数字；llama.cpp Q6_K 以 39.9 t/s 在 200K 档全场最高，
长上下文持续生成场景值得单独评估 llama.cpp 路线）。

> 勘误：本仓库 decode 列曾因"按流式 chunk 计数"低估约 2~3 倍
> （MTP 下单个 chunk 携带多个已接受 token），已改用 `usage.completion_tokens` 重测修正。

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
  短上下文 decode 约为 4~5 倍；与 PRO 5000 同架构（sm_120）但显存更大、带宽更高，各档全面领先。
- **超长上下文 decode（200K）格局反转**：llama.cpp Q6_K（39.9 t/s）> PRO 5000 vLLM FP8 无 MTP（26.4）
  > PRO 6000 MTP（18~19）> DGX Spark（14.2）。长上下文下 MTP 收益衰减，
  GDN 混合架构 + llama.cpp 的组合在此场景意外地强。
- **格式选择**：NVFP4 在 Blackwell 桌面卡上（PRO 6000/5000 均 sm_120）可行且更快，
  但需踩坑修复（见各自仓库）；FP8 是最省心的基线。
