# Qwen3.8-27B 跨设备横向对比（DGX Spark / RTX PRO 5000 / RTX PRO 6000）

> 数据来源：
> [DGX Spark NVFP4 教程](https://github.com/Deep-AI-Evo/qwen3.8-27b-nvfp4-dgx-spark-tutorial)、
> [RTX PRO 5000 Q6_K vs FP8](https://github.com/Deep-AI-Evo/qwen3.8-27b-q6k-fp8-rtx-pro5000-serving-benchmark)、
> 本仓库（RTX PRO 6000）。
>
> ⚠️ 口径差异提醒：三台设备的 vLLM 版本（0.27.1 / 0.26.0 / 0.21.0+0.26.0 双验证）、MTP 设置（×3 / n=1 / n=1+n=2）、
> 操作系统（aarch64 Linux / Windows / x86_64 Linux）与 prompt 尺寸档位并不完全相同，
> 下表取最接近的档位对比，数量级参考意义大于精确对比。

## 设备与部署概览

| | DGX Spark | RTX PRO 5000 | RTX PRO 6000（本仓库） |
|---|---|---|---|
| GPU | GB10（128GB 统一内存） | 72GB Blackwell (sm_120) | 96GB Blackwell (sm_120) |
| 系统 | aarch64 Linux | Windows 10 | x86_64 Linux |
| 引擎/格式 | vLLM 0.27.1 · NVFP4 | vLLM 0.26.0 · FP8 / NVFP4；llama.cpp Q6_K | vLLM 0.21.0 / 0.26.0 · FP8 / NVFP4；llama.cpp Q6_K |
| MTP | ×3 | n=1（另测无 MTP） | n=1 / n=2 均测（另测无 MTP） |
| 上下文 | 256K（另测双机 TP=2） | 256K（llama.cpp） | 262K（vLLM）/ 1M（llama.cpp） |

> ⚠️ llama.cpp 特别提醒：Blackwell（sm_120）上自编译 llama.cpp **必须用 CUDA 12.8**。
> 本机实测 CUDA 13.3 构建 FA kernel 异常（prefill 慢 5~26 倍，非 FA 路径直接 CUDA error），
> 换 CUDA 12.8 后恢复正常（Q6_K pp16K：696 → 3,536 t/s）。PRO 5000 数据来自官方预编译
> Windows 二进制（b9692, cuda-13.3），不受影响。

## 单并发 Decode（tok/s，越高越好；"无MTP → +MTP"）

| 上下文 | DGX Spark NVFP4 +MTP×3 | PRO 5000 FP8 | PRO 5000 NVFP4 | PRO 5000 Q6_K | PRO 6000 FP8 | PRO 6000 NVFP4 | PRO 6000 Q6_K |
|---|---|---|---|---|---|---|---|
| 短（~1-11K） | ~21 | 37.3 → 43.2 | 49.6 → 63.6 | 39.7 → 61.9 | 52.6 → 95.6 | 58.6 → **100.2** | 55.4 |
| ~30-40K | — | — | — | — | 49.9 → 53.3 | 55.4 → 62.1 | 51.6 |
| ~100K | 16.6 | — | — | — | 45.6 → 28.5 | 49.1 → 34.8 | 44.4 |
| ~148K | — | 26.4 → 15.3 ❌ | 42.1 → **57.8** ✅ | 39.9 → 40.0 | — | ≈46* → 21.4 ❌ | — |
| ~200K | 14.2 | 26.4（无MTP） | 42.1（无MTP） | 39.9 | 39.6 → 19.1 ❌ | 43.7 → 18.2 ❌ | 35.7 |
| ~256K | 11.4 | — | — | — | — | — | — |

读表要点：

\* PRO 6000 无 MTP 在 148K 未单测，取 100K（49.1）与 200K（43.7）的插值约 46；+MTP 值 21.4 为
vLLM 0.26 + marlin + n=1（与 PRO 5000 完全同栈）实测，其余 MTP 变体数据见下文专题。

- **MTP 长上下文表现因设备/OS 而反转**（详见下文 MTP 专题）：PRO 5000（Windows）的 NVFP4+MTP
  在 148K 仍 +37%，57.8 t/s 是**三设备长上下文 decode 报告最高值**；而 PRO 6000（Linux）
  用**完全相同的软件栈**（vLLM 0.26.0 + marlin + n=1）复测，148K 仅 21.4 t/s（-53%），
  未能复现。FP8+MTP 在两台设备长上下文均为负优化。
- PRO 6000 短上下文 decode 绝对值最高（MTP 开 ~100 t/s）；无 MTP 长上下文最稳（43.7 @200K）。

> 勘误：本仓库 decode 列曾因"按流式 chunk 计数"低估约 2~3 倍
> （MTP 下单个 chunk 携带多个已接受 token），已改用 `usage.completion_tokens` 重测修正。

## MTP 投机解码长上下文衰减（专题）

跨三设备、两种引擎（vLLM / llama.cpp）的实测汇总：

| 配置 | 短上下文收益 | 长上下文（~148-200K） | 判定 |
|---|---|---|---|
| PRO 5000 NVFP4+MTP（n=1, marlin, vLLM 0.26, **Windows**） | +28% | **+37% @148K** | ✅ 唯一长上下文仍正收益的 vLLM 报告 |
| PRO 6000 NVFP4+MTP（n=1, marlin, vLLM 0.26, Linux，与 PRO 5000 完全同栈） | +74%（1.4K） | **-53% @148K / -60% @200K** | ❌ 未能复现 PRO 5000 |
| PRO 6000 NVFP4+MTP（n=1, cutlass, vLLM 0.21） | +57%（1.4K） | -56% @148K / -64% @200K | ❌ |
| PRO 6000 NVFP4+MTP（n=1, marlin, vLLM 0.21） | +69%（1.4K） | -55% @148K / -62% @200K | ❌ |
| PRO 6000 NVFP4+MTP（n=2, cutlass, vLLM 0.21） | +71%（1.4K） | -58% @177K | ❌ |
| PRO 6000 FP8+MTP（n=2, vLLM 0.21） | +82%（1.4K） | -52% @177K | ❌ |
| PRO 5000 Q6_K draft-mtp（llama.cpp n_max=4） | +56% | ±0% @148K（无净损失）；两并发 -38% | ⚠️ 仅单流可用 |
| PRO 5000 FP8+MTP（n=1） | +16% | -42% @148K | ❌ |
| DGX Spark NVFP4+MTP×3 | +92%（思考模式 11.8→22.7） | 200K 仍有 14.2 t/s（缺无MTP对照） | ⚠️ 低功耗基线 |

### 变量排除实验（PRO 6000，2026-08-16 补测）

为定位"为什么 PRO 5000 的 NVFP4+MTP 长上下文不崩"，在 PRO 6000 上逐一对齐 PRO 5000 的
软件栈（原始数据：`results/prefill-nvfp4-mtp-n1.json`、`prefill-nvfp4-marlin-mtp-n1.json`、
`prefill-nvfp4-v026-marlin-mtp-n1.json`、`prefill-nvfp4-v026-marlin-mtp-n1-ms4.json`）：

| 变量 | 实验配置 | 148K decode | 结论 |
|---|---|---|---|
| n 值 | 0.21 + cutlass，n=2 → **n=1** | 19.3（n=2 为 34.8@100K / 18.2@200K） | ❌ 排除，n=1 反而更差 |
| GEMM 后端 | 0.21 + **marlin**，n=1 | 20.9 | ❌ 排除 |
| vLLM 版本 | **0.26.0** + marlin + n=1 + prefix caching（复刻 PRO 5000 全部启动参数，含 `--linear-backend marlin`、`VLLM_TEST_FORCE_FP8_MARLIN=1`） | 21.4 | ❌ 排除 |
| 调度参数 | 0.26 同上加 `--max-num-seqs 4`、util 0.90 | 19.7 | ❌ 排除 |

**关键反直觉发现：接受率全程健康。** vLLM 0.26 日志显示，即使在 148K~200K 上下文，
per-position 接受率仍 71%~100%（平均接受长度 1.7~2.0，n=1 时满分为 2.0）。
也就是说 PRO 6000 的崩盘**不是** issue #47602 描述的"单层草稿头容量不足 → 接受率坍塌"
机制，而是 **MTP 每步（draft + verify）的开销随上下文长度失控增长**——接受率接近满分
的情况下吞吐仍不到无 MTP 的一半。

**剩余未隔离变量**：操作系统（PRO 5000 Windows vs PRO 6000 Linux）、GPU 驱动分支；
另有口径警告——PRO 5000 仓库未包含其 148K decode 的计时脚本（其 `prefill_test.py`
仅测 TTFT），57.8 t/s（+37%）的具体测量细节无法完全审计，建议视为**单设备自报值**。

**根因补充**（vLLM Issue [#47602](https://github.com/vllm-project/vllm/issues/47602)）：
单层草稿头接受率坍塌机制在 llama.cpp（Vulkan/AMD）与 PRO 5000 FP8 上符合观测；
PRO 6000 的 NVFP4/FP8 崩盘呈现不同特征（接受率健康、每步开销暴增），两者可能并存，
需在 vLLM 侧按"接受率 + 每步耗时"双指标分别诊断。

**MTP 对 prefill 的影响（PRO 5000 实测，可忽略）**：±5% 以内（2.7K/37K/232K 三档），
prefill 排行不变（FP8 > NVFP4 > Q6_K）；TTFT 多出的时间主要是草稿模型 prefill。
MTP 的全部收益/损失都发生在 decode 侧。

**实践建议**：

- 短/中上下文（<30K）：放心开 MTP，收益 28%~92%（三设备一致）
- 长上下文 + **Linux + vLLM**（0.21 与 0.26 均实测）：**关 MTP**（PRO 6000 无 MTP 200K 仍有 39.6~43.7 t/s）
- 长上下文 + Windows + vLLM 0.26 + marlin + n=1：PRO 5000 报告 +37%，Linux 未复现，用前自测
- FP8+MTP 在任何设备上都别用于长上下文

## 单并发 Prefill（tok/s）与 TTFT（秒）

| Prompt | DGX Spark NVFP4 | PRO 5000 FP8 | PRO 5000 Q6_K | PRO 6000 FP8+MTP | PRO 6000 NVFP4+MTP | PRO 6000 Q6_K |
|---|---|---|---|---|---|---|
| ~1-3K | 1,800 / 0.57s | 1,033 / 2.55s | 749 / 3.60s | 6,265 / 0.23s | 8,941 / 0.17s | 1,863 / 0.81s |
| ~30-40K | — | 3,671 / 10.2s | 1,576 / 23.8s | 7,100 / 4.16s | 9,465 / 3.11s | 3,117 / 9.46s |
| ~100K | 1,230 / 83s | — | — | 5,384 / 16.5s | 6,616 / 13.4s | 2,367 / 37.6s |
| ~200-232K | 840 / 244s | 2,114 / 110s | 768 / 302s | 3,869 / 45.8s | 4,447 / 39.9s | 1,676 / 105.9s |
| ~256K（261K tok） | 715 / 366s | — | — | — | — | — |

Prefill/TTFT 维度 vLLM 全面领先 llama.cpp（2~4 倍），RTX PRO 6000 全面领先其余两设备
（agent 体感最关键的指标）。

## 并发扩展（decode 聚合吞吐 tok/s）

| 并发 | DGX Spark NVFP4（单机） | DGX Spark 双机 TP=2 | PRO 5000 FP8 | PRO 5000 NVFP4+MTP | PRO 6000 FP8+MTP | PRO 6000 NVFP4+MTP | PRO 6000 Q6_K（4 槽） |
|---|---|---|---|---|---|---|---|
| 1 | 20.0 | 20.6~22.0 | 37.3 | 63.6（单流 11K 口径） | 78.8 | 95.0 | 49.6 |
| 2 | 22.7 | — | 64.7 | 112.6（两并发总体） | 137.0 | 151.3 | 92.8 |
| 4 | 44.0 | 65.4（+49%） | — | — | 299.3 | 346.5 | 158.5 |
| 8 | 77.7 | — | — | — | 556.2 | **654.1** | — |
| 16 | 115.7（峰值 160） | — | — | — | — | — | — |

双机 TP=2（2× DGX Spark）：单流基本持平（22.7 → 20.6~22.0），并发聚合 +49%（c=4），
prefill 100K 反超单机（1,230 → 1,325 t/s），短 prompt 略吃亏；256GB 统一内存可跑更大模型。
详见 [双机部署实测](https://github.com/Deep-AI-Evo/qwen3.8-27b-nvfp4-dgx-spark-tutorial)。

## 长上下文真实性（大海捞针）

| 设备 | 规模 | 深度 | 结果 |
|---|---|---|---|
| DGX Spark | 255,376 tok | 70% | ✅ 一次答对 |
| PRO 6000 FP8 / NVFP4 | 96,886 tok | 70% | ✅ 均一次答对 |
| PRO 6000 llama.cpp Q6_K | 96,885 tok | 70% | ✅ 一次答对（48.2s） |

## 小结

- **算力档位**：PRO 6000 > PRO 5000 > DGX Spark（单机）。PRO 6000 的 prefill/TTFT/并发
  约为 DGX Spark 的 4~8 倍；双机 TP=2 能把 DGX Spark 的并发聚合拉高 ~49%，缩小差距但
  不改变档位。DGX Spark 的价值在 128GB 统一内存、低功耗桌面形态与双机扩展灵活性。
- **长上下文 decode 结论修正（2026-08-16）**：PRO 5000 报告的 NVFP4+MTP 57.8 t/s @148K（+37%）
  是三设备最高报告值，但 PRO 6000 用**完全相同的软件栈**（vLLM 0.26 + marlin + n=1）复测仅
  21.4 t/s（-53%）——n 值、GEMM 后端、vLLM 版本、调度参数逐一排除，且 PRO 6000 侧接受率
  全程健康（71%~100%），崩盘源于 MTP 每步开销失控而非接受率坍塌。差异指向 OS/驱动层
  （Windows vs Linux），在更多设备验证前该 57.8 应视为单设备自报值。**可复现的长上下文
  最佳实践仍是关 MTP**（PRO 6000 NVFP4 无 MTP 43.7 t/s @200K）。
- **FP8+MTP 长上下文在两台设备都崩**（-42% / -52%），是最不稳定组合；FP8 无 MTP
  是最省心的长上下文基线。
- **llama.cpp Q6_K 的价值**：单文件极简部署、权重质量最高、decode 与 vLLM 无 MTP 接近；
  短上下文 prefill 远弱于 vLLM。Blackwell 自编译务必用 **CUDA 12.8**（13.x kernel 异常，
  见本仓库测试报告 §8）。
- **格式选择**：NVFP4 在 Blackwell 桌面卡上可行且 decode 更快，但需踩坑修复（见各自仓库）；
  FP8 是最省心的基线。
