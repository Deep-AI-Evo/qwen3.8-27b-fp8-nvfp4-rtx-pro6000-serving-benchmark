# Qwen3.8-27B 测试报告（RTX PRO 6000 Blackwell 96GB）

> 测试日期：2026-08-15 · vLLM 0.21.0 · torch 2.11.0+cu130 · 262K 上下文 · MTP(k=2) 已开启
> 方法口径与同组织 [RTX PRO 5000 仓库](https://github.com/Deep-AI-Evo/qwen3.8-27b-q6k-fp8-rtx-pro5000-serving-benchmark) 一致：
> 流式请求，TTFT = 请求发出→首 token；prefill t/s = prompt_tokens / TTFT；
> decode t/s 剔除首 token；每轮随机前缀防缓存；关键数据 3 轮取均值。

## 1. 功能测试（temperature=0，5 项全过）

| 用例 | FP8+MTP | NVFP4+MTP |
|---|---|---|
| 数学陷阱（17 只羊剩 9 只） | ✅ | ✅ |
| 代码（回文函数） | ✅ 输出与 NVFP4 逐字一致 | ✅ |
| 逻辑推理（三人真话） | ✅ | ✅ |
| 中文总结（红楼梦） | ✅ | ✅ |
| thinking 模式（9.11 vs 9.9） | ✅ 区分小数/版本号 | ✅ |

原始输出见 `results/functional-*.json`。

## 2. 单并发 Prefill / TTFT / Decode（随上下文衰减）

### FP8+MTP

| Prompt | TTFT | Prefill | Decode |
|---|---|---|---|
| ~1.4K | 0.23 s | 6,265 t/s | 95.6 t/s |
| ~29.5K | 4.14 s | 7,124 t/s | 53.3 t/s |
| ~88.9K | 16.5 s | 5,390 t/s | 28.5 t/s |
| ~177.4K | 45.4 s | 3,906 t/s | 19.1 t/s |

### NVFP4+MTP

| Prompt | TTFT | Prefill | Decode |
|---|---|---|---|
| ~1.4K | 0.16 s | 9,431 t/s | 100.2 t/s |
| ~29.5K | 2.97 s | 9,915 t/s | 62.1 t/s |
| ~88.9K | 13.2 s | 6,766 t/s | 34.8 t/s |
| ~177.4K | 39.5 s | 4,496 t/s | 18.2 t/s |

说明：decode 随上下文衰减——64 层中 16 层全注意力的 KV 读取随长度增长，
48 层 GDN 线性注意力与长度无关；MTP 接受率也随上下文变长而下降。
200K 档 FP8 与 NVFP4 decode 基本持平（19.1 vs 18.2），此时瓶颈在注意力而非权重读取。
prefill 方面 NVFP4 各档快 FP8 约 15~50%。

> 勘误：本报告 decode 列曾因"按流式 chunk 计数"而低估约 2~3 倍
> （MTP 下单个 chunk 携带多个 token），已改为按 `usage.completion_tokens` 统计并重测。

### FP8 / NVFP4（无 MTP，Decode 列）

| Prompt | FP8 Decode | NVFP4 Decode |
|---|---|---|
| ~1.4K | 52.6 t/s | 58.6 t/s |
| ~29.5K | 49.9 t/s | 55.4 t/s |
| ~88.9K | 45.6 t/s | 49.1 t/s |
| ~177.4K | 39.6 t/s | 43.7 t/s |

对比 §2 的 MTP 开数据：**MTP 只在短上下文划算**。200K 档无 MTP（39.6 / 43.7）
是 MTP 开（19.1 / 18.2）的 2 倍以上——长上下文下 MTP 接受率衰减，验证开销反成负担。

### llama.cpp Q6_K（b9692 自编译，CUDA 12.8，q8_0 KV，FA on）

| Prompt | TTFT | Prefill | Decode |
|---|---|---|---|
| ~1.4K | 0.81 s | 1,863 t/s | 55.4 t/s |
| ~29.5K | 9.46 s | 3,117 t/s | 51.6 t/s |
| ~88.9K | 37.6 s | 2,367 t/s | 44.4 t/s |
| ~177.4K | 105.9 s | 1,676 t/s | 35.7 t/s |

decode 与 vLLM 无 MTP 接近（55 vs 53~59 t/s）；prefill/TTFT 落后 vLLM 2~4 倍。
⚠️ 此数据基于 CUDA 12.8 构建；CUDA 13.x 自编译在 sm_120 上 FA kernel 异常（见 §9）。

## 3. 并发扩展（decode profile，500 tok/流）

| 并发 | FP8+MTP 单流 | FP8+MTP 聚合 | NVFP4+MTP 单流 | NVFP4+MTP 聚合 | Q6_K 单流 | Q6_K 聚合 |
|---|---|---|---|---|---|---|
| 1 | 79.1 t/s | 78.8 t/s | 95.5 t/s | 95.0 t/s | 49.8 t/s | 49.6 t/s |
| 2 | 70.3 t/s | 137.0 t/s | 77.0 t/s | 151.3 t/s | 46.5 t/s | 92.8 t/s |
| 4 | 77.5 t/s | 299.3 t/s | 91.1 t/s | 346.5 t/s | 39.8 t/s | 158.5 t/s |
| 8 | 73.9 t/s | 556.2 t/s | 86.3 t/s | **654.1 t/s** | —（仅 4 槽） | — |

单流速度在 8 并发下仅降 ~7~10%，批处理效率极高；聚合吞吐近线性扩展。
llama.cpp（--parallel 4）4 并发聚合 158.5 t/s，扩展性也不错，但单流与聚合均低于 vLLM。

## 4. 长上下文真实性（大海捞针）

100K 上下文、70% 深度藏随机密码，三个方案均一次答对：

| 模型 | Prompt tokens | 深度 | 结果 | 耗时 |
|---|---|---|---|---|
| FP8+MTP | 96,886 | 70% | ✅ 一次答对 | 19.6 s |
| NVFP4+MTP | 96,886 | 70% | ✅ 一次答对 | 16.1 s |
| llama.cpp Q6_K | 96,885 | 70% | ✅ 一次答对 | 48.2 s |

## 5. MTP 投机采样对比

### 5.1 短上下文基准（32K 上限，decode 数数任务）

| 配置 | decode | 提升 | MTP 接受率 |
|---|---|---|---|
| FP8 | 50.9 t/s | 基线 | — |
| FP8+MTP(k=2) | 115.8 t/s | 2.28× | 91~100% |
| NVFP4 | 56.4 t/s | +11% | — |
| NVFP4+MTP(k=2) | 132.3 t/s | 2.35× | 83~99.5% |

接受率随任务可预测性变化：数数/模板化文本接近 100%，自由生成约 60~80%。

### 5.2 MTP 收益随上下文衰减（decode t/s，同机实测）

| 上下文 | FP8 无 MTP | FP8+MTP | NVFP4 无 MTP | NVFP4+MTP |
|---|---|---|---|---|
| ~1.4K | 52.6 | 95.6（1.82×） | 58.6 | 100.2（1.71×） |
| ~29.5K | 49.9 | 53.3（1.07×） | 55.4 | 62.1（1.12×） |
| ~88.9K | 45.6 | 28.5（0.63×） | 49.1 | 34.8（0.71×） |
| ~177.4K | 39.6 | 19.1（0.48×） | 43.7 | 18.2（0.42×） |

**规律：MTP 接受率随上下文变长而下降，验证（draft+verify）开销却随 KV 变长而上升，
约 30K 之后开 MTP 反而是负优化。** 长上下文生成请用无 MTP 配置。

## 6. FP8 vs NVFP4 总结（同机同口径）

| 指标 | FP8+MTP | NVFP4+MTP | NVFP4 优势 |
|---|---|---|---|
| 权重体积 | 29 GB | 22 GB | -24% |
| decode（短上下文基准） | 115.8 t/s | 132.3 t/s | +14% |
| prefill（7.5K） | 4,478 t/s | 7,880 t/s | +76% |
| prefill（177K） | 3,869 t/s | 4,447 t/s | +15% |
| TTFT（177K） | 45.8 s | 39.9 s | -13% |
| 8 并发聚合 | 556 t/s | 654 t/s | +18% |
| 功能正确性 | 全过 | 全过 | 平 |

## 7. 日常使用建议

- 交互/agent 场景直接开 MTP；NVFP4+MTP 是本机短上下文最优配置
- **长上下文（>30K）生成场景关 MTP**：200K 档无 MTP（39.6~43.7 t/s）是 MTP 开（18~19）的 2 倍多
- 100K 以内长文灌入 vLLM 约 13~17 s，随意用；200K 级约 40~46 s，适合"一次灌入 + 多轮问答"
- llama.cpp Q6_K 适合追求权重质量/极简部署的场景；decode 接近 vLLM 无 MTP，但 prefill 慢 2~4 倍
- 并发 ≤8 路时 vLLM 单流体验几乎无损（≥73 t/s），批量任务可继续加并发吃满吞吐
- 复杂推理/代码开思考模式；闲聊/翻译/摘要关思考更快
- 启动后先发预热请求；避免同机其他 GPU 任务抢占（会污染数据，参考兄弟仓库踩坑记录）

## 8. llama.cpp 在 Blackwell（sm_120）上的 CUDA 13.x kernel 异常（重要）

实测自编译 llama.cpp（master 与 b9692 均复现）在 sm_120 + CUDA 13.3 下：

| 现象 | CUDA 13.3 构建 | CUDA 12.8 构建 |
|---|---|---|
| Q6_K 27B pp16K（llama-bench） | 696 t/s | **3,536 t/s（5.1×）** |
| 0.6B 稠密模型 pp16K | 1,722 t/s | **44,864 t/s（26×）** |
| `-fa 0`（非 FlashAttention 路径） | CUDA error 崩溃 | 正常 |
| Q6_K 服务 32K prefill | 427 t/s | 3,282 t/s |

- 与模型无关：稠密小模型同样塌掉；FA_ALL_QUANTS、KV 类型、ubatch、llama.cpp 版本均无影响
- GPU 无降频、显存带宽正常（decode 权重读取 1.3 TB/s）；vLLM（torch cu130）完全正常
- 与已知案例一致：CUDA 13.x 编译的 llama.cpp 在 sm_120 上 kernel 异常
  （NVIDIA Blackwell 迁移指南亦建议 sm_120 用 CUDA 12.8 编译）
- **官方 Windows 预编译二进制（PRO 5000 仓库所用）无此问题**；Linux 自编译请务必用 CUDA 12.8，
  构建方法见部署教程 §4

## 9. 踩坑记录（部署侧 4 项）

见 `docs/Qwen3.8-27B-PRO6000-部署教程.md` §3：FlashInfer sampler JIT（sm_120）、
vLLM lm_head 漏传 quant_config（两个文件）、NVFP4 静态 KV 量化乱码、
FP8+MTP 的 Mamba cache 超界（--max-num-seqs 512）。
