# Qwen3.8-27B FP8 / NVFP4 部署教程（RTX PRO 6000 Blackwell + vLLM）

> 环境：Linux x86_64 · RTX PRO 6000 Blackwell 96GB（sm_120）· 驱动 595.84 / CUDA 13.2 ·
> vLLM 0.21.0 · PyTorch 2.11.0+cu130 · Python 3.12
>
> 本教程同样沉淀了真实踩坑，可直接交给代码智能体一键复现（见 README 的 Agent 用法）。

## 1. 环境准备

```bash
pip install -U vllm openai        # 实测 0.21.0
pip install -U modelscope hf_transfer
```

确认 GPU：`nvidia-smi` 应显示 RTX PRO 6000 Blackwell（96GB）。

## 2. 下载模型

```bash
mkdir -p ~/models

# FP8（官方，走 ModelScope）
modelscope download --model Qwen/Qwen3.8-27B-FP8 --local_dir ~/models/Qwen3.8-27B-FP8

# NVFP4（unsloth，HF 直连不通时走镜像）
HF_ENDPOINT=https://hf-mirror.com hf download unsloth/Qwen3.8-27B-NVFP4 \
  --local-dir ~/models/Qwen3.8-27B-NVFP4
```

体积：FP8 约 29GB（66 个 safetensors 分片），NVFP4 约 22GB（13 个文件）。

## 3. 必做修复（vLLM 0.21.0 + 本机环境的 3 个坑）

### 3.1 FlashInfer 采样器崩溃（FP8 / NVFP4 都需要）

系统 CUDA 工具包若为 12.x 旧版，FlashInfer 的 JIT 架构检测不认 sm_120，
引擎初始化时在采样器 profile 阶段崩溃。
**修复：所有启动命令加环境变量 `VLLM_USE_FLASHINFER_SAMPLER=0`**（回退 PyTorch 原生采样器，无性能损失）。

### 3.2 NVFP4 加载报 `lm_head.weight_scale`（vLLM bug，需补丁）

unsloth NVFP4 的 lm_head 是 FP8 量化，但 vLLM 0.21.0 的
`vllm/model_executor/models/qwen3_5.py` 和 `qwen3_5_mtp.py` 创建
`ParallelLMHead` 时漏传 `quant_config`（qwen2/qwen3 均有传，属上游疏漏）。

补丁（两个文件各一处，搜 `self.lm_head = ParallelLMHead(`，也可直接套用
`patches/lm_head_quant_config.patch`）：

```python
self.lm_head = ParallelLMHead(
    config.vocab_size,
    config.hidden_size,
    quant_config=self.quant_config,   # 补这一行
    prefix=maybe_prefix(prefix, "lm_head"),
)
```

### 3.3 NVFP4 输出乱码（静态 FP8 KV cache 量化不兼容）

unsloth NVFP4 的 `config.json` 带 `quantization_config.kv_cache_scheme`
（16 个全注意力层带静态 k_scale/v_scale）。该路径在本机不可用：
自动选后端会选中需要 JIT 的 FlashInfer attention（崩溃）；
强制 TRITON_ATTN 虽能启动但输出全是乱码。

**修复：删除 kv_cache_scheme，KV cache 回落 bf16**：

```bash
cd ~/models/Qwen3.8-27B-NVFP4
cp config.json config.json.bak
python3 -c "
import json
d = json.load(open('config.json'))
d['quantization_config'].pop('kv_cache_scheme', None)
json.dump(d, open('config.json', 'w'), ensure_ascii=False, indent=2)
"
```

另外 NVFP4 GEMM 必须显式指定 `VLLM_NVFP4_GEMM_BACKEND=cutlass`
（默认的 flashinfer-cutlass 会因 FlashInfer JIT 失败而崩）。

## 4. 启动服务

### FP8（官方 FP8 + MTP）

```bash
VLLM_USE_FLASHINFER_SAMPLER=0 vllm serve ~/models/Qwen3.8-27B-FP8 \
  --served-model-name qwen38-fp8-mtp \
  --port 8000 \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.92 \
  --max-num-seqs 512 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}'
```

### NVFP4（unsloth NVFP4 + MTP，推荐）

```bash
VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_NVFP4_GEMM_BACKEND=cutlass \
  vllm serve ~/models/Qwen3.8-27B-NVFP4 \
  --served-model-name qwen38-nvfp4-mtp \
  --port 8000 \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.92 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}'
```

参数说明：

- `--max-model-len 262144`：原生满血上下文；显存够（96GB），KV cache 仍有 ~80 万 tokens
- `--speculative-config '{"method":"mtp","num_speculative_tokens":2}'`：原生 MTP 投机采样，decode 提速约 2.3 倍
- `--max-num-seqs 512`：FP8+MTP 必需（GDN 线性注意力的 state cache 每并发序列占一块，默认 1024 会超界报错）；NVFP4 权重小，不加也行
- 思考模式默认开启；请求里用 `chat_template_kwargs: {"enable_thinking": false}` 可关闭
- 首次启动约 2 分钟（权重加载 + CUDA graph 捕获）；**启动后先发一个预热请求**

## 5. API 调用示例

```bash
curl http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen38-nvfp4-mtp",
  "messages": [{"role": "user", "content": "用一句话介绍北京"}],
  "temperature": 0.7,
  "chat_template_kwargs": {"enable_thinking": false}
}'
```

服务就绪后访问 `http://127.0.0.1:8000/docs` 查看 OpenAI 兼容 API。

## 4.5 llama.cpp Q6_K（可选第三方案，⚠️ 必须用 CUDA 12.8 编译）

**警告：Blackwell（sm_120）上不要用 CUDA 13.x 自编译 llama.cpp**——实测 FA kernel
异常（prefill 慢 5~26 倍，`-fa 0` 路径直接 CUDA error 崩溃）。NVIDIA Blackwell 迁移指南
建议 sm_120 用 CUDA 12.8 编译，本机实测 CUDA 12.8 构建一切正常（数据见测试报告 §9）。
（Windows 用户直接用官方预编译二进制即可，无此问题。）

### 4.5.1 准备 CUDA 12.8 工具链（无需 root，不动系统驱动）

```bash
# 下载 runfile（国内自动跳转 nvidia.cn 节点），只装 toolkit 到用户目录
curl -LO https://developer.download.nvidia.com/compute/cuda/12.8.1/local_installers/cuda_12.8.1_570.124.06_linux.run
sh cuda_12.8.1_570.124.06_linux.run --silent --toolkit \
  --toolkitpath=$HOME/cuda-12.8 --no-man-page --no-opengl-libs --override
$HOME/cuda-12.8/bin/nvcc --version   # 应显示 release 12.8
```

### 4.5.2 编译 llama.cpp（b9692）

```bash
git clone https://github.com/ggml-org/llama.cpp.git && cd llama.cpp
git fetch --depth 1 origin tag b9692 && git checkout b9692

cmake -B build -DGGML_CUDA=ON -DGGML_CUDA_FA_ALL_QUANTS=ON \
  -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DCUDAToolkit_ROOT=$HOME/cuda-12.8 \
  -DCMAKE_CUDA_COMPILER=$HOME/cuda-12.8/bin/nvcc \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --target llama-server llama-bench -j$(nproc)
```

- `GGML_CUDA_FA_ALL_QUANTS=ON`：q8_0 KV cache 下 FA 生效所需
- 编译后先用 `llama-bench` 自检（10 秒，可提前发现 kernel 异常构建）：

```bash
build/bin/llama-bench -m <模型.gguf> -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 -p 16384 -n 4
# 健康参考值：Q6_K 27B pp16K ≈ 3,500 t/s；若只有几百 t/s，说明构建有问题
```

### 4.5.3 下载模型并启动

```bash
# GGUF（unsloth，HF 直连不通时走镜像）
HF_ENDPOINT=https://hf-mirror.com hf download unsloth/Qwen3.8-27B-GGUF \
  Qwen3.8-27B-UD-Q6_K_XL.gguf --local-dir ~/models/Qwen3.8-27B-GGUF

LD_LIBRARY_PATH=$HOME/cuda-12.8/lib64 build/bin/llama-server \
  --model ~/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q6_K_XL.gguf \
  --host 127.0.0.1 --port 8000 \
  --ctx-size 1048576 --n-gpu-layers 99 --threads 8 --parallel 4 \
  --flash-attn on --no-mmap \
  --cache-type-k q8_0 --cache-type-v q8_0
```

- `--ctx-size 1048576 --parallel 4`：总上下文 1M，每槽上限 262K（跑 200K 单请求必须给足总上下文）
- `--cache-type-k/v q8_0`：KV 显存减半，速度代价 ~1%
- llama.cpp 兼容 OpenAI API，本仓库测试脚本可直接用

## 6. 验证

参照 `docs/Qwen3.8-27B-PRO6000-测试报告.md` 的方法，
或直接在仓库根目录执行 `./tests/run_matrix.sh <label>` 跑完整测试矩阵。

## 7. 附录：vLLM 0.26.0 + marlin 后端（对齐 PRO 5000 栈的可选配方）

为排查 MTP 长上下文崩盘（见横向对比文档 MTP 专题），本机在独立 venv 中复刻了
PRO 5000（Windows）的完整软件栈。**注意：此配置在本机并未带来 MTP 长上下文收益，
仅作为跨设备对齐/排障参考留存。**

```bash
python3 -m venv ~/vllm026-venv
~/vllm026-venv/bin/pip install vllm==0.26.0   # 自动带入 torch 2.11 + flashinfer 0.6.14
```

四个坑及解法（前两个与 PRO 5000 Windows 侧完全相同）：

1. **`--linear-backend marlin` 需要 `VLLM_TEST_FORCE_FP8_MARLIN=1`**，否则 marlin 拒接
   FP8 层（sm_120 被认为有原生 FP8，强制走 marlin 需该开关）。
2. **lm_head 缺 `output_size_per_partition`**（vLLM 0.26 marlin 路径 bug，PRO 5000 补丁①②同款）。
   修复两处（均在 venv 的 site-packages 内）：
   - `vllm/model_executor/layers/quantization/utils/marlin_utils_fp8.py` 的
     `prepare_fp8_layer_for_marlin()`：`part_size_n/k` 改为 `getattr(layer, "output_size_per_partition", None)`
     兜底 `layer.num_embeddings_per_partition` / `layer.embedding_dim`；并按 weight 实际形状
     自适应 `size_k_first`（ParallelLMHead 的 weight 是 (N, K) 布局）。
   - `vllm/model_executor/kernels/linear/scaled_mm/marlin.py` 的 `apply_weights()`：
     `size_n/size_k` 做同样的兜底。
3. **flashinfer 误认系统 nvcc**：若系统 `/usr/bin/nvcc` 低于 12.9，flashinfer 会判定
   "SM 12.x requires CUDA >= 12.9" 并清空目标架构（报 "FlashInfer requires GPUs with sm75 or higher"）。
   解法：启动时显式指定 `FLASHINFER_CUDA_ARCH_LIST="12.0f"`，并把 `CUDA_HOME` 指向 venv 自带的
   CUDA 13.3（`~/vllm026-venv/lib/python3.12/site-packages/nvidia/cu13`）。
4. **flashinfer JIT 编译 sampling 算子失败**（pip 版 nvcc 与头文件版本不齐）：直接
   `VLLM_USE_FLASHINFER_SAMPLER=0` 绕开，attention 走 FLASH_ATTN 不受影响。

启动命令（与 PRO 5000 `scripts/run_vllm_nvfp4_mtp_bench.bat` 参数逐一对应）：

```bash
export CUDA_HOME=~/vllm026-venv/lib/python3.12/site-packages/nvidia/cu13
export PATH=$CUDA_HOME/bin:$PATH
VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_TEST_FORCE_FP8_MARLIN=1 FLASHINFER_CUDA_ARCH_LIST="12.0f" \
  ~/vllm026-venv/bin/vllm serve ~/models/Qwen3.8-27B-NVFP4 \
  --served-model-name qwen38-nvfp4-026 \
  --reasoning-parser qwen3 \
  --linear-backend marlin \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 4 \
  --enable-prefix-caching \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
  --port 8000
```
