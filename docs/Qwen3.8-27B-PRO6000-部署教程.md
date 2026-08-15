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

## 6. 验证

参照 `docs/Qwen3.8-27B-PRO6000-测试报告.md` 的方法，
或直接在仓库根目录执行 `./tests/run_matrix.sh <label>` 跑完整测试矩阵。
