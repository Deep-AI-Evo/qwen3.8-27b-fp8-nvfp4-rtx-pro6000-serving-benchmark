#!/usr/bin/env python3
"""prefill / TTFT / decode 测试（随机文本防 prefix cache 命中）。

用法: python prefill_test.py <api_base> <model> <label> [sizes...]
示例: python prefill_test.py http://127.0.0.1:8000 qwen38-nvfp4-mtp nvfp4-mtp 1024 32768 100000 200000

方法（与同组织 rtx-pro5000 仓库口径一致）:
- OpenAI 兼容流式请求; TTFT = 请求发出 -> 首个 token
- prefill t/s = prompt_tokens / TTFT
- decode t/s = 首 token 之后的 tokens / 生成时长
- 每轮 prompt 用随机 token 序列开头，杜绝前缀缓存命中
- 每个尺寸 3 轮取均值，结果写 results/prefill-<label>.json
"""
import json
import os
import random
import sys
import time

import openai

WORDS = (
    "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu "
    "nu xi omicron pi rho sigma tau upsilon phi chi psi omega fox dog cat "
    "bird tree river mountain cloud stone metal water fire earth wind"
).split()


def rand_prompt(target_tokens: int) -> str:
    # 头部放 32 个随机十六进制词，保证任何前缀缓存都失效
    head = " ".join(f"{random.getrandbits(64):016x}" for _ in range(32))
    # 英文单词约 1.3 token/词，先按 0.75 词/token 估算，宁多勿少
    n_words = int(target_tokens * 0.8)
    body = " ".join(random.choice(WORDS) for _ in range(n_words))
    return head + " " + body + "\n\n用一句话总结上文。"


def run_round(client, model, prompt, max_tokens=128):
    t0 = time.time()
    ttft = None
    gen_tokens = 0
    usage = None
    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=max_tokens,
        stream=True,
        stream_options={"include_usage": True},
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    for chunk in stream:
        if chunk.usage:
            usage = chunk.usage
        if chunk.choices and chunk.choices[0].delta.content:
            if ttft is None:
                ttft = time.time() - t0
            gen_tokens += 1
    total = time.time() - t0
    decode_time = total - (ttft or total)
    prompt_tokens = usage.prompt_tokens if usage else 0
    return {
        "prompt_tokens": prompt_tokens,
        "ttft_s": round(ttft or 0, 3),
        "prefill_tps": round(prompt_tokens / ttft, 1) if ttft else None,
        "decode_tps": round(gen_tokens / decode_time, 1) if decode_time > 0 else None,
        "completion_tokens": usage.completion_tokens if usage else gen_tokens,
    }


def main():
    api_base, model, label = sys.argv[1], sys.argv[2], sys.argv[3]
    sizes = [int(s) for s in sys.argv[4:]] or [1024, 32768, 100000, 200000]
    client = openai.OpenAI(base_url=f"{api_base}/v1", api_key="none")
    out = {"label": label, "sizes": {}}
    for size in sizes:
        rounds = []
        for r in range(3):
            res = run_round(client, model, rand_prompt(size))
            rounds.append(res)
            print(f"[{label}] {size} round{r+1}: ttft={res['ttft_s']}s "
                  f"prefill={res['prefill_tps']} t/s decode={res['decode_tps']} t/s "
                  f"({res['prompt_tokens']} tok)", flush=True)
            time.sleep(2)
        avg = {k: round(sum(x[k] for x in rounds) / len(rounds), 2)
               for k in ("prompt_tokens", "ttft_s", "prefill_tps", "decode_tps")}
        out["sizes"][str(size)] = {"avg": avg, "rounds": rounds}
        print(f"[{label}] {size} AVG: {avg}", flush=True)
    os.makedirs("results", exist_ok=True)
    path = f"results/prefill-{label}.json"
    with open(path, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("saved:", path)


if __name__ == "__main__":
    main()
