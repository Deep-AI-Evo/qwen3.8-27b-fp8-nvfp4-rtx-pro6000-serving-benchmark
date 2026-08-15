#!/usr/bin/env python3
"""N 并发 decode profile 测试。

用法: python conc_test.py <api_base> <model> <N> <max_tokens> <label>
示例: python conc_test.py http://127.0.0.1:8000 qwen38-nvfp4-mtp 4 500 nvfp4-mtp

方法（与同组织 rtx-pro5000 仓库口径一致）:
- N 个线程同时发请求，分别计时
- 单流速度 = completion_tokens / 各自耗时; 总体吞吐 = Σtokens / 窗口时间
- 每条请求带随机前缀，杜绝缓存命中
- 结果写 results/conc-<label>-n<N>.json
"""
import json
import os
import random
import sys
import threading
import time

import openai


def worker(idx, api_base, model, max_tokens, out):
    client = openai.OpenAI(base_url=f"{api_base}/v1", api_key="none")
    prompt = f"{random.getrandbits(64):016x} " * 20 + "写一段关于人工智能发展的长文。"
    t0 = time.time()
    r = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=max_tokens,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    dt = time.time() - t0
    tok = r.usage.completion_tokens
    out[idx] = {"tokens": tok, "time_s": round(dt, 2), "tps": round(tok / dt, 1)}


def main():
    api_base, model, n, max_tokens, label = (
        sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5])
    out = [None] * n
    threads = [threading.Thread(target=worker, args=(i, api_base, model, max_tokens, out))
               for i in range(n)]
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    window = time.time() - t0
    total_tokens = sum(x["tokens"] for x in out)
    result = {
        "label": label, "concurrency": n, "max_tokens": max_tokens,
        "per_stream": out,
        "avg_stream_tps": round(sum(x["tps"] for x in out) / n, 1),
        "aggregate_tps": round(total_tokens / window, 1),
        "window_s": round(window, 2),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    os.makedirs("results", exist_ok=True)
    path = f"results/conc-{label}-n{n}.json"
    with open(path, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("saved:", path)


if __name__ == "__main__":
    main()
