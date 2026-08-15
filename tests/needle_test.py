#!/usr/bin/env python3
"""长上下文大海捞针（needle-in-a-haystack）测试。

用法: python needle_test.py <api_base> <model> <label> [target_tokens] [depth]
示例: python needle_test.py http://127.0.0.1:8000 qwen38-nvfp4-mtp nvfp4-mtp 100000 0.7

在约 target_tokens 的填充文本中 depth 比例处插入一个随机密码，
然后让模型找回密码。判定：回答中包含密码即通过。
"""
import json
import os
import random
import string
import sys
import time

import openai

FILLER = ("人工智能技术的发展经历了多个阶段。从早期的符号主义到后来的连接主义，"
          "再到如今的大规模预训练模型，每一次范式转移都带来了能力的跃升。"
          "研究人员不断探索更高效的架构、更大规模的数据和更优雅的算法。")


def main():
    api_base, model, label = sys.argv[1], sys.argv[2], sys.argv[3]
    target = int(sys.argv[4]) if len(sys.argv) > 4 else 100000
    depth = float(sys.argv[5]) if len(sys.argv) > 5 else 0.7
    password = "QW38-" + "".join(random.choices(string.ascii_uppercase + string.digits, k=8))

    needle = f"\n在这段文字中隐藏着一个重要信息：取回密码是 {password}。\n"
    filler = FILLER * (target * 2 // len(FILLER))  # 中文 1 字约 0.5 token，放大系数 2
    pos = int(len(filler) * depth)
    text = filler[:pos] + needle + filler[pos:]
    prompt = text + "\n\n问题：上文中隐藏的取回密码是什么？请只回答密码本身。"

    client = openai.OpenAI(base_url=f"{api_base}/v1", api_key="none")
    t0 = time.time()
    r = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=64,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    dt = time.time() - t0
    answer = (r.choices[0].message.content or "").strip()
    passed = password in answer
    result = {
        "label": label, "target_tokens": target, "depth": depth,
        "password": password, "answer": answer, "passed": passed,
        "prompt_tokens": r.usage.prompt_tokens, "time_s": round(dt, 2),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    os.makedirs("results", exist_ok=True)
    path = f"results/needle-{label}-{target}.json"
    with open(path, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("saved:", path)


if __name__ == "__main__":
    main()
