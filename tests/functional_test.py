import os
os.makedirs("results", exist_ok=True)
#!/usr/bin/env python3
"""对本地 vLLM OpenAI 服务做功能 + 吞吐测试，结果存 JSON。"""
import json, sys, time
import openai

port, tag = sys.argv[1], sys.argv[2]
client = openai.OpenAI(base_url=f"http://localhost:{port}/v1", api_key="none")
model = client.models.list().data[0].id
print(f"model id: {model}")
results = {"model_id": model, "cases": {}, "bench": {}}

CASES = {
    "math": "一个农夫有 17 只羊，除了 9 只以外都死了，还剩几只？请直接回答数字并简单解释。",
    "code": "用 Python 写一个函数判断字符串是否为回文，要求忽略大小写和非字母数字字符。只给代码和一句说明。",
    "reason": "小明、小红、小刚三人中，只有一人说了真话。小明说：是小红做的。小红说：不是我。小刚说：不是我做的。请问是谁做的？",
    "chinese": "用三句话总结《红楼梦》的主题。",
}

# 1) 功能正确性（temperature=0 保证可复现，thinking 关闭以加快速度）
for name, prompt in CASES.items():
    t0 = time.time()
    r = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0, max_tokens=1024,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    dt = time.time() - t0
    msg = r.choices[0].message
    results["cases"][name] = {
        "content": msg.content,
        "reasoning": getattr(msg, "reasoning_content", None),
        "prompt_tokens": r.usage.prompt_tokens,
        "completion_tokens": r.usage.completion_tokens,
        "time_s": round(dt, 2),
    }
    print(f"[case {name}] {r.usage.completion_tokens} tok in {dt:.1f}s")

# 2) thinking 模式抽样
t0 = time.time()
r = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "9.11 和 9.9 哪个大？"}],
    temperature=0, max_tokens=2048,
)
results["cases"]["thinking"] = {
    "content": r.choices[0].message.content,
    "reasoning": (getattr(r.choices[0].message, "reasoning_content", None) or "")[:500],
    "completion_tokens": r.usage.completion_tokens,
    "time_s": round(time.time() - t0, 2),
}
print(f"[case thinking] {r.usage.completion_tokens} tok in {time.time()-t0:.1f}s")

# 3) 吞吐：短输入长输出（decode 速度）
t0 = time.time()
r = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "从 1 数到 500，每个数字之间用空格分隔，不要输出其他内容。"}],
    temperature=0, max_tokens=2000,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
dt = time.time() - t0
out_tok = r.usage.completion_tokens
results["bench"]["decode"] = {"out_tokens": out_tok, "time_s": round(dt, 2),
                              "tok_per_s": round(out_tok / dt, 1),
                              "prompt_tokens": r.usage.prompt_tokens}
print(f"[bench decode] {out_tok} tok in {dt:.1f}s = {out_tok/dt:.1f} tok/s")

# 4) 吞吐：长输入短输出（prefill 速度）——重复长文本
long_text = "人工智能正在改变世界。" * 1500  # ~1.5万字符
t0 = time.time()
r = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": long_text + "\n\n请用一句话概括上文。"}],
    temperature=0, max_tokens=100,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
dt = time.time() - t0
in_tok = r.usage.prompt_tokens
results["bench"]["prefill"] = {"in_tokens": in_tok, "time_s": round(dt, 2),
                               "tok_per_s": round(in_tok / dt, 1),
                               "completion": r.choices[0].message.content}
print(f"[bench prefill] {in_tok} tok in {dt:.1f}s = {in_tok/dt:.1f} tok/s")

with open(f"results/functional-{tag}.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print("saved to", f"results/functional-{tag}.json")
