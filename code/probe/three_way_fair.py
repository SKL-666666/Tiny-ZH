#!/usr/bin/env python3
"""修正版三方对比：公平设置停止条件。

上一版的错误
------------
1. minimind-3 没有设 eos_token_id，一直输出到 max_tokens=300 才停
   -> 假象：看起来「话多」，实际是没停
2. minimind-3 的 <|im_end|> 是 token id 2，不是我们的 3
   -> 用错 id 会导致不停止
3. chat_template 会插入 <think> 标签，需要 skip

修正
----
- 三个模型都设 stop 条件
- max_tokens 统一 200
- minimind-3 用它的 tokenizer 的 eos_token_id
"""
import json
import os
import re
import sys

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

QUESTIONS = [
    ("简单", "你好"),
    ("简单", "你是谁？"),
    ("简单", "今天星期几？"),
    ("简单", "1+1等于几？"),
    ("简单", "说一句话。"),
    ("中等", "今天天气怎么样？"),
    ("中等", "推荐一本好书。"),
    ("中等", "怎么学习编程？"),
    ("中等", "什么是人工智能？"),
    ("中等", "介绍一下北京。"),
    ("中等", "你怎么看待友谊？"),
    ("困难", "为什么天是蓝的？"),
    ("困难", "解释一下什么是互联网。"),
    ("困难", "如何保持健康？"),
    ("困难", "介绍一下中国菜。"),
    ("困难", "给我讲讲春天。"),
    ("困难", "人为什么要睡觉？"),
    ("困难", "什么是民主？"),
    ("困难", "请写一段关于秋天的描写。"),
    ("困难", "比较一下猫和狗。"),
    ("困难", "如果我想学游泳，第一步该做什么？"),
]

MAX_TOKENS = 200


def gen_gguf(model_path, tag, n=2):
    import llama_cpp
    print(f"[{tag}] 加载 {model_path}")
    llm = llama_cpp.Llama(model_path=model_path, n_ctx=512,
                          n_threads=16, verbose=False)
    out = []
    for level, q in QUESTIONS:
        prompt = f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
        answers = []
        for _ in range(n):
            try:
                r = llm(prompt, max_tokens=MAX_TOKENS, temperature=0.7, top_p=0.9,
                        repeat_penalty=1.15, stop=["<|im_end|>"])
                answers.append(r["choices"][0]["text"].strip())
            except Exception as e:
                answers.append(f"[失败 {e}]")
        out.append({"level": level, "q": q, "answers": answers})
        print(f"  {q[:18]:<20} {len(answers[0]):>3}字")
    return out


def gen_hf(model_dir, tag, n=2):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[{tag}] 加载 {model_dir}")
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=True)
    model.eval()

    eos_id = tok.eos_token_id
    print(f"  eos_token_id = {eos_id} ({tok.eos_token})")

    out = []
    for level, q in QUESTIONS:
        msgs = [{"role": "user", "content": q}]
        try:
            text = tok.apply_chat_template(msgs, tokenize=False,
                                           add_generation_prompt=True)
        except Exception:
            text = f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"

        ids = tok(text, return_tensors="pt").to("cuda")
        answers = []
        for _ in range(n):
            with torch.no_grad():
                g = model.generate(
                    **ids,
                    max_new_tokens=MAX_TOKENS,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.15,
                    eos_token_id=eos_id,
                    pad_token_id=eos_id,
                )
            ans = tok.decode(g[0][ids["input_ids"].shape[1]:],
                             skip_special_tokens=True).strip()
            # 去掉 think 标签残留
            ans = re.sub(r"</?think>", "", ans).strip()
            answers.append(ans)
        out.append({"level": level, "q": q, "answers": answers})
        print(f"  {q[:18]:<20} {len(answers[0]):>3}字")
    return out


def main():
    results = {}

    for path, tag in (("out/tiny-zh-64m-f16.gguf", "v1_ShareGPT"),
                      ("out/tiny-zh-64m-v2-f16.gguf", "v2_minimind")):
        if os.path.exists(path):
            try:
                results[tag] = gen_gguf(path, tag)
            except Exception as e:
                print(f"[FAIL] {tag}: {e}")

    mm3 = "out_base/minimind3"
    if os.path.exists(f"{mm3}/model.safetensors"):
        try:
            results["minimind3_原版"] = gen_hf(mm3, "minimind3")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[FAIL] minimind3: {e}")

    with open("logs/three_way2.json", "w", encoding="utf-8") as f:
        json.dump({"questions": QUESTIONS, "results": results},
                  f, ensure_ascii=False, indent=2)

    # 可读报告
    lines = ["# 三方公平对比（统一 max_tokens=200，都设停止符）\n",
             "| 模型 | 参数 | 架构 |", "|---|---|---|",
             "| v1 (ShareGPT) | 64.82M | 13层640维 |",
             "| v2 (minimind) | 64.82M | 13层640维 |",
             "| minimind-3 原版 | 63.91M | Qwen3 8层768维 |", ""]

    for level in ("简单", "中等", "困难"):
        lines.append(f"\n## {level}\n")
        for lv, q in QUESTIONS:
            if lv != level:
                continue
            lines.append(f"### 问：{q}\n")
            for tag, rs in results.items():
                item = next((r for r in rs if r["q"] == q), None)
                if not item:
                    continue
                a = item["answers"][0].replace("\n", " ")
                lines.append(f"**{tag}**（{len(item['answers'][0])}字）")
                lines.append(f"> {a}\n")
            lines.append("| 模型 | 分数 /5 |")
            lines.append("|---|---|")
            lines.append("| v1 | |")
            lines.append("| v2 | |")
            lines.append("| minimind3 | |")
            lines.append("\n---\n")

    with open("logs/three_way2.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n结果: logs/three_way2.json")
    print("报告: logs/three_way2.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())