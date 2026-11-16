#!/usr/bin/env python3
"""三方对比测试：v1 vs v2 vs minimind-3 原版。

全部题目 × 全部模型，输出完整回复供人工打分（每题 5 分）。

评分维度（人工）
----------------
5分：内容正确 + 表达流畅 + 直接回答问题
4分：内容基本正确，有小瑕疵
3分：格式合格，但有明显事实错误或答非所问
2分：勉强能看，内容大量错误
1分：几乎无有效信息
0分：完全崩坏（乱码/复读/英文）
"""
import json
import os
import sys

# 必须先设环境变量再 import
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

QUESTIONS = [
    # 简单（5 题）
    ("简单", "你好"),
    ("简单", "你是谁？"),
    ("简单", "今天星期几？"),
    ("简单", "1+1等于几？"),
    ("简单", "说一句话。"),
    # 中等（6 题）
    ("中等", "今天天气怎么样？"),
    ("中等", "推荐一本好书。"),
    ("中等", "怎么学习编程？"),
    ("中等", "什么是人工智能？"),
    ("中等", "介绍一下北京。"),
    ("中等", "你怎么看待友谊？"),
    # 困难（10 题）
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


def gen_gguf(model_path, tag, n=2):
    """用 llama.cpp 生成（v1 / v2）"""
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
                r = llm(prompt, max_tokens=300, temperature=0.7, top_p=0.9,
                        repeat_penalty=1.15, stop=["<|im_end|>"])
                answers.append(r["choices"][0]["text"].strip())
            except Exception as e:
                answers.append(f"[失败 {e}]")
        out.append({"level": level, "q": q, "answers": answers})
        print(f"  {q[:20]:<22} {len(answers[0])}字")
    return out


def gen_hf(model_dir, tag, n=2):
    """用 transformers 生成（minimind-3）"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[{tag}] 加载 {model_dir}")
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=True)
    model.eval()

    out = []
    for level, q in QUESTIONS:
        # 用它的 chat template
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
                g = model.generate(**ids, max_new_tokens=300, do_sample=True,
                                   temperature=0.7, top_p=0.9,
                                   repetition_penalty=1.15,
                                   pad_token_id=tok.pad_token_id or tok.eos_token_id)
            ans = tok.decode(g[0][ids["input_ids"].shape[1]:],
                             skip_special_tokens=True).strip()
            answers.append(ans)
        out.append({"level": level, "q": q, "answers": answers})
        print(f"  {q[:20]:<22} {len(answers[0])}字")
    return out


def main():
    results = {}

    # v1 / v2（GGUF）
    for path, tag in (("out/tiny-zh-64m-f16.gguf", "v1_ShareGPT"),
                      ("out/tiny-zh-64m-v2-f16.gguf", "v2_minimind")):
        if not os.path.exists(path):
            print(f"[skip] {path} 不存在")
            continue
        try:
            results[tag] = gen_gguf(path, tag)
        except Exception as e:
            print(f"[FAIL] {tag}: {e}")

    # minimind-3（HF）
    mm3 = "out_base/minimind3"
    if os.path.exists(f"{mm3}/model.safetensors"):
        try:
            results["minimind3_原版"] = gen_hf(mm3, "minimind3")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[FAIL] minimind3: {e}")

    # 保存
    with open("logs/three_way.json", "w", encoding="utf-8") as f:
        json.dump({"questions": QUESTIONS, "results": results},
                  f, ensure_ascii=False, indent=2)

    # 生成可读报告
    with open("logs/three_way.md", "w", encoding="utf-8") as f:
        f.write("# 三方对比测试\n\n")
        f.write("| 模型 | 参数量 | 架构 |\n|---|---|---|\n")
        f.write("| v1 (ShareGPT SFT) | 64.82M | custom, 13层640维 |\n")
        f.write("| v2 (minimind SFT) | 64.82M | custom, 13层640维 |\n")
        f.write("| minimind-3 原版 | 63.91M | Qwen3, 8层768维 |\n\n")

        for level in ("简单", "中等", "困难"):
            f.write(f"\n## {level}\n\n")
            for lv, q in QUESTIONS:
                if lv != level:
                    continue
                f.write(f"### 问：{q}\n\n")
                for tag, rs in results.items():
                    item = next((r for r in rs if r["q"] == q), None)
                    if not item:
                        continue
                    a = item["answers"][0].replace("\n", " ")
                    f.write(f"**{tag}**（{len(item['answers'][0])}字）\n\n")
                    f.write(f"> {a}\n\n")
                f.write("评分：v1 ___ / v2 ___ / minimind3 ___ / 5\n\n---\n\n")

    print("\n结果: logs/three_way.json")
    print("报告: logs/three_way.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())