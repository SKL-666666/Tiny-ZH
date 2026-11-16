#!/usr/bin/env python3
"""在本机用 llama-cli 测 Tiny-ZH-31.5M，同样的 21 题。

为什么用 subprocess 调 llama-cli 而不是 llama_cpp
--------------------------------------------------
本机没装 llama-cpp-python 的 GPU 版，而 llama-cli.exe 是现成的，
CPU 跑 31.5M 模型够了（66MB）。
"""
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LLAMA = os.path.join(HERE, "tools", "llama-cli.exe")
MODEL = os.path.join(HERE, "models", "tiny-zh-f16.gguf")

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


def gen(q, max_tokens=200):
    """调 llama-cli 生成。返回纯回复文本。"""
    cmd = [
        LLAMA, "-m", MODEL,
        "-p", f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n",
        "-n", str(max_tokens),
        "--temp", "0.7",
        "--top-p", "0.9",
        "--repeat-penalty", "1.15",
        "-st",              # 单轮，不进交互
        "--no-warmup",
        "-no-cnv" if False else "-st",
        "--simple-io",
    ]
    # 去掉重复的 -st
    cmd = [c for i, c in enumerate(cmd) if not (c == "-st" and i > cmd.index("-st"))]

    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=180)
    except subprocess.TimeoutExpired:
        return "[超时]"

    out = (r.stdout or "") + (r.stderr or "")

    # llama-cli 输出格式：先打印 prompt，再打印生成内容，最后一行是统计
    # 最可靠的切法：找 assistant\n 之后的内容
    marker = "<|im_start|>assistant"
    if marker in out:
        body = out.split(marker, 1)[1]
        # 去掉开头的换行
        body = body.lstrip("\n")
    else:
        body = out

    # 去掉结尾的统计行和交互提示
    lines = []
    for line in body.splitlines():
        if re.search(r"\[ Prompt:|llama_perf|^>|Exiting|available commands|^/", line):
            break
        if line.strip().startswith("/exit") or "Ctrl+C" in line:
            break
        lines.append(line)
    text = "\n".join(lines).strip()

    # 去掉可能的特殊 token 残留
    text = text.replace("<|im_end|>", "").replace("<|im_start|>", "").strip()
    return text


def main():
    if not os.path.exists(LLAMA):
        print(f"[FATAL] 缺 {LLAMA}")
        return 1
    if not os.path.exists(MODEL):
        print(f"[FATAL] 缺 {MODEL}")
        return 1

    print(f"模型 {MODEL}")
    print(f"提示: 31.5M 模型，CPU 运行，每题约 5-15 秒\n")

    results = []
    for level, q in QUESTIONS:
        ans = gen(q)
        results.append({"level": level, "q": q, "answer": ans})
        print(f"[{level}] {q[:20]:<22} {len(ans):>3}字")
        print(f"  {ans[:120].replace(chr(10), ' ')}")
        print()

    with open("logs/tiny32_test.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 生成报告
    lines = ["# Tiny-ZH-31.5M 测试（同样 21 题）\n"]
    for level in ("简单", "中等", "困难"):
        lines.append(f"\n## {level}\n")
        for r in results:
            if r["level"] != level:
                continue
            lines.append(f"### 问：{r['q']}\n")
            lines.append(f"**Tiny-ZH-31.5M**（{len(r['answer'])}字）")
            lines.append(f"> {r['answer'].replace(chr(10), ' ')}\n")
            lines.append("| 模型 | 分数 /5 |")
            lines.append("|---|---|")
            lines.append("| 31.5M | |")
            lines.append("\n---\n")

    with open("logs/tiny32_test.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("结果: logs/tiny32_test.json")
    print("报告: logs/tiny32_test.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())