#!/usr/bin/env python3
"""新旧 64M 模型对比实测 —— 同一批 prompt，逐条比。

对比对象
--------
旧版 tiny-zh-64m-f16.gguf     ShareGPT SFT（今天失败版）
新版 tiny-zh-64m-v2-f16.gguf  minimind SFT（重训版）

判据（对应之前暴露的失败点）
---------------------------
1. 有没有编号列表 [1][2][3]
2. 简单问题是否只回几个字
3. 中文占比
4. 答案长度
5. 是否复读
"""
import json
import os
import re
import sys
from collections import Counter

import llama_cpp

PROMPTS = [
    "今天天气怎么样？",
    "你是谁？",
    "你好",
    "推荐一本好书。",
    "怎么学习编程？",
    "什么是人工智能？",
    "介绍一下北京。",
    "为什么天是蓝的？",
    "怎么才能早睡？",
    "说说你的爱好。",
    "解释一下什么是互联网。",
    "给我讲讲春天。",
    "如何保持健康？",
    "什么是友谊？",
    "介绍一下中国菜。",
]

RE_CJK = re.compile(r"[\u4e00-\u9fff]")
RE_NUMLINE = re.compile(
    r"^\s*(?:[\[\(【（]\s*\d{1,2}\s*[\]\)】）]|\d{1,2}\s*[\.、\)）]"
    r"|[一二三四五六七八九十]{1,2}\s*[\.、])")


def has_numlist(a):
    lines = [l.strip() for l in a.split("\n") if l.strip()]
    if len(lines) < 3:
        return False
    numlike = sum(1 for l in lines if RE_NUMLINE.match(l))
    return numlike / len(lines) > 0.3


def repeat_ratio(a):
    if len(a) < 8:
        return 0.0
    grams = [a[i:i + 4] for i in range(len(a) - 3)]
    if not grams:
        return 0.0
    c = Counter(grams)
    return c.most_common(1)[0][1] / len(grams)


def gen(model_path, prompts, tag):
    print(f"\n{'=' * 70}")
    print(f"模型：{tag}")
    print(f"{'=' * 70}")

    if not os.path.exists(model_path):
        print("[FATAL] 文件不存在")
        return None

    llm = llama_cpp.Llama(model_path=model_path, n_ctx=512,
                          n_threads=16, verbose=False)

    results = []
    for q in prompts:
        prompt = f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
        try:
            out = llm(prompt, max_tokens=200, temperature=0.7, top_p=0.9,
                      repeat_penalty=1.15, stop=["<|im_end|>"])
            ans = out["choices"][0]["text"].strip()
        except Exception as e:
            ans = f"[生成失败 {e}]"

        n = len(ans)
        zh = len(RE_CJK.findall(ans)) / max(n, 1)
        numlist = has_numlist(ans)
        rep = repeat_ratio(ans)
        too_short = n < 15

        results.append({"q": q, "a": ans, "len": n, "zh": round(zh, 2),
                        "numlist": numlist, "repeat": round(rep, 2),
                        "too_short": too_short})

        print(f"\n【问】{q}")
        print(f"【答】{ans[:300]}")
        flags = []
        if numlist:
            flags.append("编号列表❌")
        if too_short:
            flags.append("过短❌")
        if zh < 0.6:
            flags.append("中文不足❌")
        if rep > 0.3:
            flags.append("复读❌")
        print(f"  {n}字 中文{zh:.2f} {'  '.join(flags) if flags else 'OK'}")
    return results


def stats(rs):
    if not rs:
        return None
    return {
        "avg_len": sum(r["len"] for r in rs) / len(rs),
        "numlist": sum(1 for r in rs if r["numlist"]),
        "too_short": sum(1 for r in rs if r["too_short"]),
        "low_zh": sum(1 for r in rs if r["zh"] < 0.6),
        "repeat": sum(1 for r in rs if r["repeat"] > 0.3),
        "ok": sum(1 for r in rs if not r["numlist"] and not r["too_short"]
                  and r["zh"] >= 0.6 and r["repeat"] <= 0.3),
        "total": len(rs),
    }


def summarize(old, new):
    print(f"\n{'=' * 70}")
    print("对比汇总")
    print(f"{'=' * 70}")
    so, sn = stats(old), stats(new)
    if not so or not sn:
        print("数据不足")
        return None

    print(f"\n{'指标':<14} {'旧版(ShareGPT)':>16} {'新版(minimind)':>16} {'判定':>8}")
    print("-" * 60)

    def row(label, a, b, higher_better=False):
        if a == b:
            mark = "—"
        elif (b > a) == higher_better:
            mark = "✅"
        else:
            mark = "❌"
        print(f"{label:<14} {a:>16} {b:>16} {mark:>8}")

    row("平均字数", f"{so['avg_len']:.0f}", f"{sn['avg_len']:.0f}", True)
    row("编号列表", so["numlist"], sn["numlist"])
    row("答案过短", so["too_short"], sn["too_short"])
    row("中文不足", so["low_zh"], sn["low_zh"])
    row("复读", so["repeat"], sn["repeat"])
    row("合格", f"{so['ok']}/{so['total']}", f"{sn['ok']}/{sn['total']}", True)

    print()
    print(f"合格率：旧 {so['ok']/so['total']*100:.0f}%  ->  新 {sn['ok']/sn['total']*100:.0f}%")

    if sn["ok"] > so["ok"]:
        verdict = "新版明显更好"
    elif sn["ok"] == so["ok"]:
        verdict = "持平"
    else:
        verdict = "新版更差"
    print(f"结论：{verdict}")
    return verdict


def main():
    old = gen("out/tiny-zh-64m-f16.gguf", PROMPTS, "旧版（ShareGPT SFT）")
    new = gen("out/tiny-zh-64m-v2-f16.gguf", PROMPTS, "新版（minimind SFT）")
    v = summarize(old, new)

    with open("logs/compare_v1_v2.json", "w", encoding="utf-8") as f:
        json.dump({"prompts": PROMPTS, "old": old, "new": new, "verdict": v},
                  f, ensure_ascii=False, indent=2)
    print("\n结果写入 logs/compare_v1_v2.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())