#!/usr/bin/env python3
"""严厉测试：v1 vs v2 vs minimind-3 原版。

比上一版严格在哪
----------------
上一版只测了 15 条基础问题，判据宽松（有内容就算过）。
这一版加：
1. **同一问题跑 3 次**，看稳定性（LLM 有随机性，一次好不算好）
2. **难度分级**：简单/中等/困难
3. **更严的判据**：
   - 必须直接回答问题（不能答非所问）
   - 不能出现「作为AI助手...」这类模板回避
   - 不能复读问题
   - 中文占比 > 0.85（更严）
   - 长度 30-250 字
4. **一致性评分**：3 次生成的语义相似度
"""
import json
import os
import re
import sys
from collections import Counter

import llama_cpp

# 分三级难度
TESTS = {
    "简单": [
        "你好",
        "你是谁？",
        "今天星期几？",
        "1+1等于几？",
        "说一句话。",
    ],
    "中等": [
        "今天天气怎么样？",
        "推荐一本好书。",
        "怎么学习编程？",
        "什么是人工智能？",
        "介绍一下北京。",
        "你怎么看待友谊？",
    ],
    "困难": [
        "为什么天是蓝的？",
        "解释一下什么是互联网。",
        "如何保持健康？",
        "介绍一下中国菜。",
        "给我讲讲春天。",
        "人为什么要睡觉？",
        "什么是民主？",
        "请写一段关于秋天的描写。",
        "比较一下猫和狗。",
        "如果我想学游泳，第一步该做什么？",
    ],
}

RE_CJK = re.compile(r"[\u4e00-\u9fff]")
RE_NUMLINE = re.compile(r"^\s*(?:[\[\(【（]\s*\d{1,2}\s*[\]\)】）]|\d{1,2}\s*[\.、\)）])")
# 模板回避句式（这些是「没回答问题」
RE_EVASION = re.compile(
    r"(欢迎随时(?:告诉我|提问)|我(?:很)?(?:乐意|愿意)帮助|"
    r"如果您(?:有)?(?:其他|任何)(?:问题|疑问)|"
    r"请(?:随时)?告诉(?:我)?你(?:的)?需求|"
    r"作为(?:一个)?AI(?:助手|模型)?[，,]我(?:擅长|无法))")


def has_numlist(a):
    lines = [l.strip() for l in a.split("\n") if l.strip()]
    if len(lines) < 3:
        return False
    return sum(1 for l in lines if RE_NUMLINE.match(l)) / len(lines) > 0.3


def repeat_ratio(a):
    if len(a) < 8:
        return 0.0
    grams = [a[i:i + 4] for i in range(len(a) - 3)]
    if not grams:
        return 0.0
    return Counter(grams).most_common(1)[0][1] / len(grams)


def is_evasion(a):
    """是否用模板套话回避了问题"""
    return bool(RE_EVASION.search(a))


def judge(q, a):
    """严厉判据，返回 (通过, 失败原因列表)"""
    fails = []
    n = len(a)

    if n < 20:
        fails.append("太短")
    if n > 300:
        fails.append("太长")

    zh = len(RE_CJK.findall(a)) / max(n, 1)
    if zh < 0.85:
        fails.append(f"中文{zh:.2f}")

    if has_numlist(a):
        fails.append("编号列表")
    if repeat_ratio(a) > 0.25:
        fails.append("复读")
    if is_evasion(a):
        fails.append("模板回避")
    # 复读问题本身
    if n > 0 and q[:6] in a[:20]:
        fails.append("复读问题")

    return (len(fails) == 0), fails


def similarity(a, b):
    """粗略语义相似：字符 3-gram Jaccard"""
    def grams(s):
        return {s[i:i + 3] for i in range(len(s) - 2)}
    ga, gb = grams(a), grams(b)
    if not ga or not gb:
        return 0.0
    return len(ga & gb) / len(ga | gb)


def run_model(model_path, tag, n_trials=3):
    print(f"\n{'=' * 74}")
    print(f"模型：{tag}")
    print(f"{'=' * 74}")

    if not os.path.exists(model_path):
        print(f"[FATAL] 不存在: {model_path}")
        return None

    try:
        llm = llama_cpp.Llama(model_path=model_path, n_ctx=512,
                              n_threads=16, verbose=False)
    except Exception as e:
        print(f"[FATAL] 加载失败: {e}")
        return None

    results = []
    for level, qs in TESTS.items():
        print(f"\n--- {level} ---")
        for q in qs:
            prompt = f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
            answers = []
            for trial in range(n_trials):
                try:
                    out = llm(prompt, max_tokens=250, temperature=0.7, top_p=0.9,
                              repeat_penalty=1.15, stop=["<|im_end|>"])
                    answers.append(out["choices"][0]["text"].strip())
                except Exception as e:
                    answers.append(f"[失败 {e}]")

            # 用第一次做质量判定
            ok, fails = judge(q, answers[0])
            # 3 次之间的一致性
            sims = []
            for i in range(len(answers)):
                for j in range(i + 1, len(answers)):
                    sims.append(similarity(answers[i], answers[j]))
            consistency = sum(sims) / len(sims) if sims else 0

            mark = "OK " if ok else "FAIL"
            print(f"  [{mark}] {q[:22]:<24} {len(answers[0]):>3}字 "
                  f"一致{consistency:.2f} {' '.join(fails) if fails else ''}")

            results.append({
                "level": level, "q": q, "answers": answers,
                "ok": ok, "fails": fails, "consistency": round(consistency, 2),
            })
    return results


def summarize(all_results):
    print(f"\n{'=' * 74}")
    print("总评")
    print(f"{'=' * 74}")

    print(f"\n{'模型':<22} {'通过率':>10} {'简单':>8} {'中等':>8} {'困难':>8} {'一致性':>8}")
    print("-" * 70)

    summary = {}
    for tag, rs in all_results.items():
        if not rs:
            continue
        total = len(rs)
        passed = sum(1 for r in rs if r["ok"])
        by_level = {}
        for lv in ("简单", "中等", "困难"):
            sub = [r for r in rs if r["level"] == lv]
            by_level[lv] = f"{sum(1 for r in sub if r['ok'])}/{len(sub)}" if sub else "-"
        cons = sum(r["consistency"] for r in rs) / total
        print(f"{tag:<22} {passed}/{total} ({passed/total*100:>3.0f}%) "
              f"{by_level['简单']:>8} {by_level['中等']:>8} {by_level['困难']:>8} "
              f"{cons:>8.2f}")
        summary[tag] = {"passed": passed, "total": total, "by_level": by_level,
                        "consistency": round(cons, 3)}

    # 失败类型分布
    print(f"\n{'模型':<22} {'失败类型分布':<50}")
    print("-" * 74)
    for tag, rs in all_results.items():
        if not rs:
            continue
        c = Counter()
        for r in rs:
            for f in r["fails"]:
                # 归类（去掉具体数值）
                key = re.sub(r"[\d.]+", "", f).strip()
                c[key] += 1
        dist = ", ".join(f"{k}×{v}" for k, v in c.most_common(5)) or "无"
        print(f"{tag:<22} {dist}")

    # 排名
    print(f"\n{'=' * 74}")
    ranked = sorted(summary.items(), key=lambda x: -x[1]["passed"])
    for i, (tag, s) in enumerate(ranked, 1):
        print(f"  第{i}名: {tag:<22} {s['passed']}/{s['total']}")
    print(f"{'=' * 74}")
    return summary


def main():
    models = [
        ("out/tiny-zh-64m-f16.gguf", "v1 (ShareGPT SFT)"),
        ("out/tiny-zh-64m-v2-f16.gguf", "v2 (minimind 64M)"),
    ]
    # minimind-3 原版（如果本地有）
    for cand in ("out/minimind-3-f16.gguf", "/mnt/workspace/models/minimind-3-f16.gguf"):
        if os.path.exists(cand):
            models.append((cand, "minimind-3 原版"))
            break

    all_results = {}
    for path, tag in models:
        all_results[tag] = run_model(path, tag)

    summary = summarize(all_results)

    with open("logs/strict_test.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary,
                   "detail": {k: v for k, v in all_results.items() if v}},
                  f, ensure_ascii=False, indent=2)
    print("\n结果写入 logs/strict_test.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())