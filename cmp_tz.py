"""通过 llama-server API 测 Tiny-ZH-32M，避免 CLI 输出污染。"""
import json
import os
import re
import statistics
import urllib.request

URL = "http://127.0.0.1:8080/v1/chat/completions"

PROMPTS = [
    "什么是人工智能？",
    "推荐几本中文小说。",
    "如何摆脱拖延症？",
    "请描述一下秋天的景色。",
    "我今天心情不太好，怎么办？",
    "怎么才能早起？",
    "为什么要多读书？",
    "解释一下什么是互联网。",
    "如何做一道番茄炒蛋？",
    "养猫需要注意什么？",
    "怎样才能让自己更专注？",
    "介绍一下中国的传统节日。",
    "学习编程应该从哪里开始？",
    "什么是幸福？",
    "如何提高写作水平？",
]


def stats(text):
    if not text:
        return dict(zh=0.0, en=0, bad=0, rep=0.0, n=0, fmt=0)
    zh = sum(1 for c in text if "\u4e00" <= c <= "\u9fff") / max(len(text), 1)
    en = sum(1 for c in text if c.isascii() and c.isalpha())
    bad = text.count("\ufffd")
    t = "".join(text.split())
    g = [t[i:i + 4] for i in range(max(len(t) - 3, 0))]
    rep = 1 - len(set(g)) / len(g) if g else 0.0
    fmt = 1 if re.search(r"(1[\.、]|\*\*)", text) else 0
    return dict(zh=zh, en=en, bad=bad, rep=rep, n=len(text), fmt=fmt)


def ask(p):
    body = {
        "messages": [{"role": "user", "content": p}],
        "max_tokens": 120,
        "temperature": 0.7,
        "top_p": 0.9,
        "repeat_penalty": 1.25,
    }
    req = urllib.request.Request(
        URL, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        d = json.loads(r.read())
    return d["choices"][0]["message"]["content"].strip()


results = []
for i, p in enumerate(PROMPTS, 1):
    try:
        t = ask(p)
    except Exception as e:
        t = f"<ERR {e}>"
    results.append((p, t))
    print(f"[{i}/{len(PROMPTS)}] {p[:16]} -> {len(t)}字", flush=True)

rows = [stats(t) for _, t in results]
s = dict(
    n=len(rows),
    avg_len=statistics.mean(r["n"] for r in rows),
    avg_zh=statistics.mean(r["zh"] for r in rows),
    avg_rep=statistics.mean(r["rep"] for r in rows),
    total_bad=sum(r["bad"] for r in rows),
    en_samples=sum(1 for r in rows if r["en"] > 10),
    fmt_ok=sum(r["fmt"] for r in rows),
    empty=sum(1 for r in rows if r["n"] < 10),
)

lines = ["=" * 70, "Tiny-ZH-32M（llama-server API）", "=" * 70]
lines += [
    f"  样本数        {s['n']}",
    f"  平均长度      {s['avg_len']:.0f} 字",
    f"  中文占比      {s['avg_zh']:.3f}",
    f"  4-gram 复读   {s['avg_rep']:.3f}",
    f"  乱码总数      {s['total_bad']}",
    f"  英文样本数    {s['en_samples']}",
    f"  格式正确数    {s['fmt_ok']}",
    f"  空输出        {s['empty']}",
    "",
]
for p, t in results:
    lines += [f"  问: {p}", f"  答: {t[:200]}", ""]

text = "\n".join(lines)
with open("cmp_tz_api.txt", "w", encoding="utf-8") as f:
    f.write(text)
print(text)
