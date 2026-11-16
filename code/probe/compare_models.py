"""实测对比 minimind-3 vs Tiny-ZH-32M。

minimind-3：transformers 加载（qwen3 架构）
Tiny-ZH-32M：llama.cpp 跑 GGUF
同一批 prompt，同一套采样参数
"""
import json
import os
import re
import statistics
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MM_DIR = os.path.join(HERE, "models", "minimind-3")
TZ_GGUF = os.path.join(HERE, "models", "model.gguf")
LLAMA_CLI = os.path.join(HERE, "tools", "llama-cli.exe")

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


# ---------------- minimind-3 ----------------
def gen_minimind():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("[minimind-3] 加载中...", flush=True)
    tok = AutoTokenizer.from_pretrained(MM_DIR, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MM_DIR, dtype=torch.float32, trust_remote_code=True)
    model.eval()

    results = []
    for i, p in enumerate(PROMPTS, 1):
        msgs = [{"role": "user", "content": p}]
        try:
            text = tok.apply_chat_template(msgs, tokenize=False,
                                           add_generation_prompt=True)
        except Exception:
            text = f"<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n"
        ids = tok(text, return_tensors="pt")
        # qwen3 不接受 token_type_ids，只留 input_ids + attention_mask
        model_inputs = {"input_ids": ids["input_ids"]}
        if "attention_mask" in ids:
            model_inputs["attention_mask"] = ids["attention_mask"]
        with torch.no_grad():
            out = model.generate(**model_inputs, max_new_tokens=120, do_sample=True,
                                 temperature=0.7, top_p=0.9, top_k=40,
                                 repetition_penalty=1.25,
                                 pad_token_id=tok.pad_token_id or 0)
        resp = tok.decode(out[0][ids["input_ids"].shape[1]:],
                          skip_special_tokens=True).strip()
        results.append((p, resp))
        print(f"  [{i}/{len(PROMPTS)}]", flush=True)
    return results


# ---------------- Tiny-ZH-32M ----------------
def gen_tinyzh():
    print("[Tiny-ZH-32M] llama.cpp 生成中...", flush=True)
    results = []
    for i, p in enumerate(PROMPTS, 1):
        prompt = f"<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n"
        r = subprocess.run(
            [LLAMA_CLI, "-m", TZ_GGUF, "-p", prompt, "-n", "120",
             "--temp", "0.7", "--top-k", "40", "--top-p", "0.9",
             "--repeat-penalty", "1.25", "-ngl", "0", "--no-display-prompt",
             "--no-warmup", "-st", "--log-disable"],
            capture_output=True, text=True, encoding="utf-8", errors="replace")
        out = (r.stdout or "") + (r.stderr or "")
        # 清掉 banner 和统计行
        lines = []
        for line in out.splitlines():
            s = line.strip()
            if not s:
                continue
            if any(k in s for k in ("Loading", "llama_", "ggml_", "Prompt:",
                                    "Generation:", "Exiting", "build",
                                    "model      :", "ftype", "modalities",
                                    "system_info", "available commands",
                                    "▄", "██", "/exit", "/regen",
                                    "/clear", "/read", "/glob")):
                continue
            lines.append(s)
        resp = " ".join(lines).strip()
        for stop in ("<|im_end|>", "<|im_start|>"):
            if stop in resp:
                resp = resp.split(stop)[0]
        results.append((p, resp.strip()))
        print(f"  [{i}/{len(PROMPTS)}]", flush=True)
    return results


def summarize(name, results):
    rows = [stats(t) for _, t in results]
    out = {
        "model": name,
        "n": len(rows),
        "avg_len": statistics.mean(r["n"] for r in rows),
        "avg_zh": statistics.mean(r["zh"] for r in rows),
        "avg_rep": statistics.mean(r["rep"] for r in rows),
        "total_bad": sum(r["bad"] for r in rows),
        "en_samples": sum(1 for r in rows if r["en"] > 10),
        "fmt_ok": sum(r["fmt"] for r in rows),
        "empty": sum(1 for r in rows if r["n"] < 10),
    }
    return out


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "both"

    all_results = {}
    if which in ("mm", "both"):
        try:
            all_results["minimind-3"] = gen_minimind()
        except Exception as e:
            print(f"[minimind-3] FAILED: {type(e).__name__}: {e}")
    if which in ("tz", "both"):
        all_results["Tiny-ZH-32M"] = gen_tinyzh()

    lines = []
    for name, res in all_results.items():
        s = summarize(name, res)
        lines.append("=" * 70)
        lines.append(f"{name}")
        lines.append("=" * 70)
        lines.append(f"  样本数        {s['n']}")
        lines.append(f"  平均长度      {s['avg_len']:.0f} 字")
        lines.append(f"  中文占比      {s['avg_zh']:.3f}")
        lines.append(f"  4-gram 复读   {s['avg_rep']:.3f}")
        lines.append(f"  乱码总数      {s['total_bad']}")
        lines.append(f"  英文样本数    {s['en_samples']}")
        lines.append(f"  格式正确数    {s['fmt_ok']}")
        lines.append(f"  空输出        {s['empty']}")
        lines.append("")
        for p, t in res:
            lines.append(f"  问: {p}")
            lines.append(f"  答: {t[:180]}")
            lines.append("")

    text = "\n".join(lines)
    with open("cmp_result.txt", "w", encoding="utf-8") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
