"""用魔搭直链测 CCI3-HQ 和 SkyPile 片段（hf-mirror 的 Range 可能不支持）。"""
import json
import os
import re
import subprocess

PROBE = "data_probe"
RE_CJK = re.compile(r"[\u4e00-\u9fff]")
RE_SP = re.compile(r"(?<=[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef])\s+(?=[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef])")

TARGETS = [
    ("cci3_hq_ms",
     "https://www.modelscope.cn/api/v1/datasets/BAAI/CCI3-HQ/repo?Revision=master&FilePath=data%2Fpart_000000.jsonl",
     30_000_000),
]


def fetch(url, out, n):
    if os.path.exists(out) and os.path.getsize(out) > n * 0.8:
        print(f"  [cached] {os.path.basename(out)}")
        return True
    try:
        subprocess.run(["curl", "-fsSL", "--connect-timeout", "15", "--max-time", "300",
                        "--range", f"0-{n-1}", "-o", out, url],
                       capture_output=True, timeout=320)
        if os.path.exists(out) and os.path.getsize(out) > 1024:
            print(f"  [got] {os.path.basename(out)} {os.path.getsize(out)/1e6:.1f} MB")
            return True
    except Exception as e:
        print(f"  [FAIL] {str(e)[:80]}")
    print("  [FAIL]")
    return False


def stat(path, limit=20000):
    with open(path, "rb") as f:
        data = f.read()
    i = data.rfind(b"\n")
    if i > 0:
        data = data[:i]
    texts = []
    keys_used = {}
    for line in data.decode("utf-8", errors="ignore").splitlines():
        if len(texts) >= limit:
            break
        line = line.strip()
        if not line:
            continue
        try:
            o = json.loads(line)
        except json.JSONDecodeError:
            continue
        s = None
        for k in ("text", "content", "completion", "raw_content", "body"):
            v = o.get(k)
            if isinstance(v, str) and v.strip():
                s = v
                keys_used[k] = keys_used.get(k, 0) + 1
                break
        if not s:
            continue
        s = RE_SP.sub("", s)
        if len(s) >= 30:
            texts.append(s)
    if not texts:
        # 打印一条原始记录帮诊断
        for line in data.decode("utf-8", errors="ignore").splitlines()[:1]:
            print(f"  [诊断] 首行: {line[:200]}")
        return None
    lens = sorted(len(t) for t in texts)
    zhs = sorted(len(RE_CJK.findall(t)) / len(t) for t in texts)
    n = len(texts)
    ok = sum(1 for t in texts if len(RE_CJK.findall(t))/len(t) >= 0.65 and 100 <= len(t) <= 2000)
    print(f"  字段: {keys_used}")
    return {"n": n, "avg_len": sum(lens)/n, "p50": lens[n//2],
            "avg_zh": sum(zhs)/n, "p10_zh": zhs[n//10], "ok": ok/n}


for tag, url, n in TARGETS:
    print(f"\n--- {tag} ---")
    out = os.path.join(PROBE, f"{tag}.jsonl")
    if not fetch(url, out, n):
        continue
    st = stat(out)
    if st:
        print(f"  样本 {st['n']:,} 平均 {st['avg_len']:.0f} 字 p50={st['p50']}")
        print(f"  中文 {st['avg_zh']:.2f} (p10={st['p10_zh']:.2f}) 合格率 {st['ok']*100:.0f}%")
        v = "✅可用" if st["avg_zh"] >= 0.65 and st["ok"] >= 0.5 else "❌不合格"
        print(f"  {v}")