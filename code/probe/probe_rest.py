"""补测 SkyPile 和 COIG-PC（修正路径）。"""
import json
import os
import re
import subprocess

PROBE = "data_probe"
RE_CJK = re.compile(r"[\u4e00-\u9fff]")
RE_SP = re.compile(r"(?<=[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef])\s+(?=[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef])")
N = 25_000_000

TARGETS = [
    ("SkyPile-150B",
     "https://hf-mirror.com/datasets/Skywork/SkyPile-150B/resolve/main/data/2020-40_zh_head_0000.jsonl",
     "jsonl"),
    ("COIG-PC",
     "https://hf-mirror.com/datasets/BAAI/COIG-PC/resolve/main/data/Top50PerTask-00000-of-00001-aaeff287493ab78e.parquet",
     "parquet"),
]


def fetch(url, out, n):
    if os.path.exists(out) and os.path.getsize(out) > n * 0.7:
        print(f"  [cached] {os.path.getsize(out)/1e6:.1f}MB")
        return True
    try:
        r = subprocess.run(["curl", "-fsSL", "--connect-timeout", "15", "--max-time", "240",
                            "--range", f"0-{n-1}", "-o", out, url],
                           capture_output=True, text=True, timeout=260)
        sz = os.path.getsize(out) if os.path.exists(out) else 0
        if sz > 1024:
            print(f"  [got] {sz/1e6:.1f} MB")
            return True
        print(f"  [FAIL] curl: {(r.stderr or '')[:120]}")
    except Exception as e:
        print(f"  [FAIL] {str(e)[:80]}")
    return False


def texts_jsonl(path, limit=20000):
    with open(path, "rb") as f:
        d = f.read()
    i = d.rfind(b"\n")
    if i > 0:
        d = d[:i]
    out = []
    for line in d.decode("utf-8", errors="ignore").splitlines():
        if len(out) >= limit:
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
                break
        if s and len(s) >= 30:
            out.append(s)
    return out


def texts_parquet(path, limit=20000):
    import pyarrow.parquet as pq
    try:
        f = pq.ParquetFile(path)
    except Exception as e:
        print(f"  [parquet 打不开] {str(e)[:90]}")
        print(f"  文件大小 {os.path.getsize(path)} 字节")
        with open(path, "rb") as fh:
            print(f"  前 4 字节: {fh.read(4)}")
        return []
    cols = f.schema_arrow.names
    print(f"  列: {cols}  rows={f.metadata.num_rows:,}")
    out = []
    t = f.read_row_group(0)
    tcol = None
    for c in ("text", "content", "raw_content"):
        if c in cols:
            tcol = c
            break
    if tcol:
        for v in t.column(tcol).to_pylist():
            if v and len(v) >= 30:
                out.append(v)
                if len(out) >= limit:
                    break
    return out


def analyze(tag, texts):
    if not texts:
        print("  [无有效文本]")
        return None
    c = [RE_SP.sub("", t) for t in texts]
    lens = sorted(len(t) for t in c)
    zhs = sorted(len(RE_CJK.findall(t))/len(t) for t in c)
    n = len(c)
    ok = sum(1 for t in c if len(RE_CJK.findall(t))/len(t) >= 0.65 and 100 <= len(t) <= 2000)
    st = {"n": n, "avg_len": sum(lens)/n,
          "p50": lens[n//2], "p90": lens[int(n*0.9)],
          "avg_zh": sum(zhs)/n, "p10_zh": zhs[n//10], "ok": ok/n}
    print(f"  样本 {n:,}  平均 {st['avg_len']:.0f} 字  p50={st['p50']} p90={st['p90']}")
    print(f"  中文 {st['avg_zh']:.2f} (p10={st['p10_zh']:.2f})  合格率 {st['ok']*100:.0f}%")
    v = "✅可用" if st["avg_zh"] >= 0.65 and st["ok"] >= 0.5 else (
        "⚠️勉强" if st["avg_zh"] >= 0.5 else "❌不合格")
    print(f"  判定 {v}")
    st["verdict"] = v
    return st


res = {}
for tag, url, kind in TARGETS:
    print(f"\n--- {tag} ---")
    out = os.path.join(PROBE, f"cand2_{tag}.{kind}")
    if not fetch(url, out, N):
        continue
    texts = texts_parquet(out) if kind == "parquet" else texts_jsonl(out)
    st = analyze(tag, texts)
    if st:
        res[tag] = st

print("\n" + "=" * 70)
for tag, s in res.items():
    print(f"{tag}: {s['avg_len']:.0f}字 中文{s['avg_zh']:.2f} 合格{s['ok']*100:.0f}% {s['verdict']}")