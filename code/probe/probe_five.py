"""实测用户指定的五个数据集，各取最小片段。"""
import json
import os
import re
import subprocess

PROBE = "data_probe"
RE_CJK = re.compile(r"[\u4e00-\u9fff]")
RE_SP = re.compile(r"(?<=[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef])\s+(?=[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef])")

MS = "https://www.modelscope.cn/api/v1/datasets/{}/repo?Revision=master&FilePath={}"

TARGETS = [
    # (标签, URL, 大小, 类型)
    ("CCI3-HQ",
     MS.format("BAAI/CCI3-HQ", "data%2Fpart_000001.jsonl"), 25_000_000, "jsonl"),
    ("SkyPile-150B",
     "https://hf-mirror.com/datasets/Skywork/SkyPile-150B/resolve/main/skypile-150b-00000-of-00441.jsonl",
     25_000_000, "jsonl"),
    ("COIG-PC",
     "https://hf-mirror.com/datasets/BAAI/COIG-PC/resolve/main/data/Top50PerTask-00000-of-00001-aaeff287493ab78e.parquet",
     25_000_000, "parquet"),
    ("FineWeb-Edu-v2",
     MS.format("opencsg/chinese-fineweb-edu-v2", "data%2F00000.parquet"), 25_000_000, "parquet"),
    ("COIG-CQIA",
     MS.format("AI-ModelScope/COIG-CQIA", "COIG-CQIA-full.jsonl"), 25_000_000, "jsonl"),
]


def fetch(url, out, n):
    if os.path.exists(out) and os.path.getsize(out) > n * 0.7:
        print(f"  [cached] {os.path.basename(out)} {os.path.getsize(out)/1e6:.1f}MB")
        return True
    try:
        subprocess.run(["curl", "-fsSL", "--connect-timeout", "15", "--max-time", "240",
                        "--range", f"0-{n-1}", "-o", out, url],
                       capture_output=True, timeout=260)
        sz = os.path.getsize(out) if os.path.exists(out) else 0
        if sz > 1024:
            print(f"  [got] {os.path.basename(out)} {sz/1e6:.1f} MB")
            return True
    except Exception as e:
        print(f"  [FAIL] {str(e)[:70]}")
    print("  [FAIL] 空或不存在")
    return False


def texts_from_jsonl(path, limit=20000):
    with open(path, "rb") as f:
        data = f.read()
    i = data.rfind(b"\n")
    if i > 0:
        data = data[:i]
    out = []
    for line in data.decode("utf-8", errors="ignore").splitlines():
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
        for k in ("text", "content", "completion", "raw_content", "body", "output"):
            v = o.get(k)
            if isinstance(v, str) and v.strip():
                s = v
                break
        if s is None:
            conv = o.get("conversations") or o.get("dialog") or []
            if isinstance(conv, list):
                parts = []
                for m in conv:
                    if isinstance(m, dict):
                        parts.append(str(m.get("content") or m.get("value") or ""))
                    elif isinstance(m, str):
                        parts.append(m)
                s = "\n".join(parts) if parts else None
        if s and len(s) >= 30:
            out.append(s)
    return out


def texts_from_parquet(path, limit=20000):
    import pyarrow.parquet as pq
    try:
        f = pq.ParquetFile(path)
    except Exception as e:
        print(f"  [parquet 打不开] {str(e)[:80]}")
        return []
    cols = f.schema_arrow.names
    print(f"  列: {cols}  rows={f.metadata.num_rows:,}")
    # 分块读取（文件可能是片段）
    out = []
    try:
        t = f.read_row_group(0)
    except Exception:
        return []
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
    else:
        # 对话格式
        for c in ("conversations", "messages", "conversation"):
            if c in cols:
                for conv in t.column(c).to_pylist():
                    if isinstance(conv, list):
                        parts = []
                        for m in conv:
                            if isinstance(m, dict):
                                parts.append(str(m.get("content") or m.get("value") or ""))
                        s = "\n".join(p for p in parts if p)
                        if len(s) >= 30:
                            out.append(s)
                            if len(out) >= limit:
                                break
                break
    return out


def analyze(tag, texts):
    if not texts:
        print(f"  [无有效文本]")
        return None
    cleaned = [RE_SP.sub("", t) for t in texts]
    lens = sorted(len(t) for t in cleaned)
    zhs = sorted(len(RE_CJK.findall(t)) / len(t) for t in cleaned)
    n = len(cleaned)
    def pct(a, p):
        return a[min(int(n * p), n - 1)]
    ok = sum(1 for t in cleaned
             if len(RE_CJK.findall(t))/len(t) >= 0.65 and 100 <= len(t) <= 2000)
    st = {
        "n": n,
        "avg_len": sum(lens) / n,
        "p10": pct(lens, 0.10), "p50": pct(lens, 0.50),
        "p90": pct(lens, 0.90), "p99": pct(lens, 0.99), "max": lens[-1],
        "avg_zh": sum(zhs) / n, "p10_zh": pct(zhs, 0.10),
        "ok": ok / n,
    }
    print(f"  样本 {n:,}")
    print(f"  长度: 平均 {st['avg_len']:.0f}  p10={st['p10']} p50={st['p50']} "
          f"p90={st['p90']} p99={st['p99']} max={st['max']}")
    print(f"  中文: 平均 {st['avg_zh']:.2f}  p10={st['p10_zh']:.2f}")
    print(f"  合格率(中≥0.65 且 100-2000字): {st['ok']*100:.0f}%")
    if st["avg_zh"] >= 0.65 and st["ok"] >= 0.5:
        v = "✅ 可用"
    elif st["avg_zh"] >= 0.5:
        v = "⚠️ 勉强"
    else:
        v = "❌ 不合格"
    print(f"  判定 {v}")
    st["verdict"] = v
    return st


def main():
    print("=" * 76)
    print("五个候选数据集片段实测")
    print("=" * 76)
    results = {}
    for tag, url, n, kind in TARGETS:
        print(f"\n--- {tag} ---")
        out = os.path.join(PROBE, f"cand_{tag}.{kind}")
        if not fetch(url, out, n):
            continue
        texts = texts_from_parquet(out) if kind == "parquet" else texts_from_jsonl(out)
        st = analyze(tag, texts)
        if st:
            results[tag] = st

    print("\n" + "=" * 76)
    print("汇总")
    print("=" * 76)
    print(f"{'数据集':<20}{'平均字':>8}{'p50':>7}{'p90':>7}{'中占比':>8}{'合格率':>8}{'判定':>10}")
    print("-" * 76)
    for tag, s in results.items():
        print(f"{tag:<20}{s['avg_len']:>8.0f}{s['p50']:>7}{s['p90']:>7}"
              f"{s['avg_zh']:>8.2f}{s['ok']*100:>7.0f}%{s['verdict']:>10}")

    with open(os.path.join(PROBE, "cand_result.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())