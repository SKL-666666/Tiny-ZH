"""用 Range 请求只取每个候选数据集的最小片段，测质量。

不下载整个文件，每个源最多取 30MB。
"""
import json
import os
import subprocess
import sys

PROBE = "data_probe"
os.makedirs(PROBE, exist_ok=True)

# (标签, URL, 取多少字节, 文件类型)
TARGETS = [
    # minimind 完整版预训练（对比 mini 版）
    ("pretrain_t2t_full",
     "https://www.modelscope.cn/api/v1/datasets/gongjy/minimind_dataset/repo?Revision=master&FilePath=pretrain_t2t.jsonl",
     30_000_000, "jsonl"),

    # BAAI CCI3-HQ（高质量中文）
    ("cci3_hq",
     "https://hf-mirror.com/datasets/BAAI/CCI3-HQ/resolve/main/data/part_000000.jsonl",
     30_000_000, "jsonl"),

    # Skywork SkyPile
    ("skypile",
     "https://hf-mirror.com/datasets/Skywork/SkyPile-150B/resolve/main/data/skypile-150b-00000.jsonl",
     30_000_000, "jsonl"),

    # BAAI COIG-PC
    ("coig_pc",
     "https://hf-mirror.com/datasets/BAAI/COIG-PC/resolve/main/data/part_000000.jsonl",
     30_000_000, "jsonl"),
]


def fetch_range(url, out, nbytes):
    """只取前 nbytes 字节。"""
    if os.path.exists(out) and os.path.getsize(out) >= nbytes * 0.8:
        print(f"  [cached] {os.path.basename(out)}")
        return True
    try:
        r = subprocess.run(
            ["curl", "-fsSL", "--connect-timeout", "15", "--max-time", "300",
             "--range", f"0-{nbytes-1}", "-o", out, url],
            capture_output=True, text=True, timeout=320)
        if os.path.exists(out) and os.path.getsize(out) > 1024:
            print(f"  [got] {os.path.basename(out)} {os.path.getsize(out)/1e6:.1f} MB")
            return True
        print(f"  [FAIL] 空文件")
        return False
    except Exception as e:
        print(f"  [FAIL] {str(e)[:100]}")
        return False


def last_complete_line(path):
    """截断到最后一个完整行，避免半行 JSON 解析失败。"""
    with open(path, "rb") as f:
        data = f.read()
    idx = data.rfind(b"\n")
    if idx > 0:
        data = data[:idx]
    return data.decode("utf-8", errors="ignore")


def stat_jsonl(path, limit=20000):
    """统计 jsonl 文本质量。"""
    import re
    RE_CJK = re.compile(r"[\u4e00-\u9fff]")
    RE_SP = re.compile(r"(?<=[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef])\s+(?=[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef])")

    text = last_complete_line(path)
    texts = []
    for line in text.splitlines():
        if len(texts) >= limit:
            break
        line = line.strip()
        if not line:
            continue
        try:
            o = json.loads(line)
        except json.JSONDecodeError:
            continue
        # 尝试各种字段
        s = None
        for k in ("text", "content", "completion", "raw_content"):
            v = o.get(k)
            if isinstance(v, str) and v.strip():
                s = v
                break
        if s is None:
            # 对话格式
            conv = o.get("conversations") or o.get("dialog") or []
            if isinstance(conv, list):
                parts = []
                for m in conv:
                    if isinstance(m, dict):
                        parts.append(str(m.get("content") or m.get("value") or ""))
                    elif isinstance(m, str):
                        parts.append(m)
                s = "\n".join(parts) if parts else None
        if not s:
            continue
        s = RE_SP.sub("", s)
        if len(s) >= 30:
            texts.append(s)

    if not texts:
        return None

    lens = sorted(len(t) for t in texts)
    zhs = sorted(len(RE_CJK.findall(t)) / len(t) for t in texts)
    n = len(texts)
    ok = sum(1 for t in texts
             if len(RE_CJK.findall(t)) / len(t) >= 0.65 and 100 <= len(t) <= 2000)
    return {
        "n": n,
        "avg_len": sum(lens) / n,
        "p50": lens[n // 2],
        "avg_zh": sum(zhs) / n,
        "p10_zh": zhs[n // 10],
        "ok_rate": ok / n,
    }


def main():
    print("=" * 74)
    print("候选数据集片段实测（每源最多 30MB）")
    print("=" * 74)

    results = {}
    for tag, url, nbytes, kind in TARGETS:
        print(f"\n--- {tag} ---")
        out = os.path.join(PROBE, f"{tag}.jsonl")
        if not fetch_range(url, out, nbytes):
            continue
        st = stat_jsonl(out)
        if not st:
            print(f"  [解析失败] 没有可用文本")
            continue
        results[tag] = st
        print(f"  样本 {st['n']:,}  平均 {st['avg_len']:.0f} 字  p50={st['p50']}")
        print(f"  中文占比 平均 {st['avg_zh']:.2f}  p10={st['p10_zh']:.2f}")
        print(f"  合格率（中≥0.65 且 100-2000字）{st['ok_rate']*100:.0f}%")
        verdict = "✅ 可用" if st["avg_zh"] >= 0.65 and st["ok_rate"] >= 0.5 else (
            "⚠️ 勉强" if st["avg_zh"] >= 0.5 else "❌ 不合格")
        print(f"  判定 {verdict}")

    print("\n" + "=" * 74)
    print("汇总")
    print("=" * 74)
    print(f"{'数据集':<22}{'样本':>8}{'平均字':>8}{'中占比':>8}{'合格率':>8}")
    print("-" * 74)
    for tag, s in results.items():
        print(f"{tag:<22}{s['n']:>8,}{s['avg_len']:>8.0f}{s['avg_zh']:>8.2f}{s['ok_rate']*100:>7.0f}%")

    with open(os.path.join(PROBE, "probe_result.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())