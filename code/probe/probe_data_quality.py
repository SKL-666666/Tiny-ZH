"""实测候选数据源的质量，为 33.7M 模型的 20× 训练做配比。

预算：33.7M × 20 = 674M token

要验证的四个源
--------------
1. pretrain_t2t_mini      预训练主线（qwen3 蒸馏完整段落）
2. Infinity-Instruct 7M_core  BAAI 精筛指令集
3. 中文维基                事实性长文本
4. LCCC large             口语对话（需挤空格）

每个源测：样本数、平均长度、中文占比、token 产出、预估全量
"""
import json
import os
import re
import sys

RE_CJK = re.compile(r"[\u4e00-\u9fff]")
RE_SPACE_CJK = re.compile(r"(?<=[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef])\s+(?=[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef])")


def zh_ratio(t):
    return len(RE_CJK.findall(t)) / max(len(t), 1)


def stat(name, texts, total_count=None, squeeze=False):
    """统计一个数据源。"""
    n = len(texts)
    if n == 0:
        print(f"  {name}: 无有效样本")
        return
    lens = []
    zhs = []
    for t in texts:
        if squeeze:
            t = RE_SPACE_CJK.sub("", t)
        lens.append(len(t))
        zhs.append(zh_ratio(t))
    lens.sort()
    zhs.sort()
    avg_len = sum(lens) / n
    avg_zh = sum(zhs) / n

    print(f"\n  【{name}】")
    print(f"    样本 {n:,}  平均 {avg_len:.0f} 字")
    print(f"    长度 p10={lens[n//10]} p50={lens[n//2]} p90={lens[n*9//10]}")
    print(f"    中文占比 平均 {avg_zh:.2f}  p10={zhs[n//10]:.2f} p50={zhs[n//2]:.2f}")
    ok = sum(1 for i in range(n) if zhs[i] >= 0.65 and 100 <= lens[i] <= 2000)
    print(f"    合格率（中文≥0.65 且 100-2000字）{ok/n*100:.0f}%")
    if total_count:
        # 按合格率折算
        est = total_count * (ok / n) * avg_len
        print(f"    全量 {total_count:,} 条 -> 约 {est/1e8:.2f} 亿字")


def main():
    print("=" * 70)
    print("数据源实测")
    print("=" * 70)
    print(f"目标：33.7M × 20 = 674M token")
    print(f"按 1.5 字/token 换算，需要约 10 亿字纯文本")

    RAW = "data_probe"

    # ---- 1. pretrain_t2t_mini ----
    p = f"{RAW}/pretrain_t2t_mini.jsonl"
    if os.path.exists(p):
        texts = []
        total = 0
        with open(p, encoding="utf-8", errors="ignore") as f:
            for line in f:
                total += 1
                if len(texts) < 20000:
                    try:
                        o = json.loads(line)
                        t = o.get("text") or o.get("content") or ""
                        if t:
                            texts.append(t)
                    except Exception:
                        pass
        stat("pretrain_t2t_mini", texts, total)
    else:
        print(f"\n  [缺] {p}")

    # ---- 2. Infinity-Instruct 7M_core ----
    p = f"{RAW}/7M_core_0.parquet"
    if os.path.exists(p):
        import pyarrow.parquet as pq
        f = pq.ParquetFile(p)
        t = f.read_row_group(0)
        cols = f.schema_arrow.names
        print(f"\n  [7M_core] 列: {cols}  rows={f.metadata.num_rows:,}")
        # 提取文本
        texts = []
        for i in range(min(5000, t.num_rows)):
            parts = []
            for c in cols:
                v = t.column(c)[i].as_py()
                if isinstance(v, str):
                    parts.append(v)
                elif isinstance(v, list):
                    for m in v:
                        if isinstance(m, dict):
                            parts.append(str(m.get("content") or m.get("value") or ""))
            s = "\n".join(p for p in parts if p)
            if s.strip():
                texts.append(s)
        stat("Infinity-Instruct 7M_core（指令合并）", texts, f.metadata.num_rows * 15)
    else:
        print(f"\n  [缺] {p}")

    # ---- 3. 中文维基 ----
    p = f"{RAW}/wiki.jsonl"
    if os.path.exists(p):
        texts = []
        total = 0
        with open(p, encoding="utf-8", errors="ignore") as f:
            for line in f:
                total += 1
                if len(texts) < 20000:
                    try:
                        o = json.loads(line)
                        t = o.get("completion") or o.get("text") or ""
                        if t:
                            texts.append(t)
                    except Exception:
                        pass
        stat("中文维基", texts, total)
    else:
        print(f"\n  [缺] {p}")

    # ---- 4. LCCC ----
    p = f"{RAW}/lccc_sample.jsonl"
    if os.path.exists(p):
        import gzip
        texts = []
        total = 0
        opener = gzip.open if p.endswith(".gz") else open
        with opener(p, "rt", encoding="utf-8", errors="ignore") as f:
            for line in f:
                total += 1
                if len(texts) < 20000:
                    try:
                        o = json.loads(line)
                        dlg = o if isinstance(o, list) else (o.get("dialog") or [])
                        parts = []
                        for u in dlg:
                            if isinstance(u, dict):
                                u = u.get("content") or ""
                            if u:
                                parts.append(str(u))
                        s = " ".join(parts)
                        if s.strip():
                            texts.append(s)
                    except Exception:
                        pass
        stat("LCCC（挤空格后）", texts, 12007759, squeeze=True)
    else:
        print(f"\n  [缺] {p}")

    print("\n" + "=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())