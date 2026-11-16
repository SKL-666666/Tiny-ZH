#!/usr/bin/env python3
"""为 SFT 准备「答案 70-200 字 + 预训练未用过的数据」。

为什么不能复用预训练数据
------------------------
预训练已经见过 LCCC / ShareGPT / FineWeb / 维基 / COIG 的全部内容。
如果 SFT 再用它们，模型是在背答案（记忆），不是在学对话（泛化）。
val loss 会虚低，实际生成质量不会提升。

用什么数据（全部未进过预训练）
------------------------------
1. Firefly 中文 1.1M     165 万条指令，平均 94 字/条    <- 主力，已下载
2. Infinity-Instruct      BAAI 精选指令集，15 片
3. alpaca-zh              5.2 万条中文指令

为什么 70-200 字
----------------
- < 70 字：模型学会「一句话打发」，遇到复杂问题不够用
- > 200 字：64M 模型写长文容易漂移、复读、逻辑断裂
- 70-200 是最能体现「流利且完整」的区间

用法：
    python3 prepare_sft_fresh.py
"""
import glob
import json
import os
import re
import sys
import time
from collections import Counter

RAW = "data/raw"
OUT = "data/sft.jsonl"

RE_URL = re.compile(r"(?:https?://|www\.)\S+")
RE_HTML = re.compile(r"<[^>]{1,200}>")
RE_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
RE_CJK = re.compile(r"[\u4e00-\u9fff]")
RE_SPACE = re.compile(r"[ \t]{2,}")

MIN_ANS = 70
MAX_ANS = 200
MIN_Q = 2
MAX_Q = 300


def clean(t):
    t = RE_URL.sub("", t or "")
    t = RE_HTML.sub("", t)
    t = RE_CTRL.sub("", t)
    t = t.replace("\r", "\n")
    t = RE_SPACE.sub(" ", t)
    return t.strip()


def zh_ratio(t):
    if not t:
        return 0.0
    return len(RE_CJK.findall(t)) / len(t)


def ans_ok(a):
    """答案质量门：长度 70-200，中文占比高，不是复读，结尾像完整句子。"""
    n = len(a)
    if n < MIN_ANS or n > MAX_ANS:
        return False
    if zh_ratio(a) < 0.55:
        return False
    # 单字符重复率过高 = 复读
    top = Counter(a).most_common(1)[0][1]
    if top / n > 0.25:
        return False
    # 结尾要有标点（说明说完整了）
    if a[-1] not in "。！？…」』）.!?":
        return False
    return True


def read_firefly(path):
    """Firefly: {"instruction", "input", "output"}"""
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            q = (o.get("instruction") or "").strip()
            inp = (o.get("input") or "").strip()
            a = (o.get("output") or "").strip()
            if not q or not a:
                continue
            if inp:
                q = f"{q}\n{inp}"
            yield q, a


def read_parquet_instruct(path):
    """Infinity-Instruct: parquet，列通常是 conversations 或 instruction/output"""
    import pyarrow.parquet as pq
    f = pq.ParquetFile(path)
    names = f.schema_arrow.names
    for rg in range(f.metadata.num_row_groups):
        t = f.read_row_group(rg)
        # 形态 1: conversations / messages
        conv_col = None
        for c in ("conversations", "messages", "conversation"):
            if c in names:
                conv_col = c
                break
        if conv_col:
            for conv in t.column(conv_col).to_pylist():
                if not isinstance(conv, list) or len(conv) < 2:
                    continue
                q = a = None
                for m in conv:
                    if not isinstance(m, dict):
                        continue
                    role = m.get("role") or m.get("from")
                    val = m.get("content") or m.get("value") or ""
                    if role in ("user", "human") and q is None:
                        q = val
                    elif role in ("assistant", "gpt") and q is not None:
                        a = val
                        break
                if q and a:
                    yield q, a
            continue
        # 形态 2: instruction / output
        if "instruction" in names and "output" in names:
            qs = t.column("instruction").to_pylist()
            ins = t.column("input").to_pylist() if "input" in names else [""] * len(qs)
            outs = t.column("output").to_pylist()
            for q, i, a in zip(qs, ins, outs):
                if q and a:
                    yield (f"{q}\n{i}" if i else q), a


def main():
    t0 = time.time()
    print("=" * 66)
    print(f"SFT 数据准备  答案 {MIN_ANS}-{MAX_ANS} 字  只用预训练没见过的数据")
    print("=" * 66)

    sources = []

    # 1. Firefly 1.1M
    for p in (f"{RAW}/firefly.jsonl", f"{RAW}/firefly-train-1.1M.jsonl"):
        if os.path.exists(p) and os.path.getsize(p) > 100 << 20:
            sources.append(("firefly", p, read_firefly))
            break

    # 2. Infinity-Instruct
    inf = sorted(glob.glob(f"{RAW}/inf/*.parquet"))
    if inf:
        for p in inf:
            sources.append(("infinity", p, read_parquet_instruct))

    # 3. alpaca-zh
    for p in (f"{RAW}/alpaca_zh.jsonl", f"{RAW}/alpaca-zh.jsonl"):
        if os.path.exists(p):
            sources.append(("alpaca", p, read_firefly))
            break

    if not sources:
        print("[FATAL] 没有任何可用数据源")
        print("  需要先下载 firefly / Infinity-Instruct")
        return 1

    print(f"\n数据源 {len(sources)} 个：")
    for tag, p, _ in sources:
        sz = os.path.getsize(p) / 1e6
        print(f"  {tag:<10} {os.path.basename(p):<40} {sz:.0f} MB")

    # ---- 去重 + 过滤 ----
    seen = set()
    n_kept = 0
    per_src = Counter()
    lens = []

    with open(OUT, "w", encoding="utf-8") as out:
        for tag, path, reader in sources:
            n0 = n_kept
            n_scan = 0
            try:
                for q, a in reader(path):
                    n_scan += 1
                    q = clean(q)
                    a = clean(a)
                    if len(q) < MIN_Q or len(q) > MAX_Q:
                        continue
                    if not ans_ok(a):
                        continue
                    # 去重：问题前 48 字
                    h = hash(q[:48])
                    if h in seen:
                        continue
                    seen.add(h)
                    out.write(json.dumps({
                        "conversations": [
                            {"role": "user", "content": q},
                            {"role": "assistant", "content": a},
                        ]
                    }, ensure_ascii=False) + "\n")
                    n_kept += 1
                    lens.append(len(a))
                    if n_kept % 20000 == 0:
                        print(f"    {n_kept:,} 条  {time.time()-t0:.0f}s", flush=True)
                    if n_kept >= 300000:
                        break
            except Exception as e:
                print(f"    [ERR] {os.path.basename(path)}: {type(e).__name__}: {str(e)[:80]}")
            got = n_kept - n0
            per_src[tag] += got
            print(f"  {tag:<10} 扫描 {n_scan:,} 条 -> 保留 {got:,} 条", flush=True)
            if n_kept >= 300000:
                break

    print(f"\n{'='*66}")
    print(f"SFT 数据 {n_kept:,} 条 -> {OUT}")
    if lens:
        lens.sort()
        print(f"  答案长度 p10={lens[len(lens)//10]} p50={lens[len(lens)//2]} "
              f"p90={lens[len(lens)*9//10]} max={lens[-1]}")
        print(f"  平均 {sum(lens)/len(lens):.0f} 字")
    print(f"  来源分布：")
    for k, v in per_src.most_common():
        print(f"    {k:<10} {v:>7,} ({v/max(n_kept,1)*100:.1f}%)")
    print(f"  文件 {os.path.getsize(OUT)/1e6:.0f} MB")
    print(f"  耗时 {(time.time()-t0)/60:.1f} 分钟")
    return 0


if __name__ == "__main__":
    sys.exit(main())