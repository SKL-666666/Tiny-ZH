"""Tiny-ZH-64M 数据准备 —— 最终版

处理 4 个已实测确认的坑
-----------------------
1. LCCC 每行是 **list**，不是 dict
   ["火锅 我 在 重庆...", "哈哈哈哈 ！...", "不会 的 就是 好 油腻"]
   写 o.get("dialog") 会崩。

2. LCCC 文字是**逐字切分**的，字间空格占 38.1%
   实测：20 万段，原样 14.8M 字符，去空格 9.2M 字符
   不挤掉等于白扔 1.4 亿 token。

3. ShareGPT 字段是 conversation[].human/assistant
   不是 conversations[].from/value。旧代码就是这么错的。

4. LCCC 平均只有 46 字/段
   要"语句完整的中文稍长对话"，得设长度下限。

产物
----
  data/pretrain.txt   目标 4.5GB ≈ 12.96 亿 token
  data/sft.jsonl      SFT 对话（from ShareGPT）

用法：
  python3 prepare_64m.py            # 默认 4.5GB
  python3 prepare_64m.py 5.0        # 指定 GB
"""
import glob
import gzip
import hashlib
import json
import os
import random
import re
import sys
import time
from collections import Counter

RAW = "data/raw"
OUT_PRETRAIN = "data/pretrain.txt"
OUT_SFT = "data/sft.jsonl"

# ---------------------------------------------------------------- 清洗

RE_URL = re.compile(r"(?:https?://|www\.)\S+")
RE_HTML = re.compile(r"<[^>]{1,200}>")
RE_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
RE_SPACE = re.compile(r"[ \t]{2,}")
RE_MULTI_NL = re.compile(r"\n{3,}")

# 汉字 + 中文标点。LCCC 的字间空格就在这些字符之间。
_CJK = r"\u4e00-\u9fff\u3000-\u303f\uff00-\uffef"
RE_CJK_SPACE = re.compile(rf"(?<=[{_CJK}])\s+(?=[{_CJK}])")
RE_CJK_CHAR = re.compile(r"[\u4e00-\u9fff]")

# 句末标点
SENT_END = ("。", "！", "？", ".", "!", "?", "…")


def squeeze_spaces(t):
    """去掉汉字/中文标点之间的空格（LCCC 逐字切分的产物）。

    英文单词之间的空格保留——那是真空格。
    """
    return RE_CJK_SPACE.sub("", t)


def clean(t):
    t = RE_URL.sub("", t or "")
    t = RE_HTML.sub("", t)
    t = RE_CTRL.sub("", t)
    t = t.replace("\r", "\n")
    t = squeeze_spaces(t)
    t = RE_SPACE.sub(" ", t)
    t = RE_MULTI_NL.sub("\n\n", t)
    return t.strip()


def zh_ratio(t):
    if not t:
        return 0.0
    return len(RE_CJK_CHAR.findall(t)) / len(t)


def stat_ok(t, min_len=30, max_len=4000, min_zh=0.60,
            max_alnum=0.25, need_sent_end=True):
    """通用质量门。

    min_len 默认 30：LCCC 平均 46 字，设 50 会砍掉一半以上，
    设 30 能保留大部分同时滤掉"头不圆"这类碎片。
    """
    n = len(t)
    if n < min_len or n > max_len:
        return False
    if zh_ratio(t) < min_zh:
        return False
    alnum = sum(1 for c in t if c.isascii() and c.isalnum())
    if alnum / n > max_alnum:
        return False
    if need_sent_end and not any(p in t for p in SENT_END):
        return False
    # 单字符重复率过高 = 复读/噪声
    top = Counter(t).most_common(1)[0][1]
    if top / n > 0.30:
        return False
    return True


class Dedup:
    """前 64 字 blake2b 哈希去重。比 SimHash 快，对短文本够用。"""

    def __init__(self):
        self.seen = set()

    def add(self, t):
        h = hashlib.blake2b(t[:64].encode("utf-8"), digest_size=8).digest()
        if h in self.seen:
            return False
        self.seen.add(h)
        return True


# ---------------------------------------------------------------- 读取器

def read_lccc(path):
    """LCCC large：每行是 list（不是 dict），文字逐字切分。

    兼容三种形态，防止上游格式小改就崩：
      ["utt1", "utt2", ...]
      {"dialog": [...]}
      [{"content": "..."}, ...]
    """
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, list):
                dialog = obj
            elif isinstance(obj, dict):
                dialog = obj.get("dialog") or obj.get("conversations") or []
            else:
                continue
            parts = []
            for u in dialog:
                if isinstance(u, dict):
                    u = u.get("content") or u.get("text") or u.get("value") or ""
                if u:
                    parts.append(str(u))
            if not parts:
                continue
            text = "\n".join(parts)
            if text.strip():
                yield text


def read_sharegpt(path):
    """ShareGPT 中文：{"conversation": [{"human":..., "assistant":...}, ...]}

    坑：不是 conversations[].from/value。
    """
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            conv = o.get("conversation") or o.get("conversations") or o.get("messages") or []
            if not isinstance(conv, list) or not conv:
                continue
            parts = []
            for turn in conv:
                if not isinstance(turn, dict):
                    continue
                h = turn.get("human") or turn.get("user") or turn.get("from") or turn.get("instruction")
                a = turn.get("assistant") or turn.get("gpt") or turn.get("value") or turn.get("output")
                if h:
                    parts.append(str(h))
                if a:
                    parts.append(str(a))
            text = "\n".join(parts)
            if text.strip():
                yield text


def read_jsonl_field(path, keys=("completion", "text", "content", "output", "raw_content")):
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            for k in keys:
                v = o.get(k)
                if isinstance(v, str) and v.strip():
                    yield v
                    break


def read_parquet_text(path, col=None):
    """从 parquet 流式读文本列（FineWeb-Edu-v2 用）。"""
    import pyarrow.parquet as pq
    f = pq.ParquetFile(path)
    names = f.schema_arrow.names
    if col is None:
        for cand in ("text", "content", "raw_content"):
            if cand in names:
                col = cand
                break
    if col is None:
        print(f"    [WARN] {os.path.basename(path)} 没有文本列，跳过（列: {names}）")
        return
    for rg in range(f.metadata.num_row_groups):
        t = f.read_row_group(rg)
        for v in t.column(col).to_pylist():
            if v:
                yield v


# ---------------------------------------------------------------- 主流程

def main():
    target_gb = float(sys.argv[1]) if len(sys.argv) > 1 else 4.5
    target_bytes = target_gb * 1024 ** 3

    print("=" * 66)
    print(f"数据准备  目标 {target_gb:.1f} GB 纯文本")
    print("=" * 66)

    dedup = Dedup()
    t0 = time.time()
    n_kept = 0
    n_bytes = 0
    per_src = {}

    def exists(p, min_sz=1024):
        return os.path.exists(p) and os.path.getsize(p) > min_sz

    # (标签, 路径列表, 读取器, 目标占比, min_len, max_len)
    # 占比按"字数贡献"分配，对话合计 45%
    fw_shards = sorted(glob.glob(f"{RAW}/fwv2/*.parquet"))
    plan = [
        ("lccc",      [f"{RAW}/lccc_large.jsonl.gz"], read_lccc,        0.30, 30,  4000),
        ("sharegpt",  [f"{RAW}/common_zh_70k.jsonl"], read_sharegpt,    0.09, 50,  4000),
        ("sharegpt2", [f"{RAW}/unknow_zh_38k.jsonl"], read_sharegpt,    0.06, 50,  4000),
        ("wiki",      [f"{RAW}/wiki.jsonl"],          read_jsonl_field, 0.15, 100, 4000),
        ("coig",      [f"{RAW}/coig_full.jsonl"],     read_jsonl_field, 0.05, 30,  4000),
        ("ruozhi",    [f"{RAW}/ruozhi.jsonl"],        read_jsonl_field, 0.00, 10,  800),
        ("fineweb",   fw_shards,                      read_parquet_text, 0.35, 100, 4000),
    ]

    with open(OUT_PRETRAIN, "w", encoding="utf-8") as out:
        for tag, paths, reader, share, mn, mx in plan:
            avail = [p for p in paths if exists(p)]
            if not avail:
                print(f"  [{tag}] 跳过（无文件）")
                continue
            cap = int(target_bytes * share)
            if cap <= 0:
                cap = int(target_bytes * 0.01)     # ruozhi 这类给个小额度
            got_bytes = 0
            n0 = n_kept
            print(f"  [{tag}] {len(avail)} 个文件，目标 {cap/1024**2:.0f} MB …")
            stop = False
            for p in avail:
                if stop:
                    break
                try:
                    for raw in reader(p):
                        t = clean(raw)
                        if not stat_ok(t, mn, mx):
                            continue
                        if not dedup.add(t):
                            continue
                        line = t.replace("\n", " ") + "\n"
                        out.write(line)
                        b = len(line.encode("utf-8"))
                        n_bytes += b
                        got_bytes += b
                        n_kept += 1
                        if got_bytes >= cap:
                            stop = True
                            break
                        if n_kept % 200000 == 0:
                            el = max(time.time() - t0, 1)
                            print(f"    {n_kept:,} 条 {n_bytes/1e9:.2f}GB "
                                  f"{n_bytes/1e6/el:.2f}MB/s", flush=True)
                except Exception as e:
                    print(f"    [ERR] {os.path.basename(p)}: {type(e).__name__}: {str(e)[:90]}")
            per_src[tag] = n_kept - n0
            print(f"    +{n_kept-n0:,} 条  {got_bytes/1024**2:.0f} MB  "
                  f"({got_bytes/max(target_bytes,1)*100:.1f}%)")

    print(f"\n预训练语料 {n_kept:,} 条  {n_bytes/1024**3:.2f} GB")
    est_tok = n_bytes / 2.8 / 1.55
    print(f"  估 {est_tok/1e8:.2f} 亿 token  ({est_tok/64816000:.1f}x 参数)")
    for k, v in sorted(per_src.items(), key=lambda x: -x[1]):
        if v:
            print(f"    {k:<10} {v:>9,} 条 ({v/max(n_kept,1)*100:.1f}%)")

    # ---------- SFT ----------
    print("\n" + "=" * 66)
    print("生成 SFT 对话")
    print("=" * 66)
    n_sft = 0
    sg = f"{RAW}/common_zh_70k.jsonl"
    with open(OUT_SFT, "w", encoding="utf-8") as out:
        if exists(sg):
            with open(sg, encoding="utf-8", errors="ignore") as f:
                for line in f:
                    try:
                        o = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    conv = o.get("conversation") or []
                    msgs = []
                    for turn in conv:
                        if not isinstance(turn, dict):
                            continue
                        h = turn.get("human")
                        a = turn.get("assistant")
                        if h:
                            hc = clean(str(h))
                            if hc:
                                msgs.append({"role": "user", "content": hc})
                        if a:
                            ac = clean(str(a))
                            if ac:
                                msgs.append({"role": "assistant", "content": ac})
                    if len(msgs) < 2 or msgs[0]["role"] != "user":
                        continue
                    total = sum(len(m["content"]) for m in msgs)
                    if total < 20 or total > 2000:
                        continue
                    out.write(json.dumps({"conversations": msgs},
                                         ensure_ascii=False) + "\n")
                    n_sft += 1
    print(f"  SFT {n_sft:,} 条")
    print(f"\n总用时 {(time.time()-t0)/60:.1f} 分钟")
    return 0


if __name__ == "__main__":
    sys.exit(main())