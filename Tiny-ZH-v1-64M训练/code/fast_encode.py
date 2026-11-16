#!/usr/bin/env python3
"""快速编码 FineWeb 分片 -> 追加到 uint16 bin。

为什么单独写
------------
build_dataset.py 是单进程流式编码，2.68GB 花了 10 分钟。
6 片 FineWeb（5.5GB parquet，解压后约 38 亿字符）单进程要 1 小时+。
这里用 multiprocessing 按 row group 切分，8 核并行，快 5-6 倍。

关键：分词在 CPU 上做，GPU 帮不上忙。真正的加速是多进程。
（用户问"能用GPU快速清洗吗"——不能，tokenize 是纯 CPU 的字符串操作。）

用法：
    python3 fast_encode.py --shards data/raw/fwv2/*.parquet \
        --tokenizer tokenizer/tokenizer.json --out bin/extra.bin
"""
import argparse
import glob
import os
import sys
import time

import numpy as np
from tokenizers import Tokenizer

# 每个 worker 全局持有 tokenizer，避免重复加载
_TOK = None
_TOK_PATH = None


def _init(tok_path):
    global _TOK, _TOK_PATH
    _TOK_PATH = tok_path
    _TOK = Tokenizer.from_file(tok_path)


def _enc_group(args):
    """编码一个 row group，返回 uint16 字节串。"""
    path, rg = args
    import pyarrow.parquet as pq
    try:
        f = pq.ParquetFile(path)
        t = f.read_row_group(rg)
        names = t.schema.names
        col = None
        for c in ("text", "content", "raw_content"):
            if c in names:
                col = c
                break
        if col is None:
            return b""
        buf = []
        for s in t.column(col).to_pylist():
            if not s:
                continue
            s = s.strip()
            if len(s) < 100:
                continue
            ids = _TOK.encode(s).ids
            if ids:
                buf.extend(ids)
                buf.append(3)          # EOS
        if not buf:
            return b""
        return np.asarray(buf, dtype=np.uint16).tobytes()
    except Exception as e:
        return b""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", nargs="+", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-tokens", type=float, default=0,
                    help="达到这个 token 数就停（0=不限）")
    a = ap.parse_args()

    # 展开 glob
    shards = []
    for s in a.shards:
        shards.extend(sorted(glob.glob(s)) if any(c in s for c in "*?[") else [s])
    shards = [s for s in shards if os.path.exists(s)]
    if not shards:
        print("[FATAL] 没有可用的分片")
        return 1

    # 收集所有 row group 任务
    import pyarrow.parquet as pq
    tasks = []
    for p in shards:
        f = pq.ParquetFile(p)
        n = f.metadata.num_row_groups
        for rg in range(n):
            tasks.append((p, rg))
    print(f"{len(shards)} 片，{len(tasks)} 个 row group，{a.workers} 进程")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    limit = int(a.max_tokens) if a.max_tokens > 0 else 0

    t0 = time.time()
    total = 0
    from multiprocessing import Pool

    with open(a.out, "wb") as fo, Pool(a.workers, initializer=_init,
                                       initargs=(a.tokenizer,)) as pool:
        for i, chunk in enumerate(pool.imap_unordered(_enc_group, tasks, chunksize=1)):
            if not chunk:
                continue
            fo.write(chunk)
            total += len(chunk) // 2
            el = max(time.time() - t0, 1e-9)
            print(f"  {i+1}/{len(tasks)} rg  {total:,} token  "
                  f"{total/el/1e3:.0f}k tok/s  {el/60:.1f}min", flush=True)
            if limit and total >= limit:
                print(f"  达到上限 {limit:,}，停止")
                pool.terminate()
                break

    print(f"\n完成 {a.out}  {total:,} token = {total/1e8:.2f} 亿  "
          f"{(time.time()-t0)/60:.1f} 分钟")
    return 0


if __name__ == "__main__":
    sys.exit(main())