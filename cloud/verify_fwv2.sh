#!/usr/bin/env bash
# 验证 FineWeb-Edu-v2 单片真实产出 —— 明天开机第一件事
#
# 为什么必须先做这步
# ------------------
# 计划里 v2 每片 4.58 亿 token 是这么推的：
#   v1 实测 801MB -> 3.95 亿 token
#   v2 是 928MB，等比放大 = 4.58 亿
# 但这个"等比"没验证过。如果实际只有 2 亿，整个数据配比要重算，
# 下 2 片就不够，得下 4 片。所以开机先花 4 分钟验证，别等下载完才发现。
#
# 输出：logs/fwv2_probe.json，含实际 token 数、压缩率、建议片数
set -u
WORK=/mnt/workspace/tiny-zh-64m
cd "$WORK" || exit 1
mkdir -p data/raw/fwv2 logs
export PYTHONIOENCODING=utf-8

URL_BASE="https://hf-mirror.com/datasets/opencsg/chinese-fineweb-edu-v2/resolve/main/data"
SHARD="00000.parquet"
OUT="data/raw/fwv2/${SHARD}"

echo "===== 1. 下载第一片 ====="
if [ -f "$OUT" ] && [ "$(stat -c %s "$OUT")" -gt 900000000 ]; then
    echo "  已有，跳过"
else
    python3 smart_dl.py "$URL_BASE/$SHARD" -o "$OUT" --ncpu 32 2>&1 | tail -8
fi
SZ=$(stat -c %s "$OUT" 2>/dev/null || echo 0)
echo "  大小 $((SZ/1000000)) MB"
[ "$SZ" -gt 900000000 ] || { echo "[FATAL] 下载不完整"; exit 1; }

echo
echo "===== 2. 读 schema + 全量统计 ====="
python3 - <<'PY'
import pyarrow.parquet as pq
import json

f = pq.ParquetFile("data/raw/fwv2/00000.parquet")
print(f"  rows {f.metadata.num_rows:,}  cols {f.schema_arrow.names}")

total_chars = 0
n_docs = 0
sample = []
for rg in range(f.metadata.num_row_groups):
    t = f.read_row_group(rg)
    vals = t.column("text").to_pylist()
    n_docs += len(vals)
    total_chars += sum(len(x) for x in vals)
    if len(sample) < 200_000:
        sample.extend(vals[:3000])

print(f"  实测 {n_docs:,} 条  {total_chars/1e6:.0f}M 字符  "
      f"均 {total_chars/max(n_docs,1):.0f} 字/条")

# 存样本供训词表
with open("/tmp/fwv2_sample.txt", "w", encoding="utf-8") as fo:
    for s in sample:
        fo.write(s.replace("\n", " ") + "\n")
print(f"  样本 {len(sample):,} 条 -> /tmp/fwv2_sample.txt")
json.dump({"rows": n_docs, "chars": total_chars,
           "chars_per_doc": round(total_chars/max(n_docs,1), 1)},
          open("logs/fwv2_stats.json", "w"), indent=2)
PY

echo
echo "===== 3. 训临时词表（和最终训练同规格 6400）====="
python3 - <<'PY'
import sys
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
SPECIAL = ["<|pad|>", "<|unk|>", "<|im_start|>", "<|im_end|>"]
tok = Tokenizer(models.BPE(unk_token="<|unk|>"))
tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
tok.decoder = decoders.ByteLevel()
tr = trainers.BpeTrainer(vocab_size=6400, show_progress=False,
                         initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                         special_tokens=SPECIAL, min_frequency=2)
def it():
    with open("/tmp/fwv2_sample.txt", encoding="utf-8") as f:
        for line in f:
            yield line.rstrip("\n")
tok.train_from_iterator(it(), trainer=tr, length=None)
tok.save("/tmp/fwv2_tok.json")
print(f"  vocab {tok.get_vocab_size()}")
PY

echo
echo "===== 4. 量压缩率 + 推算 ====="
python3 - <<'PY'
import json
from tokenizers import Tokenizer

tok = Tokenizer.from_file("/tmp/fwv2_tok.json")
stats = json.load(open("logs/fwv2_stats.json"))

lines = [l.rstrip("\n") for l in open("/tmp/fwv2_sample.txt", encoding="utf-8")]
probe = lines[:20000]
tc = sum(len(s) for s in probe)
tt = sum(len(tok.encode(s).ids) for s in probe)
ratio = tc / max(tt, 1)

shard_tok = stats["chars"] / ratio
need = 64_816_000 * 20
have = 7.11e8                       # LCCC+ShareGPT+维基+COIG 实测合计
gap = need - have
shards = (gap / shard_tok) if shard_tok > 0 else 99

print(f"  压缩率 {ratio:.3f} 字/token")
print(f"  每片 token = {shard_tok/1e8:.2f} 亿")
print()
print(f"  20x 目标 {need/1e8:.2f} 亿")
print(f"  已有数据 {have/1e8:.2f} 亿（LCCC+ShareGPT+维基+COIG）")
print(f"  缺口     {gap/1e8:.2f} 亿")
print()
print(f"  >>> 需要 {shards:.1f} 片 -> 建议下 {int(shards)+1} 片")
print(f"      下 {int(shards)+1} 片 = {(int(shards)+1)*shard_tok/1e8:.2f} 亿 token")

rec = {"shard_tokens": int(shard_tok), "ratio": round(ratio, 4),
       "chars_per_doc": stats["chars_per_doc"],
       "gap_tokens": int(gap), "shards_needed": round(shards, 2),
       "shards_recommend": int(shards) + 1}
json.dump(rec, open("logs/fwv2_probe.json", "w"), ensure_ascii=False, indent=2)
print()
print("  结论写入 logs/fwv2_probe.json")
PY

echo
echo "===== 完成 ====="
cat logs/fwv2_probe.json 2>/dev/null