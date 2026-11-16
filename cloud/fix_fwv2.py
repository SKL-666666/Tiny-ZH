#!/usr/bin/env python3
"""诊断并修复 verify_fwv2.sh 第 3 步词表训练失败。

第 4 步报 "No such file or directory: /tmp/fwv2_tok.json"，
说明第 3 步的词表没训成。先看真实原因。
"""
import os
import sys
import traceback

SAMPLE = "/tmp/fwv2_sample.txt"
OUT = "/tmp/fwv2_tok.json"
SPECIAL = ["<|pad|>", "<|unk|>", "<|im_start|>", "<|im_end|>"]

print("=== 环境 ===")
try:
    import tokenizers
    print(f"tokenizers {tokenizers.__version__}")
except Exception as e:
    print(f"tokenizers 导入失败: {e}")
    sys.exit(1)

print()
print("=== 样本文件 ===")
if not os.path.exists(SAMPLE):
    print(f"[FATAL] {SAMPLE} 不存在")
    sys.exit(1)
sz = os.path.getsize(SAMPLE)
print(f"{SAMPLE}  {sz/1e6:.1f} MB")
with open(SAMPLE, encoding="utf-8", errors="ignore") as f:
    for i, line in enumerate(f):
        if i >= 3:
            break
        print(f"  样{i}: {line[:80]!r}")

print()
print("=== 训词表 ===")
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

try:
    tok = Tokenizer(models.BPE(unk_token="<|unk|>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tok.decoder = decoders.ByteLevel()
    tr = trainers.BpeTrainer(
        vocab_size=6400,
        show_progress=False,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=SPECIAL,
        min_frequency=2,
    )

    def it():
        with open(SAMPLE, encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.rstrip("\n")
                if line.strip():
                    yield line

    tok.train_from_iterator(it(), trainer=tr, length=None)
    tok.save(OUT)
    print(f"OK vocab {tok.get_vocab_size()}  -> {OUT}")

except Exception:
    print("[FATAL] 训练抛异常：")
    traceback.print_exc()
    sys.exit(1)

print()
print("=== 量压缩率 ===")
try:
    import json
    stats_path = "logs/fwv2_stats.json"
    chars = json.load(open(stats_path))["chars"] if os.path.exists(stats_path) else None
    if chars is None:
        print("[WARN] 没有 logs/fwv2_stats.json")
        sys.exit(0)

    tok = Tokenizer.from_file(OUT)
    lines = []
    with open(SAMPLE, encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if i >= 20000:
                break
            lines.append(line.rstrip("\n"))
    tc = sum(len(s) for s in lines)
    tt = sum(len(tok.encode(s).ids) for s in lines)
    ratio = tc / max(tt, 1)

    shard_tok = chars / ratio
    need = 64_816_000 * 20
    have = 7.11e8
    gap = need - have
    nshard = gap / shard_tok if shard_tok > 0 else 99

    print(f"  压缩率 {ratio:.3f} 字/token")
    print(f"  每片 {shard_tok/1e8:.2f} 亿 token（{chars/1e6:.0f}M 字符）")
    print(f"  缺口 {gap/1e8:.2f} 亿 -> 需 {nshard:.1f} 片，建议下 {int(nshard)+1} 片")

    rec = {
        "shard_tokens": int(shard_tok),
        "ratio": round(ratio, 4),
        "chars": int(chars),
        "gap_tokens": int(gap),
        "shards_needed": round(nshard, 2),
        "shards_recommend": int(nshard) + 1,
    }
    os.makedirs("logs", exist_ok=True)
    with open("logs/fwv2_probe.json", "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=2)
    print(f"  写入 logs/fwv2_probe.json")

except Exception:
    traceback.print_exc()
    sys.exit(1)

print()
print("=== 完成 ===")