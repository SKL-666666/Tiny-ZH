#!/usr/bin/env python3
"""在 191MB 样本上训词表并量压缩率。

为什么用小样本
--------------
上一版用 1.1GB 全量样本训词表，进程被 OOM Killer 杀掉（28GB 内存不够）。
train_from_iterator 虽然流式读，但 BPE trainer 内部会累积统计信息，
1.1GB 中文文本膨胀远超预期。191MB 足够训出 6400 词表。
"""
import json
import os
import sys
import traceback

SAMPLE = "/tmp/fwv2_small.txt"
OUT = "/tmp/fwv2_tok.json"
SPECIAL = ["<|pad|>", "<|unk|>", "<|im_start|>", "<|im_end|>"]


def main():
    print("=== 样本 ===")
    if not os.path.exists(SAMPLE):
        print(f"[FATAL] {SAMPLE} 不存在")
        return 1
    print(f"  {os.path.getsize(SAMPLE)/1e6:.1f} MB")

    print("\n=== 训词表 ===")
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    tok = Tokenizer(models.BPE(unk_token="<|unk|>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tok.decoder = decoders.ByteLevel()
    tr = trainers.BpeTrainer(
        vocab_size=6400,
        show_progress=True,
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
    print(f"  OK vocab {tok.get_vocab_size()}")

    print("\n=== 压缩率 ===")
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
    print(f"  {ratio:.3f} 字/token")

    stats_path = "logs/fwv2_stats.json"
    if not os.path.exists(stats_path):
        print("  [WARN] 没有 logs/fwv2_stats.json，只输出压缩率")
        return 0

    chars = json.load(open(stats_path))["chars"]
    shard_tok = chars / ratio
    need = 64_816_000 * 20
    have = 7.11e8
    gap = need - have
    nshard = gap / shard_tok if shard_tok > 0 else 99

    print(f"\n=== 推算 ===")
    print(f"  整片 {chars/1e6:.0f}M 字符 -> {shard_tok/1e8:.2f} 亿 token")
    print(f"  20x 需 {need/1e8:.2f} 亿，已有 {have/1e8:.2f} 亿，缺 {gap/1e8:.2f} 亿")
    print(f"  需要 {nshard:.1f} 片 -> 建议下 {int(nshard)+1} 片")

    rec = {
        "shard_tokens": int(shard_tok),
        "ratio": round(ratio, 4),
        "chars": int(chars),
        "gap_tokens": int(gap),
        "shards_needed": round(nshard, 2),
        "shards_recommend": int(nshard) + 1,
    }
    with open("logs/fwv2_probe.json", "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=2)
    print(f"  写入 logs/fwv2_probe.json")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)