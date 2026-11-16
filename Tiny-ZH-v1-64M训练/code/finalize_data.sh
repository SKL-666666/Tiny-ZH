#!/usr/bin/env bash
# 合并 extra.bin 并报告最终 token 数，然后复制内存盘
cd /mnt/workspace/tiny-zh-64m || exit 1

TRAIN=bin/pretrain_train.bin
EXTRA=bin/extra.bin

if [ -f "$EXTRA" ]; then
    EXTRA_N=$(($(stat -c %s "$EXTRA") / 2))
    echo "extra $EXTRA_N token，追加入 train"
    cat "$EXTRA" >> "$TRAIN"
else
    echo "没有 extra.bin"
fi

python3 - <<'PY'
import os
p = 64_816_000
n = os.path.getsize("bin/pretrain_train.bin") // 2
v = os.path.getsize("bin/pretrain_val.bin") // 2 if os.path.exists("bin/pretrain_val.bin") else 0
need = p * 20
print(f"train {n:,} token = {n/1e8:.2f} 亿 = {n/p:.1f}x 参数")
print(f"val   {v:,} token")
print(f"20x 目标 {need/1e8:.2f} 亿 -> {'达标' if n >= need else '缺 %.2f 亿' % ((need-n)/1e8)}")
use = min(n, need)
print(f"按 51,001 tok/s 训练 {use/1e8:.2f} 亿需 {use/51001/3600:.2f} 小时")
PY

echo
echo "=== 复制内存盘 ==="
NEED=$(du -sm bin | cut -f1)
AVAIL=$(df -m /dev/shm | awk 'NR==2 {print $4}')
echo "需 ${NEED}MB / 可用 ${AVAIL}MB"
rm -rf /dev/shm/tz64
if [ "${AVAIL:-0}" -gt $(( NEED + 512 )) ]; then
    mkdir -p /dev/shm/tz64/bin /dev/shm/tz64/tokenizer
    cp bin/pretrain_train.bin bin/pretrain_val.bin /dev/shm/tz64/bin/ 2>/dev/null
    cp bin/sft_train.bin bin/sft_val.bin bin/sft_train_idx.npy bin/sft_val_idx.npy /dev/shm/tz64/bin/ 2>/dev/null
    cp tokenizer/* /dev/shm/tz64/tokenizer/ 2>/dev/null
    echo "完成 $(du -sh /dev/shm/tz64 | cut -f1)"
else
    echo "[WARN] 内存盘不够，用 NAS"
fi