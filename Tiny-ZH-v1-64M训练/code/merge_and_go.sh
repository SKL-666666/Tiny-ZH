#!/usr/bin/env bash
# 合并 bin -> 切训练/验证集 -> 复制内存盘 -> 开训
#
# 背景：编码分两条路产出
#   _full.bin   build_dataset.py 单进程（基础数据 LCCC/ShareGPT/维基/COIG）
#   extra.bin   fast_encode.py 多进程（FineWeb 补充）
# 这里直接拼起来，用最后 0.1% 做验证集。
set -u
cd /mnt/workspace/tiny-zh-64m || exit 1
mkdir -p bin logs out/pretrain out/sft
export PYTHONIOENCODING=utf-8

echo "=== 等基础编码结束 ==="
for i in $(seq 1 60); do
    n=$(pgrep -c -f build_dataset 2>/dev/null || echo 0)
    [ "$n" = "0" ] && break
    sleep 10
done
echo "  build_dataset 已退出"

echo "=== 现状 ==="
ls -lh bin/ 2>/dev/null

# 等 build_dataset 完成切分（它自己会生成 pretrain_train.bin）
if [ -f bin/pretrain_train.bin ]; then
    echo "  基础切分已完成"
    BASE_N=$(($(stat -c %s bin/pretrain_train.bin) / 2))
    echo "  基础 train $BASE_N token"
else
    echo "  基础切分未完成，手动切 _full.bin"
    python3 - <<'PY'
import os
import numpy as np
src = "bin/_full.bin"
if not os.path.exists(src):
    print("[FATAL] 没有 _full.bin")
    raise SystemExit(1)
arr = np.memmap(src, dtype=np.uint16, mode="r")
total = len(arr)
n_val = max(int(total * 0.002), 20000)
split = total - n_val
with open("bin/pretrain_train.bin", "wb") as f:
    f.write(arr[:split].tobytes())
with open("bin/pretrain_val.bin", "wb") as f:
    f.write(arr[split:].tobytes())
print(f"  切分 train={split:,} val={n_val:,}")
PY
fi

echo "=== 合并 extra.bin ==="
if [ -f bin/extra.bin ]; then
    EXTRA_N=$(($(stat -c %s bin/extra.bin) / 2))
    echo "  extra $EXTRA_N token"
    cat bin/extra.bin >> bin/pretrain_train.bin
    echo "  已追加"
else
    echo "  没有 extra.bin，跳过"
fi

echo "=== 最终统计 ==="
python3 - <<'PY'
import os
n = os.path.getsize("bin/pretrain_train.bin") // 2
v = os.path.getsize("bin/pretrain_val.bin") // 2 if os.path.exists("bin/pretrain_val.bin") else 0
p = 64_816_000
need = p * 20
print(f"  train {n:,} token = {n/1e8:.2f} 亿 = {n/p:.1f}x 参数")
print(f"  val   {v:,} token")
print(f"  20x 目标 {need/1e8:.2f} 亿 -> {'达标' if n >= need else f'缺 {(need-n)/1e8:.2f} 亿'}")
# 训练耗时估算
tps = 51001
hours = min(n, need) / tps / 3600
print(f"  按 51,001 tok/s 训练 {min(n,need)/1e8:.2f} 亿需 {hours:.2f} 小时")
PY

echo "=== 复制内存盘 ==="
NEED=$(du -sm bin | cut -f1)
AVAIL=$(df -m /dev/shm | awk 'NR==2 {print $4}')
echo "  需 ${NEED}MB / 可用 ${AVAIL}MB"
if [ "${AVAIL:-0}" -gt $(( NEED + 512 )) ]; then
    mkdir -p /dev/shm/tz64/bin /dev/shm/tz64/tokenizer
    cp bin/pretrain_train.bin bin/pretrain_val.bin /dev/shm/tz64/bin/ 2>/dev/null
    [ -f bin/sft_train.bin ] && cp bin/sft_*.bin /dev/shm/tz64/bin/ 2>/dev/null
    cp tokenizer/* /dev/shm/tz64/tokenizer/ 2>/dev/null
    echo "  完成 $(du -sh /dev/shm/tz64 | cut -f1)"
else
    echo "  [WARN] 内存盘不够（需 ${NEED}MB），用 NAS"
fi

echo "=== 可以开训 ==="