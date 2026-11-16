#!/usr/bin/env bash
# 补下载 FineWeb-v2 更多片，把 token 数补到 12.96 亿以上
#
# 背景：清洗完只有 6.62 亿 token（10.2x），缺口 6.34 亿。
# 每片约 4.74 亿 token，还需 2 片。
cd /mnt/workspace/tiny-zh-64m || exit 1
mkdir -p data/raw/fwv2 logs

for n in 00003 00004 00005; do
    OUT="data/raw/fwv2/${n}.parquet"
    if [ -f "$OUT" ] && [ "$(stat -c %s "$OUT")" -gt 900000000 ]; then
        echo "[skip] ${n}.parquet 已有"
        continue
    fi
    echo "[下载] ${n}.parquet"
    python3 smart_dl.py \
        "https://hf-mirror.com/datasets/opencsg/chinese-fineweb-edu-v2/resolve/main/data/${n}.parquet" \
        -o "$OUT" --ncpu 32 2>&1 | tail -4
done

echo "=== 结果 ==="
ls -lh data/raw/fwv2/*.parquet