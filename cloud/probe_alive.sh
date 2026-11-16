#!/usr/bin/env bash
# 采样 GPU 和日志 4 次，确认训练是否真在跑
cd /mnt/workspace/tiny-zh-64m || exit 1
for i in 1 2 3 4; do
    echo "--- $(date '+%H:%M:%S') ---"
    nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
    tail -1 logs/sft_high.log
    echo "log mtime: $(stat -c %y logs/sft_high.log | cut -d. -f1)"
    sleep 8
done
echo
echo "=== 进程详情 ==="
ps aux | grep '[t]rain.py' | head -2
echo
echo "=== 日志大小变化 ==="
stat -c '%s bytes' logs/sft_high.log