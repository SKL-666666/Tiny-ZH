#!/usr/bin/env bash
# 停掉 v3 训练
pkill -9 -f 'train_v3' 2>/dev/null
pkill -9 -f 'train.py --mode sft' 2>/dev/null
# 直接按 PID 杀
for pid in $(pgrep -f 'train.py'); do
    kill -9 "$pid" 2>/dev/null && echo "killed $pid"
done
sleep 4
echo "--- 剩余 train.py 进程 ---"
ps aux | grep '[t]rain.py' | wc -l
echo "--- GPU ---"
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
echo "--- v3 训练到哪一步 ---"
tail -3 logs/sft_v3.log 2>/dev/null || echo "无日志"