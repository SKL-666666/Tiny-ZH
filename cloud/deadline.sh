#!/usr/bin/env bash
# 定时触发收尾：到点调用 finalize.sh
#
# 21:30 云端强制关机。20:30 触发收尾，留 1 小时余量：
#   停预训练 5 分 + SFT 25 分 + 导出 8 分 + 量化 5 分 + 评测 10 分 + 上传 10 分
#
# 用法：
#   bash cloud/deadline.sh 20:30
#   DEADLINE_EPOCH=1757000000 bash cloud/deadline.sh
set -u
WORK=/mnt/workspace/tiny-zh-64m
cd "$WORK" || exit 1

DEADLINE_EPOCH="${DEADLINE_EPOCH:-}"
if [ -z "$DEADLINE_EPOCH" ]; then
    HHMM="${1:-}"
    [ -z "$HHMM" ] && { echo "[FATAL] 用法: bash cloud/deadline.sh 20:30"; exit 1; }
    DEADLINE_EPOCH=$(date -d "$(date '+%Y-%m-%d') ${HHMM}" +%s)
fi

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a logs/deadline.log; }

NOW=$(date +%s)
R=$(( DEADLINE_EPOCH - NOW ))
if [ "$R" -gt 0 ]; then
    log "等待到 $(date -d @$DEADLINE_EPOCH '+%H:%M') 触发收尾（还有 $((R/60)) 分钟）"
else
    log "deadline 已过，立刻收尾"
fi

# 等，同时每 10 分钟报一次
while :; do
    NOW=$(date +%s)
    R=$(( DEADLINE_EPOCH - NOW ))
    [ "$R" -le 0 ] && break
    MOD=$(( R % 600 ))
    if [ "$MOD" -lt 20 ]; then
        STEP=$(grep -oE 'step +[0-9]+' logs/pretrain.log 2>/dev/null | tail -1 | grep -oE '[0-9]+')
        log "剩余 $((R/60)) 分钟  step=${STEP:-?}"
    fi
    sleep 20
done

log "===== 到点，开始收尾 ====="
exec bash cloud/finalize.sh >> logs/finalize.log 2>&1