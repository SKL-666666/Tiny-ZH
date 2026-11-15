#!/usr/bin/env bash
# 明天一键：验证 -> 下载 -> 清洗编码
#
# 分阶段，每阶段有产物检查，中断后重跑安全。
#
# 用法：
#   bash cloud/run_all.sh              # 全流程
#   bash cloud/run_all.sh --skip-verify # 跳过 v2 产出验证（用已有结论）
set -u
WORK=/mnt/workspace/tiny-zh-64m
cd "$WORK" || exit 1
mkdir -p logs data/raw/fwv2 bin tokenizer out/pretrain out/sft
export PYTHONIOENCODING=utf-8

log() { echo ""; echo "################ $* ################" | tee -a logs/run_all.log; }

SKIP_VERIFY=0
for a in "$@"; do
    [ "$a" = "--skip-verify" ] && SKIP_VERIFY=1
done

START=$(date +%s)

if [ "$SKIP_VERIFY" = "0" ]; then
    log "阶段 1/3  验证 FineWeb-Edu-v2 单片产出"
    bash cloud/verify_fwv2.sh 2>&1 | tee -a logs/run_all.log
    RC=${PIPESTATUS[0]}
    if [ "$RC" != "0" ]; then
        echo "[FATAL] 验证失败，检查 logs/fwv2_probe.json"
        exit 1
    fi
else
    log "跳过验证"
fi

log "阶段 2/3  下载全部数据"
bash cloud/fetch_all.sh 2>&1 | tee -a logs/run_all.log

log "阶段 3/3  清洗 + 训词表 + 编码"
bash cloud/prepare_and_encode.sh 2>&1 | tee -a logs/run_all.log

EL=$(( $(date +%s) - START ))
log "准备阶段全部完成，用时 $((EL/60)) 分钟"
echo
echo "================================================"
echo "  现在可以开训:"
echo "    bash cloud/start64.sh"
echo "  断点续训:"
echo "    bash cloud/start64.sh --resume"
echo "================================================"