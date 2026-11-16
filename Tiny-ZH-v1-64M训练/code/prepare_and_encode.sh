#!/usr/bin/env bash
# 清洗 + 训词表 + 编码 —— 明天执行
#
# 每步都有产物检查，重跑安全。
set -u
WORK=/mnt/workspace/tiny-zh-64m
cd "$WORK" || exit 1
mkdir -p data bin tokenizer logs out/pretrain out/sft
export PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS=8

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a logs/prep_stage.log; }

TARGET_GB="${TARGET_GB:-4.5}"

# ---- 1. 清洗混合 ----
if [ ! -s data/pretrain.txt ] || [ "$(stat -c %s data/pretrain.txt 2>/dev/null || echo 0)" -lt 1000000000 ]; then
    log "===== 清洗混合（目标 ${TARGET_GB}GB）====="
    python3 -u prepare_64m.py "$TARGET_GB" 2>&1 | tee -a logs/prep.log
else
    log "data/pretrain.txt 已有 $(du -h data/pretrain.txt | cut -f1)"
fi

log "语料 $(du -h data/pretrain.txt 2>/dev/null | cut -f1)  $(wc -l < data/pretrain.txt 2>/dev/null) 行"
log "SFT  $(wc -l < data/sft.jsonl 2>/dev/null) 条"

# ---- 2. 训词表 ----
if [ ! -f tokenizer/tokenizer.json ]; then
    log "===== 训练词表（6400）====="
    python3 -u train_tokenizer.py --data data/pretrain.txt \
        --out tokenizer --vocab-size 6400 2>&1 | tail -14 | tee -a logs/tok.log
fi

log "===== 词表三项校验 ====="
python3 check_tokenizer.py --tokenizer tokenizer/tokenizer.json \
    --vocab-size 6400 2>&1 | tee -a logs/tokcheck.log
if grep -q "不合格" logs/tokcheck.log; then
    log "[FATAL] 词表不合格，停止"
    exit 1
fi
log "词表通过"

# ---- 3. 编码 ----
if [ ! -f bin/pretrain_train.bin ]; then
    log "===== 编码预训练 ====="
    python3 -u build_dataset.py --mode pretrain \
        --tokenizer tokenizer/tokenizer.json \
        --src data/pretrain.txt --out-dir bin --val-ratio 0.002 \
        2>&1 | tail -5 | tee -a logs/enc.log
fi

if [ ! -f bin/sft_train.bin ] && [ -s data/sft.jsonl ]; then
    log "===== 编码 SFT ====="
    python3 -u build_dataset.py --mode sft \
        --tokenizer tokenizer/tokenizer.json \
        --src data/sft.jsonl --out-dir bin --val-ratio 0.01 --max-len 512 \
        2>&1 | tail -5 | tee -a logs/enc.log
fi

# ---- 4. 报数 ----
python3 - <<'PY' 2>&1 | tee -a logs/prep_stage.log
import json, os
p = "bin/pretrain_train.bin"
if os.path.exists(p):
    n = os.path.getsize(p) // 2
    need = 64_816_000 * 20
    print(f"预训练 token {n:,} = {n/1e8:.2f} 亿  ({n/64816000:.1f}x 参数)")
    print(f"20x 目标 {need/1e8:.2f} 亿  ->  {'达标' if n >= need else f'缺 {(need-n)/1e8:.2f} 亿'}")
if os.path.exists("bin/sft_meta.json"):
    d = json.load(open("bin/sft_meta.json"))
    print(f"SFT {d['train_samples']:,} 样本 {d['train_tokens']:,} token")
PY

# ---- 5. 内存盘 ----
NEED=$(du -sm bin | cut -f1)
AVAIL=$(df -m /dev/shm | awk 'NR==2 {print $4}')
if [ "${AVAIL:-0}" -gt $(( NEED + 512 )) ]; then
    log "复制到内存盘（${NEED}MB / 可用 ${AVAIL}MB）"
    mkdir -p /dev/shm/tz64/bin /dev/shm/tz64/tokenizer
    cp bin/* /dev/shm/tz64/bin/ 2>/dev/null
    cp tokenizer/* /dev/shm/tz64/tokenizer/ 2>/dev/null
    log "完成 $(du -sh /dev/shm/tz64 | cut -f1)"
else
    log "[WARN] /dev/shm 不足（${AVAIL:-0} < ${NEED}MB），训练将直接读 NAS，吞吐会降约 2.4 倍"
fi

log "===== 准备完成，可以开训 ====="
log "下一步: bash cloud/start64.sh"