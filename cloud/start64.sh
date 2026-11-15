#!/usr/bin/env bash
# Tiny-ZH-64M 训练主脚本（A10 / CUDA）
#
# 抗突然关机：四层防护
# --------------------
# 1. 产物全在 /mnt/workspace（NAS 持久卷），实例销毁不丢
# 2. checkpoint 原子写盘 .tmp -> fsync -> rename，断电不留半个文件
# 3. 每次存 step_NNNN.pt（留最近 3 个），last.pt 用硬链接，坏了能退
# 4. checkpoint 存 total_steps，续训沿用，LR 余弦曲线不错位
#
# 用法：
#   bash cloud/start64.sh              # 从头训
#   bash cloud/start64.sh --resume     # 断点续训（幂等）
set -u

WORK=/mnt/workspace/tiny-zh-64m
cd "$WORK" || exit 1
mkdir -p logs bin tokenizer out/pretrain out/sft
export PYTHONIOENCODING=utf-8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a logs/main.log; }

RESUME=""
for a in "$@"; do
    case "$a" in
        --resume) RESUME="--resume" ;;
    esac
done

# 12.96 亿 ÷ 51,001 tok/s = 7.06 小时 = 25,400 秒；留 3% 给存盘
PRETRAIN_BUDGET=${PRETRAIN_BUDGET:-25400}
SFT_BUDGET=${SFT_BUDGET:-2100}

log "===== Tiny-ZH-64M 训练 预训练预算 $((PRETRAIN_BUDGET/60))min ====="
df -h /mnt/workspace | tail -1

# ---- 环境检查 ----
if ! python3 - <<'PY' 2>&1 | tee -a logs/main.log
import sys
try:
    import torch
except ImportError:
    print("FATAL: torch 未安装"); sys.exit(1)
print(f"torch {torch.__version__}  cuda {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    print("FATAL: GPU 不可用 —— 先查驱动，不要继续")
    sys.exit(1)
p = torch.cuda.get_device_properties(0)
print(f"dev {p.name}  {p.total_memory/1e9:.1f}GB  sm {p.multi_processor_count}")
print(f"bf16 {torch.cuda.is_bf16_supported()}")
import os
print(f"cpu {os.cpu_count()} 核")
PY
then
    log "[FATAL] 环境检查未通过"
    exit 1
fi

# ---- 数据检查 ----
[ -f bin/pretrain_train.bin ] || { log "[FATAL] 没有 bin/pretrain_train.bin，先跑 prepare_and_encode.sh"; exit 1; }

DATADIR=$WORK/bin
if [ -f /dev/shm/tz64/bin/pretrain_train.bin ]; then
    DATADIR=/dev/shm/tz64/bin
    log "数据在内存盘 /dev/shm/tz64/bin"
else
    log "[WARN] 内存盘没有数据，直接读 NAS（吞吐会降）"
fi

python3 - <<'PY' 2>&1 | tee -a logs/main.log
import os
n = os.path.getsize("bin/pretrain_train.bin") // 2
print(f"预训练 token {n:,} = {n/1e8:.2f} 亿 ({n/64816000:.1f}x)")
PY

# ---- 6. 预训练 ----
if [ ! -f out/pretrain/best.pt ]; then
    log "===== 预训练 ====="
    python3 -u train.py --mode pretrain \
        --data "$DATADIR" --out out/pretrain \
        --seq-len 1024 --micro-batch 32 --grad-accum 1 \
        --lr 5e-4 --max-steps 39550 --time-budget "$PRETRAIN_BUDGET" \
        --probe-steps 0 --save-every 250 --log-every 25 \
        $RESUME 2>&1 | tee -a logs/pretrain.log
    log "预训练结束 $(ls -lh out/pretrain/best.pt 2>/dev/null | awk '{print $5}')"
else
    log "预训练已完成，跳过"
fi

# ---- 7. SFT ----
if [ ! -f out/sft/best.pt ] && [ -f bin/sft_train.bin ]; then
    log "===== SFT ====="
    INIT=out/pretrain/best.pt
    [ -f "$INIT" ] || INIT=out/pretrain/final.pt
    python3 -u train.py --mode sft \
        --data "$DATADIR" --out out/sft --init "$INIT" \
        --seq-len 512 --micro-batch 32 --grad-accum 2 \
        --lr 1e-4 --max-steps 1200 --time-budget "$SFT_BUDGET" \
        --probe-steps 0 --save-every 150 --log-every 50 \
        2>&1 | tee -a logs/sft.log
fi

# ---- 8. 导出 GGUF ----
if [ ! -f out/tiny-zh-64m-f16.gguf ]; then
    log "===== 导出 GGUF ====="
    STATE=out/sft/best.pt
    [ -f "$STATE" ] || STATE=out/sft/final.pt
    [ -f "$STATE" ] || STATE=out/pretrain/best.pt
    log "用 $STATE"
    python3 export_gguf.py --state "$STATE" --tokenizer tokenizer \
        --out out/tiny-zh-64m-f16.gguf --outtype f16 --name tiny-zh-64m \
        2>&1 | tail -6 | tee -a logs/export.log
fi

# ---- 9. 汇总 ----
log "===== 完成 ====="
ls -lh out/*.gguf 2>/dev/null | tee -a logs/main.log
python3 - <<'PY' 2>&1 | tee -a logs/main.log
import json, os
for tag, p in (("预训练", "out/pretrain/train_log.jsonl"),
               ("SFT", "out/sft/train_log.jsonl")):
    if not os.path.exists(p):
        continue
    vals = [json.loads(l) for l in open(p) if '"val_loss"' in l]
    if vals:
        best = min(v["val_loss"] for v in vals)
        print(f"{tag} best val_loss = {best:.4f}  (PPL {2.718281828**best:.1f})")
PY
log "产物在 $WORK/out/"