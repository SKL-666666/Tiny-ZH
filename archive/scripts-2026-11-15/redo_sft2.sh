#!/usr/bin/env bash
# 重做高质量 SFT（从干净预训练权重开始）
set -u
WORK=/mnt/workspace/tiny-zh-64m
cd "$WORK" || exit 1
export PYTHONIOENCODING=utf-8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=16

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a logs/redo_sft.log; }

log "===== 重做高质量 SFT ====="

# 环境
python3 - <<'PY' 2>&1 | tee -a logs/redo_sft.log
import sys, torch
print(f"torch {torch.__version__}  cuda {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    print("FATAL"); sys.exit(1)
p = torch.cuda.get_device_properties(0)
print(f"dev {p.name} {p.total_memory/1e9:.1f}GB sm {p.multi_processor_count}")
print(f"bf16 {torch.cuda.is_bf16_supported()}")
import os
print(f"cpu {os.cpu_count()}")
PY
[ $? -ne 0 ] && { log "[FATAL] 环境检查失败"; exit 1; }

# 预训练权重
INIT=out/pretrain/state_only.pt
[ -f "$INIT" ] || INIT=out/pretrain/best.pt
[ -f "$INIT" ] || { log "[FATAL] 没有预训练权重"; exit 1; }
log "用 $INIT ($(du -h $INIT | cut -f1))"

# 编码
if [ ! -f bin/sft_high_train.bin ]; then
    log "编码 SFT 数据"
    python3 -u build_dataset.py --mode sft \
        --tokenizer tokenizer/tokenizer.json \
        --src data/sft_high.jsonl --out-dir bin --val-ratio 0.01 --max-len 512 \
        2>&1 | tail -4 | tee -a logs/redo_sft.log
fi

# 重命名供 train.py 读取
for s in train val; do
    [ -f "bin/sft_high_${s}.bin" ] && mv -f "bin/sft_high_${s}.bin" "bin/sft_${s}.bin"
    [ -f "bin/sft_high_${s}_idx.npy" ] && mv -f "bin/sft_high_${s}_idx.npy" "bin/sft_${s}_idx.npy"
done

# 内存盘
mkdir -p /dev/shm/tzsft/bin /dev/shm/tzsft/tokenizer
cp -f bin/sft_*.bin bin/sft_*.npy /dev/shm/tzsft/bin/ 2>/dev/null
cp -f tokenizer/* /dev/shm/tzsft/tokenizer/ 2>/dev/null
DATADIR=/dev/shm/tzsft/bin
[ -f "$DATADIR/sft_train.bin" ] || DATADIR=bin
log "数据 $DATADIR ($(ls -lh $DATADIR/sft_train.bin | awk '{print $5}'))"

# 清空旧 SFT
rm -rf out/sft_high
mkdir -p out/sft_high

# 训练：3 epoch, lr 8e-5（比今天的 1e-4 低，减少分布冲击）
log "训练 SFT"
python3 -u train.py --mode sft \
    --data "$DATADIR" --out out/sft_high --init "$INIT" \
    --seq-len 512 --micro-batch 32 --grad-accum 2 \
    --lr 8e-5 --max-steps 0 --epochs 3 --time-budget 2400 \
    --probe-steps 30 --save-every 300 --log-every 50 \
    2>&1 | tee -a logs/sft_high.log

log "SFT 完成"
grep '\[val\]' logs/sft_high.log | tail -3 | tee -a logs/redo_sft.log

# 导出
log "导出 f16"
STATE=out/sft_high/best.pt
[ -f "$STATE" ] || STATE=out/sft_high/final.pt
[ -f "$STATE" ] || { log "[FATAL] 没有 SFT 权重"; exit 1; }
python3 export_gguf.py --state "$STATE" --tokenizer tokenizer \
    --out out/tiny-zh-64m-v2-f16.gguf --outtype f16 --name tiny-zh-64m-v2 \
    2>&1 | tail -3 | tee -a logs/redo_sft.log

# 量化
log "量化 Q4_K_M"
python3 - <<'PY' 2>&1 | tail -4 | tee -a logs/redo_sft.log
import os
try:
    import llama_cpp.llama_cpp as lc
    params = lc.llama_model_quantize_default_params()
    params.ftype = 12
    params.nthread = 16
    rc = lc.llama_model_quantize(b"out/tiny-zh-64m-v2-f16.gguf",
                                 b"out/tiny-zh-64m-v2-q4km.gguf", params, None, None)
    print(f"  rc={rc}")
    if os.path.exists("out/tiny-zh-64m-v2-q4km.gguf"):
        print(f"  {os.path.getsize('out/tiny-zh-64m-v2-q4km.gguf')/1e6:.0f} MB")
except Exception as e:
    print(f"  量化失败: {e}")
PY

log "===== 完成 ====="
ls -lh out/*.gguf 2>/dev/null | tee -a logs/redo_sft.log