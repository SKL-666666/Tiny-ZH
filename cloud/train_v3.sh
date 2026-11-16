#!/usr/bin/env bash
# v3：用清理过身份标签的数据重训
set -u
cd /mnt/workspace/tiny-zh-64m || exit 1
export PYTHONIOENCODING=utf-8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=16

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a logs/v3.log; }

log "===== v3 训练（身份标签已清理）====="

# 编码
if [ ! -f bin/sft_clean_train.bin ]; then
    log "编码"
    python3 -u build_dataset.py --mode sft \
        --tokenizer tokenizer/tokenizer.json \
        --src data/sft_clean.jsonl --out-dir bin --val-ratio 0.01 --max-len 512 \
        2>&1 | tail -3 | tee -a logs/v3.log
fi

# 重命名
for s in train val; do
    [ -f "bin/sft_clean_${s}.bin" ] && mv -f "bin/sft_clean_${s}.bin" "bin/sft_${s}.bin"
    [ -f "bin/sft_clean_${s}_idx.npy" ] && mv -f "bin/sft_clean_${s}_idx.npy" "bin/sft_${s}_idx.npy"
done

mkdir -p /dev/shm/tz3/bin
cp -f bin/sft_*.bin bin/sft_*.npy /dev/shm/tz3/bin/ 2>/dev/null
DATADIR=/dev/shm/tz3/bin
[ -f "$DATADIR/sft_train.bin" ] || DATADIR=bin

rm -rf out/sft_v3 && mkdir -p out/sft_v3

log "训练 3375 步"
python3 -u train.py --mode sft \
    --data "$DATADIR" --out out/sft_v3 --init out/pretrain/state_only.pt \
    --seq-len 512 --micro-batch 32 --grad-accum 2 \
    --lr 8e-5 --max-steps 3375 \
    --probe-steps 0 --save-every 300 --log-every 50 \
    2>&1 | tee -a logs/sft_v3.log

log "训练完成 $(grep -o 'best_val=[0-9.]*' logs/sft_v3.log | tail -1)"

# 导出
log "导出"
STATE=out/sft_v3/best.pt
[ -f "$STATE" ] || STATE=out/sft_v3/final.pt
python3 export_gguf.py --state "$STATE" --tokenizer tokenizer \
    --out out/tiny-zh-64m-v3-f16.gguf --outtype f16 --name tiny-zh-64m-v3 \
    2>&1 | tail -3 | tee -a logs/v3.log

log "量化"
python3 - <<'PY' 2>&1 | tail -3 | tee -a logs/v3.log
import os
try:
    import llama_cpp.llama_cpp as lc
    params = lc.llama_model_quantize_default_params()
    params.ftype = 12; params.nthread = 16
    rc = lc.llama_model_quantize(b"out/tiny-zh-64m-v3-f16.gguf",
                                 b"out/tiny-zh-64m-v3-q4km.gguf", params, None, None)
    print(f"  rc={rc} {os.path.getsize('out/tiny-zh-64m-v3-q4km.gguf')/1e6:.0f}MB")
except Exception as e:
    print(f"  失败 {e}")
PY

log "===== v3 完成 ====="
ls -lh out/*v3*.gguf 2>/dev/null