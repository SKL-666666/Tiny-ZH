#!/usr/bin/env bash
# 补装导出/量化依赖，然后完成 v2 导出
set -u
cd /mnt/workspace/tiny-zh-64m || exit 1
export PYTHONIOENCODING=utf-8

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a logs/export_v2.log; }

log "===== 1. 装 gguf ====="
pip install -q gguf 2>&1 | tail -2
pip install -q llama-cpp-python 2>&1 | tail -2
python3 -c "import gguf; print('  gguf OK')" 2>&1 | tail -1
python3 -c "import llama_cpp; print('  llama_cpp', llama_cpp.__version__)" 2>&1 | tail -1

log "===== 2. 导出 f16 ====="
STATE=out/sft_high/best.pt
[ -f "$STATE" ] || STATE=out/sft_high/final.pt
log "用 $STATE"

python3 export_gguf.py --state "$STATE" --tokenizer tokenizer \
    --out out/tiny-zh-64m-v2-f16.gguf --outtype f16 --name tiny-zh-64m-v2 \
    2>&1 | tail -5 | tee -a logs/export_v2.log

if [ ! -f out/tiny-zh-64m-v2-f16.gguf ]; then
    log "[FATAL] 导出失败"
    exit 1
fi
ls -lh out/tiny-zh-64m-v2-f16.gguf

log "===== 3. 量化 Q4_K_M ====="
python3 - <<'PY' 2>&1 | tee -a logs/export_v2.log
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
    import traceback; traceback.print_exc()
PY

log "===== 完成 ====="
ls -lh out/*v2*.gguf 2>/dev/null | tee -a logs/export_v2.log