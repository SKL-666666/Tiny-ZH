#!/usr/bin/env bash
# v2 善后：上传双平台 + 生成实验记录
set -u
cd /mnt/workspace/tiny-zh-64m || exit 1
export PYTHONIOENCODING=utf-8

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a logs/v2_upload.log; }

# 凭据（新实例没有 .secrets.env，用环境变量传入）
HF_TOKEN="${HF_TOKEN:-}"
MODELSCOPE_API_TOKEN="${MODELSCOPE_API_TOKEN:-}"
export HF_TOKEN MODELSCOPE_API_TOKEN
export HF_ENDPOINT=https://hf-mirror.com

log "===== 1. 上传 HF ====="
if [ -n "$HF_TOKEN" ]; then
    mkdir -p out_v2
    cp -f out/tiny-zh-64m-v2-f16.gguf out/tiny-zh-64m-v2-q4km.gguf out_v2/ 2>/dev/null
    cp -f out/sft_high/best.pt out_v2/sft_high_best.pt 2>/dev/null
    cp -f tokenizer/tokenizer.json out_v2/ 2>/dev/null
    cp -f logs/compare.log out_v2/compare_v1_v2.txt 2>/dev/null
    cp -f logs/compare_v1_v2.json out_v2/ 2>/dev/null
    ls -lh out_v2/ | tee -a logs/v2_upload.log

    hf upload "SKL666666/tiny-zh-64m" out_v2 \
        --repo-type model --private 2>&1 | tail -5 | tee -a logs/v2_upload.log
else
    log "[skip] 没有 HF_TOKEN"
fi

log "===== 2. 上传魔搭 ====="
if [ -n "$MODELSCOPE_API_TOKEN" ]; then
    python3 - <<'PY' 2>&1 | tee -a logs/v2_upload.log
import os, time
from modelscope.hub.api import HubApi
token = os.environ.get("MODELSCOPE_API_TOKEN", "")
a = HubApi(); a.login(token)
repo = "SKL000000/tiny-zh-64m"
files = ["out_v2/tiny-zh-64m-v2-f16.gguf",
         "out_v2/tiny-zh-64m-v2-q4km.gguf",
         "out_v2/sft_high_best.pt",
         "out_v2/compare.log"]
for f in files:
    if not os.path.exists(f):
        continue
    for attempt in (1, 2, 3):
        try:
            t0 = time.time()
            a.upload_file(path_or_fileobj=f, path_in_repo=os.path.basename(f),
                          repo_id=repo, repo_type="model",
                          commit_message=f"v2 {os.path.basename(f)}")
            print(f"  OK {os.path.basename(f)} {time.time()-t0:.0f}s")
            break
        except Exception as e:
            if attempt == 3:
                print(f"  FAIL {os.path.basename(f)} {str(e)[:80]}")
            else:
                time.sleep(4)
PY
else
    log "[skip] 没有魔搭 token"
fi

log "===== 3. 采集实验记录 ====="
python3 collect_experiment.py 2>&1 | tail -3 | tee -a logs/v2_upload.log

log "===== 完成 ====="
ls -lh out/*v2*.gguf 2>/dev/null | tee -a logs/v2_upload.log