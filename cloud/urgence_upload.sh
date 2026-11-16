#!/usr/bin/env bash
# 紧急并行上传：只传 GGUF + 权重 + 日志，跳过所有评测
#
# 为什么单独写
# ------------
# finalize.sh 里评测和上传是串行的，评测（llama_cpp 加载 132MB 模型跑 5 个 prompt）
# 要 8-10 分钟。21:30 关机，没这个时间。
# 这个脚本只做上传，和 finalize.sh 并行跑互不干扰。
set -u
WORK=/mnt/workspace/tiny-zh-64m
cd "$WORK" || exit 1
[ -f "$WORK/.secrets.env" ] && . "$WORK/.secrets.env"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a logs/urgence_upload.log; }

log "===== 紧急并行上传 ====="

# ---- 1. HF 传 GGUF（最重要，优先）----
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
if [ -n "${HF_TOKEN:-}" ]; then
    log "HF: 上传 GGUF"
    mkdir -p out_gguf
    cp -f out/*.gguf out_gguf/ 2>/dev/null
    hf upload "SKL666666/tiny-zh-64m" out_gguf \
        --repo-type model --private 2>&1 | tail -6 | tee -a logs/urgence_upload.log
    log "HF: GGUF 完成"
else
    log "[skip] 没有 HF_TOKEN"
fi

# ---- 2. 魔搭传 GGUF ----
if [ -n "${MODELSCOPE_API_TOKEN:-}" ]; then
    log "魔搭: 上传 GGUF"
    python3 - <<'PY' 2>&1 | tee -a logs/urgence_upload.log
import os, time
from modelscope.hub.api import HubApi
token = os.environ.get("MODELSCOPE_API_TOKEN", "")
owner = os.environ.get("MS_OWNER", "SKL000000")
a = HubApi(); a.login(token)
repo = f"{owner}/tiny-zh-64m"
try:
    a.create_model(model_name="tiny-zh-64m", namespace=owner,
                   chinese_name="Tiny-ZH-64M 中文小模型",
                   license="Apache-2.0", visibility=1)
    print("  创建模型仓库")
except Exception as e:
    print(f"  创建: {str(e)[:80]}")
for f in ("out/tiny-zh-64m-f16.gguf", "out/tiny-zh-64m-q4km.gguf"):
    if not os.path.exists(f):
        print(f"  缺 {f}")
        continue
    for attempt in (1, 2, 3):
        try:
            t0 = time.time()
            a.upload_file(path_or_fileobj=f, path_in_repo=os.path.basename(f),
                          repo_id=repo, repo_type="model",
                          commit_message=f"add {os.path.basename(f)}")
            print(f"  OK {os.path.basename(f)} {time.time()-t0:.0f}s")
            break
        except Exception as e:
            if attempt == 3:
                print(f"  FAIL {os.path.basename(f)}: {str(e)[:90]}")
            else:
                time.sleep(5)
PY
else
    log "[skip] 没有魔搭 token"
fi

# ---- 3. HF 传权重（GGUF 传完再传，体积大）----
if [ -n "${HF_TOKEN:-}" ]; then
    log "HF: 上传模型权重"
    mkdir -p out_models
    cp -f out/sft/best.pt out_models/ 2>/dev/null
    cp -f out/pretrain/last.pt out_models/pretrain_last.pt 2>/dev/null
    cp -f tokenizer/tokenizer.json out_models/ 2>/dev/null
    hf upload "SKL666666/tiny-zh-64m" out_models \
        --repo-type model --private 2>&1 | tail -5 | tee -a logs/urgence_upload.log
fi

# ---- 4. HF 传日志（论文材料）----
if [ -n "${HF_TOKEN:-}" ]; then
    log "HF: 上传日志"
    mkdir -p logs_up
    cp -f logs/*.log logs/*.md logs/*.jsonl logs/*.json logs_up/ 2>/dev/null
    hf upload "SKL666666/tiny-zh-64m-logs" logs_up \
        --repo-type dataset --private 2>&1 | tail -5 | tee -a logs/urgence_upload.log
fi

log "===== 上传完成 ====="
echo "HF 模型: https://huggingface.co/SKL666666/tiny-zh-64m" | tee -a logs/urgence_upload.log
echo "HF 日志: https://huggingface.co/datasets/SKL666666/tiny-zh-64m-logs" | tee -a logs/urgence_upload.log
echo "魔搭模型: https://www.modelscope.cn/models/SKL000000/tiny-zh-64m" | tee -a logs/urgence_upload.log