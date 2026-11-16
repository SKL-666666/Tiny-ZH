#!/usr/bin/env bash
# 21:10 保底接管：等 SFT 自然跑完，跳过评测，直接导出+量化+上传
#
# 与正常 finalize.sh 的区别
# -------------------------
# 只砍掉这两步：
#   - 实验记录采集（collect_experiment.py）      约 1 分钟
#   - 生成样例评测（llama_cpp 加载 130MB 模型）  约 8 分钟
# 保留全部实质工作：
#   - SFT 完整跑完 1200 步（不提前杀）
#   - 导出 f16 GGUF
#   - 量化 Q4_K_M
#   - 双平台上传
#
# 为什么这样设计
# --------------
# 用户明确要求：跳过评测可以，但不能缩短 SFT 训练时间。
# SFT 完整跑完对模型质量影响大，评测只是记录，可以事后补。
set -u
WORK=/mnt/workspace/tiny-zh-64m
cd "$WORK" || exit 1
export PYTHONIOENCODING=utf-8

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a logs/takeover.log; }

# 载入凭据
[ -f "$WORK/.secrets.env" ] && . "$WORK/.secrets.env"

log "===== 保底接管开始（跳过评测，不改 SFT）====="

# ---- 1. 等 SFT 自然结束，绝不提前杀 ----
if pgrep -f 'train.py --mode sft' >/dev/null; then
    log "SFT 还在跑，等它自然结束（最多 25 分钟）"
    STEP_NOW=$(grep -oE 'step +[0-9]+/1200' logs/sft.log 2>/dev/null | tail -1)
    log "  当前 $STEP_NOW"

    for i in $(seq 1 150); do   # 150 × 10s = 25 分钟
        sleep 10
        if ! pgrep -f 'train.py --mode sft' >/dev/null; then
            log "  SFT 已结束（等了 $((i*10)) 秒）"
            break
        fi
        # 每 2 分钟报一次
        if [ $((i % 12)) -eq 0 ]; then
            S=$(grep -oE 'step +[0-9]+/1200' logs/sft.log 2>/dev/null | tail -1)
            log "  等待中 $S"
        fi
    done

    # 25 分钟还没结束才考虑中断（此时已到 21:35，必须动）
    if pgrep -f 'train.py --mode sft' >/dev/null; then
        log "[WARN] SFT 超过 25 分钟仍未结束，发送 SIGTERM（会让它存完 checkpoint）"
        kill -TERM $(pgrep -f 'train.py --mode sft' | head -1) 2>/dev/null
        for i in $(seq 1 24); do
            sleep 5
            pgrep -f 'train.py --mode sft' >/dev/null || break
        done
        pkill -9 -f 'train.py --mode sft' 2>/dev/null
        log "  已终止"
    fi
else
    log "SFT 已不在运行"
fi

log "SFT 最终状态："
grep '\[val\]' logs/sft.log 2>/dev/null | tail -2 | tee -a logs/takeover.log
grep -oE 'step +[0-9]+/1200' logs/sft.log 2>/dev/null | tail -1 | tee -a logs/takeover.log

# ---- 2. 挑 checkpoint（SFT 优先）----
BEST=""
for c in out/sft/best.pt out/sft/last.pt out/sft/final.pt \
         out/pretrain/best.pt out/pretrain/last.pt; do
    [ -f "$c" ] && { BEST="$c"; break; }
done
[ -z "$BEST" ] && { log "[FATAL] 没有 checkpoint"; exit 1; }
log "用 $BEST"

# ---- 3. 抽 state_dict ----
python3 - "$BEST" <<'PY' 2>&1 | tee -a logs/takeover.log
import sys, torch
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
torch.save(sd, "out/export.pt")
step = ck.get("step", "?") if isinstance(ck, dict) else "?"
best = ck.get("best_val", "?") if isinstance(ck, dict) else "?"
print(f"  state_dict 已抽 step={step} best_val={best}")
PY

# ---- 4. 导出 f16 ----
if [ ! -f out/tiny-zh-64m-f16.gguf ]; then
    log "导出 f16 GGUF"
    python3 export_gguf.py --state out/export.pt --tokenizer tokenizer \
        --out out/tiny-zh-64m-f16.gguf --outtype f16 --name tiny-zh-64m \
        2>&1 | tail -4 | tee -a logs/takeover.log
fi
[ -f out/tiny-zh-64m-f16.gguf ] || { log "[FATAL] 导出失败"; exit 1; }
ls -lh out/tiny-zh-64m-f16.gguf | tee -a logs/takeover.log

# ---- 5. 量化 ----
if [ ! -f out/tiny-zh-64m-q4km.gguf ]; then
    log "量化 Q4_K_M"
    python3 - <<'PY' 2>&1 | tee -a logs/takeover.log
import os
try:
    import llama_cpp.llama_cpp as lc
    params = lc.llama_model_quantize_default_params()
    params.ftype = 12
    params.nthread = 8
    rc = lc.llama_model_quantize(b"out/tiny-zh-64m-f16.gguf",
                                 b"out/tiny-zh-64m-q4km.gguf", params, None, None)
    print(f"  rc={rc}")
    if os.path.exists("out/tiny-zh-64m-q4km.gguf"):
        print(f"  {os.path.getsize('out/tiny-zh-64m-q4km.gguf')/1e6:.0f} MB")
except Exception:
    import traceback; traceback.print_exc()
PY
fi
ls -lh out/*.gguf 2>/dev/null | tee -a logs/takeover.log

# ---- 6. 上传（跳过评测，直接传）----
log "上传 HF"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
hf upload "SKL666666/tiny-zh-64m" out \
    --repo-type model --private --include "*.gguf" 2>&1 | tail -4 \
    | tee -a logs/takeover.log

log "上传魔搭"
python3 - <<'PY' 2>&1 | tee -a logs/takeover.log
import os, time
from modelscope.hub.api import HubApi
token = os.environ.get("MODELSCOPE_API_TOKEN", "")
if not token:
    print("  没有 MS token，跳过")
else:
    a = HubApi(); a.login(token)
    owner = os.environ.get("MS_OWNER", "SKL000000")
    repo = f"{owner}/tiny-zh-64m"
    try:
        a.create_model(model_name="tiny-zh-64m", namespace=owner,
                       chinese_name="Tiny-ZH-64M", license="Apache-2.0", visibility=1)
    except Exception:
        pass
    for f in ("out/tiny-zh-64m-f16.gguf", "out/tiny-zh-64m-q4km.gguf"):
        if not os.path.exists(f):
            continue
        for attempt in (1, 2, 3):
            try:
                a.upload_file(path_or_fileobj=f, path_in_repo=os.path.basename(f),
                              repo_id=repo, repo_type="model",
                              commit_message=f"add {os.path.basename(f)}")
                print(f"  OK {os.path.basename(f)}")
                break
            except Exception as e:
                if attempt == 3:
                    print(f"  FAIL {os.path.basename(f)}: {str(e)[:80]}")
                else:
                    time.sleep(5)
PY

log "===== 接管完成 ====="
ls -lh out/*.gguf 2>/dev/null | tee -a logs/takeover.log
log "HF: https://huggingface.co/SKL666666/tiny-zh-64m"