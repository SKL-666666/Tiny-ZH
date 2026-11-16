#!/usr/bin/env bash
# 上传预训练权重（明天重做 SFT 的基础）+ 完整核验
#
# 为什么必须先传预训练权重
# ------------------------
# 明天新实例没有今天的训练结果。重做 SFT 必须从同一个预训练权重开始，
# 否则模型完全不同。pretrain/last.pt 是 778MB（含优化器），
# 但重做 SFT 只需要 state_dict（259MB）。
set -u
WORK=/mnt/workspace/tiny-zh-64m
cd "$WORK" || exit 1
[ -f "$WORK/.secrets.env" ] && . "$WORK/.secrets.env"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a logs/final_upload.log; }

log "===== 最终上传：预训练权重 ====="

# ---- 1. 抽纯 state_dict（去掉优化器，体积减 2/3）----
if [ ! -f out/pretrain/state_only.pt ]; then
    python3 - <<'PY' 2>&1 | tee -a logs/final_upload.log
import os
import torch
src = None
for c in ("out/pretrain/best.pt", "out/pretrain/last.pt", "out/pretrain/final.pt"):
    if os.path.exists(c):
        src = c
        break
if not src:
    print("[FATAL] 没有预训练权重")
    raise SystemExit(1)
ck = torch.load(src, map_location="cpu", weights_only=False)
sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
torch.save(sd, "out/pretrain/state_only.pt")
step = ck.get("step", "?") if isinstance(ck, dict) else "?"
best = ck.get("best_val", "?") if isinstance(ck, dict) else "?"
print(f"  从 {src} 抽 state_dict")
print(f"  step={step} best_val={best}")
print(f"  {os.path.getsize('out/pretrain/state_only.pt')/1e6:.0f} MB")
PY
fi

# ---- 2. 上传到 HF ----
log "HF: 上传预训练权重"
mkdir -p out_pretrain
cp -f out/pretrain/state_only.pt out_pretrain/pretrain_state.pt 2>/dev/null
cp -f out/sft/best.pt out_pretrain/sft_best.pt 2>/dev/null
cp -f tokenizer/tokenizer.json out_pretrain/ 2>/dev/null
cp -f config.py out_pretrain/ 2>/dev/null
ls -lh out_pretrain/ | tee -a logs/final_upload.log

hf upload "SKL666666/tiny-zh-64m" out_pretrain \
    --repo-type model --private 2>&1 | tail -5 | tee -a logs/final_upload.log

# ---- 3. 上传到魔搭 ----
log "魔搭: 上传预训练权重"
python3 - <<'PY' 2>&1 | tee -a logs/final_upload.log
import os, time
from modelscope.hub.api import HubApi
token = os.environ.get("MODELSCOPE_API_TOKEN", "")
if not token:
    print("  没有 token，跳过")
    raise SystemExit(0)
a = HubApi(); a.login(token)
repo = "SKL000000/tiny-zh-64m"
for f in ("out_pretrain/pretrain_state.pt", "out_pretrain/sft_best.pt",
          "out_pretrain/tokenizer.json", "out_pretrain/config.py"):
    if not os.path.exists(f):
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
                print(f"  FAIL {os.path.basename(f)} {str(e)[:80]}")
            else:
                time.sleep(4)
PY

# ---- 4. 最终核验清单 ----
log "===== 核验清单 ====="
{
echo "## 已交付产物核验"
echo ""
echo "| 文件 | 大小 | 位置 |"
echo "|---|---|---|"
for f in out/tiny-zh-64m-f16.gguf out/tiny-zh-64m-q4km.gguf \
         out/pretrain/state_only.pt out/sft/best.pt tokenizer/tokenizer.json; do
    if [ -f "$f" ]; then
        sz=$(du -h "$f" | cut -f1)
        echo "| $(basename $f) | $sz | HF + 魔搭 |"
    fi
done
echo ""
echo "## 训练统计"
echo ""
python3 - <<'PY'
import json, os
p = "out/pretrain/train_log.jsonl"
if os.path.exists(p):
    rows = [json.loads(l) for l in open(p) if '"val_loss"' in l]
    if rows:
        best = min(r["val_loss"] for r in rows)
        print(f"- 预训练 best val_loss: {best:.4f} (PPL {2.718**best:.1f})")
        print(f"- 预训练步数: {rows[-1]['step']:,}")
        tok = rows[-1]['step'] * 32768
        print(f"- 训练 token: {tok:,} = {tok/1e8:.2f} 亿 = {tok/64816000:.1f}x 参数")
q = "out/sft/train_log.jsonl"
if os.path.exists(q):
    rows = [json.loads(l) for l in open(q) if '"val_loss"' in l]
    if rows:
        best = min(r["val_loss"] for r in rows)
        print(f"- SFT best val_loss: {best:.4f} (PPL {2.718**best:.1f})")
        print(f"- SFT 步数: {rows[-1]['step']:,}")
PY
} | tee -a logs/final_upload.log

log "===== 全部完成 ====="