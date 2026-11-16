#!/usr/bin/env bash
# 紧急救场：用 minimind 的 SFT 数据重训（31.5M 那次成功用的就是它）
#
# 为什么换数据
# ------------
# ShareGPT 混了大量「列出1234」的任务指令，模型学会了编号格式，
# 导致「推荐一本好书」输出 [1][2][3]。
# minimind 的 sft_t2t_mini 是 qwen3 蒸馏的对话数据，格式规整，
# 31.5M 用它 SFT 后能稳定输出自然对话。
set -u
WORK=/mnt/workspace/tiny-zh-64m
cd "$WORK" || exit 1
export PYTHONIOENCODING=utf-8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a logs/rescue.log; }

log "===== 紧急救场开始 ====="

# ---- 1. 下载 minimind SFT ----
if [ ! -f data/raw/minimind_sft.jsonl ]; then
    log "下载 minimind sft_t2t_mini"
    python3 - <<'PY' 2>&1 | tail -3 | tee -a logs/rescue.log
import os, shutil
from modelscope.hub.file_download import dataset_file_download
try:
    p = dataset_file_download(dataset_id="gongjy/minimind_dataset",
                              file_path="sft_t2t_mini.jsonl", local_dir="data/raw")
    dst = "data/raw/minimind_sft.jsonl"
    if p and os.path.abspath(p) != os.path.abspath(dst):
        shutil.move(p, dst)
    print(f"  OK {os.path.getsize(dst)/1e6:.0f} MB")
except Exception as e:
    print(f"  FAIL {str(e)[:120]}")
PY
fi

# ---- 2. 转成我们的格式 ----
log "转换格式"
python3 - <<'PY' 2>&1 | tail -6 | tee -a logs/rescue.log
import json, os

src = "data/raw/minimind_sft.jsonl"
out = "data/sft_rescue.jsonl"
if not os.path.exists(src):
    print("[FATAL] 没有 minimind 数据")
    raise SystemExit(1)

n = 0
with open(src, encoding="utf-8", errors="ignore") as f, \
     open(out, "w", encoding="utf-8") as fo:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            o = json.loads(line)
        except json.JSONDecodeError:
            continue
        conv = o.get("conversations") or o.get("messages") or []
        if len(conv) < 2:
            continue
        msgs = []
        for m in conv:
            if not isinstance(m, dict):
                continue
            role = m.get("role") or m.get("from")
            content = m.get("content") or m.get("value") or ""
            if role in ("user", "human"):
                msgs.append({"role": "user", "content": content})
            elif role in ("assistant", "gpt"):
                msgs.append({"role": "assistant", "content": content})
        if len(msgs) < 2 or msgs[0]["role"] != "user" or msgs[-1]["role"] != "assistant":
            continue
        # 过滤：总长 20-2000，中文占比 > 0.5
        total = sum(len(m["content"]) for m in msgs)
        if total < 20 or total > 2000:
            continue
        zh = sum(1 for m in msgs for c in m["content"] if "\u4e00" <= c <= "\u9fff")
        if zh / max(total, 1) < 0.5:
            continue
        fo.write(json.dumps({"conversations": msgs}, ensure_ascii=False) + "\n")
        n += 1

print(f"  转换 {n:,} 条 -> {out}")
print(f"  大小 {os.path.getsize(out)/1e6:.0f} MB")
PY

# ---- 3. 编码 ----
log "编码 SFT"
rm -f bin/sft_rescue*.bin bin/sft_rescue*_idx.npy
python3 -u build_dataset.py --mode sft \
    --tokenizer tokenizer/tokenizer.json \
    --src data/sft_rescue.jsonl --out-dir bin --val-ratio 0.01 --max-len 512 \
    2>&1 | tail -4 | tee -a logs/rescue.log

# 重命名成 sft_* 供 train.py 读取
mv -f bin/sft_rescue_train.bin bin/sft_train.bin 2>/dev/null
mv -f bin/sft_rescue_val.bin bin/sft_val.bin 2>/dev/null
mv -f bin/sft_rescue_train_idx.npy bin/sft_train_idx.npy 2>/dev/null
mv -f bin/sft_rescue_val_idx.npy bin/sft_val_idx.npy 2>/dev/null
cp -f bin/sft_*.bin bin/sft_*.npy /dev/shm/tz64/bin/ 2>/dev/null
log "编码完成，已复制内存盘"

# ---- 4. 重训 SFT（从预训练权重开始，不用旧 SFT 权重）----
log "重训 SFT（800 步，约 13 分钟）"
rm -rf out/sft
mkdir -p out/sft

# 预训练权重
INIT=""
for c in out/pretrain/best.pt out/pretrain/last.pt out/pretrain/final.pt; do
    [ -f "$c" ] && { INIT="$c"; break; }
done
[ -z "$INIT" ] && { log "[FATAL] 没有预训练权重"; exit 1; }

python3 - "$INIT" <<'PY' 2>&1 | tee -a logs/rescue.log
import sys, torch
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
torch.save(sd, "out/pretrain/for_sft.pt")
print(f"  从 {sys.argv[1]} 抽 state_dict")
PY

DATADIR=bin
[ -f /dev/shm/tz64/bin/sft_train.bin ] && DATADIR=/dev/shm/tz64/bin

python3 -u train.py --mode sft \
    --data "$DATADIR" --out out/sft --init out/pretrain/for_sft.pt \
    --seq-len 512 --micro-batch 32 --grad-accum 2 \
    --lr 1e-4 --max-steps 800 --time-budget 1000 \
    --probe-steps 0 --save-every 200 --log-every 50 \
    2>&1 | tee -a logs/sft_rescue.log

log "SFT 完成"

# ---- 5. 重新导出 ----
log "重新导出 f16"
STATE=out/sft/best.pt
[ -f "$STATE" ] || STATE=out/sft/final.pt
python3 export_gguf.py --state "$STATE" --tokenizer tokenizer \
    --out out/tiny-zh-64m-f16.gguf --outtype f16 --name tiny-zh-64m \
    2>&1 | tail -3 | tee -a logs/rescue.log

# ---- 6. 重新量化 ----
log "重新量化"
python3 - <<'PY' 2>&1 | tail -4 | tee -a logs/rescue.log
import os
try:
    import llama_cpp.llama_cpp as lc
    params = lc.llama_model_quantize_default_params()
    params.ftype = 12
    params.nthread = 8
    rc = lc.llama_model_quantize(b"out/tiny-zh-64m-f16.gguf",
                                 b"out/tiny-zh-64m-q4km.gguf", params, None, None)
    print(f"  rc={rc} {os.path.getsize('out/tiny-zh-64m-q4km.gguf')/1e6:.0f} MB")
except Exception as e:
    print(f"  量化失败 {e}")
PY

# ---- 7. 立刻上传 ----
log "上传"
. "$WORK/.secrets.env" 2>/dev/null
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
mkdir -p out_gguf && cp -f out/*.gguf out_gguf/
hf upload "SKL666666/tiny-zh-64m" out_gguf --repo-type model --private 2>&1 | tail -3 | tee -a logs/rescue.log

python3 - <<'PY' 2>&1 | tail -4 | tee -a logs/rescue.log
import os, time
from modelscope.hub.api import HubApi
token = os.environ.get("MODELSCOPE_API_TOKEN", "")
if token:
    a = HubApi(); a.login(token)
    for f in ("out/tiny-zh-64m-f16.gguf", "out/tiny-zh-64m-q4km.gguf"):
        if not os.path.exists(f):
            continue
        try:
            a.upload_file(path_or_fileobj=f, path_in_repo=os.path.basename(f),
                          repo_id="SKL000000/tiny-zh-64m", repo_type="model",
                          commit_message="rescue rebuild")
            print(f"  魔搭 OK {os.path.basename(f)}")
        except Exception as e:
            print(f"  魔搭 FAIL {str(e)[:70]}")
PY

log "===== 救场完成 ====="
ls -lh out/*.gguf