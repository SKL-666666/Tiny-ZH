#!/usr/bin/env bash
# 明天新实例：从 HF 拉预训练权重，重做高质量 SFT
#
# 今天的教训（全部修入本脚本）
# ---------------------------
# 1. ShareGPT 混入大量「列出1234」任务指令 -> 模型输出 [1][2][3]
#    修：改用 minimind sft_t2t_mini（qwen3 蒸馏，格式规整）+ 严格过滤编号列表
# 2. 答案长度无限制 -> 模型不知何时该短，问天气只回 6 字复读
#    修：强制 70-200 字
# 3. 缺短问答 -> 简单问题只能复读
#    修：加 15% 的 10-60 字短问答
# 4. 样本仅 5.7 万，3 个 epoch 就过拟合
#    修：扩到 15 万条
#
# 用法：
#   bash cloud/redo_sft.sh
set -u
WORK=/mnt/workspace/tiny-zh-sft
mkdir -p "$WORK"/{data/raw,bin,tokenizer,logs,out/sft,out_base}
cd "$WORK" || exit 1
export PYTHONIOENCODING=utf-8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a logs/redo.log; }

log "===== 重做高质量 SFT ====="

# ---- 0. 环境检查 ----
python3 - <<'PY' 2>&1 | tee -a logs/redo.log
import sys
try:
    import torch
    print(f"torch {torch.__version__}  cuda {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("FATAL: GPU 不可用"); sys.exit(1)
    p = torch.cuda.get_device_properties(0)
    print(f"dev {p.name}  {p.total_memory/1e9:.1f}GB")
except ImportError:
    print("FATAL: torch 不可用"); sys.exit(1)
PY || exit 1

# ---- 1. 拉代码和预训练权重 ----
log "下载代码"
if [ ! -f train.py ]; then
    cd /mnt/workspace
    curl -fsSL -o tz.tar.gz "https://codeload.github.com/SKL-666666/Tiny-ZH-32M/tar.gz/refs/heads/master"
    tar xzf tz.tar.gz && mv Tiny-ZH-32M-master tzsrc
    cp tzsrc/*.py "$WORK"/ && mkdir -p "$WORK/cloud" && cp tzsrc/cloud/*.sh "$WORK/cloud/"
    cd "$WORK" || exit 1
    log "  代码就位"
fi

# 预训练权重（从 HF 拉）
if [ ! -f out_base/pretrain_state.pt ]; then
    log "下载预训练权重"
    export HF_ENDPOINT=https://hf-mirror.com
    python3 - <<'PY' 2>&1 | tail -5 | tee -a logs/redo.log
import os
from huggingface_hub import hf_hub_download
try:
    for fn in ("pretrain_state.pt", "tokenizer.json", "config.py"):
        p = hf_hub_download(repo_id="SKL666666/tiny-zh-64m", filename=fn,
                            repo_type="model",
                            local_dir="out_base")
        print(f"  OK {fn} -> {p}")
except Exception as e:
    print(f"  FAIL {str(e)[:150]}")
PY
fi

ls -lh out_base/ | tee -a logs/redo.log

# 词表就位
cp -f out_base/tokenizer.json tokenizer/ 2>/dev/null
cp -f out_base/config.py . 2>/dev/null || true

# ---- 2. 下载 SFT 数据（全部换新，不用 ShareGPT）----
log "下载 SFT 数据"
python3 - <<'PY' 2>&1 | tail -8 | tee -a logs/redo.log
import os, shutil
from modelscope.hub.file_download import dataset_file_download

DST = "data/raw"
os.makedirs(DST, exist_ok=True)

jobs = [
    # minimind SFT：31.5M 那次成功用的就是它，格式规整
    ("gongjy/minimind_dataset", "sft_t2t_mini.jsonl", "minimind_sft.jsonl"),
    # Firefly：165 万条指令，补多样性
    ("zhuangxialie/SFT-Chinese-Dataset",
     "firely-train/firefly-train-1.1M.jsonl", "firefly.jsonl"),
]
for ds, remote, local in jobs:
    out = os.path.join(DST, local)
    if os.path.exists(out) and os.path.getsize(out) > 10 << 20:
        print(f"  [cached] {local}")
        continue
    try:
        p = dataset_file_download(dataset_id=ds, file_path=remote, local_dir=DST)
        if p and os.path.abspath(p) != os.path.abspath(out):
            shutil.move(p, out)
        print(f"  [got] {local} {os.path.getsize(out)/1e6:.0f}MB")
    except Exception as e:
        print(f"  [FAIL] {local}: {str(e)[:100]}")
PY

# ---- 3. 构建高质量 SFT 数据 ----
log "构建 SFT 数据（70-200 字 + 过滤编号列表 + 加短问答）"
python3 - <<'PY' 2>&1 | tee -a logs/redo.log
import json, os, re
from collections import Counter

RAW = "data/raw"
OUT = "data/sft_high.jsonl"

RE_NUMLIST = re.compile(r"^\s*[\[\(【（]?\d{1,2}[\]\)】）]?\s*[\.、:：]?\s*$", re.M)
RE_CJK = re.compile(r"[\u4e00-\u9fff]")
RE_URL = re.compile(r"(?:https?://|www\.)\S+")

def clean(t):
    t = RE_URL.sub("", t or "")
    t = re.sub(r"<[^>]{1,200}>", "", t)
    t = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", t)
    return t.strip()

def has_numlist(a):
    """检测编号列表：连续出现 3 个以上 [1] [2] [3] 或 1. 2. 3."""
    lines = [l.strip() for l in a.split("\n") if l.strip()]
    if len(lines) < 3:
        return False
    numlike = sum(1 for l in lines if RE_NUMLIST.match(l))
    return numlike / len(lines) > 0.4

def quality(a, lo, hi):
    n = len(a)
    if n < lo or n > hi:
        return False
    if len(RE_CJK.findall(a)) / n < 0.55:
        return False
    if Counter(a).most_common(1)[0][1] / n > 0.25:
        return False
    if has_numlist(a):
        return False
    if a[-1] not in "。！？…」』）.!?":
        return False
    return True

def read_minimind(p):
    with open(p, encoding="utf-8", errors="ignore") as f:
        for line in f:
            try:
                o = json.loads(line.strip())
            except Exception:
                continue
            conv = o.get("conversations") or []
            msgs = []
            for m in conv:
                if not isinstance(m, dict):
                    continue
                r = m.get("role") or m.get("from")
                c = m.get("content") or m.get("value") or ""
                if r in ("user", "human"):
                    msgs.append(("user", c))
                elif r in ("assistant", "gpt"):
                    msgs.append(("assistant", c))
            if len(msgs) >= 2 and msgs[0][0] == "user":
                yield msgs

def read_firefly(p):
    with open(p, encoding="utf-8", errors="ignore") as f:
        for line in f:
            try:
                o = json.loads(line.strip())
            except Exception:
                continue
            q = (o.get("instruction") or "").strip()
            inp = (o.get("input") or "").strip()
            a = (o.get("output") or "").strip()
            if q and a:
                yield [("user", f"{q}\n{inp}" if inp else q), ("assistant", a)]

seen = set()
n = 0
stat = Counter()
lens = {"long": [], "short": []}

with open(OUT, "w", encoding="utf-8") as fo:
    # 主体：70-200 字
    for tag, path, reader, lo, hi, cap in (
        ("minimind_long", f"{RAW}/minimind_sft.jsonl", read_minimind, 70, 200, 110000),
        ("firefly_long",  f"{RAW}/firefly.jsonl",      read_firefly,  70, 200, 40000),
        ("minimind_short", f"{RAW}/minimind_sft.jsonl", read_minimind, 10, 60, 25000),
    ):
        if not os.path.exists(path):
            print(f"  缺 {path}")
            continue
        got = 0
        for msgs in reader(path):
            q = clean(msgs[0][1])
            a = clean(msgs[-1][1])
            if not q or len(q) > 300:
                continue
            if not quality(a, lo, hi):
                continue
            h = hash(q[:48])
            if h in seen:
                continue
            seen.add(h)
            conv = [{"role": r, "content": clean(c)} for r, c in msgs]
            fo.write(json.dumps({"conversations": conv}, ensure_ascii=False) + "\n")
            got += 1
            n += 1
            (lens["short"] if hi <= 60 else lens["long"]).append(len(a))
            if got >= cap:
                break
        stat[tag] = got
        print(f"  {tag:<16} {got:>7,} 条")

print(f"\n总计 {n:,} 条 -> {OUT}")
if lens["long"]:
    v = sorted(lens["long"])
    print(f"  长答 p10={v[len(v)//10]} p50={v[len(v)//2]} p90={v[len(v)*9//10]}")
if lens["short"]:
    v = sorted(lens["short"])
    print(f"  短答 p10={v[len(v)//10]} p50={v[len(v)//2]} p90={v[len(v)*9//10]}")
for k, v in stat.most_common():
    print(f"    {k:<16} {v/max(n,1)*100:>5.1f}%")
print(f"  文件 {os.path.getsize(OUT)/1e6:.0f} MB")
PY

# ---- 4. 编码 ----
log "编码 SFT"
rm -f bin/sft_*.bin bin/sft_*_idx.npy
python3 -u build_dataset.py --mode sft \
    --tokenizer tokenizer/tokenizer.json \
    --src data/sft_high.jsonl --out-dir bin --val-ratio 0.01 --max-len 512 \
    2>&1 | tail -4 | tee -a logs/redo.log

# ---- 5. 训练 ----
log "训练 SFT（3 epoch，约 25 分钟）"
python3 -u train.py --mode sft \
    --data bin --out out/sft --init out_base/pretrain_state.pt \
    --seq-len 512 --micro-batch 32 --grad-accum 2 \
    --lr 8e-5 --max-steps 0 --epochs 3 --time-budget 2400 \
    --probe-steps 30 --save-every 300 --log-every 50 \
    2>&1 | tee -a logs/sft_high.log

# ---- 6. 导出 + 量化 ----
log "导出 f16"
STATE=out/sft/best.pt
[ -f "$STATE" ] || STATE=out/sft/final.pt
python3 export_gguf.py --state "$STATE" --tokenizer tokenizer \
    --out out/tiny-zh-64m-f16.gguf --outtype f16 --name tiny-zh-64m \
    2>&1 | tail -3 | tee -a logs/redo.log

log "量化 Q4_K_M"
python3 - <<'PY' 2>&1 | tail -4 | tee -a logs/redo.log
import os
try:
    import llama_cpp.llama_cpp as lc
    params = lc.llama_model_quantize_default_params()
    params.ftype = 12; params.nthread = 8
    rc = lc.llama_model_quantize(b"out/tiny-zh-64m-f16.gguf",
                                 b"out/tiny-zh-64m-q4km.gguf", params, None, None)
    print(f"  rc={rc} {os.path.getsize('out/tiny-zh-64m-q4km.gguf')/1e6:.0f}MB")
except Exception as e:
    print(f"  失败 {e}")
PY

log "===== 完成 ====="
ls -lh out/*.gguf