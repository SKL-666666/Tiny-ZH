#!/usr/bin/env bash
# 下载全部数据 —— 明天执行
#
# 顺序按"能最快拿到有用的东西"排：
# 1. 魔搭 4 个小文件（内网，几分钟）
# 2. FineWeb-v2 N 片（大文件，32 路并发）
# 3. LCCC（最后，因为前面下完就能开始清洗，LCCC 可以稍后）
#
# 每步都断点续传：已存在且大小正确的文件直接跳过，重跑本脚本安全。
set -u
WORK=/mnt/workspace/tiny-zh-64m
cd "$WORK" || exit 1
mkdir -p data/raw/fwv2 logs
export PYTHONIOENCODING=utf-8

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a logs/fetch.log; }

log "===== 下载全部数据 ====="

# ---- 决定下几片 v2 ----
NSHARD=2
if [ -f logs/fwv2_probe.json ]; then
    NSHARD=$(python3 -c "import json;print(json.load(open('logs/fwv2_probe.json'))['shards_recommend'])" 2>/dev/null || echo 2)
    log "按 verify_fwv2 结论下 $NSHARD 片"
else
    log "[WARN] 没跑 verify_fwv2.sh，按默认 2 片下"
fi

# ---- 1. 魔搭：ShareGPT + 维基 + COIG ----
log "下载魔搭数据（内网）"
python3 -u - <<'PY' 2>&1 | tee -a logs/fetch.log
import os, shutil
from modelscope.hub.file_download import dataset_file_download

DST = "data/raw"
os.makedirs(DST, exist_ok=True)

jobs = [
    # (dataset_id, remote_path, local_name, 最小期望字节)
    ("zhuangxialie/SFT-Chinese-Dataset",
     "shareAIShareGPT-Chinese-English-90k/common_zh_70k.jsonl",
     "common_zh_70k.jsonl", 400<<20),
    ("zhuangxialie/SFT-Chinese-Dataset",
     "shareAIShareGPT-Chinese-English-90k/unknow_zh_38k.jsonl",
     "unknow_zh_38k.jsonl", 100<<20),
    ("AI-ModelScope/wikipedia-cn-20230720-filtered",
     "wikipedia-cn-20230720-filtered.jsonl",
     "wiki.jsonl", 400<<20),
    ("AI-ModelScope/COIG-CQIA",
     "COIG-CQIA-full.jsonl",
     "coig_full.jsonl", 100<<20),
    ("zhuangxialie/SFT-Chinese-Dataset",
     "ruozhiba/ruozhiout_qa_cn.jsonl",
     "ruozhi.jsonl", 0),
]

for ds, remote, local, min_sz in jobs:
    out = os.path.join(DST, local)
    if os.path.exists(out) and os.path.getsize(out) >= min_sz:
        print(f"  [skip] {local} {os.path.getsize(out)/1e6:.0f}MB 已有")
        continue
    try:
        p = dataset_file_download(dataset_id=ds, file_path=remote, local_dir=DST)
        if p and os.path.abspath(p) != os.path.abspath(out):
            shutil.move(p, out)
        sz = os.path.getsize(out) if os.path.exists(out) else 0
        tag = "OK" if sz >= min_sz else "SMALL"
        print(f"  [{tag}] {local} {sz/1e6:.0f}MB")
    except Exception as e:
        print(f"  [FAIL] {local}: {str(e)[:100]}")
PY

# ---- 2. FineWeb-Edu-v2 ----
log "下载 FineWeb-Edu-v2 ${NSHARD} 片"
for i in $(seq 0 $((NSHARD-1))); do
    N=$(printf "%05d" "$i")
    OUT="data/raw/fwv2/${N}.parquet"
    if [ -f "$OUT" ] && [ "$(stat -c %s "$OUT")" -gt 900000000 ]; then
        log "  ${N}.parquet 已有"
        continue
    fi
    log "  下载 ${N}.parquet"
    python3 smart_dl.py \
        "https://hf-mirror.com/datasets/opencsg/chinese-fineweb-edu-v2/resolve/main/data/${N}.parquet" \
        -o "$OUT" --ncpu 32 2>&1 | tail -6 | tee -a logs/fetch.log
done

# ---- 3. LCCC ----
if [ -f data/raw/lccc_large.jsonl.gz ] && [ "$(stat -c %s data/raw/lccc_large.jsonl.gz)" = "607605643" ]; then
    log "LCCC 已有（607605643 字节）"
else
    log "下载 LCCC（607MB，32 路并发）"
    python3 smart_dl.py \
        'https://hf-mirror.com/datasets/silver/lccc/resolve/main/lccc_large.jsonl.gz' \
        -o data/raw/lccc_large.jsonl.gz --ncpu 32 2>&1 | tail -6 | tee -a logs/fetch.log
fi

# ---- 4. 汇总 ----
log "===== 数据清单 ====="
ls -lh data/raw/*.jsonl data/raw/*.gz data/raw/fwv2/*.parquet 2>/dev/null | tee -a logs/fetch.log
echo
log "总计 $(du -sh data/raw | cut -f1)"
log "下一步: bash cloud/prepare_and_encode.sh"