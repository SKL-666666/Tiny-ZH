#!/usr/bin/env bash
# 下载 SFT 源数据（minimind + firefly）
set -u
cd /mnt/workspace/tiny-zh-64m || exit 1
mkdir -p data/raw logs
export PYTHONIOENCODING=utf-8

python3 - <<'PY' 2>&1 | tee -a logs/dl_sft.log
import os
import shutil
from modelscope.hub.file_download import dataset_file_download

DST = "data/raw"
os.makedirs(DST, exist_ok=True)

jobs = [
    ("gongjy/minimind_dataset", "sft_t2t_mini.jsonl",
     "minimind_sft.jsonl", "qwen3 蒸馏对话，格式规整"),
    ("zhuangxialie/SFT-Chinese-Dataset",
     "firely-train/firefly-train-1.1M.jsonl",
     "firefly.jsonl", "165万条中文指令"),
]

for ds, remote, local, note in jobs:
    out = os.path.join(DST, local)
    if os.path.exists(out) and os.path.getsize(out) > 10 << 20:
        print(f"  [cached] {local} {os.path.getsize(out)/1e6:.0f}MB  {note}")
        continue
    try:
        p = dataset_file_download(dataset_id=ds, file_path=remote, local_dir=DST)
        if p and os.path.abspath(p) != os.path.abspath(out):
            shutil.move(p, out)
        print(f"  [got] {local} {os.path.getsize(out)/1e6:.0f}MB  {note}")
    except Exception as e:
        print(f"  [FAIL] {local}: {str(e)[:120]}")
PY

echo "=== 结果 ==="
ls -lh data/raw/minimind_sft.jsonl data/raw/firefly.jsonl 2>/dev/null