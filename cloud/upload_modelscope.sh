#!/usr/bin/env bash
# 上传训练数据到魔搭 ModelScope（内网 47MB/s，比 HF 快 30 倍）
#
# 为什么两边都传
# --------------
# 魔搭：国内下载 47MB/s，下次开新实例几分钟就拉完
# HF：国际备份，万一魔搭挂了还有
set -u
cd /mnt/workspace/tiny-zh-64m || exit 1

REPO="${REPO:-SKL666666/tiny-zh-64m-data}"
export MODELSCOPE_API_TOKEN="${MODELSCOPE_API_TOKEN:-}"

[ -z "$MODELSCOPE_API_TOKEN" ] && { echo "[FATAL] 需要 MODELSCOPE_API_TOKEN"; exit 1; }

echo "===== 1. 登录 ====="
python3 - <<'PY'
import os
from modelscope.hub.api import HubApi
api = HubApi()
api.login(os.environ["MODELSCOPE_API_TOKEN"])
print("  login OK")
PY

echo
echo "===== 2. 创建数据集仓库 ====="
python3 - <<'PY'
import os
from modelscope.hub.api import HubApi
api = HubApi()
api.login(os.environ["MODELSCOPE_API_TOKEN"])
repo = "SKL666666/tiny-zh-64m-data"
try:
    api.create_dataset(
        dataset_name=repo,
        dataset_type="nlp",
        license="Apache-2.0",
        visibility=1,           # 1=private
        chinese_name="Tiny-ZH-64M 训练数据",
    )
    print(f"  创建 {repo}")
except Exception as e:
    print(f"  已存在或创建失败: {str(e)[:120]}")
PY

echo
echo "===== 3. 上传文件 ====="
python3 - <<'PY'
import os
from pathlib import Path
from modelscope.hub.api import HubApi

api = HubApi()
api.login(os.environ["MODELSCOPE_API_TOKEN"])
repo = "SKL666666/tiny-zh-64m-data"
src = Path("data_upload")

files = sorted(f for f in src.iterdir() if f.is_file())
print(f"  共 {len(files)} 个文件")

for f in files:
    sz = f.stat().st_size
    print(f"  上传 {f.name} ({sz/1e6:.1f} MB) …", flush=True)
    try:
        # dataset_file_upload 只接受单文件，逐个传
        api.upload_file(
            path_or_fileobj=str(f),
            path_in_repo=f.name,
            repo_id=repo,
            repo_type="dataset",
            commit_message=f"add {f.name}",
        )
    except Exception as e:
        print(f"    [FAIL] {str(e)[:150]}")
PY

echo
echo "===== 4. 完成 ====="
echo "  https://www.modelscope.cn/datasets/$REPO"