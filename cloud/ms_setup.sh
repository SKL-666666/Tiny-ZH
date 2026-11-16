#!/usr/bin/env bash
# 查魔搭用户名并创建数据集仓库，然后用 git 方式上传
set -u
cd /mnt/workspace/tiny-zh-64m || exit 1
export MODELSCOPE_API_TOKEN="${MODELSCOPE_API_TOKEN:-}"

echo "===== 1. 查用户名 ====="
python3 - <<'PY'
import os
from modelscope.hub.api import HubApi
a = HubApi()
a.login(os.environ["MODELSCOPE_API_TOKEN"])
try:
    print("user:", a.get_user_name())
except Exception as e:
    print("get_user_name 失败:", str(e)[:100])
PY

echo
echo "===== 2. 列出已有数据集 ====="
python3 - <<'PY'
import os
from modelscope.hub.api import HubApi
a = HubApi()
a.login(os.environ["MODELSCOPE_API_TOKEN"])
try:
    ds = a.list_datasets(group_or_owner=a.get_user_name(), page_number=1, page_size=10)
    print(f"  共 {len(ds)} 个")
    for d in ds[:10]:
        print("  ", getattr(d, "path", d))
except Exception as e:
    print("  列出失败:", str(e)[:150])
PY

echo
echo "===== 3. 创建仓库（显示出错细节）====="
python3 - <<'PY'
import os, traceback
from modelscope.hub.api import HubApi
a = HubApi()
a.login(os.environ["MODELSCOPE_API_TOKEN"])
user = a.get_user_name()
repo = f"{user}/tiny-zh-64m-data"
print(f"  尝试创建 {repo}")
try:
    a.create_dataset(
        dataset_name=repo,
        chinese_name="Tiny-ZH-64M 训练数据",
        license="Apache-2.0",
        visibility=1,
    )
    print("  创建成功")
except Exception:
    traceback.print_exc()
PY