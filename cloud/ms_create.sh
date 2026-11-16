#!/usr/bin/env bash
# 创建魔搭数据集仓库（create_dataset 需要 namespace 参数）
set -u
cd /mnt/workspace/tiny-zh-64m || exit 1
export MODELSCOPE_API_TOKEN="${MODELSCOPE_API_TOKEN:-}"

python3 - <<'PY'
import os
import traceback
from modelscope.hub.api import HubApi

a = HubApi()
a.login(os.environ["MODELSCOPE_API_TOKEN"])

# 先确认用户名：尝试列出自己名下的数据集
print("=== 确认用户名 ===")
CANDIDATES = ["SKL666666", "skl666666", "SKL-666666", "skl-666666"]
owner = None
for c in CANDIDATES:
    try:
        res = a.list_repos(owner=c, repo_type="dataset", page_number=1, page_size=3)
        total = getattr(res, "total_count", None)
        # 只有本人 namespace 才允许列（别人也可以列公开的，所以再看能否创建）
        print(f"  {c}: list ok, total={total}")
    except Exception as e:
        print(f"  {c}: list fail {str(e)[:80]}")

print()
print("=== 尝试创建（带 namespace）===")
for c in CANDIDATES:
    try:
        a.create_dataset(
            dataset_name="tiny-zh-64m-data",
            namespace=c,
            chinese_name="Tiny-ZH-64M 训练数据",
            license="Apache-2.0",
            visibility=1,
        )
        print(f"  >>> 成功: {c}/tiny-zh-64m-data")
        owner = c
        break
    except Exception as e:
        msg = str(e)[:150]
        print(f"  FAIL {c}: {msg}")
        if "exists" in msg.lower() or "已存在" in msg:
            print(f"  >>> 仓库已存在，用户名 {c}")
            owner = c
            break

if owner:
    print()
    print(f"用户名 = {owner}")
    with open("/tmp/ms_owner.txt", "w") as f:
        f.write(owner)
else:
    print()
    print("全部失败，看上面错误信息")
PY