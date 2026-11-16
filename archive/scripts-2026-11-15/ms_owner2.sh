#!/usr/bin/env bash
# PagedResult 有 .items，从中拿 owner
set -u
export MODELSCOPE_API_TOKEN="${MODELSCOPE_API_TOKEN:-}"

python3 - <<'PY'
import os
from modelscope.hub.api import HubApi

a = HubApi()
a.login(os.environ["MODELSCOPE_API_TOKEN"])

print("=== list_repos items ===")
try:
    res = a.list_repos(owner=None, repo_type="dataset", page_number=1, page_size=5)
    print("  total:", getattr(res, "total_count", "?"))
    for it in res.items:
        print("  item:", type(it).__name__, getattr(it, "path", it))
        d = it.to_dict() if hasattr(it, "to_dict") else {}
        if d:
            for k in ("Path", "Name", "Owner", "path", "name", "owner"):
                if k in d:
                    print(f"    {k}: {d[k]}")
except Exception as e:
    print("  FAIL:", str(e)[:200])

print()
print("=== 尝试全路径推测 ===")
# 用 SKL666666 试创建（如果用户名不同会 403）
for owner in ("SKL666666", "skl666666", "SKL-666666"):
    repo = f"{owner}/tiny-zh-64m-data"
    try:
        a.create_dataset(dataset_name=repo, chinese_name="Tiny-ZH-64M 训练数据",
                         license="Apache-2.0", visibility=1)
        print(f"  OK 创建成功: {repo}")
        print(f"  >>> 用户名是 {owner}")
        break
    except Exception as e:
        msg = str(e)[:100]
        print(f"  FAIL {repo}: {msg}")
PY