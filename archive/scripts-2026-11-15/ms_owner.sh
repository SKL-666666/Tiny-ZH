#!/usr/bin/env bash
# 从 list_repos 里挖出真实用户名
set -u
export MODELSCOPE_API_TOKEN="${MODELSCOPE_API_TOKEN:-}"

python3 - <<'PY'
import os
from modelscope.hub.api import HubApi

a = HubApi()
a.login(os.environ["MODELSCOPE_API_TOKEN"])

print("=== list_repos(owner=None) ===")
try:
    res = a.list_repos(owner=None, repo_type="dataset", page_number=1, page_size=5)
    print("  type:", type(res).__name__)
    print("  attrs:", [x for x in dir(res) if not x.startswith("_")][:15])
    # PagedResult: 尝试 .objects / .data / 迭代
    for attr in ("objects", "Data", "data", "list"):
        if hasattr(res, attr):
            v = getattr(res, attr)
            print(f"  .{attr}: {type(v).__name__}")
            try:
                for item in (v if isinstance(v, (list, tuple)) else []):
                    print("    item:", getattr(item, "path", item))
            except Exception as e:
                print("    迭代失败", e)
except Exception as e:
    print("  FAIL:", str(e)[:200])

print()
print("=== 直接读 token 对应的 owner ===")
try:
    print("  token_prefix:", os.environ["MODELSCOPE_API_TOKEN"][:12])
    # 从 SDK 内部属性找
    for attr in dir(a):
        if "user" in attr.lower() or "owner" in attr.lower() or "name" in attr.lower():
            print(f"  has attr: {attr}")
except Exception as e:
    print("  ", e)
PY