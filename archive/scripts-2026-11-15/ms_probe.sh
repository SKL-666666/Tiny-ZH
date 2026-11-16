#!/usr/bin/env bash
# 探查当前 modelscope SDK 的实际 API 名称
set -u
cd /mnt/workspace/tiny-zh-64m || exit 1
export MODELSCOPE_API_TOKEN="${MODELSCOPE_API_TOKEN:-}"

python3 - <<'PY'
import os
from modelscope.hub.api import HubApi
a = HubApi()
print("token 长度", len(os.environ["MODELSCOPE_API_TOKEN"]))

print("\n=== HubApi 公开方法 ===")
methods = [m for m in dir(a) if not m.startswith("_")]
for m in sorted(methods):
    print(" ", m)
PY