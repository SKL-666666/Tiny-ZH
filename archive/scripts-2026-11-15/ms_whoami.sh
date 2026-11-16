#!/usr/bin/env bash
# 用 token 查真实用户名（不同 SDK 版本方法不同，逐个试）
set -u
export MODELSCOPE_API_TOKEN="${MODELSCOPE_API_TOKEN:-}"

python3 - <<'PY'
import json
import os
import urllib.request

TOKEN = os.environ["MODELSCOPE_API_TOKEN"]

print("=== 试 API: /api/v1/users/me ===")
for url in (
    "https://www.modelscope.cn/api/v1/users/me",
    "https://www.modelscope.cn/api/v1/user/info",
    "https://www.modelscope.cn/api/v1/dolphin/account/me",
):
    try:
        req = urllib.request.Request(url, headers={
            "Authorization": f"Bearer {TOKEN}",
            "Content-Type": "application/json",
        })
        with urllib.request.urlopen(req, timeout=15) as r:
            body = r.read().decode("utf-8", "replace")
        print(f"  OK {url}")
        print(f"    {body[:400]}")
        try:
            d = json.loads(body)
            for k in ("Name", "name", "Username", "username", "Login", "login"):
                if isinstance(d, dict) and k in d:
                    print(f"    >>> 用户名: {d[k]}")
            data = d.get("Data") if isinstance(d, dict) else None
            if isinstance(data, dict):
                for k in ("Name", "name", "Username", "username"):
                    if k in data:
                        print(f"    >>> 用户名(Data): {data[k]}")
        except Exception:
            pass
        break
    except Exception as e:
        print(f"  FAIL {url}: {str(e)[:80]}")

print("\n=== 试 SDK: HubApi.list_datasets 不带 owner ===")
from modelscope.hub.api import HubApi
a = HubApi()
a.login(TOKEN)
for kwargs in (
    {"page_number": 1, "page_size": 5},
    {"group_or_owner": "SKL666666", "page_number": 1, "page_size": 5},
):
    try:
        ds = a.list_datasets(**kwargs)
        print(f"  OK {kwargs} -> {len(ds)} 个")
        for d in ds[:5]:
            print("   ", getattr(d, "path", d))
        break
    except Exception as e:
        print(f"  FAIL {kwargs}: {str(e)[:100]}")
PY