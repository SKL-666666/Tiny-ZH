#!/usr/bin/env bash
# 用新 token + 用户名创建魔搭数据集仓库并上传
set -u
cd /mnt/workspace/tiny-zh-64m || exit 1
export MODELSCOPE_API_TOKEN="${MODELSCOPE_API_TOKEN:-}"
OWNER="${OWNER:-SKL000000}"

[ -z "$MODELSCOPE_API_TOKEN" ] && { echo "[FATAL] 需要 MODELSCOPE_API_TOKEN"; exit 1; }

echo "===== 1. 登录并验证 ====="
python3 - <<'PY'
import os
from modelscope.hub.api import HubApi
a = HubApi()
a.login(os.environ["MODELSCOPE_API_TOKEN"])
print("  login OK")
# 验证：列自己名下的仓库
for owner in ("SKL000000",):
    try:
        res = a.list_repos(owner=owner, repo_type="dataset", page_number=1, page_size=5)
        print(f"  {owner} 名下数据集 {getattr(res,'total_count','?')} 个")
        for it in (res.items or []):
            print("   ", getattr(it, "id", it))
    except Exception as e:
        print(f"  {owner} 列出失败: {str(e)[:100]}")
PY

echo
echo "===== 2. 创建数据集仓库 ====="
python3 - <<'PY'
import os
from modelscope.hub.api import HubApi
a = HubApi()
a.login(os.environ["MODELSCOPE_API_TOKEN"])
owner = os.environ.get("OWNER", "SKL000000")
repo_short = "tiny-zh-64m-data"
try:
    a.create_dataset(
        dataset_name=repo_short,
        namespace=owner,
        chinese_name="Tiny-ZH-64M 训练数据",
        license="Apache-2.0",
        visibility=1,
    )
    print(f"  OK 创建 {owner}/{repo_short}")
except Exception as e:
    msg = str(e)
    print(f"  创建返回: {msg[:200]}")
    if "exists" in msg.lower() or "已存在" in msg or "3002" in msg:
        print(f"  仓库可能已存在，继续上传")
PY

echo
echo "===== 3. 上传文件 ====="
python3 - <<'PY'
import os
import time
from pathlib import Path
from modelscope.hub.api import HubApi

a = HubApi()
a.login(os.environ["MODELSCOPE_API_TOKEN"])
owner = os.environ.get("OWNER", "SKL000000")
repo = f"{owner}/tiny-zh-64m-data"
src = Path("data_upload")

files = sorted(f for f in src.iterdir() if f.is_file())
print(f"  {len(files)} 个文件，总 {sum(f.stat().st_size for f in files)/1e9:.2f} GB")

ok = fail = 0
for f in files:
    sz = f.stat().st_size
    t0 = time.time()
    print(f"  {f.name} ({sz/1e6:.1f} MB) …", end=" ", flush=True)
    try:
        a.upload_file(
            path_or_fileobj=str(f),
            path_in_repo=f.name,
            repo_id=repo,
            repo_type="dataset",
            commit_message=f"add {f.name}",
        )
        dt = time.time() - t0
        speed = sz / 1e6 / max(dt, 0.1)
        print(f"OK {dt:.0f}s {speed:.1f}MB/s")
        ok += 1
    except Exception as e:
        print(f"FAIL {str(e)[:110]}")
        fail += 1

print(f"\n  成功 {ok} 失败 {fail}")
PY

echo
echo "===== 4. 完成 ====="
echo "  https://www.modelscope.cn/datasets/${OWNER}/tiny-zh-64m-data"