#!/usr/bin/env bash
# 魔搭大文件上传：切块后分片传，规避单次连接超时
#
# 问题：直接传 1.27GB / 3.08GB 的文件，SDK 内部 HTTP 连接会在中途
#      TimeoutError('The write operation timed out')。
# 解法：每个大文件切成 200MB 分片，分别上传（魔搭支持任意文件名），
#      下次下载后 cat 拼接即可。同时在 README 里写清拼接方法。
set -u
cd /mnt/workspace/tiny-zh-64m || exit 1
export MODELSCOPE_API_TOKEN="${MODELSCOPE_API_TOKEN:-}"
OWNER="${OWNER:-SKL000000}"
REPO_SHORT="tiny-zh-64m-data"
UPLOAD_DIR="data_upload"
SPLIT_DIR="data_upload_split"
CHUNK_MB=200

[ -z "$MODELSCOPE_API_TOKEN" ] && { echo "[FATAL] 需要 MODELSCOPE_API_TOKEN"; exit 1; }

mkdir -p "$SPLIT_DIR"

echo "===== 1. 小文件直接传 ====="
python3 - <<'PY'
import os
from pathlib import Path
from modelscope.hub.api import HubApi

a = HubApi()
a.login(os.environ["MODELSCOPE_API_TOKEN"])
owner = os.environ.get("OWNER", "SKL000000")
repo = f"{owner}/tiny-zh-64m-data"
src = Path("data_upload")
BIG = 300 * 1024 * 1024      # 超过 300MB 的走分片

small = [f for f in sorted(src.iterdir()) if f.is_file() and f.stat().st_size <= BIG]
print(f"  小文件 {len(small)} 个")
for f in small:
    try:
        a.upload_file(path_or_fileobj=str(f), path_in_repo=f.name,
                      repo_id=repo, repo_type="dataset",
                      commit_message=f"add {f.name}")
        print(f"    OK {f.name} ({f.stat().st_size/1e6:.1f}MB)")
    except Exception as e:
        print(f"    FAIL {f.name}: {str(e)[:90]}")
PY

echo
echo "===== 2. 大文件切块 ====="
cd "$UPLOAD_DIR" || exit 1
for f in *.gz *.bin; do
    [ -f "$f" ] || continue
    SZ=$(stat -c %s "$f")
    [ "$SZ" -lt $((CHUNK_MB * 1024 * 1024)) ] && continue
    PARTS=$(( (SZ + CHUNK_MB*1024*1024 - 1) / (CHUNK_MB*1024*1024) ))
    echo "  $f  ${SZ} 字节 -> ${PARTS} 片"
    rm -f "../$SPLIT_DIR/${f}.part"* 2>/dev/null
    split -b ${CHUNK_MB}M -d -a 3 "$f" "../$SPLIT_DIR/${f}.part"
done
cd .. || exit 1

ls -lh "$SPLIT_DIR" | head -8
echo "  分片总数 $(ls $SPLIT_DIR | wc -l)"

echo
echo "===== 3. 逐个上传分片（带重试）====="
python3 - <<'PY'
import os
import time
from pathlib import Path
from modelscope.hub.api import HubApi

a = HubApi()
a.login(os.environ["MODELSCOPE_API_TOKEN"])
owner = os.environ.get("OWNER", "SKL000000")
repo = f"{owner}/tiny-zh-64m-data"
src = Path("data_upload_split")

parts = sorted(f for f in src.iterdir() if f.is_file())
print(f"  {len(parts)} 片，每片约 200MB")

ok = fail = 0
failed = []
for i, f in enumerate(parts, 1):
    sz = f.stat().st_size
    for attempt in (1, 2, 3):
        try:
            t0 = time.time()
            a.upload_file(path_or_fileobj=str(f), path_in_repo=f.name,
                          repo_id=repo, repo_type="dataset",
                          commit_message=f"part {f.name}")
            dt = time.time() - t0
            print(f"  [{i}/{len(parts)}] {f.name} {sz/1e6:.0f}MB "
                  f"{dt:.0f}s {sz/1e6/max(dt,0.1):.0f}MB/s", flush=True)
            ok += 1
            break
        except Exception as e:
            if attempt < 3:
                print(f"  [{i}/{len(parts)}] {f.name} 第{attempt}次失败，重试…", flush=True)
                time.sleep(5)
            else:
                print(f"  [{i}/{len(parts)}] {f.name} FAIL {str(e)[:90]}", flush=True)
                fail += 1
                failed.append(f.name)

print(f"\n  成功 {ok} 失败 {fail}")
if failed:
    print("  失败清单:")
    for x in failed:
        print("   ", x)
PY

echo
echo "===== 4. 写拼接说明 ====="
cat > "$SPLIT_DIR/README_SPLIT.md" <<'EOF'
# 分片文件拼接说明

本仓库的大文件被切成 200MB 分片，文件名格式：

```
pretrain_train.bin.gz.part000
pretrain_train.bin.gz.part001
...
```

## 下载后拼接

```bash
# 1. 下载全部分片（魔搭 SDK 或网页批量下载）
# 2. 按顺序拼接
cat pretrain_train.bin.gz.part* > pretrain_train.bin.gz

# 3. 校验并解压
gzip -t pretrain_train.bin.gz && echo "校验通过"
gzip -d pretrain_train.bin.gz
```

## 为什么分片

魔搭 SDK 单次上传超过约 1GB 时，HTTP 连接会中途超时（`TimeoutError`）。
切成 200MB 后每片独立上传，可重试，不会整体失败。
EOF
echo "  README_SPLIT.md 写好"

echo
echo "  https://www.modelscope.cn/datasets/${OWNER}/${REPO_SHORT}"