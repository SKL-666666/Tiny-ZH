#!/usr/bin/env bash
# SSH 一键就绪：装 tailscale + 起 daemon + 连接，全程有进度输出
#
# 今天踩过的坑，全部避开：
# 1. 不硬编码版本号 —— 1.104.5 已下架，导致 curl -f 静默失败留下空文件
# 2. 32 路分段并发 —— 单连接只有 89KB/s，并发实测 2.2MB/s
# 3. 加 tar tzf 校验 —— 防止下到半个文件还能"解压"
# 4. 不加 --ssh —— Tailscale SSH 要走浏览器鉴权，在 DERP 中继下会超时
# 5. 每步都打印状态 —— 卡住时能立刻看出卡在哪
#
# 用法：
#   bash cloud/ts_up.sh                    # 默认 hostname tiny-zh-64m
#   HOSTNAME_TS=xxx AUTHKEY=yyy bash cloud/ts_up.sh
set -u

AUTHKEY="${AUTHKEY:-tskey-auth-REPLACE_WITH_YOUR_KEY}"
TS_NAME="${TS_NAME:-tiny-zh-64m}"
SEG="${SEG:-32}"
WORK=/tmp/ts_install

echo "===== 1. 取当前版本号 ====="
VER=$(curl -fsSL --connect-timeout 15 --max-time 60 https://pkgs.tailscale.com/stable/ \
      | grep -oP 'tailscale_\K[0-9]+\.[0-9]+\.[0-9]+(?=_amd64\.tgz)' | head -1)
if [ -z "${VER:-}" ]; then
    echo "[FATAL] 拿不到版本号，检查网络"
    exit 1
fi
PKG="tailscale_${VER}_amd64.tgz"
URL="https://pkgs.tailscale.com/stable/${PKG}"
echo "  版本 $VER"

echo
echo "===== 2. 探测大小 ====="
SIZE=$(curl -fsSLI "$URL" | grep -i '^content-length:' | tail -1 | tr -d '\r' | awk '{print $2}')
[ -z "${SIZE:-}" ] && { echo "[FATAL] 拿不到 Content-Length"; exit 1; }
echo "  $((SIZE/1024/1024)) MB"

echo
echo "===== 3. 32 路并发下载 ====="
mkdir -p "$WORK" && cd "$WORK" || exit 1
rm -f pt_*.bin "$PKG"
CHUNK=$(( (SIZE + SEG - 1) / SEG ))
T0=$(date +%s)
for i in $(seq 0 $((SEG-1))); do
    a=$(( i * CHUNK ))
    b=$(( a + CHUNK - 1 ))
    [ "$b" -ge "$SIZE" ] && b=$((SIZE-1))
    curl -fsSL --connect-timeout 8 --max-time 300 --retry 5 \
         --range "${a}-${b}" -o "pt_${i}.bin" "$URL" >/dev/null 2>&1 &
done
wait
echo "  耗时 $(( $(date +%s) - T0 ))s"

for i in $(seq 0 $((SEG-1))); do
    [ -s "pt_${i}.bin" ] || { echo "[FATAL] 段 $i 失败"; exit 1; }
done
cat $(for i in $(seq 0 $((SEG-1))); do echo "pt_${i}.bin"; done) > "$PKG"
GOT=$(stat -c %s "$PKG")
[ "$GOT" = "$SIZE" ] || { echo "[FATAL] 大小 $GOT != $SIZE"; exit 1; }
tar tzf "$PKG" >/dev/null 2>&1 || { echo "[FATAL] 归档损坏"; exit 1; }
echo "  完整 OK"
rm -f pt_*.bin

echo
echo "===== 4. 解压 ====="
rm -rf "tailscale_${VER}_amd64"
tar xzf "$PKG" || exit 1
cd "tailscale_${VER}_amd64" || exit 1

echo
echo "===== 5. 起 daemon（不加 --ssh，避免浏览器鉴权）====="
pkill -f tailscaled 2>/dev/null
sleep 2
nohup ./tailscaled --tun=userspace-networking --socks5-server=localhost:1055 \
    > /tmp/tsd.log 2>&1 &
sleep 6
if ! pgrep -a tailscaled >/dev/null; then
    echo "[FATAL] daemon 没起来"
    tail -20 /tmp/tsd.log
    exit 1
fi
echo "  daemon 运行中"

echo
echo "===== 6. 连接 ====="
timeout 120 ./tailscale up --authkey="$AUTHKEY" --hostname="$TS_NAME"
echo "  exit=$?"

echo
echo "===== 7. 结果 ====="
./tailscale status 2>&1 | head -3
echo
IP=$(./tailscale ip -4 2>/dev/null | head -1)
echo "================================"
echo "  本机 IP: ${IP:-未拿到}"
echo "  本机连接: ssh -o StrictHostKeyChecking=no root@${IP:-<IP>}"
echo "================================"

echo
echo "===== 8. GPU 检查（比 SSH 更要紧）====="
python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" 2>&1 || echo "torch 不可用"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1 || echo "nvidia-smi 不可用"