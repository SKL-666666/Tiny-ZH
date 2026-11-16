#!/usr/bin/env bash
# Tiny-ZH 一键连接 —— 把今天所有踩过的坑修完
#
# 今天失败的完整清单与对应修复
# ----------------------------
# 1. tailscale_1.104.5 已下架(404)，curl -f 静默失败留下空文件
#    修：不硬编码版本，从 stable 页抓当前版本
# 2. 单连接只有 89KB/s，39MB 要 7 分钟
#    修：32 路分段并发，实测 2 分钟内完成
# 3. 关掉 --ssh 后普通 22 端口不通（userspace 模式必然）
#    修：必须 --ssh，不能省
# 4. 重启 daemon 丢登录态导致 offline
#    修：启动即带 --authkey，且 up 时也带
# 5. 本机 tailscale 会自己 stop
#    修：脚本末尾打印本机修复命令
# 6. RANGE 下载后拼接出错
#    修：三段校验（大小 / tar -t / 实际可执行）
#
# 用法（在魔搭 notebook 终端）：
#   curl -fsSL https://codeload.github.com/SKL-666666/Tiny-ZH-32M/tar.gz/refs/heads/master \
#     | tar xz -O Tiny-ZH-32M-master/cloud/connect.sh > /tmp/connect.sh && bash /tmp/connect.sh
#
# 或先拉代码再跑：
#   cd /mnt/workspace && curl -fsSL -o tz.tar.gz https://codeload.github.com/SKL-666666/Tiny-ZH-32M/tar.gz/refs/heads/master
#   tar xzf tz.tar.gz && bash Tiny-ZH-32M-master/cloud/connect.sh
set -u

AUTHKEY="${AUTHKEY-tskey-auth-REPLACE_WITH_YOUR_KEY}"
TS_NAME="${TS_NAME-tiny-zh}"
SEG="${SEG-32}"
WORK=/tmp/ts_install

say() { echo "[$(date '+%H:%M:%S')] $*"; }

say "===== 1/6 获取 tailscale 版本 ====="
VER=$(curl -fsSL --connect-timeout 15 --max-time 60 https://pkgs.tailscale.com/stable/ \
      | grep -oP 'tailscale_\K[0-9]+\.[0-9]+\.[0-9]+(?=_amd64\.tgz)' | head -1)
if [ -z "${VER:-}" ]; then
    say "[FATAL] 拿不到版本号，检查网络"
    exit 1
fi
PKG="tailscale_${VER}_amd64.tgz"
URL="https://pkgs.tailscale.com/stable/${PKG}"
say "  版本 $VER"

BIN="$WORK/tailscale_${VER}_amd64"
if [ -x "$BIN/tailscale" ]; then
    say "  已安装，跳过下载"
else
    say "===== 2/6 探测大小并分段下载 ====="
    SIZE=$(curl -fsSLI "$URL" | grep -i '^content-length:' | tail -1 | tr -d '\r' | awk '{print $2}')
    if [ -z "${SIZE:-}" ]; then
        say "[FATAL] 拿不到 Content-Length"
        exit 1
    fi
    say "  ${SIZE} 字节 ($((SIZE/1024/1024)) MB)，$SEG 路并发"

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
    say "  下载耗时 $(( $(date +%s) - T0 ))s"

    # 三段校验：存在 / 大小 / 归档完整性
    for i in $(seq 0 $((SEG-1))); do
        [ -s "pt_${i}.bin" ] || { say "[FATAL] 段 $i 缺失"; exit 1; }
    done
    cat $(for i in $(seq 0 $((SEG-1))); do echo "pt_${i}.bin"; done) > "$PKG"
    GOT=$(stat -c %s "$PKG")
    [ "$GOT" = "$SIZE" ] || { say "[FATAL] 大小 $GOT != $SIZE"; exit 1; }
    tar tzf "$PKG" >/dev/null 2>&1 || { say "[FATAL] 归档损坏"; exit 1; }
    say "  完整 OK"
    rm -f pt_*.bin

    say "===== 3/6 解压 ====="
    rm -rf "$BIN"
    tar xzf "$PKG" || exit 1
    [ -x "$BIN/tailscale" ] || { say "[FATAL] 解压后找不到 tailscale"; exit 1; }
    say "  OK"
fi

cd "$BIN" || exit 1

say "===== 4/6 重启 daemon ====="
pkill -f tailscaled 2>/dev/null
sleep 3
# userspace 模式：云端没有 /dev/net/tun，只能这样跑
nohup ./tailscaled --tun=userspace-networking --socks5-server=localhost:1055 \
    > /tmp/tsd.log 2>&1 &
sleep 8
if ! pgrep -a tailscaled >/dev/null; then
    say "[FATAL] daemon 没起来"
    tail -20 /tmp/tsd.log
    exit 1
fi
say "  daemon 运行中 (PID $(pgrep -f tailscaled | head -1))"

say "===== 5/6 登录（必须 --ssh + --authkey）====="
say "  注意：--ssh 不能省。userspace 模式下普通 22 端口不通，"
say "        只有 Tailscale SSH 能进来。"
timeout 180 ./tailscale up --ssh --hostname="$TS_NAME" --authkey="$AUTHKEY" 2>&1 | tail -5
sleep 5

say "===== 6/6 状态 ====="
./tailscale status 2>&1 | head -2
IP=$(./tailscale ip -4 2>/dev/null | head -1)

echo
echo "============================================================"
if [ -n "${IP:-}" ]; then
    echo "  ✅ 连接就绪"
    echo
    echo "  本机连接命令："
    echo "    ssh -o StrictHostKeyChecking=no root@${IP}"
else
    echo "  ❌ 没拿到 IP，检查 /tmp/tsd.log"
fi
echo "============================================================"
echo
echo "【如果本机连不上，先在本机 PowerShell 执行】"
echo '  & "C:\Program Files\Tailscale\tailscale.exe" down'
echo '  & "C:\Program Files\Tailscale\tailscale.exe" up'
echo "  这会强制本机重新同步节点列表（今天的经验）"
echo
echo "【GPU 检查】"
python3 - <<'PY' 2>&1
try:
    import torch
    print(f"  torch {torch.__version__}  cuda {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"  dev {p.name}  {p.total_memory/1e9:.1f}GB  sm {p.multi_processor_count}")
        print(f"  bf16 {torch.cuda.is_bf16_supported()}")
except ImportError as e:
    print(f"  [FATAL] torch 不可用: {e}")
import os
print(f"  cpu {os.cpu_count()} 核")
PY
echo
echo "【下一步】把上面的 100.x.x.x 发给我"