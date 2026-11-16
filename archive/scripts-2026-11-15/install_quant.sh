#!/usr/bin/env bash
# 装 llama.cpp 工具链（只为量化 GGUF）
#
# 为什么要装
# ----------
# 云端没有 llama-quantize，也没有 python gguf 包。
# 不量化只能给 f16（约 130MB），量化到 Q4_K_M 后约 40MB，小 70%。
#
# 策略：优先下载预编译二进制，其次源码编译，最后纯 python 量化。
set -u
TOOLS=/mnt/workspace/tools
mkdir -p "$TOOLS"
cd "$TOOLS" || exit 1

echo "===== 1. 查系统 ====="
uname -m
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1

echo
echo "===== 2. 试预编译二进制 ====="
# llama.cpp 官方 release 提供 Ubuntu x64 二进制
VER="b4667"
URL="https://github.com/ggml-org/llama.cpp/releases/download/${VER}/llama-${VER}-bin-ubuntu-x64.zip"
if [ ! -x "$TOOLS/llama-quantize" ]; then
    echo "  下载 $URL"
    if curl -fsSL --connect-timeout 15 --max-time 300 -o llama.zip "$URL" 2>/dev/null; then
        echo "  下载 OK $(du -h llama.zip | cut -f1)"
        if command -v unzip >/dev/null 2>&1; then
            unzip -oq llama.zip -d "$TOOLS/tmp" && echo "  解压 OK"
        else
            python3 -c "import zipfile; zipfile.ZipFile('llama.zip').extractall('$TOOLS/tmp')" && echo "  python 解压 OK"
        fi
        find "$TOOLS/tmp" -name 'llama-quantize' -type f -exec cp {} "$TOOLS/" \; 2>/dev/null
        chmod +x "$TOOLS/llama-quantize" 2>/dev/null
    else
        echo "  下载失败（GitHub 直连可能被墙）"
    fi
fi

if [ -x "$TOOLS/llama-quantize" ]; then
    echo "  llama-quantize 可用"
    "$TOOLS/llama-quantize" --help 2>&1 | head -3
    exit 0
fi

echo
echo "===== 3. 试 pip 装 gguf（纯 python 备选）====="
pip install -q gguf 2>&1 | tail -3
python3 -c "import gguf; print('  gguf', gguf.__version__ if hasattr(gguf,'__version__') else 'ok')" 2>&1 | tail -1

echo
echo "===== 4. 检查结果 ====="
if [ -x "$TOOLS/llama-quantize" ]; then
    echo "  OK: $TOOLS/llama-quantize"
else
    echo "  没有 llama-quantize"
fi
python3 -c "import gguf; print('  gguf 模块可用')" 2>/dev/null || echo "  没有 gguf 模块"