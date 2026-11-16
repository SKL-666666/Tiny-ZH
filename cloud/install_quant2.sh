#!/usr/bin/env bash
# 装 llama-quantize（走 GitHub 镜像 + 源码编译双保险）
set -u
TOOLS=/mnt/workspace/tools
mkdir -p "$TOOLS"
cd "$TOOLS" || exit 1

if [ -x "$TOOLS/llama-quantize" ]; then
    echo "已存在"; exit 0
fi

echo "===== 1. 镜像下载预编译包 ====="
VER="b4667"
GH="https://github.com/ggml-org/llama.cpp/releases/download/${VER}/llama-${VER}-bin-ubuntu-x64.zip"
for u in \
    "https://ghfast.top/$GH" \
    "https://gh-proxy.com/$GH" \
    "https://hub.gitmirror.com/$GH" \
    "https://ghproxy.net/$GH" ; do
    echo "  试 $u"
    if curl -fsSL --connect-timeout 12 --max-time 240 -o llama.zip "$u" 2>/dev/null; then
        SZ=$(stat -c %s llama.zip)
        echo "    下载 $(($SZ/1024/1024))MB"
        if python3 -c "import zipfile; zipfile.ZipFile('llama.zip')" 2>/dev/null; then
            python3 -c "import zipfile; zipfile.ZipFile('llama.zip').extractall('$TOOLS/tmp')"
            find "$TOOLS/tmp" -name 'llama-quantize*' -type f 2>/dev/null | head -3
            F=$(find "$TOOLS/tmp" -name 'llama-quantize' -type f 2>/dev/null | head -1)
            if [ -n "$F" ]; then
                cp "$F" "$TOOLS/" && chmod +x "$TOOLS/llama-quantize"
                echo "    OK -> $TOOLS/llama-quantize"
                break
            fi
        else
            echo "    不是有效 zip"
        fi
    fi
done

if [ -x "$TOOLS/llama-quantize" ]; then
    "$TOOLS/llama-quantize" --help 2>&1 | head -3
    echo "=== 量化工具就绪 ==="
    exit 0
fi

echo
echo "===== 2. 源码编译（慢但可靠）====="
echo "  检查编译依赖"
for c in git cmake make g++; do
    command -v $c >/dev/null 2>&1 && echo "    $c OK" || echo "    $c 缺失"
done

if command -v cmake >/dev/null 2>&1 && command -v g++ >/dev/null 2>&1; then
    cd "$TOOLS" || exit 1
    if [ ! -d llama.cpp ]; then
        echo "  clone 源码"
        for u in \
            "https://ghfast.top/https://github.com/ggml-org/llama.cpp.git" \
            "https://github.com/ggml-org/llama.cpp.git" ; do
            git clone --depth 1 "$u" 2>/dev/null && break
        done
    fi
    if [ -d llama.cpp ]; then
        cd llama.cpp || exit 1
        echo "  编译（只 CPU，量化不需要 GPU）"
        cmake -B build -DLLAMA_CURL=OFF -DGGML_NATIVE=OFF 2>&1 | tail -3
        cmake --build build --config Release -j8 --target llama-quantize 2>&1 | tail -5
        F=$(find build -name 'llama-quantize' -type f | head -1)
        if [ -n "$F" ]; then
            cp "$F" "$TOOLS/" && chmod +x "$TOOLS/llama-quantize"
            echo "  OK -> $TOOLS/llama-quantize"
            exit 0
        fi
    fi
else
    echo "  缺 cmake 或 g++，跳过编译"
fi

echo
echo "=== 结论：没有 llama-quantize ==="
echo "将只用 python gguf 做纯 numpy 量化（功能受限）"
exit 1