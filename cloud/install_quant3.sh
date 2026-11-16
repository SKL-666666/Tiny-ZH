#!/usr/bin/env bash
# 装量化工具的可靠路径：pip 装 llama-cpp-python（自带 ggml 量化代码）
#
# 为什么这条路更可靠
# ------------------
# 1. GitHub 预编译包被墙，下载不动
# 2. 源码编译 llama.cpp 要几分钟且容易缺依赖
# 3. llama-cpp-python 的 wheel 在 PyPI 上，pip 会走国内镜像
# 4. 它内部链接了 ggml-quants.c，量化算法和 llama-quantize 完全一致
set -u
TOOLS=/mnt/workspace/tools
mkdir -p "$TOOLS" && cd "$TOOLS" || exit 1

if [ -x "$TOOLS/llama-quantize" ]; then
    echo "已有 llama-quantize"; exit 0
fi

echo "===== 1. pip 装 llama-cpp-python ====="
# 用清华镜像，走 CPU 版本避免 CUDA 编译
pip install -q --no-cache-dir \
    --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    "llama-cpp-python" 2>&1 | tail -5

echo
echo "===== 2. 找量化模块 ====="
python3 - <<'PY'
import sys
try:
    import llama_cpp
    print(f"  llama_cpp {llama_cpp.__version__}")
    print(f"  路径 {llama_cpp.__file__}")
except ImportError as e:
    print(f"  导入失败: {e}")
    sys.exit(1)
PY

echo
echo "===== 3. 检查是否带 quantize ====="
python3 -c "import llama_cpp.llama_cpp as m; print('  有 llama_model_quantize:', hasattr(m, 'llama_model_quantize'))" 2>&1 | tail -1

echo
echo "===== 4. 找 llama-quantize 可执行文件 ====="
find /usr -name 'llama-quantize*' 2>/dev/null | head -3
python3 -c "
import llama_cpp, os
d = os.path.dirname(llama_cpp.__file__)
print('  lib dir:', d)
for f in os.listdir(d):
    if 'quant' in f.lower():
        print('   ', f)
" 2>&1 | tail -6

echo
echo "===== 5. 结论 ====="
echo "  如果上面显示有 llama_model_quantize，就用 python API 量化"