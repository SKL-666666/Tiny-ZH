#!/usr/bin/env bash
# 紧急上传：把今天所有新增的有价值产物推到 HF
set -u
cd /mnt/workspace/tiny-zh-64m || exit 1
export HF_ENDPOINT=https://hf-mirror.com
export HF_TOKEN="${HF_TOKEN:-}"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a logs/urgent_up.log; }

log "===== 紧急上传 ====="

# 1. 测试结果 + 报告（最重要，体积小）
mkdir -p up_tests
cp -f logs/strict_test.json logs/strict.log 2>/dev/null up_tests/
cp -f logs/three_way.json logs/three_way.md 2>/dev/null up_tests/
cp -f logs/three_way2.json logs/three_way2.md 2>/dev/null up_tests/
cp -f logs/compare.log logs/compare_v1_v2.json 2>/dev/null up_tests/
cp -f logs/sft_high.log logs/sft_v3.log 2>/dev/null up_tests/
cp -f three_way.md three_way2.md 2>/dev/null up_tests/
ls -lh up_tests/ | tail -12

log "1/4 上传测试结果"
hf upload "SKL666666/tiny-zh-64m-logs" up_tests --repo-type dataset --private 2>&1 | tail -3 | tee -a logs/urgent_up.log

# 2. 清理后的 SFT 数据（116MB，明天能直接用）
mkdir -p up_sftdata
cp -f data/sft_clean.jsonl up_sftdata/ 2>/dev/null
gzip -f up_sftdata/sft_clean.jsonl 2>/dev/null
cp -f build_sft_high.py clean_identity.py up_sftdata/ 2>/dev/null
ls -lh up_sftdata/ | tail -5

log "2/4 上传清理后 SFT 数据"
hf upload "SKL666666/tiny-zh-64m-data" up_sftdata --repo-type dataset --private 2>&1 | tail -3 | tee -a logs/urgent_up.log

# 3. v2 模型权重（已在之前传过 GGUF，这里传 pt）
log "3/4 上传 v2 权重"
mkdir -p up_pt
cp -f out/sft_high/best.pt up_pt/sft_v2_best.pt 2>/dev/null
hf upload "SKL666666/tiny-zh-64m" up_pt --repo-type model --private 2>&1 | tail -3 | tee -a logs/urgent_up.log

# 4. 全部日志
log "4/4 上传日志"
mkdir -p up_logs
cp -f logs/*.log logs/*.md logs/*.json 2>/dev/null up_logs/
hf upload "SKL666666/tiny-zh-64m-logs" up_logs --repo-type dataset --private 2>&1 | tail -3 | tee -a logs/urgent_up.log

log "===== 完成 ====="
echo "模型: https://huggingface.co/SKL666666/tiny-zh-64m"
echo "日志/测试: https://huggingface.co/datasets/SKL666666/tiny-zh-64m-logs"
echo "数据: https://huggingface.co/datasets/SKL666666/tiny-zh-64m-data"