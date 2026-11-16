#!/usr/bin/env bash
# 上传训练数据到 HuggingFace（走 hf-mirror 镜像，国内可通）
#
# 修正记录
# --------
# 1. huggingface.co 直连在国内不通（Network is unreachable），必须走 hf-mirror.com
# 2. 用户名是 SKL666666，不是 SKL-666666（后者会 403）
set -u
cd /mnt/workspace/tiny-zh-64m || exit 1

REPO="${REPO:-SKL666666/tiny-zh-64m-data}"
WORK=data_upload
export HF_ENDPOINT=https://hf-mirror.com
export HF_TOKEN="${HF_TOKEN:-}"

[ -z "$HF_TOKEN" ] && { echo "[FATAL] 需要 HF_TOKEN"; exit 1; }

echo "===== 写 README ====="
cat > "$WORK/README.md" <<'EOF'
# Tiny-ZH-64M 训练数据

中文小模型从零训练用的语料，已清洗、去重、编码完毕。

## 模型规格

- 参数量：64,816,000
- 架构：13 层 × 640 维，10 heads / 5 kv heads，head_dim 64，ffn 1792
- 上下文：seq 1024 原生（rope_theta 500000）
- 词表：6400（BPE，ByteLevel，GPT-2 字节映射）

## 文件说明

| 文件 | 解压后 | 说明 |
|---|---|---|
| `pretrain_train.bin.gz` | 5.5 GB | **已编码 uint16 token 流，直接训练用** |
| `pretrain_val.bin` | 3.2 MB | 验证集 |
| `pretrain.txt.gz` | 2.7 GB | 清洗后纯文本（换词表时用） |
| `sft.jsonl.gz` | 174 MB | SFT 对话，`{"conversations":[{"role","content"}]}` |
| `sft_train.bin` | 28 MB | SFT 已编码 |
| `sft_train_idx.npy` | 551 KB | SFT 索引 `(start, prompt_len, total_len)` int32 |
| `tokenizer.json` | 468 KB | **必须和 bin 配套**，否则 token id 对不上 |

## 数据来源

| 来源 | 说明 |
|---|---|
| LCCC large | 1200 万段中文日常对话，已去字间空格 |
| FineWeb-Edu-v2 | opencsg 教育级中文网页 |
| 中文维基 | AI-ModelScope/wikipedia-cn-20230720-filtered |
| ShareGPT 中文 | zhuangxialie/SFT-Chinese-Dataset |
| COIG-CQIA | 知乎/小红书/wikihow 生活场景 |

## 清洗标准

- 中文占比 > 0.60
- 单段 30-4000 字
- 必须含句末标点
- 单字符重复率 < 30%（防复读）
- 前 64 字 blake2b 哈希去重
- **LCCC 字间空格已挤除**：原始格式逐字切分，`火锅 我 在 重庆` → `我在重庆`（实测空格占 38.1%）

## 统计

- 清洗后：2.7 GB / 376 万条
- 编码后：**29.27 亿 token**（45.2× 参数）
- 压缩率：1.20 字/token
- 多字中文 token：3618 个

## 用法

```python
# 直接用编码好的数据（跳过清洗和编码，省 30 分钟）
python train.py --mode pretrain --data ./bin --seq-len 1024 \
    --micro-batch 32 --grad-accum 1 --lr 5e-4

# 或用原始文本重新编码（换词表时）
python build_dataset.py --mode pretrain \
    --tokenizer tokenizer.json --src pretrain.txt --out-dir bin
```

## 坑（实测踩过）

1. `pretrain_train.bin` 是 uint16 裸二进制，读用 `np.memmap(path, dtype=np.uint16)`
2. `sft_train_idx.npy` 是 `.tofile()` 写的裸 int32，**必须 `np.fromfile`**，`np.load` 会当 pickle 报错
3. 训练时 micro_batch 32 在 A10 23GB 上占 18.5GB，bs64 会 OOM
4. tokenizers 的 token 字符串是 GPT-2 字节映射形式（`山` 存成 `å±±`），数字符统计要先还原字节

## 参考

- 训练配置：seq 1024 / micro_batch 32 / grad_accum 1 / lr 5e-4 / A10 实测 51,001 tok/s
EOF
echo "  OK"

echo
echo "===== 上传全部 ====="
hf upload "$REPO" "$WORK" --repo-type dataset --private 2>&1 | tail -20

echo
echo "===== 完成 ====="
echo "  https://huggingface.co/datasets/$REPO"
