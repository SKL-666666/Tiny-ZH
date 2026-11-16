# Tiny-ZH-33.7M 训练计划

> 目标：33.37M 中文对话模型，**20× token（6.67 亿）**，架构上追平/超越 minimind-3
> 写于 2026-11-16，基于 11-15 的 64M 失败教训

---

## 一、为什么做 32M

| 项 | 11-15 的 64M | 本次 33.7M |
|---|---|---|
| 参数量 | 64.82M | **33.37M** |
| 预训练 token | 9.42 亿（14.5×） | **6.67 亿（20×）** |
| 数据主力 | LCCC 55%（46字碎片） | **pretrain_t2t 62%（274字完整段落）** |
| FFN/hidden | 2.80 ❌ | **4.00** ✅ |
| 三方测试得分 | 42/105（40%） | 目标 **55-65%** |

**核心逻辑：上次输在数据质量（不是量），这次数据换对了，架构也修对了。**

---

## 二、架构（33.37M，minimind-3 等比缩小）

**设计原则：不发明新结构，只把 minimind-3 按比例缩小。**

理由：minimind-3 实测 70.5 分（我们的 64M 只有 40 分），它的架构比例是经过验证的。任何偏离都是赌博。

```python
ModelConfig(
    vocab_size=6400,
    hidden_size=512,
    num_layers=9,
    num_heads=8,
    num_kv_heads=4,          # GQA 2:1，与 minimind-3 相同
    head_dim=64,             # 8×64=512
    intermediate_size=1664,  # FFN = 3.25×hidden，对齐 minimind-3 的 3.17
    max_position_embeddings=1024,
    rope_theta=1000000.0,    # 与 minimind-3 相同
    tie_word_embeddings=True,
)
# 33,367,552 参数
```

### 与 minimind-3 的比例对照

| 项 | minimind-3 | 本方案 | 比例保持 |
|---|---|---|---|
| hidden | 768 | **512** | 0.67× |
| 层数 | 8 | **9** | 1.13×（补偿变窄） |
| num_heads | 8 | **8** | 1.00× |
| num_kv_heads | 4 | **4** | 1.00× |
| head_dim | 96 | **64** | 0.67× |
| **FFN/hidden** | **3.17** | **3.25** | **1.03×** ✅ |
| GQA 比 | 2:1 | **2:1** | 相同 |
| rope_theta | 1e6 | **1e6** | 相同 |
| 词表占比 | 7.7% | 9.8% | 1.27×（小模型必然） |
| 参数量 | 63.91M | **33.37M** | 0.52× |

### 参数分解

| 项 | 参数量 | 占比 |
|---|---|---|
| embedding（tied） | 3,276,800 | 9.8% |
| 每层 q_proj | 262,144 | — |
| 每层 k_proj | 131,072 | — |
| 每层 v_proj | 131,072 | — |
| 每层 o_proj | 262,144 | — |
| 每层 attn 小计 | 786,432 | 23.5%（每层内） |
| 每层 gate/up/down | 2,555,904 | 76.4%（每层内） |
| 每层小计 | 3,343,360 | — |
| 9 层合计 | 30,090,240 | 90.2% |
| final norm | 512 | — |
| **总计** | **33,367,552** | 100% |

### 相比 11-15 失败的 64M

| 项 | 昨天 64.82M（40分） | 本方案 33.37M | 变化 |
|---|---|---|---|
| hidden | 640 | 512 | −20% |
| 层数 | 13 | 9 | −31% |
| **FFN/hidden** | **2.80** | **3.25** | **+16%** ✅ |
| num_kv_heads | 5 | 4 | −1 |
| head_dim | 64 | 64 | 相同 |

**唯一实质改动：FFN/hidden 从 2.80 提到 3.25。** 昨天 2.80 低于所有主流小模型（MobileLLM 2.67-2.80、SmolLM2 2.67、Qwen3 3.00），提到 3.25 是向主流靠拢。

### 文献查证记录

查了 MobileLLM / SmolLM2 / Qwen3 / Qwen2.5 的**实际 config.json**（非论文描述）：

| 模型 | hidden | 层数 | heads/kv | FFN | FFN/h | head_dim |
|---|---|---|---|---|---|---|
| MobileLLM-125M | 576 | 30 | 9/3 | 1536 | **2.67** | 64 |
| SmolLM2-135M | 576 | 30 | 9/3 | 1536 | **2.67** | 64 |
| MobileLLM-350M | 960 | 32 | 15/5 | 2560 | 2.67 | 64 |
| MobileLLM-1B | 1280 | 54 | 20/5 | 3584 | 2.80 | 64 |
| Qwen3-0.6B | 1024 | 28 | 16/8 | 3072 | **3.00** | 128 |
| Qwen2.5-0.5B | 896 | 24 | 14/2 | 4864 | **5.43** | 64 |
| **minimind-3** | 768 | 8 | 8/4 | 2432 | **3.17** | 96 |
| **本方案** | **512** | **9** | **8/4** | **1664** | **3.25** | **64** |

**关键发现：主流 FFN/hidden 是 2.67-3.17，不是 4.0。** 早期我建议的 4.0× 是错的，已纠正。

**为什么没有采用 MobileLLM 的「30 层」：**
在 33.37M 和 6400 词表约束下，词表占 9.8% 后，每层只分到 3.34M。如果要 30 层，每层只有 1.0M，FFN 会薄到 1.0×hidden，反而破坏比例。**这是参数预算的硬约束，不是选择。**

---

## 三、数据（三个源，全部实测过）

### 3.1 清单

| # | 数据集 | 路径 | 大小 | 实测质量 | 占比 |
|---|---|---|---|---|---|
| 1 | **pretrain_t2t.jsonl** | 魔搭 `gongjy/minimind_dataset` | **7.89GB** | 274字/中0.77/合格76% | **62%** |
| 2 | **SkyPile-150B** | hf-mirror `Skywork/SkyPile-150B` → `data/2020-40_zh_head_*.jsonl` | 每片约 350MB | **970字/中0.83/合格87%** | **34%** |
| 3 | **CCI3-HQ** | `BAAI/CCI3-HQ` → `data/part_*.jsonl` | 每片 1GB | 2253字/中0.85/合格61% | **4%** |

### 3.2 为什么这样配

| 数据源 | 提供什么 | 为什么不能更多 |
|---|---|---|
| pretrain_t2t | 对话语感、日常表达 | 62% 已足够，再加会过拟合 |
| SkyPile | 长文连贯、知识密度 | 网页语料，过多会稀释对话感 |
| CCI3-HQ | 高质量补充 | p90=4436 字太长，截断损失大 |

**对比 11-15 的错误配比：**

| 项 | 11-15 | 本次 |
|---|---|---|
| LCCC | 55%（46字碎片） | **0%** |
| FineWeb | 35%（网页噪声） | **0%** |
| pretrain_t2t | **0%** ❌ | **62%** ✅ |

**11-15 把最好的数据完全排除了，这是根本错误。**

### 3.3 需要下载的量

| 项 | 大小 | 说明 |
|---|---|---|
| pretrain_t2t.jsonl | 7.89GB | 全量（用 62% = 4.9GB 的字） |
| SkyPile 8 片 | 约 2.8GB | 8 × 350MB |
| CCI3-HQ 1 片 | 1GB | 只用 1 片 |
| **合计** | **约 11.7GB** | 魔搭 40MB/s / hf-mirror 2.2MB/s |

### 3.4 清洗标准

```python
# 通用
中文占比      > 0.65
长度          100 - 2000 字    ← 上限定 2000，超长会截断
必须有句末标点 。！？
单字符重复率  < 25%
哈希去重      前 64 字 blake2b

# CCI3-HQ 专属
长度收紧到 100-1500 字（p90=4436 太长，只取中段质量）

# 全部
不去空格（这些源不是逐字切分的，与 LCCC 不同）
```

### 3.5 token 预算（含补足方案）

**目标：33,367,552 × 20 = 667,351,040 token**

#### 第一轮估算（按 62/34/4 配比）

```
pretrain_t2t  7.89GB × 62% = 4.9GB 字 × 76% 合格 = 3.7 亿字
SkyPile       8 片 × 350MB = 2.8GB 字 × 87% 合格 = 2.6 亿字
CCI3-HQ       1 片 × 1GB  = 0.5GB 字 × 61% 合格 = 0.3 亿字
--------------------------------------------------------------
合计 6.6 亿字 ÷ 1.5 字/token = 4.4 亿 token = 13.1×
```

**缺口 2.34 亿 token（7×）。必须补足。**

#### 补足方案（定稿：方案 A + C 组合）

| 步骤 | 动作 | 新增字 | 累计 token | 达到 |
|---|---|---|---|---|
| 基线 | 62/34/4 配比 | 6.6 亿 | 4.4 亿 | 13.1× |
| **A** | **pretrain_t2t 全量用满**（7.89GB 全用，占比提到 76%） | +3.0 亿 | 6.4 亿 | **19.0×** |
| **C** | **SkyPile 加到 16 片** | +1.4 亿 | 7.4 亿 | **22.0×** |

**最终配比（全量用满）：**

| 数据源 | 用量 | 清洗后字 | token | 占比 |
|---|---|---|---|---|
| **pretrain_t2t.jsonl** | **7.89GB 全量** | 6.0 亿字 | **4.0 亿** | **54%** |
| **SkyPile-150B** | **16 片**（约 5.6GB） | 4.9 亿字 | **3.3 亿** | **44%** |
| CCI3-HQ | 1 片 | 0.3 亿字 | 0.2 亿 | 2% |
| **合计** | **约 14.5GB** | 11.2 亿字 | **7.5 亿 token** | **22.3×** ✅ |

**比 20× 多 2.3×，是刻意留的余量**——清洗损耗、去重、超长截断都会吃掉一部分，实测可能落在 6.7-7.5 亿之间。

#### 下载量变化

| 项 | 原计划 | 定稿 | 差异 |
|---|---|---|---|
| pretrain_t2t | 7.89GB | 7.89GB | 不变（本来就是全量下） |
| SkyPile | 8 片 2.8GB | **16 片 5.6GB** | +2.8GB |
| CCI3-HQ | 1GB | 1GB | 不变 |
| **合计** | 11.7GB | **14.5GB** | +2.8GB |

**多下 2.8GB，约多 2 分钟（魔搭 40MB/s）/ 20 分钟（hf-mirror 2.2MB/s）。**

#### 如果清洗后仍不足 6.67 亿 token

**最后的保底：训 1.2 epoch。** 20× 是 Chinchilla 的「算力最优」，不是「效果最优」——小模型适度重复数据是常见做法（SmolLM-135M 训了 4400×）。

重复 20% 的数据，效果损失可忽略；比强行再找新数据源划算。

---

## 四、训练配置

```bash
# 预训练
--seq-len 768              # minimind-3 推荐值，样本 p90=1899 会截断
--micro-batch 32
--grad-accum 2             # 有效 49,152 tok/step
--lr 8e-4                  # 小模型可稍高
--max-steps 13,730         # 6.74亿 / 49,152
--warmup-ratio 0.03
--time-budget 21600        # 6 小时
--save-every 500           # 约 4 分钟一次
```

**seq_len 为什么是 768 而不是 1024：**
- minimind-3 官方推荐 `max_seq_len≈768`
- pretrain_t2t 的 p90=482 字 ≈ 720 token，768 刚好覆盖
- SkyPile p90=1899 字会被截断，这是可接受的损失（长文的前 768 token 仍有信息）

**lr 为什么从 5e-4 提到 8e-4：**
- 33.7M 比 64M 小，可以用更高 lr
- 但要观察前 500 步 loss，若震荡立刻降到 5e-4

---

## 五、硬件与时间

### V100 32GB（百度 AI Studio）

| 项 | 值 |
|---|---|
| 显存占用预估 | 12-14GB（33.7M，seq 768，bs 32×2） |
| 吞吐预估 | 约 55,000 tok/s（V100 125 TFLOPS，带宽 900GB/s） |
| **预训练 6.67 亿 token** | **约 3.4 小时** |
| SFT | 40 分钟 |
| 导出 + 量化 | 15 分钟 |
| **合计** | **约 4.5 小时** |

### ⚠️ V100 没有 bf16

`torch.cuda.is_bf16_supported()` 会返回 **False**，训练会落到 fp16 + GradScaler。

**这是本次唯一的技术风险，必须先验证。**

---

## 六、执行步骤

### 阶段 0：环境（10 分钟）

```bash
# 1. 验证 GPU 和 dtype
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.is_bf16_supported())"

# 2. 装依赖
pip install -q gguf llama-cpp-python

# 3. 确认显存
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
```

### 阶段 1：fp16 烟雾测试（10 分钟）**不能省**

```bash
# 跑 50 步，确认 loss 正常下降、无 NaN
python3 train.py --mode pretrain \
    --data bin --out out/smoke \
    --seq-len 768 --micro-batch 32 --grad-accum 2 \
    --lr 8e-4 --max-steps 50 --probe-steps 0
```

**判据：**
- loss 从 8.8 降到 7.5 以下
- 无 `nan` / `inf`
- 显存 < 20GB

**失败处理：** 如果出 NaN，改用 `--dtype fp32`（慢 2 倍但稳定）。

### 阶段 2：下载数据（20 分钟）

```bash
# pretrain_t2t（7.89GB，魔搭）
python3 smart_dl.py "https://www.modelscope.cn/api/v1/datasets/gongjy/minimind_dataset/repo?Revision=master&FilePath=pretrain_t2t.jsonl" \
    -o data/raw/pretrain_t2t.jsonl --ncpu 32

# SkyPile 8 片（hf-mirror）
for i in 0000 0001 0002 0003 0004 0005 0006 0007; do
    python3 smart_dl.py "https://hf-mirror.com/datasets/Skywork/SkyPile-150B/resolve/main/data/2020-40_zh_head_${i}.jsonl" \
        -o data/raw/skypile_${i}.jsonl --ncpu 32
done

# CCI3-HQ 1 片
python3 smart_dl.py "https://www.modelscope.cn/api/v1/datasets/BAAI/CCI3-HQ/repo?Revision=master&FilePath=data%2Fpart_000000.jsonl" \
    -o data/raw/cci3_0.jsonl --ncpu 32
```

### 阶段 3：清洗混合（30 分钟）

用 `prepare_32m.py`（待写），产出 `data/pretrain.txt` 约 10 亿字。

### 阶段 4：训词表 + 校验（10 分钟）

```bash
python3 train_tokenizer.py --data data/pretrain.txt --out tokenizer --vocab-size 6400
python3 check_tokenizer.py --tokenizer tokenizer/tokenizer.json --strict
```

**必须过：** 多字中文 token > 1000、压缩率 1.45-1.90、往返一致。

### 阶段 5：编码（30 分钟）

```bash
python3 build_dataset.py --mode pretrain \
    --tokenizer tokenizer/tokenizer.json \
    --src data/pretrain.txt --out-dir bin --val-ratio 0.002
```

**17 核并行更快**，或用 `fast_encode.py`。

### 阶段 6：预训练（3.5 小时）

```bash
bash cloud/start_32m.sh
```

### 阶段 7：SFT（40 分钟）

**SFT 数据用 `sft_t2t.jsonl`（13.4GB）的完整版**，不用 mini 版。

```bash
python3 train.py --mode sft \
    --data bin --out out/sft --init out/pretrain/best.pt \
    --seq-len 512 --micro-batch 32 --grad-accum 2 \
    --lr 1e-4 --epochs 3 --time-budget 2400
```

### 阶段 8：导出 + 量化（15 分钟）

```bash
python3 export_gguf.py --state out/sft/best.pt --tokenizer tokenizer \
    --out out/tiny-zh-33m-f16.gguf --outtype f16
# 量化 Q4_K_M
```

### 阶段 9：评测（20 分钟）

用同样的 21 题，对比：

| 模型 | 已知分数 |
|---|---|
| Tiny-ZH-31.5M | 8/105（7.6%） |
| Tiny-ZH-64M v1 | 19/105（18.1%） |
| Tiny-ZH-64M v2 | 42/105（40.0%） |
| minimind-3 原版 | 74/105（70.5%） |
| **Tiny-ZH-33.7M** | **？** |

---

## 七、成功判据

| 等级 | 得分 | 说明 |
|---|---|---|
| **失败** | < 20/105 | 比 64M v2 还差，架构或数据有问题 |
| **及格** | 30-42/105 | 接近 64M v2 的水平（参数少一半但数据好） |
| **良好** | 43-55/105 | **明显超过 64M v2**，证明架构修复有效 |
| **优秀** | > 56/105 | 接近 minimind-3（70.5%），32M 达到这个水平算成功 |

---

## 八、风险与应对

| 风险 | 概率 | 应对 |
|---|---|---|
| **V100 fp16 出 NaN** | **中** | 烟雾测试拦截；降 lr 或改 fp32 |
| 清洗后 token 不足 20× | **高** | 方案 A：pretrain_t2t 全量用 |
| SkyPile 许可证问题 | 低 | 仅用于训练，模型可发布（需确认条款） |
| 编码太慢（4 核 CPU） | 中 | 用 `fast_encode.py` 多进程 |
| 显存 OOM | 低 | 33.7M seq768 bs32 约 13GB，32GB 充裕 |
| 实例中途关机 | 中 | 四层抗断电已实现，`--resume` 续 |

---

## 九、与 minimind-3 的对照（关键）

| 项 | minimind-3 | 我们 33.7M |
|---|---|---|
| 参数量 | 63.91M | **33.37M（47%）** |
| 层数 | 8 | 8（相同） |
| hidden | 768 | 512 |
| FFN/hidden | **3.17** | **4.00** ✅ |
| 每层参数 | 7.37M | 3.80M |
| 预训练 token | 约 2.5 亿（1 epoch） | **6.67 亿（20×）** ✅ |
| **token/参数比** | 4× | **20×** ✅ |

**我们的优势：FFN 更厚 + token 多 5 倍。劣势：参数少 47%。**

**这是「用数据密度换参数」的实验。如果 20× 真的有效，33.7M 应该能到 55-65%。**

---

## 十、待办

- [ ] 写 `prepare_32m.py`（清洗脚本）
- [ ] 写 `cloud/start_32m.sh`（一键训练）
- [ ] 改 `config.py` 为 33.7M
- [ ] 准备 SFT 数据下载
- [ ] V100 实例开机
- [ ] 烟雾测试
- [ ] 正式训练
