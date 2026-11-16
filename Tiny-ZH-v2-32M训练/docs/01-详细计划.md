# E3.1 — Tiny-ZH-33.37M 详细计划

> 2026-11-16 定稿 | 目标：**用一半参数追平 minimind-3 的 70.5%**
> 状态：待执行

---

## 一、目标与判断

### 1.1 要解决的问题

E2.3（64M）拿到 42/105（40.0%），minimind-3（63.9M）是 74/105（70.5%）。

**同样参数量，差 30 个百分点。**

归因（已确认）：

| 项 | E2.3 | minimind-3 |
|---|---|---|
| 预训练 token | 9.42 亿（**更多**） | 约 2.5 亿 |
| 数据主力 | LCCC 55%（**46 字碎片**） | pretrain_t2t（**274 字完整段落**） |
| 训练时长 | 5.4 小时（**更长**） | 2 小时 |
| 得分 | 40.0% | **70.5%** |

**不是 token 少，是 token 的质量密度低。**

### 1.2 本次目标

| 指标 | 目标 |
|---|---|
| 参数量 | 33.37M（minimind-3 的 52%） |
| 预训练 token | **6.9 亿（20.7×）** |
| 21 题得分 | **≥ 43/105（超过 E2.3 的 42）** |
| 理想目标 | 55-65/105 |

### 1.3 核心假设

**token/参数比从 4× 提到 20.7×，能补回一半参数的差距。**

**这个假设没有公开文献支持**——33M 级别没有架构消融数据。这是我们的实验。

**如果失败**，说明 32M 的容量天花板在 45-50%，与数据量无关。

---

## 二、架构

### 2.1 完整配置

```python
ModelConfig(
    vocab_size=6400,
    hidden_size=512,
    num_layers=9,
    num_heads=8,
    num_kv_heads=4,
    head_dim=64,                  # 8×64=512
    intermediate_size=1664,       # FFN/hidden = 3.25
    max_position_embeddings=1024,
    rope_theta=1000000.0,
    rms_norm_eps=1e-5,
    tie_word_embeddings=True,
    initializer_range=0.02,
    dropout=0.0,
)
# 33,367,552 参数
```

### 2.2 参数分解

| 组成 | 参数量 | 占总量 |
|---|---|---|
| embedding（tied with lm_head） | 3,276,800 | 9.82% |
| **每层** | | |
| ├ q_proj [512,512] | 262,144 | |
| ├ k_proj [512,256] | 131,072 | |
| ├ v_proj [512,256] | 131,072 | |
| ├ o_proj [512,512] | 262,144 | |
| ├ attn 小计 | 786,432 | 23.5%（层内） |
| ├ gate_proj [1664,512] | 851,968 | |
| ├ up_proj [1664,512] | 851,968 | |
| ├ down_proj [512,1664] | 851,968 | |
| ├ FFN 小计 | 2,555,904 | 76.4%（层内） |
| ├ 2× RMSNorm | 1,024 | |
| └ 每层合计 | 3,343,360 | |
| **9 层合计** | **30,090,240** | **90.18%** |
| final norm | 512 | |
| **总计** | **33,367,552** | **100%** |

### 2.3 与 minimind-3 的比例对照

| 项 | minimind-3 | 本方案 | 倍数 | 判断 |
|---|---|---|---|---|
| hidden | 768 | **512** | 0.67× | 等比 |
| 层数 | 8 | **9** | 1.13× | 补偿变窄 |
| num_heads | 8 | **8** | 1.00× | 不变 |
| num_kv_heads | 4 | **4** | 1.00× | 不变 |
| head_dim | 96 | **64** | 0.67× | 等比 |
| **FFN/hidden** | **3.17** | **3.25** | **1.03×** ✅ | 几乎相同 |
| GQA 比 | 2:1 | **2:1** | 相同 | 不变 |
| rope_theta | 1e6 | **1e6** | 相同 | 不变 |
| vocab | 6400 | **6400** | 相同 | 不变 |
| 词表占比 | 7.7% | 9.8% | 1.27× | 小模型的必然代价 |
| **参数量** | 63.91M | **33.37M** | 0.52× | 目标 |

**设计原则：不发明任何新结构。除了归并联调，所有比例与 minimind-3 保持一致。**

### 2.4 相比 E2.3（失败的 64M）的改动

| 项 | E2.3 64.82M | 本方案 33.37M | 影响 |
|---|---|---|---|
| hidden | 640 | 512 | 变小 |
| 层数 | 13 | 9 | 变浅 |
| num_heads | 10 | 8 | — |
| num_kv_heads | 5 | 4 | — |
| **FFN/hidden** | **2.80** | **3.25** | **+16%** ✅ |
| head_dim | 64 | 64 | 相同 |
| rope_theta | 5e5 | **1e6** | 长上下文标准 |
| 词表占比 | 6.3% | 9.8% | 上升 |

**唯一实质改动：FFN/hidden 2.80 → 3.25。**

E2.3 的 2.80 低于所有主流小模型（MobileLLM 2.67-2.80、SmolLM2 2.67、Qwen3 3.00、minimind-3 3.17）。提到 3.25 是向主流靠拢。

### 2.5 为什么不用 MobileLLM 的「30 层」

33.37M 预算下，词表占 9.82%（3.28M），剩 30.1M 分给层：

| 层数 | 每层参数 | FFN 能到多少 | 判断 |
|---|---|---|---|
| **9** | 3.34M | **3.25×hidden** | ✅ 比例正常 |
| 16 | 1.88M | 1.5×hidden | ❌ FFN 太薄 |
| 24 | 1.25M | 0.8×hidden | ❌ 崩坏 |
| 30 | 1.00M | 0.3×hidden | ❌ 不可能 |

**MobileLLM-125M 能用 30 层，是因为它总参数 125M，词表只占 14.8%，每层仍有 3.5M。**

**我们 33.37M 撑不起深窄架构。这是参数预算的硬约束，不是选择。**

### 2.6 为什么不用 4.0× FFN

早期我建议「FFN 要 4.0×」，**查证后确认是错的**：

| 模型 | FFN/hidden |
|---|---|
| MobileLLM-125M | 2.67 |
| SmolLM2-135M | 2.67 |
| MobileLLM-350M | 2.67 |
| MobileLLM-1B | 2.80 |
| Qwen3-0.6B | 3.00 |
| minimind-3 | 3.17 |
| Qwen2.5-0.5B | 5.43（异常值） |

主流是 2.67-3.17。3.25 已在上界。

---

## 三、数据

### 3.1 token 预算

```
目标 = 33,367,552 × 20 = 667,351,040 token = 6.67 亿
按 1.45 字/token 换算 = 9.68 亿字
```

**目标定 10 亿字，留 3% 余量。**

### 3.2 三源配比

| 数据源 | 字数 | 占比 | token（估） | 作用 |
|---|---|---|---|---|
| **pretrain_t2t.jsonl** | **6.0 亿** | **60%** | 4.14 亿 | 对话语感，274 字完整段落 |
| **SkyPile-150B** | **3.0 亿** | **30%** | 2.07 亿 | 长文连贯，970 字 |
| **CCI3-HQ** | **1.0 亿** | **10%** | 0.69 亿 | 知识密度，2253 字 |
| **合计** | **10.0 亿字** | 100% | **6.90 亿 = 20.7×** ✅ |

### 3.3 各源实测质量（已验证）

| 数据源 | 样本 | 平均字 | 中文占比 | 合格率 | p90 长度 |
|---|---|---|---|---|---|
| pretrain_t2t | 20,000 | 274 | **0.77** | 76% | 482 |
| SkyPile-150B | 661 | **970** | **0.83** | **87%** | 1899 |
| CCI3-HQ | 3,980 | 2253 | 0.85 | 61% | 4436 |

**合格率定义：中文占比 ≥0.65 且长度 100-2000 字。**

**SkyPile 的 87% 是三源里最干净的。**

### 3.4 需要的下载量

| 数据源 | 全量 | 本计划用量 | 需下载 |
|---|---|---|---|
| pretrain_t2t.jsonl | 7.89GB / 23.02 亿字 | 6.0 亿字（26%） | **2.1 GB** |
| SkyPile-150B | 437 片 | **3 片** × 350MB | **1.0 GB** |
| CCI3-HQ | 679 片 | **1 片** × 1GB | **1.0 GB** |
| **合计** | — | — | **约 4.1 GB** |

**下载时间**：魔搭 40MB/s → 3 分钟；hf-mirror 2.2MB/s → 31 分钟。

**⚠️ 之前估 14.5GB 是高估，实际只需 4.1GB。**

### 3.5 被排除的数据源

| 数据集 | 实测结果 | 排除原因 |
|---|---|---|
| **Infinity-Instruct 7M_core** | 中文占比 **0.04** | **是英文数据集** |
| COIG-PC | 下载 403 | 无权限 |
| **LCCC** | 46 字/段 | **E2.2 已证明有害（碎片化）** |
| **FineWeb-Edu-v2** | 网页噪声 | **E2.2 已证明稀释质量** |
| 中文维基 | 749 字 / 0.73 / 71% | 质量可以，但本计划不用（维基偏百科，不偏对话） |

### 3.6 清洗标准

```python
# 通用门（三源统一）
中文占比      ≥ 0.65
长度          100 - 2000 字      ← 上限定 2000，超长会被 seq 768 截断
必须含句末标点 。！？.!?
单字符重复率  < 25%              ← 防复读
哈希去重      前 64 字 blake2b

# SkyPile 专属
不需要挤空格（不是逐字切分）

# CCI3-HQ 专属
长度收紧到 100-1500 字（p90=4436 太长，只取中段）
```

### 3.7 为什么这样配

| 数据源 | 提供什么 | 为什么是这个比例 |
|---|---|---|
| pretrain_t2t 60% | 对话语感、日常表达、qwen3 蒸馏质量 | **主力**。minimind-3 靠它拿 70.5% |
| SkyPile 30% | 长文连贯、段落结构 | 补 pretrain_t2t 偏短的缺陷（p90 仅 482 字） |
| CCI3-HQ 10% | 知识密度、书面语 | 补前两者的事实性不足，但 p90 太长，只给 10% |

**对比 E2.2 的错误配比：**

| 项 | E2.2（40分） | 本计划 |
|---|---|---|
| LCCC | **55%**（46 字碎片） | **0%** |
| FineWeb | **35%**（网页噪声） | **0%** |
| pretrain_t2t | **0%** ❌ | **60%** ✅ |
| SkyPile | 0% | 30% |

**E2.2 把最好的数据完全排除了，这是根本错误。**

---

## 四、训练配置

### 4.1 预训练

```bash
--mode pretrain
--data /dev/shm/tz/bin
--out out/pretrain
--seq-len 768              # minimind-3 推荐值
--micro-batch 32
--grad-accum 2             # 有效 49,152 tok/step
--lr 8e-4
--min-lr-ratio 0.1         # cosine 衰减到 10%
--warmup-ratio 0.03
--max-steps 13,580         # 6.67亿 / 49,152
--save-every 500           # 约 4 分钟一次
--log-every 50
--time-budget 21600        # 6 小时兜底
```

### 4.2 参数依据

| 参数 | 值 | 依据 |
|---|---|---|
| **seq_len** | **768** | pretrain_t2t p90=482 字 ≈ 720 token，768 刚好覆盖；minimind-3 官方推荐 `≈768` |
| micro_batch | 32 | E2.2 在 64M 上用 32 占 18.5GB；33M 更小，32 只占约 12GB |
| grad_accum | 2 | 有效 batch 64，梯度更稳 |
| **lr** | **8e-4** | 33M 比 64M 小，可用更高 lr（E2.2 是 5e-4） |
| warmup | 3% | 比 E2.2 的 5% 短，小模型不需要太长预热 |
| max_steps | 13,580 | 6.67亿 ÷ 49,152 |

**lr 8e-4 的风险**：比 E2.2 高 60%。**必须观察前 500 步**，若 loss 震荡立刻降到 5e-4。

### 4.3 SFT

```bash
--mode sft
--init out/pretrain/best.pt
--data /dev/shm/tz/bin
--out out/sft
--seq-len 512
--micro-batch 32
--grad-accum 2
--lr 1e-4
--max-steps 3000           # 用完整版 sft_t2t，约 3 epoch
--time-budget 2400
```

**SFT 数据**：`sft_t2t.jsonl`（**13.4GB 完整版，不是 mini 版**）

E2.3 用的是 `sft_t2t_mini`（1.66GB，约 11 万条），完整版有 8 倍数据。

**筛选条件同 E2.3**（已证明有效）：

```
答案长度 70-200 字（长答）
或 10-60 字（短答，占比 15%）
剔除编号列表（正则）
剔除答案含身份标签（minimind/jingyaogong）
中文占比 ≥ 0.55
结尾有标点
```

---

## 五、硬件与时间

### 5.1 V100 32GB（百度 AI Studio）

| 项 | 值 |
|---|---|
| 单价 | 3.0 点/小时 |
| 余额 | 8 点 = **2.7 小时** |
| 显存 | 32GB（需求约 12-14GB，余量充足） |
| CPU | 4 核（**编码会慢，是瓶颈**） |

### 5.2 时间估算

| 阶段 | 耗时 | 说明 |
|---|---|---|
| 环境验证 | 1 分 | torch / cuda / bf16 |
| **fp16 烟雾测试** | **10 分** | **不可跳过** |
| 下载 4.1GB | 5 分（魔搭）/ 31 分（hf-mirror） | |
| 清洗混合（4 核） | **40 分** | CPU 密集 |
| 训词表 | 8 分 | |
| 编码 10 亿字（4 核） | **50 分** | **最大瓶颈** |
| **预训练 6.67 亿** | **3.4 小时** | 55,000 tok/s 估 |
| SFT | 40 分 | |
| 导出 + 量化 | 15 分 | |
| **合计** | **约 6 小时** | |

### 5.3 ⚠️ 时间超预算问题

**8 点余额只够 2.7 小时，但需要 6 小时。**

三个选择：

| 方案 | 做法 | 结果 |
|---|---|---|
| **A** | 分两次跑（2.7h + 3.3h），用 `--resume` 续训 | 需要两个时段 |
| **B** | 砍 token 到 10×（3.3 亿） | 2.9 小时，但不达 20× 目标 |
| **C** | 充值到 20 点 | 6.7 小时，一次跑完 |

**推荐 C**（如果预算允许），否则 A。

---

## 六、执行步骤

### 阶段 0：环境（5 分钟）

```bash
# 1. GPU 和 dtype
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.is_bf16_supported())"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# 2. 依赖
pip install -q gguf llama-cpp-python
```

**预期**：`bf16_supported = False`（V100 不支持），训练会落 fp16。

### 阶段 1：fp16 烟雾测试（10 分钟）**不可跳过**

```bash
# 用随机数据跑 50 步
python3 train.py --mode pretrain --data /tmp/smoke_bin --out /tmp/smoke_out \
    --seq-len 768 --micro-batch 32 --grad-accum 2 \
    --lr 8e-4 --max-steps 50 --probe-steps 0
```

**判据：**
- ✅ loss 从 8.8 降到 7.5 以下
- ✅ 无 `nan` / `inf`
- ✅ 显存峰值 < 20GB
- ✅ 50 步耗时 < 5 分钟

**失败处理**：
- 若出 NaN → `--lr 4e-4`
- 仍失败 → `--dtype fp32`（慢 2 倍但稳定）

### 阶段 2：下载数据（5-31 分钟）

```bash
# pretrain_t2t（只要前 2.1GB，用 Range 或下载后截取）
python3 smart_dl.py "https://www.modelscope.cn/api/v1/datasets/gongjy/minimind_dataset/repo?Revision=master&FilePath=pretrain_t2t.jsonl" \
    -o data/raw/pretrain_t2t_head.jsonl --ncpu 32 --head-bytes 2200000000

# SkyPile 3 片
for i in 0000 0001 0002; do
  python3 smart_dl.py "https://hf-mirror.com/datasets/Skywork/SkyPile-150B/resolve/main/data/2020-40_zh_head_${i}.jsonl" \
      -o data/raw/skypile_${i}.jsonl --ncpu 32
done

# CCI3-HQ 1 片
python3 smart_dl.py "https://www.modelscope.cn/api/v1/datasets/BAAI/CCI3-HQ/repo?Revision=master&FilePath=data%2Fpart_000000.jsonl" \
    -o data/raw/cci3_0.jsonl --ncpu 32
```

### 阶段 3：清洗混合（40 分钟）

```bash
python3 code/data/prepare_32m.py --target-chars 1000000000
```

产出：`data/pretrain.txt`（约 10 亿字，2.8GB）

### 阶段 4：训词表（8 分钟）

```bash
python3 code/train_tokenizer.py --data data/pretrain.txt --out tokenizer --vocab-size 6400
python3 code/check_tokenizer.py --tokenizer tokenizer/tokenizer.json --strict
```

**必须过三项：**
1. 含 ≥2 汉字的 token > 1000（1.55 压缩率下应有 3000+）
2. 压缩率 1.45-1.90
3. 编解码往返一致

### 阶段 5：编码（50 分钟）

```bash
# 用多进程，比单进程快 6 倍
python3 code/fast_encode.py \
    --src data/pretrain.txt \
    --tokenizer tokenizer/tokenizer.json \
    --out bin/pretrain.bin --workers 4
```

### 阶段 6：预训练（3.4 小时）

```bash
bash cloud/start_32m.sh
```

### 阶段 7：SFT（40 分钟）

### 阶段 8：导出 + 量化（15 分钟）

### 阶段 9：评测（20 分钟）

用相同的 21 题，对比四个模型：

| 模型 | 已知分数 |
|---|---|
| Tiny-ZH-31.5M | 8/105（7.6%） |
| Tiny-ZH-64M-v1 | 19/105（18.1%） |
| Tiny-ZH-64M-v2 | 42/105（40.0%） |
| minimind-3 | 74/105（70.5%） |
| **Tiny-ZH-33.37M** | **？** |

---

## 七、成功判据

| 等级 | 得分 | 含义 |
|---|---|---|
| **失败** | < 20/105 | 比 E2.3 的 42 还差，架构或数据有严重问题 |
| 及格 | 20-42 | 不如或持平 E2.3（参数少一半，持平算正常） |
| **良好** | **43-55** | **超过 E2.3，证明架构 + 数据修复有效** |
| 优秀 | 56-70 | 接近 minimind-3 |
| 突破 | > 70 | 超过 minimind-3（32M 打 64M） |

---

## 八、风险与应对

| 风险 | 概率 | 影响 | 应对 |
|---|---|---|---|
| **V100 fp16 出 NaN** | **中** | 训练全废 | 烟雾测试拦截；降 lr 到 4e-4；仍失败改 fp32 |
| **8 点余额不够 6 小时** | **高** | 中途断 | 分两次 + `--resume`，或充值 |
| 4 核编码太慢（50 分） | 高 | 超时 | `fast_encode.py` 4 进程 |
| 清洗后不足 6.67 亿 token | 中 | 不达 20× | 加 2 片 SkyPile |
| lr 8e-4 太高震荡 | 中 | loss 不降 | 前 500 步观察，立刻降到 5e-4 |
| 显存 OOM | 低 | 中断 | 32GB 需求 14GB，降 bs 到 24 |
| 实例中途关机 | 中 | 中断 | 四层抗断电，`--resume` 续 |

---

## 九、与 minimind-3 的完整对照

| 项 | minimind-3 | 本方案 | 差异 |
|---|---|---|---|
| **架构** | | | |
| 参数 | 63.91M | **33.37M** | 52% |
| 层数 | 8 | 9 | +1 |
| hidden | 768 | 512 | 67% |
| num_heads | 8 | 8 | 相同 |
| num_kv_heads | 4 | 4 | 相同 |
| head_dim | 96 | 64 | 67% |
| FFN/hidden | 3.17 | **3.25** | 103% |
| rope_theta | 1e6 | 1e6 | 相同 |
| vocab | 6400 | 6400 | 相同 |
| **训练** | | | |
| 预训练 token | 约 2.5 亿 | **6.67 亿** | 267% |
| token/参数 | 4× | **20×** | 5× |
| seq_len | 768 | 768 | 相同 |
| 训练时长 | 2 小时（3090） | 3.4 小时（V100） | — |
| **数据** | | | |
| 预训练主力 | pretrain_t2t | pretrain_t2t **60%** | 相同源，更集中 |
| 补充 | 未详 | SkyPile 30% + CCI3 10% | 增加长文和知识 |
| SFT | sft_t2t_mini | **sft_t2t 完整版** | 8 倍数据 |
| **结果** | | | |
| 21 题得分 | **70.5%** | **？** | 待测 |

---

## 十、执行清单

### 开训前
- [ ] GPU 可用，显存 ≥ 30GB
- [ ] `bf16_supported` 已确认（预期 False）
- [ ] 烟雾测试通过（50 步无 NaN）
- [ ] 磁盘 ≥ 50GB 可用
- [ ] `/dev/shm` ≥ 8GB

### 数据
- [ ] 下载 4.1GB，校验完整性
- [ ] 清洗后字数 ≥ 9.5 亿
- [ ] 词表三项校验通过
- [ ] 编码后 token 数 ≥ 6.4 亿

### 训练后
- [ ] 预训练 val loss 持续下降无震荡
- [ ] GGUF 导出无报错
- [ ] `llama-tokenize` 逐 id 一致
- [ ] `llama-cli` 输出中文不崩
- [ ] 21 题评测完成并归档

---

## 附录：文件索引

| 需要什么 | 在哪 |
|---|---|
| 架构历史与依据 | [docs/literature.md](../docs/literature.md) |
| 踩过的坑 | [docs/lessons.md](../docs/lessons.md) |
| 环境与命令 | [docs/infrastructure.md](../docs/infrastructure.md) |
| 上次失败的完整记录 | [04-64m-v2.md](04-64m-v2.md) |
| minimind-3 对比数据 | [archive/three_way2.md](../archive/three_way2.md) |
| 代码 | [code/README.md](../code/README.md) |