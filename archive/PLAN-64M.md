# Tiny-ZH-64M 训练计划（最终版）

> 目标：64.82M 中文对话模型，**20 倍数据（12.96 亿 token）**，7 小时级
> 方向：偏闲聊，但要能答常识；数据要语句完整的中文稍长对话
> 更新于 2026-11-14 20:00，GPU 实例已离线，明天开机执行

---

## 一、实测基线（全部在 A10 上跑出来的，不是估的）

| 项 | 值 | 来源 |
|---|---|---|
| 参数量 | **64,816,000** | `python config.py` |
| 架构 | 13 层 × 640 维，10 heads / 5 kv，ffn 1792，seq 1024 | 定稿 |
| 吞吐 | **51,001 tok/s** | `bench64.py` 实测 |
| 最佳形状 | seq1024 × micro_batch 32 | 实测扫描 |
| 显存 | bs32 占 20.7GB；bs64 直接 OOM | 实测 |
| 有效算力 | 19.8 TFLOPS | 吞吐反算 |
| **20× 目标** | **12.96 亿 token** | 64,816,000 × 20 |
| **20× 预训练耗时** | **7.06 小时** | 12.96 亿 ÷ 51,001 |

**决策：保倍率，训练多久都行。** 跑满 20× 约 7 小时预训练 + 40 分 SFT + 20 分导出传输 ≈ **8.5 小时**。

---

## 二、架构

```python
ModelConfig(
    vocab_size=6400, hidden_size=640, num_layers=13,
    num_heads=10, num_kv_heads=5, head_dim=64,
    intermediate_size=1792, max_position_embeddings=1024,
    rope_theta=500000.0,
)
# 64,816,000 参数
```

- 13 层 640 维，深而窄（MobileLLM：同参数下深优于宽）
- 词表占 4.10M / 64.82M = **6.3%**（上限 10%）
- RoPE 交错配对 → llama.cpp `LLAMA_ROPE_TYPE_NORM`，导出零重排
- GQA 10/5，KV cache 减半

---

## 三、数据集清单（最终确定）

### 3.1 总量核算

| # | 数据集 | 魔搭路径 / 文件 | 原始大小 | token | 依据 |
|---|---|---|---|---|---|
| 1 | LCCC large | `silver/lccc` → `lccc_large.jsonl.gz` | 607 MB | **3.74 亿** | 实测 ✅ |
| 2 | ShareGPT-zh-70k | `zhuangxialie/SFT-Chinese-Dataset` → `shareAIShareGPT-Chinese-English-90k/common_zh_70k.jsonl` | 460 MB | **0.86 亿** | 实测 ✅ |
| 3 | ShareGPT-unknow-38k | 同上 → `shareAIShareGPT-Chinese-English-90k/unknow_zh_38k.jsonl` | 144 MB | **0.43 亿** | 实测 ✅ |
| 4 | 中文维基 | `AI-ModelScope/wikipedia-cn-20230720-filtered` → `wikipedia-cn-20230720-filtered.jsonl` | 497 MB | **1.68 亿** | 实测 ✅ |
| 5 | COIG-CQIA full | `AI-ModelScope/COIG-CQIA` → `COIG-CQIA-full.jsonl` | 143 MB | **0.40 亿** | 估 |
| — | **小计** | | **1.85 GB** | **7.11 亿** | |
| 6 | **FineWeb-Edu-v2 #0** | `opencsg/chinese-fineweb-edu-v2` → `data/00000.parquet` | 928 MB | **4.58 亿** | v1 实测等比 |
| 7 | **FineWeb-Edu-v2 #1** | 同上 → `data/00001.parquet` | 927 MB | **4.58 亿** | 同上 |
| — | **合计** | | **3.71 GB** | **16.27 亿** | **25.1×** |

**为什么下 2 片而不是 1.3 片**：片是最小下载单位。下 2 片得 16.27 亿，清洗通常砍掉 15-25%，砍完仍有 12.2-13.8 亿，稳过 12.96 亿。下 1 片只有 11.7 亿，不够。

**实测锚点推导（不是拍脑袋）**：
- FineWeb-Edu **v1** 实测：每片 801MB（parquet）→ 46.9 亿字符 → **3.95 亿 token**，压缩率 1.19 字/token
- v2 每片 928MB，为 v1 的 1.159 倍 → 3.95 × 1.159 = **4.58 亿 token**
- **明天开机第一件事就是验证这个数**（见 §七）

### 3.2 备选数据源（v2 实测偏低时启用）

| 数据集 | 路径 | 规模 | 特点 |
|---|---|---|---|
| **BAAI/CCI3-HQ** | 魔搭 `BAAI/CCI3-HQ` → `data/part_*.jsonl` | 672 片 × 1GB ≈ 650GB | BAAI 高质量中文，**魔搭内网加速** |
| BAAI/CCI3-Data | 魔搭 `BAAI/CCI3-Data` | 1185 片 | 更大量，质量略低 |
| BAAI/COIG-PC | `BAAI/COIG-PC` → `data/` | 696 片 | COIG 预训练版 |
| BAAI/Infinity-Instruct | `BAAI/Infinity-Instruct` → `7M_core/*.parquet` | 15 片 | 指令数据精选 |
| pleisto/wikipedia-cn | hf-mirror | 维基镜像 | 备份源 |
| shibing624/alpaca-zh | hf-mirror | 小 | 指令补充 |

### 3.3 清洗后配比（目标 12.96 亿）

| 类别 | token | 占比 | 作用 |
|---|---|---|---|
| 对话（LCCC + ShareGPT） | 5.5 亿 | **42%** | 语感、口语化、偏闲聊 |
| 网页/教育（FineWeb-v2） | 5.5 亿 | 42% | 完整长句、知识面 |
| 百科（维基） | 1.5 亿 | 12% | 事实性表达 |
| 社区（COIG） | 0.4 亿 | 3% | 生活场景 |

**对话占 42%，偏闲聊，符合定位。**

---

## 四、🔴 三个已确认的数据坑（都用实测证实过）

### 坑 1：LCCC 是逐字切分的，空格占 38.1%

```
原始：火锅 我 在 重庆 成都 吃 了 七八 顿 火锅
必须：我在重庆成都吃了七八顿火锅
```

实测：20 万段样本，原样 14.8M 字符，去空格后 9.2M 字符，**空格占 38.1%**。
不挤掉等于白扔 1.4 亿 token 的预算。

**处理**：正则去掉汉字/中文标点之间的空格，保留英文单词之间的。
```python
_CJK = r"\u4e00-\u9fff\u3000-\u303f\uff00-\uffef"
RE_CJK_SPACE = re.compile(rf"(?<=[{_CJK}])\s+(?=[{_CJK}])")
```

### 坑 2：LCCC 每行是 list，不是 dict

```json
["火锅 我 在 重庆 成都 吃 了 七八 顿 火锅", "哈哈哈哈 ！ 那 我 的 嘴巴 可能 要 烂掉 ！", "不会 的 就是 好 油腻"]
```

写成 `o.get("dialog")` 会崩或读到 0 条。必须兼容两种：
```python
if isinstance(obj, list):
    dialog = obj
elif isinstance(obj, dict):
    dialog = obj.get("dialog") or obj.get("conversations") or []
```

### 坑 3：ShareGPT 字段是 `conversation[].human/assistant`

不是 `conversations[].from/value`。实测 top keys = `['conversation_id','category','conversation']`，turn keys = `['assistant','human']`。

旧版 `fetch_80m.py` 就是写错的，会读到 0 条。

### 坑 4：LCCC 平均只有 46 字/段

要"语句完整的中文稍长对话"，就得设长度下限。
- 不过滤：3.74 亿 token
- ≥50 字过滤：约 **1.5-2.0 亿**（砍掉一半以上）

**本计划按「≥30 字」折中**，估算 LCCC 产出降到约 **2.6 亿**。
所以 §3.1 的 16.27 亿里，LCCC 那 3.74 亿要打 7 折看。

**这意味着可能需要第 3 片 v2。明天实测后确认。**

---

## 五、清洗标准

```
对话类（LCCC / ShareGPT / COIG）：
  - 中文占比 > 0.60
  - 单段长度 30 - 4000 字
  - 必须含句末标点（。！？）
  - 单字符重复率 < 30%（防复读）
  - 前 64 字 blake2b 哈希去重

LCCC 专属：
  - 去字间空格（见坑 1）
  - 判 list / dict 两种格式（见坑 2）

正文类（维基 / FineWeb）：
  - 中文占比 > 0.65
  - 长度 100 - 4000 字
  - 英文数字占比 < 15%

产物：
  data/pretrain.txt   目标 4.5 GB ≈ 12.96 亿 token
  data/sft.jsonl      ShareGPT 里转成 user/assistant 的对话
```

---

## 六、训练配置

```bash
# 预训练
--seq-len 1024 --micro-batch 32 --grad-accum 1 \
--lr 5e-4 --max-steps 39550 --time-budget 25400 \
--save-every 250 --log-every 25

# SFT
--seq-len 512 --micro-batch 32 --grad-accum 2 \
--lr 1e-4 --max-steps 1200 --time-budget 2100
```

- **grad_accum=1**：bench 就是按 1 测的，训练必须同口径，否则时间预算算错
- **39550 步** × 32768 tok/step = 12.96 亿
- **save-every 250**：250 × 32768 ÷ 51001 = **161 秒**，即每 2.7 分钟存一次
- lr 5e-4，cosine 衰减到 10%，5% warmup

---

## 七、明天的执行顺序

### 开机后 0-5 分钟：连 SSH + 验证 GPU

```bash
# 1. 装 tailscale（32 路分段下载，自动取最新版）
bash cloud/ts_up.sh

# 2. 本机连（不加 --ssh，用密钥）
ssh -o StrictHostKeyChecking=no root@100.x.x.x

# 3. 第一件事验证 GPU
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
```
**返回 False 就停下来查驱动，不要传数据。**

### 5-15 分钟：验证 FineWeb-v2 单片产出

```bash
bash cloud/verify_fwv2.sh
```
只下一次 928MB，4 分钟，报告：
- 实际 token 数
- 与推算 4.58 亿的偏差
- 需要几片才能补足缺口

**这步决定下 2 片还是 3 片。**

### 15-50 分钟：下载全部数据

```bash
bash cloud/fetch_all.sh
```
- 魔搭 4 源（内网）
- hf-mirror v2 N 片（32 路并发）
- 每步有断点续传

### 50-90 分钟：清洗 + 编码

```bash
bash cloud/prepare_and_encode.sh
```
- 清洗到 4.5GB（含长度过滤、去空格、去重）
- 训词表 + 三项校验
- 编码成 uint16 bin
- 复制到 /dev/shm

### 90 分钟起：训练

```bash
bash cloud/start64.sh
```
- 预训练 7 小时（25400s 预算）
- SFT 35 分钟
- 导出 GGUF

---

## 八、抗突然关机（四层，已实测验证）

| 层 | 做法 | 效果 |
|---|---|---|
| 1 | 所有产物写 `/mnt/workspace/tiny-zh-64m/` | 实例销毁不丢 |
| 2 | checkpoint 原子写盘：`.tmp` → `fsync` → `rename` | 断电不留半个文件 |
| 3 | 每次存 `step_NNNN.pt`（留最近 3 个），`last.pt` 用硬链接 | `last.pt` 坏了能退到备份 |
| 4 | checkpoint 存 `total_steps`，续训沿用 | LR 余弦曲线不错位 |

**最坏情况损失：2.7 分钟（一个 save 周期）。**

`selfcheck_64m.sh` 已实测通过：
```
原子覆盖 OK  step = 2
残留 .tmp 不影响读取 OK
从 step_250.pt 恢复 step=250 OK
硬链接替换 OK
```

**续训命令**（幂等，重跑即可）：
```bash
bash cloud/start64.sh --resume
```

---

## 九、下载策略（实测数据）

| 方式 | 实测速度 | 用在哪 |
|---|---|---|
| 魔搭 SDK | 1.35 MB/s（今天实测，非文档说的 40） | 4 个小文件 |
| hf-mirror 单连接 | 150 KB/s | 不用 |
| **hf-mirror 32 路并发** | **2.2 MB/s** | FineWeb-v2 大文件 |
| 魔搭 Web API 直链 | ~3.7 MB/s（今天下 928MB 用了 247s） | 备选 |

**通用做法**：大文件一律先 Range 探测 `content-length`，再切 32 段并发下，最后拼接 + 校验（`gzip -t` 或 parquet 读一遍）。

`smart_dl.py` 会实现这套逻辑。

---

## 十、已知风险与应对

| 风险 | 概率 | 应对 |
|---|---|---|
| v2 单片产出远低于 4.58 亿 | 中 | 加片；或改用 CCI3-HQ（魔搭内网） |
| LCCC 长度过滤后不足 2.6 亿 | 高 | 已有预案：多下 v2 |
| 数据总量仍不足 12.96 亿 | 中 | 降级到 80% 目标（10.4 亿 = 16×）；或重复 LCCC 1 轮 |
| 8 核编码慢 | 中 | 编码 12.96 亿 token 约 25-35 分钟；可边编边训 |
| 显存 OOM | 低 | bs32 已验证 20.7GB；降 bs 到 24 |
| 实例中途关机 | 中 | 四层抗关机，重跑 `--resume` |
| SSH 连不上 | 中 | 见 §七，不加 `--ssh`，用密钥直连 |

---

## 附：8.5 小时时间表

| 阶段 | 耗时 | 累计 |
|---|---|---|
| 连接 + GPU 验证 | 5 分 | 0:05 |
| 验证 v2 产出 | 10 分 | 0:15 |
| 下载全部数据 | 35 分 | 0:50 |
| 清洗混合 | 25 分 | 1:15 |
| 训词表 + 校验 | 8 分 | 1:23 |
| 编码 12.96 亿 token | 30 分 | 1:53 |
| 复制内存盘 | 4 分 | 1:57 |
| **预训练 12.96 亿** | **7:04** | **9:01** |
| SFT | 35 分 | 9:36 |
| 导出 GGUF | 5 分 | 9:41 |
| 推 GitHub + 回传 | 15 分 | **9:56** |

约 10 小时。若要从早上 9 点开始，晚上 7 点前能拿到模型。
