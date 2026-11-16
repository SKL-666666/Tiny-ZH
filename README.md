# Tiny-ZH 探索日志

> **本仓库是探索记录，不是成果展示。**
> 记录从零训练中文小模型的每一次尝试、每一个失败、每一条结论。
> 模型产物在 HuggingFace / 魔搭，这里只留过程和判断。

---

## 快速导航

| 我想 | 去哪里 |
|---|---|
| 了解项目全貌 | [docs/00-overview.md](docs/00-overview.md) |
| 看当前在做什么 | [NOW.md](NOW.md) |
| 查历史每次尝试 | [experiments/](experiments/) |
| 读沉淀的经验 | [docs/lessons.md](docs/lessons.md) |
| 直接跑代码 | `code/` |

---

## 三个阶段

### 阶段一：Mao-LoRA（已完结）

用 MiniCPM5-1B 做 LoRA 学《毛泽东选集》文风。确立核心判断：
**1B + LoRA 改不了能力，但能改表达分布。**

→ [experiments/01-mao-lora.md](experiments/01-mao-lora.md)

### 阶段二：从零训练（已完结）

三次独立尝试，全部做完：

| 编号 | 模型 | 参数 | token | 测试得分 | 状态 |
|---|---|---|---|---|---|
| E2.1 | Tiny-ZH-31.5M | 31.47M | 3.09 亿（9.8×） | **8/105（7.6%）** | 已交付 |
| E2.2 | Tiny-ZH-64M-v1 | 64.82M | 9.42 亿（14.5×） | **19/105（18.1%）** | 已交付 |
| E2.3 | Tiny-ZH-64M-v2 | 64.82M | 9.42 亿（14.5×） | **42/105（40.0%）** | 已交付 |

对照组：**minimind-3 原版（63.91M）= 74/105（70.5%）**

→ [E2.1 31.5M](experiments/02-31.5m.md) · [E2.2 64M-v1](experiments/03-64m-v1.md) · [E2.3 64M-v2](experiments/04-64m-v2.md)

### 阶段三：架构优化（进行中）

**当前：Tiny-ZH-33.37M**（minimind-3 等比缩小 + 数据配比修正）

→ [experiments/05-33m-plan.md](experiments/05-33m-plan.md) ← **正在做**
→ [NOW.md](NOW.md) 实时进度

---

## 两条最重要的结论

### 1. 64M 输给 minimind-3，根因是数据不是 token 数

| 项 | 我们 64M | minimind-3 |
|---|---|---|
| 预训练 token | 9.42 亿（**更多**） | 约 2.5 亿 |
| 数据主力 | LCCC 55%（**46 字碎片**） | pretrain_t2t（**274 字完整段落**） |
| 训练时长 | 5.4 小时（**更长**） | 2 小时 |
| 得分 | 40.0% | **70.5%** |

**为凑 20× 用碎片数据填充，让模型学不会长程连贯。指标不能凌驾于数据质量。**

### 2. FFN/hidden 主流是 2.67-3.17，不是 4.0

查证六个模型真实 `config.json`：

| 模型 | hidden | 层数 | FFN/hidden |
|---|---|---|---|
| MobileLLM-125M | 576 | 30 | 2.67 |
| SmolLM2-135M | 576 | 30 | 2.67 |
| MobileLLM-1B | 1280 | 54 | 2.80 |
| Qwen3-0.6B | 1024 | 28 | 3.00 |
| minimind-3 | 768 | 8 | 3.17 |

**11-15 的 64M 用 2.80，本身没问题。真正的问题是数据。**

---

## 能力边界

| 参数 | 能做到 | 做不到 |
|---|---|---|
| 31.5M | 格式正确、语法通顺 | 任何事实、任何推理（**会崩成英文**） |
| 64M | 格式 + 少量常识 | 具体知识、逻辑推理 |
| 300M | 格式 + 一些常识 | 复杂推理 |
| 1B+ | 开始有可用知识 | 仍然大量幻觉 |

**31.5M 实测：格式 100% 正确，内容 70% 错。这不是训练方法问题，是容量问题。**

---

## 目录结构

```
Tiny-ZH/
├── README.md               ← 本文件
├── NOW.md                  ← 当前进度
├── docs/
│   ├── 00-overview.md      项目全貌
│   ├── lessons.md          踩坑清单
│   ├── literature.md       文献查证
│   └── infrastructure.md   环境与运维
├── experiments/            每次实验的完整记录
│   ├── 01-mao-lora.md
│   ├── 02-31.5m.md
│   ├── 03-64m-v1.md
│   ├── 04-64m-v2.md
│   └── 05-33m-plan.md      ← 进行中
├── code/                   训练代码
└── archive/                早期文档（保留原貌）
```

---

## 产物位置

| 内容 | 位置 |
|---|---|
| 训练数据（29.27 亿 token） | [HF: tiny-zh-64m-data](https://huggingface.co/datasets/SKL666666/tiny-zh-64m-data) |
| 模型 + GGUF | [HF: tiny-zh-64m](https://huggingface.co/SKL666666/tiny-zh-64m) |
| 训练日志 + 测试 | [HF: tiny-zh-64m-logs](https://huggingface.co/datasets/SKL666666/tiny-zh-64m-logs) |
| 魔搭镜像 | [ModelScope](https://www.modelscope.cn/models/SKL000000/tiny-zh-64m) |

---

## 一句话

**真正的价值不是模型（能力有限），而是这条链路和这份失败清单。**
从零到可运行，改几个参数就能训别的规模。