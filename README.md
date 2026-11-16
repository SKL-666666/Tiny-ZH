# Tiny-ZH

> 从零训练中文小模型的完整探索记录。
> **不是成果展示**——记录每次尝试、每个失败、每条结论。

---

## 四个部分

| 目录 | 内容 | 状态 | 得分 |
|---|---|---|---|
| [**模型微调（数学）**](模型微调（数学）/) | MiniCPM5-1B + LoRA，学毛选文风 | 已完结 | 真人 97% |
| [**Tiny-ZH-v1-32M训练**](Tiny-ZH-v1-32M训练/) | 从零训练 31.47M | 已完结 | **8/105（7.6%）** |
| [**Tiny-ZH-v1-64M训练**](Tiny-ZH-v1-64M训练/) | 从零训练 64.82M，两个 SFT 版本 | 已完结 | **42/105（40.0%）** |
| [**Tiny-ZH-v2-32M训练**](Tiny-ZH-v2-32M训练/) | 33.37M，换正确数据配比 | **进行中** | 目标 ≥43 |

**对照组**：minimind-3 原版（63.91M）= 74/105（70.5%）

---

## 演进路线

```
模型微调（数学）
  LoRA 学文风成功，四轮数学全败
  └─ 确立：1B + LoRA 改不了能力，只能改分布
        ↓
Tiny-ZH-v1-32M（31.47M，3.09 亿 token）
  格式全对，内容 70% 错，5 条崩成英文
  └─ 教训：容量不足会崩语言
        ↓
Tiny-ZH-v1-64M（64.82M，9.42 亿 token）
  v1 用 ShareGPT → 19 分（编号列表、模板回避）
  v2 换 minimind → 42 分（翻倍）
  └─ 教训：数据质量比 token 数重要
        ↓
Tiny-ZH-v2-32M（33.37M，6.67 亿 token）
  用 pretrain_t2t + SkyPile，剔除 LCCC/FineWeb
  └─ 验证：一半参数能否靠好数据追平
```

---

## 两条最重要的结论

### 1. 判断任务能不能做，先问「长出」还是「偏向」

| 需求 | 本质 | 小模型能做吗 |
|---|---|---|
| 文风、格式、语气 | 偏向某个方向 | ✅ 可以 |
| 推理、知识 | 长出新的能力 | ❌ 不行 |

### 2. 数据质量比 token 数重要

| 项 | 我们 v1-64M | minimind-3 |
|---|---|---|
| token | 9.42 亿（**更多**） | 约 2.5 亿 |
| 数据 | LCCC 55%（**46 字碎片**） | pretrain_t2t（**274 字**） |
| 时长 | 5.4 小时（**更长**） | 2 小时 |
| 得分 | 40.0% | **70.5%** |

**为凑 20× 用碎片填充，模型学不会长程连贯。**

---

## 能力边界（实测）

| 参数 | 能做到 | 做不到 |
|---|---|---|
| 31.5M | 格式正确 | 任何事实（**崩英文**） |
| 64M | 格式 + 少量常识 | 具体知识、推理 |
| 300M | + 一些常识 | 复杂推理 |
| 1B+ | 开始有可用知识 | 仍然大量幻觉 |

---

## 通用资料

| 文档 | 内容 |
|---|---|
| [docs/00-overview.md](docs/00-overview.md) | 项目全貌 |
| [docs/lessons.md](docs/lessons.md) | 踩坑清单（7 类 40+ 条） |
| [docs/literature.md](docs/literature.md) | 文献查证（含查不到的） |
| [docs/infrastructure.md](docs/infrastructure.md) | 环境、SSH、下载、抗断电 |
| [NOW.md](NOW.md) | 当前进度 |
| [code/](code/) | 共享代码库 |

---

## 产物

| 内容 | 位置 |
|---|---|
| 训练数据（29.27 亿 token） | [HF: tiny-zh-64m-data](https://huggingface.co/datasets/SKL666666/tiny-zh-64m-data) |
| 模型 + GGUF | [HF: tiny-zh-64m](https://huggingface.co/SKL666666/tiny-zh-64m) |
| 训练日志 + 测试 | [HF: tiny-zh-64m-logs](https://huggingface.co/datasets/SKL666666/tiny-zh-64m-logs) |
| 魔搭镜像 | [ModelScope](https://www.modelscope.cn/models/SKL000000/tiny-zh-64m) |

---

## 目录结构

```
Tiny-ZH/
├── README.md                     ← 本文件
├── NOW.md                        当前进度
├── 模型微调（数学）/             LoRA 路线
├── Tiny-ZH-v1-32M训练/           31.47M
├── Tiny-ZH-v1-64M训练/           64.82M（v1 + v2）
├── Tiny-ZH-v2-32M训练/           33.37M（进行中）
├── docs/                         通用资料
├── code/                         共享代码
├── tools/                        llama.cpp + 通用脚本
└── archive/                      早期文档与一次性脚本
```