# 代码索引

> 按用途分类。所有脚本零 transformers 依赖（除标注的）。

---

## 核心训练链

| 文件 | 作用 | 状态 |
|---|---|---|
| `config.py` | 模型与训练超参（唯一配置源） | 稳定 |
| `model.py` | 模型定义，Llama 同构，零外部依赖 | 稳定 |
| `train.py` | 训练主循环（断点续训 + 时间预算 + loss 尖峰保护） | 稳定 |
| `train_tokenizer.py` | 自训 BPE 词表 | 稳定 |
| `build_dataset.py` | 编码成 uint16 bin | 稳定 |
| `export_gguf.py` | state_dict → GGUF（自写导出器） | 稳定 |

---

## 推理与部署

| 文件 | 作用 |
|---|---|
| `generate.py` | PyTorch 推理 |
| `serve.py` | 本机 Web 界面（OpenAI 兼容代理） |
| `eval.py` | 客观判据 M1-M7 |

---

## 工具

| 文件 | 作用 | 重要 |
|---|---|---|
| **`check_tokenizer.py`** | **词表三项校验** | ⚠️ 用了正确的字节解码口径，网上流传的代码是错的 |
| `verify_tokenizer.py` | GGUF 分词与本机 tokenizers 逐 id 比对 | 必备 |
| `smart_dl.py` | 测速选源 + 分段并发 + 断点续传 | 下载必备 |
| `fast_encode.py` | 多进程并行编码（比单进程快 6 倍） | 大数据集必备 |
| `collect_experiment.py` | 采集实验记录（论文用） | 出报告前跑 |

---

## code/data/ — 数据准备

| 文件 | 作用 |
|---|---|
| `prepare_64m.py` | 64M 数据的清洗混合（含 LCCC 去空格、ShareGPT 字段修正） |
| `prepare_sft_fresh.py` | SFT 数据构建（答案 70-200 字 + 剔除编号列表） |
| `build_sft_high.py` | 同上，更严格的版本 |
| `clean_identity.py` | 清理 SFT 数据里的身份标签（minimind / jingyaogong） |

---

## code/probe/ — 测试与探测（一次性工具）

### 数据质量探测

| 文件 | 用途 |
|---|---|
| `probe_data_quality.py` | 测四个源的样本数/长度/中文占比 |
| `probe_candidates.py` | 用 Range 请求测候选数据集 |
| `probe_five.py` | 测用户指定的五个数据集 |
| `probe_ms.py` / `probe_rest.py` | 补测特定源 |

### 模型评测

| 文件 | 用途 |
|---|---|
| `strict_test.py` | 21 题严厉测试（含 evasion 检测、一致性） |
| `test_32m_local.py` | 本机 CPU 跑 31.5M 的 21 题 |
| `three_way_test.py` | 三方对比（含 minimind-3 原版） |
| `three_way_fair.py` | 公平版（统一 max_tokens 和停止符） |
| `compare_v1_v2.py` | v1 vs v2 对比 |

### 其他

| 文件 | 用途 |
|---|---|
| `bench_amd.py` | ROCm 吞吐 benchmark |
| `smoke_export_80m.py` | 80M 配置的导出预检 |
| `quantize_q4km.py` | 纯 Python 量化框架（**不完整**，实际用 llama-cpp-python） |
| `get_minimind3.py` | 下载 minimind-3 原版用于对比 |
| `analyze_fails.py` / `show_hard.py` | 分析测试失败案例 |

---

## 关键设计说明

### 为什么零 transformers 依赖

`model.py` 只用 PyTorch 原生算子实现 Llama 架构。好处：
1. 没有版本兼容问题
2. RoPE 配对方式完全可控（用交错配对，导出零重排）
3. 参数量精确可控

### RoPE 用交错配对

```python
def apply_rope(x, cos, sin):
    x1, x2 = x[..., 0::2], x[..., 1::2]     # 连续两维一组
    return torch.stack([x1 * cos - x2 * sin,
                        x1 * sin + x2 * cos], dim=-1).flatten(-2)
```

对应 llama.cpp 的 `LLAMA_ROPE_TYPE_NORM`，导出不需要 permute q/k 权重。

如果改成半切配对（HF Llama 的做法），必须重排权重。

### GGUF 导出的三个必须

1. `tokenizer.ggml.pre` 写 **`gpt-2`**（带连字符），不是 `gpt2`
2. 1D 张量（norm）保持 **f32**，否则计算图报类型不匹配
3. 2D 张量**不要手动反转**，gguf-py 序列化时自动反转

### 抗断电

`train.py` 的 checkpoint 机制：

```python
def save_atomic(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    with open(tmp, "rb") as f:
        os.fsync(f.fileno())        # 强制落盘
    os.replace(tmp, path)            # 原子替换
```

配合 `step_NNNN.pt` 轮转 + `last.pt` 硬链接，断电最多丢一个 save 周期。
