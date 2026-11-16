# 踩坑清单

> 每一条都是真实发生过、付出过代价的。持续追加。

---

## 一、致命错误

### 1.1 产物写 `/root`，2.5 小时训练全丢

**现象**：训练跑完准备导出，实例因 8 小时限额关机。下次开机 `/root/tiny-zh/` 整个目录不存在。

**原因**：容器里 `/root` 是 overlay 临时盘，实例销毁即清空。只有 `/mnt/workspace` 是 NAS 持久卷。

**正确做法**：所有产物写 `/mnt/workspace/项目名/`。开机先 `ls /mnt/workspace/` 确认。

**代价**：2.5 小时算力 + 重建数据 20 分钟 + 重训 45 分钟。

---

### 1.2 词表用碎片语料训练

**现象**：6400 词表训完，生成出来全是 `æĺ¥å¤©`。

**诊断**：统计「含两个以上汉字的 token」数量。

**原因**：ByteLevel BPE 把汉字当 UTF-8 字节。语料太少（63MB 维基碎片）→ BPE 合并不足 → 全是单字节碎片。

**修复**：用 ≥190MB 完整语义文本训词表。换成 minimind 40 万条后，压缩率从 1.225 提到 **1.55 字/token**。

**⚠️ 重要修正**：网上流传的检查代码是**错的**——它直接数字符串里的汉字，但词表存的是 GPT-2 字节映射形式（`山` 存成 `å±±`），按字面数永远是 0。

正确做法是先还原字节再解码：

```python
def gpt2_byte_decoder():
    cs = (list(range(ord("!"), ord("~")+1))
          + list(range(ord("¡"), ord("¬")+1))
          + list(range(ord("®"), ord("ÿ")+1)))
    table = {}
    n = 0
    for b in range(256):
        if b in cs:
            table[chr(b)] = b
        else:
            table[chr(256+n)] = b
            n += 1
    return table
```

**实测对比**（31.5M 的好词表）：

| 口径 | 结果 |
|---|---|
| 按字面数汉字 | **0**（错，会误杀好词表） |
| 还原字节后解码 | **3133**（正确） |

**工具**：`code/check_tokenizer.py` 已实现正确口径。

---

### 1.3 训练数据放 NAS

**现象**：吞吐只有 36k tok/s，预期 85k。

**原因**：`np.memmap` 在 NAS 上每次采样走网络，随机小 IO 延迟高。

**修复**：复制到 `/dev/shm`（内存盘）。

**效果**：36k → **85.5k tok/s**，快 2.4 倍。

---

### 1.4 给模型喂 system 角色

**现象**：15 条问答里 3 条崩成英文，中文占比 0.27。去掉 system 后变 0.78。

**原因**：minimind 的 SFT 数据几乎没有 system 消息，模型没见过。

**修复**：调用 API 只发 user。

---

### 1.5 用碎片数据凑 token 数（2026-11-15，最贵的一课）

**现象**：64M 用了 9.42 亿 token（14.5×），比 minimind-3 的约 2.5 亿（4×）多 3.8 倍，但只拿到 40.0%，对方 70.5%。

**原因**：为凑 20× 指标，把 LCCC（平均 **46 字/段**）提到 55% 占比，把最好的 `pretrain_t2t`（274 字完整段落）完全排除。

**后果**：9.42 亿 token 里 5.2 亿是碎片，模型只学到局部词搭配，学不会长程连贯。

**修复**：33.37M 改为 pretrain_t2t 54% + SkyPile 44% + CCI3-HQ 2%，LCCC 和 FineWeb 全部剔除。

**教训**：**指标不能凌驾于数据质量。** 20× 是 Chinchilla 的算力最优，不是效果最优。

---

## 二、数据格式坑（全部实测）

### 2.1 LCCC 每行是 list，不是 dict

```json
["火锅 我 在 重庆 成都 吃 了 七八 顿 火锅", "哈哈哈哈 ！ 那 我 的 嘴巴 可能 要 烂掉 ！", "不会 的 就是 好 油腻"]
```

写 `o.get("dialog")` 会崩（`AttributeError: 'list' object has no attribute 'get'`）。

正确：
```python
if isinstance(obj, list):
    dialog = obj
elif isinstance(obj, dict):
    dialog = obj.get("dialog") or obj.get("conversations") or []
```

### 2.2 LCCC 字间空格占 38.1%

原始格式逐字切分，`火锅 我 在 重庆`。不挤掉等于白扔 38% 的 token 预算。

```python
_CJK = r"\u4e00-\u9fff\u3000-\u303f\uff00-\uffef"
RE_CJK_SPACE = re.compile(rf"(?<=[{_CJK}])\s+(?=[{_CJK}])")
```

**实测**：20 万段样本，原样 14.8M 字符 → 去空格 9.2M 字符。

### 2.3 ShareGPT 字段是 `conversation[].human/assistant`

不是 `conversations[].from/value`。写错会读到 0 条。

实测：`top keys: ['conversation_id','category','conversation']`，`turn keys: ['assistant','human']`。

### 2.4 Infinity-Instruct 是英文数据

`7M_core` 的 `langdetect` 列标了语言。实测 5000 条样本，**中文占比 0.04**。

**必须先用 Range 请求抽样验证语言占比，再决定用不用。**

### 2.5 parquet 不能只下载一部分

Range 请求截断的 parquet 无法读取（footer 在文件末尾）：
```
Parquet magic bytes not found in footer
```

**要么完整下载，要么用 jsonl 格式的数据源做探测。**

---

## 三、GGUF 导出的十个坑

### 3.1 `tokenizer.ggml.pre` 必须写 `gpt-2`，不是 `gpt2`

写错报 `unknown pre-tokenizer type: 'gpt2'`，模型直接加载失败。

### 3.2 1D 张量（norm 权重）必须保持 f32

全转 f16 会出现 `binary_op: unsupported types: dst: f32, src0: f32, src1: f16`。

### 3.3 2D 张量不要手动反转

`gguf-py` 在序列化时自动反转（`gguf_writer.py:268`，`shape[n_dims-1-j]`）。PyTorch 的 `[out,in]` 落盘后正好是 llama.cpp 要的 `ne[0]=in, ne[1]=out`。

**验证方法**：读回 GGUF 逐张量比对，最大偏差应在 f16 误差内（实测 3.05e-05）。

### 3.4 词表大小必须与模型配置严格一致

补占位 token `<|reserved1816|>` 会被 llama.cpp 当特殊 token，一预测到就崩。

### 3.5 `llama-perplexity` 的 PPL 不能当判据

遇到 EOS 会重置上下文。我们每句以 EOS 结尾，导致 PPL 虚高几十倍（PyTorch 21.15 vs llama.cpp 2024.95，差 96 倍）。

**正确做法**：用 `llama-tokenize --ids` 逐 id 比对。

### 3.6 RoPE 用交错配对

`x[..., 0::2]` 和 `x[..., 1::2]` 配对，对应 llama.cpp 的 `LLAMA_ROPE_TYPE_NORM`，导出零重排。

用半切配对必须 permute q/k 权重。

### 3.7 SFT 的 idx 用 `np.fromfile` 而非 `np.load`

`build_dataset.py` 用 `np.asarray(idx, dtype=np.int32).tofile(path)` 写裸二进制。

```python
self.idx = np.fromfile(idx_path, dtype=np.int32).reshape(-1, 3)   # 正确
# self.idx = np.load(idx_path)                                     # 错误，当 pickle 解析
```

### 3.8 `llama-server` 必须加 `--jinja`

不加不读 GGUF 里的 chat_template，对话格式对不上，输出英文乱码。

### 3.9 `llama-cli` 已经没有 `-no-cnv` 了

新版用 `-st`（single-turn）代替。写了会报 `invalid argument: -no-cnv`。

### 3.10 Q4_K_M 对小模型会 fallback

64M 模型的 hidden 640、FFN 1792 不是 256 的整数倍，K-quant 无法完美适配。

实测：`WARNING: 79 of 120 tensor(s) required fallback quantization`，退化成 q4_0/q5_0。

**功能正常，精度略降。** 132MB → 41MB（压缩 69%）。

---

## 四、认知陷阱

### 4.1 误判「模型不会收尾」

**现象**：12 条写作测试，统计出「11 条句末标点完整度 = 0.00」，断定模型不会收尾。

**实际**：统计代码有 bug。

```python
# 错误
sents = [x for x in t.replace("\n", "。").split("。") if x.strip()]
punct = sum(1 for x in sents if x.strip()[-1] in "。！？") / len(sents)
# 永远返回 0，因为句号已经被 split 切走了
```

**正确**：判整段文本的最后一个字符。

```python
print(t.strip()[-1] in "。！？")
```

重算后发现：模型收尾完全正常。

**教训：出结论之前，先验证自己的测量工具。** 用已知答案的样本测一遍指标。

### 4.2 误判 GPU 空闲

**现象**：`nvidia-smi` 显示 0%，以为训练停了。

**实际**：采样恰好落在保存 778MB checkpoint 的间隙。

**正确做法**：连采 3-4 次，看趋势。

```bash
for i in 1 2 3 4; do
  nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader
  sleep 8
done
```

### 4.3 单次采样误判训练状态

同 4.2。2026-11-15 一天内发生两次。

---

## 五、环境坑

### 5.1 tailscale 只有 0.05 MB/s

云端没有 `/dev/net/tun`，tailscale 只能 userspace 模式，走 DERP 中继。

**正确方案**：tailscale 只做控制通道，大文件走 GitHub / 魔搭 / HF。

### 5.2 `--ssh` 不能省

userspace 模式下普通 22 端口不通，**只有 Tailscale SSH 能连进来**。关掉 `--ssh` 就完全连不上。

### 5.3 重启 daemon 丢登录态

`pkill tailscaled` 后必须 `up --authkey=...`，否则状态是 `offline`。

### 5.4 本机 tailscale 会自己停止

`tailscale status` 返回 `unexpected state: NoState` 或 `Tailscale is stopped`。

**修复**：`tailscale down && tailscale up`（不需要管理员权限）。

### 5.5 百度 AI Studio 的 BI-V150S 是国产卡

`Iluvatar CoreX`，不是 NVIDIA。`torch.cuda` 走兼容层，bf16/fused AdamW 都不可靠。

**选 V100 32GB。**

### 5.6 V100 没有 bf16

`torch.cuda.is_bf16_supported()` 返回 False，训练落到 fp16 + GradScaler。

**这条路没实测过，必须先跑烟雾测试。**

### 5.7 SSH 被墙

`ssh -T git@github.com` 报 `Connection closed by 198.18.0.29 port 22`。

修复 `~/.ssh/config`：
```
Host github.com
    HostName ssh.github.com
    Port 443
    User git
```

### 5.8 PowerShell 破坏引号

反复出现：
- `ssh host "python -c '...'"` 引号被吃掉
- `Set-Content` 带 BOM 或转 GBK
- `Get-Content | Set-Content` 中文乱码

**正确做法**：
1. 复杂命令写成脚本文件，`scp` 传上去执行
2. 用 `write` 工具写文件，不用 PowerShell 的 `Set-Content`
3. `$env:PYTHONIOENCODING="utf-8"`

### 5.9 `git push` 被密钥扫描拦截

硬编码 token 报 `GH013: Push cannot contain secrets`。

**修复**：token 写进 `.secrets.env`，加进 `.gitignore`，从环境变量读。

---

## 六、上传坑

### 6.1 魔搭 SDK 传大文件超时

传 1.27GB / 3.08GB 文件报 `TimeoutError('The write operation timed out')`。

**修复**：切成 200MB 分片，逐个上传（实测 18-19MB/s，零失败）。

### 6.2 魔搭 `create_dataset` 需要 `namespace` 参数

```python
a.create_dataset(
    dataset_name="tiny-zh-64m-data",
    namespace="SKL000000",     # ← 必须
    ...
)
```

漏了报 `missing 1 required positional argument: 'namespace'`。

### 6.3 HuggingFace 用户名可能带/不带连字符

`SKL-666666` 和 `SKL666666` 是两个不同的命名空间。用错报 `403 You don't have the rights`。

**先 `whoami` 确认**：
```python
from huggingface_hub import HfApi
print(HfApi().whoami()['name'])
```

### 6.4 HuggingFace 直连在国内不通

`hf upload` 报 `Network is unreachable`。

**修复**：`export HF_ENDPOINT=https://hf-mirror.com`

---

## 七、调试方法论

### 7.1 先用已知答案的样本测工具

4.1 的教训。

### 7.2 随机权重对比法

验证导出正确性：用随机初始化权重导出 GGUF，读回逐张量比对。

实测：111 个张量，最大绝对偏差 **3.05e-05**（f16 精度范围内）。

### 7.3 单 token 定位法

出问题时先喂 1 个 token 看 logits 是否合理，再逐步加长。

### 7.4 三次尝试法则

同一目标最多试 3 次，不成换路或报告阻塞。

**反例**：2026-11-15 的 SSH 连接试了 5 轮才找到根因（`--ssh` 被关掉）。

### 7.5 先测速再下载

下载前先用 Range 请求测 1MB 看速度，再决定并发数。

**实测速度差异**：

| 方式 | 速度 |
|---|---|
| 魔搭 SDK 内网 | **40-47 MB/s** |
| hf-mirror 单连接 | 150 KB/s |
| hf-mirror 32 路并发 | 2.2 MB/s |
| tailscale | 0.05 MB/s |