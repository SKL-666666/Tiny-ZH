# 环境与基础设施

> 记录实际跑通过的环境、命令、坑。

---

## 一、硬件

| 环境 | 配置 | 用途 |
|---|---|---|
| **本机** | Windows，Python 3.13.3，torch CPU 版，8 核 | 推理、数据探测、导出验证 |
| **魔搭 A10** | A10 24GB，torch 2.13.0+cu130，8-16 核，`/dev/shm` 16-30GB | 主训练环境 |
| **百度 AI Studio V100** | V100 32GB，4 核，**无 bf16** | 待验证 |

### 魔搭实测数据

| 项 | 值 |
|---|---|
| 下载速度 | **40-47 MB/s**（内网） |
| 单次时长 | 8 小时（实测提前 6 分钟关机） |
| 持久卷 | `/mnt/workspace`（1PB NAS） |
| 临时盘 | `/root`、`/tmp`（实例销毁即清空） |

### A10 实测训练速度

| 模型 | 配置 | 吞吐 |
|---|---|---|
| 31.5M | seq 256, bs 32×4 | 85.5k tok/s |
| 64M | seq 1024, bs 32×1 | **48,364 tok/s** |
| 64M SFT | seq 512, bs 32×2 | 49,500 tok/s |

---

## 二、SSH 连接（每次开机必做）

### 2.1 云端装 tailscale

```bash
cd /tmp
rm -rf ts_install pt_*.bin
VER=$(curl -fsSL https://pkgs.tailscale.com/stable/ | grep -oP 'tailscale_\K[0-9]+\.[0-9]+\.[0-9]+(?=_amd64\.tgz)' | head -1)
mkdir -p ts_install && cd ts_install
URL="https://pkgs.tailscale.com/stable/tailscale_${VER}_amd64.tgz"
SIZE=$(curl -fsSLI "$URL" | grep -i '^content-length:' | tail -1 | tr -d '\r' | awk '{print $2}')
CHUNK=$(( (SIZE+31)/32 ))
for i in $(seq 0 31); do
  a=$((i*CHUNK)); b=$((a+CHUNK-1)); [ $b -ge $SIZE ] && b=$((SIZE-1))
  curl -fsSL --connect-timeout 8 --max-time 300 --retry 5 --range "$a-$b" -o pt_$i.bin "$URL" >/dev/null 2>&1 &
done
wait
cat $(for i in $(seq 0 31); do echo pt_$i.bin; done) > ts.tgz
[ "$(stat -c %s ts.tgz)" = "$SIZE" ] && tar tzf ts.tgz >/dev/null 2>&1 && echo "下载完整" || { echo "下载失败"; exit 1; }
tar xzf ts.tgz && cd tailscale_${VER}_amd64
pkill -f tailscaled; sleep 3
nohup ./tailscaled --tun=userspace-networking --socks5-server=localhost:1055 > /tmp/tsd.log 2>&1 &
sleep 8
pgrep -a tailscaled || { echo "daemon没起"; tail -5 /tmp/tsd.log; exit 1; }
timeout 180 ./tailscale up --ssh --hostname=NAME --authkey=tskey-auth-xxx
sleep 3
./tailscale ip -4
```

**三个不能省的：**
1. **`--ssh`** — userspace 模式下只有 Tailscale SSH 能连进来
2. **`--authkey`** — 重启 daemon 后必须重新登录
3. **32 路分段** — 单连接只有 89KB/s

### 2.2 本机连

```powershell
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=NUL root@100.x.x.x
```

### 2.3 如果连不上

**本机 PowerShell 执行**（不需要管理员）：

```powershell
& "C:\Program Files\Tailscale\tailscale.exe" down
& "C:\Program Files\Tailscale\tailscale.exe" up
```

这会强制本机重新同步节点列表。

**检查节点状态**：
```powershell
& "C:\Program Files\Tailscale\tailscale.exe" status
```

看到 `offline` 说明云端掉线，需要云端重新 `up`。

---

## 三、代码上传（codeload，不走 git clone）

**git clone 走 22 端口会被墙**，用 HTTPS tar.gz：

```bash
cd /mnt/workspace && rm -rf tzsrc && curl -fsSL -o tz.tar.gz \
  https://codeload.github.com/SKL-666666/Tiny-ZH-32M/tar.gz/refs/heads/master \
  && tar xzf tz.tar.gz && mv Tiny-ZH-32M-master tzsrc
```

**推送到 GitHub 时注意**：token 不能硬编码，会被密钥扫描拦截（GH013）。

用 `.secrets.env` + `.gitignore`。

---

## 四、数据下载

### 4.1 测速优先

```python
# smart_dl.py 会先测 1MB 速度，再选并行策略
python3 smart_dl.py "URL" -o out.bin --ncpu 32
```

### 4.2 实测速度对比

| 方式 | 速度 | 适用 |
|---|---|---|
| **魔搭 SDK / 直链** | **40-47 MB/s** | 所有魔搭上的数据 |
| hf-mirror 单连接 | 150 KB/s | 不用 |
| hf-mirror 32 路并发 | 2.2 MB/s | HF 独有的大文件 |
| 魔搭 Web API 直链 | 14 MB/s | 单文件 |
| tailscale | 0.05 MB/s | **永不用于传文件** |

### 4.3 魔搭直链格式

```
https://www.modelscope.cn/api/v1/datasets/{owner}/{repo}/repo?Revision=master&FilePath={path}
```

路径要 URL 编码（`/` → `%2F`）。

### 4.4 Range 探测（不下载全文）

```bash
curl -fsSL --range 0-29999999 -o sample.jsonl "URL"
```

**parquet 不能用这种方式**（footer 在末尾），必须完整下载或改用 jsonl 源探测。

---

## 五、训练环境准备

### 5.1 依赖

```bash
pip install -q gguf llama-cpp-python
```

- `gguf`：导出 GGUF 用
- `llama-cpp-python`：含 `llama_model_quantize`，量化不需要单独装 llama-quantize

### 5.2 量化

```python
import llama_cpp.llama_cpp as lc
params = lc.llama_model_quantize_default_params()
params.ftype = 12          # Q4_K_M
params.nthread = 16
lc.llama_model_quantize(b"in-f16.gguf", b"out-q4km.gguf", params, None, None)
```

**为什么不用 llama-quantize 二进制**：GitHub 预编译包被墙，源码编译要几分钟且易缺依赖。`llama-cpp-python` 的 wheel 在 PyPI 上，pip 直接装。

### 5.3 内存盘

```bash
mkdir -p /dev/shm/tz/bin
cp bin/* /dev/shm/tz/bin/
# train.py --data /dev/shm/tz/bin
```

**效果**：36k → 85.5k tok/s（快 2.4 倍）。

---

## 六、后台任务启动

**PowerShell 传 SSH 时 `&` 容易失效**，用子 shell：

```bash
cd /path && (nohup setsid bash script.sh > logs/run.log 2>&1 < /dev/null &) && sleep 5 && tail -3 logs/run.log
```

**验证是否真的在跑**：

```bash
pgrep -c -f 'train.py'    # 应该 > 0
tail -2 logs/xxx.log
```

---

## 七、抗断电设计（四层）

| 层 | 做法 | 效果 |
|---|---|---|
| 1 | 产物全写 `/mnt/workspace` | 实例销毁不丢 |
| 2 | checkpoint 原子写盘 `.tmp` → `fsync` → `rename` | 断电不留半个文件 |
| 3 | 每次存 `step_NNNN.pt`（留 3 个），`last.pt` 硬链接 | `last.pt` 坏了能退到备份 |
| 4 | checkpoint 存 `total_steps`，续训沿用 | LR 曲线不错位 |

**最坏损失**：一个 save 周期（`--save-every 250` 约 2.7 分钟）。

---

## 八、定时收尾

云端实例有硬性关机时间。用 watchdog 到点触发：

```bash
# 20:30 触发收尾
nohup setsid bash cloud/deadline.sh 20:30 > logs/deadline.log 2>&1 &
```

收尾流程：停训练 → 抽 state_dict → SFT → 导出 f16 → 量化 Q4_K_M → 采集实验记录 → 评测 → 双平台上传。

**优先级**：先保证产物上传，再跑评测。评测失败不影响交付。

---

## 九、常见故障排查

| 现象 | 原因 | 修复 |
|---|---|---|
| `ssh: connect to host ... timed out` | 云端 offline | 云端 `tailscale up --authkey` |
| `unexpected state: NoState` | 本机 tailscale 停止 | 本机 `down` 然后 `up` |
| `banner exchange: timed out` | 云端关了 `--ssh` | 云端 `tailscale set --ssh` |
| `nvidia-smi` 显示 0% | 采样在存盘间隙 | 连采 3 次确认 |
| `CUDA out of memory` | batch 太大 | 降 micro_batch |
| `nan` in loss | fp16 下溢（V100） | 降 lr 或改 fp32 |
| 词表压缩率 < 1.45 | 语料不足 | 加语料到 ≥190MB |
| GGUF 加载失败 | `tokenizer.ggml.pre` 写错 | 改成 `gpt-2` |
| 生成乱码 | RoPE 配对方式不匹配 | 检查用交错还是半切 |

---

## 十、凭据管理

```
/mnt/workspace/tiny-zh-64m/.secrets.env    # 权限 600
export HF_TOKEN=hf_xxx
export HF_ENDPOINT=https://hf-mirror.com
export MODELSCOPE_API_TOKEN=ms-xxx
export MS_OWNER=SKL000000
```

加进 `.gitignore`，**不提交**。

---

## 十一、数据与模型位置

| 内容 | 位置 |
|---|---|
| 训练数据 29.27 亿 token | `SKL666666/tiny-zh-64m-data` (HF) |
| 模型 + GGUF | `SKL666666/tiny-zh-64m` (HF) |
| 训练日志 + 测试 | `SKL666666/tiny-zh-64m-logs` (HF) |
| 魔搭镜像 | `SKL000000/tiny-zh-64m` (ModelScope) |

**上传策略**：先 GGUF（最重要），再权重，最后日志。魔搭大文件切 200MB 分片。