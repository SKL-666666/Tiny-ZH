#!/usr/bin/env bash
# 21:30 关机前的完整收尾（20:30 触发）
#
# 交付物清单
# ----------
# 1. 原始模型文件
#    - out/pretrain/last.pt    预训练 checkpoint（含优化器，可续训）
#    - out/sft/best.pt         SFT 后模型（择优 state_dict）
#    - out/tiny-zh-64m-f16.gguf 原始 GGUF（f16）
#    - out/tiny-zh-64m-q4km.gguf 量化 GGUF（Q4_K_M）
# 2. 简单评测
#    - logs/eval_report.md     困惑度 + 生成样例
# 3. 持久化
#    - 全部保留在 /mnt/workspace/tiny-zh-64m（NAS 持久盘）
#    - 关键文件上传 HF + 魔搭
set -u
WORK=/mnt/workspace/tiny-zh-64m
cd "$WORK" || exit 1
export PYTHONIOENCODING=utf-8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a logs/finalize.log; }

log "===== 收尾开始 ====="

# 载入凭据（HF / 魔搭 token）
if [ -f "$WORK/.secrets.env" ]; then
    # shellcheck disable=SC1091
    . "$WORK/.secrets.env"
    log "已载入 .secrets.env"
else
    log "[WARN] 没有 .secrets.env，上传会跳过"
fi

# ---- 1. 停预训练 ----
PID=$(pgrep -f 'train.py --mode pretrain' | head -1)
if [ -n "${PID:-}" ]; then
    log "预训练 PID $PID，发 SIGTERM（它会存完 checkpoint 再退）"
    kill -TERM "$PID" 2>/dev/null
    for i in $(seq 1 36); do
        sleep 5
        pgrep -f 'train.py --mode pretrain' >/dev/null || break
    done
    pgrep -f 'train.py --mode pretrain' >/dev/null && {
        log "[WARN] 未退出，kill -9"
        pkill -9 -f 'train.py --mode pretrain'
    }
fi
log "预训练已停"
ls -lt out/pretrain/*.pt 2>/dev/null | head -3 | tee -a logs/finalize.log

# ---- 2. 抽出纯 state_dict 给 SFT ----
log "===== 准备 SFT 初始化 ====="
python3 - <<'PY' 2>&1 | tee -a logs/finalize.log
import os
import torch

src = None
for c in ("out/pretrain/last.pt", "out/pretrain/best.pt", "out/pretrain/final.pt"):
    if os.path.exists(c):
        src = c
        break
if not src:
    print("[FATAL] 没有预训练 checkpoint")
    raise SystemExit(1)

ck = torch.load(src, map_location="cpu", weights_only=False)
sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
torch.save(sd, "out/pretrain/for_sft.pt")
step = ck.get("step", "?") if isinstance(ck, dict) else "?"
best = ck.get("best_val", "?") if isinstance(ck, dict) else "?"
print(f"  从 {src} 抽 state_dict")
print(f"  step={step} best_val={best}")
print(f"  保存 out/pretrain/for_sft.pt（{len(sd)} 项）")
PY

# ---- 3. SFT ----
if [ ! -f out/sft/best.pt ]; then
    log "===== SFT ====="
    DATADIR=bin
    [ -f /dev/shm/tz64/bin/sft_train.bin ] && DATADIR=/dev/shm/tz64/bin
    log "数据 $DATADIR"
    python3 -u train.py --mode sft \
        --data "$DATADIR" --out out/sft --init out/pretrain/for_sft.pt \
        --seq-len 512 --micro-batch 32 --grad-accum 2 \
        --lr 1e-4 --max-steps 1200 --time-budget 2100 \
        --probe-steps 0 --save-every 150 --log-every 50 \
        2>&1 | tee -a logs/sft.log
fi
log "SFT 完成"

# ---- 4. 导出原始 GGUF ----
if [ ! -f out/tiny-zh-64m-f16.gguf ]; then
    log "===== 导出 f16 GGUF ====="
    STATE=out/sft/best.pt
    [ -f "$STATE" ] || STATE=out/sft/final.pt
    [ -f "$STATE" ] || STATE=out/pretrain/for_sft.pt
    python3 export_gguf.py --state "$STATE" --tokenizer tokenizer \
        --out out/tiny-zh-64m-f16.gguf --outtype f16 --name tiny-zh-64m \
        2>&1 | tail -8 | tee -a logs/export.log
fi
ls -lh out/tiny-zh-64m-f16.gguf 2>/dev/null | tee -a logs/finalize.log

# ---- 5. 量化 Q4_K_M ----
if [ ! -f out/tiny-zh-64m-q4km.gguf ] && [ -f out/tiny-zh-64m-f16.gguf ]; then
    log "===== 量化 Q4_K_M ====="
    if [ -x /mnt/workspace/tools/llama-quantize ]; then
        /mnt/workspace/tools/llama-quantize out/tiny-zh-64m-f16.gguf \
            out/tiny-zh-64m-q4km.gguf Q4_K_M 2>&1 | tail -5 | tee -a logs/quant.log
    else
        log "用 llama-cpp-python 的 API 量化"
        python3 - <<'PY' 2>&1 | tee -a logs/quant.log
import llama_cpp
import llama_cpp.llama_cpp as lc
import os, sys

src = b"out/tiny-zh-64m-f16.gguf"
dst = b"out/tiny-zh-64m-q4km.gguf"

# llama_model_quantize 的参数签名（llama_cpp 0.3.x）:
#   (fname_inp, fname_out, params) -> int
try:
    params = lc.llama_model_quantize_default_params()
    params.ftype = 12          # LLAMA_FTYPE_MOSTLY_Q4_K_M = 12
    params.nthread = 8
    params.allow_requantize = True
    print(f"  params.ftype={params.ftype}")
    rc = lc.llama_model_quantize(src, dst, params, None, None)
    print(f"  rc={rc}")
    if os.path.exists("out/tiny-zh-64m-q4km.gguf"):
        sz = os.path.getsize("out/tiny-zh-64m-q4km.gguf")
        print(f"  输出 {sz/1e6:.1f} MB")
    else:
        print("  [FAIL] 没有输出文件")
        sys.exit(1)
except Exception as e:
    import traceback
    traceback.print_exc()
    sys.exit(1)
PY
    fi
fi
ls -lh out/*.gguf 2>/dev/null | tee -a logs/finalize.log

# ---- 6. 评测 ----
log "===== 实验记录采集（论文用）====="
python3 collect_experiment.py 2>&1 | tail -4 | tee -a logs/finalize.log

log "===== 简单评测 ====="
python3 - > logs/eval_report.md 2>&1 <<'PY'
import json
import os

print("# Tiny-ZH-64M 评测报告\n")
print(f"生成时间：{__import__('datetime').datetime.now():%Y-%m-%d %H:%M}\n")

# ---- 训练指标 ----
print("## 训练指标\n")
print("| 阶段 | best val_loss | PPL | 步数 |")
print("|---|---|---|---|")
for tag, p in (("预训练", "out/pretrain/train_log.jsonl"),
               ("SFT", "out/sft/train_log.jsonl")):
    if not os.path.exists(p):
        continue
    rows = [json.loads(l) for l in open(p) if '"val_loss"' in l]
    if not rows:
        continue
    best = min(r["val_loss"] for r in rows)
    last = rows[-1]
    ppl = 2.718281828 ** best
    print(f"| {tag} | {best:.4f} | {ppl:.1f} | {last['step']} |")

# ---- 文件 ----
print("\n## 交付文件\n")
print("| 文件 | 大小 | 说明 |")
print("|---|---|---|")
for f, desc in (("out/tiny-zh-64m-f16.gguf", "原始 GGUF"),
                ("out/tiny-zh-64m-q4km.gguf", "量化 GGUF"),
                ("out/sft/best.pt", "SFT 模型"),
                ("out/pretrain/last.pt", "预训练 checkpoint（可续训）")):
    if os.path.exists(f):
        sz = os.path.getsize(f) / 1e6
        print(f"| `{f}` | {sz:.1f} MB | {desc} |")

# ---- 数据规模 ----
print("\n## 训练规模\n")
p = 64_816_000
for f, label in (("bin/pretrain_train.bin", "预训练 token")):
    if os.path.exists(f):
        n = os.path.getsize(f) // 2
        print(f"- {label}：{n:,}（{n/1e8:.2f} 亿，{n/p:.1f}× 参数）")
PY
cat logs/eval_report.md | tee -a logs/finalize.log

# ---- 7. 生成样例 ----
log "===== 生成样例 ====="
if [ -f out/tiny-zh-64m-f16.gguf ]; then
    python3 - > logs/gen_samples.md 2>&1 <<'PY'
import llama_cpp

PROMPTS = [
    "什么是人工智能？",
    "今天天气怎么样？",
    "推荐一本好书。",
    "如何学习编程？",
    "介绍一下北京。",
]

try:
    llm = llama_cpp.Llama(
        model_path="out/tiny-zh-64m-f16.gguf",
        n_ctx=512, n_threads=8, verbose=False,
    )
    print("# 生成样例\n")
    for q in PROMPTS:
        print(f"**问：{q}**\n")
        out = llm(f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n",
                  max_tokens=120, temperature=0.7, top_p=0.9,
                  repeat_penalty=1.2, stop=["<|im_end|>"])
        ans = out["choices"][0]["text"].strip()
        zh = sum(1 for c in ans if "\u4e00" <= c <= "\u9fff")
        ratio = zh / max(len(ans), 1)
        print(ans)
        print(f"\n*中文占比 {ratio:.2f}，{len(ans)} 字*\n")
except Exception as e:
    print(f"生成失败: {type(e).__name__}: {e}")
PY
    cat logs/gen_samples.md | tee -a logs/finalize.log
fi

# ---- 8. 双平台上传 ----
log "===== 上传交付物（HF + 魔搭）====="

export HF_ENDPOINT=https://hf-mirror.com
export HF_TOKEN="${HF_TOKEN:-}"
export MODELSCOPE_API_TOKEN="${MODELSCOPE_API_TOKEN:-}"
export MS_OWNER="${MS_OWNER:-SKL000000}"

# ---- 8a. HuggingFace ----
if [ -n "$HF_TOKEN" ] && [ -f out/tiny-zh-64m-q4km.gguf ]; then
    log "HF: 上传模型"
    hf upload "SKL666666/tiny-zh-64m" out \
        --repo-type model --private --include "*.gguf" "*.pt" 2>&1 | tail -6 \
        | tee -a logs/upload_model.log
    log "HF: 上传论文材料"
    hf upload "SKL666666/tiny-zh-64m-logs" logs \
        --repo-type dataset --private 2>&1 | tail -4 \
        | tee -a logs/upload_logs.log
else
    log "[skip] HF：缺 token 或没有产物"
fi

# ---- 8b. 魔搭（模型分片，规避大文件超时）----
if [ -n "$MODELSCOPE_API_TOKEN" ]; then
    log "魔搭: 创建模型仓库并上传"
    python3 - <<'PY' 2>&1 | tee -a logs/upload_ms_model.log
import os, time
from pathlib import Path
from modelscope.hub.api import HubApi

token = os.environ["MODELSCOPE_API_TOKEN"]
owner = os.environ.get("MS_OWNER", "SKL000000")
a = HubApi()
a.login(token)
repo = f"{owner}/tiny-zh-64m"

# 创建 model 仓库
try:
    a.create_model(model_name="tiny-zh-64m", namespace=owner,
                   chinese_name="Tiny-ZH-64M 中文小模型",
                   license="Apache-2.0", visibility=1)
    print(f"  创建 {repo}")
except Exception as e:
    msg = str(e)[:120]
    print(f"  创建返回: {msg}")

# 收集待传文件：GGUF + 关键 pt + 论文材料
targets = []
for pat, label in (
    ("out/tiny-zh-64m-f16.gguf", "原始 GGUF"),
    ("out/tiny-zh-64m-q4km.gguf", "量化 GGUF"),
    ("out/sft/best.pt", "SFT 模型"),
    ("logs/experiment_record.md", "实验记录"),
    ("logs/experiment_record.json", "实验记录JSON"),
    ("logs/eval_report.md", "评测报告"),
    ("logs/gen_samples.md", "生成样例"),
    ("tokenizer/tokenizer.json", "词表"),
):
    if os.path.exists(pat):
        targets.append(pat)

print(f"  {len(targets)} 个文件")

CHUNK = 200 * 1024 * 1024
for t in targets:
    sz = os.path.getsize(t)
    name = os.path.basename(t)
    if sz <= CHUNK:
        for attempt in (1, 2, 3):
            try:
                t0 = time.time()
                a.upload_file(path_or_fileobj=t, path_in_repo=name,
                              repo_id=repo, repo_type="model",
                              commit_message=f"add {name}")
                dt = time.time() - t0
                print(f"  OK {name} {sz/1e6:.0f}MB {dt:.0f}s")
                break
            except Exception as e:
                if attempt < 3:
                    time.sleep(5)
                else:
                    print(f"  FAIL {name}: {str(e)[:100]}")
    else:
        # 大文件切块
        print(f"  {name} {sz/1e6:.0f}MB，切块上传")
        import subprocess, tempfile, shutil
        tmp = tempfile.mkdtemp()
        subprocess.run(["split", "-b", "200M", "-d", "-a", "3",
                        t, os.path.join(tmp, name + ".part")], check=True)
        parts = sorted(Path(tmp).iterdir())
        for i, pf in enumerate(parts, 1):
            for attempt in (1, 2, 3):
                try:
                    a.upload_file(path_or_fileobj=str(pf),
                                  path_in_repo=pf.name,
                                  repo_id=repo, repo_type="model",
                                  commit_message=f"part {pf.name}")
                    print(f"    [{i}/{len(parts)}] {pf.name} OK")
                    break
                except Exception as e:
                    if attempt < 3:
                        time.sleep(5)
                    else:
                        print(f"    [{i}/{len(parts)}] {pf.name} FAIL {str(e)[:80]}")
        shutil.rmtree(tmp, ignore_errors=True)

print(f"\n  查看: https://www.modelscope.cn/models/{repo}")
PY
else
    log "[skip] 魔搭：缺 token"
fi

# ---- 9. 汇总 ----
log "===== 收尾完成 ====="
echo "产物清单：" | tee -a logs/finalize.log
du -sh out/* 2>/dev/null | tee -a logs/finalize.log
echo | tee -a logs/finalize.log
echo "论文材料：" | tee -a logs/finalize.log
echo "  logs/experiment_record.md     实验记录（人可读）" | tee -a logs/finalize.log
echo "  logs/experiment_record.json   实验记录（机器可读，含完整曲线）" | tee -a logs/finalize.log
echo "  logs/eval_report.md           评测报告" | tee -a logs/finalize.log
echo "  logs/gen_samples.md           生成样例" | tee -a logs/finalize.log
echo "  logs/pretrain.log             预训练原始日志" | tee -a logs/finalize.log
echo "  logs/sft.log                  SFT 原始日志" | tee -a logs/finalize.log
echo "生成样例：logs/gen_samples.md" | tee -a logs/finalize.log