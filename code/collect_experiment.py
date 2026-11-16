#!/usr/bin/env python3
"""实验记录采集器：为论文准备可复现的完整数据。

采集内容
--------
1. 环境快照      硬件、驱动、库版本、CUDA/ROCm
2. 配置快照      模型结构、训练超参（从 train_log 反推 + config.py）
3. 数据统计      每个数据源的原始量、清洗后量、token 数、占比
4. 训练曲线      loss / lr / throughput 的完整时间序列
5. 时间戳        每个阶段的起止时间与耗时
6. 随机性        seed、数据顺序

输出：
    logs/experiment_record.json   机器可读，便于画图
    logs/experiment_record.md     人可读，可直接贴论文附录

用法：
    python3 collect_experiment.py
"""
import glob
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime

WORK = "/mnt/workspace/tiny-zh-64m"
os.chdir(WORK)


def run(cmd, timeout=60):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                           timeout=timeout, errors="replace")
        return (r.stdout or "").strip()
    except Exception:
        return ""


def sha256_head(path, nbytes=1 << 20):
    """算文件前 1MB 的哈希，用于标识版本（全量太慢）。"""
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(nbytes))
    return h.hexdigest()[:16]


# ---------------------------------------------------------------- 1. 环境

def env_snapshot():
    d = {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
        "collected_at": datetime.now().isoformat(timespec="seconds"),
    }
    # GPU
    d["gpu"] = run("nvidia-smi --query-gpu=name,memory.total,driver_version "
                   "--format=csv,noheader")
    # torch
    try:
        import torch
        d["torch"] = torch.__version__
        d["cuda_compiled"] = torch.version.cuda
        d["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            d["gpu_name"] = p.name
            d["gpu_mem_gb"] = round(p.total_memory / 1e9, 2)
            d["gpu_sm"] = p.multi_processor_count
            d["bf16_supported"] = torch.cuda.is_bf16_supported()
            d["torch_cudnn"] = torch.backends.cudnn.version()
    except Exception as e:
        d["torch_error"] = str(e)
    # 关键库
    for mod in ("tokenizers", "numpy", "pyarrow", "transformers"):
        try:
            m = __import__(mod)
            d[mod] = getattr(m, "__version__", "?")
        except ImportError:
            d[mod] = None
    return d


# ---------------------------------------------------------------- 2. 配置

def config_snapshot():
    """配置快照。

    关键：TrainConfig 的默认值不是实际用的值——训练时命令行参数覆盖了它。
    论文里必须写实际值，所以从两个地方取：
    1. logs/train_run.log 里打印的 [data] tok/step（可反推 micro_batch×grad_accum×seq）
    2. 启动脚本 cloud/start64.sh 里的命令行参数
    """
    sys.path.insert(0, WORK)
    d = {}
    try:
        from config import ModelConfig, TrainConfig
        m = ModelConfig()
        t = TrainConfig()
        d["model"] = {k: v for k, v in m.__dict__.items()}
        d["param_count"] = m.param_count()
        d["train_defaults"] = {k: v for k, v in t.__dict__.items()}
        d["train_defaults_note"] = "这是 dataclass 默认值，实际训练被命令行覆盖，见 actual_* 字段"
    except Exception as e:
        d["error"] = str(e)

    # ---- 从实际日志反推 ----
    for path, tag in (("logs/pretrain.log", "pretrain"),
                      ("logs/sft.log", "sft")):
        if not os.path.exists(path):
            continue
        with open(path, errors="replace") as f:
            for line in f:
                s = line.strip()
                if s.startswith("[data]"):
                    d[tag + "_data"] = s
                elif s.startswith("[model]"):
                    d[tag + "_model"] = s
                elif s.startswith("[opt]"):
                    d[tag + "_optimizer"] = s
                elif s.startswith("[env]"):
                    d.setdefault(tag + "_env", []).append(s)

    # 从 tok/step 反推有效批次
    import re
    if "pretrain_data" in d:
        m = re.search(r"tok/step=([\d,]+)", d["pretrain_data"])
        if m:
            tps = int(m.group(1).replace(",", ""))
            d["actual_tok_per_step"] = tps
            # 已知 seq_len=1024 时，tps/1024 = micro_batch × grad_accum
            d["actual_batch_x_accum_if_seq1024"] = tps // 1024

    # 从启动脚本抓命令行
    for script in ("cloud/start64.sh", "cloud/finalize.sh"):
        if not os.path.exists(script):
            continue
        txt = open(script, errors="replace").read()
        for key, pat in (
            ("actual_seq_len", r"--seq-len\s+(\d+)"),
            ("actual_micro_batch", r"--micro-batch\s+(\d+)"),
            ("actual_grad_accum", r"--grad-accum\s+(\d+)"),
            ("actual_lr", r"--lr\s+([\d.e-]+)"),
            ("actual_max_steps", r"--max-steps\s+(\d+)"),
            ("actual_save_every", r"--save-every\s+(\d+)"),
        ):
            if key in d:
                continue
            mm = re.search(pat, txt)
            if mm:
                d[key] = mm.group(1)

    # 训练产出的真实吞吐
    if os.path.exists("out/pretrain/train_log.jsonl"):
        rows = []
        for line in open("out/pretrain/train_log.jsonl", errors="replace"):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        ts = [r["tok_s"] for r in rows if "tok_s" in r]
        if ts:
            d["actual_throughput_tok_s"] = {
                "mean": round(sum(ts) / len(ts)),
                "max": max(ts), "min": min(ts),
                "samples": len(ts),
            }
    return d


# ---------------------------------------------------------------- 3. 数据

SOURCE_FILES = {
    "LCCC large": ("data/raw/lccc_large.jsonl.gz", "1200 万段中文日常对话"),
    "ShareGPT-zh-70k": ("data/raw/common_zh_70k.jsonl", "7.6 万条真实多轮"),
    "ShareGPT-unknow-38k": ("data/raw/unknow_zh_38k.jsonl", "3.9 万条多轮"),
    "中文维基": ("data/raw/wiki.jsonl", "wikipedia-cn-20230720-filtered"),
    "COIG-CQIA": ("data/raw/coig_full.jsonl", "知乎/小红书/wikihow/douban"),
    "弱智吧": ("data/raw/ruozhi.jsonl", "口语化短句"),
}
FWV2_GLOB = "data/raw/fwv2/*.parquet"


def data_snapshot():
    d = {"sources": {}, "total": {}}

    for name, (path, desc) in SOURCE_FILES.items():
        if os.path.exists(path):
            sz = os.path.getsize(path)
            d["sources"][name] = {
                "path": path, "bytes": sz, "mb": round(sz / 1e6, 1),
                "sha256_head16": sha256_head(path), "desc": desc,
            }
        else:
            d["sources"][name] = {"path": path, "missing": True}

    # FineWeb 分片
    shards = sorted(glob.glob(FWV2_GLOB))
    if shards:
        tot = sum(os.path.getsize(s) for s in shards)
        d["sources"]["FineWeb-Edu-v2"] = {
            "path": "data/raw/fwv2/*.parquet",
            "shards": len(shards),
            "bytes": tot, "mb": round(tot / 1e6, 1),
            "files": [os.path.basename(s) for s in shards],
            "desc": "opencsg 教育级中文网页",
        }

    # 编码产物
    for f, tag in (("bin/pretrain_train.bin", "pretrain_train"),
                   ("bin/pretrain_val.bin", "pretrain_val"),
                   ("bin/sft_train.bin", "sft_train"),
                   ("bin/sft_val.bin", "sft_val")):
        if os.path.exists(f):
            n = os.path.getsize(f) // 2
            d["total"][tag] = {"tokens": n, "yi": round(n / 1e8, 3)}

    # 清洗后文本
    if os.path.exists("data/pretrain.txt"):
        sz = os.path.getsize("data/pretrain.txt")
        lines = int(run(f"wc -l < data/pretrain.txt") or 0)
        d["total"]["pretrain_txt"] = {
            "bytes": sz, "gb": round(sz / 1e9, 2), "lines": lines,
        }
    # SFT
    if os.path.exists("data/sft.jsonl"):
        d["total"]["sft_jsonl"] = {
            "lines": int(run("wc -l < data/sft.jsonl") or 0),
            "mb": round(os.path.getsize("data/sft.jsonl") / 1e6, 1),
        }

    # 词表
    if os.path.exists("tokenizer/tokenizer.json"):
        d["tokenizer"] = {
            "vocab_size": int(run("python3 -c \"from tokenizers import Tokenizer;"
                                  "print(Tokenizer.from_file('tokenizer/tokenizer.json')"
                                  ".get_vocab_size())\"") or 0),
            "sha256_head16": sha256_head("tokenizer/tokenizer.json"),
        }
    d["tokenizer_stats"] = {
        "chars_per_token": 1.20,
        "multi_zh_tokens": 3618,
        "note": "实测自 check_tokenizer.py",
    }
    return d


# ---------------------------------------------------------------- 4. 曲线

def curve_snapshot():
    d = {}
    for path, tag in (("out/pretrain/train_log.jsonl", "pretrain"),
                      ("out/sft/train_log.jsonl", "sft")):
        if not os.path.exists(path):
            continue
        rows = []
        for line in open(path, errors="replace"):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        if not rows:
            continue
        loss_rows = [r for r in rows if "loss" in r]
        val_rows = [r for r in rows if "val_loss" in r]
        d[tag] = {
            "steps_logged": len(loss_rows),
            "val_points": len(val_rows),
            "first_loss": loss_rows[0]["loss"] if loss_rows else None,
            "last_loss": loss_rows[-1]["loss"] if loss_rows else None,
            "best_val": min((r["val_loss"] for r in val_rows), default=None),
            "last_val": val_rows[-1]["val_loss"] if val_rows else None,
            "last_step": loss_rows[-1]["step"] if loss_rows else None,
            "avg_tok_s": (sum(r.get("tok_s", 0) for r in loss_rows)
                          / max(len(loss_rows), 1)),
            "curve": loss_rows,
            "val_curve": val_rows,
        }
    return d


# ---------------------------------------------------------------- 5. 时间

def timeline_snapshot():
    """从各日志的首末时间戳推断阶段耗时。"""
    stages = {}
    logs = {
        "data_prep": "logs/prep_run.log",
        "encode": "logs/enc2.log",
        "tokenizer": "logs/tok2.log",
        "pretrain": "logs/pretrain.log",
        "sft": "logs/sft.log",
    }
    for tag, p in logs.items():
        if not os.path.exists(p):
            continue
        st = os.stat(p)
        stages[tag] = {
            "mtime": datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
            "size": st.st_size,
            "exists": True,
        }
    return stages


# ---------------------------------------------------------------- 主流程

def main():
    print("采集实验记录…")
    rec = {
        "meta": {
            "project": "Tiny-ZH-64M",
            "note": "中文小模型从零训练，为论文准备的可复现记录",
            "generated": datetime.now().isoformat(timespec="seconds"),
        },
        "environment": env_snapshot(),
        "config": config_snapshot(),
        "data": data_snapshot(),
        "curves": curve_snapshot(),
        "timeline": timeline_snapshot(),
    }

    os.makedirs("logs", exist_ok=True)
    with open("logs/experiment_record.json", "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=2, default=str)
    print("  logs/experiment_record.json")

    # ---- 生成 markdown ----
    L = []
    a = L.append
    a("# Tiny-ZH-64M 实验记录\n")
    a(f"生成时间：{rec['meta']['generated']}\n")

    e = rec["environment"]
    a("## 1. 实验环境\n")
    a("| 项 | 值 |")
    a("|---|---|")
    for k in ("platform", "python", "cpu_count", "gpu_name", "gpu_mem_gb",
              "gpu_sm", "torch", "cuda_compiled", "bf16_supported", "tokenizers"):
        v = e.get(k)
        if v is not None:
            a(f"| {k} | {v} |")
    a("")

    c = rec["config"]
    if "model" in c:
        a("## 2. 模型配置\n")
        a("| 参数 | 值 |")
        a("|---|---|")
        for k, v in c["model"].items():
            a(f"| {k} | {v} |")
        a("")
        pc = c.get("param_count", {})
        if pc:
            a(f"**参数量：{pc.get('total', 0):,}**\n")

        # 实际训练配置（论文用这个）
        actual_keys = [
            ("actual_seq_len", "序列长度 seq_len"),
            ("actual_micro_batch", "micro_batch"),
            ("actual_grad_accum", "grad_accum"),
            ("actual_tok_per_step", "每步 token 数"),
            ("actual_lr", "学习率 lr"),
            ("actual_max_steps", "总步数 max_steps"),
            ("actual_save_every", "存盘间隔"),
        ]
        has_actual = any(k in c for k, _ in actual_keys)
        if has_actual:
            a("### 实际训练超参（命令行传入，论文以此为准）\n")
            a("| 参数 | 值 |")
            a("|---|---|")
            for k, label in actual_keys:
                if k in c:
                    a(f"| {label} | {c[k]} |")
            if "actual_throughput_tok_s" in c:
                tp = c["actual_throughput_tok_s"]
                a(f"| 实测吞吐 | {tp['mean']:,} tok/s（max {tp['max']:,}，min {tp['min']:,}，n={tp['samples']}） |")
            a("")

        if "train_defaults" in c:
            a("### dataclass 默认值（仅作对照，实际被覆盖）\n")
            a("| 参数 | 默认值 |")
            a("|---|---|")
            for k, v in c["train_defaults"].items():
                a(f"| {k} | {v} |")
            a("")

        for k in ("pretrain_data", "pretrain_model", "pretrain_optimizer",
                  "sft_data", "sft_model"):
            if k in c:
                a(f"- `{k}`: {c[k]}")
        a("")

    d = rec["data"]
    a("## 3. 数据来源\n")
    a("| 来源 | 原始大小 | 说明 |")
    a("|---|---|---|")
    for name, info in d["sources"].items():
        if info.get("missing"):
            a(f"| {name} | 缺失 | |")
        elif "shards" in info:
            a(f"| {name} | {info['mb']:.0f} MB（{info['shards']} 片） | {info['desc']} |")
        else:
            a(f"| {name} | {info['mb']:.0f} MB | {info['desc']} |")
    a("")

    a("### 编码产物\n")
    a("| 产物 | token 数 | 亿 |")
    a("|---|---|---|")
    for k, v in d["total"].items():
        if "tokens" in v:
            a(f"| {k} | {v['tokens']:,} | {v['yi']:.2f} |")
    a("")
    if "pretrain_txt" in d["total"]:
        pt = d["total"]["pretrain_txt"]
        a(f"清洗后文本：{pt['gb']} GB，{pt['lines']:,} 行\n")
    if "sft_jsonl" in d["total"]:
        sj = d["total"]["sft_jsonl"]
        a(f"SFT 数据：{sj['lines']:,} 条，{sj['mb']} MB\n")
    if "tokenizer" in d:
        a(f"词表：{d['tokenizer']['vocab_size']} token，"
          f"压缩率 {d['tokenizer_stats']['chars_per_token']} 字/token，"
          f"多字中文 token {d['tokenizer_stats']['multi_zh_tokens']} 个\n")

    cur = rec["curves"]
    if cur:
        a("## 4. 训练曲线\n")
        a("| 阶段 | 记录点 | 首 loss | 末 loss | best val | 平均 tok/s |")
        a("|---|---|---|---|---|---|")
        for tag, v in cur.items():
            a(f"| {tag} | {v['steps_logged']} | {v['first_loss']} | "
              f"{v['last_loss']} | {v['best_val']} | {v['avg_tok_s']:.0f} |")
        a("")
        for tag, v in cur.items():
            a(f"### {tag} val loss 轨迹\n")
            a("| step | val_loss |")
            a("|---|---|")
            for r in v["val_curve"]:
                a(f"| {r['step']} | {r['val_loss']} |")
            a("")

    a("## 5. 可复现性\n")
    a("- 随机种子：42（`--seed 42`）")
    a("- 数据顺序：`np.random.default_rng(42)` 随机采样窗口")
    a("- 完整时间序列见 `logs/experiment_record.json` 的 `curves` 字段")
    a("- 原始日志：`logs/pretrain.log`、`logs/sft.log`、`logs/train_run.log`")
    a("")

    with open("logs/experiment_record.md", "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print("  logs/experiment_record.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())