#!/usr/bin/env python3
"""下载并验证 minimind-3 原版（Qwen3 架构，safetensors）。

为什么单独处理
--------------
minimind-3 不是 GGUF，是 HuggingFace 格式 + Qwen3 架构 + 它自己的分词器。
不能和我们的 GGUF 用同一套 llama.cpp 加载，必须用 transformers。

维度对比
--------
minimind-3:  Qwen3, hidden 768, 8 层, vocab 6400（待确认）
Tiny-ZH-64M: custom, hidden 640, 13 层, vocab 6400
"""
import json
import os
import sys

from huggingface_hub import hf_hub_download

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

REPO = "jingyaogong/minimind-3"
DST = "out_base/minimind3"

FILES = [
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
]


def main():
    os.makedirs(DST, exist_ok=True)
    print(f"下载 {REPO} -> {DST}")

    ok = 0
    for fn in FILES:
        out = os.path.join(DST, fn)
        if os.path.exists(out) and os.path.getsize(out) > 100:
            print(f"  [cached] {fn}")
            ok += 1
            continue
        try:
            p = hf_hub_download(repo_id=REPO, filename=fn,
                                local_dir=DST, repo_type="model")
            sz = os.path.getsize(p)
            print(f"  [got] {fn} {sz/1e6:.1f} MB" if sz > 1e6 else f"  [got] {fn} {sz} bytes")
            ok += 1
        except Exception as e:
            print(f"  [FAIL] {fn}: {str(e)[:120]}")

    print(f"\n{ok}/{len(FILES)} 个文件就位")

    # 读 config 看架构
    cfg_path = os.path.join(DST, "config.json")
    if os.path.exists(cfg_path):
        cfg = json.load(open(cfg_path))
        print("\n=== minimind-3 架构 ===")
        for k in ("architectures", "hidden_size", "num_hidden_layers",
                  "num_attention_heads", "num_key_value_heads",
                  "intermediate_size", "vocab_size", "max_position_embeddings",
                  "tie_word_embeddings", "torch_dtype"):
            if k in cfg:
                print(f"  {k:<28} {cfg[k]}")
        # 估算参数
        h = cfg.get("hidden_size", 0)
        L = cfg.get("num_hidden_layers", 0)
        I = cfg.get("intermediate_size", 0)
        V = cfg.get("vocab_size", 0)
        hd = cfg.get("head_dim", h // max(cfg.get("num_attention_heads", 1), 1))
        nh = cfg.get("num_attention_heads", 1)
        nkv = cfg.get("num_key_value_heads", nh)
        attn = h * nh * hd + 2 * h * nkv * hd + nh * hd * h
        mlp = 3 * h * I
        per = attn + mlp + 2 * h
        total = V * h + L * per + h
        print(f"\n  估算参数 {total:,} ({total/1e6:.2f}M)")
    return 0


if __name__ == "__main__":
    sys.exit(main())