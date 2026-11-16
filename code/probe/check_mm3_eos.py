#!/usr/bin/env python3
"""检查 minimind-3 的 EOS 配置，修正测试脚本。"""
import json
import os

D = "out_base/minimind3"

tc = json.load(open(f"{D}/tokenizer_config.json"))
cfg = json.load(open(f"{D}/config.json"))
tj = json.load(open(f"{D}/tokenizer.json"))

print("=== tokenizer_config ===")
for k in ("eos_token", "bos_token", "pad_token", "unk_token"):
    print(f"  {k}: {tc.get(k)}")

print("\n=== config ===")
for k in ("eos_token_id", "bos_token_id", "pad_token_id"):
    print(f"  {k}: {cfg.get(k)}")

print("\n=== 特殊 token id ===")
vocab = tj["model"]["vocab"]
inv = {v: k for k, v in vocab.items()}
for i in range(6):
    print(f"  {i}: {inv.get(i)!r}")

print("\n=== added_tokens ===")
for a in tj.get("added_tokens", [])[:8]:
    print(f"  {a.get('id')}: {a.get('content')!r}")

# chat template 结尾是什么
jinja = f"{D}/chat_template.jinja"
if os.path.exists(jinja):
    txt = open(jinja, encoding="utf-8").read()
    print(f"\n=== chat_template 结尾 300 字 ===")
    print(txt[-300:])
