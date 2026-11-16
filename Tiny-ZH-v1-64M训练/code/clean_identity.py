#!/usr/bin/env python3
"""清理 SFT 数据里的身份标签，重训一个「无品牌」版本。

问题
----
minimind 的 SFT 数据里大量出现「我是 minimind」「由中国的个人开发者开发」，
模型学会了这些身份声明，问「你是谁」就答「我叫 minimind」。

清理方式
--------
把身份相关的表述替换成中性的：
  minimind / MiniMind     -> 小助手
  由中国的个人开发者开发    -> 删掉该从句
  参考官方文档             -> 删掉该从句

注意：不删除整条样本（会损失数据），只替换文本。
"""
import json
import os
import re
import sys

SRC = "data/sft_high.jsonl"
OUT = "data/sft_clean.jsonl"

# 身份标签替换表（按顺序应用）
REPLACEMENTS = [
    # minimind 作者名（实测发现数据里大量出现）
    (re.compile(r"jingyaogong", re.I), ""),
    (re.compile(r"Jingyao\s*Gong", re.I), ""),
    (re.compile(r"minimind|MiniMind|MINIMIND", re.I), "小助手"),
    # 「由 XX 创建/开发」整句
    (re.compile(r"我是由[^，。]{0,30}(?:创建|开发|训练|制作)的", re.I), "我是一个"),
    (re.compile(r"由中国的个人开发者(?:独立)?开发(?:的)?", re.I), ""),
    (re.compile(r"中国的个人开发者", re.I), ""),
    (re.compile(r"由[^，。]{0,20}(?:创建|开发|训练)的(?:高效)?", re.I), ""),
    (re.compile(r"有关模型和产品的详细内容请参考官方文档[。！]?", re.I), ""),
    (re.compile(r"请参考官方文档[。！]?", re.I), ""),
    (re.compile(r"设计原型为基于VR技术的智能助理[，,]?", re.I), ""),
    # 兜底：剩余的品牌式修饰
    (re.compile(r"(?:高效|小参数)AI模型", re.I), "AI模型"),
]


def clean_identity(t):
    for pat, rep in REPLACEMENTS:
        t = pat.sub(rep, t)
    # 清理替换后产生的不通顺结构
    t = re.sub(r"我是由创造的", "我是一个", t)
    t = re.sub(r"我是由(?:的)?", "我是一个", t)
    t = re.sub(r"我是由[^，。]{0,10}的", "我是一个", t)
    t = re.sub(r"由创造的", "", t)
    t = re.sub(r"[，,]{2,}", "，", t)
    t = re.sub(r"^[，,。.]", "", t)
    t = re.sub(r"^\s*的", "", t)
    t = re.sub(r"\s{2,}", " ", t)
    return t.strip()


def main():
    if not os.path.exists(SRC):
        print(f"[FATAL] 缺 {SRC}")
        return 1

    n_in = n_out = n_changed = 0
    with open(SRC, encoding="utf-8", errors="ignore") as f, \
         open(OUT, "w", encoding="utf-8") as fo:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            conv = o.get("conversations") or []
            new_conv = []
            changed = False
            for m in conv:
                c = m.get("content", "")
                nc = clean_identity(c)
                if nc != c:
                    changed = True
                # 替换后太短的回答就丢掉这条
                if m.get("role") == "assistant" and len(nc) < 10:
                    changed = True
                    nc = None
                if nc is not None:
                    new_conv.append({"role": m.get("role"), "content": nc})
            if len(new_conv) < 2:
                continue
            if changed:
                n_changed += 1
            fo.write(json.dumps({"conversations": new_conv},
                                ensure_ascii=False) + "\n")
            n_out += 1

    print(f"输入 {n_in:,} 条")
    print(f"输出 {n_out:,} 条")
    print(f"改动 {n_changed:,} 条 ({n_changed/max(n_out,1)*100:.1f}%)")
    print(f"文件 {os.path.getsize(OUT)/1e6:.0f} MB -> {OUT}")

    # 抽样验证
    print("\n=== 抽样验证（含身份词的行） ===")
    shown = 0
    with open(OUT, encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            for m in o["conversations"]:
                if m["role"] == "assistant" and shown < 3:
                    if "助手" in m["content"] or "我是" in m["content"]:
                        print(f"  {m['content'][:100]}")
                        shown += 1
            if shown >= 3:
                break
    return 0


if __name__ == "__main__":
    sys.exit(main())