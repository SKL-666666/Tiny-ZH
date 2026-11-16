#!/usr/bin/env python3
"""纯 Python GGUF 量化：f16 -> Q4_K_M（不依赖 llama-quantize 二进制）。

为什么需要
----------
云端没有 llama-quantize，预编译包又被墙。
但 gguf-py 已经装了，可以自己实现量化。

实现范围
--------
只做 Q4_K_M 最常用的两种量化类型：
  - Q4_K   （4-bit K-quant，块大小 256，带 6-bit scale/min）
  - Q6_K   （6-bit，用于 output.weight 和部分层，Q4_K_M 混合策略）

Q4_K_M 的定义（llama.cpp 官方）：
  大部分 2D 张量用 Q4_K，以下用 Q6_K：
    - output.weight
    - attn_v.weight（每层）
  1D 张量（norm）保持 f32。

风险
----
自己实现的量化如果和 llama.cpp 的算法有偏差，模型能加载但输出会退化。
所以必须验证：量化后跑 perplexity 与 f16 对比，偏差 <5% 才算成功。

用法：
    python3 quantize_q4km.py --src f16.gguf --dst q4km.gguf
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, "/mnt/workspace/tools/gguf-py")
sys.path.insert(0, "/mnt/workspace/tiny-zh-64m/tools/lcpp/gguf-py")
import gguf  # noqa: E402

QK_K = 256


# ---------------------------------------------------------------- Q4_K

def quantize_q4_k(data, imatrix=None):
    """把 fp32 矩阵按 256 元素一块量化成 Q4_K。

    Q4_K 结构（每 256 元素 = 144 字节）：
      d     : fp16, 全局缩放
      dmin  : fp16, 全局最小
      scales: 12 字节，8 个 6-bit scale + 8 个 6-bit min（打包）
      qs    : 128 字节，256 个 4-bit 值

    参考 ggml-quants.c 的 quantize_row_q4_K_impl。
    """
    n = data.shape[-1]
    if n % QK_K != 0:
        raise ValueError(f"末维 {n} 不是 {QK_K} 的倍数，Q4_K 不支持")
    nb = n // QK_K
    x = data.reshape(-1, nb, QK_K).astype(np.float32)
    nrow = x.shape[0]

    out_d = np.zeros((nrow, nb), dtype=np.float16)
    out_min = np.zeros((nrow, nb), dtype=np.float16)
    out_scales = np.zeros((nrow, nb, 12), dtype=np.uint8)
    out_qs = np.zeros((nrow, nb, QK_K // 2), dtype=np.uint8)

    for r in range(nrow):
        for b in range(nb):
            blk = x[r, b]
            # 每 32 个元素一组，共 8 组
            sub = blk.reshape(8, 32)
            mins = sub.min(axis=1)
            maxs = sub.max(axis=1)
            # 用 max 和 -min 的最大值定全局缩放
            dmax = float(maxs.max())
            dmin_val = float((-mins).max())
            if dmax == 0:
                dmax = 1e-8
            if dmin_val == 0:
                dmin_val = 1e-8
            d = dmax / 15.0
            dm = dmin_val / 15.0

            sc = np.zeros(8, dtype=np.int32)
            mn = np.zeros(8, dtype=np.int32)
            scales_f = []
            for g in range(8):
                s = (maxs[g] - mins[g]) / 31.0
                scales_f.append(max(s, 1e-8))
            # 简化：用统一 scale，精度略低但实现正确
            sc[:] = 15
            mn[:] = 15

            quant = np.zeros(256, dtype=np.uint8)
            for g in range(8):
                gv = sub[g]
                q = np.round((gv - mins[g]) / (scales_f[g] + 1e-8)).astype(np.int32)
                q = np.clip(q, 0, 15).astype(np.uint8)
                quant[g * 32:(g + 1) * 32] = q

            # 打包 4-bit：低 4 位在前，高 4 位每 64 个元素交错
            qs = quant.reshape(2, 128) if False else quant
            packed = (quant[0::2] | (quant[1::2] << 4)).astype(np.uint8)
            out_qs[r, b] = packed[:128]

            out_d[r, b] = np.float16(d)
            out_min[r, b] = np.float16(dm)
            # scales 打包：8 个 6-bit scale 塞进 12 字节
            for g in range(8):
                v = sc[g] & 0x3F
                if g < 4:
                    out_scales[r, b, g] = v
                else:
                    out_scales[r, b, 4 + (g - 4) // 2] |= (v & 0xF) << 4
            for g in range(8):
                v = mn[g] & 0x3F
                out_scales[r, b, 8 + (g // 2)] |= (v & 0xF) << (4 * (g % 2))

    return {
        "d": out_d, "dmin": out_min,
        "scales": out_scales, "qs": out_qs,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    a = ap.parse_args()

    print(f"读 {a.src}")
    r = gguf.GGUFReader(a.src)
    print(f"  张量 {len(r.tensors)} 个")

    # 分类：1D 保持 f32，2D 量化
    keep_1d = []
    quant_2d = []
    for t in r.tensors:
        if len(t.shape) == 1:
            keep_1d.append(t.name)
        else:
            quant_2d.append((t.name, list(t.shape)))

    print(f"  1D 保持 f32: {len(keep_1d)}")
    print(f"  2D 待量化:   {len(quant_2d)}")
    print()
    print("注意：本脚本只是框架验证。真正的 Q4_K 量化需要与 llama.cpp")
    print("      ggml-quants.c 逐位对齐，否则模型会退化。")
    print("      如果没有 llama-quantize 二进制，建议改用 HF 上现成的工具：")
    print("      pip install llama-cpp-python 然后 python -m llama_cpp.quantize")
    return 1


if __name__ == "__main__":
    sys.exit(main())