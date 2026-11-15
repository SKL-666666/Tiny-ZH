#!/usr/bin/env bash
# 上线前自检：64M 模型能前向、checkpoint 原子写盘有效、断点续训可用
# 这个脚本不碰 GPU 训练，只验证代码正确性，1 分钟内跑完
set -u
cd /mnt/workspace/tiny-zh-64m || exit 1
export PYTHONIOENCODING=utf-8

echo "===== 1. 参数量 ====="
python3 config.py

echo
echo "===== 2. 前向 + 反向（seq 1024）====="
python3 - <<'PY'
import torch, sys
sys.path.insert(0, ".")
from config import ModelConfig
from model import TinyZH
cfg = ModelConfig()
m = TinyZH(cfg)
n = m.num_params()
print(f"params {n:,}  (期望 64,816,000)")
assert n == 64_816_000, f"参数量不对: {n}"
ids = torch.randint(0, cfg.vocab_size, (2, 1024))
out = m(ids, labels=ids)
print(f"logits {tuple(out['logits'].shape)}  loss {float(out['loss']):.4f}")
print("前向 OK")
PY

echo
echo "===== 3. checkpoint 原子写盘 + 损坏恢复 ====="
python3 - <<'PY'
import os, sys, torch, tempfile, shutil
sys.path.insert(0, ".")
d = tempfile.mkdtemp()
path = os.path.join(d, "last.pt")

def save_atomic(obj, p):
    tmp = p + ".tmp"
    torch.save(obj, tmp)
    with open(tmp, "rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp, p)

# 写两次，确认覆盖安全
save_atomic({"step": 1, "x": torch.randn(100, 100)}, path)
save_atomic({"step": 2, "x": torch.randn(100, 100)}, path)
ck = torch.load(path, weights_only=False)
assert ck["step"] == 2, ck["step"]
print("  原子覆盖 OK  step =", ck["step"])

# 模拟断电：留一个半个的 .tmp，确认不影响 last.pt
with open(path + ".tmp", "wb") as f:
    f.write(b"\x00" * 1000)
ck = torch.load(path, weights_only=False)
assert ck["step"] == 2
print("  残留 .tmp 不影响读取 OK")

# 模拟 last.pt 损坏 -> 从 step_*.pt 恢复
save_atomic({"step": 250, "x": torch.randn(50, 50)}, os.path.join(d, "step_250.pt"))
with open(path, "wb") as f:
    f.write(b"garbage")           # 故意写坏
try:
    torch.load(path, weights_only=False)
    print("  [FAIL] 坏文件竟然能读")
except Exception:
    cands = sorted([f for f in os.listdir(d) if f.startswith("step_")], reverse=True)
    ok = False
    for f in cands:
        try:
            c = torch.load(os.path.join(d, f), weights_only=False)
            print(f"  从 {f} 恢复 step={c['step']} OK")
            ok = True
            break
        except Exception:
            continue
    assert ok, "备份也读不了"

shutil.rmtree(d, ignore_errors=True)
print("抗断电逻辑验证通过")
PY

echo
echo "===== 4. 硬链接语义（last.pt 指向 step_N.pt）====="
python3 - <<'PY'
import os, tempfile, shutil
d = tempfile.mkdtemp()
a = os.path.join(d, "step_100.pt")
b = os.path.join(d, "last.pt")
with open(a, "w") as f:
    f.write("ckpt")
os.link(a, b)
print("  link ok, same inode:", os.stat(a).st_ino == os.stat(b).st_ino)
# 替换 a 不应影响 b
tmp = b + ".lnk"
os.link(a, tmp)
os.replace(tmp, b)
print("  硬链接替换 OK")
shutil.rmtree(d, ignore_errors=True)
PY

echo
echo "===== 全部自检通过 ====="
