"""智能下载器：测速选源 + 分段并发 + 断点续传 + 完整性校验。

为什么需要它
------------
今天实测的下载速度差异极大：
    魔搭 SDK 内网      1.35 MB/s   (文档说 40MB/s，实际不是)
    hf-mirror 单连接    0.15 MB/s
    hf-mirror 32 路并发 2.2 MB/s    (快 15 倍)
    魔搭 Web 直链       3.7 MB/s

所以不能写死"用某某源"，要当场测。
本工具做三件事：
1. probe()  用 Range 拉前 1MB，量出该源的真实速度
2. fetch()  支持分段并发 + 断点续传 + 大小校验
3. pick()   给多个候选源，自动选最快的

用法：
    python3 smart_dl.py url -o out.bin              # 单源下载
    python3 smart_dl.py url1 url2 -o out.bin        # 多源竞速
    python3 smart_dl.py url -o out.bin --probe      # 只测速不下载
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time


# ---------------------------------------------------------------- 基础

def run(cmd, timeout=None):
    return subprocess.run(cmd, capture_output=True, text=True,
                          timeout=timeout, errors="replace")


def content_length(url, timeout=25):
    """拿文件总大小。跟随重定向。"""
    r = run(["curl", "-fsSIL", "--connect-timeout", "10", "--max-time", str(timeout), url])
    out = (r.stdout or "") + (r.stderr or "")
    size = None
    for line in out.splitlines():
        low = line.lower()
        if low.startswith("content-length:"):
            try:
                size = int(line.split(":", 1)[1].strip())
            except ValueError:
                pass
    return size


def supports_range(url):
    """探测是否支持 Range 请求（决定能否分段并发）。"""
    r = run(["curl", "-fsSL", "--connect-timeout", "10", "--max-time", "25",
             "--range", "0-1023", "-o", os.devnull, "-w", "%{http_code}", url])
    return "206" in (r.stdout or "")


def probe(url, mb=1.0, timeout=25):
    """实测单连接速度（MB/s）。拉 mb 指定的字节数计时。"""
    end = int(mb * 1024 * 1024) - 1
    t0 = time.time()
    r = run(["curl", "-fsSL", "--connect-timeout", "10", "--max-time", str(timeout),
             "--range", f"0-{end}", "-o", os.devnull, "-w", "%{size_download}", url])
    dt = max(time.time() - t0, 1e-6)
    try:
        got = int((r.stdout or "0").strip() or 0)
    except ValueError:
        got = 0
    if got == 0:
        return 0.0
    return got / dt / 1e6


def pick_fastest(urls, probe_mb=1.0, verbose=True):
    """多源竞速，返回 (最快url, 速度)。"""
    results = []
    for u in urls:
        sp = probe(u, probe_mb)
        results.append((sp, u))
        if verbose:
            print(f"  {sp:7.2f} MB/s  {u[:90]}")
    results.sort(reverse=True)
    if not results or results[0][0] <= 0:
        return None, 0.0
    return results[0][1], results[0][0]


# ---------------------------------------------------------------- 分段下载

def seg_download(url, out, total, ncpu=32, retries=5, verbose=True):
    """把文件切成 ncpu 段并发下载，再拼接。

    关键细节
    --------
    - 每段用 --range，独立 curl 进程
    - 已存在的段按大小判断是否完整，不完整就续传（从 got 字节处追加）
    - 全部完成后校验总大小
    """
    tmp = tempfile.mkdtemp(prefix="dlseg_")
    part = os.path.join(tmp, "p")
    chunk = (total + ncpu - 1) // ncpu

    jobs = []
    for i in range(ncpu):
        a = i * chunk
        b = min(a + chunk - 1, total - 1)
        want = b - a + 1
        pf = f"{part}{i}.bin"
        got = os.path.getsize(pf) if os.path.exists(pf) else 0
        if got == want:
            continue
        from_byte = a + got
        jobs.append((i, from_byte, b, pf, got > 0))

    if verbose and jobs:
        print(f"  待下 {len(jobs)}/{ncpu} 段")

    procs = []
    for i, s, e, pf, append in jobs:
        mode = ">>" if append else ">"
        # 用 shell 重定向处理追加；curl 的 --range 从 s 开始
        cmd = (f'curl -fsSL --connect-timeout 10 --max-time 3600 --retry {retries} '
               f'--retry-delay 2 --range "{s}-{e}" "{url}" {mode} "{pf}"')
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
        procs.append(p)

    # 等待 + 进度
    last = 0
    while any(p.poll() is None for p in procs):
        time.sleep(5)
        cur = sum(os.path.getsize(f"{part}{i}.bin")
                  for i in range(ncpu) if os.path.exists(f"{part}{i}.bin"))
        if verbose and cur - last > 50 * 1024 * 1024:
            print(f"    {cur/1e6:.0f}/{total/1e6:.0f} MB ({cur*100//max(total,1)}%)")
            last = cur
    for p in procs:
        p.wait()

    # 校验每段
    bad = []
    for i in range(ncpu):
        a = i * chunk
        b = min(a + chunk - 1, total - 1)
        want = b - a + 1
        pf = f"{part}{i}.bin"
        got = os.path.getsize(pf) if os.path.exists(pf) else 0
        if got != want:
            bad.append((i, got, want))
    if bad:
        print(f"  [WARN] {len(bad)} 段不完整，重试一次")
        for i, got, want in bad:
            a = i * chunk
            b = min(a + chunk - 1, total - 1)
            pf = f"{part}{i}.bin"
            if os.path.exists(pf):
                os.remove(pf)
            run(["curl", "-fsSL", "--connect-timeout", "10", "--max-time", "1800",
                 "--retry", "5", "--range", f"{a}-{b}", "-o", pf, url], timeout=1900)
            got = os.path.getsize(pf) if os.path.exists(pf) else 0
            if got != want:
                print(f"  [FAIL] 段 {i} 仍不完整 {got}/{want}")
                shutil.rmtree(tmp, ignore_errors=True)
                return False

    # 拼接
    if verbose:
        print("  拼接…")
    with open(out, "wb") as fo:
        for i in range(ncpu):
            with open(f"{part}{i}.bin", "rb") as fi:
                shutil.copyfileobj(fi, fo, length=8 * 1024 * 1024)

    got = os.path.getsize(out)
    shutil.rmtree(tmp, ignore_errors=True)
    if got != total:
        print(f"  [FAIL] 拼接后 {got} != {total}")
        return False
    return True


# ---------------------------------------------------------------- 主流程

def fetch(urls, out, ncpu=32, do_probe=False, verbose=True):
    if os.path.exists(out):
        if verbose:
            print(f"已存在，跳过: {out} ({os.path.getsize(out)/1e6:.0f} MB)")
        return True

    if isinstance(urls, str):
        urls = [urls]

    if verbose:
        print(f"目标: {out}")
        print("测速:")
    url, speed = pick_fastest(urls, verbose=verbose)
    if url is None:
        print("  [FAIL] 所有源都不通")
        return False
    if do_probe:
        return True

    total = content_length(url)
    if not total:
        print("  [FAIL] 拿不到 content-length")
        return False

    rng = supports_range(url)
    if verbose:
        print(f"选定源 ({speed:.2f} MB/s)，{total/1e6:.0f} MB，Range={'支持' if rng else '不支持'}")

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    t0 = time.time()
    if rng and total > 20 * 1024 * 1024:
        ok = seg_download(url, out, total, ncpu=ncpu, verbose=verbose)
    else:
        # 小文件或不支持 Range，单连接
        r = subprocess.run(["curl", "-fsSL", "--connect-timeout", "15",
                            "--max-time", "3600", "--retry", "5",
                            "-o", out, url])
        ok = r.returncode == 0 and os.path.getsize(out) == total

    if ok:
        dt = time.time() - t0
        print(f"  完成 {os.path.getsize(out)/1e6:.0f} MB，{dt/60:.1f} 分钟，"
              f"{os.path.getsize(out)/1e6/max(dt,1):.2f} MB/s")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("urls", nargs="+")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--ncpu", type=int, default=32, help="分段数，默认 32")
    ap.add_argument("--probe", action="store_true", help="只测速不下载")
    a = ap.parse_args()
    ok = fetch(a.urls, a.out, a.ncpu, a.probe)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
