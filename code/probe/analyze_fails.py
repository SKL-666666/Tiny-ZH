#!/usr/bin/env python3
"""分析严厉测试的失败案例"""
import json

d = json.load(open("logs/strict_test.json"))

for tag, rs in d["detail"].items():
    print(f"\n{'='*70}")
    print(f"{tag}")
    print(f"{'='*70}")
    fails = [r for r in rs if not r["ok"]]
    print(f"失败 {len(fails)}/{len(rs)} 条\n")
    for r in fails:
        print(f"[{r['level']}] {r['q']}")
        print(f"  失败: {', '.join(r['fails'])}")
        a = r["answers"][0].replace("\n", " ")
        print(f"  答: {a[:150]}")
        print()