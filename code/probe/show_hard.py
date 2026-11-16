#!/usr/bin/env python3
"""打印困难题的完整回复，供人工评估。"""
import json

d = json.load(open("logs/strict_test.json"))

# 只打印困难题，两个模型并排
for tag, rs in d["detail"].items():
    print(f"\n{'#' * 76}")
    print(f"# {tag}")
    print(f"{'#' * 76}")
    for r in rs:
        if r["level"] != "困难":
            continue
        print(f"\n{'─' * 76}")
        print(f"【问】{r['q']}")
        print(f"【判】{'通过' if r['ok'] else '失败: ' + ', '.join(r['fails'])}")
        print(f"{'─' * 76}")
        ans = r["answers"][0]
        # 按句号断行，便于阅读
        print(ans)
        print(f"\n  [共 {len(ans)} 字，3次一致性 {r['consistency']}]")