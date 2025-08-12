"""
parse_single_file.py
----------------------------------------------------
|对单个文件                  rank0                   |
----------------------------------------------------
统计 all_gather_into_tensor / broadcast 各 Shape 的出现次数与累计耗时
"""

from collections import defaultdict
from pretty_print import pretty_print, parse_pat

# ---------------------------------------- 可配置项 ---------------------------------------
LOG_FLODER_FILE = (
    "/home/yang/Downloads/coll-0811/collective_trace_0811.log-17"
)
# ----------------------------------------------------------------------------------------


def parse_log(path: str):
    """解析日志，返回双层 defaultdict：{op: {shape: {'count': int, 'total_ms': float}}}"""
    stats = defaultdict(lambda: defaultdict(lambda: {"count": 0, "total_ms": 0.0}))

    pat = parse_pat()

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            m = pat.match(line.strip())
            if not m:
                continue
            op, shape, dur = m.group("op"), m.group("shape"), float(m.group("dur"))
            stats[op][shape]["count"] += 1
            stats[op][shape]["total_ms"] += dur
    return stats

def main():
    """主函数"""
    stat = parse_log(LOG_FLODER_FILE)
    pretty_print(stat)


if __name__ == "__main__":
    main()
