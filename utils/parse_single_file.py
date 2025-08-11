"""
parse_single_file.py
----------------------------------------------------
|对单个文件                  rank0                   |
----------------------------------------------------
统计 all_gather_into_tensor / broadcast 各 Shape 的出现次数与累计耗时
"""

import re
from collections import defaultdict

# ---------------------------------------- 可配置项 ---------------------------------------
LOG_FLODER_FILE = (
    "/home/yang/Downloads/coll-0808-v3/collective_trace_0808.log-0"  # 日志路径
)
# ----------------------------------------------------------------------------------------


def parse_log(path: str):
    """解析日志，返回双层 defaultdict：{op: {shape: {'count': int, 'total_ms': float}}}"""
    stats = defaultdict(lambda: defaultdict(lambda: {"count": 0, "total_ms": 0.0}))

    pat = re.compile(
        r"\[TRACE\] global rank (?P<rank>\d+) in (?P<group>\S+) - "
        r"(?P<op>\w+) - "
        r"async:(?P<async>\d+), "
        r"Size: (?P<size>\d+\.\d+) MB, "
        r"Shape: (?P<shape>\([^)]+\)),"
        r"Dtype: (?P<dtype>[^,]+), "
        r"Duration: (?P<dur>\d+(?:\.\d+)?) ms, "
        r"GROUP size (?P<gs>\d+)  = (?P<ranks>\[[\d, ]+\])"
    )

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            m = pat.match(line.strip())
            if not m:
                continue
            op, shape, dur = m.group("op"), m.group("shape"), float(m.group("dur"))
            stats[op][shape]["count"] += 1
            stats[op][shape]["total_ms"] += dur
    return stats


def pretty_print(stats: dict):
    """控制台打印结果"""
    if not stats:
        print("没有统计数据可显示")
        return

    # 先获取排序后的操作列表，避免在循环中直接使用迭代变量
    sorted_ops = sorted(stats.keys())

    for current_op in sorted_ops:
        print(f"\n=== {current_op} ===")

        # 按总耗时降序排列
        sorted_shapes = sorted(
            stats[current_op].items(),
            key=lambda item: item[1]["total_ms"],
            reverse=True,
        )

        for shape, info in sorted_shapes:
            count = info["count"]
            total_ms = info["total_ms"]
            avg_ms = total_ms / count if count != 0 else 0.0

            print(
                f"输出Shape {str(shape):<20} | "
                f"count={count:>8} | "
                f"total= {total_ms:>10.2f} ms | "
                f"avg= {avg_ms:>8.2f} ms"
            )

    # for op in sorted(stats):
    #     print(f'\n=== {op} ===')

    #     for shape in sorted(stats[op], key=lambda s: stats[op][s]['total_ms'], reverse=True):
    #         info = stats[op][shape]
    #         print(f"输出Shape {shape:<20} | count={info['count']:>8} | "
    #               f"total= {info['total_ms']:.2f} ms | "
    #               f"avg= {info['total_ms']/info['count']:.2f} ms")


def main():
    """主函数"""
    stat = parse_log(LOG_FLODER_FILE)
    pretty_print(stat)


if __name__ == "__main__":
    main()
