"""
Print the result to console
"""
import re


def parse_pat():
    """正则表达式模式"""
    pat = re.compile(
    r"\[TRACE\] global rank (?P<rank>\d+) "
    r"in (?P<group>\S+) - "
    r"(?P<op>\w+) - "
    r"async:(?P<async>\d+), "
    r"Size: (?P<size>\d+(?:\.\d+)?) MB, "
    r"Shape: (?P<shape>\([^)]+\)), "              # ← 圆括号
    r"Dtype: (?P<dtype>[^,]+), "
    r"Duration: (?P<dur>\d+(?:\.\d+)?) ms, "
    r"GROUP size (?P<gs>\d+)\s*=\s*"             # ← 空格任意
    r"(?P<ranks>\[[\d,\s]+\]), call count: (?P<count>\d+)"
    )
#    pat = re.compile(
#        r"\[TRACE\] global rank (?P<rank>\d+) in (?P<group>\S+) - "
#        r"(?P<op>\w+) - "
#        r"async:(?P<async>\d+), "
#        r"Size: (?P<size>\d+\.\d+) MB, "
#        r"Shape: (?P<shape>\([^)]+\)), "
#        r"Dtype: (?P<dtype>[^,]+), "
#        r"Duration: (?P<dur>\d+(?:\.\d+)?) ms, "
#        r"GROUP size (?P<gs>\d+)  = (?P<ranks>\[[\d, ]+\]) "
#    )
    return pat

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
