"""
parse_coll_info.py
统计 all_gather_into_tensor / broadcast 各 Shape 的出现次数与累计耗时
"""
import os
import re
from collections import defaultdict

# ----------------------------- 可配置项 ------------------------
LOG_FLODER_FILE = '/home/yang/Downloads/coll-0808-v2'          # 日志路径
# --------------------------------------------------------------


def parse_log(path: str):
    """解析日志，返回双层 defaultdict：{op: {shape: {'count': int, 'total_ms': float}}}"""
    # 创建一个双层 defaultdict，用于存储日志信息
    stats = defaultdict(lambda: defaultdict(lambda: {'count': 0, 'total_ms': 0.0}))

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

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            m = pat.match(line.strip())
            if not m:
                continue
            op, shape, dur = m.group('op'), m.group('shape'), float(m.group('dur'))
            stats[op][shape]['count'] += 1
            stats[op][shape]['total_ms'] += dur
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

def parse_folder(path: str):
    """遍历文件夹下的所有文件，并调用 parse_log 函数进行处理"""
    stats = defaultdict(lambda: defaultdict(lambda: {'count': 0, 'total_ms': 0.0}))
    for root, _, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            print(f'==================== {file_path} =====================')
            stat = parse_log(file_path)
            for op in stat:
                for shape, info in stat[op].items():
                    stats[op][shape]['count'] += info['count']
                    stats[op][shape]['total_ms'] += info['total_ms']
    pretty_print(stats)

def main():
    """主函数"""
    parse_folder(LOG_FLODER_FILE)

if __name__ == '__main__':
    main()
