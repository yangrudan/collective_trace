"""
parse_coll_info.py
统计 all_gather_into_tensor / broadcast 各 Shape 的出现次数与累计耗时
"""
import os

from collections import defaultdict
from pretty_print import pretty_print, parse_pat

# ----------------------------- 可配置项 ------------------------
LOG_FLODER_FILE = '/home/yang/Downloads/0821'          # 日志路径
# --------------------------------------------------------------


def parse_log(path: str):
    """解析日志，返回双层 defaultdict：{op: {shape: {'count': int, 'total_ms': float}}}"""
    # 创建一个双层 defaultdict，用于存储日志信息
    stats = defaultdict(lambda: defaultdict(lambda: {'count': 0, 'total_ms': 0.0}))

    pat = parse_pat()

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            m = pat.match(line.strip())
            if not m:
                continue
            op, shape, dur = m.group('op'), m.group('shape'), float(m.group('dur'))
            stats[op][shape]['count'] += 1
            stats[op][shape]['total_ms'] += dur
    return stats

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
    print(f"{stats}")
    pretty_print(stats)

def main():
    """主函数"""
    parse_folder(LOG_FLODER_FILE)

if __name__ == '__main__':
    main()
