import argparse
import torch
import torch.distributed as dist
import os
import time
from functools import wraps

from collective_trace.collective_trace import trace_all_collectives

tracer = trace_all_collectives(trace_file="collective_trace.log", verbose=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sync_mode", action="store_true", help="启用同步通信模式")
    args = parser.parse_args()

    # CPU 场景使用 gloo 后端
    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()

    # Mock gradient（CPU 张量）
    gradient = torch.tensor([rank] * 3, dtype=torch.float32)
    print(f"Rank {rank} 初始梯度: {gradient.numpy()}")

    if args.sync_mode:
        dist.all_reduce(gradient, op=dist.ReduceOp.SUM)
    else:
        work = dist.all_reduce(gradient, op=dist.ReduceOp.SUM, async_op=True)

    # Mock computation to simulate workload（CPU 上随便算点）
    dummy_tensor = torch.ones(1)
    for _ in range(1000):
        dummy_tensor = dummy_tensor * 2 + 1

    if args.sync_mode:
        print(f"Rank {rank} [同步]all_reduce后梯度: {gradient.numpy()}")
    else:
        work.wait()
        print(f"Rank {rank} [异步]all_reduce后梯度: {gradient.numpy()}")

    # 导出追踪数据
    global tracer
    tracer.export_to_csv(f"aaa_{rank}.txt")

    print(tracer.get_all_call_counts())

    dist.destroy_process_group()


if __name__ == "__main__":
    # 运行示例（4 进程 CPU）
    # torchrun --nproc_per_node=4 test_in_cpu.py [--sync_mode]
    main()
