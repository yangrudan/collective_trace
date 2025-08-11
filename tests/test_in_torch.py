import argparse
import torch
import torch.distributed as dist
import os
import time
from functools import wraps

from collective_trace.collective_trace import trace_all_collectives


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sync_mode", action="store_true", help="启用同步通信模式")
    args = parser.parse_args()

    tracer = trace_all_collectives(trace_file="collective_trace.log", verbose=True)
    dist.init_process_group(backend="nccl")

    # 启用追踪
    # tracer = trace_all_collectives(trace_file='collective_trace.log')

    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # Mock gradient
    gradient = torch.tensor([rank] * 3, dtype=torch.float32, device=device)
    print(f"Rank {rank} 初始梯度: {gradient.cpu().numpy()}")

    if args.sync_mode:
        dist.all_reduce(gradient, op=dist.ReduceOp.SUM)
    else:
        work = dist.all_reduce(gradient, op=dist.ReduceOp.SUM, async_op=True)

    # Mock computation to simulate workload
    dummy_tensor = torch.ones(1, device=device)
    for _ in range(1000):
        dummy_tensor = dummy_tensor * 2 + 1

    if args.sync_mode:
        print(f"Rank {rank} [同步]all_reduce后梯度: {gradient.cpu().numpy()}")
    else:
        work.wait()
        print(f"Rank {rank} [异步]all_reduce后梯度: {gradient.cpu().numpy()}")

    # Export trace data to CSV file
    tracer.export_to_csv(f"aaa_{rank}.csv")

    dist.destroy_process_group()


if __name__ == "__main__":
    # cmd：torchrun --nproc_per_node=4 this_script.py
    main()
