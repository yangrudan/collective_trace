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

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # 为reduce_scatter准备数据
    # 每个进程创建一个包含world_size个元素的张量
    # 最终每个进程会收到归约后对应位置的元素
    input_tensor = torch.tensor(
        [rank * world_size + i for i in range(world_size)],
        dtype=torch.float32,
        device=device,
    )
    print(f"Rank {rank} 输入张量: {input_tensor.cpu().numpy()}")

    # 输出张量大小应为输入张量大小 / world_size
    output_tensor = torch.empty(
        input_tensor.numel() // world_size, dtype=torch.float32, device=device
    )

    if args.sync_mode:
        # 同步版本的reduce_scatter_tensor
        dist.reduce_scatter_tensor(output_tensor, input_tensor, op=dist.ReduceOp.SUM)
    else:
        # 异步版本的reduce_scatter_tensor
        work = dist.reduce_scatter_tensor(
            output_tensor, input_tensor, op=dist.ReduceOp.SUM, async_op=True
        )

    # 模拟计算负载
    dummy_tensor = torch.ones(1, device=device)
    for _ in range(1000):
        dummy_tensor = dummy_tensor * 2 + 1

    if args.sync_mode:
        print(f"Rank {rank} [同步]reduce_scatter后结果: {output_tensor.cpu().numpy()}")
    else:
        # 等待异步操作完成
        work.wait()
        print(f"Rank {rank} [异步]reduce_scatter后结果: {output_tensor.cpu().numpy()}")

    # 导出追踪数据到CSV文件
    # tracer.export_to_csv(f"reduce_scatter_{rank}.csv")

    dist.destroy_process_group()


if __name__ == "__main__":
    # 运行命令：torchrun --nproc_per_node=4 this_script.py
    # 启用同步模式：torchrun --nproc_per_node=4 this_script.py --sync_mode
    main()
