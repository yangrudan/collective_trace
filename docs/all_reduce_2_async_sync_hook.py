import argparse
import torch
import torch.distributed as dist
import os
import time
from functools import wraps


class TimedWork:
    def __init__(self, work, start_time, func_name, data_size):
        self.work = work
        self.start_time = start_time
        self.func_name = func_name
        self.data_size = data_size

    def wait(self):
        result = self.work.wait()
        end_time = time.time()
        duration = (end_time - self.start_time) * 1000
        print(
            f"Rank {dist.get_rank()}: {self.func_name} | 数据量[异步模式]: {self.data_size/1e6:.2f}MB | 耗时: {duration:.2f}ms"
        )
        return result

    def is_completed(self):
        return self.work.is_completed()


def trace_async_collective(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        tensor = args[0] if args else None
        print(
            f"tensor.numel={tensor.numel()}   tensor.element_size={tensor.element_size()}\n"
        )
        data_size = tensor.numel() * tensor.element_size() if tensor is not None else 0

        is_async = kwargs.get("async_op", False)
        if is_async:
            work = func(*args, **kwargs)
            return TimedWork(work, start_time, func.__name__, data_size)
        else:
            work = func(*args, **kwargs)
            end_time = time.time()
            duration = (end_time - start_time) * 1000  # 毫秒
            print(
                f"Rank {dist.get_rank()}: {func.__name__} | 数据量[同步模式]: {data_size/1e6:.2f}MB | 耗时: {duration:.2f}ms"
            )
            return work

    return wrapper


# 对同步/异步通信函数进行包装
dist.all_reduce = trace_async_collective(dist.all_reduce)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sync_mode", action="store_true", help="启用同步通信模式")
    args = parser.parse_args()

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # 模拟梯度
    gradient = torch.tensor([rank] * 3, dtype=torch.float32, device=device)
    print(f"Rank {rank} 初始梯度: {gradient.cpu().numpy()}")

    if args.sync_mode:
        dist.all_reduce(gradient, op=dist.ReduceOp.SUM)
    else:
        work = dist.all_reduce(gradient, op=dist.ReduceOp.SUM, async_op=True)

    # 模拟计算重叠
    dummy_tensor = torch.ones(1, device=device)
    for _ in range(1000):
        dummy_tensor = dummy_tensor * 2 + 1

    if args.sync_mode:
        print(f"Rank {rank} [同步]all_reduce后梯度: {gradient.cpu().numpy()}")
    else:
        work.wait()
        print(f"Rank {rank} [异步]all_reduce后梯度: {gradient.cpu().numpy()}")

    dist.destroy_process_group()


if __name__ == "__main__":
    # 启动命令：torchrun --nproc_per_node=4 this_script.py
    main()
