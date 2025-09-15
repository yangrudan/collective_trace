import argparse
import torch
import torch.distributed as dist

from torch.distributed.distributed_c10d import _coalescing_manager

from collective_trace.collective_trace import trace_all_collectives


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sync_mode", action="store_true", help="启用同步通信模式")

    trace_all_collectives(trace_file="collective_trace.log", verbose=True)
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")

    # 示例1：异步模式 - wait在with块内调用
    print("\n=== 异步模式（wait在with块内） ===")
    tensors = [torch.randn(1024, device=device) for _ in range(4)]
    with _coalescing_manager(device=device, async_ops=True) as cm:
        for tensor in tensors:
            dist.all_reduce(tensor)
        cm.wait()  # 在with块内调用wait

    # 示例2：异步模式 - wait在with块外调用
    print("\n=== 异步模式（wait在with块外） ===")
    tensors = [torch.randn(1024, device=device) for _ in range(4)]
    with _coalescing_manager(device=device, async_ops=True) as cm:
        for tensor in tensors:
            dist.all_reduce(tensor)
    # 在with块外调用wait
    cm.wait()


main()
