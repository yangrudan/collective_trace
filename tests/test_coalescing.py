import argparse
import torch
import torch.distributed as dist

#from torch.distributed import _coalescing_manager
from collective_trace.collective_trace import trace_all_collectives

import sys

def trace_assign(frame, event, arg):
    if event == "assign":
        if frame.f_code.co_name == "_coalescing_manager":
            print(f"_coalescing_manager modified at: {frame.f_code.co_filename}:{frame.f_lineno}")
    return trace_assign

sys.settrace(trace_assign)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sync_mode", action="store_true", help="启用同步通信模式")

    print(f"Test hook before: {dist._coalescing_manager.__name__}")    

    trace_all_collectives(trace_file="collective_trace.log", verbose=True)
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")

    from torch.distributed import _coalescing_manager
    print(f"!!!Test hook After: {dist._coalescing_manager.__name__}")    
    print(f"!!!Test hook After ccccccccccccccccccccccccc: {_coalescing_manager.__name__}")    
    # 示例1：异步模式 - wait在with块内调用
    print("\n=== 异步模式（wait在with块内） ===")
    tensors = [torch.randn(1024, device=device) for _ in range(4)]
    with _coalescing_manager(device=device, async_ops=True) as cm:
        for tensor in tensors:
            dist.all_reduce(tensor)
        cm.wait()  # 在with块内调用wait

    # 示例2：异步模式 - wait在with块外调用
    print("\n=== 异步模式（wait在with块外） ===")
    tensors = [torch.randn(4*1024*1024, device=device) for _ in range(4)]
    with _coalescing_manager(device=device, async_ops=True) as cm:
        for tensor in tensors:
            dist.all_reduce(tensor)
    # 在with块外调用wait
    cm.wait()
   
    print("\n=== 同步模式 ===")
    tensors = [torch.randn(4*1024*1024, device=device) for _ in range(4)]
    with _coalescing_manager(device=device) as cm:
        for tensor in tensors:
            dist.all_reduce(tensor)
    # 在with块外调用wait
    cm.wait()

main()
