import os
import torch
import argparse
import torch.distributed as dist
from torch.distributed import ReduceOp
from datetime import datetime
import time
import argparse
import numpy as np
import torch.distributed

from collective_trace.collective_trace import trace_all_collectives

tracer = trace_all_collectives(trace_file="collective_trace.log", verbose=True)


def main():
    dist.init_process_group(backend="nccl")
    if not torch.distributed.is_initialized():
        return

    torch.manual_seed(1)
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    print("local rank is:", local_rank)
    torch.cuda.set_device(local_rank)

    send_tensor = []
    recv_tensor = []

    ops_list = []
    for i in range(256):
        send_tensor.append(
            torch.ones((1280, 1280), dtype=torch.float32, device=f"cuda") * rank
            + i * 0.001
        )
        recv_tensor.append(
            torch.randn((1280, 1280), dtype=torch.float32, device=f"cuda")
        )

    for i in range(256):
        send_op = dist.P2POp(dist.isend, send_tensor[i], (rank + 1) % world_size)
        recv_op = dist.P2POp(
            dist.irecv, recv_tensor[i], (rank - 1 + world_size) % world_size
        )
        ops_list.append(send_op)
        ops_list.append(recv_op)

    reqs = dist.batch_isend_irecv(ops_list)
    torch.distributed.barrier()

    import time

    time.sleep(rank * 3)

    for i in range(256):
        print(
            "recv tensor is:",
            i,
            recv_tensor[i].reshape(-1),
            torch.distributed.get_rank(),
        )


if __name__ == "__main__":
    main()
