import os
import torch
import torch.distributed as dist
import time


# 猴子补丁：监控 batch_isend_irecv 通信数据量
def hook_batch_isend_irecv():
    original_batch_isend_irecv = dist.batch_isend_irecv

    def wrapped_batch_isend_irecv(ops_list):
        send_total = 0
        recv_total = 0
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        for op in ops_list:
            if isinstance(op, dist.P2POp):
                tensor = op.tensor
                data_size = tensor.numel() * tensor.element_size()  # 计算数据量

                # 通过进程号逻辑判断操作类型（结合你的脚本逻辑）
                # 发送操作的目标进程是 (rank + 1) % world_size
                # 接收操作的源进程是 (rank - 1 + world_size) % world_size
                if op.peer == (rank + 1) % world_size:
                    send_total += data_size
                elif op.peer == (rank - 1 + world_size) % world_size:
                    recv_total += data_size

        # 转换为 MB 单位
        send_mb = send_total / (1024 * 1024)
        recv_mb = recv_total / (1024 * 1024)

        print(f"进程 {rank} 通信统计:")
        print(f"  发送数据: {send_total} 字节 ({send_mb:.2f} MB)")
        print(f"  接收数据: {recv_total} 字节 ({recv_mb:.2f} MB)")
        print(
            f"  总数据: {send_total + recv_total} 字节 ({(send_mb + recv_mb):.2f} MB)"
        )

        return original_batch_isend_irecv(ops_list)

    dist.batch_isend_irecv = wrapped_batch_isend_irecv
    print(f"进程 {dist.get_rank()}: 已安装通信监控钩子")


def main():
    dist.init_process_group(backend="nccl")
    if not dist.is_initialized():
        return

    torch.manual_seed(1)
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    print(f"进程 {rank} 的 local rank: {local_rank}")
    torch.cuda.set_device(local_rank)

    # 安装钩子
    hook_batch_isend_irecv()

    send_tensor = []
    recv_tensor = []
    ops_list = []

    for i in range(256):
        # 明确指定设备为 local_rank 对应的 GPU
        send_tensor.append(
            torch.ones((1280, 1280), dtype=torch.float32, device=f"cuda:{local_rank}")
            * rank
            + i * 0.001
        )
        recv_tensor.append(
            torch.randn((1280, 1280), dtype=torch.float32, device=f"cuda:{local_rank}")
        )

    for i in range(256):
        # 创建 P2P 操作（你的原始逻辑）
        send_op = dist.P2POp(dist.isend, send_tensor[i], (rank + 1) % world_size)
        recv_op = dist.P2POp(
            dist.irecv, recv_tensor[i], (rank - 1 + world_size) % world_size
        )
        ops_list.append(send_op)
        ops_list.append(recv_op)

    # 执行批量通信（会被钩子监控）
    start_time = time.time()
    reqs = dist.batch_isend_irecv(ops_list)
    for req in reqs:
        req.wait()  # 等待所有操作完成
    end_time = time.time()

    dist.barrier()

    if rank == 0:
        print(f"总通信时间: {end_time - start_time:.4f} 秒")

    time.sleep(rank * 0.1)  # 避免输出混乱
    if rank == 0:
        for i in range(2):
            print(f"进程 {rank} 接收张量 {i} 前5元素: {recv_tensor[i].flatten()[:5]}")


if __name__ == "__main__":
    main()
