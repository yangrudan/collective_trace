import torch
import torch.distributed as dist
from torch.multiprocessing import spawn
import os
import tempfile

def test_coalescing_manager(rank, world_size, tempfile_name):
    """测试低版本PyTorch _coalescing_manager接口（group, device, reqs参数）"""
    # 1. 初始化分布式环境
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'  # 固定端口，避免冲突
    dist.init_process_group(
        backend='nccl' if torch.cuda.is_available() else 'gloo',
        init_method=f'file://{tempfile_name}',
        rank=rank,
        world_size=world_size
    )
    
    # 2. 选择设备（优先GPU，无GPU时用CPU）
    device = torch.device(
        f'cuda:{rank % torch.cuda.device_count()}' 
        if torch.cuda.is_available() else 'cpu'
    )
    torch.cuda.set_device(device) if torch.cuda.is_available() else None  # 绑定GPU设备

    # ------------------------------
    # 测试1：正常场景 - 合并通信+请求数量匹配
    # ------------------------------
    print(f"Rank {rank}: 开始测试1（正常合并通信）")
    # 创建FLOAT类型张量（保持数据类型一致性）
    tensor1 = torch.ones(10, device=device, dtype=torch.float32) * rank  # 如rank=0→全0，rank=1→全1
    tensor2 = torch.ones(10, device=device, dtype=torch.float32) * rank
    reqs = []  # 用于收集通信请求的列表

    try:
        # 核心：调用低版本_coalescing_manager（参数：group=None, device, reqs）
        with dist.distributed_c10d._coalescing_manager(
            group=None,  # None表示使用默认进程组
            device=device,
            reqs=reqs
        ) as _:
            # 提交2个异步通信操作（必须用async_op=True才能生成req）
            req1 = dist.all_reduce(tensor1, async_op=True)  # 对tensor1做all_reduce
            req2 = dist.all_reduce(tensor2, async_op=True)  # 对tensor2做all_reduce
            reqs.extend([req1, req2])  # 关键：确保请求数量与通信操作数量一致

        # 验证结果（all_reduce后，每个元素应为 0+1=1（world_size=2时））
        expected_sum = float(sum(range(world_size)))  # 转为float，与张量类型匹配
        expected_tensor = torch.full_like(tensor1, expected_sum, dtype=torch.float32)
        
        # 验证tensor1结果
        assert torch.allclose(tensor1, expected_tensor, atol=1e-5), \
            f"Rank {rank}: tensor1结果错误 | 预期: {expected_sum} | 实际: {tensor1[0].item()} | 类型: {tensor1.dtype}"
        # 验证tensor2结果
        assert torch.allclose(tensor2, expected_tensor, atol=1e-5), \
            f"Rank {rank}: tensor2结果错误 | 预期: {expected_sum} | 实际: {tensor2[0].item()} | 类型: {tensor2.dtype}"
        # 验证请求数量（2个通信操作 → 2个请求）
        assert len(reqs) == 2, \
            f"Rank {rank}: 请求数量不匹配 | 预期: 2 | 实际: {len(reqs)}"

        if rank == 0:  # 仅主进程打印成功信息，避免重复
            print("测试1：✅ 正常合并通信 - 成功（结果正确+请求数量匹配）")

    except Exception as e:
        print(f"Rank {rank}: 测试1：❌ 失败 - {str(e)}")


    # ------------------------------
    # 测试2：异常场景 - 请求数量不匹配
    # ------------------------------
    print(f"Rank {rank}: 开始测试2（请求数量不匹配）")
    reqs_err = []
    tensor_err = torch.ones(5, device=device, dtype=torch.float32) * rank

    try:
        with dist.distributed_c10d._coalescing_manager(
            group=None,
            device=device,
            reqs=reqs_err
        ) as _:
            # 提交1个通信操作，但不添加到reqs_err（制造数量不匹配）
            dist.all_reduce(tensor_err, async_op=True)  # 生成req但未收集

        # 如果未抛出错误，说明测试失败
        if rank == 0:
            print("测试2：❌ 失败 - 未捕获请求数量不匹配错误")

    except RuntimeError as e:
        # 验证是否抛出预期的错误（低版本PyTorch的固定错误信息）
        if "Number of requests do not match number of collectives" in str(e):
            if rank == 0:
                print("测试2：✅ 请求数量不匹配 - 成功（捕获预期错误）")
        else:
            print(f"Rank {rank}: 测试2：❌ 失败 - 非预期错误: {str(e)}")
    except Exception as e:
        print(f"Rank {rank}: 测试2：❌ 失败 - 意外错误: {str(e)}")


    # ------------------------------
    # 测试3：边界场景 - 单个通信操作
    # ------------------------------
    print(f"Rank {rank}: 开始测试3（单个通信操作）")
    reqs_single = []
    tensor_single = torch.ones(3, device=device, dtype=torch.float32) * rank

    try:
        with dist.distributed_c10d._coalescing_manager(
            group=None,
            device=device,
            reqs=reqs_single
        ) as _:
            # 提交1个通信操作，且收集请求
            req_single = dist.all_reduce(tensor_single, async_op=True)
            reqs_single.append(req_single)

        # 验证结果
        expected_single = float(sum(range(world_size)))
        assert torch.allclose(tensor_single, torch.full_like(tensor_single, expected_single), atol=1e-5), \
            f"Rank {rank}: tensor_single结果错误 | 预期: {expected_single} | 实际: {tensor_single[0].item()}"
        assert len(reqs_single) == 1, \
            f"Rank {rank}: 单个请求数量错误 | 预期:1 | 实际:{len(reqs_single)}"

        if rank == 0:
            print("测试3：✅ 单个通信操作 - 成功")

    except Exception as e:
        print(f"Rank {rank}: 测试3：❌ 失败 - {str(e)}")


    # 3. 清理分布式环境
    dist.destroy_process_group()
    print(f"Rank {rank}: 测试结束")


def run_tests():
    """启动测试（支持CPU/GPU，自动适配2进程）"""
    # 预处理：检查PyTorch版本（确保是低版本）
    torch_version = torch.__version__.split('.')
    if int(torch_version[0]) > 2 or (int(torch_version[0]) == 2 and int(torch_version[1]) > 0):
        print("⚠️ 警告：当前PyTorch版本可能不是低版本（>2.0.1），_coalescing_manager参数可能不匹配")

    # 创建临时文件（用于进程组初始化，自动清理）
    with tempfile.NamedTemporaryFile(delete=False) as f:
        temp_file = f.name

    try:
        # 启动2个进程测试（分布式通信至少需要2个进程）
        world_size = 2
        spawn(
            fn=test_coalescing_manager,
            args=(world_size, temp_file),
            nprocs=world_size,  # 进程数=world_size
            join=True  # 等待所有进程完成
        )
    finally:
        # 强制删除临时文件（避免残留）
        if os.path.exists(temp_file):
            os.unlink(temp_file)


if __name__ == "__main__":
    # 确保测试在主进程中启动（避免多进程重复执行）
    run_tests()
    