import torch
import torch.distributed as dist
import os
import time
from torch.multiprocessing import spawn

from collective_trace.collective_trace import trace_all_collectives

tracer = trace_all_collectives(trace_file="collective_trace.log", verbose=True)

def run(rank, size):
    """执行屏障测试的函数"""
    # 每个进程初始化不同的随机种子
    torch.manual_seed(rank)
    
    # 打印当前进程开始工作的信息
    print(f"进程 {rank} 开始执行任务...", flush=True)
    
    # 让不同进程执行不同时长的"工作"，模拟实际计算差异
    work_duration = 0.5 + rank * 0.2  # 进程0: 0.5s, 进程1: 0.7s, 等
    time.sleep(work_duration)
    print(f"进程 {rank} 完成初步工作，准备进入屏障...", flush=True)
    
    # 记录进入屏障前的时间
    start_time = time.time()
    
    # 执行屏障同步
    dist.barrier()
    
    # 记录离开屏障的时间
    end_time = time.time()
    
    # 计算每个进程在屏障处等待的时间
    wait_time = end_time - start_time
    print(f"进程 {rank} 已通过屏障，在屏障处等待了 {wait_time:.4f} 秒", flush=True)
    
    # 验证所有进程通过屏障的时间差应在很小范围内（理论上应同时通过）
    # 收集所有进程的结束时间
    all_end_times = [torch.tensor(0.0) for _ in range(size)]
    dist.all_gather(all_end_times, torch.tensor(end_time))
    
    # 计算最大时间差
    max_time_diff = max(all_end_times) - min(all_end_times)
    if rank == 0:  # 只在主进程打印结果
        print(f"\n所有进程通过屏障的最大时间差: {max_time_diff.item():.6f} 秒")
        if max_time_diff < 0.01:  # 10ms 阈值，可根据系统调整
            print("测试通过: 所有进程在屏障处成功同步")
        else:
            print("测试失败: 进程同步存在明显延迟")

def init_process(rank, size, fn, backend='gloo'):
    """初始化进程组并运行测试函数"""
    # 设置环境变量
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # 初始化进程组
    dist.init_process_group(backend, rank=rank, world_size=size)
    
    # 运行测试函数
    fn(rank, size)
    
    # 销毁进程组
    dist.destroy_process_group()

# python test_barrier.py
if __name__ == "__main__":
    # 测试的进程数量
    processes = 4
    
    # 使用spawn启动多个进程
    spawn(init_process, args=(processes, run), nprocs=processes, join=True)
