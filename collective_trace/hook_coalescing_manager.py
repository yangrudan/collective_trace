"""
hook _coalesing_manager

1. 注意torch版本, 低版本无_CoalesingManager
2. 注意hook时机, 在torch.distributed.distributed_c10d._coalescing_manager被赋值之前
3. 捕获合并的原语名称, 便于后续统计
"""
import time
from contextlib import contextmanager
from typing import Optional
import torch
import torch.distributed as dist
from torch.distributed.distributed_c10d import _coalescing_manager


# 检查是否有_CoalescingManager类
try:
    from torch.distributed.distributed_c10d import _CoalescingManager
    HAS_COALESCING_MANAGER_CLASS = True
except ImportError:
    HAS_COALESCING_MANAGER_CLASS = False


def print_timing(details):
    """默认的计时打印函数"""
    print("合并通信计时结果:")
    for name, duration in details.items():
        print(f"  {name}: {duration:.6f}秒")

def hook_coalescing_manager():
    global _coalescing_manager
    origin_coalescing_manager = _coalescing_manager

    @contextmanager
    def timed_coalescing_manager(
        group: Optional[dist.ProcessGroup] = None,
        device: Optional[torch.device] = None,
        async_ops: bool = False,
    ):
        """
        灵活的合并通信计时管理器，支持cm.wait()在with块内外调用
        
        异步模式下：
        - 若在with块内调用cm.wait()：计时包含等待时间
        - 若在with块外调用cm.wait()：仍能捕获等待时间并在最终输出
        
        Args:
            group: 进程组
            device: 设备
            async_ops: 是否异步执行
        """
        # 记录整个操作的开始时间
        total_start = time.perf_counter()
        timing_details = {}
        wait_called = False
        wait_duration = 0.0

        work_items = []

        # 低版本PyTorch兼容：定义带计时功能的管理器
        class TimedCoalescingWrapper:
            def __init__(self, inner_obj):
                self.inner_obj = inner_obj
                self._wait_start = None
                self._wait_duration = 0.0

            def append(self, work):
                work_items.append(work)
                if hasattr(self.inner_obj, 'append'):
                    self.inner_obj.append(work)

            def wait(self):
                nonlocal wait_called, wait_duration
                # 记录wait()开始时间
                self._wait_start = time.perf_counter()
                
                # 执行原始wait()
                if HAS_COALESCING_MANAGER_CLASS and hasattr(self.inner_obj, 'wait'):
                    result = self.inner_obj.wait()
                else:
                    # 低版本兼容：直接等待所有工作完成
                    result = [work.wait() for work in work_items]
                
                # 记录wait()耗时
                self._wait_duration = time.perf_counter() - self._wait_start
                wait_duration = self._wait_duration
                wait_called = True
                return result

            # 转发其他属性和方法
            def __getattr__(self, name):
                return getattr(self.inner_obj, name)

        # 创建原始上下文管理器
        with origin_coalescing_manager(group=group, device=device, async_ops=async_ops) as original_cm:
            # 创建包装后的管理器
            timed_cm = TimedCoalescingWrapper(original_cm)

            try:
                # 向用户代码返回带计时功能的管理器
                yield timed_cm
            finally:
                # 计算上下文块的执行时间（不包含with块外的wait）
                context_end = time.perf_counter()
                context_duration = context_end - total_start
                timing_details["context_duration"] = context_duration

        # with块退出后，处理计时（此时可能已调用或未调用wait）
        if async_ops:
            # 异步模式处理
            if not wait_called:
                # 用户可能在with块外调用wait，需要额外处理
                # 这里使用装饰器模式包装wait方法，确保最终能捕获时间
                original_wait = timed_cm.wait

                def wrapped_wait():
                    nonlocal wait_called, wait_duration
                    if not wait_called:
                        start = time.perf_counter()
                        result = original_wait()
                        wait_duration = time.perf_counter() - start
                        wait_called = True
                        # 重新计算总耗时
                        timing_details["wait_duration"] = wait_duration
                        timing_details["total_duration"] = total_start + context_duration + wait_duration - total_start
                        # 触发日志回调
                        print_timing(timing_details)
                        return result
                    return original_wait()
                
                timed_cm.wait = wrapped_wait
                timing_details["wait_duration"] = wait_duration
            else:
                # wait在with块内已调用
                timing_details["wait_duration"] = wait_duration
                timing_details["total_duration"] = context_duration + wait_duration
        else:
            # 同步模式处理
            timing_details["total_duration"] = context_duration
    
        # 如果wait已在with块内调用，直接输出结果
        if async_ops and wait_called:
            print_timing(timing_details)
    
    _coalescing_manager = timed_coalescing_manager

# ------------------------------
# 使用示例
# ------------------------------
if __name__ == "__main__":
    # 初始化分布式环境（示例配置）
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    
    hook_coalescing_manager()
    
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
