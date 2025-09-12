"""
hook _coalesing_manager

1. 注意torch版本, 低版本无_CoalesingManager,  _coalescing_manager参数为(group, device, reqs)!!
2. 注意hook时机, 在torch.distributed.distributed_c10d._coalescing_manager被赋值之前
3. TODO 捕获合并的原语名称, 便于后续统计
4. TODO from 方式可能无法替换成功
"""

import time
from contextlib import contextmanager
from typing import Optional, List

try:
    import torch
    import torch.distributed as dist
    from torch.distributed.distributed_c10d import _coalescing_manager
except ImportError:
    pass


TORCH_VERSION = torch.__version__.split(".")
TORCH_MAJOR = int(TORCH_VERSION[0])
TORCH_MINOR = int(TORCH_VERSION[1])

IS_OLD_VERSION = (TORCH_MAJOR < 2) or (TORCH_MAJOR == 2 and TORCH_MINOR <= 0)


def print_timing(details):
    """tmp print timing results"""
    print("======== timing details =======:")
    for name, duration in details.items():
        print(f"  {name}: {duration:.6f}秒")


def hook_coalescing_manager():
    """hook _coalescing_manager"""
    global _coalescing_manager
    origin_coalescing_manager = _coalescing_manager

    if IS_OLD_VERSION:
        # PyTorch (<=2.0.1)，_coalescing_manager参数为(group, device, reqs)
        @contextmanager
        def timed_coalescing_manager(
            group: Optional[dist.ProcessGroup] = None,
            device: Optional[torch.device] = None,
            reqs: Optional[List] = None,
        ):
            # 记录开始时间
            total_start = time.perf_counter()
            timing_details = {}
            reqs_list = reqs if reqs is not None else []


            with origin_coalescing_manager(
                group=group, device=device, reqs=reqs_list
            ) as _:
                try:
                    yield
                finally:
                    context_end = time.perf_counter()
                    context_duration = context_end - total_start
                    timing_details["context_duration"] = context_duration

            # 处理计时结果
            timing_details["total_duration"] = context_duration
            print_timing(timing_details)
    else:
        # 适配高版本PyTorch (>=2.0.2)，_coalescing_manager参数为(group, device, async_ops)

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

            class TimedCoalescingWrapper:
                """
                CoalescingManager with timing"""

                def __init__(self, inner_obj):
                    self.inner_obj = inner_obj
                    self._wait_start = None
                    self._wait_duration = 0.0

                def append(self, work):
                    """
                    Append a work item to the list of work items"""
                    work_items.append(work)
                    if hasattr(self.inner_obj, "append"):
                        self.inner_obj.append(work)

                def wait(self):
                    """
                    Wait for all work items to complete"""
                    nonlocal wait_called, wait_duration
                    self._wait_start = time.perf_counter()

                    if hasattr(self.inner_obj, "wait"):
                        result = self.inner_obj.wait()
                    else:
                        result = [work.wait() for work in work_items]

                    self._wait_duration = time.perf_counter() - self._wait_start
                    wait_duration = self._wait_duration
                    wait_called = True
                    print("wait duration: ", self._wait_duration)
                    return result

                def __getattr__(self, name):
                    return getattr(self.inner_obj, name)

            with origin_coalescing_manager(
                group=group, device=device, async_ops=async_ops
            ) as original_cm:
                timed_cm = TimedCoalescingWrapper(original_cm)

                try:
                    yield timed_cm
                finally:
                    context_end = time.perf_counter()
                    context_duration = context_end - total_start
                    timing_details["context_duration"] = context_duration

                    if not async_ops:
                        print_timing(timing_details)

            # # with块退出后，处理计时（此时可能已调用或未调用wait）
            # if async_ops:
            #     # 异步模式处理
            #     if not wait_called:
            #         # 用户可能在with块外调用wait，需要额外处理
            #         # 这里使用装饰器模式包装wait方法，确保最终能捕获时间
            #         original_wait = timed_cm.wait

            #         def wrapped_wait():
            #             nonlocal wait_called, wait_duration
            #             if not wait_called:
            #                 start = time.perf_counter()
            #                 result = original_wait()
            #                 wait_duration = time.perf_counter() - start
            #                 wait_called = True
            #                 # 重新计算总耗时
            #                 timing_details["wait_duration"] = wait_duration
            #                 timing_details["total_duration"] = (
            #                     total_start
            #                     + context_duration
            #                     + wait_duration
            #                     - total_start
            #                 )
            #                 # 触发日志回调
            #                 print_timing(timing_details)
            #                 return result
            #             return original_wait()

            #         timed_cm.wait = wrapped_wait
            #         timing_details["wait_duration"] = wait_duration
            #     else:
            #         # wait在with块内已调用
            #         timing_details["wait_duration"] = wait_duration
            #         timing_details["total_duration"] = context_duration + wait_duration
            # else:
            #     # 同步模式处理
            #     timing_details["total_duration"] = context_duration

            # # 如果wait已在with块内调用，直接输出结果
            # if async_ops and wait_called:
            #     print_timing(timing_details)

    _coalescing_manager = timed_coalescing_manager


# ------------------------------
# 使用示例
# ------------------------------
# if __name__ == "__main__":
#     # 初始化分布式环境（示例配置）
#     dist.init_process_group(backend="nccl")
#     rank = dist.get_rank()
#     device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")

#     hook_coalescing_manager()


#     if IS_OLD_VERSION:
#         print("\n=== 2.0.1 同步版本 ===")
#         tensor1 = torch.ones(10, device=device, dtype=torch.float32) * rank
#         tensor2 = torch.ones(10, device=device, dtype=torch.float32) * rank
#         reqs = []  # 用于收集通信请求的列表

#         hook_coalescing_manager()

#         with _coalescing_manager(
#                 group=None,  # None表示使用默认进程组
#                 device=device,
#                 reqs=reqs
#             ) as _:
#                 # 提交2个异步通信操作（必须用async_op=True才能生成req）
#             req1 = dist.all_reduce(tensor1, async_op=True)  # 对tensor1做all_reduce
#             req2 = dist.all_reduce(tensor2, async_op=True)  # 对tensor2做all_reduce
#             reqs.extend([req1, req2])  # 关键：确保请求数量与通信操作数量一致

#     else:
#         # 示例1：异步模式 - wait在with块内调用
#         print("\n=== 异步模式（wait在with块内） ===")
#         tensors = [torch.randn(1024, device=device) for _ in range(4)]
#         with _coalescing_manager(device=device, async_ops=True) as cm:
#             for tensor in tensors:
#                 dist.all_reduce(tensor)
#             cm.wait()  # 在with块内调用wait

#         # 示例2：异步模式 - wait在with块外调用
#         print("\n=== 异步模式（wait在with块外） ===")
#         tensors = [torch.randn(1024, device=device) for _ in range(4)]
#         with _coalescing_manager(device=device, async_ops=True) as cm:
#             for tensor in tensors:
#                 dist.all_reduce(tensor)
#         # 在with块外调用wait
#         cm.wait()
