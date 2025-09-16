"""
hook _coalesing_manager

1. 注意torch版本, 低版本无_CoalesingManager,  _coalescing_manager参数为(group, device, reqs)!!
2. 注意hook时机, 在torch.distributed.distributed_c10d._coalescing_manager被赋值之前
3. TODO 捕获合并的原语名称, 便于后续统计
4. from 方式可能无法替换成功
"""
import time
from contextlib import contextmanager
from typing import Optional, List

try:
    import torch
    import torch.distributed as dist
except ImportError:
    pass


TORCH_VERSION = torch.__version__.split(".")
TORCH_MAJOR = int(TORCH_VERSION[0])
TORCH_MINOR = int(TORCH_VERSION[1])

IS_OLD_VERSION = (TORCH_MAJOR < 2) or (TORCH_MAJOR == 2 and TORCH_MINOR <= 0)
print(f"IS_OLD_VERSION: {IS_OLD_VERSION}, Torch version: {torch.__version__}")


def print_timing(details):
    """tmp print timing results"""
    print("======== timing details =======:")
    for name, duration in details.items():
        print(f"  {name}: {duration:.6f}秒")


# pylint: disable=protected-access
def hook_coalescing_manager():
    """hook _coalescing_manager"""
    if IS_OLD_VERSION:
        origin_coalescing_manager = dist.distributed_c10d._coalescing_manager
    else:
        origin_coalescing_manager = dist._coalescing_manager

    if IS_OLD_VERSION:
        # PyTorch (<=2.0.1)，_coalescing_manager参数为(group, device, reqs)
        @contextmanager
        def timed_coalescing_manager(
            group: Optional[dist.ProcessGroup] = None,
            device: Optional[torch.device] = None,
            reqs: Optional[List] = None,
        ):
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

            timing_details["total_duration"] = context_duration
            print_timing(timing_details)
    else:
        # PyTorch (>=2.0.2)，_coalescing_manager(group, device, async_ops)

        @contextmanager
        def timed_coalescing_manager(
            group: Optional[dist.ProcessGroup] = None,
            device: Optional[torch.device] = None,
            async_ops: bool = False,
        ):
            """
            Context manager for timing _coalescing_manager

            Args:
                group: ProcessGroup
                device: device
                async_ops: whether to use async ops
            """
            print("======!!!!hooked _coalescing_manager enter \n")
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

    # pylint: disable=protected-access
    dist._coalescing_manager = timed_coalescing_manager
