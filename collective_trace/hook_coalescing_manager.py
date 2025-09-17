"""
hook _coalesing_manager
"""

import uuid
import time
from contextlib import contextmanager
from typing import Optional

from .shared_coealescing_state import coalescing_state

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


def hook_coalescing_manager(tracer):
    """hook _coalescing_manager"""
    if IS_OLD_VERSION:
        return  # 低版本暂不支持hook

    # pylint: disable=protected-access
    origin_coalescing_manager = dist._coalescing_manager

    if IS_OLD_VERSION:
        pass
        # # PyTorch (<=2.0.1)，_coalescing_manager参数为(group, device, reqs)
        # @contextmanager
        # def timed_coalescing_manager(
        #     group: Optional[dist.ProcessGroup] = None,
        #     device: Optional[torch.device] = None,
        #     reqs: Optional[List] = None,
        # ):
        #     total_start = time.perf_counter()
        #     timing_details = {}
        #     reqs_list = reqs if reqs is not None else []

        #     with origin_coalescing_manager(
        #         group=group, device=device, reqs=reqs_list
        #     ) as _:
        #         try:
        #             yield
        #         finally:
        #             context_end = time.perf_counter()
        #             context_duration = context_end - total_start
        #             timing_details["context_duration"] = context_duration

        #     timing_details["total_duration"] = context_duration
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
            tracer.update_group_info(group)

            total_start = time.perf_counter()
            work_items = []

            class TimedCoalescingWrapper:
                """
                CoalescingManager with timing"""

                def __init__(self, inner_obj):
                    self.inner_obj = inner_obj

                def append(self, work):
                    """
                    Append a work item to the list of work items"""
                    work_items.append(work)
                    if hasattr(self.inner_obj, "append"):
                        self.inner_obj.append(work)

                def wait(self):
                    """
                    Wait for all work items to complete"""
                    if hasattr(self.inner_obj, "wait"):
                        result = self.inner_obj.wait()
                    else:
                        result = [work.wait() for work in work_items]
                    return result

                def __getattr__(self, name):
                    return getattr(self.inner_obj, name)

            cm_id = uuid.uuid4().hex
            coalescing_state.active_cm_id = cm_id
            coalescing_state.counter[cm_id] = 0

            with origin_coalescing_manager(
                group=group, device=device, async_ops=async_ops
            ) as original_cm:
                timed_cm = TimedCoalescingWrapper(original_cm)

                try:
                    yield timed_cm
                finally:
                    end_time = time.perf_counter()
                    duration_ms = (end_time - total_start) * 1000

                    # 使用tracer.log()输出日志
                    tracer.log(
                        f"[COALESCE] global rank {tracer.config.global_rank} "
                        f"in GROUP_{tracer.group_info.my_idx_in_group} "
                        f"- coalesced_ops - async:{1 if async_ops else 0}, "
                        f"conuters: {coalescing_state.counter.get(cm_id, 0)}, "
                        f"name: {coalescing_state.names.get(cm_id, 'unknown')}, "
                        f"sizes: {coalescing_state.sizes.get(cm_id, 0)}, "
                        f"Duration: {duration_ms:.3f} ms, "
                        f"GROUP size {tracer.group_info.my_size}  = "
                        f"{tracer.group_info.participate_ranks}, "
                    )

                    # cleanup
                    del coalescing_state.counter[cm_id]
                    if coalescing_state.active_cm_id == cm_id:
                        coalescing_state.active_cm_id = None

    # pylint: disable=protected-access
    dist._coalescing_manager = timed_coalescing_manager
