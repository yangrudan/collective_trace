"""
Core functions for collective tracing
"""

import csv
import time
import threading
from functools import wraps
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict

try:
    import torch
    import torch.distributed as dist
except ImportError:
    print("!!! 未找到 PyTorch，已跳过")

from .get_group import get_participating_ranks


function_names = [
    "all_reduce",
    "all_gather",
    "reduce_scatter",
    "broadcast",
    "reduce_scatter_base",
    "all_gather_base",
    "all_gather_into_tensor",
    "reduce_scatter_tensor",
    "all_gather_into_tensor_coalesced",
    "reduce_scatter_tensor_coalesced",
]

# '_all_gather_base'
# '_reduce_scatter_base',
# 'batch_isend_irecv',


@dataclass
class TraceConfig:
    """Configuration for tracing"""

    trace_file: str = None
    verbose: bool = True
    has_cuda: bool = False
    global_rank = 0


@dataclass
class GroupState:
    """State information about the current group"""

    my_rank: int = 0
    my_size: int = 1
    my_idx_in_group: int = 0
    participate_ranks: list = field(default_factory=list)


@dataclass
class RealTimeState:
    """Real-time state information"""

    running_primitives: List[Dict] = field(default_factory=list)
    completed_primitives: List[Dict] = field(default_factory=list)
    display_interval = 1


class CollectiveTracer:
    """
    Trace collective operations for distributed training.
    """

    def __init__(self, trace_file=None, verbose=True):
        """
        Args:
            trace_file: Log file to store trace data, if None, no file will be created
            verbose: Whether to print messages or not
        """
        self.config = TraceConfig(
            trace_file=trace_file, verbose=verbose, has_cuda=torch.cuda.is_available()
        )
        self.trace_data = []
        self.original_functions = {}
        self.hooked_functions = {}

        for func_name in function_names:
            if hasattr(dist, func_name):
                self.hooked_functions[func_name] = getattr(dist, func_name)
            else:
                print(f"!!! torch.distributed 中未找到函数 {func_name}，已跳过")

        if not self.hooked_functions:
            print("!!! WARNING !!! 没有找到任何要追踪的函数")

        self.call_counts = defaultdict(lambda: defaultdict(lambda: {"count": 0}))
        self.group_info = GroupState()

        self.rt_info = RealTimeState()
        self._start_display_thread()  # 启动实时显示线程

    def _start_display_thread(self):
        """启动后台线程，定期显示原语状态"""

        def display_loop():
            while True:
                time.sleep(self.rt_info.display_interval)
                self._display_primitives_status()

        thread = threading.Thread(target=display_loop, daemon=True)
        thread.start()

    def _display_primitives_status(self):
        """打印当前rank的原语状态"""
        if not self.config.verbose:
            return
        print(
            f"\n[Rank {self.config.global_rank}] 原语状态更新 ({time.strftime('%H:%M:%S')}):"
        )
        print(f"  正在运行: {len(self.rt_info.running_primitives)} 个")
        for idx, prim in enumerate(self.rt_info.running_primitives[:5]):  # 显示前5个
            print(
                f"    {idx+1}. {prim['func_name']} (Shape: {prim['tensor_shape']}, "
                f" 已运行: {time.time() - prim['start_time']:.2f}s)"
            )
        if len(self.rt_info.running_primitives) > 5:
            print(f"    ... 还有 {len(self.rt_info.running_primitives) - 5} 个未显示")

        print(f"  已完成: {len(self.rt_info.completed_primitives)} 个")
        for idx, prim in enumerate(
            reversed(self.rt_info.completed_primitives[-5:])
        ):  # 显示最近5个
            print(
                f"    {idx+1}. {prim['func_name']} (Shape: {prim['tensor_shape']}, "
                f" 耗时: {prim['duration']*1000:.2f}ms)"
            )
        if len(self.rt_info.completed_primitives) > 5:
            print(f"    ... 还有 {len(self.rt_info.completed_primitives) - 5} 个未显示")

    def log(self, message):
        """Log a message to console and/or file."""
        if self.config.verbose:
            print(message)
        if self.config.trace_file:
            ranked_filename = f"{self.config.trace_file}-{self.config.global_rank}"
            with open(ranked_filename, "a", encoding="utf-8") as f:
                f.write(message + "\n")

    def create_trace_entry(self, func_name, start_time, duration, tensor_info):
        """Create a trace entry."""
        if tensor_info is None:
            tensor_info = {"shape": "unknown", "dtype": "unknown", "size": 0}

        return {
            "function": func_name,
            "timestamp": start_time,
            "duration": duration,
            "tensor_shape": tensor_info["shape"],
            "tensor_dtype": str(tensor_info["dtype"]),
            "tensor_size": tensor_info["size"],
        }

    def _trace_wrapper(self, func_name, orig_func):
        """Create a wrapper for the original function to trace its execution."""

        class TimedWork:
            """
            A class to wrap the work and time it.
            """

            def __init__(self, work, start_time, func_name, **kwargs):
                self.work = work
                self.start_time = start_time
                self.func_name = func_name
                self.tensor_info = kwargs.get(
                    "tensor_info", {"shape": "unknown", "dtype": "unknown", "size": 0}
                )
                self.tracer = kwargs.get("tracer")

                # 新增：记录正在运行的原语
                self.prim_id = id(self)  # 用对象ID作为唯一标识
                self.tracer.running_primitives.append(
                    {
                        "prim_id": self.prim_id,
                        "func_name": func_name,
                        "tensor_shape": self.tensor_info["shape"],
                        "start_time": start_time,
                    }
                )

            def wait(self):
                """Wait for the wrapped work to complete and record the timing info."""
                result = self.work.wait()

                # if self.tracer.has_cuda:
                #     _cuda_sync()

                end_time = time.perf_counter()
                duration = end_time - self.start_time

                # 新增：将原语从运行中移至已完成
                self.tracer.rt_info.running_primitives = [
                    p
                    for p in self.tracer.rt_info.running_primitives
                    if p["prim_id"] != self.prim_id
                ]
                self.tracer.rt_info.completed_primitives.append(
                    {
                        "prim_id": self.prim_id,
                        "func_name": self.func_name,
                        "tensor_shape": self.tensor_info["shape"],
                        "start_time": self.start_time,
                        "end_time": end_time,
                        "duration": duration,
                    }
                )

                # Create a trace entry
                trace_entry = self.tracer.create_trace_entry(
                    func_name, self.start_time, duration, self.tensor_info
                )
                self.tracer.trace_data.append(trace_entry)

                # Print trace information
                self.tracer.log(
                    f"[TRACE] global rank {self.tracer.config.global_rank} "
                    f"in GROUP_{self.tracer.group_info.my_idx_in_group} "
                    f"- {func_name} - async:1, "
                    f"Size: {self.tensor_info['size']/1024/1024:.2f} MB, "
                    f"Shape: {self.tensor_info['shape']}, "
                    f"Dtype: {self.tensor_info['dtype']}, "
                    f"Duration: {duration*1e3:.3f} ms, "
                    f"GROUP size {self.tracer.group_info.my_size}  = "
                    f"{self.tracer.group_info.participate_ranks}, "
                    f"call count: "
                    f"{self.tracer.call_counts[func_name][self.tensor_info['shape']]['count']}"
                )

                return result

            def is_completed(self):
                """Check whether the wrapped work is completed."""
                return self.work.is_completed()

        @wraps(orig_func)
        def wrapper(*args, **kwargs):

            tensor_info = self._extract_tensor_info(args, kwargs)

            shape = tensor_info["shape"] if tensor_info else "unknown"
            op = func_name
            self.call_counts[op][shape]["count"] += 1

            group = kwargs.get("group") or (args[2] if len(args) > 2 else None)
            (
                self.group_info.my_rank,
                self.group_info.my_size,
                self.group_info.my_idx_in_group,
                self.group_info.participate_ranks,
            ) = get_participating_ranks(group)

            self.config.global_rank = dist.get_rank()

            if self.config.has_cuda:
                _cuda_sync()
            start_time = time.perf_counter()

            is_async = kwargs.get("async_op", False)
            if is_async:
                work = orig_func(*args, **kwargs)

                return TimedWork(
                    work, start_time, func_name, tensor_info=tensor_info, tracer=self
                )

            prim_id = id(args)  # 用参数ID作为临时标识
            self.rt_info.running_primitives.append(
                {
                    "prim_id": prim_id,
                    "func_name": func_name,
                    "tensor_shape": tensor_info["shape"],
                    "start_time": start_time,
                }
            )
            work = orig_func(*args, **kwargs)
            # if self.has_cuda:
            #     _cuda_sync()
            end_time = time.perf_counter()
            duration = end_time - start_time
            # 同步操作：移至已完成
            self.rt_info.running_primitives = [
                p for p in self.rt_info.running_primitives if p["prim_id"] != prim_id
            ]
            self.rt_info.completed_primitives.append(
                {
                    "prim_id": prim_id,
                    "func_name": func_name,
                    "tensor_shape": tensor_info["shape"],
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration": duration,
                }
            )
            trace_entry = self.create_trace_entry(
                func_name, start_time, duration, tensor_info
            )
            self.trace_data.append(trace_entry)
            # Print trace information
            self.log(
                f"[TRACE] global rank {self.config.global_rank} "
                f"in GROUP_{self.group_info.my_idx_in_group} "
                f"- {func_name} - async:0, "
                f"Size: {tensor_info['size']/1024/1024:.2f} MB, "
                f"Shape: {tensor_info['shape']}, "
                f"Dtype: {tensor_info['dtype']}, "
                f"Duration: {duration*1e3:.3f} ms, "
                f"GROUP size {self.group_info.my_size}  = "
                f"{self.group_info.participate_ranks}, "
                f"call count: {self.call_counts[func_name][tensor_info['shape']]['count']}"
            )
            return work

        return wrapper

    def _extract_tensor_info(self, args, kwargs):
        """sub function to extract tensor information from arguments."""
        tensor = None

        # Try to find a tensor in positional arguments
        for arg in args:
            if isinstance(arg, torch.Tensor):
                tensor = arg
                break

        # If not found, try to find a tensor in keyword arguments
        if tensor is None:
            for _, value in kwargs.items():
                if isinstance(value, torch.Tensor):
                    tensor = value
                    break

        # If still not found, check if the first argument is an object with a tensor attribute
        if tensor is None and args:
            first_arg = args[0]
            for attr in dir(first_arg):
                try:
                    value = getattr(first_arg, attr)
                    if isinstance(value, torch.Tensor):
                        tensor = value
                        break
                except (AttributeError, TypeError):
                    continue

        if tensor is None:
            return {"shape": "unknown", "dtype": "unknown", "size": 0}

        return {
            "shape": tuple(tensor.shape),
            "dtype": tensor.dtype,
            "size": tensor.element_size() * tensor.numel(),
        }

    def hook_batch_isend_irecv(self):
        """Hook the `batch_isend_irecv` method to log batch operations."""
        if not hasattr(dist, "batch_isend_irecv"):
            print("!!! WARNING !!! 没有找到 batch_isend_irecv 函数")
            return

        original_batch_isend_irecv = dist.batch_isend_irecv

        def wrapped_batch_isend_irecv(ops_list):
            stats = {
                "send_total": 0,
                "recv_total": 0,
                "send_targets": [],
                "send_shapes": [],
                "recv_sources": [],
                "recv_shapes": [],
            }

            for op in ops_list:
                if isinstance(op, dist.P2POp):
                    tensor = op.tensor
                    data_size = tensor.numel() * tensor.element_size()
                    shape = tuple(tensor.shape)

                    if op.op == dist.isend:
                        stats["send_total"] += data_size
                        stats["send_targets"].append(op.peer)
                        stats["send_shapes"].append(shape)
                    elif op.op == dist.irecv:
                        stats["recv_total"] += data_size
                        stats["recv_sources"].append(op.peer)
                        stats["recv_shapes"].append(shape)

            send_mb = stats["send_total"] / (1024 * 1024)
            recv_mb = stats["recv_total"] / (1024 * 1024)

            send_targets_str = (
                ", ".join(map(str, stats["send_targets"]))
                if stats["send_targets"]
                else "Null"
            )
            recv_sources_str = (
                ", ".join(map(str, stats["recv_sources"]))
                if stats["recv_sources"]
                else "Null"
            )
            send_shapes_str = (
                ", ".join(map(str, stats["send_shapes"]))
                if stats["send_shapes"]
                else "Null"
            )
            recv_shapes_str = (
                ", ".join(map(str, stats["recv_shapes"]))
                if stats["recv_shapes"]
                else "Null"
            )

            self.log(
                f"[BATCH] global rank {dist.get_rank()} - "
                f"send: {stats['send_total']} bytes ({send_mb:.2f} MB), "
                f"shape: [{send_shapes_str}] 目标: [{send_targets_str}], "
                f"recv: {stats['recv_total']} bytes ({recv_mb:.2f} MB), "
                f"shape: [{recv_shapes_str}[ 来源: [{recv_sources_str}], "
                f"ops count: {len(ops_list)}"
            )

            return original_batch_isend_irecv(ops_list)

        dist.batch_isend_irecv = wrapped_batch_isend_irecv

    def _trace_barrier(self, original_barrier):

        @wraps(original_barrier)
        def barrier_traced(*args, **kwargs):
            start_time = time.perf_counter()
            result = original_barrier(*args, **kwargs)
            end_time = time.perf_counter()
            duration = end_time - start_time

            trace_entry = self.create_trace_entry("barrier", start_time, duration, None)
            self.trace_data.append(trace_entry)

            self.log(
                f"[BARRIER] global rank {dist.get_rank()} - barrier - async:0, "
                f"Duration: {duration*1e3:.3f} ms"
            )

            return result

        return barrier_traced

    def apply_hooks(self):
        """Apply all tracing hooks on distributed functions"""
        for func_name, orig_func in self.hooked_functions.items():
            if hasattr(dist, func_name):
                self.original_functions[func_name] = getattr(dist, func_name)
                setattr(dist, func_name, self._trace_wrapper(func_name, orig_func))
                print(f"Applyed hook to function: {func_name}")

        if hasattr(dist, "batch_isend_irecv"):
            self.hook_batch_isend_irecv()
            print("Applyed hook to batch_isend_irecv")

        if hasattr(dist, "barrier"):
            original_barrier = getattr(dist, "barrier")
            setattr(dist, "barrier", self._trace_barrier(original_barrier))
            print("Applyed hook to barrier")

    def remove_hooks(self):
        """Remove all tracing hooks from distributed functions"""
        for func_name, orig_func in self.original_functions.items():
            if hasattr(dist, func_name):
                setattr(dist, func_name, orig_func)
                self.log(f"Removed hook from function: {func_name}")

    def get_trace_data(self):
        """Return the collected trace data."""
        return self.trace_data

    def get_all_call_counts(self):
        """Get a copy of call counts dictionary."""
        return self.call_counts.copy()

    def export_to_csv(self, filename):
        """Export trace data to CSV format"""
        if not self.trace_data:
            self.log("No trace data to export.")
            return

        with open(filename, "w", newline="", encoding="utf-8") as csvfile:
            fieldnames = self.trace_data[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in self.trace_data:
                writer.writerow(row)

        self.log(f"Exported trace data to {filename}")


def _cuda_sync():
    """Synchronize CUDA devices."""
    torch.cuda.synchronize()
