"""
Core hook modlue
"""

import csv
from functools import wraps
from collections import defaultdict
import os
import signal

try:
    import torch
    import torch.distributed as dist
except ImportError:
    print("!!! PyTorch not found, skipped")
from .async_logger import RankedAsyncLogger
from .trace_config import TraceConfig, GroupState
from .timeout_manager import TimeoutManager
from .trace_wrapper import (
    create_function_wrapper,
    create_barrier_wrapper,
    create_batch_isend_irecv_wrapper,
)
from .get_group import get_participating_ranks

from .hook_coalescing_manager import hook_coalescing_manager

function_names = [
    "all_reduce",
    "all_gather",
    "reduce_scatter",
    "broadcast",
    "all_gather_into_tensor",
    "reduce_scatter_tensor",
]


class CollectiveTracer:
    """Trace collective operations for distributed training with timeout detection."""

    def __init__(self, trace_file=None, verbose=True):
        self.config = TraceConfig(
            trace_file=trace_file,
            verbose=verbose,
            has_cuda=torch.cuda.is_available(),
            global_rank=0,
            async_logger=None,
        )
        self.trace_data = []
        self.original_functions = {}
        self.hooked_functions = {}
        self.call_counts = defaultdict(lambda: defaultdict(lambda: {"count": 0}))
        self.group_info = GroupState()

        # Init timeout manager
        self.timeout_manager = TimeoutManager(self.log, self._timeout_callback)

        # Init hooked functions
        for func_name in function_names:
            if hasattr(dist, func_name):
                self.hooked_functions[func_name] = getattr(dist, func_name)
            else:
                print(
                    f"!!! Function {func_name} not found in torch.distributed, skipped"
                )

        if not self.hooked_functions:
            print("!!! WARNING !!! No functions to trace found")

    def update_group_info(self, group):
        """Update group info from the given group."""
        (
            self.group_info.my_rank,
            self.group_info.my_size,
            self.group_info.my_idx_in_group,
            self.group_info.participate_ranks,
        ) = get_participating_ranks(group)

    def _timeout_callback(self, op_id, func_name, is_async, timed_out_type):
        """Timeout callback: log timeout events"""
        async_str = "async" if is_async else "sync"
        if timed_out_type == "unfinished":
            msg = (
                f"[TIMEOUT] {async_str} Operation {func_name} (ID: {op_id}) "
                f"timed out without completion! "
                f"Exceeded threshold {self.timeout_manager.get_timeout_threshold()}s"
            )
        else:
            msg = (
                f"[TIMEOUT] {async_str} Operation {func_name} (ID: {op_id}) "
                f"completed but timed out! "
                f"Exceeded threshold {self.timeout_manager.get_timeout_threshold()}s"
            )

        self.log(msg)
        os.kill(os.getpid(), signal.SIGUSR1)
        self.timeout_manager.unregister_operation(op_id)

    def log(self, message):
        """Log a message to console and/or file."""
        if self.config.verbose:
            print(message)
        if self.config.async_logger:
            self.config.async_logger.log(message)

    def with_global_rank(self, func):
        """Decorator that updates global rank before calling the wrapped function"""

        @wraps(func)
        def wrapper(*args, **kwargs):
            if dist.is_available() and dist.is_initialized():
                self.config.global_rank = dist.get_rank()

                if self.config.async_logger is None and self.config.trace_file:
                    self.config.async_logger = RankedAsyncLogger(
                        base_log_file=self.config.trace_file,
                        rank=self.config.global_rank,
                    )
            return func(*args, **kwargs)

        return wrapper

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

    def apply_hooks(self):
        """Apply all tracing hooks on distributed functions"""
        for func_name, orig_func in self.hooked_functions.items():
            if hasattr(dist, func_name):
                self.original_functions[func_name] = getattr(dist, func_name)
                setattr(
                    dist, func_name, create_function_wrapper(func_name, orig_func, self)
                )
                print(f"Applied hook to function: {func_name}")

        if hasattr(dist, "batch_isend_irecv"):
            self.original_functions["batch_isend_irecv"] = dist.batch_isend_irecv
            dist.batch_isend_irecv = create_batch_isend_irecv_wrapper(
                dist.batch_isend_irecv, self
            )
            print("Applied hook to batch_isend_irecv")

        if hasattr(dist, "barrier"):
            self.original_functions["barrier"] = dist.barrier
            dist.barrier = create_barrier_wrapper(dist.barrier, self)
            print("Applied hook to barrier")

        hook_coalescing_manager(self)
        print("Applied hook to _coalescing_manager")

    def remove_hooks(self):
        """Remove all tracing hooks from distributed functions"""
        for func_name, orig_func in self.original_functions.items():
            if hasattr(dist, func_name):
                setattr(dist, func_name, orig_func)
                self.log(f"Removed hook for {func_name}")

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

    def __del__(self):
        """Clean up resources and stop timer"""
        self.timeout_manager.stop()
        if hasattr(self, "config") and self.config.async_logger:
            self.config.async_logger.close()
