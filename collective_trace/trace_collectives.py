"""
Core functions for collective tracing
"""

import csv
import time
from functools import wraps
from collections import defaultdict
from dataclasses import dataclass
import os
import signal
import uuid

try:
    import torch
    import torch.distributed as dist
except ImportError:
    print("!!! PyTorch not found, skipped")

from .async_logger import RankedAsyncLogger
from .get_group import get_participating_ranks
from .timeout_daemon import OperationTimer


function_names = [
    "all_reduce",
    "all_gather",
    "reduce_scatter",
    "broadcast",
    "all_gather_into_tensor",
    "reduce_scatter_tensor",
]

# "_all_gather_base",
# "_reduce_scatter_base",


@dataclass
class TraceConfig:
    """Configuration for tracing"""

    trace_file: str = None
    verbose: bool = True
    has_cuda: bool = False
    global_rank: int = 0
    async_logger: RankedAsyncLogger = None # Lazy initialization(wait for global ranks verified)


@dataclass
class GroupState:
    """State information about the current group"""

    my_rank: int = 0
    my_size: int = 1
    my_idx_in_group: int = 0
    participate_ranks: list = None

    def __post_init__(self):
        if self.participate_ranks is None:
            self.participate_ranks = []


class TimeOutDaemon(OperationTimer):
    """
    Daemon thread class that monitors and handles timeouts for
    collective operations.

    This class retrieves the timeout threshold from environment
    variables or parameters,creates and manages a timer, and
    invokes the specified callback function when a timeout occurs.
    """

    # Define default timeout threshold as integer
    DEFAULT_TIMEOUT_THRESHOLD = 480  # 8 minutes (integer value)

    def __init__(self, callback):
        """Initialize a TimeOutDaemon instance"""
        timeout_threshold = self._resolve_timeout_threshold()
        super().__init__(timeout_threshold, callback)
        self.start()  # Start the monitoring thread in the constructor

    def _resolve_timeout_threshold(self):
        """Resolve the timeout threshold with priority: environment variable > default value"""
        env_timeout = os.environ.get("COLLECTIVE_TIMEOUT")
        if env_timeout is not None:
            try:
                return int(env_timeout)
            except ValueError:
                pass

        return self.DEFAULT_TIMEOUT_THRESHOLD

    def get_timeout_threshold(self):
        """Get the currently set timeout threshold in seconds"""
        return self.timeout_threshold


class CollectiveTracer:
    """
    Trace collective operations for distributed training with timeout detection.
    """

    def __init__(self, trace_file=None, verbose=True):
        """
        Args:
            trace_file: Log file to store trace data, if None, no file will be created
            verbose: Whether to print messages or not
            timeout_threshold: Timeout threshold in seconds. If None, read from \
                            COLLECTIVE_TIMEOUT environment variable, default 50s
        """
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

        for func_name in function_names:
            if hasattr(dist, func_name):
                self.hooked_functions[func_name] = getattr(dist, func_name)
            else:
                print(
                    f"!!! Function {func_name} not found in torch.distributed, skipped"
                )

        if not self.hooked_functions:
            print("!!! WARNING !!! No functions to trace found")

        self.call_counts = defaultdict(lambda: defaultdict(lambda: {"count": 0}))
        self.group_info = GroupState()

        self.timeout_daemon = TimeOutDaemon(self._timeout_callback)

    def _timeout_callback(self, op_id, func_name, is_async, timed_out_type):
        """Timeout callback: log timeout events"""
        async_str = "async" if is_async else "sync"
        if timed_out_type == "unfinished":
            msg = f"[TIMEOUT] {async_str} Operation {func_name} (ID: {op_id}) \
                  timed out without completion! \
                  Exceeded threshold {self.timeout_daemon.get_timeout_threshold()}s"
        else:
            msg = f"[TIMEOUT] {async_str} Operation {func_name} (ID: {op_id}) \
                  completed but timed out! \
                  Exceeded threshold {self.timeout_daemon.get_timeout_threshold()}s"

        self.log(msg)
        os.kill(os.getpid(), signal.SIGUSR1)
        self.timeout_daemon.unregister_operation(op_id)

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
                        rank=self.config.global_rank
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

    def _trace_wrapper(self, func_name, orig_func):
        """Create a wrapper for the original function to trace its execution and detect timeouts."""

        class TimedWork:
            """
            A class to wrap the work and time it, with timeout detection.
            """

            def __init__(self, work, op_id, start_time, func_name, **kwargs):
                self.work = work
                self.op_id = op_id
                self.start_time = start_time
                self.func_name = func_name
                self.tensor_info = kwargs.get(
                    "tensor_info", {"shape": "unknown", "dtype": "unknown", "size": 0}
                )
                self.tracer = kwargs.get("tracer")

            def wait(self):
                """ Wait for the wrapped work to complete and record the timing info """
                self.tracer.timeout_daemon.register_operation(self.op_id, self.func_name, True)

                result = self.work.wait()

                # Mark operation as completed
                self.tracer.timeout_daemon.mark_completed(self.op_id)

                # if self.tracer.has_cuda:
                #     _cuda_sync()

                end_time = time.perf_counter()
                duration = end_time - self.start_time

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

        @self.with_global_rank
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

            if self.config.has_cuda:
                _cuda_sync()
            start_time = time.perf_counter()

            is_async = kwargs.get("async_op", False)
            # Generate unique operation ID
            op_id = uuid.uuid4()

            if is_async:
                work = orig_func(*args, **kwargs)
                return TimedWork(
                    work,
                    op_id,
                    start_time,
                    func_name,
                    tensor_info=tensor_info,
                    tracer=self,
                )

            # Register operation with timer
            self.timeout_daemon.register_operation(op_id, func_name, is_async)

            # Synchronous operation
            result = orig_func(*args, **kwargs)

            # Check if already timed out
            if self.timeout_daemon.is_timed_out(op_id):
                self.log(
                    f"[ERROR] Synchronous operation {func_name} (ID: {op_id}) has timed out"
                )

            # Mark operation as completed
            self.timeout_daemon.mark_completed(op_id)

            end_time = time.perf_counter()
            duration = end_time - start_time

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

            return result

        return wrapper

    def _extract_tensor_info(self, args, kwargs):
        """Sub function to extract tensor information from arguments."""
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

        # If still not found, check if the first argument is an object with a
        # tensor attribute
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
        """Hook the `batch_isend_irecv` method to log batch operations with timeout detection."""
        if not hasattr(dist, "batch_isend_irecv"):
            print("!!! WARNING !!! batch_isend_irecv function not found")
            return

        original_batch_isend_irecv = dist.batch_isend_irecv

        @self.with_global_rank
        def wrapped_batch_isend_irecv(ops_list):
            # Generate unique operation ID
            op_id = uuid.uuid4()
            # Register operation with timer
            self.timeout_daemon.register_operation(
                op_id, "batch_isend_irecv", is_async=True
            )

            # Extract statistics
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

            # Execute original operation
            result = original_batch_isend_irecv(ops_list)

            # Mark operation as completed
            self.timeout_daemon.mark_completed(op_id)

            # Format output information
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
                f"[BATCH] global rank {self.config.global_rank} - "
                f"send: {stats['send_total']} bytes ({send_mb:.2f} MB), "
                f"shape: [{send_shapes_str}] 目标: [{send_targets_str}], "
                f"recv: {stats['recv_total']} bytes ({recv_mb:.2f} MB), "
                f"shape: [{recv_shapes_str}[ 来源: [{recv_sources_str}], "
                f"ops count: {len(ops_list)}"
            )

            return result

        dist.batch_isend_irecv = wrapped_batch_isend_irecv

    def _trace_barrier(self, original_barrier):
        @self.with_global_rank
        @wraps(original_barrier)
        def barrier_traced(*args, **kwargs):
            # Generate unique operation ID
            op_id = uuid.uuid4()
            # Register operation with timer
            self.timeout_daemon.register_operation(
                op_id, "barrier", is_async=False
            )

            start_time = time.perf_counter()
            result = original_barrier(*args, **kwargs)

            # Mark operation as completed
            self.timeout_daemon.mark_completed(op_id)
            end_time = time.perf_counter()
            duration = end_time - start_time

            trace_entry = self.create_trace_entry("barrier", start_time, duration, None)
            self.trace_data.append(trace_entry)

            self.log(
                f"[BARRIER] global rank {self.config.global_rank} -"
                f"barrier - async:0, "
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
                print(f"Applied hook to function: {func_name}")

        if hasattr(dist, "batch_isend_irecv"):
            self.hook_batch_isend_irecv()
            print("Applied hook to batch_isend_irecv")

        if hasattr(dist, "barrier"):
            original_barrier = getattr(dist, "barrier")
            setattr(dist, "barrier", self._trace_barrier(original_barrier))
            print("Applied hook to barrier")

    def remove_hooks(self):
        """Remove all tracing hooks from distributed functions"""
        for func_name, orig_func in self.original_functions.items():
            if hasattr(dist, func_name):
                setattr(dist, func_name, orig_func)
                self.log(f"Removed hook for {func_name}")

        # Restore batch_isend_irecv
        if "batch_isend_irecv" in self.original_functions:
            setattr(
                dist, "batch_isend_irecv", self.original_functions["batch_isend_irecv"]
            )
            self.log("Removed hook for batch_isend_irecv")

        # Restore barrier
        if "barrier" in self.original_functions:
            setattr(dist, "barrier", self.original_functions["barrier"])
            self.log("Removed hook for barrier")

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
        self.timeout_daemon.stop()

        if hasattr(self, 'config') and self.config.async_logger:
            self.config.async_logger.close()


def _cuda_sync():
    """Synchronize CUDA devices."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
