import time
import threading
from queue import Queue
from functools import wraps
from collections import defaultdict
from dataclasses import dataclass
import csv
import os
import signal

try:
    import torch
    import torch.distributed as dist
except ImportError:
    print("!!! PyTorch not found, skipped")

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
            "_all_gather_base",
            "_reduce_scatter_base",
        ]

class OperationTimer:
    """Independent timer class for operation timeout detection"""
    def __init__(self, timeout_threshold, callback):
        """
        Args:
            timeout_threshold: Timeout threshold in seconds
            callback: Timeout callback function with format: func(op_id, func_name, is_async, timed_out_type)
                      timed_out_type: "unfinished" (timed out without completion) / "finished_late" (completed but timed out)
        """
        self.timeout_threshold = timeout_threshold
        self.callback = callback  # Timeout callback
        self.pending_ops = {}  # Operations to monitor: {op_id: (start_time, func_name, is_async, is_completed)}
        self.lock = threading.Lock()  # Thread-safe lock for pending_ops
        self.monitor_thread = None
        self.running = False
        self.timeout_events = {}  # Store operation timeout events

    def start(self):
        """Start the monitoring thread"""
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor, daemon=True)
        self.monitor_thread.start()

    def stop(self):
        """Stop the monitoring thread"""
        self.running = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=1.0)

    def _monitor(self):
        """Periodically check if operations have timed out"""
        while self.running:
            current_time = time.perf_counter()
            with self.lock:
                # Iterate through all operations to monitor
                for op_id, (start_time, func_name, is_async, is_completed) in list(self.pending_ops.items()):
                    # Only check incomplete operations for timeouts
                    if not is_completed:
                        if current_time - start_time > self.timeout_threshold:
                            # Operation incomplete and timed out, trigger callback
                            self.callback(op_id, func_name, is_async, "unfinished")
                            # Mark as completed to avoid duplicate triggers
                            self.pending_ops[op_id] = (start_time, func_name, is_async, True)
            time.sleep(0.1)  # Reduce CPU usage

    def register_operation(self, op_id, func_name, is_async):
        """Register a new operation and start timing"""
        with self.lock:
            self.pending_ops[op_id] = (time.perf_counter(), func_name, is_async, False)
            self.timeout_events[op_id] = threading.Event()

    def unregister_operation(self, op_id):
        if op_id in self.pending_ops:
            del self.pending_ops[op_id]
        if op_id in self.timeout_events:
            del self.timeout_events[op_id]

    def mark_completed(self, op_id):
        """Mark operation as completed and check if it finished late"""
        with self.lock:
            if op_id in self.pending_ops:
                start_time, func_name, is_async, _ = self.pending_ops[op_id]
                end_time = time.perf_counter()
                # Check if completed but timed out
                if end_time - start_time > self.timeout_threshold:
                    self.callback(op_id, func_name, is_async, "finished_late")
                # Remove completed operation
                del self.pending_ops[op_id]
                if op_id in self.timeout_events:
                    self.timeout_events[op_id].set()
                    del self.timeout_events[op_id]

    def is_timed_out(self, op_id):
        """Check if an operation has timed out"""
        with self.lock:
            if op_id not in self.pending_ops:
                return False
            start_time, _, _, is_completed = self.pending_ops[op_id]
            return not is_completed and (time.perf_counter() - start_time > self.timeout_threshold)


@dataclass
class TraceConfig:
    """Configuration for tracing"""
    trace_file: str = None
    verbose: bool = True
    has_cuda: bool = False
    global_rank: int = 0


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


class CollectiveTracer:
    """
    Trace collective operations for distributed training with timeout detection.
    """

    def __init__(self, trace_file=None, verbose=True, timeout_threshold=None):
        """
        Args:
            trace_file: Log file to store trace data, if None, no file will be created
            verbose: Whether to print messages or not
            timeout_threshold: Timeout threshold in seconds. If None, read from COLLECTIVE_TIMEOUT environment variable, default 50s
        """
        # Get timeout threshold from environment variable, use default 50s if not set
        if timeout_threshold is None:
            env_timeout = os.environ.get('COLLECTIVE_TIMEOUT')
            if env_timeout is not None:
                try:
                    self.timeout_threshold = float(env_timeout)
                    print(f"Read timeout threshold from environment variable COLLECTIVE_TIMEOUT: {self.timeout_threshold}s")
                except ValueError:
                    print(f"Invalid value for environment variable COLLECTIVE_TIMEOUT: {env_timeout}, using default 50s")
                    self.timeout_threshold = 50.0
            else:
                self.timeout_threshold = 50.0  # Default value
        else:
            self.timeout_threshold = timeout_threshold

        self.config = TraceConfig(
            trace_file=trace_file,
            verbose=verbose,
            has_cuda=torch.cuda.is_available() if 'torch' in globals() else False,
            global_rank=0,
        )
        self.trace_data = []
        self.original_functions = {}
        self.hooked_functions = {}

        for func_name in function_names:
            if hasattr(dist, func_name):
                self.hooked_functions[func_name] = getattr(dist, func_name)
            else:
                print(f"!!! Function {func_name} not found in torch.distributed, skipped")

        if not self.hooked_functions:
            print("!!! WARNING !!! No functions to trace found")

        self.call_counts = defaultdict(lambda: defaultdict(lambda: {"count": 0}))
        self.group_info = GroupState()

        self.timer = OperationTimer(
            timeout_threshold=self.timeout_threshold,  # 这里改为使用成员变量
            callback=self._timeout_callback  
        )
        self.timer.start()

    def _timeout_callback(self, op_id, func_name, is_async, timed_out_type):
        """Timeout callback: log timeout events"""
        if timed_out_type == "unfinished":
            msg = f"[TIMEOUT] Operation {func_name} (ID: {op_id}) timed out without completion! Exceeded threshold {self.timeout_threshold}s"
        else:
            msg = f"[TIMEOUT] Operation {func_name} (ID: {op_id}) completed but timed out! Exceeded threshold {self.timeout_threshold}s"
        
        self.log(msg)
        os.kill(os.getpid(), signal.SIGUSR1)
        self.timer.unregister_operation(op_id)

    def log(self, message):
        """Log a message to console and/or file."""
        if self.config.verbose:
            print(message)
        if self.config.trace_file:
            ranked_filename = f"{self.config.trace_file}-{self.config.global_rank}"
            with open(ranked_filename, "a", encoding="utf-8") as f:
                f.write(message + "\n")

    def with_global_rank(self, func):
        """Decorator that updates global rank before calling the wrapped function"""

        @wraps(func)
        def wrapper(*args, **kwargs):
            if 'dist' in globals() and dist.is_available() and dist.is_initialized():
                self.config.global_rank = dist.get_rank()
            return func(*args, **kwargs)

        return wrapper

    def create_trace_entry(self, func_name, start_time, duration, tensor_info, timed_out=False, is_async=False):
        """Create a trace entry with timeout information."""
        if tensor_info is None:
            tensor_info = {"shape": "unknown", "dtype": "unknown", "size": 0}

        return {
            "function": func_name,
            "timestamp": start_time,
            "duration": duration,
            "tensor_shape": tensor_info["shape"],
            "tensor_dtype": str(tensor_info["dtype"]),
            "tensor_size": tensor_info["size"],
            "timed_out": timed_out,
            "is_async": is_async
        }

    def _trace_wrapper(self, func_name, orig_func):
        """Create a wrapper for the original function to trace its execution and detect timeouts."""

        class TimedWork:
            """
            A class to wrap the work and time it, with timeout detection.
            """

            def __init__(self, work, op_id, start_time, func_name, tensor_info, tracer):
                self.work = work
                self.op_id = op_id
                self.start_time = start_time
                self.func_name = func_name
                self.tensor_info = tensor_info
                self.tracer = tracer

            def wait(self):
                result = self.work.wait()

                if self.tracer.config.has_cuda:
                    _cuda_sync()

                # Mark operation as completed and check for timeout
                self.tracer.timer.mark_completed(self.op_id)
                
                end_time = time.perf_counter()
                duration = end_time - self.start_time
                timed_out = duration > self.tracer.timer.timeout_threshold

                # Create a trace entry
                trace_entry = self.tracer.create_trace_entry(
                    self.func_name, self.start_time, duration, self.tensor_info,
                    timed_out, is_async=True
                )
                self.tracer.trace_data.append(trace_entry)

                # Print trace information
                self.tracer.log(
                    f"[TRACE] global rank {self.tracer.config.global_rank} "
                    f"in GROUP_{self.tracer.group_info.my_idx_in_group} "
                    f"- {self.func_name} - async:1, "
                    f"Size: {self.tensor_info['size']/1024/1024:.2f} MB, "
                    f"Shape: {self.tensor_info['shape']}, "
                    f"Dtype: {self.tensor_info['dtype']}, "
                    f"Duration: {duration*1e3:.3f} ms, "
                    f"Timed out: {'Yes' if timed_out else 'No'}, "
                    f"GROUP size {self.tracer.group_info.my_size}  = "
                    f"{self.tracer.group_info.participate_ranks}, "
                    f"call count: "
                    f"{self.tracer.call_counts[self.func_name][self.tensor_info['shape']]['count']}"
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
            op_id = id((args, kwargs, time.time()))

            # Register operation with timer
            self.timer.register_operation(op_id, func_name, is_async)

            if is_async:
                work = orig_func(*args, **kwargs)
                return TimedWork(
                    work, op_id, start_time, func_name, tensor_info, self
                )

            # Synchronous operation
            result = orig_func(*args, **kwargs)

            # Check if already timed out
            if self.timer.is_timed_out(op_id):
                self.log(f"[ERROR] Synchronous operation {func_name} (ID: {op_id}) has timed out")

            # Mark operation as completed
            self.timer.mark_completed(op_id)
            
            end_time = time.perf_counter()
            duration = end_time - start_time
            timed_out = duration > self.timer.timeout_threshold

            trace_entry = self.create_trace_entry(
                func_name, start_time, duration, tensor_info, timed_out
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
                f"Timed out: {'Yes' if timed_out else 'No'}, "
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
            if 'torch' in globals() and isinstance(arg, torch.Tensor):
                tensor = arg
                break

        # If not found, try to find a tensor in keyword arguments
        if tensor is None:
            for _, value in kwargs.items():
                if 'torch' in globals() and isinstance(value, torch.Tensor):
                    tensor = value
                    break

        # If still not found, check if the first argument is an object with a tensor attribute
        if tensor is None and args:
            first_arg = args[0]
            for attr in dir(first_arg):
                try:
                    value = getattr(first_arg, attr)
                    if 'torch' in globals() and isinstance(value, torch.Tensor):
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
        if not ('dist' in globals() and hasattr(dist, "batch_isend_irecv")):
            print("!!! WARNING !!! batch_isend_irecv function not found")
            return

        original_batch_isend_irecv = dist.batch_isend_irecv

        @self.with_global_rank
        def wrapped_batch_isend_irecv(ops_list):
            # Generate unique operation ID
            op_id = id((ops_list, time.time()))
            # Register operation with timer
            self.timer.register_operation(op_id, "batch_isend_irecv", is_async=True)
            
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

            start_time = time.perf_counter()
            # Execute original operation
            result = original_batch_isend_irecv(ops_list)
            
            # Mark operation as completed
            self.timer.mark_completed(op_id)
            end_time = time.perf_counter()
            duration = end_time - start_time
            timed_out = duration > self.timer.timeout_threshold

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

            # Record trace information
            trace_entry = self.create_trace_entry(
                "batch_isend_irecv", start_time, duration, 
                {
                    "shape": f"send: {send_shapes_str}, recv: {recv_shapes_str}",
                    "dtype": "mixed",
                    "size": stats["send_total"] + stats["recv_total"]
                },
                timed_out,
                is_async=True
            )
            self.trace_data.append(trace_entry)

            self.log(
                f"[BATCH] global rank {self.config.global_rank} - "
                f"send: {stats['send_total']} bytes ({send_mb:.2f} MB), "
                f"shape: [{send_shapes_str}] targets: [{send_targets_str}], "
                f"recv: {stats['recv_total']} bytes ({recv_mb:.2f} MB), "
                f"shape: [{recv_shapes_str}] sources: [{recv_sources_str}], "
                f"ops count: {len(ops_list)}, "
                f"Duration: {duration*1e3:.3f} ms, "
                f"Timed out: {'Yes' if timed_out else 'No'}"
            )

            return result

        dist.batch_isend_irecv = wrapped_batch_isend_irecv

    def _trace_barrier(self, original_barrier):
        @self.with_global_rank
        @wraps(original_barrier)
        def barrier_traced(*args, **kwargs):
            # Generate unique operation ID
            op_id = id((args, kwargs, time.time()))
            # Register operation with timer
            self.timer.register_operation(op_id, "barrier", is_async=False)
            
            start_time = time.perf_counter()
            result = original_barrier(*args, **kwargs)
            
            # Mark operation as completed
            self.timer.mark_completed(op_id)
            end_time = time.perf_counter()
            duration = end_time - start_time
            timed_out = duration > self.timer.timeout_threshold

            trace_entry = self.create_trace_entry(
                "barrier", start_time, duration, None, timed_out
            )
            self.trace_data.append(trace_entry)

            self.log(
                f"[BARRIER] global rank {self.config.global_rank} -"
                f"barrier - async:0, "
                f"Duration: {duration*1e3:.3f} ms, "
                f"Timed out: {'Yes' if timed_out else 'No'}"
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
            if 'dist' in globals() and hasattr(dist, func_name):
                setattr(dist, func_name, orig_func)
                self.log(f"Removed hook for {func_name}")

        # Restore batch_isend_irecv
        if 'batch_isend_irecv' in self.original_functions:
            setattr(dist, 'batch_isend_irecv', self.original_functions['batch_isend_irecv'])
            self.log("Removed hook for batch_isend_irecv")

        # Restore barrier
        if 'barrier' in self.original_functions:
            setattr(dist, 'barrier', self.original_functions['barrier'])
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
        self.timer.stop()


def _cuda_sync():
    """Synchronize CUDA devices."""
    if 'torch' in globals() and torch.cuda.is_available():
        torch.cuda.synchronize()
    