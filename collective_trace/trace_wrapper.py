"""Wrapper functions for tracing PyTorch distributed collective operations."""

import time
import uuid

from functools import wraps

import torch
try:
    import torch.distributed as dist
except ImportError:
    print("!!! PyTorch not found, skipped")

from async_timer import AsyncTimer
from .shared_coealescing_state import coalescing_state
from .trace_utils import extract_tensor_info




class TimedWork:
    """Wrap async work to track completion and timing"""

    def __init__(self, work, op_id, timer, func_name, **kwargs):
        self.work = work
        self.op_id = op_id
        self.timer = timer
        self.func_name = func_name
        self.tensor_info = kwargs.get(
            "tensor_info", {"shape": "unknown", "dtype": "unknown", "size": 0}
        )
        self.tracer = kwargs.get("tracer")

    def wait(self):
        """Wait for the work to complete"""
        current_stream = torch.cuda.current_stream(torch.cuda.current_device())
        stream_ptr = current_stream.cuda_stream
        self.timer.start(stream_ptr)

        self.tracer.timeout_manager.register_operation(self.op_id, self.func_name, True)
        result = self.work.wait()
        self.tracer.timeout_manager.mark_completed(self.op_id)

        self.timer.end(stream_ptr)

        while not self.timer.is_completed():
            time.sleep(0.1)

        duration = self.timer.get_elapsed()

        trace_entry = self.tracer.create_trace_entry(
            self.func_name, "", duration, self.tensor_info
        )
        self.tracer.trace_data.append(trace_entry)

        self.tracer.log(
            f"[TRACE] global rank {self.tracer.config.global_rank} "
            f"in GROUP_{self.tracer.group_info.my_idx_in_group} "
            f"- {self.func_name} - async:1, "
            f"Size: {self.tensor_info['size'] / 1024 / 1024:.2f} MB, "
            f"Shape: {self.tensor_info['shape']}, "
            f"Dtype: {self.tensor_info['dtype']}, "
            f"Duration: {duration} ms, "
            f"GROUP size {self.tracer.group_info.my_size}  = "
            f"{self.tracer.group_info.participate_ranks}, "
            f"call count: "
            f"{self.tracer.call_counts[self.func_name][self.tensor_info['shape']]['count']}"
        )
        return result

    def is_completed(self):
        """Check if the work is completed"""
        return self.work.is_completed()


def create_function_wrapper(func_name, orig_func, tracer):
    """Create wrapper for collective functions"""

    @tracer.with_global_rank
    @wraps(orig_func)
    def wrapper(*args, **kwargs):
        tensor_info = extract_tensor_info(args, kwargs)
        tracer.call_counts[func_name] \
            [tensor_info["shape"] if tensor_info else "unknown"]["count"] += 1

        # Update group info
        tracer.update_group_info(kwargs.get("group") or (args[2] if len(args) > 2 else None))

        if coalescing_state.active_cm_id is not None:
            cm_id = coalescing_state.active_cm_id
            coalescing_state.counter[cm_id] = coalescing_state.counter.get(cm_id, 0) + 1
            coalescing_state.names[cm_id] = func_name
            if coalescing_state.sizes.get(cm_id) is None:
                coalescing_state.sizes[cm_id] = tensor_info["size"]
            else:
                coalescing_state.sizes[cm_id] += tensor_info["size"]

        # cuda_sync()
        start_time = time.time()  # tmp
        timer = AsyncTimer()

        is_async = kwargs.get("async_op", False)
        op_id = uuid.uuid4()

        if is_async:
            work = orig_func(*args, **kwargs)
            return TimedWork(
                work,
                op_id,
                timer,
                func_name,
                tensor_info=tensor_info,
                tracer=tracer,
            )

        # Synchronous operation
        tracer.timeout_manager.register_operation(op_id, func_name, is_async)
        kwargs_for_call = dict(kwargs)
        kwargs_for_call["async_op"] = True
        result = orig_func(*args, **kwargs_for_call)
        # result = orig_func(*args, **kwargs)

        current_stream = torch.cuda.current_stream(torch.cuda.current_device())
        stream_ptr = current_stream.cuda_stream
        timer.start(stream_ptr)
        result.wait()
        timer.end(stream_ptr)

        if tracer.timeout_manager.is_timed_out(op_id):
            tracer.log(
                f"[ERROR] Synchronous operation {func_name} (ID: {op_id}) has timed out"
            )

        timer.end()
        while not timer.is_completed():
            time.sleep(0.1)
        duration = timer.get_elapsed()

        tracer.timeout_manager.mark_completed(op_id)

        trace_entry = tracer.create_trace_entry(
            func_name, start_time, duration, tensor_info
        )
        tracer.trace_data.append(trace_entry)

        tracer.log(
            f"[TRACE] global rank {tracer.config.global_rank} "
            f"in GROUP_{tracer.group_info.my_idx_in_group} "
            f"- {func_name} - async:0, "
            f"Size: {tensor_info['size'] / 1024 / 1024:.2f} MB, "
            f"Shape: {tensor_info['shape']}, "
            f"Dtype: {tensor_info['dtype']}, "
            f"Duration: {duration} ms, "
            f"GROUP size {tracer.group_info.my_size}  = "
            f"{tracer.group_info.participate_ranks}, "
            f"call count: {tracer.call_counts[func_name][tensor_info['shape']]['count']}"
        )
        return result

    return wrapper


def create_barrier_wrapper(original_barrier, tracer):
    """Create wrapper for barrier function"""

    @tracer.with_global_rank
    @wraps(original_barrier)
    def wrapper(*args, **kwargs):
        op_id = uuid.uuid4()
        tracer.timeout_manager.register_operation(op_id, "barrier", is_async=False)

        start_time = time.perf_counter()
        result = original_barrier(*args, **kwargs)

        tracer.timeout_manager.mark_completed(op_id)
        end_time = time.perf_counter()
        duration = end_time - start_time

        trace_entry = tracer.create_trace_entry("barrier", start_time, duration, None)
        tracer.trace_data.append(trace_entry)

        tracer.log(
            f"[BARRIER] global rank {tracer.config.global_rank} -"
            f"barrier - async:0, "
            f"Duration: {duration * 1e3:.3f} ms"
        )
        return result

    return wrapper


def create_batch_isend_irecv_wrapper(original_func, tracer):
    """Create wrapper for batch_isend_irecv function"""

    @tracer.with_global_rank
    def wrapper(ops_list):
        op_id = uuid.uuid4()
        tracer.timeout_manager.register_operation(
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
        result = original_func(ops_list)
        tracer.timeout_manager.mark_completed(op_id)

        # Format log
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

        tracer.log(
            f"[BATCH] global rank {tracer.config.global_rank} - "
            f"send: {stats['send_total']} bytes ({send_mb:.2f} MB), "
            f"shape: [{send_shapes_str}] 目标: [{send_targets_str}], "
            f"recv: {stats['recv_total']} bytes ({recv_mb:.2f} MB), "
            f"shape: [{recv_shapes_str}[ 来源: [{recv_sources_str}], "
            f"ops count: {len(ops_list)}"
        )
        return result

    return wrapper
