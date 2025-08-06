import time
import threading
from queue import Queue
from functools import wraps
from collections import defaultdict

# 假设以下导入已存在
from .get_group import get_participating_ranks
from . import torch
from . import dist
import os
import signal


class OperationTimer:
    """独立的计时器类，用于检测操作超时"""
    def __init__(self, timeout_threshold, callback):
        """
        Args:
            timeout_threshold: 超时阈值（秒）
            callback: 超时回调函数，格式: func(op_id, func_name, is_async, timed_out_type)
                      timed_out_type: "unfinished"（未完成超时）/"finished_late"（完成但超时）
        """
        self.timeout_threshold = timeout_threshold
        self.callback = callback  # 超时回调
        self.pending_ops = {}  # 待监控操作: {op_id: (start_time, func_name, is_async, is_completed)}
        self.lock = threading.Lock()  # 保护pending_ops的线程安全锁
        self.monitor_thread = None
        self.running = False

    def start(self):
        """启动监控线程"""
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor, daemon=True)
        self.monitor_thread.start()

    def stop(self):
        """停止监控线程"""
        self.running = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=1.0)

    def _monitor(self):
        """定期检查操作是否超时"""
        while self.running:
            current_time = time.perf_counter()
            with self.lock:
                # 遍历所有待监控操作
                for op_id, (start_time, func_name, is_async, is_completed) in list(self.pending_ops.items()):
                    # 未完成的操作才需要检查是否超时
                    if not is_completed:
                        if current_time - start_time > self.timeout_threshold:
                            # 未完成且超时，触发回调
                            self.callback(op_id, func_name, is_async, "unfinished")
                            # 标记为已完成（避免重复触发）
                            self.pending_ops[op_id] = (start_time, func_name, is_async, True)
            time.sleep(0.1)  # 降低CPU占用

    def register_operation(self, op_id, func_name, is_async):
        """注册一个新操作，开始计时"""
        with self.lock:
            self.pending_ops[op_id] = (time.perf_counter(), func_name, is_async, False)

    def mark_completed(self, op_id):
        """标记操作已完成，并检查是否完成超时"""
        with self.lock:
            if op_id in self.pending_ops:
                start_time, func_name, is_async, _ = self.pending_ops[op_id]
                end_time = time.perf_counter()
                # 检查是否完成但超时
                if end_time - start_time > self.timeout_threshold:
                    self.callback(op_id, func_name, is_async, "finished_late")
                # 移除已完成的操作（可选，减少内存占用）
                del self.pending_ops[op_id]


class CollectiveTracer:
    """集合通信追踪器，使用OperationTimer处理超时"""
    def __init__(self, trace_file=None, verbose=True, timeout_threshold=50):
        self.trace_file = trace_file
        self.verbose = verbose
        self.trace_data = []
        self.original_functions = {}
        self.hooked_functions = {}
        self.has_cuda = torch.cuda.is_available()
        self.global_rank = 0
        self.call_counts = defaultdict(lambda: defaultdict(lambda: {'count': 0}))
        self.my_rank = 0
        self.my_size = 1
        self.my_id_in_group = 0
        self.participate_ranks = []

        # 初始化超时计时器
        self.timer = OperationTimer(
            timeout_threshold=timeout_threshold,
            callback=self._timeout_callback  # 超时回调函数
        )
        # self.timer.start()

        # 初始化待监控的函数
        function_names = [
            'all_reduce', 'all_gather', 'reduce_scatter', 'broadcast',
            'reduce_scatter_base', 'all_gather_base', '_all_gather_base',
            '_reduce_scatter_base', 'reduce_scatter_tensor', 'all_gather_into_tensor'
        ]
        for func_name in function_names:
            if hasattr(dist, func_name):
                self.hooked_functions[func_name] = getattr(dist, func_name)
            else:
                print(f"!!! torch.distributed 中未找到函数 {func_name}，已跳过")

        if not self.hooked_functions:
            print("!!! WARNING !!! 没有找到任何要追踪的函数")

    def _timeout_callback(self, op_id, func_name, is_async, timed_out_type):
        """超时回调：记录超时日志"""
        timed_out = True
        os.kill(os.getpid(), signal.SIGINT)
        if timed_out_type == "unfinished":
            msg = f"[TIMEOUT] 操作 {func_name} (ID: {op_id}) 未完成且超时！"
        else:
            msg = f"[TIMEOUT] 操作 {func_name} (ID: {op_id}) 完成但超时！"
        self._log(msg)

    def _log(self, message):
        """日志输出"""
        if self.verbose:
            print(message)
        if self.trace_file:
            ranked_filename = f"{self.trace_file}-{self.global_rank}"
            with open(ranked_filename, 'a') as f:
                f.write(message + '\n')

    def create_trace_entry(self, func_name, start_time, duration, tensor_info, timed_out):
        """创建追踪记录"""
        return {
            'function': func_name,
            'timestamp': start_time,
            'duration': duration,
            'tensor_shape': tensor_info['shape'],
            'tensor_dtype': str(tensor_info['dtype']),
            'tensor_size': tensor_info['size'],
            'timed_out': timed_out,
            'is_async': False
        }

    def _extract_tensor_info(self, args, kwargs):
        """提取张量信息"""
        tensor = None
        # 从参数中查找张量
        for arg in args:
            if isinstance(arg, torch.Tensor):
                tensor = arg
                break
        if tensor is None:
            for key, value in kwargs.items():
                if isinstance(value, torch.Tensor):
                    tensor = value
                    break
        if tensor is None and args:
            first_arg = args[0]
            for attr in dir(first_arg):
                try:
                    value = getattr(first_arg, attr)
                    if isinstance(value, torch.Tensor):
                        tensor = value
                        break
                except:
                    continue
        if tensor is None:
            return {'shape': 'unknown', 'dtype': 'unknown', 'size': 0}
        return {
            'shape': tuple(tensor.shape),
            'dtype': tensor.dtype,
            'size': tensor.element_size() * tensor.numel()
        }

    def _trace_wrapper(self, func_name, orig_func):
        """包装原函数，添加追踪和超时监控"""
        class TimedWork:
            """包装异步操作的Work对象"""
            def __init__(self, work, op_id, start_time, func_name, tensor_info, tracer):
                self.work = work
                self.op_id = op_id
                self.start_time = start_time
                self.func_name = func_name
                self.tensor_info = tensor_info
                self.tracer = tracer

            def wait(self):
                result = self.work.wait()
                if self.tracer.has_cuda:
                    _cuda_sync()
                # 标记操作完成，由timer判断是否超时
                self.tracer.timer.mark_completed(self.op_id)
                # 记录耗时
                end_time = time.perf_counter()
                duration = end_time - self.start_time
                timed_out = duration > self.tracer.timer.timeout_threshold
                # 记录追踪信息
                entry = self.tracer.create_trace_entry(
                    self.func_name, self.start_time, duration, self.tensor_info, timed_out
                )
                entry['is_async'] = True
                self.tracer.trace_data.append(entry)
                self.tracer._log(f"[TRACE] 异步 {self.func_name} 耗时: {duration*1e3:.3f}ms")
                return result

            def is_completed(self):
                return self.work.is_completed()

        @wraps(orig_func)
        def wrapper(*args, **kwargs):
            # 提取张量信息和初始化参数
            tensor_info = self._extract_tensor_info(args, kwargs)
            shape = tensor_info['shape']
            self.call_counts[func_name][shape]['count'] += 1
            group = kwargs.get('group') or (args[2] if len(args) > 2 else None)
            self.my_rank, self.my_size, self.my_id_in_group, self.participate_ranks = get_participating_ranks(group)
            self.global_rank = dist.get_rank()

            if self.has_cuda:
                _cuda_sync()
            start_time = time.perf_counter()
            self.timer.start()
            is_async = kwargs.get('async_op', False)
            op_id = id((args, kwargs, time.time()))  # 生成唯一操作ID

            # 注册操作到计时器
            self.timer.register_operation(op_id, func_name, is_async)

            if is_async:
                # 异步操作：直接执行并返回包装后的Work
                work = orig_func(*args, **kwargs)
                return TimedWork(work, op_id, start_time, func_name, tensor_info, self)
            else:
                # 同步操作：用线程执行，支持超时检测
                result = [None]
                error = [None]

                def sync_executor():
                    try:
                        result[0] = orig_func(*args, **kwargs)
                    except Exception as e:
                        error[0] = e
                    finally:
                        if self.has_cuda:
                            _cuda_sync()

                # 启动线程执行同步操作
                exec_thread = threading.Thread(target=sync_executor)
                exec_thread.start()
                # 等待线程完成（最多等待超时阈值时间）
                exec_thread.join(timeout=self.timer.timeout_threshold)

                # 检查线程状态（是否超时）
                if exec_thread.is_alive():
                    # 未完成超时（由timer的回调已处理日志）
                    raise TimeoutError(f"同步 {func_name} 超时（>{self.timer.timeout_threshold}s）")
                if error[0] is not None:
                    raise error[0]

                # 操作完成，标记并检查是否超时
                self.timer.mark_completed(op_id)
                end_time = time.perf_counter()
                duration = end_time - start_time
                timed_out = duration > self.timer.timeout_threshold

                # 记录追踪信息
                entry = self.create_trace_entry(func_name, start_time, duration, tensor_info, timed_out)
                self.trace_data.append(entry)
                self._log(f"[TRACE] 同步 {func_name} 耗时: {duration*1e3:.3f}ms")
                return result[0]

        return wrapper

    def apply_hooks(self):
        """安装钩子"""
        for func_name, orig_func in self.hooked_functions.items():
            if hasattr(dist, func_name):
                self.original_functions[func_name] = getattr(dist, func_name)
                setattr(dist, func_name, self._trace_wrapper(func_name, orig_func))
                self._log(f"已为 {func_name} 安装钩子")

    def remove_hooks(self):
        """移除钩子"""
        for func_name, orig_func in self.original_functions.items():
            if hasattr(dist, func_name):
                setattr(dist, func_name, orig_func)
                self._log(f"已移除 {func_name} 的钩子")

    def __del__(self):
        """销毁时停止计时器"""
        self.timer.stop()


def _cuda_sync():
    torch.cuda.synchronize()