import time
from functools import wraps
from typing import List, Optional, Union, Tuple
from collections import defaultdict

from .get_group import get_participating_ranks
from . import torch
from . import dist

"""
export PYTHONPATH=/home/yang:$PYTHONPATH  # 设置环境变量

import sys
sys.path.insert(0, '/home/yang')  # 把 /home/yang 路径添加到搜索路径的最前面
"""

function_names = [
    'all_reduce',
    'all_gather',
    'reduce_scatter',
    'broadcast',
    'reduce_scatter_base',
    'all_gather_base',
    '_all_gather_base',
    'all_gather_into_tensor',
    '_reduce_scatter_base',
]

# '_all_gather_base'
# 'reduce_scatter_tensor',
# 'batch_isend_irecv',


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
        self.trace_file = trace_file
        self.verbose = verbose
        self.trace_data = []
        self.original_functions = {}
        self.hooked_functions = {}
        self.has_cuda = torch.cuda.is_available()
        for func_name in function_names:
            if hasattr(dist, func_name):
                self.hooked_functions[func_name] = getattr(dist, func_name)
            else:
                print(f"!!! torch.distributed 中未找到函数 {func_name}，已跳过")

        if not self.hooked_functions:
            print("!!! WARNING !!! 没有找到任何要追踪的函数")

        self.call_counts = defaultdict(lambda: defaultdict(lambda: {'count': 0}))
        self.my_rank = 0  # partly rank in group
        self.my_size = 1
        self.my_id_in_group = 0
        self.participate_ranks = []

        self.global_rank = 0
        
    def _log(self, message):
        """Log a message to console and/or file."""
        if self.verbose:
            print(message)
        if self.trace_file:
            ranked_filename = f"{self.trace_file}-{self.global_rank}"
            with open(ranked_filename, 'a') as f:
                f.write(message + '\n')
    
    def create_trace_entry(self, func_name, start_time, duration, tensor_info):
        """Create a trace entry."""
        return {
            'function': func_name,
            'timestamp': start_time,
            'duration': duration,
            'tensor_shape': tensor_info['shape'],
            'tensor_dtype': str(tensor_info['dtype']),
            'tensor_size': tensor_info['size']
        }
    
    def _trace_wrapper(self, func_name, orig_func):
        """Create a wrapper for the original function to trace its execution."""
        class TimedWork:
            def __init__(self, work, start_time, func_name, data_size, tensor_info=None, Tracer=None):
                self.work = work
                self.start_time = start_time
                self.func_name = func_name
                self.data_size = data_size
                self.tensor_info = tensor_info if tensor_info else {'shape': 'unknown', 'dtype': 'unknown', 'size': 0}
                self.tracer = Tracer
                
            def wait(self):
                result = self.work.wait()

                # if self.tracer.has_cuda:
                #     _cuda_sync()

                end_time = time.perf_counter()
                duration = end_time - self.start_time
                
                # Create a trace entry
                trace_entry = self.tracer.create_trace_entry(func_name, self.start_time, duration, self.tensor_info)
                self.tracer.trace_data.append(trace_entry)
                
                # Print trace information
                self.tracer._log(f"[TRACE] global rank {self.tracer.global_rank} in GROUP_{self.tracer.my_id_in_group} - {func_name} - async:1, "
                        f"Size: {self.tensor_info['size']/1024/1024:.2f} MB, "
                        f"Shape: {self.tensor_info['shape']},"
                        f"Dtype: {self.tensor_info['dtype']}, "
                        f"Duration: {duration*1e3:.3f} ms, "
                        f"GROUP size {self.tracer.my_size}  = {self.tracer.participate_ranks},"
                        f"call count: {self.tracer.call_counts[func_name][self.tensor_info['shape']]['count']}"
                    )
  
                return result
            
            def is_completed(self):
                return self.work.is_completed()
            
        @wraps(orig_func)
        def wrapper(*args, **kwargs):

            tensor_info = self._extract_tensor_info(args, kwargs)

            shape = tensor_info['shape'] if tensor_info else 'unknown'
            op = func_name
            self.call_counts[op][shape]['count'] += 1


            tensor = args[0] if args else None
            # print(f"tensor.numel={tensor.numel()}   tensor.element_size={tensor.element_size()}\n")  不能在这打印
            data_size = tensor.numel() * tensor.element_size() if tensor is not None else 0

            group = kwargs.get('group') or (args[2] if len(args) > 2 else None)
            self.my_rank, self.my_size, self.my_id_in_group, self.participate_ranks = get_participating_ranks(group)

            self.global_rank = dist.get_rank()


            if self.has_cuda:
                _cuda_sync()
            start_time = time.perf_counter()

            
            is_async = kwargs.get('async_op', False)
            if is_async:
                work = orig_func(*args, **kwargs)

                return TimedWork(work, start_time, func_name, data_size, tensor_info, self)
            else:
                work = orig_func(*args, **kwargs)
                
                # if self.has_cuda:
                #     _cuda_sync()

                end_time = time.perf_counter()
                duration = end_time - start_time
                
                trace_entry = self.create_trace_entry(func_name, start_time, duration, tensor_info)
                self.trace_data.append(trace_entry)
                
                # Print trace information
                self._log(f"[TRACE] global rank {self.global_rank} in GROUP_{self.my_id_in_group} - {func_name} - async:0, "
                        f"Size: {tensor_info['size']/1024/1024:.2f} MB, "
                        f"Shape: {tensor_info['shape']},"
                        f"Dtype: {tensor_info['dtype']}, "
                        f"Duration: {duration*1e3:.3f} ms, "
                        f"GROUP size {self.my_size}  = {self.participate_ranks},"
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
            for key, value in kwargs.items():
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
                except:
                    continue
        
        if tensor is None:
            return {'shape': 'unknown', 'dtype': 'unknown', 'size': 0}
            
        return {
            'shape': tuple(tensor.shape),
            'dtype': tensor.dtype,
            'size': tensor.element_size() * tensor.numel()
        }
    
    def hook_batch_isend_irecv(self):
        if not hasattr(dist, 'batch_isend_irecv'):
            print("!!! WARNING !!! 没有找到 batch_isend_irecv 函数")
            return
        
        original_batch_isend_irecv = dist.batch_isend_irecv
    
        def wrapped_batch_isend_irecv(ops_list):
            send_total = 0
            recv_total = 0
            
            for op in ops_list:
                if isinstance(op, dist.P2POp):
                    tensor = op.tensor
                    data_size = tensor.numel() * tensor.element_size() 
                    
                    if op.op == dist.isend:
                        send_total += data_size
                    elif op.op == dist.irecv:
                        recv_total += data_size
            
            send_mb = send_total / (1024 * 1024)
            recv_mb = recv_total / (1024 * 1024)
            
            self._log(f"[TRACE] global rank {dist.get_rank()} - send: {send_total} bytes ({send_mb:.2f} MB), "
                      f"recv: {recv_total} bytes ({recv_mb:.2f} MB), "
                      f"ops count: {len(ops_list)}")
            
            return original_batch_isend_irecv(ops_list)
    
        dist.batch_isend_irecv = wrapped_batch_isend_irecv
     
    
    def apply_hooks(self):
        for func_name, orig_func in self.hooked_functions.items():
            if hasattr(dist, func_name):
                self.original_functions[func_name] = getattr(dist, func_name)
                setattr(dist, func_name, self._trace_wrapper(func_name, orig_func))
                self._log(f"Applyed hook to function: {func_name}")
        
        if hasattr(dist, 'batch_isend_irecv'):
            self.hook_batch_isend_irecv()
        
        
    
    def remove_hooks(self):
        for func_name, orig_func in self.original_functions.items():
            if hasattr(dist, func_name):
                setattr(dist, func_name, orig_func)
                self._log(f"Removed hook from function: {func_name}")
    
    def get_trace_data(self):
        return self.trace_data
    
    def get_all_call_counts(self):
        return self.call_counts.copy()
    
    def export_to_csv(self, filename):
        import csv
        if not self.trace_data:
            self._log("No trace data to export.")
            return
            
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = self.trace_data[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in self.trace_data:
                writer.writerow(row)
                
        self._log(f"Exported trace data to {filename}")

def _cuda_sync():
    torch.cuda.synchronize()
    