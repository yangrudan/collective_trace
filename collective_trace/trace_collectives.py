import torch
import torch.distributed as dist
import time
from functools import wraps


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
        self.hooked_functions = {
            'all_reduce': dist.all_reduce,
            'all_gather': dist.all_gather,
            'reduce_scatter': dist.reduce_scatter,
            'broadcast': dist.broadcast
        }
        
    def _log(self, message):
        """Log a message to console and/or file."""
        if self.verbose:
            print(message)
        if self.trace_file:
            with open(self.trace_file, 'a') as f:
                f.write(message + '\n')
    
    def _trace_wrapper(self, func_name, orig_func):
        """Create a wrapper for the original function to trace its execution."""
        class TimedWork:
            def __init__(self, work, start_time, func_name, data_size, tensor_info=None):
                self.work = work
                self.start_time = start_time
                self.func_name = func_name
                self.data_size = data_size
                self.tensor_info = tensor_info if tensor_info else {'shape': 'unknown', 'dtype': 'unknown', 'size': 0}
                
            def wait(self):
                result = self.work.wait()
                end_time = time.time()
                duration = end_time - self.start_time
                
                # Create a trace entry
                trace_entry = {
                    'function': func_name,
                    'timestamp': self.start_time,
                    'duration': duration,
                    'tensor_shape': self.tensor_info['shape'],
                    'tensor_dtype': str(self.tensor_info['dtype']),
                    'tensor_size': self.tensor_info['size']
                }
                #self.trace_data.append(trace_entry)
                
                # Print trace information
                self.print(f"[TRACE] {func_name} - Shape: {self.tensor_info['shape']}, "
                        f"Dtype: {self.tensor_info['dtype']}, Size: {self.tensor_info['size']/1024/1024:.2f} MB, "
                        f"Duration: {duration*1000:.2f} ms")
                
                return result
            
            def is_completed(self):
                return self.work.is_completed()
            
        @wraps(orig_func)
        def wrapper(*args, **kwargs):
            tensor_info = self._extract_tensor_info(args, kwargs)

            start_time = time.time()
            tensor = args[0] if args else None
            print(f"tensor.numel={tensor.numel()}   tensor.element_size={tensor.element_size()}\n")
            data_size = tensor.numel() * tensor.element_size() if tensor is not None else 0
            
            is_async = kwargs.get('async_op', False)
            if is_async:
                work = orig_func(*args, **kwargs)

                return TimedWork(work, start_time, func_name, data_size, tensor_info)
            else:
                work = orig_func(*args, **kwargs)
                
                end_time = time.time()
                duration = end_time - start_time
                
                # Create a trace entry
                trace_entry = {
                    'function': func_name,
                    'timestamp': start_time,
                    'duration': duration,
                    'tensor_shape': tensor_info['shape'],
                    'tensor_dtype': str(tensor_info['dtype']),
                    'tensor_size': tensor_info['size']
                }
                self.trace_data.append(trace_entry)
                
                # Print trace information
                self._log(f"[TRACE] {func_name} - Shape: {tensor_info['shape']}, "
                        f"Dtype: {tensor_info['dtype']}, Size: {tensor_info['size']/1024/1024:.2f} MB, "
                        f"Duration: {duration*1000:.2f} ms")
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
    
    def apply_hooks(self):
        for func_name, orig_func in self.hooked_functions.items():
            if hasattr(dist, func_name):
                self.original_functions[func_name] = getattr(dist, func_name)
                setattr(dist, func_name, self._trace_wrapper(func_name, orig_func))
                self._log(f"Applyed hook to function: {func_name}")
    
    def remove_hooks(self):
        for func_name, orig_func in self.original_functions.items():
            if hasattr(dist, func_name):
                setattr(dist, func_name, orig_func)
                self._log(f"Removed hook from function: {func_name}")
    
    def get_trace_data(self):
        return self.trace_data
    
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
