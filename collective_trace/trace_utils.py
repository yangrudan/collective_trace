"""
Utility functions for tracing PyTorch operations.
"""

try:
    import torch
except ImportError:
    pass


def cuda_sync():
    """Synchronize CUDA devices."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def extract_tensor_info(args, kwargs):
    """Extract tensor information from arguments."""
    tensor = None

    # Check positional arguments
    for arg in args:
        if isinstance(arg, torch.Tensor):
            tensor = arg
            break

    # Check keyword arguments
    if tensor is None:
        for _, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                tensor = value
                break

    # Check attributes of first argument
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
