__ALL__ = ["trace_all_collectives"]

import torch
import torch.distributed as dist
from typing import List, Optional, Union, Tuple

from .trace_collectives import CollectiveTracer

def trace_all_collectives(trace_file=None, verbose=False):
    """Fast API to trace all collective operations in PyTorch."""
    tracer = CollectiveTracer(trace_file=trace_file, verbose=verbose)
    tracer.apply_hooks()
    return tracer
