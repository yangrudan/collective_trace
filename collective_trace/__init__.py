"""
CollectiveTracer Module

Support distributed training with PyTorch. 
This module provides a fast API to trace all collective operations in PyTorch.
"""

__ALL__ = ["trace_all_collectives"]

from .trace_collectives import CollectiveTracer


def trace_all_collectives(trace_file=None, verbose=False):
    """Fast API to trace all collective operations in PyTorch."""
    tracer = CollectiveTracer(trace_file=trace_file, verbose=verbose)
    tracer.apply_hooks()
    return tracer
