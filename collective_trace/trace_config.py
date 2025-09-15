"""
Data classes for tracing configuration and group state
"""
from dataclasses import dataclass
from .async_logger import RankedAsyncLogger

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
