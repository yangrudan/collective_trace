"""
all_gather ... && _coalescing_manager 's share
"""

from typing import Dict, Optional


class CoalescingState:
    """
    Shared state for coalescing manager tracing"""

    def __init__(self):
        """
        Initialize the coalescing state"""
        self.active_cm_id: Optional[int] = None  # current active context manager id
        self.counter: Dict[int, int] = {}  # suche as {cm_id: counter}
        self.names: Dict[int, str] = {}  # such as {cm_id: name}
        self.sizes: Dict[int, int] = {}  # such as {cm_id: size}

    def reset(self):
        """
        Reset the coalescing state"""
        self.active_cm_id = None
        self.counter = {}
        self.names = {}


# Since we are using a global state, we need to make sure it is only initialized once
coalescing_state = CoalescingState()
