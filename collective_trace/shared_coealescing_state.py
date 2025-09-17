"""
all_gather ... && _coalescing_manager 's share
"""
from typing import Dict, Optional


class CoalescingState:
    def __init__(self):
        self.active_cm_id: Optional[int] = None  # current active context manager id
        self.counter: Dict[int, int] = {}  # suche as {cm_id: counter}

# Since we are using a global state, we need to make sure it is only initialized once
coalescing_state = CoalescingState()