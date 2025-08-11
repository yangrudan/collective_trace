"""
Group management module for distributed operations.

This module provides functionality to create and manage groups of processes
in a distributed computing environment, with utilities to track group IDs
and handle group creation exceptions.
"""

import os
from typing import Optional, Tuple, List

try:
    import torch
    import torch.distributed as dist
except ImportError:
    pass


class GroupState:
    """Manages state for process group tracking, including caches and counters."""

    def __init__(self):
        self.group_ranks_cache = {}
        self.group_id_counter = 0
        self.group_id_index_map = {}

    def get_group_index(self, group_id: int) -> int:
        """Retrieve the index of a group by its ID if it exists."""
        return self.group_id_index_map.get(group_id)

    def cache_group_ranks(self, group_id: int, ranks: List[int]) -> None:
        """Cache the ranks of a group."""
        self.group_ranks_cache[group_id] = ranks

    def update_counter_and_map(self, group_id: int) -> int:
        """Increment counter and update group index map, returning the new index."""
        self.group_id_counter += 1
        self.group_id_index_map[group_id] = self.group_id_counter
        return self.group_id_counter


group_state = GroupState()


def get_participating_ranks(
    group: Optional[dist.ProcessGroup] = None,
) -> Tuple[int, int, int, List[int]]:
    """
    Get participating ranks in a given process group

    Args:
        group (Optional[dist.ProcessGroup], optional):
        The process group to retrieve ranks from. Defaults to None.

    Returns:
        Tuple[int, int, int, List[int]]:
        group_rank, group_size, index of group, and list of participating ranks
    """
    if not dist.is_initialized():
        return 0, 0, 0, []

    group_rank = dist.get_rank(group=group)
    group_size = dist.get_world_size(group=group)

    if group is None or group == dist.group.WORLD:
        return group_rank, group_size, 0, list(range(dist.get_world_size()))

    group_id = id(group)

    if (
        group_id in group_state.group_ranks_cache
        and group_state.get_group_index(group_id) is not None
    ):
        return (
            group_rank,
            group_size,
            group_state.get_group_index(group_id),
            group_state.group_ranks_cache[group_id],
        )

    # Method 1: Use all_gather_object to collect all ranks
    try:
        ranks_list = [None] * group_size
        global_rank = dist.get_rank()
        dist.all_gather_object(ranks_list, global_rank, group=group)
        ranks = [int(r) for r in ranks_list]

        group_state.group_id_counter += 1
        group_state.cache_group_ranks(group_id, ranks)
        new_index = group_state.update_counter_and_map(group_id)
        return group_rank, group_size, new_index, ranks

    except (RuntimeError, ValueError, TypeError) as e:
        print(
            f"[Rank {dist.get_rank()}] all_gather_object failed: {e}. Using fallback method."
        )

    # Method 2: Use TCPStore to collect all ranks
    try:
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        store = dist.TCPStore(
            host_name=os.environ["MASTER_ADDR"],
            port=int(os.environ["MASTER_PORT"]),
            world_size=world_size,
            is_master=(rank == 0),
            timeout=torch.timedelta(seconds=30),
        )

        store_key = f"rank_in_group_{group_id}"

        store_key = f"rank_in_group_{group_id}"
        store.set(store_key, str(rank))

        # If rank is 0, collect all ranks from the store
        if rank == 0:
            ranks = []
            for _ in range(group_size):
                r = int(store.get(store_key).decode())
                ranks.append(r)
            ranks_tensor = torch.tensor(ranks, dtype=torch.int32)
        else:
            ranks_tensor = torch.zeros(group_size, dtype=torch.int32)

        # Broadcast the ranks_tensor to all ranks in the group
        dist.broadcast(ranks_tensor, src=0, group=group)
        ranks = ranks_tensor.tolist()

        # Clean up the store
        if rank == 0:
            store.delete_key(store_key)

        group_state.cache_group_ranks(group_id, ranks)
        new_index = group_state.update_counter_and_map(group_id)
        return group_rank, group_size, new_index, ranks

    except RuntimeError as e:
        print(f"Rank {rank} failed to create group (RuntimeError): {str(e)}")
        raise
    except ValueError as e:
        print(f"Rank {rank} invalid group parameters (ValueError): {str(e)}")
        raise
