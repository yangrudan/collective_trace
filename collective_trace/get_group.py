from . import torch
from . import dist
from . import Optional, Union, Tuple, List

# Store the group ranks for each function in a dictionary
GROUP_RANKS_CACHE = {}

"""
This function returns a list of participating ranks within a given process group."""
def get_participating_ranks(group: Optional[dist.ProcessGroup] = None) ->  Tuple[int, int, int, List[int]]:
    if not dist.is_initialized():
        return 0, 0, []

    group_rank = dist.get_rank(group=group)
    group_size = dist.get_world_size(group=group)

    if group is None or group == dist.group.WORLD:
        return group_rank, group_size, 0, list(range(dist.get_world_size()))
    
    group_id = id(group)
    
    if group_id in GROUP_RANKS_CACHE:
        return group_rank, group_size, group_id, GROUP_RANKS_CACHE[group_id]

    
    # Method 1: Use all_gather_object to collect all ranks
    try:
        ranks_list = [None] * group_size
        global_rank = dist.get_rank()
        dist.all_gather_object(ranks_list, global_rank, group=group)
        ranks = [int(r) for r in ranks_list]
        GROUP_RANKS_CACHE[group_id] = ranks
        return group_rank, group_size, group_id, ranks
    
    except Exception as e:
        print(f"[Rank {dist.get_rank()}] all_gather_object failed: {e}. Using fallback method.")
    
    # Method 2: Use TCPStore to collect all ranks
    try:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        
        import os
        store = dist.TCPStore(
            host_name=os.environ['MASTER_ADDR'],
            port=int(os.environ['MASTER_PORT']),
            world_size=world_size,
            is_master=(rank == 0),
            timeout=torch.timedelta(seconds=30)
        )
        
        store_key = f'rank_in_group_{group_id}'
        store.set(store_key, str(rank))
        
        # If rank is 0, collect all ranks from the store
        if rank == 0:
            ranks = []
            for i in range(group_size):
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
        
        GROUP_RANKS_CACHE[group_id] = ranks
        return group_rank, group_size, group_id, ranks
    
    except Exception as e:
        print(f"[Rank {rank}] Failed to get ranks via TCPStore: {e}")
        # If all methods fail, return a list of all ranks in the group
        return group_rank, group_size, 0, [dist.get_rank() for _ in range(group_size)]
    