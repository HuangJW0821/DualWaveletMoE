#!/usr/bin/env python
# -*- coding:utf-8 _*-
import random
from typing import List, Optional
from torch.utils.data import BatchSampler

from wavelet_moe.utils.log_util import log_in_local_rank_0
from wavelet_moe.datasets.dataset_time300B import WaveletMoeWindowDataset


class DistributedBatchSampler(BatchSampler):
    """
    Wrap a (possibly infinite) BatchSampler and shard batches across DDP ranks. \n
    Use `batch_id % world_size` to assure different rank get different batch when train distributively.
    """

    def __init__(self, batch_sampler: BatchSampler, num_replicas: int, rank: int):
        self.batch_sampler = batch_sampler
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)

    def __iter__(self):
        for i, batch in enumerate(self.batch_sampler):
            if (i % self.num_replicas) == self.rank:
                yield batch

    def __len__(self):
        try:
            base_len = len(self.batch_sampler)
            return max(0, base_len // self.num_replicas)
        except Exception:
            return 0

    @staticmethod
    def maybe_wrap(batch_sampler: BatchSampler) -> BatchSampler:
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                return DistributedBatchSampler(
                    batch_sampler=batch_sampler,
                    num_replicas=dist.get_world_size(),
                    rank=dist.get_rank(),
                )
        except Exception:
            pass
        return batch_sampler


class MultiDatasetBalancedSampler(BatchSampler):
    """
    Balanced sampler across multiple datasets. \n
    Uniform sample acorss datasets to assure example balance.
    Supports unlimited generation (`while True`) unil trainer's `max_steps`.
    """
    
    def __init__(
        self, 
        dataset: WaveletMoeWindowDataset,
        batch_size: int,
        shuffle: bool = True,
        seed: Optional[int] = None
    ):
        """
        Args: 
            dataset: `WaveletMoeWindowDataset` (should support `subset_id`)
        """
        self.dataset = dataset
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        
        # Compatible two kinds of dataset class:
        # 1) Windowed dataset: where index of `dataset.window_list ==` idx of `dataset.__getitem__`
        # 2) train/test wrapper: map `dataset.indices`` to `base_ds.window_list`(map `dataset.__getitem__` first)
        base_ds = getattr(dataset, 'base_ds', None)
        is_wrapper = hasattr(dataset, 'indices') and base_ds is not None and hasattr(base_ds, 'window_list')

        if is_wrapper:
            window_list = base_ds.window_list
            index_map = [int(i) for i in dataset.indices]  # local_idx -> base_window_idx
            total_windows = len(index_map)
        else:
            if not hasattr(dataset, 'window_list') or len(dataset.window_list) == 0:
                raise ValueError("Dataset must have window_list with subset_id information (or base_ds.window_list for wrappers)")
            window_list = dataset.window_list
            index_map = None
            total_windows = len(window_list)

        # prefer using what is passed though wrapper first, and then use base_ds
        num_subsets = getattr(dataset, 'num_subsets', getattr(base_ds, 'num_subsets', 1))
        if num_subsets == 0:
            raise ValueError('No subsets found in dataset. Check dataset loading logic.')

        subset_names = getattr(dataset, 'subset_names', getattr(base_ds, 'subset_names', None))
        if subset_names is None:
            subset_names = [f'subset_{i}' for i in range(num_subsets)]

        # index by subset window
        self.subset_window_indices = [[] for _ in range(num_subsets)]

        if index_map is None:
            # Windowed dataset: window_idx == dataloader idx
            for window_idx, window_info in enumerate(window_list):
                subset_id = window_info.get('subset_id', None)
                if subset_id is None:
                    raise ValueError(f"Window {window_idx} missing 'subset_id'. Check window generation logic.")
                if subset_id < 0 or subset_id >= num_subsets:
                    raise ValueError(f"Window {window_idx} has invalid subset_id={subset_id}, expected range [0, {num_subsets})")
                self.subset_window_indices[subset_id].append(window_idx)
        else:
            # wrapper：local_idx == dataloader idx，get subset_id from base window_list[base_idx]
            for local_idx, base_window_idx in enumerate(index_map):
                window_info = window_list[base_window_idx]
                subset_id = window_info.get('subset_id', None)
                if subset_id is None:
                    raise ValueError(f"Base window {base_window_idx} missing 'subset_id'. Check window generation logic.")
                if subset_id < 0 or subset_id >= num_subsets:
                    raise ValueError(f"Base window {base_window_idx} has invalid subset_id={subset_id}, expected range [0, {num_subsets})")
                self.subset_window_indices[subset_id].append(local_idx)

        self.num_subsets = num_subsets
        self.total_windows = total_windows

        log_in_local_rank_0('MultiDatasetBalancedSampler initialized:')
        log_in_local_rank_0(f'  Total subsets: {self.num_subsets}')
        log_in_local_rank_0(f'  Total windows: {self.total_windows}')
        log_in_local_rank_0(f'  Batch size: {self.batch_size}')
        log_in_local_rank_0('  Sampling: uniform (all subsets equal probability)')
        log_in_local_rank_0('  Window distribution by subset:')
        for subset_idx, indices in enumerate(self.subset_window_indices):
            subset_name = subset_names[subset_idx] if subset_idx < len(subset_names) else f'subset_{subset_idx}'
            log_in_local_rank_0(f'    {subset_name}: {len(indices)} windows')

        self._rng = random.Random(seed) if seed is not None else random
    
    def __iter__(self):
        """
        Generate an infinite index sequence to ensure each batch is sampled uniformly from all datasets.
        """

        # creat index pools for every subst (shuffle)
        subset_indices_pools = []
        for subset_idx, indices in enumerate(self.subset_window_indices):
            if len(indices) == 0:
                subset_indices_pools.append([])
                continue
            pool = indices.copy()
            if self.shuffle:
                self._rng.shuffle(pool)
            subset_indices_pools.append(pool)
        
        # maintain circular indices for each subset
        subset_pointers = [0] * self.num_subsets
        
        # get valid ubset num
        num_valid_subsets = sum(1 for pool in subset_indices_pools if len(pool) > 0)
        if num_valid_subsets == 0:
            return iter([])
        
        use_random_subset_mode = self.batch_size < num_valid_subsets
        
        batch_allocation = None
        if not use_random_subset_mode:
            batch_allocation = self._allocate_batch_samples()
        
        # gen batch
        def infinite_batch_generator():
            while True:
                batch_indices = []
                
                if use_random_subset_mode:
                    # mode 1: batch_size < subset num, then choose batch_size subsets randomly
                    # sample one example each subset, assure that one subset appear once in each batch
                    valid_subset_indices = [s_idx for s_idx in range(self.num_subsets) 
                                           if len(subset_indices_pools[s_idx]) > 0]
                    if len(valid_subset_indices) == 0:
                        break
                    
                    # sample without replacement
                    selected_subsets = self._rng.sample(valid_subset_indices, k=self.batch_size)
                    
                    for subset_idx in selected_subsets:
                        pool = subset_indices_pools[subset_idx]
                        pool_size = len(pool)
                        if pool_size == 0:
                            continue
                        idx_in_pool = subset_pointers[subset_idx] % pool_size
                        window_idx = pool[idx_in_pool]
                        batch_indices.append(window_idx)
                        subset_pointers[subset_idx] += 1
                        if subset_pointers[subset_idx] % pool_size == 0 and self.shuffle and subset_pointers[subset_idx] > 0:
                            self._rng.shuffle(pool)
                else:
                    # mode 2: batch_size >= subset num
                    for subset_idx, count in enumerate(batch_allocation):
                        if len(subset_indices_pools[subset_idx]) == 0:
                            continue
                        pool = subset_indices_pools[subset_idx]
                        pool_size = len(pool)
                        for _ in range(count):
                            idx_in_pool = subset_pointers[subset_idx] % pool_size
                            window_idx = pool[idx_in_pool]
                            batch_indices.append(window_idx)
                            subset_pointers[subset_idx] += 1
                            if subset_pointers[subset_idx] % pool_size == 0 and self.shuffle and subset_pointers[subset_idx] > 0:
                                self._rng.shuffle(pool)
                
                if len(batch_indices) > 0:
                    if self.shuffle:
                        self._rng.shuffle(batch_indices)
                    yield batch_indices
                else:
                    break
        
        return infinite_batch_generator()
    
    def _allocate_batch_samples(self) -> List[int]:
        """
        Calculate the number of samples each subset should contribute in each batch (balance sample)
        
        **Note:** the remainder (`extra`) will randomly distributed among all subsets, rather than assigned to the first few subsets in order.
        This ensures that the distribution of `extra` is more random and farier at the begining of each epoch.
        """
        allocation = []
        
        # filter non empty subset
        valid_subsets = []
        for subset_idx in range(self.num_subsets):
            if len(self.subset_window_indices[subset_idx]) > 0:
                valid_subsets.append(subset_idx)
        
        if len(valid_subsets) == 0:
            return [0] * self.num_subsets
        
        # assign equal probable for all subsets
        allocation_map = {}
        
        # batch_size >= subset num: assign extra randomly
        base_per_subset = self.batch_size // len(valid_subsets)
        extra = self.batch_size % len(valid_subsets)
        
        for subset_idx in valid_subsets:
            allocation_map[subset_idx] = base_per_subset
        
        if extra > 0:
            selected_for_extra = self._rng.sample(valid_subsets, k=extra)
            for subset_idx in selected_for_extra:
                allocation_map[subset_idx] += 1
        
        total_allocated = sum(allocation_map.values())
        if total_allocated != self.batch_size:
            print(f'Warning: Batch allocation mismatch: {total_allocated} != {self.batch_size}')
        
        # build allocation list
        for subset_idx in range(self.num_subsets):
            allocation.append(allocation_map.get(subset_idx, 0))
        
        return allocation
    
    def __len__(self):
        if self.total_windows > 0:
            return self.total_windows * 100  # inf
        else:
            return 0

