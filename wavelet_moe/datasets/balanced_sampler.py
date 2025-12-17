#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
子集均衡采样器
类似 Chronos 的 MultiDatasetBalancedSampler，确保各个数据集均匀采样
"""

import random
from typing import List, Optional

# torch 是可选的，只有在实际使用时才需要
try:
    from torch.utils.data import Sampler, BatchSampler
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    # 定义一个简单的 BatchSampler 基类（用于类型提示）
    class BatchSampler:
        pass


class DistributedBatchSampler(BatchSampler):
    """Wrap a (possibly infinite) BatchSampler and shard batches across DDP ranks.

    accelerate/Trainer 对自定义 batch_sampler 通常不会自动分布式 shard，
    这里用 batch_id % world_size 的方式保证各 rank 拿到不同 batch。
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
    多数据集平衡采样器
    
    确保每个 batch 从各个数据集中均匀采样，而不是简单随机采样。
    支持无限生成（while True），直到 Trainer 的 max_steps 停止。
    """
    
    def __init__(
        self, 
        dataset: 'TimeMoEWindowDataset',
        batch_size: int,
        shuffle: bool = True,
        seed: Optional[int] = None
    ):
        """
        参数：
            dataset: TimeMoEWindowDataset 实例（需要支持 subset_id）
            batch_size: 批次大小
            shuffle: 是否在每个 epoch 开始时打乱
            seed: 随机种子
        """
        self.dataset = dataset
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        
                # 兼容两种 dataset 形态：
        # 1) 直接窗口数据集：dataset.window_list 的下标 == dataset.__getitem__ 的 idx
        # 2) train/test wrapper：dataset.indices 映射到 base_ds.window_list（dataset.__getitem__ 会先映射）
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

        # 子集数 / 子集名：优先用 wrapper 上透传的，其次用 base_ds 的
        num_subsets = getattr(dataset, 'num_subsets', getattr(base_ds, 'num_subsets', 1))
        if num_subsets == 0:
            raise ValueError('No subsets found in dataset. Check dataset loading logic.')

        subset_names = getattr(dataset, 'subset_names', getattr(base_ds, 'subset_names', None))
        if subset_names is None:
            subset_names = [f'subset_{i}' for i in range(num_subsets)]

        # 按子集分组窗口索引（注意：保存的是 *DataLoader 可用的 idx*）
        self.subset_window_indices = [[] for _ in range(num_subsets)]

        if index_map is None:
            # 直接窗口数据集：window_idx 就是 dataloader idx
            for window_idx, window_info in enumerate(window_list):
                subset_id = window_info.get('subset_id', None)
                if subset_id is None:
                    raise ValueError(f"Window {window_idx} missing 'subset_id'. Check window generation logic.")
                if subset_id < 0 or subset_id >= num_subsets:
                    raise ValueError(f"Window {window_idx} has invalid subset_id={subset_id}, expected range [0, {num_subsets})")
                self.subset_window_indices[subset_id].append(window_idx)
        else:
            # wrapper：local_idx 才是 dataloader idx，subset_id 需要从 base window_list[base_idx] 拿
            for local_idx, base_window_idx in enumerate(index_map):
                window_info = window_list[base_window_idx]
                subset_id = window_info.get('subset_id', None)
                if subset_id is None:
                    raise ValueError(f"Base window {base_window_idx} missing 'subset_id'. Check window generation logic.")
                if subset_id < 0 or subset_id >= num_subsets:
                    raise ValueError(f"Base window {base_window_idx} has invalid subset_id={subset_id}, expected range [0, {num_subsets})")
                self.subset_window_indices[subset_id].append(local_idx)

        # 统计信息
        self.num_subsets = num_subsets
        self.total_windows = total_windows

        # 打印子集统计信息（rank0 打印即可；这里保持简洁，不做分布式判断）
        print('')
        print('MultiDatasetBalancedSampler initialized:')
        print(f'  Total subsets: {self.num_subsets}')
        print(f'  Total windows: {self.total_windows}')
        print(f'  Batch size: {self.batch_size}')
        print('  Sampling: uniform (all subsets equal probability)')
        print('  Window distribution by subset:')
        for subset_idx, indices in enumerate(self.subset_window_indices):
            subset_name = subset_names[subset_idx] if subset_idx < len(subset_names) else f'subset_{subset_idx}'
            print(f'    {subset_name}: {len(indices)} windows')

# 初始化随机数生成器
        self._rng = random.Random(seed) if seed is not None else random
    
    def __iter__(self):
        """
        无限生成索引序列，确保每个 batch 从各个数据集均匀采样
        
        类似 Chronos 的无限循环生成方式，直到 Trainer 的 max_steps 停止训练
        """
        # 为每个子集创建索引池（打乱）
        subset_indices_pools = []
        for subset_idx, indices in enumerate(self.subset_window_indices):
            if len(indices) == 0:
                subset_indices_pools.append([])
                continue
            pool = indices.copy()
            if self.shuffle:
                self._rng.shuffle(pool)
            subset_indices_pools.append(pool)
        
        # 为每个子集维护循环索引
        subset_pointers = [0] * self.num_subsets
        
        # 计算有效的子集数量
        num_valid_subsets = sum(1 for pool in subset_indices_pools if len(pool) > 0)
        if num_valid_subsets == 0:
            return iter([])
        
        # 检查是否为"随机选择子集"模式
        use_random_subset_mode = self.batch_size < num_valid_subsets
        
        # 只有在非随机模式下才需要计算 batch_allocation（延迟计算）
        batch_allocation = None
        if not use_random_subset_mode:
            batch_allocation = self._allocate_batch_samples()
        
        # 无限生成 batch
        def infinite_batch_generator():
            while True:
                batch_indices = []
                
                if use_random_subset_mode:
                    # 模式1：batch_size < 子集数，随机选择 batch_size 个子集（均匀）
                    # 每个选中的子集只取一个样本，确保 batch 中每个子集最多出现一次
                    valid_subset_indices = [s_idx for s_idx in range(self.num_subsets) 
                                           if len(subset_indices_pools[s_idx]) > 0]
                    if len(valid_subset_indices) == 0:
                        break
                    
                    # 无放回选择：随机选择 batch_size 个不同的子集，每个子集只取一个样本
                    # 由于 use_random_subset_mode 保证 batch_size < num_valid_subsets，
                    # 而 len(valid_subset_indices) == num_valid_subsets，所以一定有足够的子集可选
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
                    # 模式2：batch_size >= 子集数，使用固定分配
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
        """计算每个 batch 中，每个子集应该贡献的样本数（均匀分配）
        
        注意：余数（extra）会随机分配给所有子集，而不是按顺序分配给前几个子集。
        这样可以确保每个 epoch 开始时，余数的分配是随机的，更加公平。
        """
        allocation = []
        
        # 过滤出非空子集
        valid_subsets = []
        for subset_idx in range(self.num_subsets):
            if len(self.subset_window_indices[subset_idx]) > 0:
                valid_subsets.append(subset_idx)
        
        if len(valid_subsets) == 0:
            return [0] * self.num_subsets
        
        # 均匀分配（所有子集等概率）
        # 注意：此方法只在 batch_size >= len(valid_subsets) 时被调用
        # 如果 batch_size < len(valid_subsets)，会使用随机选择模式，不调用此方法
        allocation_map = {}
        
        # batch_size >= 子集数：每个子集至少1个，剩余按均匀分配
        base_per_subset = self.batch_size // len(valid_subsets)
        extra = self.batch_size % len(valid_subsets)
        
        # 先给所有子集分配基础数量
        for subset_idx in valid_subsets:
            allocation_map[subset_idx] = base_per_subset
        
        # 将余数随机分配给所有子集（无放回，确保每个子集最多多分1个）
        if extra > 0:
            selected_for_extra = self._rng.sample(valid_subsets, k=extra)
            for subset_idx in selected_for_extra:
                allocation_map[subset_idx] += 1
        
        # 验证分配总和等于 batch_size
        total_allocated = sum(allocation_map.values())
        if total_allocated != self.batch_size:
            print(f'Warning: Batch allocation mismatch: {total_allocated} != {self.batch_size}')
        
        # 构建完整的分配列表
        for subset_idx in range(self.num_subsets):
            allocation.append(allocation_map.get(subset_idx, 0))
        
        return allocation
    
    def __len__(self):
        """返回总索引数（理论上无限）"""
        if self.total_windows > 0:
            return self.total_windows * 100  # 返回一个很大的数，表示"无限"
        else:
            return 0

