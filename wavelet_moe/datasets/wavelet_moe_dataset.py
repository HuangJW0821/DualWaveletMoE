import os
import numpy as np
import pickle
import hashlib
import json
from pathlib import Path
from typing import Optional, Dict, Any
import warnings
from torch.utils.data import Dataset

from wavelet_moe.datasets.time_series_dataset import TimeSeriesDataset, GeneralDataset, BinaryDataset


def binary_search(sorted_list, value):
    low = 0
    high = len(sorted_list) - 1
    best_index = -1

    while low <= high:
        mid = (low + high) // 2
        if sorted_list[mid] <= value:
            best_index = mid
            low = mid + 1
        else:
            high = mid - 1

    return best_index


class WaveletMoeMultipleDataset(TimeSeriesDataset):
    """
    This class is only responsible for dataset list management. \n
    Dataset class for multiple datasets, applicable for large scale training scenarios, especially Time-300B. \n
    Datasets must be organized as follows, each subset must be individual TimeSeriesDataset. \n
    ```
    root_path
    ├ domain_1
    │ ├ time_series_dataset_1
    │ │ ├ data-000-of-xxx.bin
    │ │ ├ ...
    │ │ └ meta.json
    │ ├ time_series_dataset_2
    │ └ ...
    ├ domain_2
    └ ...
    ```
    """

    def __init__(
        self, 
        root_path: str, 
        dataset_cache_path: Optional[str] = None,
        use_dataset_cache: bool = True
    ):
        self.root_path = root_path

        self.datasets = []
        self.num_tokens = None
        self.use_dataset_cache = use_dataset_cache
        self.dataset_cache_path = Path(dataset_cache_path) if dataset_cache_path else None

        # try to load cached datasets info
        cached_data, cache_path = None, None
        if use_dataset_cache:
            cache_path = self._get_dataset_cache_path()
            if cache_path and cache_path.exists():
                cached_data = self._load_dataset_from_cache(cache_path)

        if cached_data is not None:
            # successfully loaded from cache
            print(f'Loaded datasets list from cache, total {len(cached_data["dataset_paths"])} datasets')
            
            # reload every dataset, it might be time-consuming while running
            for ds_path in cached_data['dataset_paths']:
                try:
                    if BinaryDataset.is_valid_path(ds_path):
                        ds = BinaryDataset(ds_path)
                    elif GeneralDataset.is_valid_path(ds_path):
                        ds = GeneralDataset(ds_path)
                    else:
                        continue

                    if len(ds) > 0:
                        self.datasets.append(ds)
                
                except Exception as e:
                    warnings.warn(f" ⚠ Failed to load dataset [{ds_path}] due to error: {e}", RuntimeWarning)
            
            print(f"Totally {len(self.datasets)} datasets loaded for training.")
        
        # no cache data are provided, scan through root_path & log datasets info
        else:
            if BinaryDataset.is_valid_path(self.root_path):
                print(f"Loading BinaryDataset: {self.root_path}")
                ds = BinaryDataset(self.root_path)
                if len(ds) > 0:
                    self.datasets.append(ds)
                    print(f"  ✓ [{len(ds)}] sequences are loaded.")
            elif GeneralDataset.is_valid_path(self.root_path):
                print(f"Loading GeneralDataset: {self.root_path}")
                ds = GeneralDataset(self.root_path)
                if len(ds) > 0:
                    self.datasets.append(ds)
                    print(f"  ✓ [{len(ds)}] sequences are loaded.")
            else:
                # walk through the root_path
                print(f"Scanning: {self.root_path}")
                binary_count = 0
                general_count = 0
                dataset_paths = []  # load dataset paths for cache

                for root, dirs, files in os.walk(self.root_path):
                    # skip hidden files & folders
                    dirs[:] = [d for d in dirs if not d.startswith('.')]

                    for file in files:
                        if file.startswith('.') or file in ['repo_metadata.json', 'README.md']:
                            continue

                        fn_path = os.path.join(root, file)
                        if file != BinaryDataset.meta_file_name and GeneralDataset.is_valid_path(fn_path):
                            try:
                                ds = GeneralDataset(fn_path)
                                if len(ds) > 0:
                                    self.datasets.append(ds)
                                    dataset_paths.append(fn_path)
                                    general_count += 1
                                    rel_path = os.path.relpath(fn_path, self.root_path)
                                    print(f'  No.{general_count} GeneralDataset [{rel_path}]: [{len(ds)}] sequences loaded')
                            except Exception as e:
                                rel_path = os.path.relpath(fn_path, self.root_path)
                                warnings.warn(f" ⚠ Failed to load dataset [{rel_path}] due to error: {e}", RuntimeWarning)
                    
                    for sub_folder in dirs:
                        folder_path = os.path.join(root, sub_folder)
                        if BinaryDataset.is_valid_path(folder_path):
                            try:
                                ds = BinaryDataset(folder_path)
                                if len(ds) > 0:
                                    self.datasets.append(ds)
                                    dataset_paths.append(folder_path)
                                    binary_count += 1
                                    rel_path = os.path.relpath(folder_path, self.root_path)
                                    print(f'  No.{binary_count} BinaryDataset [{rel_path}]: [{len(ds)}] sequences loaded')
                            except Exception as e:
                                rel_path = os.path.relpath(folder_path, self.root_path)
                                warnings.warn(f" ⚠ Failed to load dataset [{rel_path}] due to error: {e}", RuntimeWarning)

                print(f'\n End scanning: total [{binary_count}] BinaryDataset and [{general_count}] GeneralDataset were found.')

                if use_dataset_cache and cache_path:
                    self._save_dataset_to_cache(cache_path, dataset_paths)

        self.cumsum_lengths = [0]
        for ds in self.datasets:
            self.cumsum_lengths.append(self.cumsum_lengths[-1] + len(ds))
        self.num_sequences = self.cumsum_lengths[-1]

        print(f'\nFinshed datasets loading: total [{len(self.datasets)}] datasets ([{self.num_sequences:,}] sequences) were loaded.')

    def _get_dataset_cache_path(self) -> Optional[Path]:
        if not self.use_dataset_cache:
            return None

        if not self.dataset_cache_path:
            dataset_cache_path = Path(os.getcwd()) / '.cache' / 'wavelet_moe_datasets'
        else:
            dataset_cache_path = self.dataset_cache_path / 'datasets'

        # generate cache key
        cache_key = self._compute_dataset_cache_key()
        cache_path = dataset_cache_path / f'{cache_key}'
        return cache_path

    def _compute_dataset_cache_key(self) -> str:
        params = {'root_path': str(self.root_path)}
        key_str = json.dumps(params, sort_keys=True)
        cache_key = hashlib.md5(key_str.encode()).hexdigest()[:16]
        return cache_key

    def _load_dataset_from_cache(self, cache_path: Path) -> Optional[Dict[str, Any]]:
        try:
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)

            if 'dataset_paths' not in cached_data:
                return None

            if cached_data.get('root_path') != str(self.root_path):
                warnings.warn(f"Cache datasets\' root_path mismatch, dataset class will scan root_path again.", RuntimeWarning)
                return None

            return cached_data
        except Exception as e:
            warnings.warn(f"Error while loading dataset cache: {e}", RuntimeWarning)
            return None

    def _save_dataset_to_cache(self, cache_path: Path, dataset_paths: list):
        try:
            dataset_cache_path = cache_path.parent
            dataset_cache_path.mkdir(parents=True, exist_ok=True)

            cached_data = {
                'root_path': str(self.root_path),
                'dataset_paths': dataset_paths,
                'num_datasets': len(dataset_paths),
            }

            # use tmp file to assure write-time atomicity
            temp_path = cache_path.with_suffix('.tmp')
            with open(temp_path, 'wb') as f:
                pickle.dump(cached_data, f, protocol=pickle.HIGHEST_PROTOCOL)

            # atomicity rename
            temp_path.replace(cache_path)
            print(f'✓ Datasets list save at [{cache_path}] ( total [{len(dataset_paths)}] datasets)')

        except Exception as e:
            warnings.warn(f"Error while saving cache: [{e}]", RuntimeWarning)

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, seq_idx):
        if not (0 <= seq_idx and seq_idx < self.cumsum_lengths[-1]):
            raise IndexError(f"Index out of range while getitem from dataset: idx [{seq_idx}] are provided but length [{self.cumsum_lengths[-1]}].")

        dataset_idx = binary_search(self.cumsum_lengths, seq_idx)
        dataset_offset = seq_idx - self.cumsum_lengths[dataset_idx]
        seq = self.datasets[dataset_idx][dataset_offset]
        return seq

    def get_sequence_length_by_idx(self, seq_idx):
        if not (0 <= seq_idx and seq_idx < self.cumsum_lengths[-1]):
            raise IndexError(f"Index out of range while getitem from dataset: idx [{seq_idx}] are provided but length [{self.cumsum_lengths[-1]}].")

        dataset_idx = binary_search(self.cumsum_lengths, seq_idx)
        dataset_offset = seq_idx - self.cumsum_lengths[dataset_idx]
        return self.datasets[dataset_idx].get_sequence_length_by_idx(dataset_offset)

    def get_num_tokens(self):
        if self.num_tokens is None:
            self.num_tokens = sum([ds.get_num_tokens() for ds in self.datasets])

        return self.num_tokens


# TODO: clean rebundent annotation
class WaveletMoeWindowDataset:
    """
    A dataset class for generating non-overlapping sliding windows from a time series dataset.
    Includes quality checks and proper padding with loss masks.

    Attributes:
        dataset (`TimeSeriesDataset`): The underlying time series dataset.
        context_length (`int`): Length of the input context window.
        prediction_length (`int`): Length of the prediction window. Defaults to `0`.
        window_size (`int`): Total size of the sliding window (`context_length + prediction_length`).
        stride (`int`): Step size for sliding the window. Defaults to `window_size`.
        sub_seq_indexes (`list`): List of tuples containing sequence indices and their corresponding offsets.

    **Methods**:
        `__len__()`:
            Returns the total number of sliding windows in the dataset.
        `__iter__()`:
            Iterates over the dataset, yielding one sliding window at a time.
        `__getitem__(seq_idx)`:
            Retrieves a sliding window with input_ids and loss_masks.

    **Example**:
    ```
        dataset = TimeSeriesDataset(...)
        context_length, prediction_length = 10, 5
        window_dataset = TimeMoEWindowDataset(dataset, context_length, prediction_length)
        for sample in window_dataset:
            print(sample['input_ids'], sample['loss_masks'])
    ```
    """

    def __init__(
            self, 
            dataset: TimeSeriesDataset, 
            context_length: int=4096, 
            prediction_length: int = 0, 
            stride: int = None,
            lazy: bool = False, 
            dataset_cache_path: Optional[str] = None, 
            use_dataset_cache: bool = True, 
            **kwargs
        ):
        """
        Args:
         dataset (`TimeSeriesDataset`): The underlying time series dataset.
         context_length (`int`): Length of the input context window.
         prediction_length (`int`): Total size of the sliding window (`context_length + prediction_length`).
         stride: (`int`): Step size for sliding the window. Defaults to `window_size`.
         lazy: 是否使用 Lazy 模式（默认False）
            - False (非 Lazy): 预计算所有窗口索引（所有可能的 offset）
                * 初始化：遍历所有序列，预计算所有窗口索引，存储到 sub_seq_indexes
                * 访问：直接索引 sub_seq_indexes[window_idx]，O(1) 访问
                * 窗口数量：准确（所有可能的窗口，可能包含无效窗口）
                * 内存：中等（存储所有窗口索引）
            - True (Lazy): 只记录序列元数据，访问时动态生成窗口
                * 初始化：只扫描序列元数据，估算窗口数量，生成估算的 window_list
                * 访问：二分查找定位序列，动态生成窗口并做质量检查，O(log n) + 动态生成
                * 窗口数量：估算值（实际可能因质量检查而减少）
                * 内存：小（只有元数据和估算的 window_list）
         dataset_cache_path: 缓存目录（默认 None，使用临时目录）
         use_dataset_cache: 是否使用缓存（默认 True）
        """
        self.dataset = dataset
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.window_size = context_length + prediction_length
        self.stride = stride if stride else self.window_size
        self.lazy = lazy
        self.use_dataset_cache = use_dataset_cache
        self.dataset_cache_path = Path(dataset_cache_path) if dataset_cache_path else Path(
            os.path.join(os.path.dirname(__file__), '.cache', 'wavelet_moe_windows'))

        # 质量检查阈值（参考 Chronos）
        self.zero_threshold = 0.2

        # 获取数据集信息（用于记录 subset_id）
        self._init_dataset_info()

        # 尝试从缓存加载
        cached_data = None
        cache_path = None
        if use_dataset_cache:
            cache_path = self._get_cache_path()
            print("缓存路径为:", cache_path)
            if cache_path and cache_path.exists():

                cached_data = self._load_from_cache(cache_path)

        if cached_data is not None:
            # 从缓存加载成功
            cached_lazy = cached_data.get('lazy', False)
            cache_loaded = False

            if cached_lazy and self.lazy:
                # lazy 模式的缓存：只包含元数据，不包含完整窗口列表
                if 'sequence_metadata' in cached_data and 'window_cumsum' in cached_data and 'window_list' in cached_data:
                    # 验证 subset_names 是否匹配（数据集结构可能变化）
                    cached_subset_names = cached_data.get('subset_names', [])
                    if cached_subset_names != self.subset_names:
                        print(f'警告: 缓存中的子集名称不匹配，将重新生成')
                        print(f'  缓存: {cached_subset_names[:5]}...' if len(
                            cached_subset_names) > 5 else f'  缓存: {cached_subset_names}')
                        print(f'  当前: {self.subset_names[:5]}...' if len(
                            self.subset_names) > 5 else f'  当前: {self.subset_names}')
                        cached_data = None  # 强制重新生成
                    else:
                        self.sequence_metadata = cached_data['sequence_metadata']
                        self.window_cumsum = cached_data['window_cumsum']
                        # lazy 模式下，window_list 只包含轻量级元数据（用于采样器）
                        self.window_list = cached_data.get('window_list', [])
                        self.sub_seq_indexes = []  # lazy 模式下不生成 sub_seq_indexes
                        print(
                            f'\n✓ 从缓存加载 lazy 模式元数据: {len(self.sequence_metadata)} 个序列，{self.window_cumsum[-1]:,} 个窗口（估算）')
                        cache_loaded = True
                else:
                    missing_fields = []
                    if 'sequence_metadata' not in cached_data:
                        missing_fields.append('sequence_metadata')
                    if 'window_cumsum' not in cached_data:
                        missing_fields.append('window_cumsum')
                    if 'window_list' not in cached_data:
                        missing_fields.append('window_list')
                    print(f'警告: lazy 缓存缺少必要字段 ({", ".join(missing_fields)})，将重新生成')
                    cached_data = None  # 强制重新生成
            elif cached_data is not None:
                # 非 lazy 模式的缓存：包含完整窗口列表
                # 验证 subset_names 是否匹配（数据集结构可能变化）
                cached_subset_names = cached_data.get('subset_names', [])
                if cached_subset_names != self.subset_names:
                    print(f'警告: 缓存中的子集名称不匹配，将重新生成')
                    print(f'  缓存: {cached_subset_names[:5]}...' if len(
                        cached_subset_names) > 5 else f'  缓存: {cached_subset_names}')
                    print(f'  当前: {self.subset_names[:5]}...' if len(
                        self.subset_names) > 5 else f'  当前: {self.subset_names}')
                    cached_data = None  # 强制重新生成
                else:
                    import time
                    start_time = time.time()
                    print(f'  正在构建窗口索引列表...')
                    self.window_list = cached_data['window_list']
                    # 优先使用缓存的 sub_seq_indexes（如果存在），否则从 window_list 构建
                    if 'sub_seq_indexes' in cached_data and cached_data['sub_seq_indexes'] is not None:
                        self.sub_seq_indexes = cached_data['sub_seq_indexes']
                    else:
                        self.sub_seq_indexes = [(w['seq_idx'], w['offset']) for w in self.window_list]
                    build_time = time.time() - start_time
                    print(f'\n✓ 从缓存加载窗口列表: {len(self.window_list):,} 个窗口')
                    if build_time > 0.1:
                        print(f'  索引构建耗时: {build_time:.1f}秒')

                    # 如果缓存是非 lazy 的，但当前要求 lazy，则禁用 lazy（因为缓存包含完整信息）
                    if self.lazy and not cached_lazy:
                        print(f'  注意: 缓存包含完整窗口信息，lazy 模式自动禁用')
                        self.lazy = False

                    # 确保 window_cumsum 存在（非 lazy 模式下，window_cumsum 应该等于 window_list 的长度）
                    if not hasattr(self, 'window_cumsum'):
                        # 非 lazy 模式：window_cumsum 应该等于 window_list 的长度（每个窗口对应一个累积值）
                        self.window_cumsum = list(range(len(self.window_list) + 1))

                    cache_loaded = True

            # 统计各子集的窗口数（只有在成功加载缓存时才执行）
            # if cache_loaded and hasattr(self, 'num_subsets') and hasattr(self, 'window_list') and len(
            #         self.window_list) > 0:
            #     subset_window_counts = {}
            #     for w in self.window_list:
            #         subset_id = w.get('subset_id', 0)
            #         subset_window_counts[subset_id] = subset_window_counts.get(subset_id, 0) + 1
            #     print(f'  窗口分布（按子集）:')
            #     for subset_id in sorted(subset_window_counts.keys()):
            #         subset_name = self.subset_names[subset_id] if subset_id < len(
            #             self.subset_names) else f'subset_{subset_id}'
            #         print(f'    {subset_name}: {subset_window_counts[subset_id]:,} 个窗口')

        # 如果缓存加载失败或没有缓存，需要重新生成
        if cached_data is None:
            print(f'  未找到缓存，开始生成窗口列表...')
            # 需要生成窗口
            print(f'\n生成窗口列表...')
            if not lazy:
                # 一次性生成所有窗口索引（原始方式）
                self._generate_all_windows()
            else:
                # 真正的 Lazy 模式：只记录序列元数据，不加载数据，不生成窗口
                self._init_lazy_windows()

            # 保存到缓存
            if use_dataset_cache and cache_path:
                self._save_to_cache(cache_path)

            # 打印窗口生成统计信息
            self._print_window_statistics()

        # 确保 window_cumsum 存在（用于 lazy 模式和 __len__ 方法）
        if not hasattr(self, 'window_cumsum'):
            if self.lazy and hasattr(self, 'sequence_metadata'):
                # lazy 模式下，从 sequence_metadata 计算（应该已经在 _init_lazy_windows 中初始化）
                self.window_cumsum = [0]
                for meta in self.sequence_metadata:
                    self.window_cumsum.append(self.window_cumsum[-1] + meta['num_windows'])
            else:
                # 非 lazy 模式或从缓存加载，使用 window_list 的长度
                self.window_cumsum = [0]
                num_windows = len(self.window_list) if hasattr(self, 'window_list') else len(
                    self.sub_seq_indexes) if hasattr(self, 'sub_seq_indexes') else 0
                for _ in range(num_windows):
                    self.window_cumsum.append(self.window_cumsum[-1] + 1)

    def _init_dataset_info(self):
        """初始化数据集信息，用于记录 subset_id"""
        # 获取子数据集信息
        if hasattr(self.dataset, 'datasets') and hasattr(self.dataset, 'cumsum_lengths'):
            # TimeMoEDataset 结构
            self.num_subsets = len(self.dataset.datasets)
            self.subset_map = {}  # seq_idx -> subset_id
            self.subset_names = []

            for subset_id, ds in enumerate(self.dataset.datasets):
                # 获取数据集名称
                if hasattr(ds, 'data_path'):
                    name = os.path.basename(ds.data_path)
                else:
                    name = f'subset_{subset_id}'
                self.subset_names.append(name)

                # 记录每个序列属于哪个子集
                start_idx = self.dataset.cumsum_lengths[subset_id]
                end_idx = self.dataset.cumsum_lengths[subset_id + 1]
                for seq_idx in range(start_idx, end_idx):
                    self.subset_map[seq_idx] = subset_id
        else:
            # 单个数据集
            self.num_subsets = 1
            self.subset_map = {i: 0 for i in range(len(self.dataset))}
            self.subset_names = ['single_dataset']

    def _build_window_list(self):
        """构建 window_list，包含 subset_id 信息（用于平衡采样器）"""
        self.window_list = []
        for window_idx, (seq_idx, offset) in enumerate(self.sub_seq_indexes):
            subset_id = self.subset_map.get(seq_idx, 0)
            self.window_list.append({
                'window_idx': window_idx,
                'seq_idx': seq_idx,
                'offset': offset,
                'subset_id': subset_id
            })

    def _generate_all_windows(self):
        """非 Lazy 模式：预计算所有窗口索引，并在生成时做质量检查（类似 ChronosDataset）"""
        print(f'  模式: 非 Lazy（预计算所有窗口，包含质量检查）')
        num_seqs = len(self.dataset)
        print(f'  总序列数: {num_seqs:,}')
        iterator = range(num_seqs)
        try:
            from tqdm import tqdm
            iterator = tqdm(iterator, total=num_seqs,
                            desc='Pre-computing window indexes with quality check (non-lazy mode)')
        except ImportError:
            pass

        self.sub_seq_indexes = []
        skipped_sequences = 0
        dropped_windows = 0
        processed_sequences = 0
        last_print_idx = 0
        print_interval = 10000  # 每处理 10000 个序列打印一次

        import time
        start_time = time.time()

        for seq_idx in iterator:
            try:
                # 获取序列长度
                n_points = self.dataset.get_sequence_length_by_idx(seq_idx)
            except (KeyError, IndexError, TypeError, ValueError, Exception) as e:
                skipped_sequences += 1
                if skipped_sequences <= 10:
                    print(f'⚠ 警告: 跳过序列 {seq_idx}: {type(e).__name__} - {e}')
                continue

            # 保留至少 1 个时间步的序列
            if n_points < 1:
                continue

            # 如果序列太短（小于窗口长度），加载数据并做质量检查
            if n_points < self.window_size:
                try:
                    # 加载短序列数据并做质量检查
                    win = self.dataset[seq_idx][0: n_points]
                    win = np.array(win, dtype=np.float32)
                    if len(win) > 0 and self._window_is_valid(win):
                        self.sub_seq_indexes.append((seq_idx, 0))
                    else:
                        dropped_windows += 1
                except Exception:
                    dropped_windows += 1
                continue

            # 序列长度足够，生成窗口并做质量检查
            max_offset = n_points - self.window_size
            for offset in range(0, max_offset + 1, self.stride):
                try:
                    # 加载窗口数据并做质量检查
                    win = self.dataset[seq_idx][offset: offset + self.window_size]
                    win = np.array(win, dtype=np.float32)
                    if len(win) == self.window_size and self._window_is_valid(win):
                        self.sub_seq_indexes.append((seq_idx, offset))
                    else:
                        dropped_windows += 1
                except Exception:
                    dropped_windows += 1
                    continue

            # 末尾补窗（只有当末尾窗口不在 stride 循环中时才添加，避免重复）
            tail_start = n_points - self.window_size
            if tail_start > 0 and tail_start % self.stride != 0:
                # 检查是否已存在（避免重复，虽然理论上不应该存在）
                if len(self.sub_seq_indexes) == 0 or self.sub_seq_indexes[-1] != (seq_idx, tail_start):
                    try:
                        # 加载末尾窗口数据并做质量检查
                        win = self.dataset[seq_idx][tail_start: tail_start + self.window_size]
                        win = np.array(win, dtype=np.float32)
                        if len(win) == self.window_size and self._window_is_valid(win):
                            self.sub_seq_indexes.append((seq_idx, tail_start))
                        else:
                            dropped_windows += 1
                    except Exception:
                        dropped_windows += 1

            processed_sequences += 1
            # 定期打印进度信息（每 10000 条）
            if processed_sequences - last_print_idx >= print_interval:
                elapsed_time = time.time() - start_time
                progress = processed_sequences / num_seqs
                if progress > 0:
                    estimated_total_time = elapsed_time / progress
                    remaining_time = estimated_total_time - elapsed_time
                    # 格式化剩余时间
                    if remaining_time < 60:
                        time_str = f'{remaining_time:.1f}秒'
                    elif remaining_time < 3600:
                        time_str = f'{remaining_time / 60:.1f}分钟'
                    else:
                        hours = int(remaining_time // 3600)
                        minutes = int((remaining_time % 3600) // 60)
                        time_str = f'{hours}小时{minutes}分钟'
                else:
                    time_str = '计算中...'

                print(f'  进度: {processed_sequences:,}/{num_seqs:,} ({100 * progress:.1f}%) | '
                      f'有效窗口: {len(self.sub_seq_indexes):,} | '
                      f'过滤: {dropped_windows:,} | '
                      f'剩余时间: {time_str}')
                last_print_idx = processed_sequences

        if skipped_sequences > 0:
            print(f'  跳过 {skipped_sequences} 个无法访问的序列')

        if len(self.sub_seq_indexes) == 0:
            print(f'  ⚠ 警告: 没有生成任何窗口索引！数据集可能为空或所有序列都被跳过。')
        else:
            print(f'  预计算完成: {len(self.sub_seq_indexes):,} 个有效窗口（已过滤 {dropped_windows} 个无效窗口）')

        # 构建 window_list（用于采样器）
        self._build_window_list()

    def _init_lazy_windows(self):
        """真正的 Lazy 模式：只记录序列元数据，不加载数据，不生成窗口"""
        print(f'  模式: Lazy（只记录元数据，访问时生成窗口）')
        # 记录每个序列的元数据（序列索引、长度、所属子集）
        # 不加载实际数据，不生成窗口，不做质量检查
        self.sequence_metadata = []
        num_seqs = len(self.dataset)
        print(f'  总序列数: {num_seqs:,}')

        iterator = range(num_seqs)
        try:
            from tqdm import tqdm
            iterator = tqdm(iterator, total=num_seqs, desc='Scanning sequences (lazy mode, no data loading)')
        except ImportError:
            pass

        skipped_sequences = 0
        processed_sequences = 0
        last_print_idx = 0
        print_interval = 10000  # 每处理 10000 个序列打印一次

        import time
        start_time = time.time()

        for seq_idx in iterator:
            try:
                # 只获取序列长度（不加载数据）
                n_points = self.dataset.get_sequence_length_by_idx(seq_idx)
                subset_id = self.subset_map.get(seq_idx, 0)

                # 跳过长度为 0 的序列
                if n_points < 1:
                    continue

                # 估算该序列可能产生的窗口数量（不实际生成）
                if n_points >= self.window_size:
                    # 估算窗口数（不考虑质量检查，因为还没加载数据）
                    num_windows = (n_points - self.window_size) // self.stride + 1
                    # 可能还有末尾窗口
                    tail_start = n_points - self.window_size
                    if tail_start > 0 and tail_start % self.stride != 0:
                        num_windows += 1
                elif n_points >= 1:
                    num_windows = 1  # 短序列至少一个窗口
                else:
                    num_windows = 0

                if num_windows > 0:
                    self.sequence_metadata.append({
                        'seq_idx': seq_idx,
                        'length': n_points,
                        'subset_id': subset_id,
                        'num_windows': num_windows  # 估算值，实际可能因质量检查而减少
                    })
            except Exception as e:
                skipped_sequences += 1
                if skipped_sequences <= 10:
                    print(f'⚠ 警告: 跳过序列 {seq_idx}: {type(e).__name__} - {e}')
                continue

            processed_sequences += 1
            # 定期打印进度信息（每 10000 条）
            if processed_sequences - last_print_idx >= print_interval:
                elapsed_time = time.time() - start_time
                progress = processed_sequences / num_seqs
                if progress > 0:
                    estimated_total_time = elapsed_time / progress
                    remaining_time = estimated_total_time - elapsed_time
                    # 格式化剩余时间
                    if remaining_time < 60:
                        time_str = f'{remaining_time:.1f}秒'
                    elif remaining_time < 3600:
                        time_str = f'{remaining_time / 60:.1f}分钟'
                    else:
                        hours = int(remaining_time // 3600)
                        minutes = int((remaining_time % 3600) // 60)
                        time_str = f'{hours}小时{minutes}分钟'
                else:
                    time_str = '计算中...'

                print(f'  进度: {processed_sequences:,}/{num_seqs:,} ({100 * progress:.1f}%) | '
                      f'有效序列: {len(self.sequence_metadata):,} | '
                      f'剩余时间: {time_str}')
                last_print_idx = processed_sequences

        # 预计算窗口累积索引（用于快速定位）
        self.window_cumsum = [0]
        for meta in self.sequence_metadata:
            self.window_cumsum.append(self.window_cumsum[-1] + meta['num_windows'])

        # 为了支持均衡采样器，生成轻量级的 window_list（只包含估算的元数据，不加载数据）
        # 这些元数据用于采样器分组，实际访问时会重新生成并做质量检查
        print(f'  生成估算窗口列表（用于采样器）...')
        self.window_list = []
        self.sub_seq_indexes = []  # Lazy 模式下为空

        window_idx = 0
        for meta in self.sequence_metadata:
            seq_idx = meta['seq_idx']
            subset_id = meta['subset_id']
            n_points = meta['length']

            # 生成估算的窗口元数据（不加载数据，不做质量检查）
            if n_points >= self.window_size:
                for offset in range(0, n_points - self.window_size + 1, self.stride):
                    self.window_list.append({
                        'window_idx': window_idx,
                        'seq_idx': seq_idx,
                        'offset': offset,
                        'subset_id': subset_id,
                        'is_estimated': True  # 标记为估算值，访问时需要验证
                    })
                    window_idx += 1
                # 末尾窗口
                tail_start = n_points - self.window_size
                if tail_start > 0 and tail_start % self.stride != 0:
                    self.window_list.append({
                        'window_idx': window_idx,
                        'seq_idx': seq_idx,
                        'offset': tail_start,
                        'subset_id': subset_id,
                        'is_estimated': True
                    })
                    window_idx += 1
            elif n_points >= 1:
                self.window_list.append({
                    'window_idx': window_idx,
                    'seq_idx': seq_idx,
                    'offset': 0,
                    'subset_id': subset_id,
                    'is_estimated': True
                })
                window_idx += 1

        if skipped_sequences > 0:
            print(f'  跳过了 {skipped_sequences} 个无法访问的序列')

        if len(self.sequence_metadata) == 0:
            print(f'  ⚠ 警告: 没有找到任何有效序列！数据集可能为空或所有序列都被跳过。')
        elif len(self.window_list) == 0:
            print(f'  ⚠ 警告: 没有生成任何窗口！数据集可能为空或所有序列都太短。')
        else:
            print(
                f'  ✓ Lazy 模式初始化完成: {len(self.sequence_metadata):,} 个序列，{len(self.window_list):,} 个估算窗口（未加载数据，未做质量检查）')

    def _generate_window_for_seq(self, seq_idx: int, n_points: int) -> list:
        """为单个序列生成所有窗口索引（流式加载时使用）"""
        windows = []

        if n_points < 1:
            return windows

        if n_points < self.window_size:
            # 短序列也需要做质量检查
            try:
                win = self.dataset[seq_idx][0: n_points]
                win = np.array(win, dtype=np.float32)
                if len(win) > 0 and self._window_is_valid(win):
                    windows.append((seq_idx, 0))
            except Exception:
                pass
            return windows

        max_offset = n_points - self.window_size
        for offset in range(0, max_offset + 1, self.stride):
            try:
                win = self.dataset[seq_idx][offset: offset + self.window_size]
                win = np.array(win, dtype=np.float32)
                if len(win) == self.window_size and self._window_is_valid(win):
                    windows.append((seq_idx, offset))
            except Exception:
                continue

        # 末尾补窗（只有当末尾窗口不在 stride 循环中时才添加，避免重复）
        tail_start = n_points - self.window_size
        if tail_start > 0 and tail_start % self.stride != 0:
            try:
                win = self.dataset[seq_idx][tail_start: tail_start + self.window_size]
                win = np.array(win, dtype=np.float32)
                if len(win) == self.window_size and self._window_is_valid(win):
                    # 检查是否已存在（虽然理论上不应该存在）
                    if not windows or windows[-1] != (seq_idx, tail_start):
                        windows.append((seq_idx, tail_start))
            except Exception:
                pass

        return windows

    def _find_sequence_for_window(self, window_idx: int) -> int:
        """使用二分查找找到 window_idx 对应的序列索引"""
        # 在 window_cumsum 中查找
        low, high = 0, len(self.window_cumsum) - 1
        while low < high:
            mid = (low + high + 1) // 2
            if self.window_cumsum[mid] <= window_idx:
                low = mid
            else:
                high = mid - 1
        return low if low < len(self.sequence_metadata) else None

    def _print_window_statistics(self):
        """打印窗口生成统计信息"""
        if not hasattr(self, 'window_list') or len(self.window_list) == 0:
            return

        print(f'\n窗口生成完成:')
        print(f'  总窗口数: {len(self.window_list):,}')

        if hasattr(self, 'num_subsets'):
            # 统计各子集的窗口数
            subset_window_counts = {}
            for w in self.window_list:
                subset_id = w.get('subset_id', 0)
                subset_window_counts[subset_id] = subset_window_counts.get(subset_id, 0) + 1

            print(f'  子集数量: {self.num_subsets}')
            print(f'  窗口分布（按子集）:')
            for subset_id in sorted(subset_window_counts.keys()):
                subset_name = self.subset_names[subset_id] if subset_id < len(
                    self.subset_names) else f'subset_{subset_id}'
                count = subset_window_counts[subset_id]
                percentage = 100.0 * count / len(self.window_list)
                print(f'    {subset_name}: {count:,} 个窗口 ({percentage:.2f}%)')

    def _get_cache_path(self) -> Optional[Path]:
        """生成缓存文件路径"""
        if not self.use_dataset_cache:
            return None

        # self.dataset_cache_path 在 __init__ 中已经初始化为 Path 对象（不会是 None）
        dataset_cache_path = self.dataset_cache_path

        # 生成缓存键（基于数据集路径、窗口参数等）
        cache_key = self._compute_cache_key()
        cache_path = dataset_cache_path / f'{cache_key}'#f'1f041c1173a00a70'

        return cache_path

    def _compute_cache_key(self) -> str:
        """计算缓存键（基于数据集和窗口参数）"""
        # 获取数据集路径信息
        dataset_info = []
        if hasattr(self.dataset, 'data_folder'):
            dataset_info.append(str(self.dataset.data_folder))
        elif hasattr(self.dataset, 'datasets'):
            # TimeMoEDataset
            for ds in self.dataset.datasets:
                if hasattr(ds, 'data_path'):
                    dataset_info.append(str(ds.data_path))

        # 组合参数
        params = {
            'dataset': '|'.join(dataset_info),
            'context_length': self.context_length,
            'prediction_length': self.prediction_length,
            'window_size': self.window_size,
            'stride': self.stride,
            'lazy': self.lazy,
            'zero_threshold': self.zero_threshold,
        }

        # 生成哈希
        key_str = json.dumps(params, sort_keys=True)
        cache_key = hashlib.md5(key_str.encode()).hexdigest()[:16]
        return cache_key

    def _load_from_cache(self, cache_path: Path) -> Optional[Dict[str, Any]]:
        """从缓存文件加载窗口列表"""
        import time
        start_time = time.time()

        try:
            print(f'  正在加载缓存文件: {cache_path.name}...')
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)

            load_time = time.time() - start_time

            # 验证缓存数据
            if 'window_list' not in cached_data:
                return None

            if len(cached_data['window_list']) == 0:
                print('警告: 缓存文件包含空窗口列表，将重新生成')
                return None

            # 验证参数一致性（所有影响窗口生成的参数）
            if cached_data.get('context_length') != self.context_length:
                print(
                    f'警告: 缓存参数不匹配 (context_length: {cached_data.get("context_length")} != {self.context_length})，将重新生成')
                return None

            if cached_data.get('prediction_length') != self.prediction_length:
                print(
                    f'警告: 缓存参数不匹配 (prediction_length: {cached_data.get("prediction_length")} != {self.prediction_length})，将重新生成')
                return None

            if cached_data.get('stride') != self.stride:
                print(f'警告: 缓存参数不匹配 (stride: {cached_data.get("stride")} != {self.stride})，将重新生成')
                return None
            '''
            if cached_data.get('zero_threshold') != self.zero_threshold:
                print(cached_data.get('zero_threshold'))
                print(self.zero_threshold)
                print(
                    f'警告: 缓存参数不匹配 (zero_threshold: {cached_data.get("zero_threshold")} != {self.zero_threshold})，将重新生成')
                return None
'''
            # 显示加载时间
            if load_time < 1:
                print(f'  ✓ 缓存加载完成: {load_time * 1000:.0f}毫秒')
            elif load_time < 60:
                print(f'  ✓ 缓存加载完成: {load_time:.1f}秒')
            else:
                minutes = int(load_time // 60)
                seconds = int(load_time % 60)
                print(f'  ✓ 缓存加载完成: {minutes}分{seconds}秒')

            return cached_data
        except Exception as e:
            print(f'缓存加载错误: {e}')
            return None

    def _save_to_cache(self, cache_path: Path):
        """保存窗口列表到缓存文件"""
        if not hasattr(self, 'window_list') or len(self.window_list) == 0:
            return

        try:
            dataset_cache_path = cache_path.parent
            dataset_cache_path.mkdir(parents=True, exist_ok=True)

            # 保存窗口列表和相关信息
            cached_data = {
                'context_length': self.context_length,
                'prediction_length': self.prediction_length,
                'window_size': self.window_size,
                'stride': self.stride,
                'lazy': self.lazy,
                'zero_threshold': self.zero_threshold,  # 质量检查阈值
                'num_subsets': self.num_subsets if hasattr(self, 'num_subsets') else 1,
                'subset_names': self.subset_names if hasattr(self, 'subset_names') else [],
            }

            if self.lazy:
                # lazy 模式：只保存元数据（轻量级）
                cached_data.update({
                    'sequence_metadata': self.sequence_metadata if hasattr(self, 'sequence_metadata') else [],
                    'window_cumsum': self.window_cumsum if hasattr(self, 'window_cumsum') else [0],
                    'window_list': self.window_list,  # 轻量级元数据列表（用于采样器）
                    'num_windows': self.window_cumsum[-1] if hasattr(self, 'window_cumsum') and len(
                        self.window_cumsum) > 0 else 0,
                })
            else:
                # 非 lazy 模式：保存完整窗口列表
                cached_data.update({
                    'window_list': self.window_list,
                    'sub_seq_indexes': self.sub_seq_indexes if hasattr(self, 'sub_seq_indexes') else None,
                    'num_windows': len(self.window_list),
                })

            # 使用临时文件，确保原子性写入
            temp_path = cache_path.with_suffix('.tmp')
            with open(temp_path, 'wb') as f:
                pickle.dump(cached_data, f, protocol=pickle.HIGHEST_PROTOCOL)

            # 原子性重命名
            temp_path.replace(cache_path)
            print(f'✓ 窗口列表已缓存: {cache_path} ({len(self.window_list)} 个窗口)')

        except Exception as e:
            print(f'警告: 缓存保存失败 ({e})，但不影响功能')

    def _window_is_valid(self, seq_1d: np.ndarray) -> bool:
        """判断窗口质量是否合格：
        - NaN/Inf 不允许
        - 值为 0 的比例 <= 阈值
        - 一阶/二阶差分为 0 的比例 <= 阈值self.zero_threshold =0.2
        """
        if not isinstance(seq_1d, np.ndarray):
            seq_1d = np.asarray(seq_1d)
        if seq_1d.ndim != 1:
            seq_1d = seq_1d.reshape(-1)
        n = len(seq_1d)
        if n == 0:
            return False
        # NaN/Inf 检查
        if np.isnan(seq_1d).any():
            return False
        if np.isinf(seq_1d).any():
            return False
        # 数值 0 比例
        zero_ratio = float((seq_1d == 0).sum()) / float(n)
        if zero_ratio > 0.2:
            return False
        # 一阶差分
        if n >= 2:
            d1 = seq_1d[1:] - seq_1d[:-1]
            d1_zero_ratio = float((d1 == 0).sum()) / float(len(d1))
            if d1_zero_ratio > 0.2:
                return False
        # 二阶差分
        if n >= 3:
            d2 = seq_1d[2:] - seq_1d[:-2]
            d2_zero_ratio = float((d2 == 0).sum()) / float(len(d2))
            if d2_zero_ratio > 0.2:
                return False
        return True

    def __len__(self):
        if self.lazy:
            # Lazy 模式：返回估算的窗口数量（实际有效窗口数可能更少，因为质量检查）
            if hasattr(self, 'window_cumsum') and len(self.window_cumsum) > 0:
                return self.window_cumsum[-1]
            else:
                # 如果 window_cumsum 不存在，回退到 window_list 的长度
                return len(self.window_list) if hasattr(self, 'window_list') else 0
        else:
            # 非 lazy 模式，使用 sub_seq_indexes 或 window_list
            if hasattr(self, 'sub_seq_indexes') and len(self.sub_seq_indexes) > 0:
                return len(self.sub_seq_indexes)
            elif hasattr(self, 'window_list') and len(self.window_list) > 0:
                return len(self.window_list)
            else:
                return 0

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, window_idx):
        if self.lazy:
            # Lazy 模式：动态生成窗口并做质量检查
            # 使用二分查找找到对应的序列
            seq_meta_idx = self._find_sequence_for_window(window_idx)
            if seq_meta_idx is None or seq_meta_idx >= len(self.sequence_metadata):
                raise IndexError(f"Window index {window_idx} out of range")

            meta = self.sequence_metadata[seq_meta_idx]
            seq_idx = meta['seq_idx']
            n_points = meta['length']

            # 计算窗口在该序列中的相对索引
            window_offset_in_seq = window_idx - self.window_cumsum[seq_meta_idx]

            # 生成该序列的所有有效窗口（只生成一次，然后缓存）
            if not hasattr(self, '_seq_window_cache'):
                self._seq_window_cache = {}

            if seq_idx not in self._seq_window_cache:
                # 动态生成窗口并做质量检查
                self._seq_window_cache[seq_idx] = self._generate_window_for_seq(seq_idx, n_points)

            windows = self._seq_window_cache[seq_idx]

            # 如果该序列没有有效窗口（所有窗口都未通过质量检查），尝试查找下一个有效窗口
            if len(windows) == 0:
                # 递归查找下一个有效窗口（最多尝试 100 个窗口，避免无限循环）
                max_attempts = 100
                attempt = 0
                next_window_idx = window_idx + 1
                found_valid_window = False

                while attempt < max_attempts and next_window_idx < len(self):
                    next_seq_meta_idx = self._find_sequence_for_window(next_window_idx)
                    if next_seq_meta_idx is None or next_seq_meta_idx >= len(self.sequence_metadata):
                        break
                    next_meta = self.sequence_metadata[next_seq_meta_idx]
                    next_seq_idx = next_meta['seq_idx']

                    if next_seq_idx not in self._seq_window_cache:
                        self._seq_window_cache[next_seq_idx] = self._generate_window_for_seq(next_seq_idx,
                                                                                             next_meta['length'])

                    next_windows = self._seq_window_cache[next_seq_idx]
                    if len(next_windows) > 0:
                        # 找到有效窗口，使用它
                        next_window_offset = next_window_idx - self.window_cumsum[next_seq_meta_idx]
                        if next_window_offset < len(next_windows):
                            seq_i, offset_i = next_windows[next_window_offset]
                            found_valid_window = True
                            break

                    next_window_idx += 1
                    attempt += 1

                if not found_valid_window:
                    # 如果找不到有效窗口，抛出更详细的错误信息
                    raise IndexError(
                        f"Window index {window_idx} (sequence {seq_idx}) has no valid windows after quality check. "
                        f"估算窗口数: {meta['num_windows']}, 实际有效窗口数: 0. "
                        f"尝试查找后续窗口也失败（已尝试 {attempt} 个窗口）。"
                    )
            else:
                # 正常情况：使用当前序列的窗口
                if window_offset_in_seq >= len(windows):
                    raise IndexError(
                        f"Window offset {window_offset_in_seq} out of range for sequence {seq_idx} (实际有效窗口数: {len(windows)}, 估算窗口数: {meta['num_windows']})")

                seq_i, offset_i = windows[window_offset_in_seq]

            # 提取窗口（lazy 模式）
            seq = self.dataset[seq_i][offset_i: offset_i + self.window_size]
            seq = np.array(seq, dtype=np.float32)
        else:
            # 非 Lazy 模式：从预计算的索引获取（已过滤无效窗口，类似 ChronosDataset）
            seq_i, offset_i = self.sub_seq_indexes[window_idx]

            # 提取窗口
            seq = self.dataset[seq_i][offset_i: offset_i + self.window_size]
            seq = np.array(seq, dtype=np.float32)

        # 统一处理缺失值（NaN/Inf）：填充为 0，并生成 loss_mask
        is_valid = np.isfinite(seq)  # 检查是否为有限数（非 NaN/Inf）

        # 记录原始 0 值位置（在填充 NaN/Inf 之前）
        original_zero_mask = (seq == 0.0) & is_valid  # 原始值为 0 且有效的位置

        # 处理缺失值（NaN/Inf）：填充为 0
        seq = np.where(is_valid, seq, 0.0)

        # 创建 loss mask：缺失值（NaN/Inf）位置为 0，其他位置为 1
        loss_mask = is_valid.astype(np.int32)  # 有效位置为 1，缺失位置为 0

        # 追加规则：原始数值为 0 的位置也屏蔽（0 值不参与 loss 计算）
        # 注意：只屏蔽原始 0 值，不包括填充后的 0 值（NaN/Inf 填充的 0 已经被 is_valid 屏蔽了）
        loss_mask[original_zero_mask] = 0

        # Padding 到窗口长度（末尾 padding 0）
        n_pad = self.window_size - len(seq)
        if n_pad > 0:
            seq = np.pad(seq, (0, n_pad), 'constant', constant_values=0)
            loss_mask = np.pad(loss_mask, (0, n_pad), 'constant', constant_values=0)

        return {
            'input_ids': seq,  # [window_size] 维度为 [seq_length]
            'loss_masks': loss_mask  # [window_size] 维度为 [seq_length]
        }


# TODO: clean rebundent annotation
class WaveletMoeWindowTensorDataset(Dataset):
    """
    Wrapper of WaveletMoeWindowDataset, in order to split train & test (validation) set
    返回:
        {
            "input_ids": seq,       # [window_size] float32，一维时间序列
            "loss_masks": loss_mask # [window_size] 0/1 掩码
        }
    """

    def __init__(
        self,
        window_dataset: Dataset,    # why Dataset ?
        split: str = "train",
        test_size: float = 0.0001,
        seed: int = 42,
    ):
        """
        Args:
            window_dataset:
                TimeMoEWindowDataset 实例（已经是窗口级数据）。
            split:
                "train" 或 "test"。
            test_size:
                按窗口随机切分 test 比例。
            seed:
                划分用的随机种子。
        """
        if not (0.0 <= float(test_size) < 1.0):
            raise ValueError(f"test_size must be in [0, 1), got {test_size}")

        self.base_ds = window_dataset

        all_indices = np.arange(len(self.base_ds), dtype=np.int64)
        self.indices = all_indices
        rng = np.random.default_rng(seed)
        rng.shuffle(all_indices)

        split_point = int(len(all_indices) * (1.0 - float(test_size)))
        if split == "train":
            self.indices = all_indices[:split_point]
        elif split == "test":
            self.indices = all_indices[split_point:]
        else:
            raise ValueError(f"Split must choose from ['train', 'test'], not {split}!")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # 映射到底层 TimeMoEWindowDataset 的窗口索引
        win_idx = int(self.indices[idx])
        sample = self.base_ds[win_idx]

        # TimeMoEWindowDataset 返回的一维窗口

        seq = np.asarray(sample["input_ids"], dtype=np.float32)      # [T]
        loss_mask = np.asarray(
            sample.get("loss_masks", sample.get("loss_mask")),
            dtype=np.float32,
        )  # [T]

        if seq.ndim != 1:
            raise ValueError(
                f"Expect 1D sequence from TimeMoEWindowDataset, got shape {seq.shape}"
            )
        if loss_mask.shape[0] != seq.shape[0]:
            raise ValueError(
                f"loss_mask length {loss_mask.shape[0]} != seq length {seq.shape[0]}"
            )

        input_seq = seq[:, None]         # [T] -> [T, 1]

        # 给 WaveletTimeSeriesDataset 使用：
        #   - input_ids: [C, T]，这里 C=1
        #   - loss_mask: [T,] step-wise 掩码
        input_ids = input_seq.T
        step_loss_mask = loss_mask
        new_sample = {
            "input_ids": input_ids,           # [1, T]
            "loss_mask": step_loss_mask,      # [T,]
        }


        return new_sample