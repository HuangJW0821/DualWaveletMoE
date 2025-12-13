import json
import os
import random
from typing import Dict, List, Any, Optional
from datasets import load_from_disk
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from .dwt_tokenizer import DWTTokenizer
from transformers.data.data_collator import DataCollatorMixin
from functools import lru_cache
from datasets.utils.logging import disable_progress_bar
disable_progress_bar()

class TimeSeriesSingleDataset(Dataset):
    """
    Dataset class for a single HF dataset.
    """
    def __init__(self, ds_path: str):
        self.ds = load_from_disk(ds_path)
        self.ds.set_format("torch")
    
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):
        """
        Get sample from dataset, each sample is a time-series group. \n
        A group refers to a se of related time series and may refer to different things depending on forecasting task. \n
        For example, a group may consist of: \n
        - **univariate ts**: a signle ts corresponding to univariate forecasting task.
        - **related ts**: a set of univariate ts with shared source or metadata, forecast could be promote through in context learning.
        - **multivaraite ts**: a set of variates with shared dynamics.
        - **covariate-informed ts**: ts forecasting with poast-only covariate.

        Returns:
         sample:
         - **id**: sample id.
         - **task_id**: subset id, use to identify subset info.
         - **data**: tensor shape *[group_size, seq_len]*, *seq_len* default to 4096.
         - **loss_mask**: tensor shape *[seq_len]*, loss mask of the group
        """
        sample = self.ds[idx]

        return sample

def load_dataset_lru(path: str):
    ds = load_from_disk(path)
    ds.set_format("torch")
    return ds

class TimeSeriesMultipleDataset(Dataset):
    """
    Dataset class for multiple datasets, applicable for large scale training scenarios. \n
    Datasets must be organized as follows, each subset must be individual HF dataset: \n
    ```
    root_path
    ├ dataset_1
    │ ├ subset_1
    │ │ ├ train-000-of-xxx.arrow
    │ │ ├ train-001-of-xxx.arrow
    │ │ └ ...
    │ ├ subset_2
    │ └ ...
    ├ dataset_2
    └ ...
    ```
    During sampling, choose a dataset by lottery sampling (if weights are provided),
    then randomly choose a subset, and randomly sample it.
    """
    def __init__(
            self, 
            root_path: str, 
            dataset_names: List[str] = None, 
            dataset_weights: List[int] = None, 
            seed: int = 42,
            MAX_CACHED_DATASETS = 128
        ):
        self.rng = random.Random(seed)
        self.cache = lru_cache(maxsize=MAX_CACHED_DATASETS)(load_dataset_lru)

        if dataset_names is None:
            dataset_names = os.listdir(root_path)
        self.dataset_names = dataset_names
        
        if dataset_weights is None:
            dataset_weights = [1.0] * len(self.dataset_names)
        elif len(self.dataset_names) != len(dataset_weights):
            raise ValueError(f"Only {len(dataset_weights)} weights given, but there are {len(self.dataset_names)} datasets in {self.dataset_names}")
        self.dataset_weights = dataset_weights

        # index meta data
        self.dataset_to_subsets = {}    # dataset -> list of subsets path
        self.dataset_sizes = {}         # dataset -> total sample number of a dataset
        self.subsets_cache = {}         # path -> loaded subset (HF dataset instance)

        # generate meta data
        print(f"Loading datasets: {dataset_names}")
        for dataset_name in dataset_names:
            dataset_path = os.path.join(root_path, dataset_name)
            if not os.path.isdir(dataset_path):
                continue

            metadata_file = os.path.join(dataset_path, "waveletmoe_dataset_metadata.json")
            if os.path.exists(metadata_file):
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)
                    subset_path_list = metadata["subset_path_list"]
                    total_size = metadata["total_size"]
            
            else:
                subset_path_list = []
                subset_sizes = {}
                total_size = 0

                for subset_name in tqdm(os.listdir(dataset_path), desc=f"Loading [{dataset_name}]"):
                    subset_path = os.path.join(dataset_path, subset_name)
                    if not os.path.isdir(subset_path):
                        continue

                    # load subset metadata
                    subset_path_list.append(subset_path)

                    subset = load_from_disk(subset_path)
                    subset_len = len(subset)
                    total_size += subset_len
                    subset_sizes[subset_name] = subset_len
                
                metadata = {"subset_path_list": subset_path_list, "total_size": total_size, "subset_sizes": subset_sizes}
                with open(metadata_file, "w") as f:
                    json.dump(metadata, f, indent=3)

            if subset_path_list and len(subset_path_list) > 0:
                self.dataset_to_subsets[dataset_name] = subset_path_list
                self.dataset_sizes[dataset_name] = total_size
        
        # prepare sampling probs
        self.dataset_probs = dataset_weights / np.sum(dataset_weights)
        
        # compute total dataste length
        self.total_size = sum(self.dataset_sizes.values())

        print(f"TimeSeriesMultipleDataset object inited, total size [{self.total_size}] examples.")
    
    def __len__(self):
        return self.total_size
    
    def __getitem__(self, idx):
        # neglect idx, randomly load sample from datasets
        dataset_name = self.rng.choices(self.dataset_names, weights=self.dataset_probs, k=1)[0]
        subset_path = self.rng.choice(self.dataset_to_subsets[dataset_name])
        dataset = self.cache(subset_path)
        local_idx = self.rng.randint(0, len(dataset) - 1)
        return dataset[local_idx]

class TimeSeriesWindowedSingleDataset(Dataset):
    """
    Dataset class for single dataset, applicable for evaluation. \n
    Datasets must be a local HF dataset: \n
    ```
    dataset_1
    ├ train-000-of-xxx.arrow
    ├ train-001-of-xxx.arrow
    └ ...
    ```
    During itering, this class would iter through the dataset & return windows.
    """
    def __init__(
            self,
            dataset_path: str, 
            window_length: int, 
            stride: Optional[int] = None,
            drop_remainder: bool = True,
            min_window_length: int = 1
        ):
        self.dataset_path = dataset_path
        self.window_length = window_length
        self.stride = stride if stride is not None else window_length
        self.drop_remainder = drop_remainder
        self.min_window_length = min_window_length

        if self.window_length <= 0:
            raise ValueError(f"Args \'window_length\' must larger than 0, not [{window_length}].")
        if self.stride <= 0:
            raise ValueError(f"Args \'stride\' must larger than 0, not [{stride}].")
        if self.min_window_length <= 0:
            raise ValueError(f"Args \'min_window_length\' must larger than 0, not [{min_window_length}].")

        self.dataset = load_from_disk(dataset_path)
        self.dataset.set_format("torch")

        self.dataset_name = os.path.basename(dataset_path)

        self._precompute_window_info()

        self._get_target_idx()
    
    def _precompute_window_info(self):
        self.window_indices = []

        for sample_idx in range(len(self.dataset)):
            input_ids = self.dataset[sample_idx]["data"]
            seq_length = input_ids.shape[1]

            if seq_length < self.window_length:
                if not self.drop_remainder and seq_length >= self.min_window_length:
                    num_windows = 1
                else:
                    num_windows = 0
            
            else:
                if self.drop_remainder:
                    num_windows = (seq_length - self.window_length) // self.stride + 1
                else:
                    num_windows = (seq_length - self.window_length + self.stride - 1) // self.stride + 1
            
            for window_idx in range(num_windows):
                start_pos = window_idx * self.stride
                end_pos = start_pos + self.window_length

                # for last window
                if end_pos > seq_length:
                    if not self.drop_remainder and (seq_length - start_pos) >= self.min_window_length:
                        end_pos = seq_length
                    else:
                        continue
                
                self.window_indices.append((sample_idx, start_pos, end_pos)) 
    
    def _get_target_idx(self):
        task_info_path = os.path.join(self.dataset_path, "task_info.json")
        
        if os.path.exists(task_info_path):
            with open(task_info_path, 'r', encoding='utf-8') as f:
                task_info = json.load(f)
            task_info = list(task_info.values())[0]
            self.target_start_idx = task_info['target_start_idx']
            self.target_end_idx = task_info['target_end_idx']

            # # if target is filtered during preprocess, choose first covariate as target
            # if task_info['target_end_idx'] == task_info['target_start_idx']:
            #     self.target_end_idx = self.target_start_idx + 1
        else:
            self.target_start_idx = 0
            self.target_end_idx = len(self.dataset[0]['data'])

    def __len__(self):
        return len(self.window_indices)
    
    def __getitem__(self, idx):
        if idx >= len(self):
            raise ValueError(f"Index [{idx}] of of range, max idx [{len(self.window_indices)-1}].")

        sample_idx, start_pos, end_pos = self.window_indices[idx]
        original_sample = self.dataset[sample_idx]

        window_sample = {}
        for key, value in original_sample.items():
            if key == "data":
                window_sample[key] = value[:, start_pos : end_pos]
            elif key == "loss_mask":
                window_sample[key] = value[start_pos : end_pos]
            else:
                window_sample[key] = value
        
        window_sample["window_info"] = {
            "original_sample_idx": sample_idx,
            "window_idx": idx,
            "start_pos": start_pos,
            "window_length": len(window_sample)
        }

        return window_sample
    
    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]