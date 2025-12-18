import json
import os
import random
from typing import Dict, List, Any
import torch
from tqdm import tqdm
from .dwt_tokenizer import DWTTokenizer
from transformers.data.data_collator import DataCollatorMixin
from datasets.utils.logging import disable_progress_bar
disable_progress_bar()

# def zero_scaler(batch: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
#     mean = batch.mean(dim=1, keepdim=True)
#     std = batch.std(dim=1, keepdim=True, unbiased=False)
#     return (batch - mean) / (std + eps)
#
# def max_scaler(seq: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
#     max_val = torch.abs(seq).amax()
#     return seq / (max_val + eps)

def zero_scaler(seq: torch.Tensor):
    mean_val = seq.mean()
    std_val = seq.std()
    if std_val == 0:
        normed_seq = seq - mean_val
    else:
        normed_seq = (seq - mean_val) / std_val
    return normed_seq

def max_scaler(seq: torch.Tensor):
    max_val = torch.abs(seq).max()
    if max_val == 0:
        normed_seq = seq
    else:
        normed_seq = seq / max_val
    return normed_seq

class WaveletTimeSeriesDataCollator(DataCollatorMixin):
    """
    Data collator for WaveletMoE. \n
    For a single time series, first perfrom scaling, then DWT and finally patch-wise tokenize.

    Args:
     batch_size: Number of sequences contained in batch. If the batch contains multiple groups, \
        any part exceeding the *batch_size* will be truncated. The number of groups in the batch is denoted as *group_num*.
     patch_size: Patch size of sequences, final length of token will be *token_len = patch_size X 2*.
     wavelet_function: Wavelet function use for DWT.
     signal_extension_mode: Signal extension mode of DWT.
     level: DWT level.
     normalization_method: Normalization method (scaling method) of sequences.
     mode: `str`, should be one of `["TRAIN", "TEST"]`
     
    Returns:
     batch:
     - **input_ids**: tensor shape *(batch_size, token_num, token_len)*, *batch_size* processed sequences from *group_num* groups. \
        When testing, `input_ids.shape[0]` might smaller than `batch_size` to assure groups in batch are complete.
     - **labels**: tensor shape *(batch_size, token_num, token_len)*.
     - **group_ids**: tensor shape *(batch_size, )*, sequences within same group have same group_id.
     - **loss_masks**: tensor shape *(batch_size, token_num)*, sequences within same group have same loss mask.
    """

    def __init__(
        self, 
        batch_size: int, 
        patch_size: int = 8, 
        wavelet_function: str = "bior2.2", 
        signal_extension_mode: str = "periodization", 
        level: int = 2, 
        normalization_method: str = 'zero', 
        mode: str = "TRAIN"
    ):
        if patch_size%2 != 0:
            raise ValueError(f"Patch size should be an even number, not {patch_size}.")
        
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.tokenizer = DWTTokenizer(wavelet_function, signal_extension_mode, level, patch_size=patch_size)

        if mode not in ["TRAIN", "TEST"]:
            raise ValueError(f"Arg str should be one of [\"TRAIN\", \"TEST\"], not \"{mode}\". ")
        self.mode = mode

        if normalization_method is None:
            self.normalization_method = None
        elif isinstance(normalization_method, str):
            if normalization_method.lower() == 'max':
                self.normalization_method = max_scaler
            elif normalization_method.lower() == 'zero':
                self.normalization_method = zero_scaler
            else:
                raise ValueError(f'Unknown normalization method: {normalization_method}')
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # collect batch
        batch_data = torch.tensor([])
        batch_loss_masks = torch.tensor([])

        current_batch_size = 0
        for feature in features:
            # process group data
            group_data = feature['data']  # shape: (group_size, seq_len)
            group_size, seq_len = group_data.shape
            loss_mask = feature['loss_mask'].unsqueeze(0)  # shape: (1, seq_len)
            # group size (number of variates) differ across subsets.
            # if reamaining space is 0, drop unprocessed group, so group_num groups are collected in batch finally.
            remaining = self.batch_size - current_batch_size
            if remaining <= 0:
                break
            # when testing, if contains more than one groups, drop the last incomplete group,
            # assure that every group in batch is complete
            if self.mode == "TEST" and current_batch_size > 0 and remaining < group_size:
                break
            current_group_size = min(remaining, group_size)
            batch_data = torch.cat([batch_data, group_data[:current_group_size]], dim=0)

            current_batch_size += current_group_size
            # process loss_mask
            group_loss_mask = loss_mask.repeat(current_group_size, 1)
            batch_loss_masks = torch.cat([batch_loss_masks, group_loss_mask], dim=0)

        # process batch data: scaling & wavelet dwt tokenize
        # shape: (batch_size, seq_len) -> (batch_size, seq_len*2)
        if self.normalization_method is not None:
            batch_data = self.normalization_method(batch_data)
        device, dtype = batch_data.device, batch_data.dtype
        batch_data, _ = self.tokenizer.patch_wise_tokenize(batch_data)
        batch_data = torch.tensor(batch_data, dtype=dtype, device=device)    # pywt run on numpy

        # validate shape
        token_len = self.patch_size*2
        batch_size, seq_len = batch_data.shape      # the last batch might be smaller than self.batch_size
        if seq_len % token_len != 0:
            raise ValueError(f"Length of sequence {seq_len} must be multiple of token_len {token_len}.")
        token_num = int(seq_len / token_len)

        # process labels: drop first token & concat with a PAD token
        # shape: (batch_size, seq_len*2)
        batch_labels = torch.cat([batch_data[:, token_len:], torch.full(size = (batch_size, token_len), fill_value=0)], dim=1)

        # shape: (batch_size, seq_len*2) -> (batch_size, token_num, token_len)
        batch_data = batch_data.reshape(batch_size, token_num, token_len)
        batch_labels = batch_labels.reshape(batch_size, token_num, token_len)
        batch_loss_masks = batch_loss_masks.reshape(batch_size, token_num, self.patch_size)

        # process loss_masks: reshape & token-wise down-sample
        # shape: (batch_size, token_num, patch_size) -> (batch_size, token_num)
        batch_loss_masks = (batch_loss_masks != 0).any(dim=-1).long()

        # silcng to adapt new dual structure
        time_seq = batch_data[:,:, :self.patch_size]
        wavelet_seq = batch_data[:,:, self.patch_size:]

        time_seq_labels = batch_labels[:,:, :self.patch_size]
        wavelet_seq_labels = batch_labels[:,:, self.patch_size:]

        return {
            'time_seq': time_seq,                           # shape: (batch_size, token_num, patch_sz)
            'wavelet_seq': wavelet_seq,                     # shape: (batch_size, token_num, patch_sz)
            'time_seq_labels': time_seq_labels,             # shape: (batch_size, token_num, patch_sz)
            'wavelet_seq_labels': wavelet_seq_labels,       # shape: (batch_size, token_num, patch_sz)
            'loss_masks': batch_loss_masks                  # shape: (batch_size, token_num)
        }