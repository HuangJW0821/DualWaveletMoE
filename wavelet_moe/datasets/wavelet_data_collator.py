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

def zero_scaler(batch: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mean = batch.mean(dim=1, keepdim=True)
    std = batch.std(dim=1, keepdim=True, unbiased=False)
    return (batch - mean) / (std + eps)

def max_scaler(seq: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    max_val = torch.abs(seq).amax()
    return seq / (max_val + eps)


class WaveletTimeSeriesDataCollator(DataCollatorMixin):
    """
    Data collator for WaveletMoE. \n
    For a single time series, first perfrom scaling, then DWT and finally patch-wise tokenize.

    Args:
     batch_size: Number of sequences contained in batch. If the batch contains multiple groups, \
        any part exceeding the *batch_size* will be truncated. The number of groups in the batch is denoted as *group_num*.
     patch_size: Patch size of sequences, final length of token will be *token_len = patch_size*.
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

        for feature in features:
            if batch_data.shape[0] >= self.batch_size:
                break
            feature_data = feature["data"]  # [C, L] / [1, L] / [L]
            loss_mask = feature["loss_mask"]  # [L] / [1, L]
            if feature_data.dim() == 1:
                feature_data = feature_data.unsqueeze(0)  # [1, L]
            C, L = feature_data.shape
            if loss_mask.dim() == 1:
                loss_mask = loss_mask.unsqueeze(0)  # [1, L]
            if loss_mask.shape[0] == 1 and C > 1:
                loss_mask = loss_mask.repeat(C, 1)  # [C, L]
            remaining = self.batch_size - batch_data.shape[0]
            if C > remaining:
                feature_data = feature_data[:remaining]
                loss_mask = loss_mask[:remaining]
            batch_data = torch.cat([batch_data, feature_data], dim=0)
            batch_loss_masks = torch.cat([batch_loss_masks, loss_mask], dim=0)

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