import torch
import warnings
import numpy as np

from wavelet_moe.datasets.dwt_tokenizer import DWTTokenizer

class SumEvalMetric:
    def __init__(self, name, patch_size: int = 8):
        self.name = name
        self.patch_size = patch_size
        self.time_seq_loss = 0.0
        self.wavelet_seq_loss = 0.0

    def push(self, preds, labels):
        if len(preds.shape) > 2 and preds.shape[2] > self.patch_size:
            self.time_seq_loss += self._calculate(preds[:, :, : self.patch_size], labels[:, :, : self.patch_size])
            self.wavelet_seq_loss += self._calculate(preds[:, :, self.patch_size :], labels[:, :, self.patch_size :])
        else:
            self.time_seq_loss += self._calculate(preds, labels)
            self.wavelet_seq_loss = None

    def _calculate(self, preds: torch.Tensor, labels: torch.Tensor):
        pass


class MSEMetric(SumEvalMetric):
    def _calculate(self, preds: torch.Tensor, labels: torch.Tensor):
        return torch.sum((preds - labels) ** 2)


class MAEMetric(SumEvalMetric):
    def _calculate(self, preds: torch.Tensor, labels: torch.Tensor):
        return torch.sum(torch.abs(preds - labels))