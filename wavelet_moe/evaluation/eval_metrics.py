import torch
import warnings
import numpy as np

from wavelet_moe.datasets.dwt_tokenizer import DWTTokenizer

class SumEvalMetric:
    def __init__(self, name, patch_size: int = 8):
        self.name = name
        self.patch_size = patch_size
        self.exmaple_loss = 0.0
        self.sequence_part_loss = 0.0
        self.coeff_part_loss = 0.0

        warnings.warn("DWTTokenizer in SumEvalMetric is init with default params, it should be change if you are conducting ablation on wavelet features.")
        self.dwt_tokenizer = DWTTokenizer()
        self.idwt_reconstruct_loss = 0.0
        self.idwt_pred_seq_loss = 0.0
        

    def push(self, preds, labels):
        one_example_loss = self._calculate(preds, labels)
        self.exmaple_loss += one_example_loss

        if len(preds.shape) > 2 and preds.shape[2] > self.patch_size:
            self.sequence_part_loss += self._calculate(preds[:, :, : self.patch_size], labels[:, :, : self.patch_size])
            self.coeff_part_loss += self._calculate(preds[:, :, self.patch_size :], labels[:, :, self.patch_size :])
            
            # calculating dwt loss, copied from .wavelet_moe.models.modeling_wavelet_moe.WaveletModeForPrediction.calc_dwt_loss()
            batch_size, pred_len, token_len = preds.shape
            flattened_preds = preds.reshape(batch_size, pred_len * token_len)
            pred_seqs, pred_coeffs = self.dwt_tokenizer.patch_wise_detokenize(flattened_preds)
                
            rec_seqs = self.dwt_tokenizer.waverec(pred_coeffs)

            if isinstance(pred_seqs, np.ndarray):
                pred_seqs = torch.from_numpy(pred_seqs).to(dtype=preds.dtype, device=preds.device)
            if isinstance(rec_seqs, np.ndarray):
                rec_seqs = torch.from_numpy(rec_seqs).to(dtype=preds.dtype, device=preds.device)

            self.idwt_reconstruct_loss += self._calculate(pred_seqs, rec_seqs)

            flattened_labels = labels[:, :, : self.patch_size].reshape(batch_size, pred_len * self.patch_size)
            self.idwt_pred_seq_loss += self._calculate(rec_seqs, flattened_labels)
        else:
            self.sequence_part_loss += one_example_loss
            self.coeff_part_loss += one_example_loss
            self.idwt_reconstruct_loss += one_example_loss
            self.idwt_pred_seq_loss += one_example_loss

    def _calculate(self, preds: torch.Tensor, labels: torch.Tensor):
        pass


class MSEMetric(SumEvalMetric):
    def _calculate(self, preds: torch.Tensor, labels: torch.Tensor):
        return torch.sum((preds - labels) ** 2)


class MAEMetric(SumEvalMetric):
    def _calculate(self, preds: torch.Tensor, labels: torch.Tensor):
        return torch.sum(torch.abs(preds - labels))