from typing import Dict, Optional, Tuple
import torch
from abc import abstractmethod
import numpy as np
import pandas as pd

from wavelet_moe.models.modeling_wavelet_moe import WaveletMoeForPrediction

class ModelForEvaluation:
    def __init__(self, model_path: str, input_length: int, prediction_length: int, patch_size: int = 8):
        self.model_path = model_path
        self.input_length = input_length
        self.prediction_length = prediction_length
        self.patch_size = patch_size

    @abstractmethod
    def _prepare_items_for_generate(self, batch: Dict):
        pass

    @abstractmethod
    def generate(self, batch: Dict):
        pass
    
    @abstractmethod
    def prepare_items_for_plt(self, batch: Dict, preds: torch.Tensor):
        pass


class WaveletMoEForEvaluation(ModelForEvaluation):
    def __init__(
            self, 
            model_path: str, 
            device: torch.device, 
            input_length: int, 
            prediction_length: int,
            patch_size: int = None
        ):

        super().__init__(model_path, input_length, prediction_length, patch_size)
        
        model = WaveletMoeForPrediction.from_pretrained(
            model_path,
            device_map = device,
            torch_dtype = 'auto',
        )

        self.patch_size = model.config.patch_size   # load from model config

        self.model = model
        self.device = device
        self.model.eval()

    def _prepare_items_for_generate(self, batch: Dict):
        """
        Args:
         batch: `Dict`
        
        Returns:
         (inputs, labels):
         - **inputs**: `torch.Tensor`, shape `(batch_size, input_length, patch_size * 2)`.
         - **labels**: `torch.Tensor`, shape `(batch_size, pred_length, patch_size * 2)`.
        """
        model = self.model
        device = self.device

        input_length = self.input_length
        prediction_length = self.prediction_length

        input_ids = (torch.cat((batch["time_seq"], batch["wavelet_seq"]), dim=2)).to(device)

        seq_len = input_ids.shape[1]
        if input_length + prediction_length > seq_len:
            raise ValueError(f"Input length + Pred length [{input_length} + {prediction_length} = {input_length + prediction_length}] should be shorter than seq_len [{seq_len}]")

        inputs = input_ids[:, : input_length, :].to(device).to(model.dtype)
        labels = input_ids[:, input_length : input_length + prediction_length, :].to(device).to(model.dtype)
        
        return inputs, labels

    def generate(self, batch: Dict):
        """
        Args:
         batch: `Dict`
        
        Returns:
         (preds, labels):
         - **preds**:  `torch.Tensor`, shape `(batch_size, pred_length, patch_size * 2)`.
         - **labels**: `torch.Tensor`, shape `(batch_size, pred_length, patch_size * 2)`.
        """
        input_ids, labels = self._prepare_items_for_generate(batch)

        outputs = self.model.generate(
            input_ids = input_ids,
            max_length = self.input_length + self.prediction_length,
        )

        preds = outputs[:, -self.prediction_length :, :]

        return preds, labels

    def prepare_items_for_plt(self, batch: Dict, preds: torch.Tensor):
        """
        Args:
         batch: `Dict`
         preds: `torch.Tensor`, shape `(batch_size, pred_length, patch_size * 2)`
        
        Returns:
         (group_ids, inputs, labels, seq_len):
         - **inputs**: `np.ndarray`, shape `(batch_size, input_length * patch_size)`.
         - **labels**: `np.ndarray`, shape `(batch_size, pred_length * patch_size)`.
         - **preds**:  `np.ndarray`, shape `(batch_size, pred_length * patch_size)`.
        """
        input_length = self.input_length
        pred_length = self.prediction_length
        patch_size = self.patch_size

        inputs, labels = self._prepare_items_for_generate(batch)

        batch_size = inputs.shape[0]

        inputs = inputs[:,:, :patch_size].reshape(batch_size, input_length * patch_size)
        labels = labels[:,:, :patch_size].reshape(batch_size, pred_length * patch_size)
        preds = preds[:,:, :patch_size].reshape(batch_size, pred_length * patch_size)

        inputs = np.array(inputs.cpu())
        labels = np.array(labels.cpu())
        preds = np.array(preds.cpu())

        return inputs, labels, preds


class TimeMoEForEvaluation(ModelForEvaluation):
    def __init__(
            self, 
            model_path: str, 
            device: torch.device, 
            input_length: int, 
            prediction_length: int,
            patch_size: int = 8
        ):
        from transformers import AutoModelForCausalLM

        super().__init__(model_path, input_length, prediction_length, patch_size)
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map = device,
            trust_remote_code=True,
        )

        self.model = model
        self.device = device
        self.model.eval()

    def _prepare_items_for_generate(self, batch: Dict):
        """
        Args:
         batch: `Dict`
        
        Returns:
         (inputs, labels):
         - **inputs**: `torch.Tensor`, shape `(batch_size, input_length * patch_size)`.
         - **labels**: `torch.Tensor`, shape `(batch_size, pred_length * patch_size)`.
        """
        model = self.model
        device = self.device

        input_length = self.input_length
        prediction_length = self.prediction_length
        patch_size = self.patch_size

        input_ids = batch["time_seq"]
        batch_size, seq_len, _ = input_ids.shape

        if input_length + prediction_length > seq_len:
            raise ValueError(f"Input length + Pred length [{input_length} + {prediction_length} = {input_length + prediction_length}] should be shorter than seq_len [{seq_len}]")

        inputs = input_ids[:, : input_length, : patch_size].to(device).to(model.dtype)
        labels = input_ids[:, input_length : input_length + prediction_length, : patch_size].to(device).to(model.dtype)
        
        inputs = inputs.reshape(batch_size, input_length * patch_size)
        labels = labels.reshape(batch_size, prediction_length * patch_size)

        return inputs, labels

    def generate(self, batch: Dict):
        """
        Args:
         batch: `Dict`
        
        Returns:
         (preds, labels):
         - **preds**:  `torch.Tensor`, shape `(batch_size, pred_length * patch_size)`.
         - **labels**: `torch.Tensor`, shape `(batch_size, pred_length * patch_size)`.
        """
        inputs, labels = self._prepare_items_for_generate(batch)

        outputs = self.model.generate(
            input_ids = inputs,
            max_length = (self.input_length + self.prediction_length) * self.patch_size,
        )

        preds = outputs[:, -(self.prediction_length * self.patch_size) :]

        return preds, labels

    def prepare_items_for_plt(self, batch: Dict, preds: torch.Tensor):
        """
        Args:
         batch: `Dict`
         preds: `torch.Tensor`, shape `(batch_size, pred_length, token_len)`
        
        Returns:
         (inputs, labels, seq_len):
         - **inputs**: `np.ndarray`, shape `(batch_size, input_length * patch_size)`.
         - **labels**: `np.ndarray`, shape `(batch_size, pred_length * patch_size)`.
         - **preds**:  `np.ndarray`, shape `(batch_size, pred_length * patch_size)`.
        """

        inputs, labels = self._prepare_items_for_generate(batch)

        inputs = np.array(inputs.cpu())
        labels = np.array(labels.cpu())
        preds = np.array(preds.cpu())

        return inputs, labels, preds


class ChronosForEvaluation(ModelForEvaluation):
    def __init__(
            self,
            model_path: str,
            device: torch.device,
            input_length: int,
            prediction_length: int,
            patch_size: int = 8
    ):
        from chronos import ChronosPipeline

        super().__init__(model_path, input_length, prediction_length, patch_size)

        model = ChronosPipeline.from_pretrained(
            model_path,
            device_map=device,
        )

        self.model = model
        self.device = device

    def _prepare_items_for_generate(self, batch: Dict):
        """
        Args:
         batch: `Dict`

        Returns:
         (inputs, labels):
         - **inputs**: `torch.Tensor`, shape `(batch_size, input_length * patch_size)`.
         - **labels**: `torch.Tensor`, shape `(batch_size, pred_length * patch_size)`.
        """
        model = self.model
        device = self.device

        input_length = self.input_length
        prediction_length = self.prediction_length
        patch_size = self.patch_size

        input_ids = batch["time_seq"]
        batch_size, seq_len, _ = input_ids.shape

        if input_length + prediction_length > seq_len:
            raise ValueError(
                f"Input length + Pred length [{input_length} + {prediction_length} = {input_length + prediction_length}] should be shorter than seq_len [{seq_len}]")

        inputs = input_ids[:, : input_length, : patch_size]
        labels = input_ids[:, input_length: input_length + prediction_length, : patch_size]

        inputs = inputs.reshape(batch_size, input_length * patch_size)
        labels = labels.reshape(batch_size, prediction_length * patch_size)

        return inputs, labels

    def generate(self, batch: Dict):
        """
        Args:
         batch: `Dict`

        Returns:
         (preds, labels):
         - **preds**:  `torch.Tensor`, shape `(batch_size, pred_length * patch_size)`.
         - **labels**: `torch.Tensor`, shape `(batch_size, pred_length * patch_size)`.
        """
        inputs, labels = self._prepare_items_for_generate(batch)

        outputs = self.model.predict(inputs, self.prediction_length * self.patch_size)

        preds = outputs[:, -(self.prediction_length * self.patch_size):]

        mid_q = preds.size(1) // 2  # 20 // 2 = 10
        preds = preds[:, mid_q, :]

        return preds, labels

    def prepare_items_for_plt(self, batch: Dict, preds: torch.Tensor):
        """
        Args:
         batch: `Dict`
         preds: `torch.Tensor`, shape `(batch_size, pred_length, token_len)`

        Returns:
         (inputs, labels, seq_len):
         - **inputs**: `np.ndarray`, shape `(batch_size, input_length * patch_size)`.
         - **labels**: `np.ndarray`, shape `(batch_size, pred_length * patch_size)`.
         - **preds**:  `np.ndarray`, shape `(batch_size, pred_length * patch_size)`.
        """

        inputs, labels = self._prepare_items_for_generate(batch)

        inputs = np.array(inputs.cpu())
        labels = np.array(labels.cpu())
        preds = np.array(preds.cpu())

        return inputs, labels, preds


# wrap moirai family in one class
class MoiraiFamilyForEvaluation(ModelForEvaluation):
    def __init__(
        self, 
        model_path: str, 
        device: torch.device, 
        input_length: int, 
        prediction_length: int,
        patch_size: int = 8     # consistent with WaveletMoeDataCollator, unused in real Moirai model        
    ):
        from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
        from uni2ts.model.moirai_moe import MoiraiMoEForecast, MoiraiMoEModule
        from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module

        super().__init__(model_path, input_length, prediction_length, patch_size)

        if "moirai-1" in model_path:
            # Moirai-1.0 uses patch_size=32 during eval as mentioned in Section 4.1 in its paper (https://arxiv.org/abs/2402.02592).
            model = MoiraiForecast(
                module = MoiraiModule.from_pretrained(model_path),
                prediction_length = prediction_length * patch_size,
                context_length = input_length * patch_size,
                patch_size = 32,
                target_dim = 1,
                feat_dynamic_real_dim = 0,
                past_feat_dynamic_real_dim = 0
            )
        elif "moirai-moe" in model_path:
            # Moirai-MoE uses patch_size=16 during eval as mentioned in Appendix B in its paper (https://arxiv.org/abs/2410.10469.
            model = MoiraiMoEForecast(
                module = MoiraiMoEModule.from_pretrained(model_path),
                prediction_length = prediction_length * patch_size,
                context_length = input_length * patch_size,
                patch_size = 16,
                target_dim = 1,
                feat_dynamic_real_dim = 0,
                past_feat_dynamic_real_dim = 0
            )
        elif "moirai-2" in model_path:
            model = Moirai2Forecast(
                module = Moirai2Module.from_pretrained(model_path),
                prediction_length = prediction_length * patch_size,
                context_length = input_length * patch_size,
                target_dim = 1,
                feat_dynamic_real_dim = 0,
                past_feat_dynamic_real_dim = 0
            )
        else:
            raise ValueError(f"Unsupport Moirai model name: {model_path}, support version: ['moirai-1.0', 'moirai-1.1', 'moirai-moe', 'moirai-2.0']")

        model.to(device).eval()

        self.model = model
        self.device = device

    def _prepare_items_for_generate(self, batch: Dict):
        """
        Args:
         batch: `Dict`
        
        Returns:
         (inputs, labels, moirai_input_dict):
         - **inputs**: `torch.Tensor`, shape `(batch_size, input_length * patch_size)`.
         - **labels**: `torch.Tensor`, shape `(batch_size, pred_length * patch_size)`.
         - **moirai_input_dict**: `Dict[str, torch.Tensor]`, input items in moirai's style
        """
        model = self.model
        device = self.device

        input_length = self.input_length
        prediction_length = self.prediction_length
        patch_size = self.patch_size

        input_ids = batch["time_seq"]
        batch_size, seq_len, _ = input_ids.shape

        if input_length + prediction_length > seq_len:
            raise ValueError(f"Input length + Pred length [{input_length} + {prediction_length} = {input_length + prediction_length}] should be shorter than seq_len [{seq_len}]")

        inputs = input_ids[:, : input_length, : patch_size].to(device).to(model.dtype)
        labels = input_ids[:, input_length : input_length + prediction_length, : patch_size].to(device).to(model.dtype)
        
        inputs = inputs.reshape(batch_size, input_length * patch_size)
        labels = labels.reshape(batch_size, prediction_length * patch_size)

        # shape [batch_sz, input_len * patch_sz, target_dim]
        past_target = inputs.unsqueeze(-1)

        # shape [batch_sz, input_len * patch_sz, target_dim]
        past_observed_target = torch.ones_like(past_target, dtype=torch.bool, device=device)  

        # shape [batch_sz, input_len * patch_sz]
        past_is_pad = torch.zeros_like(inputs, dtype=torch.bool, device=device)

        # prepare input in Moirai's style
        moirai_input_dict = {
            "past_target": past_target,
            "past_observed_target": past_observed_target,
            "past_is_pad": past_is_pad
        }

        return inputs, labels, moirai_input_dict

    def generate(self, batch: Dict):
        """
        Args:
         batch: `Dict`
        
        Returns:
         (preds, labels):
         - **preds**:  `torch.Tensor`, shape `(batch_size, pred_length * patch_size)`.
         - **labels**: `torch.Tensor`, shape `(batch_size, pred_length * patch_size)`.
        """
        _, labels, input_dict = self._prepare_items_for_generate(batch)

        with torch.no_grad():
            preds = self.model(**input_dict)

        # shape [batch_sz, num_samples, pred_len * patch_sz] -> [batch_sz, pred_len * patch_sz]
        preds = preds.mean(dim=1)

        return preds, labels

    def prepare_items_for_plt(self, batch: Dict, preds: torch.Tensor):
        """
        Args:
         batch: `Dict`
         preds: `torch.Tensor`, shape `(batch_size, pred_length, token_len)`
        
        Returns:
         (inputs, labels, seq_len):
         - **inputs**: `np.ndarray`, shape `(batch_size, input_length * patch_size)`.
         - **labels**: `np.ndarray`, shape `(batch_size, pred_length * patch_size)`.
         - **preds**:  `np.ndarray`, shape `(batch_size, pred_length * patch_size)`.
        """

        inputs, labels, _ = self._prepare_items_for_generate(batch)

        inputs = np.array(inputs.cpu())
        labels = np.array(labels.cpu())
        preds = np.array(preds.cpu())

        return inputs, labels, preds
