from typing import Dict, Optional, Tuple
import torch
from abc import abstractmethod
import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM
import timesfm

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
    

class SundialForEvaluation(ModelForEvaluation):
    def __init__(
        self, 
        model_path: str, 
        device: torch.device, 
        input_length: int, 
        prediction_length: int,
        patch_size: int = 8,
        num_samples: int = 20
    ):
        super().__init__(model_path, input_length, prediction_length, patch_size)
        self.num_samples = num_samples
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map=device,
        )
        
        self.model = model
        self.device = device
        self.model.eval()
    
    def _prepare_items_for_generate(self, batch: Dict):

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
    
    def generate(self, batch: Dict, target_idx: Optional[Tuple[int]] = None):

        inputs, labels = self._prepare_items_for_generate(batch)
        
        # Sundial 支持多预测序列
        outputs = self.model.generate(
            inputs=inputs,
            max_length=(self.input_length + self.prediction_length) * self.patch_size,
            num_samples=self.num_samples
        )  

        # print("outputs:", outputs.shape)
        # 输出是 (batch, num_samples, pred_len), 求平均到 (batch, pred_len)
        if outputs.dim() == 3:
            preds = outputs.mean(dim=1)
        else:
            preds = outputs
        
        # print("preds:", preds.shape)

        return preds, labels

    def prepare_items_for_plt(self, batch: Dict, preds: torch.Tensor):

        inputs, labels = self._prepare_items_for_generate(batch)
        inputs = np.array(inputs.cpu())
        labels = np.array(labels.cpu())
        preds = np.array(preds.cpu())
        return inputs, labels, preds


class TimerForEvaluation(ModelForEvaluation):
    def __init__(
            self, 
            model_path: str, 
            device: torch.device, 
            input_length: int, 
            prediction_length: int,
            patch_size: int = 8
        ):

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
         (group_ids, inputs, labels):
         - **group_ids**: `torch.Tensor`, shape `(batch_size, )`.
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

    def generate(self, batch: Dict, target_idx: Optional[Tuple[int]]=int):
        """
        Args:
         batch: `Dict`
        
        Returns:
         (preds, labels):
         - **preds**:  `torch.Tensor`, shape `(batch_size, pred_length * patch_size)`.
         - **labels**: `torch.Tensor`, shape `(batch_size, pred_length * patch_size)`.
        """
        inputs, labels = self._prepare_items_for_generate(batch)
        # print("inputs shape:", inputs.shape)

        outputs = self.model.generate(
            inputs = inputs,
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


# class TimesfmForEvaluation(ModelForEvaluation):
#     def __init__(
#         self,
#         model_path: str,
#         device: torch.device,
#         input_length: int,
#         prediction_length: int,
#         patch_size: int = 8,  # TimesFM 不使用 patch
#         forecast_config: timesfm.ForecastConfig = None,
#     ):
#         super().__init__(model_path, input_length, prediction_length, patch_size)

#         torch.set_float32_matmul_precision("high")

#         # Load model
#         model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(model_path)

#         if forecast_config is None:
#             forecast_config = timesfm.ForecastConfig(
#                 max_context=input_length*patch_size,
#                 max_horizon=prediction_length*patch_size,
#                 normalize_inputs=False,
#                 use_continuous_quantile_head=True,
#                 force_flip_invariance=True,
#                 infer_is_positive=True,
#                 fix_quantile_crossing=True,
#             )

#         model.compile(forecast_config)

#         self.model = model
#         self.device = device
    

#     def _prepare_items_for_generate(self, batch: Dict):
#         """
#         Args:
#          batch: Dict

#         Returns:
#          (inputs, labels):
#          - inputs: List[np.ndarray], each with shape (input_length,)
#          - labels: torch.Tensor, shape (batch_size, prediction_length)
#         """

#         model = self.model
#         device = self.device

#         input_length = self.input_length
#         prediction_length = self.prediction_length
#         patch_size=self.patch_size

#         input_ids = batch["time_seq"]  # (B, T, token_len)
#         batch_size, seq_len, _ = input_ids.shape

#         if input_length + prediction_length > seq_len:
#             raise ValueError(
#                 f"Input length + Pred length [{input_length} + {prediction_length}] "
#                 f"should be <= seq_len [{seq_len}]"
#             )

#         inputs = input_ids[:, : input_length, : patch_size].float()
#         labels = input_ids[:, input_length : input_length + prediction_length, : patch_size].float()
        
#         inputs = inputs.reshape(batch_size, input_length * patch_size)
#         labels = labels.reshape(batch_size, prediction_length * patch_size)

#         return inputs, labels
    
#     def generate(self, batch: Dict):
#         """
#         Returns:
#          (preds, labels):
#          - preds: torch.Tensor, shape (batch_size, prediction_length)
#          - labels: torch.Tensor, shape (batch_size, prediction_length)
#         """

#         inputs, labels = self._prepare_items_for_generate(batch)

#         with torch.no_grad():
#             point_forecast, _ = self.model.forecast(
#                 horizon=self.prediction_length*self.patch_size,
#                 inputs=inputs,
#             )

#         preds = torch.tensor(point_forecast, dtype=torch.float32)
        

#         return preds, labels
    
#     def prepare_items_for_plt(self, batch: Dict, preds: torch.Tensor):
#         """
#         Returns:
#          (inputs, labels, preds):
#          - inputs: np.ndarray, shape (batch_size, input_length)
#          - labels: np.ndarray, shape (batch_size, prediction_length)
#          - preds:  np.ndarray, shape (batch_size, prediction_length)
#         """

#         inputs, labels = self._prepare_items_for_generate(batch)

#         inputs = np.array(inputs.cpu())
#         labels = labels.cpu().numpy()
#         preds = preds.cpu().numpy()

#         return inputs, labels, preds


class TimesfmoldForEvaluation(ModelForEvaluation):
    def __init__(
        self,
        model_path: str,
        device: torch.device,
        input_length: int,
        prediction_length: int,
        batch_size: int,
        patch_size: int = 8,  # TimesFM 不使用 patch
    ):
        super().__init__(model_path, input_length, prediction_length, patch_size)

        # export HF_ENDPOINT=https://hf-mirror.com
        model = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(
                backend="gpu",
                per_core_batch_size = batch_size, #32,
                horizon_len=prediction_length*patch_size,
                num_layers=50,
                use_positional_embedding=False,
                context_len=input_length*patch_size,
            ),
            checkpoint=timesfm.TimesFmCheckpoint(
                huggingface_repo_id="google/timesfm-2.0-500m-pytorch"
            ),
        )

        self.model = model
        self.device = device
    

    def _prepare_items_for_generate(self, batch: Dict):
        """
        Args:
         batch: Dict

        Returns:
         (inputs, labels):
         - inputs: List[np.ndarray], each with shape (input_length,)
         - labels: torch.Tensor, shape (batch_size, prediction_length)
        """

        model = self.model
        device = self.device

        input_length = self.input_length
        prediction_length = self.prediction_length
        patch_size=self.patch_size

        input_ids = batch["time_seq"]  # (B, T, token_len)
        batch_size, seq_len, _ = input_ids.shape

        if input_length + prediction_length > seq_len:
            raise ValueError(
                f"Input length + Pred length [{input_length} + {prediction_length}] "
                f"should be <= seq_len [{seq_len}]"
            )

        inputs = input_ids[:, : input_length, : patch_size].float()
        labels = input_ids[:, input_length : input_length + prediction_length, : patch_size].float()
        
        inputs = inputs.reshape(batch_size, input_length * patch_size)
        labels = labels.reshape(batch_size, prediction_length * patch_size)

        return inputs, labels
    
    def generate(self, batch: Dict):
        """
        Returns:
         (preds, labels):
         - preds: torch.Tensor, shape (batch_size, prediction_length)
         - labels: torch.Tensor, shape (batch_size, prediction_length)
        """

        inputs, labels = self._prepare_items_for_generate(batch)

        input_size = len(inputs)
        freq = [0] * input_size

        with torch.no_grad():
            point_forecast, _ = self.model.forecast(
                inputs,
                freq=freq,
            )

        preds = torch.tensor(point_forecast, dtype=torch.float32)
        

        return preds, labels
    
    def prepare_items_for_plt(self, batch: Dict, preds: torch.Tensor):
        """
        Returns:
         (inputs, labels, preds):
         - inputs: np.ndarray, shape (batch_size, input_length)
         - labels: np.ndarray, shape (batch_size, prediction_length)
         - preds:  np.ndarray, shape (batch_size, prediction_length)
        """

        inputs, labels = self._prepare_items_for_generate(batch)

        # inputs = np.stack(inputs, axis=0)
        inputs = np.array(inputs.cpu())
        labels = labels.cpu().numpy()
        preds = preds.cpu().numpy()

        return inputs, labels, preds
