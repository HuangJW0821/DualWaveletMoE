from typing import Dict, Optional, Tuple
import torch
from abc import abstractmethod
from transformers import AutoModelForCausalLM
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
            patch_size: int = 8
        ):

        super().__init__(model_path, input_length, prediction_length, patch_size)
        
        model = WaveletMoeForPrediction.from_pretrained(
            model_path,
            device_map = device,
            torch_dtype = 'auto',
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
         - **inputs**: `torch.Tensor`, shape `(batch_size, input_length, patch_size * 2)`.
         - **labels**: `torch.Tensor`, shape `(batch_size, pred_length, patch_size * 2)`.
        """
        model = self.model
        device = self.device

        input_length = self.input_length
        prediction_length = self.prediction_length

        # input_ids = batch["input_ids"]
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
         target_idx: Optional, idx of target varaiates in group.
        
        Returns:
         (preds, labels):
         - **preds**:  `torch.Tensor`, shape `(batch_size, pred_length, token_len)`.
         - **labels**: `torch.Tensor`, shape `(batch_size, pred_length, token_len)`.
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
         preds: `torch.Tensor`, shape `(batch_size, pred_length, token_len)`
        
        Returns:
         (group_ids, inputs, labels, seq_len):
         - **group_ids**: `torch.Tensor`, shape `(batch_size, )`.
         - **inputs**: `np.ndarray`, shape `(batch_size, input_length * patch_size)`.
         - **labels**: `np.ndarray`, shape `(batch_size, pred_length * patch_size)`.
         - **preds**:  `np.ndarray`, shape `(batch_size, pred_length * patch_size)`.
        """
        input_length = self.input_length
        pred_length = self.prediction_length
        patch_size = self.patch_size

        _, inputs, labels = self._prepare_items_for_generate(batch)

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

        input_ids = batch["input_ids"]
        batch_size, seq_len, _ = input_ids.shape

        if input_length + prediction_length > seq_len:
            raise ValueError(f"Input length + Pred length [{input_length} + {prediction_length} = {input_length + prediction_length}] should be shorter than seq_len [{seq_len}]")

        inputs = input_ids[:, : input_length, : patch_size].to(device).to(model.dtype)
        labels = input_ids[:, input_length : input_length + prediction_length, : patch_size].to(device).to(model.dtype)
        
        inputs = inputs.reshape(batch_size, input_length * patch_size)
        labels = labels.reshape(batch_size, prediction_length * patch_size)

        group_ids = batch["group_ids"].to(device).to(model.dtype)

        return group_ids, inputs, labels

    def generate(self, batch: Dict, target_idx: Optional[Tuple[int]]=int):
        """
        Args:
         batch: `Dict`
        
        Returns:
         (preds, labels):
         - **preds**:  `torch.Tensor`, shape `(batch_size, pred_length * patch_size)`.
         - **labels**: `torch.Tensor`, shape `(batch_size, pred_length * patch_size)`.
        """
        group_ids, inputs, labels = self._prepare_items_for_generate(batch)

        outputs = self.model.generate(
            input_ids = inputs,
            max_length = (self.input_length + self.prediction_length) * self.patch_size,
        )

        preds = outputs[:, -(self.prediction_length * self.patch_size) :]

        return group_ids, preds, labels

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

        _, inputs, labels = self._prepare_items_for_generate(batch)

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
        #self.model.eval()

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

        input_ids = batch["input_ids"]
        batch_size, seq_len, _ = input_ids.shape

        if input_length + prediction_length > seq_len:
            raise ValueError(
                f"Input length + Pred length [{input_length} + {prediction_length} = {input_length + prediction_length}] should be shorter than seq_len [{seq_len}]")

        inputs = input_ids[:, : input_length, : patch_size]
        labels = input_ids[:, input_length: input_length + prediction_length, : patch_size]

        inputs = inputs.reshape(batch_size, input_length * patch_size)
        labels = labels.reshape(batch_size, prediction_length * patch_size)

        group_ids = batch["group_ids"]

        return group_ids, inputs, labels

    def generate(self, batch: Dict, target_idx: Optional[Tuple[int]] = int):
        """
        Args:
         batch: `Dict`

        Returns:
         (preds, labels):
         - **preds**:  `torch.Tensor`, shape `(batch_size, pred_length * patch_size)`.
         - **labels**: `torch.Tensor`, shape `(batch_size, pred_length * patch_size)`.
        """
        group_ids, inputs, labels = self._prepare_items_for_generate(batch)

        outputs = self.model.predict(inputs,self.prediction_length * self.patch_size)


        preds = outputs[:, -(self.prediction_length * self.patch_size):]

        mid_q = preds.size(1) // 2  # 20 // 2 = 10
        preds = preds[:, mid_q, :]

        return group_ids, preds, labels

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

        _, inputs, labels = self._prepare_items_for_generate(batch)

        inputs = np.array(inputs.cpu())
        labels = np.array(labels.cpu())
        preds = np.array(preds.cpu())

        return inputs, labels, preds


class Chronos2ForEvaluation(ModelForEvaluation):
    def __init__(
            self, 
            model_path: str, 
            device: torch.device, 
            input_length: int, 
            prediction_length: int,
            patch_size: int = 8
        ):
        from chronos import Chronos2Pipeline

        super().__init__(model_path, input_length, prediction_length, patch_size)
        
        self.model = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map=device)

    def _prepare_items_for_generate(self, batch: Dict, target_idx: Optional[Tuple[int]] = None):
        """
        Args:
         batch: `Dict`
         target_idx: Optional, idx of target varaiates in group.
        
        Returns:
         (group_ids, input_df, labels):
         - **group_ids**: `torch.Tensor`, shape `(batch_size, )`
         - **input_df**: `pandas.DataFrame`, input dataframe in Chronos2 style.
         - **labels**: `torch.Tensor`, shape `(batch_size, pred_length, token_len)`.
        """
        input_length = self.input_length
        prediction_length = self.prediction_length
        patch_size = self.patch_size

        group_ids = batch["group_ids"]
        input_ids = batch["input_ids"]
        batch_size, seq_len, _ = input_ids.shape

        if input_length + prediction_length > seq_len:
            raise ValueError(f"Input length + Pred length [{input_length} + {prediction_length} = {input_length + prediction_length}] should be shorter than seq_len [{seq_len}]")

        inputs = input_ids[:, : input_length, : patch_size].reshape(batch_size, input_length * patch_size)
        labels = input_ids[:, input_length : input_length + prediction_length, : patch_size].reshape(batch_size, prediction_length * patch_size)
        
        # process to dataframe, in adaption for Chronos2
        group_num = len(torch.unique_consecutive(group_ids))
        group_size = batch_size // group_num        # TEST mode of datacollator assure that batch contains complete groups

        # (batch_size, input_len * patch_size) -> (group_num, group_size, input_len * patch_size)
        inputs = inputs.reshape(group_num, group_size, input_length * patch_size)

        # -> (group_size, group_num, input_len * patch_size) -> (group_size, group_num * input_len * patch_size)
        inputs = inputs.transpose(0, 1).reshape(group_size, group_num * input_length * patch_size)

        # to dataframe
        input_df = pd.DataFrame(inputs.T.numpy())

        # set columns names
        col_names, target_col_names = [], []
        if target_idx is None:
            target_idx = (0, group_size)
        for i in range(group_size):
            if target_idx[0]<=i<target_idx[1]:
                col_name = f"target_{i}"
                target_col_names.append(col_name)
            else:
                col_name = f"cov_{i}"
            col_names.append(col_name)
        input_df.columns = col_names

        # add id & timestamp column
        input_df["item_id"] = torch.repeat_interleave(torch.arange(group_num), repeats = input_length * patch_size)
        input_df["timestamp"] = torch.arange(input_length * patch_size).repeat(group_num)

        return group_ids, input_df, labels, target_col_names, group_num, group_size

    def generate(self, batch: Dict, target_idx: Optional[Tuple[int]] = None):
        """
        Args:
         batch: `Dict`
         target_idx: Optional, idx of target varaiates in group.
        
        Returns:
         (preds, labels):
         - **preds**:  `torch.Tensor`, shape `(batch_size, pred_length, token_len)`.
         - **labels**: `torch.Tensor`, shape `(batch_size, pred_length, token_len)`.
        """
        group_ids, input_df, labels, target_col_names, group_num, group_size = self._prepare_items_for_generate(batch, target_idx)
        target_nums = len(target_col_names)
        prediction_length = self.prediction_length * self.patch_size

        outputs = self.model.predict_df(
            input_df,
            prediction_length = prediction_length,
            quantile_levels = [0.5],
            target = target_col_names
        )

        # shape (group_num * target_nums * (pred_len * patch_size) )
        preds = torch.Tensor(outputs['predictions'].values)

        # shape (group_num, target_nums, pred_len * patch_size)
        preds = preds.reshape(group_num, target_nums, prediction_length)

        # if covariate exsist, restore preds' shape to
        # (group_num, group_size, pred_len * patch_size)
        if target_nums < group_size:
            restored_preds = torch.zeros((group_num, group_size, prediction_length),  dtype=preds.dtype, device=preds.device)
            restored_preds[:, target_idx[0] : target_idx[1], :] = preds
            preds = restored_preds
        
        if group_num * group_size != labels.shape[0]:
            raise ValueError(f"group_num * group_size should be {labels.shape[0]}, not {group_num * group_size}!")

        preds = preds.reshape(group_num * group_size, prediction_length)

        return group_ids, preds, labels

    def prepare_items_for_plt(self, batch: Dict, preds: torch.Tensor):
        """
        Args:
         batch: `Dict`
         preds: `torch.Tensor`, shape `(batch_size, pred_length, token_len)`
        
        Returns:
         (group_ids, inputs, labels, seq_len):
         - **group_ids**: `torch.Tensor`, shape `(batch_size, )`.
         - **inputs**: `np.ndarray`, shape `(batch_size, input_length * patch_size)`.
         - **labels**: `np.ndarray`, shape `(batch_size, pred_length * patch_size)`.
         - **preds**:  `np.ndarray`, shape `(batch_size, pred_length * patch_size)`.
        """
        input_length = self.input_length
        pred_length = self.prediction_length
        patch_size = self.patch_size

        input_ids = batch["input_ids"]
        batch_size, _, _ = input_ids.shape

        inputs = input_ids[:, : input_length, :patch_size].reshape(batch_size, input_length * patch_size)
        labels = input_ids[:, input_length : input_length + pred_length , : patch_size].reshape(batch_size, pred_length * patch_size)

        inputs = np.array(inputs.cpu())
        labels = np.array(labels.cpu())
        preds = np.array(preds.cpu())

        return inputs, labels, preds

