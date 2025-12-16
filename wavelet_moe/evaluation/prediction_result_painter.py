from typing import List, Optional 
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import torch
import json
from typing import Dict, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt

from wavelet_moe.datasets.wavelet_moe_dataset import TimeSeriesWindowedSingleDataset
from wavelet_moe.datasets.wavelet_data_collator import WaveletTimeSeriesDataCollator
from wavelet_moe.evaluation.eval_models import ModelForEvaluation
from wavelet_moe.evaluation.eval_metrics import MAEMetric, MSEMetric

class PredictionResultPainter:
    def __init__(
        self,
        output_path: str,
        file_name: str,
        input_length: int,
        predict_length: int,
        patch_size: int = 8
    ):
        self.output_path = output_path
        self.file_name = file_name
        self.input_length = input_length
        self.predict_length = predict_length
        self.patch_size = patch_size

    def _draw_group_prediction_result(
        self, 
        dataset_name: str, 
        group_inputs: np.ndarray, 
        group_labels: np.ndarray, 
        group_predicts: np.ndarray, 
        batch_idx: int, 
        group_idx: int,
        target_idx: Optional[Tuple[int]] = None
    ):
        x_input = np.asarray(list(range(self.input_length * self.patch_size)))
        x_label = np.asarray(list(range(self.input_length * self.patch_size, (self.input_length + self.predict_length) * self.patch_size)))
        
        group_size = group_inputs.shape[0]

        figsize = (48, group_size*6)
        fig, axes = plt.subplots(nrows = group_size, ncols = 1, figsize = figsize)

        if group_size == 1:
            axes = np.array([axes])
        else:
            axes = np.array(axes).reshape(-1)

        for i in range(group_size):
            axes[i].plot(x_input, group_inputs[i], c='black', label='input')
            if not (target_idx is not None and not (target_idx[0] <= i and i < target_idx[1])):
                axes[i].plot(x_label, group_labels[i], c='grey', label='label')
                axes[i].plot(x_label, group_predicts[i], c='red', label='prediction')
            axes[i].legend()

        output_path = os.path.join(self.output_path, f"{self.file_name}_examples", dataset_name)
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        fig.tight_layout()
        fig.suptitle(f"Dataset[{dataset_name}]_Batch[{batch_idx}]_Example[{group_idx}]", fontsize=16, fontweight='bold')  
        fig.savefig(os.path.join(output_path, f"Batch[{batch_idx}]_Example[{group_idx}].png"))
        plt.close()

    def draw_batch_prediction_result(
        self, 
        dataset_name: str, 
        batch_inputs: np.ndarray, 
        batch_labels: np.ndarray, 
        batch_preds: np.ndarray, 
        batch_idx: int
    ):
        x_input = np.asarray(list(range(self.input_length * self.patch_size)))
        x_label = np.asarray(list(range(self.input_length * self.patch_size, (self.input_length + self.predict_length) * self.patch_size)))
        
        batch_size = batch_inputs.shape[0]

        figsize = (48, batch_size*6)
        fig, axes = plt.subplots(nrows = batch_size, ncols = 1, figsize = figsize)

        if batch_size == 1:
            axes = np.array([axes])
        else:
            axes = np.array(axes).reshape(-1)

        for i in range(batch_size):
            axes[i].plot(x_input, batch_inputs[i], c='black', label='input')
            axes[i].plot(x_label, batch_labels[i], c='grey', label='label')
            axes[i].plot(x_label, batch_preds[i], c='red', label='prediction')
            axes[i].legend()

        output_path = os.path.join(self.output_path, f"{self.file_name}_examples", dataset_name)
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        fig.tight_layout()
        fig.suptitle(f"Dataset[{dataset_name}]_Batch[{batch_idx}]", fontsize=16, fontweight='bold')  
        fig.savefig(os.path.join(output_path, f"Batch[{batch_idx}].png"))
        plt.close()


class MultipleModelPredictionResultPainter():
    def __init__(
        self,
        model_list: List[ModelForEvaluation],
        model_name_list: List[str],
        color_name_list: List[str],
        root_path: str,
        output_path: str,
        input_length: int,
        predict_length: int,
        batch_size: int,
        batch_num: int,
        patch_size: int = 8,
        wavelet: str = "bior2.2",
        mode: str = "periodization", 
        level: int = 2, 
        normalization_method: str = 'zero',
        num_worker: int = 8,
        predict_target_only: bool = False
    ):
        raise NotImplementedError("Not implemted for DualWaveletMoE yet")
        if not os.path.exists(root_path):
            raise ValueError(f"Path not exists: [{root_path}]!")
        if not os.path.exists(output_path):
            raise ValueError(f"Path not exists: [{output_path}]!")
        if input_length <= 0 or predict_length <= 0:
            raise ValueError(f"Both input_length & predict_length should larger than 0.")
        if len(model_list) != len(model_name_list) or len(model_list) != len(color_name_list):
            raise ValueError(f"Argument `model_list`, `model_name_list` & `color_name_list` should have same length, " \
                             f"not [{len(model_list)}, {len(model_name_list)}, {len(color_name_list)}]")

        self.model_list = model_list
        self.model_name_list = model_name_list
        self.color_name_list = color_name_list

        self.root_path = root_path
        self.output_path = output_path

        self.dataset_names = os.listdir(root_path)
        self.benchmark_name = os.path.basename(root_path)

        self.input_length = input_length
        self.predict_length = predict_length
        self.batch_size = batch_size
        self.batch_num = batch_num

        self.patch_size = patch_size
        self.num_worker = num_worker
        self.predict_target_only = predict_target_only

        self.data_collator = WaveletTimeSeriesDataCollator(
            batch_size = batch_size,
            patch_size = patch_size,
            wavelet_function = wavelet,
            signal_extension_mode = mode,
            level = level,
            normalization_method = normalization_method,
            mode = "TEST"
        )

        self.file_name = f"MULTIPLE_MODELs_ON_BENCHMARK[{self.benchmark_name}]_[{self.input_length} to {self.predict_length} tokens]"

        self.painter = PredictionResultPainter(
            output_path = self.output_path,
            file_name = self.file_name,
            input_length = self.input_length,
            predict_length = self.predict_length,
            patch_size = self.patch_size,
            predict_target_only = predict_target_only
        )

    def _draw_one_dataset(
        self, 
        dataset: TimeSeriesWindowedSingleDataset
    ):
        if self.predict_target_only:
            target_idx = (dataset.target_start_idx, dataset.target_end_idx)
        else:
            target_idx = None

        dataloader = DataLoader(
            dataset = dataset,
            batch_size = self.batch_size,
            shuffle = True,
            collate_fn = self.data_collator,
            num_workers = self.num_worker,
        )

        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i > self.batch_num:
                    break

                group_ids, preds, _ = self.model_list[0].generate(batch, target_idx = target_idx)
                batch_inputs, batch_labels, batch_preds = self.model_list[0].prepare_items_for_plt(batch, preds)

                batch_preds_list = [batch_preds]

                for model_idx in range(1, len(self.model_list)):
                    _, preds, _ = self.model_list[model_idx].generate(batch, target_idx = target_idx)
                    _, _, batch_preds = self.model_list[model_idx].prepare_items_for_plt(batch, preds)
                    batch_preds_list.append(batch_preds)

                self.painter.draw_batch_prediction_result_for_pred_list(
                    model_name_list = self.model_name_list,
                    color_name_list = self.color_name_list,
                    dataset_name = dataset.dataset_name, 
                    group_ids = group_ids, 
                    batch_inputs = batch_inputs, 
                    batch_labels = batch_labels, 
                    batch_preds_list = batch_preds_list, 
                    batch_idx = i, 
                    target_idx = target_idx
                )


    def draw_result(self):
        for dataset_name in tqdm(self.dataset_names):

            dataset_path = os.path.join(self.root_path, dataset_name)
            if (not os.path.isdir(dataset_path)) or (len(os.listdir(dataset_path)) == 0):
                continue

            dataset = TimeSeriesWindowedSingleDataset(
                dataset_path = dataset_path,
                window_length = (self.input_length + self.predict_length) * self.patch_size
            )

            if len(dataset) < self.batch_size:
                continue

            self._draw_one_dataset(dataset)