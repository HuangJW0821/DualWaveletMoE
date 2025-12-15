import random
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import torch
import json
from typing import Dict, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt

from DualWaveletMoE.wavelet_moe.datasets.wavelet_moe_dataset import TimeSeriesWindowedSingleDataset
from wavelet_moe.datasets.wavelet_data_collator import WaveletTimeSeriesDataCollator
from wavelet_moe.evaluation.eval_models import ModelForEvaluation
from wavelet_moe.evaluation.eval_metrics import MAEMetric, MSEMetric
from wavelet_moe.evaluation.prediction_result_painter import PredictionResultPainter

class EvaluationRunner():
    def __init__(
        self,
        model: ModelForEvaluation,
        root_path: str,
        output_path: str,
        input_length: int,
        predict_length: int,
        batch_size: int,
        patch_size: int = 8,
        wavelet: str = "bior2.2",
        mode: str = "periodization", 
        level: int = 2, 
        normalization_method: str = 'zero',
        num_worker: int = 16,
        draw_prediciton_result: bool = True,
        predict_target_only: bool = False
    ):
        if not os.path.exists(root_path):
            raise ValueError(f"Path not exists: [{root_path}]!")
        if not os.path.exists(output_path):
            raise ValueError(f"Path not exists: [{output_path}]!")
        if input_length <= 0 or predict_length <= 0:
            raise ValueError(f"Both input_length & predict_length should larger than 0.")

        self.model = model
        self.root_path = root_path
        self.output_path = output_path

        self.dataset_names = os.listdir(root_path)
        self.benchmark_name = os.path.basename(root_path)

        self.input_length = input_length
        self.predict_length = predict_length
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.num_worker = num_worker
        self.draw_prediciton_result = draw_prediciton_result
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

        self.file_name = f"BENCHMARK[{self.benchmark_name}]_[{self.input_length} to {self.predict_length} tokens]"

        self.painter = PredictionResultPainter(
            output_path = self.output_path,
            file_name = self.file_name,
            input_length = self.input_length,
            predict_length = self.predict_length,
            patch_size = self.patch_size,
            predict_target_only = self.predict_target_only
        )

    def _eval_one_dataset(self, dataset: TimeSeriesWindowedSingleDataset):
        if self.predict_target_only:
            target_idx = (dataset.target_start_idx, dataset.target_end_idx)
        else:
            target_idx = None

        dataloader = DataLoader(
            dataset = dataset,
            batch_size = self.batch_size,
            shuffle = False,
            collate_fn = self.data_collator,
            num_workers = self.num_worker 
        )

        metric_list = [
            MSEMetric(name='mse'),
            MAEMetric(name='mae'),
        ]

        acc_count = 0
        with torch.no_grad():
            for i, batch in tqdm(enumerate(dataloader), desc=f"{dataset.dataset_name}-[{len(dataset)}]"):
                group_ids, preds, labels = self.model.generate(batch, target_idx = target_idx)

                if target_idx is not None:
                    _, _, counts = group_ids.unique_consecutive(return_inverse=True, return_counts=True)
                    group_pos = torch.arange(len(group_ids), device=group_ids.device) - torch.repeat_interleave(torch.cumsum(counts, 0) - counts, counts)
                    is_target = (group_pos >= target_idx[0]) & (group_pos < target_idx[1])
                    target_preds, target_labels = preds[is_target, :], labels[is_target, :]
                else:
                    target_preds, target_labels = preds, labels

                for metric in metric_list:
                    metric.push(target_preds, target_labels)

                acc_count += target_preds.numel()

                if i==100:
                    break
                
                if self.draw_prediciton_result and i%25==0:
                    batch_inputs, batch_labels, batch_preds = self.model.prepare_items_for_plt(batch, preds)
                    self.painter.draw_batch_prediction_result(dataset.dataset_name, group_ids, batch_inputs, batch_labels, batch_preds, i, target_idx)
        
        dataset_metrics = {
            "avg_exmaple_loss": {},
            "avg_sequence_part_loss": {},
            "avg_coeff_part_loss": {},
            "avg_idwt_reconstruct_loss": {},
            "avg_idwt_pred_seq_loss": {}
        }

        for metric in metric_list:
            dataset_metrics["avg_exmaple_loss"][metric.name] = (metric.exmaple_loss / acc_count).tolist()
            dataset_metrics["avg_sequence_part_loss"][metric.name] = (metric.sequence_part_loss / acc_count * 2).tolist()
            dataset_metrics["avg_coeff_part_loss"][metric.name] = (metric.coeff_part_loss / acc_count * 2).tolist()
            dataset_metrics["avg_idwt_reconstruct_loss"][metric.name] = (metric.idwt_reconstruct_loss / acc_count * 2).tolist()
            dataset_metrics["avg_idwt_pred_seq_loss"][metric.name] = (metric.idwt_pred_seq_loss / acc_count * 2).tolist()

        return {
            "dataset_metrics": dataset_metrics,
            "metric_list": metric_list,
            "acc_count": acc_count
        }

    def evaluate(self):
        per_dataset_results = {}
        metric_lists = []
        acc_count = 0.0

        for dataset_name in tqdm(self.dataset_names):

            # ###
            # if dataset_name != "bitbrains_fast_storage_H_processed_strictly":
            #     continue
            # ###

            dataset_path = os.path.join(self.root_path, dataset_name)
            if (not os.path.isdir(dataset_path)) or (len(os.listdir(dataset_path)) == 0):
                continue

            dataset = TimeSeriesWindowedSingleDataset(
                dataset_path = dataset_path,
                window_length = (self.input_length + self.predict_length) * self.patch_size
            )

            if len(dataset) < self.batch_size:
                continue

            result = self._eval_one_dataset(dataset)

            per_dataset_results[dataset_name] = result["dataset_metrics"]
            metric_lists.append(result["metric_list"])

            acc_count += result["acc_count"]
        
        final_eval_result = {
            "model_path": self.model.model_path,
            "data_path": self.root_path,
            "input_length": self.input_length,
            "predict_length": self.predict_length,
            "batch_size": self.batch_size,
            "predict_target_only": self.predict_target_only,
            "benchmark_result": {
                "avg_exmaple_loss": {},
                "avg_sequence_part_loss": {},
                "avg_coeff_part_loss": {},
                "avg_idwt_reconstruct_loss": {},
                "avg_idwt_pred_seq_loss": {}
            },
            "per_dataset_results": per_dataset_results
        }

        for i in range(len(metric_lists[0])):
            metric_name = metric_lists[0][i].name
            final_eval_result["benchmark_result"]["avg_exmaple_loss"][metric_name] = ( sum(metric[i].exmaple_loss for metric in metric_lists) / acc_count ).tolist()
            final_eval_result["benchmark_result"]["avg_sequence_part_loss"][metric_name] = ( sum(metric[i].sequence_part_loss for metric in metric_lists) / acc_count * 2 ).tolist()
            final_eval_result["benchmark_result"]["avg_coeff_part_loss"][metric_name] = ( sum(metric[i].coeff_part_loss for metric in metric_lists) / acc_count * 2 ).tolist()
            final_eval_result["benchmark_result"]["avg_idwt_reconstruct_loss"][metric_name] = ( sum(metric[i].idwt_reconstruct_loss for metric in metric_lists) / acc_count * 2 ).tolist()
            final_eval_result["benchmark_result"]["avg_idwt_pred_seq_loss"][metric_name] = ( sum(metric[i].idwt_pred_seq_loss for metric in metric_lists) / acc_count * 2 ).tolist()

        with open(os.path.join(self.output_path, f"{self.file_name}.txt"), "w", encoding="utf-8") as f:
            json.dump(final_eval_result, f, ensure_ascii=False, indent=4)

        return final_eval_result
        


        



