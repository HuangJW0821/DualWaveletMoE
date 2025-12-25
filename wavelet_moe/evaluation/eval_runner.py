import random
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
from wavelet_moe.evaluation.eval_models import ModelForEvaluation, WaveletMoEForEvaluation
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
        use_per_sample_norm: bool = False,
        num_worker: int = 16,
        draw_prediciton_result: bool = True
    ):
        if not os.path.exists(root_path):
            raise ValueError(f"Path not exists: [{root_path}]!")
        if not os.path.exists(output_path):
            raise ValueError(f"Path not exists: [{output_path}]!")
        if input_length <= 0 or predict_length <= 0:
            raise ValueError(f"Both input_length & predict_length should larger than 0.")

        self.model = model
        self.is_wavelet_moe = isinstance(model, WaveletMoEForEvaluation)

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

        self.data_collator = WaveletTimeSeriesDataCollator(
            batch_size = batch_size,
            patch_size = patch_size,
            wavelet_function = wavelet,
            signal_extension_mode = mode,
            level = level,
            normalization_method = normalization_method,
            use_per_sample_norm = use_per_sample_norm
        )

        self.file_name = f"BENCHMARK[{self.benchmark_name}]_[{self.input_length} to {self.predict_length} tokens]"

        if draw_prediciton_result:
            self.painter = PredictionResultPainter(
                output_path = self.output_path,
                file_name = self.file_name,
                input_length = self.input_length,
                predict_length = self.predict_length,
                patch_size = self.patch_size,
            )

    def _eval_one_dataset(self, dataset: TimeSeriesWindowedSingleDataset):

        dataloader = DataLoader(
            dataset = dataset,
            batch_size = self.batch_size,
            shuffle = False,
            collate_fn = self.data_collator,
            num_workers = self.num_worker 
        )

        if self.draw_prediciton_result:
            draw_result_step = max(1, len(dataset) // (5 * self.batch_size))    # draw 5 batch for each dataset

        metric_list = [
            MSEMetric(name='mse', patch_size = self.patch_size),
            MAEMetric(name='mae', patch_size = self.patch_size),
        ]

        timestep_cnt = 0
        with torch.no_grad():
            for i, batch in tqdm(enumerate(dataloader), desc=f"{dataset.dataset_name}-[{len(dataset)}]"):
                preds, labels = self.model.generate(batch)

                for metric in metric_list:
                    metric.push(preds, labels)

                timestep_cnt += preds.numel()

                # if i==100:
                #     break
                
                if self.draw_prediciton_result and i % draw_result_step == 0:
                    batch_inputs, batch_labels, batch_preds = self.model.prepare_items_for_plt(batch, preds)
                    self.painter.draw_batch_prediction_result(dataset.dataset_name, batch_inputs, batch_labels, batch_preds, i)
        
        if self.is_wavelet_moe:
            timestep_cnt /= 2

        dataset_metrics = {"time_seq_loss": {}}

        for metric in metric_list:
            dataset_metrics["time_seq_loss"][metric.name] = (metric.time_seq_loss / timestep_cnt).tolist()

        if self.is_wavelet_moe:
            dataset_metrics["wavelet_seq_loss"] = {}
            for metric in metric_list:
                dataset_metrics["wavelet_seq_loss"][metric.name] = (metric.wavelet_seq_loss / timestep_cnt).tolist()

        return {
            "dataset_metrics": dataset_metrics,
            "metric_list": metric_list,
            "timestep_cnt": timestep_cnt
        }

    def evaluate(self):
        per_dataset_results = {}
        metric_lists = []
        timestep_cnt = 0

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

            timestep_cnt += result["timestep_cnt"]
        
        final_eval_result = {
            "model_path": self.model.model_path,
            "data_path": self.root_path,
            "input_length": self.input_length,
            "predict_length": self.predict_length,
            "batch_size": self.batch_size,
            "benchmark_result": {
                "time_seq_loss": {},
            },
            "per_dataset_results": per_dataset_results
        }

        for i in range(len(metric_lists[0])):
            metric_name = metric_lists[0][i].name
            final_eval_result["benchmark_result"]["time_seq_loss"][metric_name] = ( sum(metric[i].time_seq_loss for metric in metric_lists) / timestep_cnt ).tolist()
        
        if self.is_wavelet_moe:
            final_eval_result["benchmark_result"]["wavelet_seq_loss"] = {}
            for i in range(len(metric_lists[0])):
                metric_name = metric_lists[0][i].name  
                final_eval_result["benchmark_result"]["wavelet_seq_loss"][metric_name] = ( sum(metric[i].wavelet_seq_loss for metric in metric_lists) / timestep_cnt ).tolist()

        with open(os.path.join(self.output_path, f"{self.file_name}.txt"), "w", encoding="utf-8") as f:
            json.dump(final_eval_result, f, ensure_ascii=False, indent=4)

        return final_eval_result
        


        



