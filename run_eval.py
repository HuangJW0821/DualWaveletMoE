#!/usr/bin/env python
# -*- coding:utf-8 _*-
import os
import argparse

from wavelet_moe.evaluation.eval_runner import EvaluationRunner
from wavelet_moe.evaluation.eval_models import WaveletMoEForEvaluation, TimeMoEForEvaluation, ChronosForEvaluation

def main(args):
    local_rank = int(os.getenv('LOCAL_RANK') or 0)

    input_length = args.input_length
    prediction_length = args.prediction_length

    if "TimeMoE" in args.model:
        model = TimeMoEForEvaluation(
            model_path = args.model,
            device = f"cuda:{local_rank}",
            input_length = input_length,
            prediction_length = prediction_length,
        )
    elif "chronos" in args.model:
        model = ChronosForEvaluation(
            model_path = args.model,
            device = f"cuda:{local_rank}",
            input_length = input_length,
            prediction_length = prediction_length,
        )
    else:
        model = WaveletMoEForEvaluation(
            model_path = args.model,
            device = f"cuda:{local_rank}",
            input_length = input_length,
            prediction_length = prediction_length,
        )

    eval_runner = EvaluationRunner(
        model = model,
        root_path = args.dataset_path,
        output_path = args.output_path,
        input_length = args.input_length,
        predict_length = args.prediction_length,
        batch_size = args.batch_size,
        patch_size = model.patch_size,  # load patch_size from model wrapper since WaveletMoE migh have dynamic patch_size
        num_worker = args.num_worker,
        draw_prediciton_result = args.draw_prediciton_result,
    )

    eval_runner.evaluate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('WaveletMoE Evaluate')
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='Maple728/TimeMoE-50M',
        help='Model path'
    )
    parser.add_argument(
        '--dataset_path', '-d',
        type=str,
        default="/data/home/dataset/gifteval_benchmark_strictly_processed_copy",
        help='Benchmark data path'
    )
    parser.add_argument(
        '--output_path', '-o',
        type=str,
        default="/data/home/jiawei/PersonalFiles/Wavelet_Time_Series/DualWaveletMoE/logs/timemoe_50M",
        help='Output path'
    )

    parser.add_argument(
        '--batch_size', '-b',
        type=int,
        default=16,
        help='Batch size of evaluation'
    )

    parser.add_argument(
        '--input_length', '-i',
        type=int,
        default=64,
        help='Input length'
    )
    parser.add_argument(
        '--prediction_length', '-p',
        type=int,
        default=12,
        help='Prediction length'
    )

    parser.add_argument(
        '--num_worker',
        type=int,
        default=8,
    )

    parser.add_argument(
        "--draw_prediciton_result", 
        default=True,
        help="draw prediction result of first batch, save in args.output_path"
    )
   
    args = parser.parse_args()

    main(args)
