#!/usr/bin/env python
# -*- coding:utf-8 _*-
import os
import argparse

from wavelet_moe.evaluation.eval_runner import EvaluationRunner
from wavelet_moe.evaluation.eval_models import WaveletMoEForEvaluation, TimeMoEForEvaluation, Chronos2ForEvaluation

def main(args):
    local_rank = int(os.getenv('LOCAL_RANK') or 0)

    if "TimeMoE" in args.model:
        model = TimeMoEForEvaluation(
            model_path = args.model,
            device = f"cuda:{local_rank}",
            input_length = args.input_length,
            prediction_length = args.prediction_length
        )
    elif "chronos2" in args.model:
        model = Chronos2ForEvaluation(
            model_path = args.model,
            device = f"cuda:{local_rank}",
            input_length = args.input_length,
            prediction_length = args.prediction_length
        )
    else:
        model = WaveletMoEForEvaluation(
            model_path = args.model,
            device = f"cuda:{local_rank}",
            input_length = args.input_length,
            prediction_length = args.prediction_length
        )

    eval_runner = EvaluationRunner(
        model = model,
        root_path = args.dataset_path,
        output_path = args.output_path,
        input_length = args.input_length,
        predict_length = args.prediction_length,
        batch_size = args.batch_size,
        num_worker = args.num_worker,
        draw_prediciton_result = args.draw_prediciton_result,
        predict_target_only = args.predict_target_only
    )

    eval_runner.evaluate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('WaveletMoE Evaluate')
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='/data/home/jiawei/PersonalFiles/Wavelet_Time_Series/WaveletMoE_multivariate/logs/uni_50M_syn_from_scratch/checkpoint-50000',
        help='Model path'
    )
    parser.add_argument(
        '--dataset_path', '-d',
        type=str,
        default="/data/home/dataset/wavelet_moe_multivariate/bench/high_low_freq_syn_test",
        help='Benchmark data path'
    )
    parser.add_argument(
        '--output_path', '-o',
        type=str,
        default="/data/home/jiawei/PersonalFiles/Wavelet_Time_Series/WaveletMoE_multivariate/logs/uni_50M_syn_from_scratch/checkpoint-50000",
        help='Output path'
    )

    parser.add_argument(
        '--batch_size', '-b',
        type=int,
        default=5,
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
        default=False,
        help="draw prediction result of first batch, save in args.output_path"
    )

    parser.add_argument(
        "--predict_target_only", 
        action="store_true",
        help="only predict target variates, dataset should provide info about target var & covariate, " \
        "otherwise all channels will be consider as target variates"
    )
    # parser.add_argument(
    #     "--predict_target_only", 
    #     default=True,
    #     help="only predict target variates, dataset should provide info about target var & covariate, " \
    #     "otherwise all channels will be consider as target variates"
    # )
   
    args = parser.parse_args()

    main(args)
