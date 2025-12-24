#!/usr/bin/env python
# -*- coding:utf-8 _*-
import os
import argparse

from wavelet_moe.evaluation.eval_runner import EvaluationRunner
from wavelet_moe.evaluation.eval_models import WaveletMoEForEvaluation, TimeMoEForEvaluation, ChronosForEvaluation

def main(args):
    local_rank = int(os.getenv('LOCAL_RANK') or 0)
    os.makedirs(args.output_path, exist_ok=True)

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

    for dataset_path in args.dataset_paths:
        print(f"\n[INFO] Evaluating benchmark: {dataset_path}")

        eval_runner = EvaluationRunner(
            model=model,
            root_path= dataset_path,
            output_path=args.output_path,
            input_length=args.input_length,
            predict_length=args.prediction_length,
            batch_size=args.batch_size,
            patch_size=model.patch_size,
            # load patch_size from model wrapper since WaveletMoE migh have dynamic patch_size
            use_per_sample_norm=args.use_per_sample_norm,
            num_worker=args.num_worker,
            draw_prediciton_result=args.draw_prediciton_result,
        )

        eval_runner.evaluate()



if __name__ == '__main__':
    parser = argparse.ArgumentParser('WaveletMoE Evaluate')
    parser.add_argument(
        '--model', '-m',
        type=str,
        # default='Maple728/TimeMoE-50M',
        # default='logs/14_uniform_soup',
        # default='amazon/chronos-t5-small',
        default='/data/home/weibin/ckpt_new50M/14/checkpoint-15000',
        help='Model path'
    )
    parser.add_argument(
        '--dataset_paths', '-d',
        type=str,
        nargs='+',
        default=[
            "/data/home/dataset/gifteval_benchmark_strictly_processed_copy",
            "/data/home/dataset/chronos_zero_shot_benchmark",
            "/data/home/dataset/fev_benchmark_processed"
        ],
        # default=["/data/home/dataset/USTD_12G_zero_shot_processed"],
        help='Benchmark data paths'
    )
    parser.add_argument(
        '--output_path', '-o',
        type=str,
        default="results/WaveletMoE",
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
        "--use_per_sample_norm", 
        action="store_true",
        help="use per-sample, sequence-wise norm in data collator, otherwise use batch-level norm."
    )

    parser.add_argument(
        '--num_worker',
        type=int,
        default=8,
    )

    parser.add_argument(
        "--draw_prediciton_result", 
        action="store_true",
        help="draw prediction result of first batch, save in args.output_path"
    )
   
    args = parser.parse_args()

    main(args)
