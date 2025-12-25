#!/usr/bin/env python
# -*- coding:utf-8 _*-
import os
import argparse
from warnings import warn

from wavelet_moe.evaluation.eval_runner import EvaluationRunner
from wavelet_moe.evaluation.eval_models import WaveletMoEForEvaluation, TimeMoEForEvaluation, ChronosForEvaluation, MoiraiFamilyForEvaluation

def main(args):
    local_rank = int(os.getenv('LOCAL_RANK') or 0)

    input_length = args.input_length
    prediction_length = args.prediction_length

    if "TimeMoE" in args.model:
        # model_path = f"Maple728/TimeMoE-{size}", choose size from ["50M", "200M"]
        model = TimeMoEForEvaluation(
            model_path = args.model,
            device = f"cuda:{local_rank}",
            input_length = input_length,
            prediction_length = prediction_length,
        )
    elif "chronos" in args.model:
        # WARNING: Chronos family requires package `chronos-forecasting==2.2.1`, may cause dependency conflicts.
        warn("Chronos family requires package `chronos-forecasting==2.2.1`, may cause dependency conflicts.", UserWarning)

        # model_path = f"amazon/chronos-t5-{size}", choose size from ["tiny", "mini", "small", "base", "large"]
        # respectively correspond to param size [8M, 20M, 46M, 200M, 710M]
        model = ChronosForEvaluation(
            model_path = args.model,
            device = f"cuda:{local_rank}",
            input_length = input_length,
            prediction_length = prediction_length,
        )
    elif "moirai" in args.model:
        # WARNING: Moirai family requires package `uni2ts==2.0.0`, may cause dependency conflicts.
        warn("Moirai family requires package `uni2ts==2.0.0`, may cause dependency conflicts.", UserWarning)
        
        # wrap moirai family in one class

        # for moirai-1, model_path = f"Salesforce/moirai-1.0-R-{size}", choose size from ["small", "base", "large"]
        # respectively correspond to param size [14M, 91M, 311M]

        # for moirai-moe, model_path = f"Salesforce/moirai-moe-1.0-R-{size}", choose size from ["small", "base"]
        # respectively correspond to activated param size [11M, 86M] ([117M, 935M] in total)

        # for moirai-2, model_path = f"Salesforce/moirai-2.0-R-{size}", choose size from ["small"]
        # respectively correspond to param size [11M]
        model = MoiraiFamilyForEvaluation(
            model_path = args.model,
            device = f"cuda:{local_rank}",
            input_length = input_length,
            prediction_length = prediction_length,
        )
    else:
        # model_path = <local_model_ckpt_path>
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
        use_per_sample_norm = args.use_per_sample_norm,
        num_worker = args.num_worker,
        draw_prediction_result = args.draw_prediction_result,
    )

    eval_runner.evaluate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('WaveletMoE Evaluate')
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='Salesforce/moirai-1.0-R-base',
        help='Model path'
    )
    parser.add_argument(
        '--dataset_path', '-d',
        type=str,
        default="/data/home/dataset/wavelet_moe_benchmark",
        help='Benchmark data path'
    )
    parser.add_argument(
        '--output_path', '-o',
        type=str,
        default="/data/home/jiawei/PersonalFiles/Wavelet_Time_Series/DualWaveletMoE/logs/moirai/moirai_1_0_base",
        help='Output path'
    )

    parser.add_argument(
        '--batch_size', '-b',
        type=int,
        default=16,
        help='Batch size of evaluation'
    )

    # WARNING: the argument `input_length` & `prediction_length` denote the number of patches, with `patch_size==8` by default.
    # If `patch_size` is adjusted, please ensure that `input_length` & `prediction_length` are updated accordingly.
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
        "--draw_prediction_result", 
        action="store_true",
        help="draw prediction result of first batch, save in args.output_path"
    )
   
    args = parser.parse_args()

    main(args)
