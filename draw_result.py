import os
import argparse

from wavelet_moe.evaluation.prediction_result_painter import MultipleModelPredictionResultPainter
from wavelet_moe.evaluation.eval_models import ModelForEvaluation, WaveletMoEForEvaluation, TimeMoEForEvaluation, ChronosForEvaluation, MoiraiFamilyForEvaluation

# -------------- Set Arguments Manually ------------
LOCAL_RANK = int(os.getenv('LOCAL_RANK') or 0)
INPUT_LENGTH = 64
PREDICTION_LENGTH = 12
PATCH_SIZE = 8

# -------------- Please Add Model Here Manually --------------
# tips: you can add arbitrary number of models
# MODEL_LIST: List[model: ModelForEvaluation]
MODEL_LIST = [
    WaveletMoEForEvaluation(
        model_path = "/data/home/weibin/ckpt_new50m/2/checkpoint-20000",
        device = f"cuda:{LOCAL_RANK}",
        input_length = INPUT_LENGTH,
        prediction_length = PREDICTION_LENGTH
    ),

    # TimeMoEForEvaluation(
    #     model_path = "Maple728/TimeMoE-50M",
    #     device = f"cuda:{LOCAL_RANK}",
    #     input_length = INPUT_LENGTH,
    #     prediction_length = PREDICTION_LENGTH
    # ),

    MoiraiFamilyForEvaluation(
        model_path = "Salesforce/moirai-1.1-R-base",
        device = f"cuda:{LOCAL_RANK}",
        input_length = INPUT_LENGTH,
        prediction_length = PREDICTION_LENGTH
    ),

    # MoiraiFamilyForEvaluation(
    #     model_path = "Salesforce/moirai-moe-1.0-R-small",
    #     device = f"cuda:{LOCAL_RANK}",
    #     input_length = INPUT_LENGTH,
    #     prediction_length = PREDICTION_LENGTH
    # ),

    MoiraiFamilyForEvaluation(
        model_path = "Salesforce/moirai-2.0-R-small",
        device = f"cuda:{LOCAL_RANK}",
        input_length = INPUT_LENGTH,
        prediction_length = PREDICTION_LENGTH
    )
]

# MODEL_NAME_LIST: List[model_name: str]
# use as label name in fig
MODEL_NAME_LIST = [
    "DualWaveletMoE-50M (weibin, 2-20k)",
    # "TimeMoE-50M",
    "Moirai-1.1-base",
    # "Moirai-MoE-small",
    "Moirai-2.0-small"
]

# COLOR_NAME_LIST: List[color_name: str]
# use as color in fig
COLOR_NAME_LIST = [
    "red",
    "green",
    "blue",
    # "orange",
    # "purple"
]

def main(args):
    painter = MultipleModelPredictionResultPainter(
        model_list = MODEL_LIST,
        model_name_list = MODEL_NAME_LIST,
        color_name_list = COLOR_NAME_LIST,
        root_path = args.dataset_path,
        output_path = args.output_path,
        input_length = INPUT_LENGTH,
        predict_length = PREDICTION_LENGTH,
        batch_size = args.batch_size,
        batch_num = args.batch_num,
        patch_size = PATCH_SIZE,
        num_worker = args.num_worker,
        use_per_sample_norm = args.use_per_sample_norm
    )

    painter.draw_result()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('WaveletMoE Evaluate')

    parser.add_argument(
        '--dataset_path', '-d',
        type=str,
        default="/data/home/dataset/gifteval_benchmark_strictly_processed_copy",
        help='Benchmark data path'
    )
    
    parser.add_argument(
        '--output_path', '-o',
        type=str,
        default="/data/home/jiawei/PersonalFiles/Wavelet_Time_Series/DualWaveletMoE/figs/wavelet_moirai_base",
        help='Output path'
    )
    
    parser.add_argument(
        '--batch_size', '-bsz',
        type=int,
        default=16,
        help='Batch size to paint'
    )
    
    parser.add_argument(
        '--batch_num', '-bn',
        type=int,
        default=4,
        help='Batch num to paint'
    )
    
    parser.add_argument(
        '--num_worker',
        type=int,
        default=8,
    )

    parser.add_argument(
        "--use_per_sample_norm", 
        action="store_true",
        help="use per-sample, sequence-wise norm in data collator, otherwise use batch-level norm."
    )
   
    args = parser.parse_args()
    
    main(args)