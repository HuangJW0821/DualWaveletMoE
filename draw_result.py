import os
import argparse

from wavelet_moe.evaluation.prediction_result_painter import MultipleModelPredictionResultPainter
from wavelet_moe.evaluation.eval_models import ModelForEvaluation, WaveletMoEForEvaluation, TimeMoEForEvaluation, Chronos2ForEvaluation, ChronosForEvaluation

# -------------- Set Arguments Manually ------------
LOCAL_RANK = int(os.getenv('LOCAL_RANK') or 0)
INPUT_LENGTH = 64
PREDICTION_LENGTH = 12

# -------------- Please Add Model Here Manually --------------
# tips: you can add arbitrary number of models
# MODEL_LIST: List[model: ModelForEvaluation]
MODEL_LIST = [

    WaveletMoEForEvaluation(
        model_path = "/data/home/jiawei/PersonalFiles/Wavelet_Time_Series/WaveletMoE_multivariate/logs/uni_50M_syn_from_scratch/checkpoint-50000",
        device = f"cuda:{LOCAL_RANK}",
        input_length = INPUT_LENGTH,
        prediction_length = PREDICTION_LENGTH
    ),

    TimeMoEForEvaluation(
        model_path = "Maple728/TimeMoE-50M",
        device = f"cuda:{LOCAL_RANK}",
        input_length = INPUT_LENGTH,
        prediction_length = PREDICTION_LENGTH
    )
]

# MODEL_NAME_LIST: List[model_name: str]
# use as label name in fig
MODEL_NAME_LIST = [
    "WaveletMoE-50M (synth data, 50k)",
    "TimeMoE-50M"
]

# COLOR_NAME_LIST: List[color_name: str]
# use as color in fig
COLOR_NAME_LIST = [
    "red",
    # "blue",
    # "yellow",
    "green"
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
        predict_target_only = args.predict_target_only,
    )

    painter.draw_result()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('WaveletMoE Evaluate')

    parser.add_argument(
        '--dataset_path', '-d',
        type=str,
        default="/data/home/dataset/wavelet_moe_multivariate/bench/high_low_freq_syn_test",
        help='Benchmark data path'
    )
    parser.add_argument(
        '--output_path', '-o',
        type=str,
        default="/data/home/jiawei/PersonalFiles/Wavelet_Time_Series/WaveletMoE_multivariate/figs/high_low_freq_eval",
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