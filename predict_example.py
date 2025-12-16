#!/usr/bin/env python
# -*- coding:utf-8 _*-
import os
import argparse
import torch

from wavelet_moe.models.modeling_wavelet_moe import WaveletMoeForPrediction

# Quick start

def predict_once(args):
    batch_size = args.batch_size
    input_length = args.input_length
    prediction_length = args.prediction_length
    patch_size = 8

    local_rank = int(os.getenv('LOCAL_RANK') or 0)
    device = f"cuda:{local_rank}"

    # init model
    model = WaveletMoeForPrediction.from_pretrained(
        args.model,
        device_map = device,
        torch_dtype = 'auto',
    )
    model.eval()

    # generate random tensor as input
    # since transformers model only take `input_ids` or `input_embeds` as input,
    # we should concat `time_seq` and `wavelet_seq` at `dim=2` before input
    time_seq = torch.randn((batch_size, input_length, patch_size))
    wavelet_seq = torch.randn((batch_size, input_length, patch_size))
    input_ids = (torch.cat((time_seq, wavelet_seq), dim=2)).to(device)

    # predict
    outputs = model.generate(
        input_ids = input_ids,
        max_length = input_length + prediction_length
    )

    # model's outputs are also concat of `time_seq` & `wavelet_seq`
    predictions = outputs[:, -prediction_length:, :]
    time_prediction = predictions[:, :, : patch_size ]
    wavelet_prediction = predictions[:, :, patch_size :]

    print(time_prediction, wavelet_prediction)
    print(time_prediction.shape, wavelet_prediction.shape)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('WaveletMoE prediction example')
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='/data/home/jiawei/PersonalFiles/Wavelet_Time_Series/DualWaveletMoE/logs/small_model_for_pred_test',
        help='Model path'
    )

    parser.add_argument(
        '--batch_size', '-b',
        type=int,
        default=10,
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
   
    args = parser.parse_args()
    
    predict_once(args)