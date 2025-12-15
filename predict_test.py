#!/usr/bin/env python
# -*- coding:utf-8 _*-
import json
import os
import argparse
import numpy as np
import logging
import torch
import torch.distributed as dist
from torch.utils.data import DistributedSampler, DataLoader
from tqdm import tqdm

from DualWaveletMoE.wavelet_moe.datasets.wavelet_moe_dataset import ChronosTensorDataset, WaveletTimeSeriesDataset
from wavelet_moe.models.modeling_wavelet_moe import WaveletMoeForPrediction

def setup_nccl(rank, world_size, master_addr='127.0.0.1', master_port=9899):
    dist.init_process_group("nccl", init_method='tcp://{}:{}'.format(master_addr, master_port), rank=rank,
                            world_size=world_size)


class WaveletMoE:
    def __init__(self, model_path, device, input_length, prediction_length, **kwargs):
        
        model = WaveletMoeForPrediction.from_pretrained(
            model_path,
            device_map=device,
            # attn_implementation='flash_attention_2',
            torch_dtype='auto',
        )

        self.model = model
        self.device = device
        self.input_length = input_length
        self.prediction_length = prediction_length
        self.model.eval()

    def predict(self, batch):
        model = self.model
        device = self.device

        input_length = self.input_length
        prediction_length = self.prediction_length

        if input_length + prediction_length > batch.shape[1]:
            raise ValueError(f"Input length + Pred length [{input_length} + {prediction_length} = {input_length + prediction_length}] should be shorter than token_num [{batch.shape[1]}]")
        
        inputs = batch[:, : input_length, :].to(device).to(model.dtype)
        labels = batch[:, input_length : input_length + prediction_length, :].to(device).to(model.dtype)
        outputs = model.generate(
            inputs=inputs,
            max_length=batch.shape[1],
        )

        preds = outputs[:, -prediction_length :, :]

        return preds, labels


def evaluate(args):
    batch_size = args.batch_size
    input_length = args.input_length
    prediction_length = args.prediction_length

    master_addr = os.getenv('MASTER_ADDR', '127.0.0.1')
    master_port = os.getenv('MASTER_PORT', args.port)
    world_size = int(os.getenv('WORLD_SIZE') or 1)
    rank = int(os.getenv('RANK') or 0)
    local_rank = int(os.getenv('LOCAL_RANK') or 0)

    if torch.cuda.is_available():
        try:
            setup_nccl(rank=rank, world_size=world_size, master_addr=master_addr, master_port=master_port)
            device = f"cuda:{local_rank}"
            is_dist = True
        except Exception as e:
            print('Error: ', f'Setup nccl fail, so set device to cpu: {e}')
            device = 'cpu'
            is_dist = False
    else:
        device = 'cpu'
        is_dist = False

    model = WaveletMoE(
        args.model,
        device,
        input_length = input_length,
        prediction_length=prediction_length
    )


    context_length = args.input_length + args.prediction_length
    batch = torch.randn((batch_size, context_length, 16))
    preds, labels = model.predict(batch)

    a, b = preds, labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser('WaveletMoE Evaluate')
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='/data/home/jiawei/PersonalFiles/Wavelet_Time_Series/WaveletMoE/log/20251106_lr/3e-4/from_ckpt_160000',
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
    parser.add_argument(
        '--port',
        type=int,
        default=9899,
    )
   
    args = parser.parse_args()
    
    evaluate(args)