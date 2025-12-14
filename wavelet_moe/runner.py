import os
import math
import random
from functools import reduce
from operator import mul

import torch

from wavelet_moe.datasets.wavelet_moe_dataset import TimeSeriesSingleDataset, TimeSeriesMultipleDataset
from wavelet_moe.datasets.wavelet_data_collator import WaveletTimeSeriesDataCollator
from wavelet_moe.models.modeling_wavelet_moe import WaveletMoeForPrediction, WaveletMoeConfig
from wavelet_moe.trainer.hf_trainer import WaveletMoETrainingArguments, WaveletMoeTrainer
from wavelet_moe.utils.dist_util import get_world_size
from wavelet_moe.utils.log_util import logger, log_in_local_rank_0


class WaveletMoeRunner:
    def __init__(
            self,
            model_path: str = None,
            output_path: str = 'logs/wavelet_moe',
            seed: int = 9899
    ):
        self.model_path = model_path
        self.output_path = output_path
        self.seed = seed

    def load_model(self, model_path: str = None, from_scatch: bool = False, **kwargs):
        if model_path is None:
            model_path = self.model_path

        loss_func = kwargs.pop("loss_func", None)
        if loss_func is None:
            loss_func = "huber"
        
        if loss_func in ["huber", "mse"]:
            log_in_local_rank_0(f"Use loss function: {loss_func}")
        else:
            raise ValueError(f"Unknown loss function: {loss_func}")
        kwargs["loss_func"] = loss_func

        if from_scatch:
            config = WaveletMoeConfig.from_pretrained(model_path, loss_func=loss_func)
            model = WaveletMoeForPrediction(config)
        else:
            model = WaveletMoeForPrediction.from_pretrained(model_path, **kwargs)
        return model

    def train_model(self, from_scratch: bool = False, **kwargs):
        setup_seed(self.seed)

        train_config = kwargs

        num_devices = get_world_size()

        global_batch_size = train_config.get('global_batch_size', None)
        micro_batch_size = train_config.get('micro_batch_size', None)

        # check paparelism batch size validility
        if global_batch_size is None and micro_batch_size is None:
            raise ValueError('Must set at lease one argument: "global_batch_size" or "micro_batch_size"')
        elif global_batch_size is None:
            gradient_accumulation_steps = 1
            global_batch_size = micro_batch_size * num_devices
        elif micro_batch_size is None:
            micro_batch_size = math.ceil(global_batch_size / num_devices)
            gradient_accumulation_steps = 1
        else:
            if micro_batch_size * num_devices > global_batch_size:
                if num_devices > global_batch_size:
                    micro_batch_size = 1
                    global_batch_size = num_devices
                else:
                    micro_batch_size = math.ceil(global_batch_size / num_devices)
            gradient_accumulation_steps = math.ceil(global_batch_size / num_devices / micro_batch_size)
            global_batch_size = int(gradient_accumulation_steps * num_devices * micro_batch_size)

        # use epoch or train step
        # one is valid, another is set to -1
        if ('train_steps' in train_config
                and train_config['train_steps'] is not None
                and train_config['train_steps'] > 0):
            train_steps = int(train_config["train_steps"])
            num_train_epochs = -1
        else:
            train_steps = -1
            num_train_epochs = _safe_float(train_config.get("num_train_epochs", 1))


        # set training precision (torch_dtype)
        precision = train_config.get('precision', 'bf16')
        if precision not in ['bf16', 'fp16', 'fp32']:
            log_in_local_rank_0(f'Precision {precision} is not set, use fp32 default!', type='warn')
            precision = 'fp32'

        if precision == 'bf16':
            torch_dtype = torch.bfloat16
        elif precision == 'fp16':
            # use fp32 to load model but uses fp15 to train model
            torch_dtype = torch.float32
        elif precision == 'fp32':
            torch_dtype = torch.float32
        else:
            raise ValueError(f'Unsupported precision {precision}')

        # log in main node (rank 0 process)
        log_in_local_rank_0(f'Set global_batch_size to {global_batch_size}')
        log_in_local_rank_0(f'Set micro_batch_size to {micro_batch_size}')
        log_in_local_rank_0(f'Set gradient_accumulation_steps to {gradient_accumulation_steps}')
        log_in_local_rank_0(f'Set precision to {precision}')
        log_in_local_rank_0(f'Set normalization to {train_config["normalization_method"]}')

        # set training arguments
        # class WaveletMoETrainingArguments is herit from transformers.TrainingArguments
        # only added a component min_learning_rate
        training_args = WaveletMoETrainingArguments(
            output_dir=self.output_path,
            num_train_epochs=num_train_epochs,
            # use_cpu=True,
            max_steps=train_steps,
            evaluation_strategy=train_config.get("evaluation_strategy", 'no'),
            eval_steps=_safe_float(train_config.get("eval_steps", None)),
            save_strategy=train_config.get("save_strategy", "no"),
            save_steps=_safe_float(train_config.get("save_steps", None)),
            learning_rate=float(train_config.get("learning_rate", 1e-5)),
            min_learning_rate=float(train_config.get("min_learning_rate", 0)),
            adam_beta1=float(train_config.get("adam_beta1", 0.9)),
            adam_beta2=float(train_config.get("adam_beta2", 0.95)),
            adam_epsilon=float(train_config.get("adam_epsilon", 1e-8)),
            lr_scheduler_type=train_config.get("lr_scheduler_type", 'constant'),
            warmup_ratio=float(train_config.get("warmup_ratio") or 0.0),
            warmup_steps=int(train_config.get("warmup_steps", 0)),
            weight_decay=float(train_config.get("weight_decay", 0.1)),
            per_device_train_batch_size=int(micro_batch_size),
            per_device_eval_batch_size=int(micro_batch_size * 2),
            gradient_accumulation_steps=int(gradient_accumulation_steps),
            gradient_checkpointing=train_config.get("gradient_checkpointing", False),
            bf16=True if precision == 'bf16' else False,
            fp16=True if precision == 'fp16' else False,
            deepspeed=train_config.get("deepspeed"),
            push_to_hub=False,
            logging_first_step=True,
            log_on_each_node=False,
            logging_steps=int(train_config.get('logging_steps', 1)),
            seed=self.seed,
            data_seed=self.seed,
            max_grad_norm=train_config.get('max_grad_norm', 1.0),
            optim=train_config.get('optim', 'adamw_torch'),
            torch_compile=train_config.get('torch_compile', False),
            dataloader_num_workers=train_config.get('dataloader_num_workers') or 2,
            ddp_find_unused_parameters=False,

            logging_dir=os.path.join(self.output_path, 'tb_logs'),
            save_only_model=train_config.get('save_only_model', True),
            save_total_limit=train_config.get('save_total_limit'),
        )

        # check model_path
        # if train from scatch, load model by init a class WaveletMoeForPrediction object
        # otherwise, load from previous checkpoint
        model_path = train_config.pop('model_path', None) or self.model_path
        if model_path is not None:
            model = self.load_model(
                model_path=model_path,
                from_scatch=from_scratch,
                torch_dtype=torch_dtype,
                loss_func = train_config.get("loss_func", "huber")
            )
            log_in_local_rank_0(f'Load model parameters from: {model_path}')
        else:
            raise ValueError('Model path is None')

        # calculate total params num
        num_total_params = 0
        for p in model.parameters():
            num_total_params += reduce(mul, p.shape)

        # print statistics info
        log_in_local_rank_0(f"Train config: {train_config}")
        log_in_local_rank_0(f"Training args: {training_args}")
        log_in_local_rank_0(f"Model config: {model.config}")
        log_in_local_rank_0(f'Number of the model parameters: {length_to_str(num_total_params)}')

        if train_steps > 0:
            total_train_tokens = train_steps * global_batch_size * train_config['max_length']
            log_in_local_rank_0(f'Tokens will consume: {length_to_str(total_train_tokens)}')

        # Training
        # load dataset & data collator
        dataset = TimeSeriesMultipleDataset(root_path = train_config["data_path"])
        data_collator = WaveletTimeSeriesDataCollator(
            batch_size = micro_batch_size,
            patch_size = train_config["patch_size"],
            wavelet_function = train_config["wavelet_function"],
            signal_extension_mode = train_config["wavelet_signal_extension_mode"],
            level = train_config["wavelet_dwt_level"],
            normalization_method = train_config["normalization_method"]
        )

        # init trainer, start training & save result
        trainer = WaveletMoeTrainer(
            model = model,
            args = training_args,
            train_dataset = dataset,
            data_collator= data_collator,
            needed_column_names = ["data", "loss_mask"]
        )
        trainer.train()
        trainer.save_model()
        log_in_local_rank_0(f'Saving model to {self.output_path}')

        return trainer.model


def setup_seed(seed: int = 9899):
    """
    Setup seed for all known operations.

    Args:
        seed (int): seed number.

    Returns:

    """
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def length_to_str(length):
    if length >= 1e12:
        return f'{length / 1e12:.3f}T'
    if length >= 1e9:
        return f'{length / 1e9:.3f}B'
    elif length >= 1e6:
        return f'{length / 1e6:.3f}M'
    else:
        return f'{length / 1e3:.3f}K'


def _safe_float(number):
    if number is None:
        return None
    else:
        return float(number)
