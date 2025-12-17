import os
import math
import random
from functools import reduce
from operator import mul

import torch

from wavelet_moe.datasets.wavelet_moe_dataset import TimeSeriesSingleDataset, TimeSeriesMultipleDataset
from wavelet_moe.datasets.dataset_time300B import WaveletMoeMultipleDomainDataset, WaveletMoeWindowDataset, WaveletMoeWindowTensorDataset
from wavelet_moe.datasets.wavelet_data_collator import WaveletTimeSeriesDataCollator
from wavelet_moe.datasets.balanced_sampler import MultiDatasetBalancedSampler, DistributedBatchSampler
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

    def load_model(self, model_path: str = None, from_scratch: bool = False, **kwargs):
        if model_path is None:
            model_path = self.model_path

        if from_scratch:
            config = WaveletMoeConfig.from_pretrained(model_path)
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

        # set training arguments
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
                from_scratch=from_scratch,
                torch_dtype=torch_dtype
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
        # dataset = TimeSeriesMultipleDataset(root_path = train_config["data_path"])
        train_dataset, val_dataset = self._prepare_chronos_dataset(train_config)
        # train_dataset, val_dataset = self._prepare_single_dataset(train_config)

        # #use Time-300B dataset
        # train_dataset, val_dataset = self._prepare_time300b_dataset(train_config)

        data_collator = WaveletTimeSeriesDataCollator(
            batch_size = micro_batch_size,
            patch_size = model.config.patch_size,
            wavelet_function = model.config.wavelet_function,
            signal_extension_mode = model.config.wavelet_signal_extension_mode,
            level = model.config.wavelet_dwt_level,
            normalization_method = model.config.normalization_method
        )

        # init trainer, start training & save result

        # use balanced sampler or not
        # use_balanced_sampler = bool(train_config.get('use_balanced_sampler', False))
        # TrainerCls = BalancedWaveletMoeTrainer if use_balanced_sampler else WaveletMoeTrainer
        # trainer = TrainerCls(
        #     model = model,
        #     args = training_args,
        #     train_dataset = train_dataset,
        #     data_collator= data_collator,
        #     needed_column_names = ["data", "loss_mask"],
        # )
        trainer = WaveletMoeTrainer(
            model = model,
            args = training_args,
            train_dataset = train_dataset,
            data_collator= data_collator,
            needed_column_names = ["data", "loss_mask"],
        )
        
        trainer.train()
        trainer.save_model()
        log_in_local_rank_0(f'Saving model to {self.output_path}')

        return trainer.model
    

    # Adaption for Time-300B
    def _prepare_time300b_dataset(self, config):
        data_path = config["data_path"]

        context_length = int(config.get("context_length", config["max_length"]))
        prediction_length = int(config.get("prediction_length", 0))
        window_size = context_length + prediction_length
        log_in_local_rank_0("Window size:", window_size)

        stride = window_size

        use_lazy_window = bool(config.get("lazy_window", False))
        cache_dir = config.get("cache_dir", None)
        use_cache = bool(config.get("use_dataset_cache", True))

        base_ds = WaveletMoeMultipleDomainDataset(
            root_path=data_path,
            dataset_cache_path=cache_dir,
            use_dataset_cache=use_cache,
        )

        window_ds = WaveletMoeWindowDataset(
            dataset=base_ds,
            context_length=context_length,
            prediction_length=prediction_length,
            stride=stride,
            use_lazy_window=False,
            dataset_cache_path=cache_dir,
            use_dataset_cache=use_cache,
        )

        train_ds = WaveletMoeWindowTensorDataset(
            window_dataset=window_ds,
            split= "train",
            test_size=float(config.get("test_size", 0.0)),
            seed=config.get("data_seed", self.seed),
        )

        # val_ds = WaveletMoeWindowTensorDataset(
        #     window_dataset=window_ds,
        #     split= "test",
        #     test_size=float(config.get("test_size", 0.0)),
        #     seed=config.get("data_seed", self.seed),
        # )
        val_ds = None


        if hasattr(window_ds, "window_list"):
            train_ds.window_list = window_ds.window_list
        if hasattr(window_ds, "num_subsets"):
            train_ds.num_subsets = window_ds.num_subsets
        if hasattr(window_ds, "subset_names"):
            train_ds.subset_names = window_ds.subset_names

        return train_ds, val_ds


    # TODO: mind this.
    def _prepare_chronos_dataset(self, config):
        train_dataset = TimeSeriesMultipleDataset(
            root_path = config["data_path"],
            dataset_names = ["chronos_processed"]
        )
        val_dataset = None
        return train_dataset, val_dataset
    

    def _prepare_single_dataset(self, config):
        train_dataset = TimeSeriesSingleDataset(ds_path = config["data_path"])
        val_dataset = None
        return train_dataset, val_dataset


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

# Balanced sampler trainer
class BalancedWaveletMoeTrainer(WaveletMoeTrainer):
    """WaveletMoeTrainer + MultiDatasetBalancedSampler (batch-level balanced sampling)."""

    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        per_device_bs = int(self.args.per_device_train_batch_size)
        sampler = MultiDatasetBalancedSampler(
            dataset=self.train_dataset,
            batch_size=per_device_bs,
            shuffle=True,
            seed=getattr(self.args, "data_seed", None),
        )

        sampler = DistributedBatchSampler.maybe_wrap(sampler)
        from torch.utils.data import DataLoader
        return DataLoader(
            self.train_dataset,
            batch_sampler=sampler,
            collate_fn=self.data_collator,
            num_workers=int(getattr(self.args, "dataloader_num_workers", 0) or 0),
            pin_memory=bool(getattr(self.args, "dataloader_pin_memory", True)),
            persistent_workers=(int(getattr(self.args, "dataloader_num_workers", 0) or 0) > 0),
        )
