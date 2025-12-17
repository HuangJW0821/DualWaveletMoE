## TODO

Check `TODO` mark in src for more details.

1. **推理加速**：添加sparse tensor (when fitering KV), KV cache, fast attention的适配。

2. **数据准备**：注意当前代码数据加载逻辑与上一个版本有所不同，tokenize的部分（包括DWT与patching）交由collator `./wavelet_moe/datasets/wavelet_data_collator.py`中的`WaveletTimeSeriesDataCollator`类处理。


## Quick Start

`./prediction_example.py` provide a example to predict with WaveletMoE

```
python prediction_example.py
```

## Train

### Train configuration prepare

Prepare your model setting by `config.json`

### Train from scratch

To leverage a single GPU or multiple GPUs on a single node, use this command:

```
python torch_dist_run.py main.py -m <model_config_path> -d <data_path> -o <output_path> --from_scratch
```

### Train from checkpoint

If a checkpoint is provided, use this command:

```
python torch_dist_run.py main.py -m <checkpoint_path> -d <data_path> -o <output_path> --learning_rate <learning_rate from checkpoint> --warmup_ratio <warmup_ratio>
```

### Train distributedly

Dont forget `CUDA_VISIBLE_DEVICES` & `LOCAL_WORLD_SIZE` if you want to train on multiple GPUS, check `./torch_dist_run.py` for more details.

```
CUDA_VISIBLE_DEVICES=0,1 LOCAL_WORLD_SIZE=2 python torch_dist_run.py main.py
```

### Draw loss curve

You can run ` ./utils/draw_loss_curve.py ` to draw loss curve from a checkpoint

## Evaluation

### Evaluate with local dataset

You can run ` ./run_eval.py` to run evaluation:

```
python run_eval.py -m <model_path> -d <data_path> -o <output_path> --input_length <input_token_num> --prediction_length <prediction_token_num>
```

It would automatically load test dataset from `<data_path>` and use it for evaluation. 

If argument `use_per_sample_norm == True`, data collator will conduct per-sample, sequence-wise scaling; otherwise batch-level scaling.

If argument `draw_prediciton_result == True`, visualization prediction result will also be saved.

The evaluation result will be saved at `<model_path>/eval_result_<input_length>_to_<prediction_length>_tokens.txt`, MAE & MSE are used as metrics.


### Draw test result

Run  ` ./draw_result.py` using command:

``` 
python draw_result.py -m <model_path> -b <baseline_path> -o <output_path> --input_length <input_token_num> --prediction_length <prediction_token_num>

```

It would load models from `<model_path>` & `<baseline_path>`, and draw prediction result.
