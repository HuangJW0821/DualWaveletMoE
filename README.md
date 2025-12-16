## TODO

Check `TODO` mark in src for more details.

1. **`./wavelet_moe/evaluation` 需要修改以适配Dual结构**：建议从`./wavelet_moe/evaluation/eval_models.py`开始入手。注意`model.generate()`方法只接收`input_ids`作为输入，而`WaveletTimeSeriesDataCollator`生成的batch将时序序列和小波序列分别输出（`time_seq`与`wavelet_seq`，形状均为`[batch_sz, token_num, patch_sz]`）。在调用`model.generate()`前需将两个序列concat为`[batch_sz, token_num, patch_sz * 2]`的`input_ids`。`./prediction_example.py`提供了参考。

2. **Time-300B Dataset**：已将原先用于加载Time-300B的若干`Dataset`类转移至`./wavelet_moe/datasets/dataset_time300B.py`，还需添加均衡采样sampler与sequence patching相关组件，并修改`./wavelet_moe/runner.py`中的`WaveletMoeRunner._prepare_time300b_dataset()`方法。若要使用Time-300B进行训练，还需将`WaveletMoeRunner.train_model()`方法中的`train_dataset, val_dataset = self._prepare_chronos_dataset(train_config)`语句（`line 189`）修改为`train_dataset, val_dataset = self._prepare_time300b_dataset(train_config)`。

3. **数据准备**：注意当前代码数据加载逻辑与上一个版本有所不同，tokenize的部分（包括DWT与patching）交由collator `./wavelet_moe/datasets/wavelet_data_collator.py`中的`WaveletTimeSeriesDataCollator`类处理。


## Quick Start

`./prediction_example.py` provide a example to predict with WaveletMoE

```
python prediction_example.py
```

## Train

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

It would automatically load test set of Chronos dataset from `<data_path>` and use it for evaluation. Modify as you need if you want to load other dataset.

The evaluation result will be saved at `<model_path>/eval_result_<input_length>_to_<prediction_length>_tokens.txt`, MAE & MSE are used as metrics.


### Draw test result

Run  ` ./draw_result.py` using command:

``` 
python draw_result.py -m <model_path> -b <baseline_path> -o <output_path> --input_length <input_token_num> --prediction_length <prediction_token_num>

```

It would load models from `<model_path>` & `<baseline_path>`, and draw prediction result.
