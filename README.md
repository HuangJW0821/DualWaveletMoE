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
