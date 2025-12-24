#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import argparse
import os
import csv
import re


def main(args):
    # 1. 读取 json / txt
    with open(args.input_path, "r") as f:
        content = json.load(f)

    per_dataset = content["per_dataset_results"]

    # 2. 确保输出目录存在
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    # 3. 写 CSV
    with open(args.output_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)

        # 表头：mse 在前，mae 在后
        writer.writerow(["dataset", "mse", "mae"])

        # 4. 写入每个数据集的结果
        for dataset_name, result in per_dataset.items():
            mse = round(result["time_seq_loss"]["mse"], 4)
            mae = round(result["time_seq_loss"]["mae"], 4)
            clean_name = re.sub(r"_processed.*$", "", dataset_name)

            writer.writerow([clean_name, mse, mae])

    print(f"CSV saved to: {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Convert evaluation txt to csv")
    parser.add_argument(
        "--input_path",
        type=str,
        default="results/WaveletMoE/BENCHMARK[fev_benchmark_processed]_[64 to 12 tokens].txt",
        help="Path to evaluation result txt/json file",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="results/WaveletMoE/fev_benchmark.csv",
        help="Output csv path",
    )

    args = parser.parse_args()
    main(args)
