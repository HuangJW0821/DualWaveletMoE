from datasets import load_from_disk
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os

dataset_path = "/data/home/jiawei/PersonalFiles/Wavelet_Time_Series/datasets/PEMS03_processed"
output_path = "/data/home/jiawei/PersonalFiles/Wavelet_Time_Series/WaveletMoE_multivariate/figs/exmple_figs"
batch_size = 1

dataset = load_from_disk(dataset_path=dataset_path)
dataset.set_format("numpy")

print(dataset.column_names)

dataloader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=True
)

def plot_timeseries_batch(batch, figsize=(48, 12)):
    for idx in range(len(batch["data"])):
        data = batch["data"][idx]

        data = np.array(data)
        num_series, seq_len = data.shape

        cols = 1
        rows = num_series

        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if rows * cols == 1:
            axes = np.array([axes])
        else:
            axes = np.array(axes).reshape(-1)

        for i in range(num_series):
            axes[i].plot(data[i])
            axes[i].set_title(f"Series {i}")
            axes[i].set_xlabel("Time")
            axes[i].set_ylabel("Value")

        for j in range(i + 1, rows * cols):
            axes[j].axis("off")

        plt.tight_layout()
        plt.show()
        plt.legend()
        plt.savefig(os.path.join(output_path, f"example{idx}.png"))

for batch in dataloader:
    plot_timeseries_batch(batch)
    break

