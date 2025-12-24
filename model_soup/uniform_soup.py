import os
import argparse
import torch
from transformers import AutoConfig
from safetensors.torch import load_file, save_file

def uniform_soup(
    ckpt_dirs,
    output_dir,
    dtype=torch.bfloat16,
):
    """
    ckpt_dirs: list of checkpoint directories (Trainer.save_model outputs)
    output_dir: directory to save souped model
    """

    assert len(ckpt_dirs) >= 2, "Need at least two checkpoints for soup"

    print(f"==> Loading checkpoints:")
    for d in ckpt_dirs:
        print(f"  - {d}")

    # ---------- load first model weights ----------
    base_ckpt = ckpt_dirs[0]
    base_weights_path = os.path.join(base_ckpt, "model.safetensors")
    assert os.path.exists(base_weights_path)

    base_state = load_file(base_weights_path)
    soup_state = {}

    # init accumulator
    for k, v in base_state.items():
        soup_state[k] = v.to(dtype).clone()

    # ---------- accumulate ----------
    for ckpt_dir in ckpt_dirs[1:]:
        weight_path = os.path.join(ckpt_dir, "model.safetensors")
        assert os.path.exists(weight_path)

        state = load_file(weight_path)

        # safety check
        if state.keys() != soup_state.keys():
            raise ValueError(f"State dict keys mismatch in {ckpt_dir}")

        for k in soup_state:
            if state[k].shape != soup_state[k].shape:
                raise ValueError(f"Shape mismatch for {k}")
            soup_state[k] += state[k].to(dtype)

    # ---------- average ----------
    num_models = len(ckpt_dirs)
    for k in soup_state:
        soup_state[k] /= num_models

    # ---------- save ----------
    os.makedirs(output_dir, exist_ok=True)

    # copy config files from first checkpoint
    for fname in [
        "config.json",
        "generation_config.json",
    ]:
        src = os.path.join(base_ckpt, fname)
        if os.path.exists(src):
            dst = os.path.join(output_dir, fname)
            os.system(f"cp {src} {dst}")

    save_file(
        soup_state,
        os.path.join(output_dir, "model.safetensors"),
        metadata={"format": "pt"}
    )

    print(f"\n✅ Uniform soup saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpts",
        type=str,
        nargs="+",
        required=True,
        help="List of checkpoint directories",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for souped model",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["fp32", "bf16", "fp16"],
    )

    args = parser.parse_args()

    dtype_map = {
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }

    uniform_soup(
        ckpt_dirs=args.ckpts,
        output_dir=args.output,
        dtype=dtype_map[args.dtype],
    )
