import os
from typing import List
from transformers import PretrainedConfig
from time_moe.models.configuration_time_moe import TimeMoeConfig

# derived from time_moe.models.configuration_time_moe.TimeMoeConfig
class WaveletMoeConfig(TimeMoeConfig):
    model_type = "wavelet_moe"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
            self,
            horizon_lengths: List[int] = 1,
            patch_size: int = 8,
            tie_word_embeddings: bool = False,
            wavelet_function: str = "bior2.2",
            wavelet_signal_extension_mode: str = "periodization",
            wavelet_dwt_level: int = 2,
            loss_func: str = "huber",
            time_axis_loss_factor: float = 0.5,
            use_group_attn: bool = False,
            use_channel_axis_loss: bool = False,
            **kwargs,
    ):  
        if patch_size%2 != 0:
            raise ValueError(f"Patch size should be multiple of 2, not {patch_size}.")
        self.patch_size = patch_size
        
        if isinstance(horizon_lengths, int):
            horizon_lengths = [horizon_lengths]
        self.horizon_lengths = horizon_lengths

        self.wavelet_function = wavelet_function
        self.wavelet_signal_extension_mode = wavelet_signal_extension_mode
        self.wavelet_dwt_level = wavelet_dwt_level
        self.loss_func = loss_func
        self.time_axis_loss_factor = time_axis_loss_factor

        # if these two attribute are not provided in config.json
        # the model will be init as previous pure univariate version
        self.use_group_attn = use_group_attn
        self.use_channel_axis_loss = use_channel_axis_loss

        kwargs.pop('tie_word_embeddings', None)
        kwargs.pop('horizon_lengths', None)
        kwargs.pop("input_size", None)
        super().__init__(
            input_size=patch_size*2,
            tie_word_embeddings=tie_word_embeddings,
            horizon_lengths=horizon_lengths,
            **kwargs,
        )
