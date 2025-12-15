import os
from typing import List
from transformers import PretrainedConfig

# derived from time_moe.models.configuration_time_moe.TimeMoeConfig
class WaveletMoeConfig(PretrainedConfig):
    model_type = "wavelet_moe"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
            self,
            patch_size: int = 8,
            hidden_size: int = 4096,
            intermediate_size: int = 22016,
            horizon_lengths: List[int] = 1,
            num_hidden_layers: int = 32,
            num_attention_heads: int = 32,
            num_key_value_heads: int = None,
            hidden_act: str = "silu",
            num_experts_per_token: int = 2,
            num_experts: int = 1,
            max_position_embeddings: int = 32768,
            initializer_range: float = 0.02,
            rms_norm_eps: float = 1e-6,
            use_dense: bool = False,
            rope_theta: int = 10000,
            attention_dropout: float = 0.0,
            use_load_balance_loss: bool = True,
            load_balance_loss_factor: float = 0.02,
            tie_word_embeddings: bool = False,
            wavelet_function: str = "bior2.2",
            wavelet_signal_extension_mode: str = "periodization",
            wavelet_dwt_level: int = 2,
            loss_func: str = "huber",
            # use_topk_kv: bool = True,
            # topk_kv: int = None,
            **kwargs,
    ):  
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        
        self.num_experts_per_token = num_experts_per_token
        self.num_experts = num_experts
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_dense = use_dense
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout
        self.use_load_balance_loss = use_load_balance_loss
        self.load_balance_loss_factor = load_balance_loss_factor

        assert self.use_dense ^ self.use_load_balance_loss, 'Both use_dense and use_load_balance_loss cannot be set to True or False at the same time.'

        if patch_size%2 != 0:
            raise ValueError(f"Patch size should be multiple of 2, not {patch_size}.")
        self.patch_size = patch_size
        self.token_len = patch_size
        
        if isinstance(horizon_lengths, int):
            horizon_lengths = [horizon_lengths]
        self.horizon_lengths = horizon_lengths

        self.wavelet_function = wavelet_function
        self.wavelet_signal_extension_mode = wavelet_signal_extension_mode
        self.wavelet_dwt_level = wavelet_dwt_level
        self.loss_func = loss_func

        # self.use_topk_kv = use_topk_kv
        # self.topk_kv = topk_kv
        # if use_topk_kv and topk_kv is None:
        #     raise ValueError(f"topk_kv should not be None if filter MHA is used.")

        kwargs.pop('tie_word_embeddings', None)
        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
