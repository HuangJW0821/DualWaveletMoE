import os
import warnings
import math
from typing import Optional, Tuple, List, Union

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from transformers import PreTrainedModel, Cache, DynamicCache, StaticCache
from transformers.activations import ACT2FN
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.utils import logging
from .configuration_wavelet_moe import WaveletMoeConfig
from .wavelet_moe_output import WaveletMoeModelOutputWithPast, WaveletMoeCausalLMOutputWithPast
from transformers.modeling_outputs import MoeModelOutputWithPast, MoeCausalLMOutputWithPast
from .wavelet_generation_mixin import WaveletGenerationMixin
from ..datasets.dwt_tokenizer import DWTTokenizer

logger = logging.get_logger(__name__)

# get info about unpad data
def _get_unpad_data(attention_mask):
    """
    Get info about unpad token (real token), used in class TimeMoeFlashAttention2

    Args:
        attention_mask: [batch_size, seq_len], 1 for real token, 0 for pad

    Returns:
        (indices, cu_seqlens, max_seqlen_in_batch): 
        indices: index of every real token (non-pad token) \n
        cu_seqlens: cumulative sum of seq len, ie offset of every token \n
        max_seqlen_in_batch: max seq len in batch
    """

    # cal real len for every seq in batch, by summing the last dim of attention_mask
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)

    # get index of every real token
    # first flatten attention_mask and get non-zero value's index:
    #   torch.nonzero(.., as_tuple=False) return a 2D tensor, each row is the index of non-zero value
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()

    # get max seq len in batch
    max_seqlen_in_batch = seqlens_in_batch.max().item()

    # cal cumulative sum, get offset of every seq's end,
    # then pad a 0 in front, get sth like [0, len(seq1), len(seq1)+len(seq2), ...]
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))

    # return 
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


# load balancing loss for MoE layers (shared expert are not included)
def load_balancing_loss_func(
        gate_logits: Union[torch.Tensor, Tuple[torch.Tensor], List[torch.Tensor]],
        top_k: int,
        num_experts: int = None,
        attention_mask: Optional[torch.Tensor] = None,
        group_ids: Optional[torch.Tensor] = None
) -> torch.Tensor:
    r"""
    Computes auxiliary load balancing loss as in Switch Transformer - implemented in Pytorch.

    See Switch Transformer (https://arxiv.org/abs/2101.03961) for more details. This function implements the loss
    function presented in equations (4) - (6) of the paper. It aims at penalizing cases where the routing between
    experts is too unbalanced.

    Args:
        gate_logits: (Union[`torch.Tensor`, Tuple[torch.Tensor], List[torch.Tensor])
            Logits from the `gate`, should be a tuple of `model.config.num_hidden_layers` tensors of
            shape `[batch_size X sequence_length, num_experts]`.
        top_k: (`int`)
            Selected Top k over the experts.
        num_experts (`int`, *optional*):
            Number of experts
        attention_mask: (`torch.Tensor`, None)
            The attention_mask used in forward function
            shape [batch_size X sequence_length] if not None.
        group_ids: `(batch_size, )`

    Returns:
        The auxiliary loss.
    """
    if gate_logits is None or not isinstance(gate_logits, (tuple, list)) or gate_logits[0] is None:
        return 0.0

    # choose top-k experts, rated by softmax, get a expert_mask
    compute_device = gate_logits[0].device

    # concatenated_gate_logits: [num_hidden_layers * batch_size * seq_len, num_experts]
    concatenated_gate_logits = torch.cat([layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0)

    # shape [num_hidden_layers * batch_size * seq_len, num_experts]
    routing_weights = torch.nn.functional.softmax(concatenated_gate_logits, dim=-1)

    # select topk & mask onehot
    # shape [num_layers * batch * seq_len, topk, num_experts]
    # expert_mask[i] indicate the experts that token i chosen
    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)

    # situation 1: all token are real
    if attention_mask is None:
        # Compute the percentage of tokens routed to each expert
        tokens_per_expert = torch.mean(expert_mask.float(), dim=0)

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.mean(routing_weights, dim=0)
    
    # situation 2:
    else:
        batch_size, sequence_length = attention_mask.shape

        # concatenated_gate_logits: [num_hidden_layers * batch_size * seq_len, num_experts], dont forget
        num_hidden_layers = concatenated_gate_logits.shape[0] // (batch_size * sequence_length)

        # Compute the mask that masks all padding tokens as 0 with the same shape of expert_mask:
        # 1. increase dim of attn_mask to [1, batch_size, seq_len, 1, 1]
        # 2. expand shape of attn_mask to [num_hidden_layers, batch_size, sequence_length, top_k, num_experts]
        # 3. flatten first 2 dim, reshape to [num_layers * batch_size * seq_len, top_k, num_experts], same shape of expert_mask
        expert_attention_mask = (
            attention_mask[None, :, :, None, None]
            .expand((num_hidden_layers, batch_size, sequence_length, top_k, num_experts))
            .reshape(-1, top_k, num_experts)
            .to(compute_device)
        )

        # Compute the percentage of tokens routed to each experts
        # 1. sum(expert_mask * expert_attn_mask): times an exact expert be selected through out all transformer blocks
        # 2. sum(expert_attn_mask): how many times an expert be selected
        # 3. divide, get percentage (ideal probability)
        # shape [topk, num_experts]
        tokens_per_expert = torch.sum(expert_mask.float() * expert_attention_mask, dim=0) / torch.sum(
            expert_attention_mask, dim=0
        )

        # Compute the mask that masks all padding tokens as 0 with the same shape of tokens_per_expert
        # shape: [num_layers * batch_size * seq_len, num_experts]
        router_per_expert_attention_mask = (
            attention_mask[None, :, :, None]
            .expand((num_hidden_layers, batch_size, sequence_length, num_experts))
            .reshape(-1, num_experts)
            .to(compute_device)
        )

        # Compute the average probability of routing to these experts
        # 1. sum(routing_weight * router_per_..._mask): sum all unpad token's routing weight
        # 2. sum(router_per_..._mask): nums of unpad token
        # 3. divide, get average probality (actual probability)
        # shape [num_experts, ]
        router_prob_per_expert = torch.sum(routing_weights * router_per_expert_attention_mask, dim=0) / torch.sum(
            router_per_expert_attention_mask, dim=0
        )

    # overall_loss = sum(ideal_prob * actual_prob)
    # in idealize situation, every expert get token with same prob, thus ideal_prob * actual_prob = 1/N^2
    # then sum, overall_loss = 1/N, a float number
    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(dim=0))

    # finally, loss==1 in when idealize
    return overall_loss * num_experts


def channel_load_balancing_loss(
    gate_logits: Union[torch.Tensor, Tuple[torch.Tensor], List[torch.Tensor]],
    top_k: int,
    num_experts: int = None,
    attention_mask: Optional[torch.Tensor] = None,
):
    """
    Compute load balanc loss along batch axis.

    Args:
        gate_logits: (Union[`torch.Tensor`, Tuple[torch.Tensor], List[torch.Tensor])
            Logits from the `gate`, should be a tuple of `model.config.num_hidden_layers` tensors of
            shape `[batch_size X sequence_length, num_experts]`.
        top_k: (`int`)
            Selected Top k over the experts.
        num_experts (`int`, *optional*):
            Number of experts
        attention_mask: (`torch.Tensor`, None)
            The attention_mask used in forward function
            shape [batch_size, sequence_length] if not None.
    """

    num_layers = len(gate_logits)
    batch_size, seq_len = attention_mask.shape
    compute_device = gate_logits[0].device

    # concatenated_gate_logits: [num_hidden_layers * batch_size * seq_len, num_experts]
    concatenated_gate_logits = torch.cat([layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0)
    
    # gate routing weights [num_hidden_layers * batch_size * seq_len, num_experts]
    routing_weights = torch.nn.functional.softmax(concatenated_gate_logits, dim=-1)

    # expert_mask shape [num_layers * batch * seq_len, topk, num_experts]
    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts).float()

    # reshape
    routing_weights = routing_weights.view(batch_size, num_layers * seq_len, num_experts)
    expert_mask = expert_mask.view(batch_size, num_layers * seq_len, top_k, num_experts)

    if attention_mask is None:
        # shape [batch_size, num_experts]
        tokens_per_expert = torch.mean(expert_mask.float().sum(dim=2), dim=1) / top_k

        # shape [batch_size, num_experts]
        router_prob_per_batch = torch.mean(routing_weights, dim=1)
    
    else:

        # extend to [batch_size, num_layers, seq_len]
        expert_attention_mask = attention_mask.unsqueeze(1).repeat(1, num_layers, 1)

        # [batch_size, num_layers * seq_len]
        expert_attention_mask = expert_attention_mask.view(batch_size, -1).float()

        # [batch_size, 1]
        num_valid_tokens_per_batch = torch.sum(expert_attention_mask, dim=1, keepdim=True)

        # [batch_size, num_experts]
        tokens_per_expert = torch.sum(
            expert_mask.float().sum(dim=2)  * expert_attention_mask.unsqueeze(-1), 
            dim=1
        ) / (num_valid_tokens_per_batch * top_k)


        # [batch_size, num_experts]
        router_prob_per_batch = torch.sum(
            routing_weights * expert_attention_mask.unsqueeze(-1), 
            dim=1
        ) / (num_valid_tokens_per_batch)

    overall_loss = torch.sum(torch.mean(tokens_per_expert, dim=0) * torch.mean(router_prob_per_batch, dim=0))

    return overall_loss * num_experts


# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)

    Repeat key value, use when num_key_value_head < num_attention_head situation like Grouped Query Attn or Multi-Query Attn,
    so that we can storage lesser key_value_head (reduce KV matrix's needed params)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    # (x, y) → (-y, x), ie rotate in imagine number's space
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.mistral.modeling_mistral.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors,
      therefore encode relative position relations between tokens.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)    # choose which index to rotate & expand into new shape
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

# derived from time_moe.models.modeling_time_moe.TimeMoeInputEmbedding
class WaveletMoeInputEmbedding(nn.Module):
    """
    Use a mlp layer to embedding the time-series.
    Embed [batch_sz, token_num, token_len] to [batch_sz, token_num, hidden_sz], 
    in which token_num == seq_len/(patch_sz)*2, token_len == patch_sz*2 == input_sz
    """

    def __init__(self, config: WaveletMoeConfig):
        super().__init__()
        self.config = config
        self.input_size = config.input_size  # default patch_sz*2
        self.hidden_size = config.hidden_size
        self.emb_layer = nn.Linear(self.input_size, self.hidden_size, bias=False)
        self.gate_layer = nn.Linear(self.input_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        """[batch_sz, token_num, token_len] -> [batch_sz, token_num, hidden_sz]"""
        emb = self.act_fn(self.gate_layer(x)) * self.emb_layer(x)
        return emb


# Copied from transformers.models.mistral.modeling_mistral.MistralRotaryEmbedding with Mistral->WaveletMoe
class WaveletMoeRotaryEmbedding(torch.nn.Module):
    """
    Apply RoPE to input
    """
    def __init__(self, dim, max_position_embeddings=512, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


# Copied from transformers.models.llama.modeling_llama.LlamaRMSNorm with Llama->WaveletMoe
class WaveletMoeRMSNorm(torch.nn.Module):
    """
    RMSNorm for Root Mean Square: x = x / sqrt(mean(x^2) + eps)
    """
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


# Copied from from time_moe.models.modeling_time_moe.TimeMoeTemporalBlock with TimeMoe -> WaveletMoe
class WaveletMoeTemporalBlock(nn.Module):
    """
    Tempolate for FFN, used in MoE layer and dense model
    """
    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, hidden_state):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))


# Copied from from time_moe.models.modeling_time_moe.WaveletMoeMLP with TimeMoe -> WaveletMoe
class WaveletMoeMLP(WaveletMoeTemporalBlock):
    """
    Wrapper of class WaveletMoeTemporalBlock
    """
    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str):
        super().__init__(hidden_size, intermediate_size, hidden_act)

    def forward(self, hidden_state):
        return super().forward(hidden_state), None


# Copied from from time_moe.models.modeling_time_moe.TimeMoeSparseExpertsLayer with TimeMoe -> WaveletMoe
class WaveletMoeSparseExpertsLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok     # top-k per token
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts
        self.norm_topk_prob = False

        # scaling expert's intermediate layer's size, s.t. keep compute cost
        moe_intermediate_size = self.config.intermediate_size // self.top_k

        # gating
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [WaveletMoeTemporalBlock(
                hidden_size=self.config.hidden_size,
                intermediate_size=moe_intermediate_size,
                hidden_act=self.config.hidden_act,
            ) for _ in range(self.num_experts)]
        )

        # shared expert with individual gate
        self.shared_expert = WaveletMoeTemporalBlock(
            hidden_size=self.config.hidden_size,
            intermediate_size=self.config.intermediate_size,
            hidden_act=self.config.hidden_act,
        )
        self.shared_expert_gate = torch.nn.Linear(config.hidden_size, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits -> (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)    # gating

        # calculate routing weights: softmax & top-k
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)
        
        # init output shape
        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        # expert_mask: [num_experts, top_k, batch*seq_len]
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        # ie forward thorugh selected experts
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

        # forward through shared expert
        shared_expert_output = self.shared_expert(hidden_states)
        shared_expert_output = F.sigmoid(self.shared_expert_gate(hidden_states)) * shared_expert_output

        # fuse selected expert's outputs & shared expert's output
        final_hidden_states = final_hidden_states + shared_expert_output

        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits


# Copied from transformers.models.qwen2.modeling_qwen2.Qwen2Attention with Qwen2->Wavelet
class WaveletMoeTimeAttention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper with kv cache.
    """

    def __init__(self, config: WaveletMoeConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        # attention head config, num_query_head (num_heads) >= num_kv_head in GQA
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        # RoPE config
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta

        self.is_causal = True
        self.attention_dropout = config.attention_dropout

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        
        # projection layers
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.rotary_emb = WaveletMoeRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: torch.Tensor,
            position_ids: Optional[torch.LongTensor] = None,
            output_attentions: bool = False,
            **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        Classic MHA.

        Args:
         hidden_states: `[batch_sz, q_len, hidden_sz]`
         attention_mask: `[batch_sz, 1, q_len, kv_len]`
         position_ids: `[batch_sz, q_len]`
         output_attentions: `bool`
        
        Returns:
         (attn_output, attn_weights):
         - attn_output: `[batch_sz, q_len, hidden_sz]`
         - attn_weifhts: `None` or `[batch_sz, num_heads, q_len, kv_len]`
        """
        
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        
        # forward thourgh projection layers
        bsz, q_len, _ = hidden_states.size()

        # q,k,v states, shape [batch_sz, q_len, hidden_sz]
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # q_states  shape -> [batch_sz, num_heads, q_len, head_dim], where hidden_states = num_heads * head_dim
        # kv_states shape -> [batch_sz, num_k_v_heads, q_len, head_dim]
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # restore kv_len
        kv_seq_len = key_states.shape[-2]
        
        # forward through RoPE layer (only QK states)
        # q,k_states shape unchange
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        # if n_kv_heads != n_heads (when GQA or MQA), reshape k,v_sates  -> [batch_sz, num_heads, kv_len, head_dim]
        # otherwise, keep shape unchange
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # cal attention weights
        # shape [batch_sz, num_heads, q_len, kv_len]
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Time attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        # masking attention weights with attn_mask
        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Time Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            
            # apply mask
            attn_weights = attn_weights + attention_mask

        # forward through softmax & dropout
        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        
        # mul with v_states, shape [batch_sz, num_heads, q_len, head_dim]
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        # collect heads outputs
        # shape: [bsz, num_heads, q_len, head_dim] -> [bsz, q_len, hidden_size]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        # forward thorugh output_proj
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights


class WaveletMoeGroupAttention(nn.Module):
    """
    Multi-headed attention within groups to capture cross-channel information.
    No REPo since there's no position information between groups.
    """

    def __init__(self, config: WaveletMoeConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        # attention head config, num_query_head (num_heads) >= num_kv_head in GQA
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self.is_causal = True
        self.attention_dropout = config.attention_dropout

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        
        # projection layers
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)


    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: torch.Tensor,
            output_attentions: bool = False,
            **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        Group attention, MHA within groups by transpose hidden_states & sepecif attention mask.

        Args:
         hidden_states: `[batch_sz, q_len, hidden_sz]`
         attention_mask: `[kv_len, 1, batch_sz, batch_sz]`
         output_attentions: `bool`
        
        Returns:
         (attn_output, attn_weights):
         - attn_output: `[batch_sz, q_len, hidden_sz]`
         - attn_weights: `None` or `[q_len, num_heads, batch_sz, batch_sz]`
        """
        
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        
        # forward thourgh projection layers
        bsz, q_len, _ = hidden_states.size()

        # transpose to [q_len, batch_sz, hidden_sz] to apply group attention
        hidden_states = hidden_states.transpose(0, 1)

        # q states, shape [q_len, batch_sz, hidden_sz]
        # kv states, shape [kv_len, batch_sz, hidden_sz]
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # q_states  shape -> [q_len, num_heads, batch_sz, head_dim], where hidden_states = num_heads * head_dim
        # kv_states shape -> [kv_len, num_k_v_heads, batch_sz, head_dim]
        query_states = query_states.view(q_len, bsz, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(q_len, bsz, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(q_len, bsz, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_len = key_states.shape[0]
        if q_len != kv_len:
            raise ValueError(f"q_len [{q_len}] should equal to kv_len [{kv_len}]!")

        # if n_kv_heads != n_heads (when GQA or MQA), reshape k,v_sates  -> [kv_len, num_heads, batch_sz, head_dim]
        # otherwise, keep shape unchange
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # cal attention weights
        # [q_len, num_heads, batch_sz, head_dim] * [kv_len, num_heads, head_dim, batch_sz]
        # = [q_len, num_heads, batch_sz, batch_sz], since no q_len==kv_len (no kv cache)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (q_len, self.num_heads, bsz, bsz):
            raise ValueError(
                f"Group attention weights should be of size {(q_len, self.num_heads, bsz, bsz)}, but is"
                f" {attn_weights.size()}"
            )

        # masking attention weights with attn_mask
        if attention_mask is not None:
            if attention_mask.size() != (kv_len, 1, bsz, bsz):
                raise ValueError(
                    f"Group ttention mask should be of size {(kv_len, 1, bsz, bsz)}, but is {attention_mask.size()}"
                )
            
            # apply mask
            attn_weights = attn_weights + attention_mask

        # forward through softmax & dropout
        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        
        # mul with v_states
        # [q_len, num_heads, batch_sz, batch_sz] * [kv_len, num_heads, batch_sz, head_dim]
        # == shape [q_len, num_heads, batch_sz, head_dim] since q_len==kv_len
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (q_len, self.num_heads, bsz, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(q_len, self.num_heads, bsz, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        # collect heads outputs
        # shape: [q_len, num_heads, bsz, head_dim] -> [q_len, bsz, hidden_size]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(q_len, bsz, self.hidden_size)

        # forward thorugh output_proj
        attn_output = self.o_proj(attn_output)

        # shape: [bsz, seq_len, hidden_size], consist with hidden shape in model
        attn_output = attn_output.transpose(0, 1)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights


# Copied from from time_moe.models.modeling_time_moe.TimeMoeDecoderLayer with TimeMoe -> WaveletMoe
class WaveletMoeDecoderLayer(nn.Module):
    """
    Implement of WaveletMoE transformer block
    """
    def __init__(self, config: WaveletMoeConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size

        if self.config.use_group_attn:
            # Time Attention
            self.time_attn = WaveletMoeTimeAttention(config, layer_idx)

            # Group Attention
            self.group_attn = WaveletMoeGroupAttention(config, layer_idx)
        
        
        else:
            if self.config.use_channel_axis_loss:
                self.time_attn = WaveletMoeTimeAttention(config, layer_idx)
            else:
                # adaption for previous univariate version chekcpoint, should delete when release
                self.self_attn = WaveletMoeTimeAttention(config, layer_idx)

        # MoE or dense FFN
        if self.config.use_dense:
            self.ffn_layer = WaveletMoeMLP(
                hidden_size=self.config.hidden_size,
                intermediate_size=self.config.intermediate_size,
                hidden_act=self.config.hidden_act,
            )
        else:
            self.ffn_layer = WaveletMoeSparseExpertsLayer(config)
        
        # Before- , Between- & After-Attention Norm layer
        if self.config.use_group_attn:
            self.before_attention_layernorm = WaveletMoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.inter_attention_layernorm = WaveletMoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            if self.config.use_channel_axis_loss:
                self.before_attention_layernorm = WaveletMoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            else:
                # no inter attention norm layer for previous univariate version chekcpoint
                self.input_layernorm = WaveletMoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        self.post_attention_layernorm = WaveletMoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
            self,
            hidden_states: torch.Tensor,
            time_attn_mask: Optional[torch.Tensor] = None,
            group_attn_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            output_attentions: Optional[bool] = False,
            **kwargs,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, Optional[torch.FloatTensor], Optional[torch.FloatTensor]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, q_len, kv_len)` where padding elements are indicated by 0.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail. 
        """
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. "
                "Please make sure use `attention_mask` instead.`"
            )

        if self.config.use_group_attn:
            # Time Attention Layer
            residual = hidden_states
            hidden_states = self.before_attention_layernorm(hidden_states)
            hidden_states, time_attn_weights = self.time_attn(
                hidden_states=hidden_states,
                attention_mask=time_attn_mask,
                position_ids=position_ids,
                output_attentions=output_attentions
            )
            hidden_states = residual + hidden_states

            # Group Attention Layer
            residual = hidden_states
            hidden_states = self.inter_attention_layernorm(hidden_states)
            hidden_states, group_attn_weights = self.group_attn(
                hidden_states=hidden_states,
                attention_mask=group_attn_mask,
                position_ids=position_ids,
                output_attentions=output_attentions
            )
            hidden_states = residual + hidden_states
        
        else:
            if self.config.use_channel_axis_loss:
                residual = hidden_states
                hidden_states = self.before_attention_layernorm(hidden_states)
                hidden_states, time_attn_weights = self.time_attn(
                    hidden_states=hidden_states,
                    attention_mask=time_attn_mask,
                    position_ids=position_ids,
                    output_attentions=output_attentions
                )
                hidden_states = residual + hidden_states
            else:
                residual = hidden_states
                hidden_states = self.input_layernorm(hidden_states)
                hidden_states, time_attn_weights = self.self_attn(
                    hidden_states=hidden_states,
                    attention_mask=time_attn_mask,
                    position_ids=position_ids,
                    output_attentions=output_attentions
                )
                hidden_states = residual + hidden_states

            group_attn_weights = None

        # Feed Forward Layer
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, router_logits = self.ffn_layer(hidden_states)
        hidden_states = residual + hidden_states

        # output settings
        if not output_attentions:
            time_attn_weights = None
            group_attn_weights = None
        
        return {
            "hidden_states": hidden_states,
            "time_attn_weights": time_attn_weights,
            "group_attn_weights": group_attn_weights,
            "router_logits": router_logits
        }


# Copied from from time_moe.models.modeling_time_moe.WaveletMoePreTrainedModel with TimeMoe -> WaveletMoe
class WaveletMoePreTrainedModel(PreTrainedModel):
    """
    Base class of WaveletMoE
    """
    config_class = WaveletMoeConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["WaveletMoeDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = False
    _supports_sdpa = False
    _supports_cache_class = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, torch.nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, torch.nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


# Derived from from time_moe.models.modeling_time_moe.TimeMoeModel with TimeMoe -> WaveletMoe
# Edit the implementation of causal attention mask to adpat patch-wise tokenization.
class WaveletMoeModel(WaveletMoePreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`TimeMoeDecoderLayer`]
    Embedding layer + WaveletMoE backbone: WaveletMoeDecoderLayer * N
    
    Args:
        config: WaveletMoeConfig
    """

    def __init__(self, config: WaveletMoeConfig):
        super().__init__(config)
        self.embed_layer = WaveletMoeInputEmbedding(config)
        self.layers = nn.ModuleList(
            [WaveletMoeDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = WaveletMoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def _prepare_time_attn_mask(
        self,
        attention_mask: torch.Tensor,
        input_shape: Union[torch.Size, Tuple, List],
        inputs_embeds: torch.Tensor
    ):
        """
        Generate 4d causal attention mask for time attention, mask along time axis. \n

        Args:
         attention_mask: `[batch_sz, kv_len]`
         input_shape: `(batch_sz, q_len)`
         inputs_embeds: `torch.Tensor`, use it to get device and dtype info

        Returns:
         time_attn_mask: `[batch_sz, 1, q_len, kv_len]`
        """

        time_attn_mask = _prepare_4d_causal_attention_mask(
            attention_mask,
            input_shape,
            inputs_embeds,
            0,
        )
    
        return time_attn_mask
    
    def _prepare_group_attn_mask(
        self,
        attention_mask: torch.Tensor,
        group_ids: torch.Tensor,
        inputs_embeds: torch.Tensor
    ):
        """
        Generate 4d attention mask for group attention, mask within groups. \n
        `group_attn_mask` shapes `[kv_len, 1, batch_sz, batch_sz]`; 
        `group_attn_mask[i][j][k]` means that in time step `i`, `batch[i]` should apply attention on `batch[j]`. \n

        Args:
         attention_mask: `[batch_sz, kv_len]`
         group_ids: `[batch_sz,]`
         input_shape: `(batch_sz, q_len)`
         inputs_embeds: `torch.Tensor`, use it to get device and dtype info.
        
        Returns:
         group_attn_mask: `[kv_len, 1, batch_sz, batch_sz]`
        """

        # shape [batch_sz, batch_sz]
        # group_mask[i][j]==True when batch[i] & batch[j] are in same group
        group_mask = group_ids[:, None] == group_ids[None, :]

        # outer product of group_mask & attention_mask
        # thus combines group & time masks to ensure attention only uses
        # tokens from same group which are also not mask in time
        # shape: [batch_sz, batch_sz, kv_len]
        group_attn_mask = torch.einsum("qb, bt -> qbt", group_mask, attention_mask)

        # shape -> [kv_len, 1, batch_sz, batch_sz]
        group_attn_mask = group_attn_mask.permute(2, 0, 1).unsqueeze(1)

        # invert mask
        group_attn_mask = (1.0 - group_attn_mask) * torch.finfo(inputs_embeds.dtype).min

        return group_attn_mask

    def forward(
            self,
            input_ids: torch.FloatTensor = None,
            group_ids: torch.Tensor = None,
            loss_masks: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,           
            inputs_embeds: Optional[torch.FloatTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, WaveletMoeModelOutputWithPast]:
        """
        **Notation**
        - `q_len` equals to token nums to process in this round, \
            (ie `horizon_len` of prediction head chosen in previous forward or input tokens nums in first forward).
        - `kv_len` equals to tokens nums that have already processed, including input prompt & generated tokens.
        - `token_num == q_len`

        Args:
         input_ids: `[batch_sz, token_num, token_len]`
         group_ids: `[batch_sz, ]`
         loss_masks: `[batch_sz, token_num]`

         attention_mask: 
          - when training: first `None`; then generate in forward with shape `[batch_sz, 1, token_num, token_num]`.
          - when inference: shape `[batch_sz, kv_len]` -> `[batch_sz, 1, q_len, kv_len]`
         
         position_ids: Position ids for RoPE, generate in first forward & update during forward
          - when first generate: shape ``[1, token_num]`
          - when provided: update shape to `[batch_size, token_num]`.
         input_embeds: shape `[batch_sz, token_num, hidden_sz]`, generate during forward.
        
         output_attentions: output setting, load from self.config.
         ouput_hidden_states: output setting, load from self.config.
         return_dict: output setting, load from self.config.
        """

        # set default param
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        # valid input ts be [batch_size, seq_len, input_size]
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            # for univariate input, convert to 3D vector to unify process method
            if len(input_ids.shape) == 2:   
                input_ids.unsqueeze_(dim=-1)
            batch_size, token_num, _ = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, token_num, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")


        # generate position_ids for RoPE
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(0, token_num, dtype=torch.long, device=device)
            position_ids = position_ids.view(-1, token_num)
        else:
            position_ids = position_ids.view(-1, token_num).long()

        # start forwarding from here
        # forward through embe_layer if input arent embedding
        # shape: [batch_sz, token_num, token_len] -> [batch_sz, token_num, hidden_sz]
        if inputs_embeds is None:
            inputs_embeds = self.embed_layer(input_ids)

        # generate time_attn_mask & group_attn_mask using loss_mask (training) or attention_mask (inferencing)
        if attention_mask is None and loss_masks is None:
            raise ValueError("attention_mask and loss_mask shouldn't be None at same time!")

        # use loss_masks when training
        if loss_masks is not None:
            attention_mask = loss_masks

        time_attn_mask = self._prepare_time_attn_mask(
            attention_mask,
            (batch_size, token_num),
            inputs_embeds
        )

        group_attn_mask = self._prepare_group_attn_mask(
            attention_mask,
            group_ids,
            inputs_embeds
        )

        hidden_states = inputs_embeds

        # decoder layers
        # init output as config setting
        all_hidden_states = () if output_hidden_states else None
        all_time_attns = () if output_attentions else None
        all_group_attns = () if output_attentions else None
        all_router_logits = ()

        # forwarding through backbone layers
        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # forward setting: load gradient checkpoint or normal forwarding
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    time_attn_mask,
                    group_attn_mask,
                    position_ids,
                    output_attentions
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    time_attn_mask=time_attn_mask,
                    group_attn_mask=group_attn_mask,
                    position_ids=position_ids,
                    output_attentions=output_attentions
                )

            # hidden_states from layer output
            hidden_states = layer_outputs["hidden_states"]

            # collect MoE router logits
            all_router_logits += (layer_outputs["router_logits"],)

            if output_attentions:
                all_time_attns += (layer_outputs["time_attn_weights"],)
                all_group_attns += (layer_outputs["group_attn_weights"],)

        # normalize last layer's hidden_states
        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # return
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, all_hidden_states, all_time_attns, all_group_attns, all_router_logits]
                if v is not None
            )
        return WaveletMoeModelOutputWithPast(
            last_hidden_state=hidden_states,    
            hidden_states=all_hidden_states,
            time_attentions=all_time_attns,
            gruop_attentions=all_group_attns,
            router_logits=all_router_logits
        )


class WaveletMoeOutputLayer(nn.Module):
    """
    Prediction head: a MLP from [batch_size, seq_len, hidden_size] → [batch_size, token_num, horizon_length, token_len], 
    mind that seq_len==token_num, input_size==token_len
    """
    def __init__(self, hidden_size: int, horizon_length: int, input_size: int = 1):
        super().__init__()
        self.horizon_length = horizon_length
        self.input_size = input_size
        self.out_layer = nn.Linear(
            hidden_size,
            input_size * horizon_length,    # multi-step prediction
            bias=False,
        )

    def forward(self, x):
        """

        Args:
            x (torch.FloatTensor): with shape [batch_size, token_num, hidden_size]

        Returns:
    `       torch.FloatTensor: final prediction with shape [batch_size, token_num, horizon_length, input_size]
        """

        batch_size, token_num, _ = x.shape

        # [batch_size, token_num, hidden_size] → [batch_size, token_num, token_len * horizon_length]
        out = self.out_layer(x)

        # shape -> [batch_size, token_num, horizon_length, token_len]
        out = out.view(batch_size, token_num, self.horizon_length, self.input_size)
        return out


# Derived from from time_moe.models.modeling_time_moe.TimeMoeForPrediction with TimeMoe -> WaveletMoe
# Edit the loss function to adpat patch-wise tokenization.
class WaveletMoeForPrediction(WaveletMoePreTrainedModel, WaveletGenerationMixin):

    def __init__(self, config: WaveletMoeConfig):
        super().__init__(config)
        self.config = config
        self.apply_aux_loss = config.apply_aux_loss
        self.num_experts_per_tok = config.num_experts_per_tok
        self.router_aux_loss_factor = config.router_aux_loss_factor

        # self.model = input_layer + transformer_block * N
        self.model = WaveletMoeModel(config)

        # output layer
        lm_head_list = []
        self.horizon_length_map = {}    # Mind this: horizon_length := token_len * predict_token_num == (patch_sz*2)*predict_token_num
        for i, horizon_length in enumerate(config.horizon_lengths):
            lm_head_list.append(
                WaveletMoeOutputLayer(
                    hidden_size=self.config.hidden_size,
                    input_size=self.config.input_size,
                    horizon_length=horizon_length,
                )
            )
            self.horizon_length_map[horizon_length] = i
        self.lm_heads = nn.ModuleList(lm_head_list)

        # select loss function
        self.loss_function_name = config.loss_func
        if self.loss_function_name == "huber":
            self.loss_function = torch.nn.HuberLoss(reduction='none', delta=2.0)
        elif self.loss_function_name == "mse":
            self.loss_function = torch.nn.MSELoss(reduction='none')
        else:
            raise ValueError(f"Unsupport loss function: {self.loss_function_name}")

        # add dwt tokenizer to calculate loss
        self.dwt_tokenizer = DWTTokenizer(
                                 wavelet=self.config.wavelet_function, 
                                 mode=self.config.wavelet_signal_extension_mode,
                                 level=self.config.wavelet_dwt_level,
                                 patch_size=self.config.patch_size,
                                )

        # Initialize weights and apply final processing
        self.post_init()

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
            self,
            input_ids: torch.FloatTensor = None,
            group_ids: torch.Tensor = None,
            loss_masks: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            max_horizon_length: Optional[int] = None,
            target_idx: Optional[Tuple[int]] = None
    ) -> Union[Tuple, WaveletMoeCausalLMOutputWithPast]:
        """
        Args:
         input_ids:
          `[batch_sz, token_num, token_len]`
          - When training: `token_num` is the token nums of sample.
          - When inferencing: `token_num` equals to token nums of prompt in first round of forward, \
          after that `token_num` equals to token nums generated in last round of forward (`horizon_len` chosen in last forward).
         group_ids: `[batch_sz, ]`
         loss_masks: `[batch_sz, token_num]`
         labels: `[batch_sz, token_num, token_len]`, use in training.
         attention_mask: `None` or `[batch_sz, token_num]`
         position_ids: `None` or `[batch_sz, token_num]`
         inputs_embeds: `None` or `[batch_sz, token_num, hidden_size]`
        """

        # output setting
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids = input_ids,
            group_ids = group_ids,
            loss_masks = loss_masks,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,

            # output setting
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        predictions = None

        loss = None
        aux_loss = None

        # when training
        if labels is not None:  
            # AutoRegressive loss
            ar_loss = 0.0
            dwt_loss = 0.0

            for lm_head, horizon_length in zip(self.lm_heads, self.config.horizon_lengths):
                one_predictions = lm_head(hidden_states)

                # Mind this: 1. should modify loss func; 2. whats loss_mask
                one_ar_loss = self.calc_ar_loss(one_predictions, labels, loss_masks, horizon_length)
                ar_loss += one_ar_loss

                one_dwt_loss = self.calc_dwt_loss(one_predictions)
                dwt_loss += one_dwt_loss
                
                if predictions is None:
                    predictions = one_predictions

            loss = (ar_loss + dwt_loss) / len(self.config.horizon_lengths)

            if self.apply_aux_loss:
                # len(router_logits)==num_layers, router_logits.shape==(batch_sz * token_num, expert_num)
                router_logits = outputs.router_logits if return_dict else outputs[-1]

                temporal_aux_loss = self.calc_load_balancing_loss(
                    router_logits,
                    group_ids = group_ids,
                    top_k=self.num_experts_per_tok,
                    num_experts=self.config.num_experts,
                    attention_mask=loss_masks,
                    time_axis_loss_factor=self.config.time_axis_loss_factor
                )

                loss += self.router_aux_loss_factor * temporal_aux_loss.to(loss.device)
        
        # when inferencing
        else:   
            # scheduling for multi-resolution forecasting, ie Algo 1 in paper
            # schedule the shortest horizon
            if max_horizon_length is None:  
                horizon_length = self.config.horizon_lengths[0]
                max_horizon_length = horizon_length
            
            # greedly schedule the longest horizon
            else:
                horizon_length = self.config.horizon_lengths[0]
                for h in self.config.horizon_lengths[1:]:
                    if h > max_horizon_length:
                        break
                    else:
                        horizon_length = h
            lm_head = self.lm_heads[self.horizon_length_map[horizon_length]]
            predictions = lm_head(hidden_states)

            if horizon_length > max_horizon_length:
                predictions = predictions[:, :, :max_horizon_length, :]

        if not return_dict:
            output = (predictions,) + outputs[1:]
            return (loss, aux_loss) + output if loss is not None else output

        return WaveletMoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=predictions,
            hidden_states=outputs.hidden_states,
            time_attentions=outputs.time_attentions,
            gruop_attentions=outputs.gruop_attentions,
        )


    def calc_ar_loss(self, predictions, labels, loss_masks, horizon_len):
        """
        Args:
         predictions: [B, token_num, horizon_len, token_len]
         labels:      [B, token_num, token_len]
         loss_masks:  [B, token_num]
         horizon_len: int
         loss_func:   function(input, target)
        """
        
        if predictions.shape[2] != horizon_len:
                raise ValueError(f"BUG FOUND: predictions.shape[2] == {predictions.shape[2]} should be the same with horizon_length [{horizon_len}].")

        # reshape labels & loss_masks to [B, token_num, horizon_len, token_len], same with predictions
        # to ensure token-wise loss calculation
        batch_size, token_num, horizon_len, token_len = predictions.shape

        # pad horizon_len-1 zero in dim of token_num, make sure windowing wont out of idx.
        # shape: [B, token_num, token_len] -> [B, token_num + horizon_len -1, token_len]
        padded_labels = F.pad(labels, (0, 0, 0, horizon_len - 1), mode='constant', value=0)  

        # collect labels for every tokens & every horizon step
        # shape -> [B, token_num, horizon_len, token_len]
        new_labels = padded_labels.unfold(dimension=1, size=horizon_len, step=1).transpose(2,3)  

        # expand loss mask
        loss_masks_padded = F.pad(loss_masks, (0, horizon_len - 1))  # [B, token_num + horizon_len - 1]
        new_loss_masks = loss_masks_padded.unfold(dimension=1, size=horizon_len, step=1)  # [B, token_num, horizon_len]
        new_loss_masks = new_loss_masks.unsqueeze(-1).expand(-1, -1, -1, token_len)  # [B, token_num, horizon_len, token_len]

        # apply loss masks
        masked_predictions = predictions * new_loss_masks
        masked_labels = new_labels * new_loss_masks

        # Calculate loss with mask
        losses = self.loss_function(masked_predictions, masked_labels)

        if loss_masks is not None:
            losses = losses * new_loss_masks
            loss = losses.sum() / new_loss_masks.sum()
        else:
            loss = torch.mean(losses)

        return loss


    def calc_dwt_loss(self, predictions):
        """
        Calculate the loss between prediction seq & dwt coeffs.
        """

        batch_size, token_num, horizon_len, token_len = predictions.shape

        # flatten [B, token_num, horizon_len, token_len] -> [B*token_num, horizon_len*token_len]
        # s.t. every row is a tokenized predict time series
        flattened_preds = predictions.reshape(batch_size * token_num, horizon_len * token_len)
            
        pred_seqs, pred_coeffs = self.dwt_tokenizer.patch_wise_detokenize(flattened_preds)
            
        rec_seqs = self.dwt_tokenizer.waverec(pred_coeffs)

        if isinstance(pred_seqs, np.ndarray):
            pred_seqs = torch.from_numpy(pred_seqs).to(dtype=predictions.dtype, device=predictions.device)
        if isinstance(rec_seqs, np.ndarray):
            rec_seqs = torch.from_numpy(rec_seqs).to(dtype=predictions.dtype, device=predictions.device)

        losses = self.loss_function(pred_seqs, rec_seqs)

        loss = torch.mean(losses)

        return loss


    def cal_time_axis_load_balancing_loss(
            self,
            expert_mask: torch.Tensor,
            routing_weights: torch.Tensor,
            attention_mask: torch.Tensor,
            top_k: int,
            num_layers: int,
            batch_size: int,
            seq_len: int,
            num_experts: int,
            device: torch.device
    ) -> torch.Tensor:
        """
        Computes load balancing loss along time axis.

        Args:
        expert_mask: (`torch.Tensor`) shape `[num_layers * batch * seq_len, topk, num_experts]`
        routing_weights: (`torch.Tensor`) shape `[num_layers * batch_size * seq_len, num_experts]`
        attention_mask: (`torch.Tensor`) shape `[batch_size, seq_length]`
        top_k: `int`
        num_layers: `int`
        batch_size: `int`
        seq_len: `int`
        num_experts: `int`
        device: `torch.device`
        
        Returns:
        time_axis_load_balacing_loss: `torch.Tensor`
        """

        # Compute the mask that masks all padding tokens as 0 with the same shape of expert_mask:
        # 1. increase dim of attn_mask to [1, batch_size, seq_len, 1, 1]
        # 2. expand shape of attn_mask to [num_hidden_layers, batch_size, sequence_length, top_k, num_experts]
        # 3. flatten first 2 dim, reshape to [num_layers * batch_size * seq_len, top_k, num_experts], same shape of expert_mask
        expert_attention_mask = (
            attention_mask[None, :, :, None, None]
            .expand((num_layers, batch_size, seq_len, top_k, num_experts))
            .reshape(-1, top_k, num_experts)
            .to(device)
        )

        # Compute the percentage of tokens routed to each experts
        # 1. sum(expert_mask * expert_attn_mask): times an exact expert be selected through out all transformer blocks
        # 2. sum(expert_attn_mask): how many times an expert be selected
        # 3. divide, get percentage (ideal probability)
        # shape [topk, num_experts]
        tokens_per_expert = torch.sum(expert_mask.float() * expert_attention_mask, dim=0) / torch.sum(expert_attention_mask, dim=0)

        # Compute the mask that masks all padding tokens as 0 with the same shape of tokens_per_expert
        # shape: [num_layers * batch_size * seq_len, num_experts]
        router_per_expert_attention_mask = (
            attention_mask[None, :, :, None]
            .expand((num_layers, batch_size, seq_len, num_experts))
            .reshape(-1, num_experts)
            .to(device)
        )

        # Compute the average probability of routing to these experts
        # 1. sum(routing_weight * router_per_..._mask): sum all unpad token's routing weight
        # 2. sum(router_per_..._mask): nums of unpad token
        # 3. divide, get average probality (actual probability)
        # shape [num_experts, ]
        router_prob_per_expert = torch.sum(routing_weights * router_per_expert_attention_mask, dim=0) / torch.sum(router_per_expert_attention_mask, dim=0)

        # overall_loss = sum(ideal_prob * actual_prob), ideally 1/N
        overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(dim=0))

        # finally, loss==1 in when idealize
        return overall_loss * num_experts


    def cal_channel_axis_load_balancing_loss(
            self,
            expert_mask: torch.Tensor,
            routing_weights: torch.Tensor,
            attention_mask: torch.Tensor,
            group_ids: torch.Tensor,
            top_k: int,
            num_layers: int,
            batch_size: int,
            seq_len: int,
            num_experts: int,
            device: torch.device
    ) -> torch.Tensor:
        """
        Computes load balancing loss along channel axis.

        Args:
        expert_mask: (`torch.Tensor`) shape `[num_layers * batch * seq_len, topk, num_experts]`
        routing_weights: (`torch.Tensor`) shape `[num_layers * batch_size * seq_len, num_experts]`
        attention_mask: (`torch.Tensor`) shape `[batch_size, seq_length]`
        group_ids: (`torch.Tensor`)
        top_k: `int`
        num_layers: `int`
        batch_size: `int`
        seq_len: `int`
        num_experts: `int`
        device: `torch.device`
        
        Returns:
        channel_axis_load_balacing_loss: `torch.Tensor`
        """

        # reshape
        routing_weights = routing_weights.view(batch_size, num_layers * seq_len, num_experts)
        expert_mask = expert_mask.view(batch_size, num_layers * seq_len, top_k, num_experts)

        # extend to [batch_size, num_layers, seq_len]
        expert_attention_mask = attention_mask.unsqueeze(1).repeat(1, num_layers, 1)

        # [batch_size, num_layers * seq_len]
        expert_attention_mask = expert_attention_mask.view(batch_size, -1).float().to(device)

        # [batch_size, 1]
        num_valid_tokens_per_batch = torch.sum(expert_attention_mask, dim=1, keepdim=True)

        # [batch_size, num_experts]
        tokens_per_expert = torch.sum(
            expert_mask.float().sum(dim=2)  * expert_attention_mask.unsqueeze(-1), 
            dim=1
        ) / (num_valid_tokens_per_batch * top_k)

        # [batch_size, num_experts]
        router_prob_per_batch = torch.sum(
            routing_weights * expert_attention_mask.unsqueeze(-1), 
            dim=1
        ) / (num_valid_tokens_per_batch)

        overall_loss = torch.sum(torch.mean(tokens_per_expert, dim=0) * torch.mean(router_prob_per_batch, dim=0))

        return overall_loss * num_experts


    def calc_load_balancing_loss(
            self,
            gate_logits: Union[torch.Tensor, Tuple[torch.Tensor], List[torch.Tensor]],
            group_ids: torch.Tensor,
            top_k: int,
            num_experts: int,
            attention_mask: torch.Tensor,
            time_axis_loss_factor: float = 0.5
    ) -> torch.Tensor:
        """
        Computes auxiliary load balancing loss within time & channel axis

        Args:
            gate_logits: (`Tuple[torch.Tensor]`)
                Logits from the `gate`, should be a tuple of `model.config.num_hidden_layers` tensors of
                shape `[batch_size * seq_length, num_experts]`.
            top_k: (`int`)
                Selected Top k over the experts.
            num_experts (`int`):
                Number of experts
            attention_mask: (`torch.Tensor`)
                The attention_mask used in forward function
                shape [batch_size, seq_length]
            time_axis_loss_factor: (`float`)
                Weight of time axis load balancing loss.

        Returns:
            loss:
                Auxiliary load balancing loss. \n
                `loss = time_axis_loss_factor * time_axis_loss + (1-time_axis_loss_factor) * channel_axis_loss`
        """

        num_layers = len(gate_logits)
        batch_size, seq_len = attention_mask.shape
        compute_device = gate_logits[0].device

        # concatenated_gate_logits: [num_layers * batch_size * seq_len, num_experts]
        concatenated_gate_logits = torch.cat([layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0)
        
        # gate routing weights [num_layers * batch_size * seq_len, num_experts]
        routing_weights = torch.nn.functional.softmax(concatenated_gate_logits, dim=-1)

        # expert_mask shape [num_layers * batch * seq_len, topk, num_experts]
        _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts).float()

        time_axis_loss = self.cal_time_axis_load_balancing_loss(
            expert_mask = expert_mask,
            routing_weights = routing_weights,
            attention_mask = attention_mask,
            top_k = top_k,
            num_layers = num_layers,
            batch_size = batch_size,
            seq_len = seq_len,
            num_experts = num_experts,
            device = compute_device
        )

        if self.config.use_channel_axis_loss == True:
            channel_axis_loss = self.cal_channel_axis_load_balancing_loss(
                expert_mask = expert_mask,
                routing_weights = routing_weights,
                attention_mask = attention_mask,
                group_ids = group_ids,
                top_k = top_k,
                num_layers = num_layers,
                batch_size = batch_size,
                seq_len = seq_len,
                num_experts = num_experts,
                device = compute_device
            )

            load_balancing_loss = time_axis_loss_factor * time_axis_loss + (1-time_axis_loss_factor) * channel_axis_loss
        else:
            load_balancing_loss = time_axis_loss
        

        return load_balancing_loss


    def prepare_inputs_for_generation(
            self, input_ids, group_ids, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        """
        Data format prepration for auto-regression input. Including KV Cache management, position ID gengeration and
        attention mask clipping, ect. 

        Call before every forward during inferecing.
        During inference: predict() -> _greedy_search() -> prepare_inputs_for_generation()
        """

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None:
            logger.info('Use input_embedding')
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "attention_mask": attention_mask,
                "group_ids": group_ids
            }
        )
        return model_inputs

