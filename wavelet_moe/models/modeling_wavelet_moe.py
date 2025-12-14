import os
import warnings
import math
from typing import Optional, Tuple, List, Union

import numpy as np
import torch
from torch import nn
from transformers import PreTrainedModel, Cache, DynamicCache, StaticCache
from transformers.activations import ACT2FN
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.utils import logging
from transformers.modeling_outputs import MoeModelOutputWithPast, MoeCausalLMOutputWithPast

from wavelet_moe.models.configuration_wavelet_moe import WaveletMoeConfig
from wavelet_moe.models.wavelet_moe_output import WaveletMoeModelOutputWithPast, WaveletMoeCausalLMOutputWithPast
from wavelet_moe.models.wavelet_generation_mixin import WaveletGenerationMixin

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


class WaveletMoeEmbedderBlock(torch.nn.Module):
    """
    Use a mlp layer to embedding the input ids.
    Embed `[batch_sz, token_num, token_len]` to `[batch_sz, token_num, hidden_sz]`.
    """

    def __init__(self, config: WaveletMoeConfig):
        super().__init__()
        self.config = config
        self.token_len = config.token_len
        self.hidden_size = config.hidden_size
        self.emb_layer = nn.Linear(self.token_len, self.hidden_size, bias=False)
        self.gate_layer = nn.Linear(self.token_len, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        """Embed `[batch_sz, token_num, token_len]` -> `[batch_sz, token_num, hidden_sz]`"""
        emb = self.act_fn(self.gate_layer(x)) * self.emb_layer(x)
        return emb


class WaveletMoeInputLayer(torch.nn.Module):
    """
    Input layer of WaveletMoE, contains two embedding layers, respectively for time sequence & wavelet coefficient sequence.
    """
    def __init__(self, config: WaveletMoeConfig):
        super().__init__()
        self.time_embed_layer = WaveletMoeEmbedderBlock(config)
        self.wavelet_embed_layer = WaveletMoeEmbedderBlock(config)
    
    def forward(self, time_seq: torch.FloatTensor, wavelet_seq: torch.FloatTensor):
        """
        Embed `time_seq` & `wavelet_seq` respectively.
        
        Args:
         time_seq: `torch.FloatTensor`, shape `[batch_sz, token_num, token_len]`
         wavelet_seq: `torch.FloatTensor`, shape `[batch_sz, token_num, token_len]`
        
        Returns:
         (time_seq_embeds, wavelet_seq_embeds): two embeddings, shape `[batch_sz, token_num, hidden_sz]`
        """
        return self.time_embed_layer(time_seq), self.wavelet_embed_layer(wavelet_seq)


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


class WaveletMoeMLPBlock(torch.nn.Module):
    """
    Tempolate for MLP, used in MoE layer and dense model
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


class WaveletMoeDenseFFN(WaveletMoeMLPBlock):
    """
    Wrapper of class WaveletMoeMLPBlock, use in ablation.
    """
    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str):
        super().__init__(hidden_size, intermediate_size, hidden_act)

    def forward(self, hidden_state):
        return super().forward(hidden_state), None


class WaveletMoeDualRouter(torch.nn.Module):
    """
    Router for dual MoE, combining time-sequence & wavelet coeff information for routing.
    """

    def __init__(self, config: WaveletMoeConfig):
        self.hidden_size = config.hidden_size
        self.before_routing_norm = WaveletMoeRMSNorm(config.hidden_size * 2, eps = config.rms_norm_eps)
        self.routed_experts_gate = torch.nn.Linear(config.hidden_size * 2, config.num_experts, bias=False)
        self.shared_experts_gate = torch.nn.Linear(config.hidden_size * 2, 1, bias=False)

    def forward(
        self,
        time_hidden_states: torch.FloatTensor,
        wavelet_hidden_states: torch.FloatTensor,
    ):
        """
        Calculate router logits for routed experts & shared experts, 
        combining time-sequence & wavelet coeff information for routing.

        Args:
         time_hidden_states (`torch.FloatTensor`): shape `[batch_sz, token_num, hidden_sz]`.
         wavelet_hidden_states (`torch.FloatTensor`): shape `[batch_sz, token_num, hidden_sz]`.
        
        Returns:
         (routed_experts_router_logits, shared_experts_router_logits):
         - **routed_experts_router_logits**: (`torch.FloatTensor`), shape `[batch_sz * token_num, routed_experts_num]`.
         - **shared_experts_router_logits**: (`torch.FloatTensor`), shape `[batch_sz * token_num, shared_experts_num]`, `shared_experts_num==1` by default.
        """

        # shape [batch_sz, token_num, hidden_sz * 2] -> [batch_sz * token_num, hidden_sz * 2]
        concat_hidden_states = torch.cat([time_hidden_states, wavelet_hidden_states], dim=2)
        concat_hidden_states = concat_hidden_states.view(-1, self.hidden_size * 2)

        # normalize between time & wavelet modality
        concat_hidden_states = self.before_routing_norm(concat_hidden_states)

        # routing for routed experts, shape [batch_sz * token_num, hidden_sz * 2]
        routed_experts_router_logits = self.routed_experts_gate(concat_hidden_states)

        # routing for shared experts, shape `[batch_sz * token_num, shared_experts_num]`, `shared_experts_num==1` by default.
        shared_experts_router_logits = self.shared_experts_gate(concat_hidden_states)

        return routed_experts_router_logits, shared_experts_router_logits


class WaveletMoeSparseExpertsBlock(torch.nn.Module):
    """
    MoE with dual-modality routering & residual add.
    """
    def __init__(self, config: WaveletMoeConfig):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok     # top-k per token
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts

        # scaling expert's intermediate layer's size, s.t. keep compute cost
        moe_intermediate_size = self.config.intermediate_size // self.top_k

        self.routed_experts = nn.ModuleList(
            [WaveletMoeMLPBlock(
                hidden_size = self.config.hidden_size,
                intermediate_size = moe_intermediate_size,
                hidden_act = self.config.hidden_act,
            ) for _ in range(self.num_experts)]
        )

        self.shared_expert = WaveletMoeMLPBlock(
            hidden_size = self.config.hidden_size,
            intermediate_size = self.config.intermediate_size,
            hidden_act = self.config.hidden_act,
        )

    def forward(
        self, 
        hidden_states: torch.FloatTensor,
        routed_experts_router_logits: torch.FloatTensor,
        shared_experts_router_logits: torch.FloatTensor
    ):
        """
        Args:
         hidden_states (`torch.FloatTensor`): shape `[batch_sz, token_num, hidden_sz]`.
         routed_experts_router_logits (`torch.FloatTensor`): shape `[batch_sz * token_num, routed_experts_num]`.
         shared_experts_router_logits (`torch.FloatTensor`): shape `[batch_sz * token_num, shared_experts_num]`, `shared_experts_num==1` by default.
        
        Returns:
         final_hidden_states: (`torch.FloatTensor`), shape `[batch_sz, token_num, hidden_sz]`.
        """

        # Prepare data & routing
        
        batch_sz, token_num, hidden_sz = hidden_states.shape

        # shape [batch_sz * token_num, hidden_sz]
        hidden_states = hidden_states.view(-1, hidden_sz)

        # Forward thourgh routed experts
        # calculate routing weights: softmax & top-k
        routing_weights = nn.functional.softmax(routed_experts_router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)
        
        # init output shape
        final_hidden_states = torch.zeros((batch_sz * token_num, hidden_sz), dtype = hidden_states.dtype, device = hidden_states.device)

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        # expert_mask: [num_experts, top_k, batch_sz * token_num]
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        # ie forward thorugh selected experts
        for expert_idx in range(self.num_experts):
            expert_layer = self.routed_experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x].reshape(-1, hidden_sz)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

        # forward through shared expert
        shared_expert_output = self.shared_expert(hidden_states)
        shared_expert_output = nn.functional.sigmoid(shared_experts_router_logits) * shared_expert_output

        # fuse selected expert's outputs & shared expert's output
        final_hidden_states = final_hidden_states + shared_expert_output

        final_hidden_states = final_hidden_states.reshape(batch_sz, token_num, hidden_sz)
        return final_hidden_states


class WaveletMoeDualMoeLayer(torch.nn.Module):
    """
    MoE layer contain two parallel sparse experts block, respectively for time-series sequences & wavelet coeff sequences.
    Time & wavelet information are combined in routing. 
    Conduct residual add after forward through experts.
    """
    def __init__(self, config: WaveletMoeConfig):
        super().__init__()
        self.config = config
        self.before_moe_time_norm = WaveletMoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.before_moe_wavelet_norm = WaveletMoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.router = WaveletMoeDualRouter(config)
        self.time_sparse_experts_block = WaveletMoeSparseExpertsBlock(config)
        self.wavelet_sparse_experts_block = WaveletMoeSparseExpertsBlock(config)
    
    def forward(
        self,
        time_hidden_states: torch.FloatTensor,
        wavelet_hidden_states: torch.FloatTensor
    ):
        """
        Args:
         time_hidden_states (`torch.FloatTensor`): shape `[batch_sz, token_num, hidden_sz]`.
         wavelet_hidden_states (`torch.FloatTensor`): shape `[batch_sz, token_num, hidden_sz]`.
        
        Returns:
         (time_hidden_states, wavelet_hidden_states, routed_experts_router_logits):
         - **time_hidden_states**: (`torch.FloatTensor`), shape `[batch_sz, token_num, hidden_sz]`.
         - **wavelet_hidden_states**: (`torch.FloatTensor`), shape `[batch_sz, token_num, hidden_sz]`.
         - **routed_experts_router_logits**: (`torch.FloatTensor`), shape `[batch_sz * token_num, routed_experts_num]`.
        """

        time_residual, wavelet_residual = time_hidden_states, wavelet_hidden_states

        # norm before MoE
        time_hidden_states = self.before_moe_time_norm(time_hidden_states)
        wavelet_hidden_states = self.before_moe_wavelet_norm(time_hidden_states)

        # shape [batch_sz * token_num, routed_experts_num] & [batch_sz * token_num, shared_experts_num]
        routed_experts_router_logits, shared_experts_router_logits = self.router(time_hidden_states, wavelet_hidden_states)

        # forward through experts
        # shape [batch_sz, token_num, hidden_sz]
        time_hidden_states = self.time_sparse_experts_block(
            hidden_states = time_hidden_states,
            routed_experts_router_logits = routed_experts_router_logits,
            shared_experts_router_logits = shared_experts_router_logits
        )
        wavelet_hidden_states = self.wavelet_sparse_experts_block(
            hidden_states = wavelet_hidden_states,
            routed_experts_router_logits = routed_experts_router_logits,
            shared_experts_router_logits = shared_experts_router_logits
        )

        # residual add
        time_hidden_states = time_hidden_states + time_residual
        wavelet_hidden_states = wavelet_hidden_states + wavelet_residual

        return time_hidden_states, wavelet_hidden_states, routed_experts_router_logits


class WaveletMoeFilteredMHABlock(torch.nn.Module):
    """
    Multi-headed attention with KV filtering & droupout.
    """

    def __init__(self, config: WaveletMoeConfig):
        super().__init__()
        self.config = config

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
    
    def _select_topk_kv(
        self, 
        attn_weights: torch.FloatTensor, 
        causal_attn_mask: torch.FloatTensor
    ):
        """
        Select topk KV pair by ranking Q@K^T and update attention mask.

        Args:
         attn_weights: shape `[batch_sz, head_num, q_len, kv_len]`
         causal_attn_mask: shape `[batch_sz, 1, q_len, kv_len]`, a down traingle 4d attn mask
        
        Returns:
         causal_attn_mask: shape `[batch_sz, head_num, q_len, kv_len]
        """

        topk = self.config.topk_kv
        bsz, head_num, q_len, kv_len = attn_weights.shape

        # broadcast causal mask to head_num
        causal_attn_mask = causal_attn_mask.expand(-1, head_num, -1, -1)

        # mask future token before select topk
        masked_attn_scores = attn_weights + causal_attn_mask

        # select topk along kv_len
        _, topk_indices = torch.topk(masked_attn_scores, k = min(topk, kv_len), dim=-1)

        # build topk mask
        topk_mask = torch.full_like(masked_attn_scores, float("-inf"))
        topk_mask.scatter_(dim = -1, index = topk_indices, value = 0.0)

        causal_attn_mask = causal_attn_mask + topk_mask

        return causal_attn_mask

    def forward(
            self,
            hidden_states: torch.FloatTensor,
            causal_attn_mask: torch.Tensor,
            position_ids: Optional[torch.LongTensor] = None,
            output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        MHA, select top-k KV pair by ranking Q@K^T, in order to filter outliers & anomalies.

        Args:
         hidden_states: `[batch_sz, q_len, hidden_sz]`
         causal_attn_mask: `[batch_sz, 1, q_len, kv_len]`
         position_ids: `[batch_sz, q_len]`
         output_attentions: `bool`
        
        Returns:
         (attn_output, attn_weights):
         - attn_output: `[batch_sz, q_len, hidden_sz]`
         - attn_weights: `None` or `[batch_sz, head_num, q_len, kv_len]`
        """
        
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

        if attn_weights.size() != (bsz, 1, q_len, kv_seq_len):
            raise ValueError(f"Self attention weights should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attn_weights.size()}")

        # masking attention weights with attn_mask
        if causal_attn_mask is not None:

            if causal_attn_mask.size() != (bsz, self.num_heads, q_len, kv_seq_len):
                raise ValueError(f"Time Attention mask should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is {causal_attn_mask.size()}")
            
            if self.config.use_topk_kv:
                causal_attn_mask = self._select_topk_kv(attn_weights, causal_attn_mask)
            
            # apply mask
            attn_weights = attn_weights + causal_attn_mask

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


class WaveletMoeAttentionBlock(torch.nn.Module):
    """
    Attention block: Layer norm -> Filter MHA & droupout -> Redisual add -> LayerNorm
    """
    def __init__(self, config: WaveletMoeConfig):
        super().__init__()
        self.before_attn_layernorm = WaveletMoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = WaveletMoeFilteredMHABlock(config)
    
    def forward(
        self,
        hidden_states: torch.FloatTensor,
        causal_attn_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = False
    ):
        """
        Args:
         hidden_states (`torch.FloatTensor`): shape `[batch_sz, token_num, hidden_sz]`.
         causal_attn_mask (`torch.Tensor`, *optional*): shape `[batch_sz, 1, q_len, kv_len]`.
         position_ids (`torch.LongTensor`, *optional*): shape `[batch_size, token_num]`.
         output_attentions (`bool`, *optional*):
        
        Returns:
         (hidden_states, attn_weights):
         - hidden_states: `[batch_sz, token_num, hidden_sz]`
         - attn_weights: `None` or `[batch_sz, head_num, q_len, kv_len]`
        """

        residual = hidden_states
        hidden_states = self.before_attn_layernorm(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states = hidden_states,
            causal_attn_mask = causal_attn_mask,
            position_ids = position_ids,
            output_attentions = output_attentions
        )
        hidden_states = residual + hidden_states
        
        return hidden_states, attn_weights


class WaveletMoeDualAttentionLayer(torch.nn.Module):
    """
    Attention layer contain two parallel attention block, respectively for time-series sequences & wavelet coeff sequences.
    """
    def __init__(self, config: WaveletMoeConfig):
        super().__init__()
        self.time_attn_block = WaveletMoeAttentionBlock(config)
        self.wavelet_attn_block = WaveletMoeAttentionBlock(config)
    
    def forward(
        self,
        time_hidden_states: torch.FloatTensor,
        wavelet_hidden_states: torch.FloatTensor,
        causal_attn_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = False
    ):
        """
        Args:
         time_seq_embeds (`torch.FloatTensor`): shape `[batch_sz, token_num, hidden_sz]`.
         wavelet_seq_embeds (`torch.FloatTensor`): shape `[batch_sz, token_num, hidden_sz]`.
         causal_attn_mask (`torch.Tensor`, *optional*): shape `[batch_sz, 1, q_len, kv_len]`.
         position_ids (`torch.LongTensor`, *optional*): shape `[batch_size, token_num]`.
         output_attentions (`bool`, *optional*):
        
        Returns:
         (time_hidden_states, wavelet_hidden_states, time_attn_weights, wavelet_attn_weights):
        """

        time_hidden_states, time_attn_weights = self.time_attn_block(
            hidden_states = time_hidden_states,
            causal_attn_mask = causal_attn_mask,
            position_ids = position_ids,
            output_attentions = output_attentions
        )

        wavelet_hidden_states, wavelet_attn_weights = self.wavelet_attn_block(
            hidden_states = wavelet_hidden_states,
            causal_attn_mask = causal_attn_mask,
            position_ids = position_ids,
            output_attentions = output_attentions
        )

        return time_hidden_states, wavelet_hidden_states, time_attn_weights, wavelet_attn_weights


class WaveletMoeDecoderLayer(torch.nn.Module):
    """
    Implement of WaveletMoE transformer block
    """
    def __init__(self, config: WaveletMoeConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        # Dual Filtered MHA
        # TODO: add setting for potential ablation
        self.dual_attn_layer = WaveletMoeDualAttentionLayer(config)

        # MoE or dense FFN
        if self.config.use_dense:
            # TODO: potential ablation setting
            raise NotImplementedError("Dense FFN not yet implemented for dual attention setting")
            self.ffn_layer = WaveletMoeDenseFFN(
                hidden_size=self.config.hidden_size,
                intermediate_size=self.config.intermediate_size,
                hidden_act=self.config.hidden_act,
            )
        else:
            self.ffn_layer = WaveletMoeDualMoeLayer(config)

    def forward(
            self,
            time_hidden_states: torch.FloatTensor,
            wavelet_hidden_states: torch.FloatTensor,
            causal_attn_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            output_attentions: Optional[bool] = False
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, Optional[torch.FloatTensor], Optional[torch.FloatTensor]]:
        """
        Args:
         time_seq_embeds (`torch.FloatTensor`): shape `[batch_sz, token_num, hidden_sz]`.
         wavelet_seq_embeds (`torch.FloatTensor`): shape `[batch_sz, token_num, hidden_sz]`.
         causal_attn_mask (`torch.Tensor`, *optional*): shape `[batch_sz, 1, q_len, kv_len]`.
         position_ids (`torch.LongTensor`, *optional*): shape `[batch_size, token_num]`.
         output_attentions (`bool`, *optional*):
        
        Returns:
        - **time_hidden_states**: `torch.FloatTensor`, shape `[batch_sz, token_num, hidden_sz]`.
        - **wavelet_hidden_states**: `torch.FloatTensor`, shape `[batch_sz, token_num, hidden_sz]`.
        - **time_attn_weights**: `tuple(torch.FloatTensor)`, *optional*
        - **wavelet_attn_weights**: `tuple(torch.FloatTensor)`, *optional*
        - **router_logits**: `tuple(torch.FloatTensor)`, *optional*
        """

        if time_hidden_states.shape != wavelet_hidden_states.shape:
            raise ValueError(f"time_hidden_states.shape [{time_hidden_states.shape}] should equal to wavelet_hidden_states.shape [{wavelet_hidden_states.shape}]!")
        if time_hidden_states.dtype != wavelet_hidden_states.dtype:
            raise ValueError(f"time_hidden_states.dtype [{time_hidden_states.dtype}] should equal to wavelet_hidden_states.dtype [{wavelet_hidden_states.dtype}]!")
        if time_hidden_states.device != wavelet_hidden_states.device:
            raise ValueError(f"time_hidden_states.device [{time_hidden_states.device}] should equal to wavelet_hidden_states.device [{wavelet_hidden_states.device}]!")
        

        # Dual Attn Layer
        time_hidden_states, wavelet_hidden_states, time_attn_weights, wavelet_attn_weights = self.dual_attn_layer(
            time_hidden_states,
            wavelet_hidden_states,
            causal_attn_mask,
            position_ids,
            output_attentions
        )

        time_hidden_states, wavelet_hidden_states, router_logits = self.ffn_layer(time_hidden_states, wavelet_hidden_states)

        # output settings
        if not output_attentions:
            time_attn_weights = None
            wavelet_attn_weights = None
        
        return {
            "time_hidden_states": time_hidden_states,
            "wavelet_hidden_states": wavelet_hidden_states,
            "time_attn_weights": time_attn_weights,
            "wavelet_attn_weights": wavelet_attn_weights,
            "router_logits": router_logits
        }


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


class WaveletMoeModel(WaveletMoePreTrainedModel):
    """
    `WaveletMoeModel = WaveletMoeInputLayer + WaveletMoeDecoderLayer * config.num_hidden_layers`
    
    Args:
        config: WaveletMoeConfig
    """

    def __init__(self, config: WaveletMoeConfig):
        super().__init__(config)
        self.input_layer = WaveletMoeInputLayer(config)

        self.decoder_layers = nn.ModuleList(
            [WaveletMoeDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

        self.time_norm = WaveletMoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.wavelet_norm = WaveletMoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def _prepare_causal_attn_mask(
        self,
        attention_mask: torch.Tensor,
        input_shape: Union[torch.Size, Tuple, List],
        inputs_embeds: torch.Tensor
    ):
        """
        Generate 4d causal attention mask for self attention, mask along time axis. \n

        Args:
         attention_mask: `[batch_sz, kv_len]`
         input_shape: `(batch_sz, q_len)`
         inputs_embeds: `torch.Tensor`, use it to get device and dtype info

        Returns:
         causal_attn_mask: `[batch_sz, 1, q_len, kv_len]`
        """

        causal_attn_mask = _prepare_4d_causal_attention_mask(
            attention_mask,
            input_shape,
            inputs_embeds,
            0,
        )
    
        return causal_attn_mask

    def forward(
            self,
            time_seq: torch.FloatTensor = None,
            wavelet_seq: torch.FloatTensor = None,
            time_seq_embeds: torch.FloatTensor = None,
            wavelet_seq_embeds: torch.FloatTensor = None,
            loss_masks: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,    
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None
    ) -> Union[Tuple, WaveletMoeModelOutputWithPast]:
        """
        **Notation**
        - Since KV cache are not yet implemented, we have `token_num == q_len == kv_len`

        Args:
         time_seq: `torch.FloatTensor`, shape `[batch_sz, token_num, token_len]`.
         wavelet_seq: `torch.FloatTensor`, shape `[batch_sz, token_num, token_len]`.
         time_seq_embeds: `torch.FloatTensor`, shape `[batch_sz, token_num, hidden_sz]`.
         wavelet_seq_embeds: `torch.FloatTensor`, shape `[batch_sz, token_num, hidden_sz]`.
         
         loss_masks: `[batch_sz, token_num]`
         attention_mask: `None` or `[batch_sz, kv_len]`
          - when training: first `None`; then generate in forward with shape `[batch_sz, 1, token_num, token_num]`.
          - when inference: shape `[batch_sz, kv_len]` -> `[batch_sz, 1, q_len, kv_len]`
         
         position_ids: Position ids for RoPE, generate in first forward & update during forward
          - when first generate: shape ``[1, token_num]`
          - when provided: update shape to `[batch_size, token_num]`.
         input_embeds: shape `[batch_sz, token_num, hidden_sz]`, generate during forward.
        
         output_attentions: output setting, load from self.config.
         ouput_hidden_states: output setting, load from self.config.
        
        Returns:
         WaveletMoeModelOutputWithPast: 
        """

        # set default param
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # check input shapes & devices
        if time_seq is not None and time_seq_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif time_seq is not None and wavelet_seq is not None:
            if time_seq.device != wavelet_seq.devices:
                wavelet_seq.to(time_seq.device)
            batch_size, token_num, _ = time_seq.shape
        elif time_seq_embeds is not None and wavelet_seq_embeds is not None:
            if time_seq_embeds.device != wavelet_seq_embeds.device:
                wavelet_seq_embeds.to(time_seq_embeds.device)
            batch_size, token_num, _ = time_seq_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        # generate position_ids for RoPE
        if position_ids is None:
            device = time_seq.device if time_seq is not None else time_seq_embeds.device
            position_ids = torch.arange(0, token_num, dtype=torch.long, device=device)
            position_ids = position_ids.view(-1, token_num)
        else:
            position_ids = position_ids.view(-1, token_num).long()

        # start forwarding from here
        # forward through input layer if input arent embedding
        # shape: [batch_sz, token_num, token_len] -> [batch_sz, token_num, hidden_sz]
        if time_seq_embeds is None:
            time_seq_embeds, wavelet_seq_embeds = self.input_layer(time_seq, wavelet_seq)

        # generate time_attn_mask & group_attn_mask using loss_mask (training) or attention_mask (inferencing)
        if attention_mask is None and loss_masks is None:
            raise ValueError("attention_mask and loss_mask shouldn't be None at same time!")

        # use loss_masks when training
        if loss_masks is not None:
            attention_mask = loss_masks

        # generate 4d causal attn mask
        causal_attn_mask = self._prepare_causal_attn_mask(
            attention_mask,
            (batch_size, token_num),
            time_seq_embeds
        )

        time_hidden_states, wavelet_hidden_states = time_seq_embeds, wavelet_seq_embeds

        # decoder layers
        # init output as config setting
        all_time_hidden_states = () if output_hidden_states else None
        all_time_self_attns = () if output_attentions else None

        all_wavelet_hidden_states = () if output_hidden_states else None
        all_wavelet_self_attns = () if output_attentions else None

        all_router_logits = ()

        # forwarding through decoder layers
        for decoder_layer in self.decoder_layers:
            if output_hidden_states:
                all_time_hidden_states += (time_hidden_states,)
                all_wavelet_hidden_states += (wavelet_hidden_states, )


            # forward setting: load gradient checkpoint or normal forwarding
            if self.gradient_checkpointing and self.training:
                decoder_layer_output = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    time_hidden_states,
                    wavelet_hidden_states,
                    causal_attn_mask,
                    position_ids,
                    output_attentions
                )
            else:
                decoder_layer_output = decoder_layer(
                    time_hidden_states,
                    wavelet_hidden_states,
                    causal_attn_mask=causal_attn_mask,
                    position_ids=position_ids,
                    output_attentions=output_attentions
                )

            # hidden_states from layer output
            time_hidden_states = decoder_layer_output["time_hidden_states"]
            wavelet_hidden_states = decoder_layer_output["wavelet_hidden_states"]

            if output_attentions:
                all_time_self_attns += (decoder_layer_output["time_self_attn_weights"],)
                all_wavelet_self_attns += (decoder_layer_output["wavelet_self_attn_weights"],)
            
            # collect MoE router logits
            all_router_logits += (decoder_layer_output["router_logits"],)

        # normalize last layer's hidden_states
        time_hidden_states = self.time_norm(time_hidden_states)
        wavelet_hidden_states = self.wavelet_norm(wavelet_hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_time_hidden_states += (time_hidden_states,)
            all_wavelet_hidden_states += (wavelet_hidden_states,)

        return WaveletMoeModelOutputWithPast(
            last_time_hidden_state = time_hidden_states, 
            last_wavelet_hidden_state = wavelet_hidden_states,
            all_time_hidden_states = all_time_hidden_states,
            all_wavelet_hidden_states = all_wavelet_hidden_states,
            all_time_self_attns = all_time_self_attns,
            all_wavelet_self_attns = all_wavelet_self_attns,
            all_router_logits = all_router_logits
        )


class WaveletMoePredictionHead(torch.nn.Module):
    """
    Prediction head: a MLP from [batch_sz, token_num, hidden_sz] → [batch_sz, token_num, horizon_len, token_len]
    """
    def __init__(self, hidden_sz: int, horizon_len: int, token_len: int = 8):
        super().__init__()
        self.horizon_len = horizon_len
        self.token_len = token_len
        self.out_layer = nn.Linear(hidden_sz, token_len * horizon_len, bias=False) # multi-step prediction

    def forward(self, hidden_states: torch.FloatTensor):
        """
        Args:
         hidden_states (`torch.FloatTensor`): shape [batch_sz, token_num, hidden_sz]

        Returns:
         prediction_outputs: (`torch.FloatTensor`), shape [batch_sz, token_num, horizon_len, token_len]
        """

        batch_sz, token_num, _ = hidden_states.shape

        # shape [batch_sz, token_num, horizon_len * token_len] -> [batch_sz, token_num, horizon_len, token_len]
        prediction_outputs = self.out_layer(hidden_states)
        prediction_outputs = prediction_outputs.view(batch_sz, token_num, self.horizon_len, self.token_len)

        return prediction_outputs


class WaveletMoeDualOutputHead(torch.nn.Module):
    """
    Dual prediction head contains two `WaveletMoePredictionHead`
    """
    def __init__(self, hidden_sz: int, horizon_len: int, token_len: int = 8):
        super().__init__()
        self.time_predict_head = WaveletMoePredictionHead(hidden_sz, horizon_len, token_len)
        self.wavelet_predict_head = WaveletMoePredictionHead(hidden_sz, horizon_len, token_len)
    
    def forward(
        self,
        time_hidden_states: torch.FloatTensor,
        wavelet_hidden_states: torch.FloatTensor
    ):
        """
        Args:
         time_hidden_states (`torch.FloatTensor`): shape `[batch_sz, token_num, hidden_sz]`.
         wavelet_hidden_states (`torch.FloatTensor`): shape `[batch_sz, token_num, hidden_sz]`.
        
        Returns:
         (time_predictions, wavelet_predictions):
         - **time_hidden_states**: (`torch.FloatTensor`), shape `[batch_sz, token_num, horizon_len, token_len]`.
         - **wavelet_hidden_states**: (`torch.FloatTensor`), shape `[batch_sz, token_num, horizon_len, token_len]`.
        """
        time_hidden_states = self.time_predict_head(time_hidden_states)
        wavelet_hidden_states = self.wavelet_predict_head(wavelet_hidden_states)
        return time_hidden_states, wavelet_hidden_states
        


# TODO: delete rebundent argments like output_attentions & return dict
class WaveletMoeForPrediction(WaveletMoePreTrainedModel, WaveletGenerationMixin):

    def __init__(self, config: WaveletMoeConfig):
        super().__init__(config)
        self.config = config
        self.num_experts_per_tok = config.num_experts_per_tok
        self.apply_aux_loss = config.apply_aux_loss
        self.router_aux_loss_factor = config.router_aux_loss_factor

        self.model = WaveletMoeModel(config)

        # output layer
        dual_output_head_list = []
        self.horizon_length_map = {}
        for i, horizon_length in enumerate(config.horizon_lengths):
            dual_output_head_list.append(
                WaveletMoeDualOutputHead(
                    hidden_sz = self.config.hidden_size,
                    horizon_len = horizon_length,
                    token_len = self.config.token_len
                )
            )
            self.horizon_length_map[horizon_length] = i
        self.dual_output_heads = nn.ModuleList(dual_output_head_list)

        # select loss function
        self.loss_function_name = config.loss_func
        if self.loss_function_name == "huber":
            self.loss_function = torch.nn.HuberLoss(reduction='none', delta=2.0)
        elif self.loss_function_name == "mse":
            self.loss_function = torch.nn.MSELoss(reduction='none')
        else:
            raise ValueError(f"Unsupport loss function: {self.loss_function_name}")

        # Initialize weights and apply final processing
        self.post_init()

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
            self,
            time_seq: torch.FloatTensor = None,
            wavelet_seq: torch.FloatTensor = None,

            time_seq_embeds: Optional[torch.FloatTensor] = None,
            wavelet_seq_embeds: Optional[torch.FloatTensor] = None,

            time_seq_labels: Optional[torch.FloatTensor] = None,
            wavelet_seq_labels: Optional[torch.FloatTensor] = None,

            loss_masks: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            max_horizon_length: Optional[int] = None,
            
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
    ) -> Union[Tuple, WaveletMoeCausalLMOutputWithPast]:
        """
        Args:
         time_seq: `torch.FloatTensor`, shape `[batch_sz, token_num, token_len]`.
         wavelet_seq: `torch.FloatTensor`, shape `[batch_sz, token_num, token_len]`.
         time_seq_embeds: `torch.FloatTensor`, shape `[batch_sz, token_num, hidden_sz]`.
         wavelet_seq_embeds: `torch.FloatTensor`, shape `[batch_sz, token_num, hidden_sz]`.
         time_seq_labels: `torch.FloatTensor`, shape `[batch_sz, token_num, token_len]`, use in training.
         wavelet_seq_labels: `torch.FloatTensor`, shape `[batch_sz, token_num, token_len]`, use in training.
         labels: `torch.FloatTensor`, shape `[batch_sz, token_num, token_len]`, use in training.
         loss_masks: `torch.FloatTensor`, shape `[batch_sz, token_num]`.
         attention_mask: `None` or `[batch_sz, token_num]`.
         position_ids: `None` or `[batch_sz, token_num]`.
         max_horizon_length: `None` or `int`, use in inferencing.
        
        Returns:
         WaveletMoeCausalLMOutputWithPast: 
        """

        # output setting
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            time_seq = time_seq,
            wavelet_seq = wavelet_seq,
            time_seq_embeds = time_seq_embeds,
            wavelet_seq_embeds = wavelet_seq_embeds,
            loss_masks = loss_masks,
            attention_mask=attention_mask,
            position_ids=position_ids,

            # output setting
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        time_hidden_state, wavelet_hidden_state = outputs.last_time_hidden_state, outputs.last_wavelet_hidden_state
        time_predictions, wavelet_predictions = None, None
        loss, aux_loss = None, None

        # when training
        if time_seq_labels is not None:  
            # auto regressive loss
            ar_loss, time_ar_loss, wavelet_ar_loss = 0.0, 0.0, 0.0

            for output_head, horizon_length in zip(self.dual_output_heads, self.config.horizon_lengths):
                one_time_predictions, one_wavelet_predictions = output_head(time_hidden_state, wavelet_hidden_state)

                one_time_ar_loss = self._calc_ar_loss(one_time_predictions, time_seq_labels, loss_masks, horizon_length)
                one_wavelet_ar_loss = self._calc_ar_loss(one_wavelet_predictions, wavelet_seq_labels, loss_masks, horizon_length)
                
                ar_loss += (one_time_ar_loss + one_wavelet_ar_loss)
                
                if time_predictions is None:
                    time_predictions, wavelet_predictions = one_time_predictions, one_wavelet_predictions

            loss = ar_loss / len(self.config.horizon_lengths)

            if self.apply_aux_loss:
                router_logits = outputs.router_logits

                temporal_aux_loss = self._calc_load_balancing_loss(
                    router_logits
                    top_k = self.num_experts_per_tok,
                    num_experts = self.config.num_experts,
                    attention_mask = loss_masks
                )

                loss += self.router_aux_loss_factor * temporal_aux_loss.to(loss.device)
        
        # TODO: inference modules for DualWaveletMoE
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

        return WaveletMoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=predictions,
            hidden_states=outputs.hidden_states,
            time_attentions=outputs.time_attentions,
            gruop_attentions=outputs.gruop_attentions,
        )


    def _calc_ar_loss(self, predictions, labels, loss_masks, horizon_len):
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
        loss_masks_padded = nn.functional.pad(loss_masks, (0, horizon_len - 1))  # [B, token_num + horizon_len - 1]
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


    def _cal_time_axis_load_balancing_loss(
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


    def _calc_load_balancing_loss(
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

        load_balancing_loss = self._cal_time_axis_load_balancing_loss(
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
        
        return load_balancing_loss


    # TODO: merge loss_masks & attn_mask here.
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

