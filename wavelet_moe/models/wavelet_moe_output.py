from transformers.modeling_outputs import ModelOutput
from typing import Optional, Tuple
from dataclasses import dataclass
import torch

@dataclass
class WaveletMoeModelOutputWithPast(ModelOutput):
    """
    Base class for WaveletMoE model's outputs, with potential hidden states and attentions.

    Args:
        last_time_hidden_state: (`torch.FloatTensor`, shape `(batch_sz, token_num, hidden_sz)`)
            Time series sequences hidden-states at the output of the last layer of the model.
        
        last_wavelet_hidden_state: (`torch.FloatTensor`, shape `(batch_sz, token_num, hidden_sz)`)
            Wavelet coeff sequences hidden-states at the output of the last layer of the model.
        
        all_time_hidden_states: (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`)
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_sz, token_num, hidden_sz)`.

            Time series sequences hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        
        all_wavelet_hidden_states: (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`)
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_sz, token_num, hidden_sz)`.

            Wavelet coeff sequences hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
            
        
        all_time_self_attns: (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`)
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_sz, head_num, token_num, token_num)`.
            
            Attentions weights of time series sequence after the attention softmax, used to compute the weighted average in the self-attention heads.
        
        all_wavelet_self_attns: (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`)
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_sz, head_num, token_num, token_num)`.
            
            Attentions weights of wavelet coeff sequence after the attention softmax, used to compute the weighted average in the self-attention heads.
           
        all_router_logits: (`tuple(torch.FloatTensor)`, *optional*, returned when `output_router_probs=True` and `config.add_router_probs=True` is passed or when `config.output_router_probs=True`)
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_sz, token_num, expert_num)`.

            Raw router logtis (post-softmax) that are computed by MoE routers, these terms are used to compute the auxiliary loss for Mixture of Experts models.
    """

    last_time_hidden_state: torch.FloatTensor = None
    last_wavelet_hidden_state: torch.FloatTensor = None
    all_time_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    all_wavelet_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    all_time_self_attns: Optional[Tuple[torch.FloatTensor, ...]] = None
    all_wavelet_self_attns: Optional[Tuple[torch.FloatTensor, ...]] = None
    all_router_logits: Optional[Tuple[torch.FloatTensor]] = None


# Derived from transformers.modeling_outputs.MoeCausalLMOutputWithPast, modify to adapt time & group attn
@dataclass
class WaveletMoeCausalLMOutputWithPast(ModelOutput):
    """
    Base class for causal language model (or autoregressive) with mixture of experts outputs.

    Args:
        loss: (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided)
            Language modeling loss (for next-token prediction).

        logits: (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`)
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).

        aux_loss: (`torch.FloatTensor`, *optional*, returned when `labels` is provided)
            aux_loss for the sparse modules.

        router_logits: (`tuple(torch.FloatTensor)`, *optional*, returned when `output_router_probs=True` and `config.add_router_probs=True` is passed or when `config.output_router_probs=True`)
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, sequence_length, num_experts)`.

            Raw router logtis (post-softmax) that are computed by MoE routers, these terms are used to compute the auxiliary
            loss for Mixture of Experts models.
        
        hidden_states: (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`)
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        
        time_attentions: (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`)
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        
        gruop_attentions: (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`)
    """

    loss: Optional[torch.FloatTensor] = None
    aux_loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    time_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    gruop_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    router_logits: Optional[Tuple[torch.FloatTensor]] = None
