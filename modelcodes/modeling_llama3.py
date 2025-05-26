import math
import warnings
import time
import gc
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers import LlamaTokenizer
from transformers.models.llama.modeling_llama import (
    LlamaMLP,
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaPreTrainedModel, 
    LlamaForCausalLM, 
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
    LLAMA_INPUTS_DOCSTRING,
    CausalLMOutputWithPast,
    _CONFIG_FOR_DOC,
    apply_rotary_pos_emb,
)
from transformers.generation.utils import DynamicCache, Cache

from codes.MLAConfig import MLAConfig

length: torch.Tensor
def set_length_for_training(new_length: torch.Tensor):
    global length
    length = new_length

total_loss = 0.
def get_total_loss() -> float:
    global total_loss
    return total_loss
def clear_total_loss() -> float:
    global total_loss
    total_loss = 0.

class LlamaRotaryEmbeddingModified(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, arr=None, device=None, scaling_factor=1.0):
        super().__init__()
        if arr is None:
            arr = torch.arange(0, dim, 2, dtype=torch.int64)
        self.scaling_factor = scaling_factor
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (arr.float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        # For BC we register cos and sin cached
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)
        t = t / self.scaling_factor
        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("_cos_cached", emb.cos().to(torch.get_default_dtype()), persistent=False)
        self.register_buffer("_sin_cached", emb.sin().to(torch.get_default_dtype()), persistent=False)

    @property
    def sin_cached(self):
        return self._sin_cached
    @property
    def cos_cached(self):
        return self._cos_cached

    @torch.no_grad()
    def forward(self, x, position_ids):
        # x: [bs, num_attention_heads, seq_len, head_size]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class LlamaLinearWithLoRA(nn.Module):
    def __init__(self, original_linear: nn.Linear, r, alpha, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.main_linear = original_linear
        self.rank = r
        self.ratio = alpha / r
        self.LoRA_A = nn.Linear(self.in_features, self.rank, bias=False, 
                                device=original_linear.weight.device, 
                                dtype=original_linear.weight.dtype
                                )
        self.LoRA_B = nn.Linear(self.rank, self.out_features, bias=False, 
                                device=original_linear.weight.device,
                                dtype=original_linear.weight.dtype
                                )
        self.LoRA_B.weight = nn.Parameter(torch.zeros_like(self.LoRA_B.weight))
    
    def save_param(self, path_base: str, name: str):
        torch.save(self.LoRA_A.weight, f"{path_base}{name}_A.pt")
        torch.save(self.LoRA_B.weight, f"{path_base}{name}_B.pt")
    
    def load_param(self, path_base: str, name: str):
        A_weight = torch.load(f"{path_base}{name}_A.pt")
        B_weight = torch.load(f"{path_base}{name}_B.pt")
        self.LoRA_A.weight = nn.Parameter(A_weight)
        self.LoRA_B.weight = nn.Parameter(B_weight)

    def forward(self, x):
        output = self.main_linear(x) + self.LoRA_B(self.LoRA_A(x) * self.ratio)
        return output

    def set_training(self, new_state: bool):
        self.main_linear.requires_grad_(False)
        self.LoRA_A.requires_grad_(new_state)
        self.LoRA_B.requires_grad_(new_state)


class LlamaMLPWithLoRA(nn.Module):
    def __init__(self, original_mlp: LlamaMLP, lora_rank: int, lora_alpha: int):
        super().__init__()
        self.config = original_mlp.config
        self.hidden_size = original_mlp.config.hidden_size
        self.intermediate_size = original_mlp.config.intermediate_size
        self.gate_proj = LlamaLinearWithLoRA(original_mlp.gate_proj, r=lora_rank, alpha=lora_alpha)
        self.up_proj = LlamaLinearWithLoRA(original_mlp.up_proj, r=lora_rank, alpha=lora_alpha)
        self.down_proj = LlamaLinearWithLoRA(original_mlp.down_proj, r=lora_rank, alpha=lora_alpha)
        self.act_fn = original_mlp.act_fn

    def save_param(self, path_base):
        self.up_proj.save_param(path_base, "up_proj")
        self.down_proj.save_param(path_base, "down_proj")
        self.gate_proj.save_param(path_base, "gate_proj")
    
    def load_param(self, path_base):
        self.up_proj.load_param(path_base, "up_proj")
        self.down_proj.load_param(path_base, "down_proj")
        self.gate_proj.load_param(path_base, "gate_proj")

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

    def set_training(self, new_state: bool):
        self.up_proj.set_training(new_state)
        self.down_proj.set_training(new_state)
        self.gate_proj.set_training(new_state) 


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class LlamaAttentionMLA(nn.Module):
    """Multi-headed latent attention"""

    def __init__(self, original_attn: LlamaAttention, 
                 reduce_ratio_qk=1, reduce_ratio_vo=1,
                 ):
        super().__init__()
        self.config = original_attn.config
        self.layer_idx = original_attn.layer_idx

        self.attention_dropout = original_attn.attention_dropout
        self.hidden_size = original_attn.hidden_size
        self.num_heads = original_attn.num_heads
        # self.head_dim = self.hidden_size // self.num_heads
        self.head_dim_qk = original_attn.head_dim // reduce_ratio_qk
        self.head_dim_vo = original_attn.head_dim // reduce_ratio_vo
        self.num_key_value_heads = original_attn.num_key_value_heads
        self.num_key_value_groups = original_attn.num_key_value_groups
        self.max_position_embeddings = original_attn.max_position_embeddings
        self.rope_theta = original_attn.rope_theta
        self.is_causal = True

        self.q_proj = nn.Linear(self.hidden_size, self.head_dim_qk * self.num_heads, False)
        self.q_proj.weight = nn.Parameter(original_attn.q_proj.weight[::reduce_ratio_qk, ...].clone())
        self.k_proj = nn.Linear(self.hidden_size, self.head_dim_qk * self.num_key_value_heads, False)
        self.k_proj.weight = nn.Parameter(original_attn.k_proj.weight[::reduce_ratio_qk, ...].clone())
        self.v_proj = nn.Linear(self.hidden_size, self.head_dim_vo * self.num_key_value_heads, False)
        self.v_proj.weight = nn.Parameter(original_attn.v_proj.weight[::reduce_ratio_vo, ...].clone())
        self.o_proj = nn.Linear(self.head_dim_vo * self.num_key_value_heads, self.hidden_size, False)
        self.o_proj.weight = nn.Parameter(original_attn.o_proj.weight[..., ::reduce_ratio_vo].clone())
        self._init_rope(original_attn)

    def _init_rope(self, original_attn: LlamaAttention):
        arr = torch.cat([torch.arange(self.head_dim_qk // 4, (self.head_dim_qk * 3) // 4, 2), 
                         torch.arange((self.head_dim_qk * 3) // 4, self.head_dim_qk, 1)])
        self.rotary_emb = LlamaRotaryEmbeddingModified(
            self.head_dim_qk,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
            arr=arr
        ).to(original_attn.o_proj.weight.device)
    
    def set_training(self, new_state: bool=False):
        self.q_proj.requires_grad_(new_state), self.k_proj.requires_grad_(new_state)  
        self.v_proj.requires_grad_(new_state), self.o_proj.requires_grad_(new_state)

    def save_param(self, path_base):
        torch.save(self.q_proj.weight, path_base + "q_proj.pt")
        torch.save(self.k_proj.weight, path_base + "k_proj.pt")
        torch.save(self.v_proj.weight, path_base + "v_proj.pt")
        torch.save(self.o_proj.weight, path_base + "o_proj.pt")
    
    def load_param(self, path_base):
        q_weight = torch.load(path_base + "q_proj.pt")
        k_weight = torch.load(path_base + "k_proj.pt")
        self.q_proj.weight = nn.Parameter(q_weight)
        self.k_proj.weight = nn.Parameter(k_weight)
        v_weight = torch.load(path_base + "v_proj.pt")
        o_weight = torch.load(path_base + "o_proj.pt")
        self.v_proj.weight = nn.Parameter(v_weight)
        self.o_proj.weight = nn.Parameter(o_weight)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim_qk).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim_qk).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim_vo).transpose(1, 2)

        past_key_value = getattr(self, "past_key_value", past_key_value)
        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # ==== Eager implementation of scaled dot-product attention ====
        
        if not self.training:
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim_qk)

            if attention_mask is not None:  # no matter the length, we just slice it
                causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
                attn_weights = attn_weights + causal_mask

            # upcast attention to fp32
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
            attn_output = torch.matmul(attn_weights, value_states)

            if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim_vo):
                raise ValueError(
                    f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim_vo)}, but is"
                    f" {attn_output.size()}"
                )
            
            attn_output = attn_output.transpose(1, 2).contiguous()

            attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim_vo)
            attn_output = self.o_proj(attn_output)

        # ==== We use torch.nn.functional.scaled_dot_product_attention for training here ====

        else:
            causal_mask = attention_mask
            # if attention_mask is not None and cache_position is not None:
            if attention_mask is not None:
                causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

            # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
            # Reference: https://github.com/pytorch/pytorch/issues/112577.
            if query_states.device.type == "cuda" and causal_mask is not None:
                query_states = query_states.contiguous()
                key_states = key_states.contiguous()
                value_states = value_states.contiguous()

            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=causal_mask,
                dropout_p=self.attention_dropout if self.training else 0.0,
            )

            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(bsz, q_len, self.head_dim_vo * self.num_heads)
            attn_output = self.o_proj(attn_output)

            if not output_attentions:
                attn_weights = None

        return attn_output, None, past_key_value


class LlamaDecoderLayerWithMLA(nn.Module):
    def __init__(self, original_layer: LlamaDecoderLayer, 
                 reduce_ratio_qk: int, reduce_ratio_vo: int, 
                 lora_rank: int, lora_alpha: int):
        super().__init__()
        self.hidden_size = original_layer.hidden_size

        self.self_attn = LlamaAttentionMLA(original_layer.self_attn, 
                                           reduce_ratio_qk, 
                                           reduce_ratio_vo
                                           )
        self.mlp = LlamaMLPWithLoRA(original_layer.mlp, lora_rank, lora_alpha)

        self.training = False

        self.input_layernorm = original_layer.input_layernorm
        self.post_attention_layernorm = original_layer.post_attention_layernorm

        self.set_training(False)

    def save_param(self, path_base: str, layer_idx: int):
        layer_idx = str(layer_idx)
        if len(layer_idx) < 2:
            layer_idx = '0' + layer_idx 
        self.self_attn.save_param(f"{path_base}layer{layer_idx}_")
        self.mlp.save_param(f"{path_base}layer{layer_idx}_")

    def load_param(self, path_base: str, layer_idx: int):
        layer_idx = str(layer_idx)
        if len(layer_idx) < 2:
            layer_idx = '0' + layer_idx 
        self.self_attn.load_param(f"{path_base}layer{layer_idx}_")
        self.mlp.load_param(f"{path_base}layer{layer_idx}_")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            is_teaching (`bool`): Whether or not return standard output.
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

    def set_training(self, new_state: bool):
        self.training = new_state
        self.mlp.set_training(new_state)
        self.self_attn.set_training(new_state)

class LlamaForCausalLMWithMLA(LlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, original_model: LlamaForCausalLM, MLAconfig: MLAConfig):
        super().__init__(original_model.config)
    
        self.model = original_model.model
        self.vocab_size = original_model.vocab_size
        self.lm_head = original_model.lm_head
        self.lm_head.requires_grad_(False)
        self.model.embed_tokens.requires_grad_(False)
        
        self.file_path = MLAconfig.path
        for i in range(len(self.model.layers)):
            self.model.layers[i] = LlamaDecoderLayerWithMLA(self.model.layers[i], 
                                                            MLAconfig.reduce_ratio_qk, 
                                                            MLAconfig.reduce_ratio_vo, 
                                                            MLAconfig.r,
                                                            MLAconfig.alpha)
            self.model.layers[i].set_training(True)

    def save_model(self, path: Optional[str]=None):
        if path is None: path = self.file_path
        if path[-1] != '/': path = path + '/'
        for i in range(len(self.model.layers)):
            self.model.layers[i].save_param(path, i)

    def load_model(self, path: Optional[str]=None):
        if path is None: path = self.file_path
        if path[-1] != '/': path = path + '/'
        for i in range(len(self.model.layers)):
            self.model.layers[i].load_param(path, i)

    def set_training(self, new_state: bool):
        if new_state: self.train()
        else: self.eval()
        for i in range(len(self.model.layers)):
            self.model.layers[i].set_training(new_state)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, cache_position=None, **kwargs
    ):
        # With static cache, the `past_key_values` is None
        # TODO joao: standardize interface for the different Cache classes and remove of this if
        has_static_cache = False
        if past_key_values is None:
            past_key_values = getattr(getattr(self.model.layers[0], "self_attn", {}), "past_key_value", None)
            has_static_cache = past_key_values is not None

        past_length = 0
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                past_length = cache_position[0] if cache_position is not None else past_key_values.get_seq_length()
                max_cache_length = (
                    torch.tensor(past_key_values.get_max_length(), device=input_ids.device)
                    if past_key_values.get_max_length() is not None
                    else None
                )
                cache_length = past_length if max_cache_length is None else torch.min(max_cache_length, past_length)
            # TODO joao: remove this `else` after `generate` prioritizes `Cache` objects
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            # The `contiguous()` here is necessary to have a static stride during decoding. torchdynamo otherwise
            # recompiles graphs as the stride of the inputs is a guard. Ref: https://github.com/huggingface/transformers/pull/29114
            # TODO: use `next_tokens` directly instead.
            model_inputs = {"input_ids": input_ids.contiguous()}

        input_length = position_ids.shape[-1] if position_ids is not None else input_ids.shape[-1]
        if cache_position is None:
            cache_position = torch.arange(past_length, past_length + input_length, device=input_ids.device)
        else:
            cache_position = cache_position[-input_length:]

        if has_static_cache:
            past_key_values = None

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past


    @torch.no_grad()
    def generate(self, tokenizer: LlamaTokenizer, messages: list[str], max_new_token=256):
        key_value_cache = DynamicCache()
        device = self.model.embed_tokens.weight.device

        inputs = tokenizer(messages, return_tensors="pt", padding=True, truncation=True)
        # del inputs["token_type_ids"]
        inputs = inputs.to(device)

        new_tokens = []
        while len(new_tokens) < max_new_token:
            outputs = self.forward(**inputs, 
                                   past_key_values=key_value_cache, 
                                   use_cache=True,
                                   )
            key_value_cache = outputs.past_key_values
            next_token = outputs.logits[..., -1:, :].argmax(dim=-1).cpu()

            new_tokens.append(next_token)

            inputs["input_ids"] = next_token.to(device)
            inputs["attention_mask"] = torch.ones_like(next_token, device=device)

        new_tokens = torch.cat(new_tokens, dim=-1)
        return  tokenizer.batch_decode(new_tokens)
    
    @torch.no_grad()
    def generatewithtime(self, tokenizer: LlamaTokenizer, messages: list[str], max_new_token=256):
        key_value_cache = DynamicCache()
        device = self.model.embed_tokens.weight.device

        inputs = tokenizer(messages, return_tensors="pt", padding=True, truncation=True)
        # del inputs["token_type_ids"]
        inputs = inputs.to(device)

        new_tokens = []
        while len(new_tokens) < max_new_token:
            if len(new_tokens) == 0:
                time1 = time.time()
            elif len(new_tokens) == 1:
                time2 = time.time()
            outputs = self.forward(**inputs, 
                                   past_key_values=key_value_cache, 
                                   use_cache=True,
                                   )
            key_value_cache = outputs.past_key_values
            next_token = outputs.logits[..., -1:, :].argmax(dim=-1).cpu()

            new_tokens.append(next_token)

            inputs["input_ids"] = next_token.to(device)
            inputs["attention_mask"] = torch.ones_like(next_token, device=device)

        time3 = time.time()

        new_tokens = torch.cat(new_tokens, dim=-1)
        return  tokenizer.batch_decode(new_tokens), \
                (time2 - time1), \
                (time3 - time2) / max_new_token
    
    @torch.no_grad()
    def generatewithtoken(self, tokenizer: LlamaTokenizer, tokens: torch.Tensor, max_new_token=256):
        key_value_cache = DynamicCache()
        device = self.model.embed_tokens.weight.device

        inputs = tokenizer("Message", return_tensors="pt", padding=True, truncation=True)
        inputs["input_ids"] = tokens
        del inputs["attention_mask"]
        del inputs["token_type_ids"]
        inputs = inputs.to(device)

        new_tokens = []
        while len(new_tokens) < max_new_token:
            if len(new_tokens) == 0:
                time1 = time.time()
            elif len(new_tokens) == 1:
                time2 = time.time()
            outputs = self.forward(**inputs, 
                                   past_key_values=key_value_cache, 
                                   use_cache=True,
                                   )
            key_value_cache = outputs.past_key_values
            next_token = outputs.logits[..., -1:, :].argmax(dim=-1).cpu()

            new_tokens.append(next_token)

            inputs["input_ids"] = next_token.to(device)
            inputs["attention_mask"] = torch.ones_like(next_token, device=device)

        time3 = time.time()

        new_tokens = torch.cat(new_tokens, dim=-1)
        return  tokenizer.batch_decode(new_tokens), \
                (time2 - time1), \
                (time3 - time2) / max_new_token
    
            