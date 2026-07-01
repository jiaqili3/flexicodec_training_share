# Copyright 2025 Xiaomi Corporation.

# input feature is processed by: self.mel_transform = MelSpectrogram(
        #     sample_rate=self.mimo_audio_tokenizer.config.sampling_rate,
        #     n_fft=self.mimo_audio_tokenizer.config.nfft,
        #     hop_length=self.mimo_audio_tokenizer.config.hop_length,
        #     win_length=self.mimo_audio_tokenizer.config.window_size,
        #     f_min=self.mimo_audio_tokenizer.config.fmin,
        #     f_max=self.mimo_audio_tokenizer.config.fmax,
        #     n_mels=self.mimo_audio_tokenizer.config.n_mels,
        #     power=1.0,
        #     center=True,
        # ).to(self.device)
# self.mimo_audio_tokenizer.config:  
# {
#   "max_audio_seconds": 1800,
#   "stride_size": 2,
#   "avg_pooler": 2,
#   "d_model": 1280,
#   "scale_embedding": false,
#   "kernel_size": 3,
#   "activation_function": "gelu",
#   "encoder_layers": 32,
#   "encoder_skip_layer_id": 3,
#   "encoder_attention_heads": 20,
#   "encoder_ffn_dim": 5120,
#   "encoder_causal": false,
#   "encoder_attn_window_size": [
#     -1,
#     -1
#   ],
#   "decoder_layers": 32,
#   "decoder_attention_heads": 20,
#   "decoder_ffn_dim": 5120,
#   "decoder_kernel_size": 3,
#   "decoder_stride_size": 2,
#   "decoder_causal": true,
#   "decoder_attn_window_size": [
#     -1,
#     -1
#   ],
#   "nfft": 960,
#   "vocoder_dim": 256,
#   "vocoder_intermediate_dim": 1024,
#   "vocoder_num_layers": 16,
#   "n_mels": 128,
#   "sampling_rate": 24000,
#   "hop_length": 240,
#   "window_size": 960,
#   "vocoder_padding": "same",
#   "fmin": 0,
#   "fmax": null,
#   "num_quantizers": 20,
#   "codebook_size": [
#     1024,
#     1024,
#     128,
#     128,
#     128,
#     128,
#     128,
#     128,
#     128,
#     128,
#     128,
#     128,
#     128,
#     128,
#     128,
#     128,
#     128,
#     128,
#     128,
#     128
#   ],
#   "threshold_ema_dead_code": 2,
#   "position_embedding_type": "rope",
#   "rope_theta": 10000,
#   "rope_type": "default",
#   "ln_type": "LayerNorm",
#   "vocoder_attention_heads": 16,
#   "vocoder_attn_window_size": [
#     40,
#     10
#   ]
# }
import math

import numpy as np
import torch
import torch.nn as nn
from flash_attn import flash_attn_varlen_func
from torch.nn import functional as F
from transformers.activations import ACT2FN
from transformers.modeling_utils import PreTrainedModel

from .configuration_mimoencoder import MiMoAudioTokenizerConfig
from .modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update, apply_rotary_pos_emb
from dataclasses import dataclass, field
from typing import List

def get_sequence_mask(inputs, inputs_length):
    if inputs.dim() == 3:
        bsz, tgt_len, _ = inputs.size()
    else:
        bsz, tgt_len = inputs_length.shape[0], torch.max(inputs_length)
    sequence_mask = torch.arange(0, tgt_len).to(inputs.device)
    sequence_mask = torch.lt(sequence_mask, inputs_length.reshape(bsz, 1)).view(
        bsz, tgt_len, 1
    )
    unpacking_index = torch.cumsum(sequence_mask.to(torch.int64).view(-1), dim=0) - 1
    return sequence_mask, unpacking_index


def unpack_hidden_states(
    hidden_states, lengths, sequence_mask=None, unpacking_index=None
):
    bsz = lengths.shape[0]
    if sequence_mask is None or unpacking_index is None:
        sequence_mask, unpacking_index = get_sequence_mask(hidden_states, lengths)
    hidden_states = torch.index_select(hidden_states, 0, unpacking_index).view(
        bsz, torch.max(lengths), hidden_states.shape[-1]
    )
    hidden_states = torch.where(sequence_mask, hidden_states, 0)
    return hidden_states


def get_position_ids(lengths):
    total_len = lengths.sum()
    offset = torch.cat([torch.zeros(1).to(lengths), lengths[:-1].cumsum(dim=0)])
    offset = torch.repeat_interleave(offset, lengths)
    position_ids = torch.arange(0, total_len).to(offset) - offset
    return position_ids


class RotaryEmbedding(nn.Module):
    def __init__(self, base, dim, max_seq_len, rope_type="default", device=None):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.rope_type = rope_type

        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(
            device=device, base=base, dim=dim
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[:, None].float().expand(-1, 1).to(x.device)
        position_ids_expanded = position_ids[None, :].float()

        device_type = (
            x.device.type
            if isinstance(x.device.type, str) and x.device.type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
            ).transpose(0, 1)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states


LAYER_NORM = {"LayerNorm": nn.LayerNorm, "RMSNorm": RMSNorm}


class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads, window_size=(-1, -1), causal=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.window_size = window_size

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

        self.causal = causal

    def forward(
        self,
        hidden_states: torch.Tensor,
        seq_len: torch.Tensor,
        rope_position_embeddings=None,
    ):
        bsz, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).view(
            bsz, self.num_heads, self.head_dim
        )
        key_states = self.k_proj(hidden_states).view(bsz, self.num_heads, self.head_dim)
        value_states = self.v_proj(hidden_states).view(
            bsz, self.num_heads, self.head_dim
        )

        if rope_position_embeddings is not None:
            cos, sin = rope_position_embeddings
            query_states = apply_rotary_pos_emb(query_states, cos, sin)
            key_states = apply_rotary_pos_emb(key_states, cos, sin)

        cu_len = F.pad(torch.cumsum(seq_len, dim=0), (1, 0), "constant", 0).to(
            torch.int32
        )
        max_seqlen = torch.max(seq_len).to(torch.int32).detach()
        attn_output = flash_attn_varlen_func(
            query_states,
            key_states,
            value_states,
            cu_len,
            cu_len,
            max_seqlen,
            max_seqlen,
            causal=self.causal,
            window_size=self.window_size,
        )
        attn_output = attn_output.reshape(bsz, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        return attn_output


class TransformerLayer(nn.Module):
    def __init__(
        self,
        act,
        d_model,
        encoder_attention_heads,
        encoder_ffn_dim,
        causal,
        ln_type="LayerNorm",
        attn_window_size=(-1, -1),
    ):
        super().__init__()
        self.embed_dim = d_model
        self.self_attn = Attention(
            self.embed_dim, encoder_attention_heads, attn_window_size, causal
        )

        self.self_attn_layer_norm = LAYER_NORM[ln_type](self.embed_dim)

        self.activation_fn = act
        self.fc1 = nn.Linear(self.embed_dim, encoder_ffn_dim)
        self.fc2 = nn.Linear(encoder_ffn_dim, self.embed_dim)

        self.final_layer_norm = LAYER_NORM[ln_type](self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        seq_len: torch.Tensor,
        rope_position_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states, seq_len, rope_position_embeddings=rope_position_embeddings
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states

        if (
            hidden_states.dtype == torch.float16
            or hidden_states.dtype == torch.bfloat16
        ) and (torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(
                hidden_states, min=-clamp_value, max=clamp_value
            )
        return hidden_states

class AudioEncoder(nn.Module):
    def __init__(self, config: MiMoAudioTokenizerConfig):
        super().__init__()
        config._attn_implementation = "flash_attention_2"
        self.config = config
        self.max_source_positions = (
            config.max_audio_seconds * config.sampling_rate // config.hop_length
        ) // config.stride_size
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        self.skip_layer_idx = config.encoder_skip_layer_id
        self.conv1 = nn.Conv1d(
            config.n_mels, config.d_model, kernel_size=config.kernel_size, padding=1
        )
        self.conv2 = nn.Conv1d(
            config.d_model,
            config.d_model,
            kernel_size=config.kernel_size,
            stride=config.stride_size,
            padding=1,
        )

        self.position_embedding = RotaryEmbedding(
            config.rope_theta,
            config.d_model // config.encoder_attention_heads,
            self.max_source_positions,
            config.rope_type,
        )

        self.layers = nn.ModuleList(
            [
                TransformerLayer(
                    ACT2FN[config.activation_function],
                    config.d_model,
                    config.encoder_attention_heads,
                    config.encoder_ffn_dim,
                    causal=self.config.encoder_causal,
                    ln_type=self.config.ln_type,
                    attn_window_size=self.config.encoder_attn_window_size,
                )
                for _ in range(config.encoder_layers)
            ]
        )

        self.layer_norm = LAYER_NORM[config.ln_type](config.d_model)

        # Optional mid-encoder downsampling (2x) to reach 12.5Hz tokens when starting from 25Hz
        self.mid_downsample = None
        self.mid_downsample_norm = None
        self.mid_downsample_layer_idx = None
        if getattr(self.config, "mid_downsample", False):
            self.mid_downsample = nn.Sequential(
                nn.Conv1d(
                    config.d_model,
                    config.d_model,
                    kernel_size=config.kernel_size,
                    stride=getattr(config, "mid_downsample_stride", 2),
                    padding=1,
                    bias=False,
                ),
                nn.GELU(),
            )
            self.mid_downsample_norm = LAYER_NORM[config.ln_type](config.d_model)
            if getattr(config, "mid_downsample_layer_idx", None) is None:
                self.mid_downsample_layer_idx = max(1, config.encoder_layers // 2)
            else:
                self.mid_downsample_layer_idx = int(config.mid_downsample_layer_idx)
            self.mid_downsample_stride = getattr(config, "mid_downsample_stride", 2)
        else:
            self.mid_downsample_stride = 1

        if self.config.avg_pooler != 1:
            self.down_sample_layer = nn.Sequential(
                nn.Conv1d(
                    config.d_model,
                    config.d_model,
                    config.avg_pooler,
                    config.avg_pooler,
                    bias=False,
                ),
                nn.GELU(),
            )
            self.down_sample_norm = LAYER_NORM[config.ln_type](config.d_model)
        else:
            self.down_sample_layer = None

        
        self.quantizer = None

    def get_features(self, input_features, output_length):
        input_features = input_features.to(self.conv1.weight)
        inputs_embeds = nn.functional.gelu(self.conv1(input_features))
        inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds))
        inputs_embeds = inputs_embeds.permute(0, 2, 1) # 50hz
        bsz, tgt_len, _ = inputs_embeds.size()

        hidden_states = inputs_embeds

        position_ids = (
            get_position_ids(output_length).long().to(input_features.device)
        )
        rope_position_embeddings = self.position_embedding(
            input_features, position_ids
        )

        attention_mask, unpacking_index = get_sequence_mask(
            hidden_states, output_length
        )

        hidden_states = torch.masked_select(hidden_states, attention_mask).view(
            torch.sum(output_length), self.config.d_model
        )

        skip_connect_hidden_states = 0.0
        for idx, encoder_layer in enumerate(self.layers):
            breakpoint()
            hidden_states = encoder_layer(
                hidden_states,
                output_length,
                rope_position_embeddings=rope_position_embeddings,
            )
            # Insert optional mid-encoder downsampling block after the specified layer
            if (
                self.mid_downsample is not None
                and self.mid_downsample_layer_idx is not None
                and idx == self.mid_downsample_layer_idx - 1
            ):
                # Repack to (B, T, D)
                hidden_states_bt = torch.index_select(hidden_states, 0, unpacking_index).view(
                    bsz, tgt_len, self.config.d_model
                )
                # Pad to be divisible by stride if needed
                if hidden_states_bt.size(1) % self.mid_downsample_stride:
                    pad_len = self.mid_downsample_stride - hidden_states_bt.size(1) % self.mid_downsample_stride
                    hidden_states_bt = torch.nn.functional.pad(
                        hidden_states_bt, (0, 0, 0, pad_len), mode="constant", value=0.0
                    )
                    tgt_len = tgt_len + pad_len
                # Downsample with stride-2 conv
                hidden_states_bt = self.mid_downsample(hidden_states_bt.transpose(1, 2)).transpose(1, 2)
                # Update sequence lengths (ceil division)
                output_length = (
                    output_length // self.mid_downsample_stride
                    + (output_length % self.mid_downsample_stride != 0).int()
                )
                # Update tgt_len to new temporal length
                tgt_len = hidden_states_bt.size(1)
                # Recompute masks and repack to (sum(L), D)
                attention_mask, unpacking_index = get_sequence_mask(hidden_states_bt, output_length)
                hidden_states = torch.masked_select(hidden_states_bt, attention_mask).view(
                    torch.sum(output_length), self.config.d_model
                )
                hidden_states = self.mid_downsample_norm(hidden_states)
                # Recompute RoPE embeddings for the new positions
                position_ids = get_position_ids(output_length).long().to(input_features.device)
                rope_position_embeddings = self.position_embedding(input_features, position_ids)

            if (self.skip_layer_idx is not None) and idx == self.skip_layer_idx - 1:
                skip_connect_hidden_states = hidden_states.clone()

        hidden_states += skip_connect_hidden_states
        hidden_states = self.layer_norm(hidden_states)

        if self.down_sample_layer is not None:
            hidden_states = torch.index_select(hidden_states, 0, unpacking_index).view(
                bsz, tgt_len, self.config.d_model
            )
            if hidden_states.size(1) % self.config.avg_pooler:
                pad_len = (
                    self.config.avg_pooler
                    - hidden_states.size(1) % self.config.avg_pooler
                )
                hidden_states = torch.nn.functional.pad(
                    hidden_states, (0, 0, 0, pad_len), mode="constant", value=0.0
                )
                tgt_len += pad_len
            tgt_len = tgt_len // self.config.avg_pooler
            hidden_states = self.down_sample_layer(hidden_states.transpose(1, 2))
            output_length = (
                output_length // self.config.avg_pooler
                + (output_length % self.config.avg_pooler != 0).int()
            )
            hidden_states = hidden_states.transpose(1, 2)
            attention_mask, unpacking_index = get_sequence_mask(
                hidden_states, output_length
            )
            hidden_states = torch.masked_select(hidden_states, attention_mask).view(
                torch.sum(output_length), self.config.d_model
            )
            hidden_states = self.down_sample_norm(hidden_states)

        return (
            hidden_states,
            output_length,
            attention_mask,
            unpacking_index,
            tgt_len,
            bsz,
        )

    def get_output_length(self, mel_len):
        tgt_len = mel_len + 3 - self.config.kernel_size
        return (tgt_len + 2 - self.config.kernel_size) // self.config.stride_size + 1

    @torch.no_grad()
    def encode(
        self,
        input_features,
        input_lens=None,
        output_length=None,
        return_codes_only=False,
        n_q=None,
        use_quantizer=True,
    ):
        if output_length is None:
            output_length = self.get_output_length(input_lens)
        input_features = unpack_hidden_states(input_features, input_lens)
        hidden_states, output_length, attention_mask, unpacking_index, tgt_len, bsz = (
            self.get_features(
                input_features=input_features.transpose(1, 2),
                output_length=output_length,
            )
        )

        dtype = hidden_states.dtype

        if use_quantizer and self.quantizer is not None:
            self.quantizer.float()

            codes = self.quantizer.encode(hidden_states.float(), n_q=n_q)
            if return_codes_only:
                return codes, output_length
            hidden_states = self.quantizer.decode(codes)
            hidden_states = hidden_states.to(dtype)
        else:
            codes = None

        hidden_states_packed = hidden_states.clone()

        # unpacking
        hidden_states = torch.index_select(hidden_states, 0, unpacking_index).view(
            bsz, tgt_len, self.config.d_model
        )
        hidden_states = torch.where(attention_mask, hidden_states, 0)
        return hidden_states, hidden_states_packed, output_length, codes