from .cnn import ConvNeXtBlock
from .dac_model import DAC
import torch.nn as nn
import math
from typing import List
from typing import Union

import numpy as np
import torch
from audiotools import AudioSignal
from audiotools.ml import BaseModel
from torch import nn

# from .base import CodecMixin
from dac.nn.layers import Snake1d
from dac.nn.layers import WNConv1d
from dac.nn.layers import WNConvTranspose1d
from .dac_quantize import ResidualVectorQuantize
from easydict import EasyDict as edict
import torch.nn.functional as F
import random
from einops import rearrange
class UpsampleConvNeXtBlock(nn.Module):
    def __init__(self, in_dim, out_dim, upsample_factor=2, intermediate_dim=2048, is_causal=False, n_layers=4):
        super().__init__()
        
        # Upsampling layer (transposed convolution-based)
        self.upsample = WNConvTranspose1d(in_dim, out_dim, kernel_size=upsample_factor, stride=upsample_factor)
        
        # ConvNeXt block sequence
        self.convnext_blocks = nn.Sequential(
            *[ConvNeXtBlock(dim=out_dim, intermediate_dim=intermediate_dim, is_causal=is_causal) for _ in range(n_layers)]
        )

    def forward(self, x):
        # Upsample then apply ConvNeXt blocks
        x = self.upsample(x)
        return self.convnext_blocks(x)
class DownsampleConvNeXtBlock(nn.Module):
    def __init__(self, in_dim, out_dim, downsample_factor=2, intermediate_dim=2048, is_causal=False, n_layers=4):
        super().__init__()
        # Downsampling layer (stride-based)
        self.downsample = WNConv1d(in_dim, out_dim, kernel_size=downsample_factor, stride=downsample_factor)
        
        # ConvNeXt block sequence
        self.convnext_blocks = nn.Sequential(
            *[ConvNeXtBlock(dim=out_dim, intermediate_dim=intermediate_dim, is_causal=is_causal) for _ in range(n_layers)]
        )

    def forward(self, x):
        # Downsample then apply ConvNeXt blocks
        x = self.downsample(x)
        return self.convnext_blocks(x)

import torch
import torch.nn as nn

class DualCodec(nn.Module):
    def __init__(
        self,
        encoder_dim: int = 64,
        encoder_rates: List[int] = [2, 4, 8, 8],
        latent_dim: int = None,
        decoder_dim: int = 1536,
        decoder_rates: List[int] = [8, 8, 4, 2],
        n_codebooks: int = 9,
        codebook_size: int = 1024,
        semantic_codebook_size: int = 16384,
        codebook_dim: Union[int, list] = 8,
        semantic_codebook_dim=8,
        quantizer_dropout: bool = False,
        sample_rate: int = 24000,
        distill_projection_out_dim=1024,
        convnext_dim=768,
        convnext_layers=4,
        decode_semantic_for_codec=True,
        is_causal=False,
        semantic_downsample_factor=2,
        use_concat_downsampling=False,
        use_conv_downsampling=False,
        override_dac_encoder=None, # torch.nn.Module
        override_vocos_decoder=None,
        semantic_encoder=None,
        semantic_decoder=None,
        ssl_dim=1024,
    ):
        super().__init__()
        self.semantic_downsample_factor = semantic_downsample_factor
        self.concat_downsample_factor = 1
        self.use_concat_downsampling = use_concat_downsampling
        self.use_conv_downsampling = use_conv_downsampling

        self.semantic_encoder = semantic_encoder
        self.semantic_decoder = semantic_decoder

        self.dac = DAC(
            encoder_dim,
            encoder_rates,
            latent_dim,
            decoder_dim,
            decoder_rates,
            n_codebooks,
            codebook_size,
            codebook_dim,
            quantizer_dropout,
            sample_rate,
            distill_projection_out_dim,
            distill=False,
        )
        if override_dac_encoder is not None:
            self.dac.encoder = override_dac_encoder
            self.override_dac_encoder = True
        else:
            self.override_dac_encoder = False
        if override_vocos_decoder is not None:
            self.dac.decoder = override_vocos_decoder
            self.override_vocos_decoder = True
        else:
            self.override_vocos_decoder = False

        self.decode_semantic_for_codec = decode_semantic_for_codec
        self.encoder_rates = encoder_rates
        self.ssl_dim = ssl_dim

        self.dac_bn_dim = self.dac.latent_dim
        # get the dim after downsampling
        if self.use_concat_downsampling:
            assert not self.use_conv_downsampling
            print('using concat downsampling')
            # reset semantic_downsample_factor so that it will not perform avg pooling
            self.concat_downsample_factor = semantic_downsample_factor
            self.semantic_downsample_factor = 1

        self.convnext_encoder = nn.Sequential(
            WNConv1d(
                self.ssl_dim, convnext_dim, kernel_size=1,
            ),
            *[ConvNeXtBlock(
                dim=convnext_dim,
                intermediate_dim=2048,
                is_causal=is_causal
            ) for _ in range(convnext_layers)],  # Unpack the list directly into nn.Sequential
        )
        if semantic_encoder is not None:
            self.convnext_encoder = semantic_encoder
        self.semantic_vq = ResidualVectorQuantize(
            convnext_dim, n_codebooks=1, codebook_size=semantic_codebook_size,
            codebook_dim=semantic_codebook_dim,
        )
        self.convnext_decoder = nn.Sequential(
            *[ConvNeXtBlock(
                dim=convnext_dim,
                intermediate_dim=2048,
                is_causal=is_causal,
            ) for _ in range(convnext_layers)],  # Unpack the list directly into nn.Sequential
            WNConv1d(
                convnext_dim, self.dac_bn_dim, kernel_size=1,
            ),
        )
        if semantic_decoder is not None:
            self.convnext_decoder = semantic_decoder
            self.semantic_vq = ResidualVectorQuantize(
                1024, n_codebooks=1, codebook_size=semantic_codebook_size,
                codebook_dim=semantic_codebook_dim,
            )
        # if not self.decode_semantic_for_codec:
        #     assert convnext_dim == 1024

    def semantic_quantize(self, semantic_repr):
        if self.override_dac_encoder:
            pad_amount = audio_data.shape[1] % self.concat_downsample_factor
            audio_data = audio_data[:, :audio_data.shape[1] - pad_amount]
            semantic_repr = semantic_repr[..., :semantic_repr.shape[-1] - pad_amount]
            # audio_data = torch.nn.functional.pad(audio_data, (0, pad_amount))
            # semantic_repr = torch.nn.functional.pad(semantic_repr, (0, pad_amount))
            semantic_repr = rearrange(semantic_repr, 'b c (t k) -> b (k c) t', k=self.concat_downsample_factor)
            # encoded_feature = self.dac.encoder(audio_data)
        elif self.use_concat_downsampling:
            # semantic_repr = semantic_repr[..., 1:]
            # left pad the same as first frame
            semantic_repr = torch.nn.functional.pad(semantic_repr, (1,0), mode='reflect')
            semantic_repr = rearrange(semantic_repr, 'b c (t k) -> b (k c) t', k=self.concat_downsample_factor)
        semantic = self.convnext_encoder(semantic_repr)
            
        semantic, codes, latents, commitment_loss, codebook_loss, first_layer_quantized = self.semantic_vq(semantic)
        codes = rearrange(codes, 'b 1 t -> b t')
        return codes

    def encode(self, audio_data, num_quantizers=None, sample_rate=24000, semantic_repr=None, return_semantic_feat=False):
        assert not self.training
        if self.override_dac_encoder:
            # audio_data: mel
            pad_amount = audio_data.shape[1] % self.concat_downsample_factor
            audio_data = audio_data[:, :audio_data.shape[1] - pad_amount]
            semantic_repr = semantic_repr[..., :semantic_repr.shape[-1] - pad_amount]
            # audio_data = torch.nn.functional.pad(audio_data, (0, pad_amount))
            # semantic_repr = torch.nn.functional.pad(semantic_repr, (0, pad_amount))
            semantic_repr = rearrange(semantic_repr, 'b c (t k) -> b (k c) t', k=self.concat_downsample_factor)
            encoded_feature = self.dac.encoder(audio_data)
        elif self.use_concat_downsampling:
            # semantic_repr = semantic_repr[..., 1:]
            # left pad the same as first frame
            semantic_repr = torch.nn.functional.pad(semantic_repr, (1,0), mode='reflect')
            if not self.training:
                # pad to multiple of downsample factor
                pad_amount = semantic_repr.shape[-1] % self.concat_downsample_factor
                # semantic_repr = torch.nn.functional.pad(semantic_repr, (pad_amount,0), mode='reflect')
                semantic_repr = semantic_repr[..., :semantic_repr.shape[-1]-pad_amount]

            semantic_repr = rearrange(semantic_repr, 'b c (t k) -> b (k c) t', k=self.concat_downsample_factor)
        semantic = self.convnext_encoder(semantic_repr)
            
        semantic, codes, latents, commitment_loss, codebook_loss, first_layer_quantized = self.semantic_vq(semantic)
        if self.decode_semantic_for_codec:
            semantic = self.convnext_decoder(semantic)

        semantic_codes = codes

        if num_quantizers == 1:
            return semantic_codes, None

        if num_quantizers is not None:
            num_quantizers -= 1

        if self.override_dac_encoder:
            acoustic_codes = self.dac.encode(encoded_feature=encoded_feature, sample_rate=sample_rate, n_quantizers=num_quantizers, subtracted_latent=semantic)[1]
        else:
            acoustic_codes = self.dac.encode(audio_data, sample_rate=sample_rate, n_quantizers=num_quantizers, subtracted_latent=semantic)[1]
        
        if return_semantic_feat:
            return semantic_codes, acoustic_codes, semantic
        else:
            return semantic_codes, acoustic_codes # [B, n_q, T]

    def decode_from_codes(self, semantic_codes, acoustic_codes):
        """both [B, n_q, T]"""
        semantic = self.semantic_vq.from_codes(semantic_codes)[0]
        if self.decode_semantic_for_codec:
            semantic = self.convnext_decoder(semantic)

        audio = self.dac.decode_from_codes(acoustic_codes, semantic_latent=semantic)
        return audio

    def forward(self, 
            audio_data: torch.Tensor,
            sample_rate: int = 24000,
            n_quantizers: int = None,
            semantic_repr=None,
            bypass_quantize_rate=0.125,
            possibly_no_quantizer=False,
        ):
        """
        semantic_repr: [B, C, T]
        """
        if self.override_dac_encoder:
            pad_amount = audio_data.shape[1] % self.concat_downsample_factor
            audio_data = audio_data[:, :audio_data.shape[1] - pad_amount]
            semantic_repr = semantic_repr[..., :semantic_repr.shape[-1] - pad_amount]
            # audio_data = torch.nn.functional.pad(audio_data, (0, pad_amount))
            # semantic_repr = torch.nn.functional.pad(semantic_repr, (0, pad_amount))
            semantic_repr = rearrange(semantic_repr, 'b c (t k) -> b (k c) t', k=self.concat_downsample_factor)
            encoded_feature = self.dac.encoder(audio_data)
        elif self.use_concat_downsampling:
            raise NotImplementedError
            pad_amount = semantic_repr.shape[-1] % self.concat_downsample_factor
            semantic_repr = semantic_repr[..., :semantic_repr.shape[1] - pad_amount]
            semantic_repr = rearrange(semantic_repr, 'b c (t k) -> b (k c) t', k=self.concat_downsample_factor)
        
        semantic_repr_ret = semantic_repr.clone().detach()
        semantic = self.convnext_encoder(semantic_repr)
            
        semantic, codes, latents, commitment_loss, codebook_loss, first_layer_quantized = self.semantic_vq(semantic)
        if self.decode_semantic_for_codec:
            semantic = self.convnext_decoder(semantic)

        bypass_quantize = random.random() < bypass_quantize_rate
        if not self.training:
            bypass_quantize = False
        if n_quantizers == 1:
            bypass_quantize = True
        if n_quantizers is not None:
            n_quantizers = n_quantizers - 1
        # cut_from_front = self.use_concat_downsampling # we cut from front to align the two features if we're using concat downsampling method
        if self.override_dac_encoder:
            acoustic_edict = self.dac(None, sample_rate, n_quantizers, subtracted_latent=semantic, bypass_quantize=bypass_quantize, possibly_no_quantizer=possibly_no_quantizer, \
                encoded_feature=encoded_feature)
        else:
            acoustic_edict = self.dac(audio_data, sample_rate, n_quantizers, subtracted_latent=semantic, bypass_quantize=bypass_quantize, possibly_no_quantizer=possibly_no_quantizer, \
                cut_from_front=False)
        if not self.decode_semantic_for_codec:
            semantic = self.convnext_decoder(semantic)
            semantic_repr_ret = semantic_repr_ret[..., :semantic.shape[-1]]

        semantic_edict = edict({
            "x": semantic,
            "codes": codes,
            "latents": latents,
            "penalty": commitment_loss,
            "vq/codebook_loss": codebook_loss,
            "metrics": {},
            "bypassed_quantize": bypass_quantize,
            "semantic_repr": semantic_repr_ret,
        })
        return acoustic_edict, semantic_edict

if __name__ == '__main__':
    model = DualCodec()
    model(torch.rand((3, 1, 24000)), 24000)