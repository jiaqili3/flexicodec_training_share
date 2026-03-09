from random import random
from einops import rearrange
from easydict import EasyDict as edict

import torch
import torch.nn as nn
import random
class DualCodec(nn.Module):
    def __init__(
        self,
        encoder_model,
        decoder_model,
        quantizer_model,
        quantization_method='rvq',
        transform_model=None,
        dropout=True,
        n_quantizers=8,
    ):
        super().__init__()
        self.encoder_model = encoder_model
        self.decoder_model = decoder_model
        self.quantizer = quantizer_model
        self.transform_model = transform_model if transform_model is not None else nn.Identity()
        self.quantization_method = quantization_method
        self.dropout = dropout
        self.n_quantizers = n_quantizers

        self.downsample_rate = self.encoder_model.downsample_rate
        self.semantic_downsample_factor = 1 # for compatibility with dualCodec 
        self.override_dac_encoder = False # for compatibility with dualCodec

    @torch.no_grad()
    def encode(self, audio_data, num_quantizers=None, sample_rate=24000, semantic_repr=None):
        """
        both codes: [b,q,t]
        """
        assert not self.training
        z = self.encoder_model(audio_data)
        
        if self.quantization_method == 'rvq':
            quantized, codes, commitment_loss, quantized_list = self.quantizer(z, n_q=num_quantizers)
            codebook_loss = 0
            codes = rearrange(codes, 'q b t -> b q t', q=num_quantizers)
        elif self.quantization_method == 'proj':
            # codes: [b,q,t]
            quantized, codes, _, commitment_loss, codebook_loss, first_layer_quantized = self.quantizer(z, num_quantizers)
        elif self.quantization_method == 'fsq':
            quantized, ret_dict = self.quantizer(z)
            codes = ret_dict['quantizer_indices'] # b q t
        else:
            raise ValueError(f"Unknown quantization method: {self.quantization_method}")
        semantic_codes = codes[:, :1]
        if codes.shape[1] == 1:
            acoustic_codes = None
        else:
            acoustic_codes = codes[:, 1:]
        return semantic_codes, acoustic_codes

    def decode_from_codes(self, semantic_codes, acoustic_codes):
        """
        Args:
            codes: Quantized codes from the encoder. (batch, n_q, timesteps)
        Returns:
            Decoded audio
        """
        if acoustic_codes is not None:
            codes = torch.cat([semantic_codes, acoustic_codes], dim=1)
        else:
            codes = semantic_codes
        # Reconstruct quantized representation from codes
        if self.quantization_method == 'rvq':
            codes = rearrange(codes, 'b q t -> q b t')
            quantized = self.quantizer.decode(codes)
        elif self.quantization_method == 'proj':
            quantized = self.quantizer.from_codes(codes)[0]
        elif self.quantization_method == 'fsq':
            quantized = self.quantizer.decode_tokens(codes)
        else:
            raise ValueError(f"Unknown quantization method: {self.quantization_method}")
        
        # Decode through decoder model
        audio = self.decoder_model(quantized)
        
        return audio

    def forward(self, 
            audio_data: torch.Tensor,
            sample_rate: int = 24000,
            n_quantizers: int = None,
            semantic_repr=None,
            bypass_quantize_rate=0.125,
            possibly_no_quantizer=False,
        ):
        z = self.encoder_model(audio_data) # (b,d,t)

        if n_quantizers is None and self.dropout:
            n_quantizers = random.randint(1, self.n_quantizers)

        if self.quantization_method == 'rvq':
            quantized, codes, commitment_loss, quantized_list = self.quantizer(z, n_q=n_quantizers, layers=[0])
            codebook_loss = 0
            feature = quantized_list[0] # (b,c,t)
        elif self.quantization_method == 'proj':
            quantized, codes, _, commitment_loss, codebook_loss, first_layer_quantized = self.quantizer(z, n_quantizers)
            feature = first_layer_quantized # (b,c,t)
        elif self.quantization_method == 'fsq':
            quantized, ret_dict = self.quantizer.encode(z, return_info=True)
            codes = ret_dict['quantizer_indices']
            feature = None
            commitment_loss = 0.0
            codebook_loss = 0.0
        else:
            raise ValueError(f"Unknown quantization method: {self.quantization_method}")

        if feature is not None:
            first_layer_quantized = self.transform_model(feature)
        else:
            first_layer_quantized = None
        x = self.decoder_model(quantized)

        semantic_edict = None
        acoustic_edict = edict({
            "x": x,
            "z": z,
            "codes": codes,
            # "latents": latents,
            "penalty": commitment_loss,
            "vq/codebook_loss": codebook_loss,
            "metrics": {},
            "first_layer_quantized": first_layer_quantized,
        })
        return acoustic_edict, semantic_edict
