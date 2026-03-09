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
from .cnn import ConvNeXtBlock

def init_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        nn.init.constant_(m.bias, 0)
def pad_to_length(x, length, pad_value=0):
    # Get the current size along the last dimension
    current_length = x.shape[-1]

    # If the length is greater than current_length, we need to pad
    if length > current_length:
        pad_amount = length - current_length
        # Pad on the last dimension (right side), keeping all other dimensions the same
        x_padded = F.pad(x, (0, pad_amount), value=pad_value)
    else:
        # If no padding is required, simply slice the tensor
        x_padded = x[..., :length]

    return x_padded

# -----------------------------------------------------------------------------
# Helper functions for Residual Autoencoding (DC-AE)
# -----------------------------------------------------------------------------

def _space_to_channel_avg(x: torch.Tensor, stride: int, out_channels: int):
    """Space-to-channel operation followed by channel averaging.

    This implements the non-parametric shortcut for down-sampling blocks as
    described in the Deep Compression Autoencoder (DC-AE) paper.  Works on
    1-D feature maps with shape (B, C, T).
    """
    if stride == 1:
        return x

    B, C, T = x.shape
    assert T % stride == 0, "T must be divisible by stride for space-to-channel."

    # Move `stride` positions from time dimension into channels
    x = x.view(B, C, T // stride, stride)           # B C T/s s
    x = x.permute(0, 3, 1, 2).contiguous()          # B s C T/s
    x = x.view(B, C * stride, T // stride)          # B C*s T/s

    if x.shape[1] != out_channels:
        # ------------------------------------------------------------------
        # Relaxed handling when channels are not an exact multiple
        # ------------------------------------------------------------------
        if x.shape[1] % out_channels == 0:
            # Perfectly divisible – original behaviour
            group_size = x.shape[1] // out_channels
            x = x.view(B, out_channels, group_size, T // stride).mean(dim=2)
        else:
            # Not divisible – fallback to cropping or repeating channels to
            # match the desired `out_channels` before averaging.
            if x.shape[1] > out_channels:
                # Crop the extra channels (keeps deterministic, non-parametric)
                group_size = x.shape[1] // out_channels  # floor
                required_channels = out_channels * group_size
                x = x[:, :required_channels, :]
                x = x.view(B, out_channels, group_size, T // stride).mean(dim=2)
            else:
                # When we have fewer channels than required, repeat channels
                repeat_factor = math.ceil(out_channels / x.shape[1])
                x = x.repeat_interleave(repeat_factor, dim=1)[:, :out_channels, :]
    return x


def _channel_to_space_dup(x: torch.Tensor, stride: int, out_channels: int):
    """Channel-to-space operation followed by channel duplication.

    Implements the non-parametric shortcut for up-sampling blocks in DC-AE.
    """
    if stride == 1:
        return x

    B, C, T = x.shape

    # Ensure the channel dimension is divisible by `stride`.
    if C % stride != 0:
        # Pad channels by repeating the last channel so that it becomes divisible
        pad_channels = stride - (C % stride)
        x = torch.cat([x, x[:, -1:, :].repeat(1, pad_channels, 1)], dim=1)
        C = x.shape[1]

    x = x.view(B, C // stride, stride, T)           # B C/stride s T
    x = x.permute(0, 1, 3, 2).contiguous()          # B C/stride T s
    x = x.view(B, C // stride, T * stride)          # B C/stride T*s

    if x.shape[1] != out_channels:
        if out_channels % x.shape[1] == 0:
            dup_factor = out_channels // x.shape[1]
            x = x.repeat_interleave(dup_factor, dim=1)
        else:
            # If exact duplication isn't possible, crop or pad to match
            if x.shape[1] > out_channels:
                x = x[:, :out_channels, :]
            else:
                dup_factor = math.ceil(out_channels / x.shape[1])
                x = x.repeat_interleave(dup_factor, dim=1)[:, :out_channels, :]
    return x

class ResidualUnit(nn.Module):
    def __init__(self, dim: int = 16, dilation: int = 1):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.block = nn.Sequential(
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=7, dilation=dilation, padding=pad),
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x):
        y = self.block(x)
        pad = (x.shape[-1] - y.shape[-1]) // 2
        if pad > 0:
            x = x[..., pad:-pad]
        return x + y


class EncoderBlock(nn.Module):
    """Down-sampling block with optional residual autoencoding shortcut."""

    def __init__(self, dim: int = 16, stride: int = 1, use_shortcut: bool = False):
        super().__init__()
        self.stride = stride
        self.use_shortcut = use_shortcut and stride > 1  # shortcut only when down-sampling

        self.block = nn.Sequential(
            ResidualUnit(dim // 2, dilation=1),
            ResidualUnit(dim // 2, dilation=3),
            ResidualUnit(dim // 2, dilation=9),
            Snake1d(dim // 2),
            WNConv1d(
                dim // 2,
                dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
        )

    def forward(self, x):
        y = self.block(x)
        if self.use_shortcut:
            shortcut = _space_to_channel_avg(x, self.stride, y.shape[1])
            y = y + shortcut
        return y


class Encoder(nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        strides: list = [2, 4, 8, 8],
        d_latent: int = 64,
        residual_autoencode: bool = False,
    ):
        super().__init__()
        # Create first convolution
        self.block = [WNConv1d(1, d_model, kernel_size=7, padding=3)]

        self.downsample_rate = np.prod(strides)

        # Create EncoderBlocks that double channels as they downsample by `stride`
        for stride in strides:
            d_model *= 2
            self.block += [EncoderBlock(d_model, stride=stride, use_shortcut=residual_autoencode)]

        # Create last convolution
        self.block += [
            Snake1d(d_model),
            WNConv1d(d_model, d_latent, kernel_size=3, padding=1),
        ]

        # Wrap black into nn.Sequential
        self.block = nn.Sequential(*self.block)
        self.enc_dim = d_model

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    """Up-sampling block with optional residual autoencoding shortcut."""

    def __init__(self, input_dim: int = 16, output_dim: int = 8, stride: int = 1, use_shortcut: bool = False):
        super().__init__()
        self.stride = stride
        self.use_shortcut = use_shortcut and stride > 1

        self.block = nn.Sequential(
            Snake1d(input_dim),
            WNConvTranspose1d(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
            ResidualUnit(output_dim, dilation=1),
            ResidualUnit(output_dim, dilation=3),
            ResidualUnit(output_dim, dilation=9),
        )

    def forward(self, x):
        y = self.block(x)
        if self.use_shortcut:
            shortcut = _channel_to_space_dup(x, self.stride, y.shape[1])
            y = y + shortcut[..., :y.shape[-1]]
        return y


class Decoder(nn.Module):
    def __init__(
        self,
        input_channel,
        channels,
        rates,
        d_out: int = 1,
        residual_autoencode: bool = False,
    ):
        super().__init__()

        # Add first conv layer
        layers = [WNConv1d(input_channel, channels, kernel_size=7, padding=3)]

        # Add upsampling + MRF blocks
        for i, stride in enumerate(rates):
            input_dim = channels // 2**i
            output_dim = channels // 2 ** (i + 1)
            layers += [DecoderBlock(input_dim, output_dim, stride, use_shortcut=residual_autoencode)]

        # Add final conv layer
        layers += [
            Snake1d(output_dim),
            WNConv1d(output_dim, d_out, kernel_size=7, padding=3),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class DAC(BaseModel):
    def __init__(
        self,
        encoder_dim: int = 64,
        encoder_rates: List[int] = [2, 4, 8, 8],
        latent_dim: int = None,
        decoder_dim: int = 1536,
        decoder_rates: List[int] = [8, 8, 4, 2],
        n_codebooks: int = 9,
        codebook_size: int = 1024,
        codebook_dim: Union[int, list] = 8,
        quantizer_dropout: bool = False,
        sample_rate: int = 44100,
        distill_projection_out_dim=1024,
        distill=False,
        convnext=True,
        is_causal=False,
        residual_autoencode: bool = False,
    ):
        super().__init__()

        self.encoder_dim = encoder_dim
        self.encoder_rates = encoder_rates
        self.decoder_dim = decoder_dim
        self.decoder_rates = decoder_rates
        self.sample_rate = sample_rate

        if latent_dim is None:
            latent_dim = encoder_dim * (2 ** len(encoder_rates))

        self.latent_dim = latent_dim

        self.hop_length = np.prod(encoder_rates)
        self.encoder = Encoder(encoder_dim, encoder_rates, latent_dim, residual_autoencode=residual_autoencode)

        self.n_codebooks = n_codebooks
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.quantizer = ResidualVectorQuantize(
            input_dim=latent_dim,
            n_codebooks=n_codebooks,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            quantizer_dropout=quantizer_dropout,
        )
        print(f'residual_autoencode: {residual_autoencode}')

        self.decoder = Decoder(
            latent_dim,
            decoder_dim,
            decoder_rates,
            residual_autoencode=residual_autoencode,
        )
        self.sample_rate = sample_rate
        self.apply(init_weights)

        self.distill = distill
        if self.distill:
            self.distill_projection = WNConv1d(
                latent_dim, distill_projection_out_dim, kernel_size=1,
            )
            if convnext:
                self.convnext = nn.Sequential(
                    *[ConvNeXtBlock(
                        dim=distill_projection_out_dim,
                        intermediate_dim=2048,
                        is_causal=is_causal,
                    ) for _ in range(5)],  # Unpack the list directly into nn.Sequential
                    WNConv1d(
                        distill_projection_out_dim, 1024, kernel_size=1,
                    )
                )
            else:
                self.convnext = nn.Identity()

    def preprocess(self, audio_data, sample_rate):
        if sample_rate is None:
            sample_rate = self.sample_rate
        assert sample_rate == self.sample_rate

        length = audio_data.shape[-1]
        right_pad = math.ceil(length / self.hop_length) * self.hop_length - length
        audio_data = nn.functional.pad(audio_data, (0, right_pad))

        return audio_data

    def encode(
        self,
        audio_data: torch.Tensor=None,
        sample_rate=24000,
        n_quantizers: int = None,
        subtracted_latent = None,
        encoded_feature: torch.Tensor=None,
        return_info=False,
    ):
        """Encode given audio data and return quantized latent codes

        Parameters
        ----------
        audio_data : Tensor[B x 1 x T]
            Audio data to encode
        n_quantizers : int, optional
            Number of quantizers to use, by default None
            If None, all quantizers are used.
        return_info : bool, optional
            Whether to return additional information, by default False.
            if return_info, should be same as vae (first argument is z with 32 quantizers)
        Returns
        -------
        dict
            A dictionary with the following keys:
            "z" : Tensor[B x D x T]
                Quantized continuous representation of input
            "codes" : Tensor[B x N x T]
                Codebook indices for each codebook
                (quantized discrete representation of input)
            "latents" : Tensor[B x N*D x T]
                Projected latents (continuous representation of input before quantization)
            "vq/commitment_loss" : Tensor[1]
                Commitment loss to train encoder to predict vectors closer to codebook
                entries
            "vq/codebook_loss" : Tensor[1]
                Codebook loss to update the codebook
            "length" : int
                Number of samples in input audio
        """
        assert not self.training
        if encoded_feature is None:
            audio_data = self.preprocess(audio_data, sample_rate)
            z = self.encoder(audio_data)
        else:
            assert audio_data is None
            z = encoded_feature
        if subtracted_latent is not None:
            assert np.abs(z.shape[-1] - subtracted_latent.shape[-1]) <= 2
            z = z[..., :subtracted_latent.shape[-1]] - subtracted_latent
        z, codes, latents, commitment_loss, codebook_loss, first_layer_quantized = self.quantizer(
            z, n_quantizers, possibly_no_quantizer=False,
        )
        if return_info:
            assert n_quantizers == 32
            assert subtracted_latent is None
            assert encoded_feature is None
            assert audio_data is not None
            assert sample_rate == 24000
            assert n_quantizers is not None
            info = {}
            return z, info
        if subtracted_latent is not None:
            z = z + subtracted_latent
        return z, codes, latents, commitment_loss, codebook_loss, first_layer_quantized

    def decode_from_codes(self, acoustic_codes: torch.Tensor, semantic_latent=None):
        # acoustic codes should not contain any semantic code
        z = 0.0
        if acoustic_codes is not None:
            z = self.quantizer.from_codes(acoustic_codes)[0]
        if semantic_latent is not None:
            z = z + semantic_latent

        z = self.decoder(z) # audio
        return z

    def decode(self, latent):
        return self.decoder(latent)

    def forward(
        self,
        audio_data: torch.Tensor=None,
        sample_rate: int = None,
        n_quantizers: int = None,
        subtracted_latent = None,
        bypass_quantize=False,
        possibly_no_quantizer=False,
        cut_from_front=False,
        encoded_feature: torch.Tensor=None,
    ):
        """Model forward pass

        Parameters
        ----------
        audio_data : Tensor[B x 1 x T]
            Audio data to encode
        sample_rate : int, optional
            Sample rate of audio data in Hz, by default None
            If None, defaults to `self.sample_rate`
        n_quantizers : int, optional
            Number of quantizers to use, by default None.
            If None, all quantizers are used.

        Returns
        -------
        dict
            A dictionary with the following keys:
            "z" : Tensor[B x D x T]
                Quantized continuous representation of input
            "codes" : Tensor[B x N x T]
                Codebook indices for each codebook
                (quantized discrete representation of input)
            "latents" : Tensor[B x N*D x T]
                Projected latents (continuous representation of input before quantization)
            "vq/commitment_loss" : Tensor[1]
                Commitment loss to train encoder to predict vectors closer to codebook
                entries
            "vq/codebook_loss" : Tensor[1]
                Codebook loss to update the codebook
            "length" : int
                Number of samples in input audio
            "audio" : Tensor[B x 1 x length]
                Decoded audio data.
        """
        if encoded_feature is not None:
            assert audio_data is None
            z = encoded_feature # [b, c, t]
        else:
            length = audio_data.shape[-1]
            audio_data = self.preprocess(audio_data, sample_rate)
            z = self.encoder(audio_data)
        if subtracted_latent is not None:
            assert (z.shape[-1] - subtracted_latent.shape[-1]) <= 3, f"shape mismatch, {z.shape[-1], subtracted_latent.shape[-1]}"
            if cut_from_front:
                z = z[..., 1:]
            z = z[..., :subtracted_latent.shape[-1]] - subtracted_latent
        if bypass_quantize:
            codes, latents, commitment_loss, codebook_loss, first_layer_quantized = \
                None, None, 0.0, 0.0, None
            z = 0.0
        else:
            z, codes, latents, commitment_loss, codebook_loss, first_layer_quantized = self.quantizer(
                z, n_quantizers, possibly_no_quantizer=possibly_no_quantizer,
            )
        if subtracted_latent is not None:
            z = z + subtracted_latent

        x = self.decoder(z)
        if encoded_feature is None:
            x = pad_to_length(x, length)

        if self.distill:
            first_layer_quantized = self.distill_projection(first_layer_quantized)
            first_layer_quantized = self.convnext(first_layer_quantized)
        
        return edict({
            "x": x,
            "z": z,
            "codes": codes,
            "latents": latents,
            "penalty": commitment_loss,
            "vq/codebook_loss": codebook_loss,
            "metrics": {},
            "first_layer_quantized": first_layer_quantized,
        })


if __name__ == "__main__":
    import numpy as np
    from functools import partial

    model = DAC(
        sample_rate=16000,
        encoder_dim=32,
        encoder_rates=[4,5,6,8],
        decoder_dim=960*2,
        decoder_rates=[8,6,5,4],
        n_codebooks=11,
        codebook_size=1024,
        codebook_dim=8,
        quantizer_dropout=1.0,
        residual_autoencode=True,
    ).to("cpu")

    for n, m in model.named_modules():
        o = m.extra_repr()
        p = sum([np.prod(p.size()) for p in m.parameters()])
        fn = lambda o, p: o + f" {p/1e6:<.3f}M params."
        setattr(m, "extra_repr", partial(fn, o=o, p=p))
    print(model)
    print("Total # of params: ", sum([np.prod(p.size()) for p in model.parameters()]))

    length = 88200 * 2
    x = torch.randn(1, 1, length).to(model.device)
    x.requires_grad_(True)
    x.retain_grad()

    # Make a forward pass
    out = model(x)["x"]
    print("Input shape:", x.shape)
    print("Output shape:", out.shape)

    # Create gradient variable
    grad = torch.zeros_like(out)
    grad[:, :, grad.shape[-1] // 2] = 1

    # Make a backward pass
    out.backward(grad)

    # Check non-zero values
    gradmap = x.grad.squeeze(0)
    gradmap = (gradmap != 0).sum(0)  # sum across features
    rf = (gradmap != 0).sum()

    print(f"Receptive field: {rf.item()}")

    x = AudioSignal(torch.randn(1, 1, 44100 * 60), 44100)
    # model.decompress(model.compress(x, verbose=True), verbose=True)