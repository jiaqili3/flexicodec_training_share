import torch
from torch import nn
import numpy as np

from encodec.modules import SConv1d, SConvTranspose1d, SLSTM

from audio_codec.generator.model_blocks.big_codec.module import WNConv1d, GeneralEncoderBlock, ResLSTM
from audio_codec.generator.model_blocks.big_codec.alias_free_torch import *
from audio_codec.generator.model_blocks.big_codec import activations


def init_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        nn.init.constant_(m.bias, 0)


class ConvEncoder(nn.Module):
    def __init__(
        self,
        ngf=16,
        up_ratios=(2, 2, 2, 5, 5),
        dilations=(1, 3, 9),
        out_channels=1024,
        causal=True,
        norm="weight_norm",
        pad_mode="constant",
        filter_type="skip",  # updown, skip
    ):
        super().__init__()
        assert filter_type in ["updown", "causal_updown", "skip"]

        self.hop_length = np.prod(up_ratios)
        self.ngf = ngf
        self.up_ratios = up_ratios

        # Create first convolution
        d_model = ngf
        self.block = [
            SConv1d(1, d_model, kernel_size=7, norm=norm, causal=causal, pad_mode=pad_mode)
        ]

        # Create EncoderBlocks that double channels as they downsample by `stride`
        for i, stride in enumerate(up_ratios):
            d_model *= 2
            self.block += [
                GeneralEncoderBlock(
                    d_model,
                    stride=stride,
                    dilations=dilations,
                    norm=norm,
                    causal=causal,
                    pad_mode=pad_mode,
                    filter_type=filter_type,
                )
            ]

        # Create last convolution
        self.block += [
            GeneralActivation1d(
                activation=activations.SnakeBeta(d_model, alpha_logscale=True),
                causal=causal,
                filter_type=filter_type,
            ),
            SConv1d(
                d_model, out_channels, kernel_size=3, norm=norm, causal=causal, pad_mode=pad_mode
            ),
        ]

        # Wrap black into nn.Sequential
        self.block = nn.Sequential(*self.block)
        self.enc_dim = d_model

        num_params = sum(p.numel() for p in self.block.parameters())
        print(f"CodecEncoder: {num_params} parameters")
        # import pdb; pdb.set_trace()

        self.reset_parameters()

    def forward(self, x):
        out = self.block(x)
        out = out.transpose(1, 2).contiguous()
        return out

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""

        def _remove_weight_norm(m):
            try:
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""

        def _apply_weight_norm(m):
            if isinstance(m, nn.Conv1d):
                torch.nn.utils.weight_norm(m)

        self.apply(_apply_weight_norm)

    def reset_parameters(self):
        self.apply(init_weights)
