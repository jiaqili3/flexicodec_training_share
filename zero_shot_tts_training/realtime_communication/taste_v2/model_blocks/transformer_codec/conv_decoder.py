import numpy as np
import torch
import torch.nn as nn
from encodec.modules import SConv1d, SConvTranspose1d, SLSTM

from zero_shot_tts_training.realtime_communication.taste_v2.model_blocks.big_codec.module import WNConv1d, GeneralDecoderBlock, ResLSTM
from zero_shot_tts_training.realtime_communication.taste_v2.model_blocks.big_codec.alias_free_torch import *
from zero_shot_tts_training.realtime_communication.taste_v2.model_blocks.big_codec import activations

def init_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        nn.init.constant_(m.bias, 0)


class ConvDecoder(nn.Module):
    def __init__(
        self,
        in_channels=1024,
        upsample_initial_channel=512,
        up_ratios=(5, 5, 2, 2, 2),
        dilations=(1, 3, 9),
        causal=True,
        norm="weight_norm",
        pad_mode="reflect",
        filter_type="skip",  # updown, skip
    ):
        super().__init__()
        assert filter_type in ["updown", "causal_updown", "skip"]

        self.hop_length = np.prod(up_ratios)
        self.up_ratios = up_ratios

        channels = upsample_initial_channel
        layers = [
            SConv1d(
                in_channels, channels, kernel_size=7, norm=norm, causal=causal, pad_mode=pad_mode
            )
        ]

        for i, stride in enumerate(up_ratios):
            input_dim = channels // 2**i
            output_dim = channels // 2 ** (i + 1)
            input_dim = channels
            output_dim = channels
            layers += [
                GeneralDecoderBlock(
                    input_dim,
                    output_dim,
                    stride,
                    dilations,
                    norm=norm,
                    causal=causal,
                    pad_mode=pad_mode,
                    filter_type=filter_type,
                )
            ]

        layers += [
            GeneralActivation1d(
                activation=activations.SnakeBeta(output_dim, alpha_logscale=True),
                causal=causal,
                filter_type=filter_type,
            ),
            SConv1d(output_dim, 1, kernel_size=7, norm=norm, causal=causal, pad_mode=pad_mode),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*layers)
        
        num_params = sum(p.numel() for p in self.model.parameters())
        print(f"CodecDecoder: {num_params} parameters")
        # import pdb; pdb.set_trace()

        self.reset_parameters()

    def forward(self, x):
        x = x.transpose(1, 2).contiguous()
        x = self.model(x)
        return x

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
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                torch.nn.utils.weight_norm(m)

        self.apply(_apply_weight_norm)

    def reset_parameters(self):
        self.apply(init_weights)
