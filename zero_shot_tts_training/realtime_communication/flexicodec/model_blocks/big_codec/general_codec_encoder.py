import torch
from torch import nn
import numpy as np

from encodec.modules import SConv1d, SConvTranspose1d, SLSTM

from .module import WNConv1d, GeneralEncoderBlock, ResLSTM
from .alias_free_torch import *
from . import activations

import audio_codec.generator.model_blocks.Mimi.transformer as Stransformer

def init_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        nn.init.constant_(m.bias, 0)


class GeneralCodecEncoder(nn.Module):
    def __init__(
        self,
        ngf=48,
        use_rnn=True,
        rnn_bidirectional=False,
        rnn_num_layers=2,
        use_transformer=True,
        transformer_dim_feedforward=2048,
        transformer_num_layers=8,
        up_ratios=(2, 2, 2, 5, 5),
        dilations=(1, 3, 9),
        out_channels=1024,
        causal=False,
        norm="weight_norm",
        pad_mode="reflect",
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
        # RNN
        if use_rnn:
            self.block += [
                ResLSTM(d_model, num_layers=rnn_num_layers, bidirectional=rnn_bidirectional)
            ]
        
        elif use_transformer:
            # NOTE: d_model: input dimension; causal: causal or not; 
            # set num_layers=3 and transformer_dim_feedforward=1024 to get similar para # as RNN
            # context need to tune? currently 250 / frame_rate
            _transformer_kwargs = {'d_model': d_model, 'num_heads': 8, 'num_layers': transformer_num_layers, 
                'causal': causal, 'layer_scale': 0.01, 'context': 250, 'conv_layout': True, 
                'max_period': 10000, 'gating': 'none', 'norm': 'layer_norm', 'positional_embedding': 'rope', 
                'dim_feedforward': transformer_dim_feedforward, 'input_dimension': d_model, 'output_dimensions': [d_model]}
            self.block += [
                Stransformer.ProjectedTransformer(**_transformer_kwargs)
            ]
        
        else:
            print('No RNN or Transformer is used, please check the configuration')
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

        self.reset_parameters()

    def forward(self, x):
        out = self.block(x)
        return out

    def inference(self, x):
        return self.block(x)

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
