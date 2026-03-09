import numpy as np
import torch
import torch.nn as nn
from encodec.modules import SConv1d, SConvTranspose1d, SLSTM

from .residual_vq import ResidualVQ
from .module import WNConv1d, GeneralDecoderBlock, ResLSTM
from .alias_free_torch import *
from . import activations

import audio_codec.generator.model_blocks.Mimi.transformer as Stransformer

def init_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        nn.init.constant_(m.bias, 0)


class GeneralCodecDecoder(nn.Module):
    def __init__(
        self,
        in_channels=1024,
        upsample_initial_channel=1536,
        ngf=48,
        use_rnn=True,
        rnn_bidirectional=False,
        rnn_num_layers=2,
        use_transformer=True,
        transformer_dim_feedforward=2048,
        transformer_num_layers=8,
        up_ratios=(5, 5, 2, 2, 2),
        dilations=(1, 3, 9),
        vq_num_quantizers=1,
        vq_dim=1024,
        vq_commit_weight=0.25,
        vq_weight_init=False,
        vq_full_commit_loss=False,
        codebook_size=8192,
        codebook_dim=8,
        causal=False,
        norm="weight_norm",
        pad_mode="reflect",
        filter_type="skip",  # updown, skip
        RVQ_dropout=False,
        RVQ_dropout_weight=[1.0],  # weight to select each layer when RVQ_droput is True
    ):
        super().__init__()
        assert filter_type in ["updown", "causal_updown", "skip"]

        self.hop_length = np.prod(up_ratios)
        self.ngf = ngf
        self.up_ratios = up_ratios

        self.quantizer = ResidualVQ(  # no need to change
            num_quantizers=vq_num_quantizers,
            dim=vq_dim,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            threshold_ema_dead_code=2,
            commitment=vq_commit_weight,
            weight_init=vq_weight_init,
            full_commit_loss=vq_full_commit_loss,
            RVQ_dropout=RVQ_dropout,
            RVQ_dropout_weight=RVQ_dropout_weight,
        )
        channels = upsample_initial_channel
        layers = [
            SConv1d(
                in_channels, channels, kernel_size=7, norm=norm, causal=causal, pad_mode=pad_mode
            )
        ]

        if use_rnn: # Para num of RNN: 37.77M
            layers += [
                ResLSTM(
                    channels, num_layers=rnn_num_layers, bidirectional=rnn_bidirectional
                )  # no need to change
            ]
        
        elif use_transformer:
            # NOTE: d_model: input dimension; causal: causal or not; 
            # set num_layers=3 and transformer_dim_feedforward=1024 to get similar para # as RNN
            # context need to tune? currently 250 / frame_rate
            _transformer_kwargs = {'d_model': channels, 'num_heads': 8, 'num_layers': transformer_num_layers, 
                'causal': causal, 'layer_scale': 0.01, 'context': 250, 'conv_layout': True, 
                'max_period': 10000, 'gating': 'none', 'norm': 'layer_norm', 'positional_embedding': 'rope', 
                'dim_feedforward': transformer_dim_feedforward, 'input_dimension': channels, 'output_dimensions': [channels]}
            layers += [
                Stransformer.ProjectedTransformer(**_transformer_kwargs)
            ]
        
        else:
            print('No RNN or Transformer is used, please check the configuration')

        for i, stride in enumerate(up_ratios):
            input_dim = channels // 2**i
            output_dim = channels // 2 ** (i + 1)
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

        self.reset_parameters()

    def forward(self, x, vq=True):
        if vq is True:
            x, q, commit_loss = self.quantizer(x)
            return x, q, commit_loss
        x = self.model(x)
        return x

    def vq2emb(self, vq):
        self.quantizer = self.quantizer.eval()
        x = self.quantizer.vq2emb(vq)
        return x

    def get_emb(self):
        self.quantizer = self.quantizer.eval()
        embs = self.quantizer.get_emb()
        return embs

    def inference_vq(self, vq):
        x = vq[None, :, :]
        x = self.model(x)
        return x

    def inference_0(self, x):
        x, q, loss, perp = self.quantizer(x)
        x = self.model(x)
        return x, None

    def inference(self, x):
        x = self.model(x)
        return x, None

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
