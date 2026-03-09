
import torch.nn as nn
from .dual_model import DualCodec

from .utils_dualcodec_dasheng import *
from .utils_dualcodec_dasheng import infer_dasheng_encoder
# import .utils_dualcodec_dasheng as utils_dualcodec_dasheng
import torchaudio
from einops import rearrange 
from loguru import logger
from functools import partial

class DualCodecWithDasheng(nn.Module):
    def __init__(self, dual_model, dasheng_model_name='dasheng_06B',
    freeze_dasheng=False, dasheng_encoder_ckpt_path='/gluster-ssd-tts/jiaqi_repos/dasheng_06b.pt', dasheng_ckpt_path=None, semantic_downsample_factor=1):
        super(DualCodecWithDasheng, self).__init__()
        self.dualcodec_model = dual_model
        self.dasheng_model = globals()[dasheng_model_name]()

        if dasheng_ckpt_path is not None:
            raise NotImplementedError
        if dasheng_encoder_ckpt_path is not None:
            self.dasheng_model.encoder.load_state_dict(torch.load(dasheng_encoder_ckpt_path)['model'])
            logger.info(f'loaded dasheng model ckpt from {dasheng_encoder_ckpt_path}')

        self.freeze_dasheng = freeze_dasheng
        self.dasheng_encoder_ckpt_path = dasheng_encoder_ckpt_path
        self.dasheng_ckpt_path = dasheng_ckpt_path
        self.semantic_downsample_factor = semantic_downsample_factor
        if freeze_dasheng:
            for param in self.dasheng_model.parameters():
                param.requires_grad = False
            self.dasheng_model.eval()

        # compatibility to dualcodec
        self.override_dac_encoder = False

    @torch.inference_mode()
    def encode(self, x, num_quantizers=None, sample_rate=24000, semantic_repr=None, return_semantic_feat=False):
        # x: [B, 1, T] audio waveform
        # num_quantizers: number of quantizers
        # sample_rate: sample rate of x
        # semantic_repr: not used
        # return: semantic_codes, acoustic_codes
        self.dualcodec_model.eval()
        self.dasheng_model.eval()

        # resample x to 16kHz
        x_16k = torchaudio.functional.resample(x, orig_freq=24000, new_freq=16000)
        
        semantic_repr = infer_dasheng_encoder(self.dasheng_model, x_16k.squeeze(1), no_grad=self.freeze_dasheng)
        # semantic_repr: [B, T, C], C=1280
        semantic_repr = rearrange(semantic_repr, "b t c -> b c t")

        # pool the semantic_repr
        semantic_repr = torch.nn.functional.avg_pool1d(semantic_repr, self.semantic_downsample_factor, self.semantic_downsample_factor)

        try:
            if return_semantic_feat:
                semantic_codes, acoustic_codes, semantic_feat = self.dualcodec_model.encode(
                    audio_data=x,
                    semantic_repr=semantic_repr,
                    num_quantizers=num_quantizers,
                    return_semantic_feat=True)
                return semantic_codes, acoustic_codes, semantic_feat
            else:
                semantic_codes, acoustic_codes = self.dualcodec_model.encode(
                    audio_data=x,
                    semantic_repr=semantic_repr,
                    num_quantizers=num_quantizers)
                return semantic_codes, acoustic_codes
        except:
            breakpoint()

    def get_semantic_feature(self, x, layer=0, num_quantizers=None, sample_rate=24000, semantic_repr=None):
        if layer != 0:
            raise NotImplementedError
        semantic_feature = self.encode(x, num_quantizers=num_quantizers, sample_rate=sample_rate, semantic_repr=semantic_repr, return_semantic_feat=True)[-1]
        return semantic_feature # [b c t]

    def decode_from_codes(self, semantic_codes, acoustic_codes):
        # semantic_codes: [B, 1, T] semantic codes
        # acoustic_codes: [B, 1, T] acoustic codes
        # return: wav
        return self.dualcodec_model.decode_from_codes(
            semantic_codes=semantic_codes,
            acoustic_codes=acoustic_codes
        )

    def forward(self, x, **kwargs):
        # x: [B, 1, T] audio waveform
        # kwargs: same as DualCodec forward
        # the sample rate of x should be 24kHz

        # resample x to 16kHz
        x_16k = torchaudio.functional.resample(x, orig_freq=24000, new_freq=16000)
        
        semantic_repr = infer_dasheng_encoder(self.dasheng_model, x_16k.squeeze(1), no_grad=self.freeze_dasheng)
        # semantic_repr: [B, T, C], C=1280
        semantic_repr = rearrange(semantic_repr, "b t c -> b c t")

        # pool the semantic_repr
        semantic_repr = torch.nn.functional.avg_pool1d(semantic_repr, self.semantic_downsample_factor, self.semantic_downsample_factor)

        # pop out the original semantic_repr
        kwargs.pop("semantic_repr", None)

        return self.dualcodec_model(
            audio_data=x,
            semantic_repr=semantic_repr,
            **kwargs,
            )


def test_25hz():
    # test DualCodecWithDasheng
    dualcodec_model = DualCodec(
        encoder_rates=[4,5,6,8],
        decoder_rates=[8,6,5,4],
        convnext_dim=1024,
        is_causal=True,
        semantic_downsample_factor=1,
        ssl_dim=1280,
        encoder_dim=80,
    )
    # dualcodec_model.load_state_dict(torch.load('checkpoints/dualcodec.pt'))
    dualcodec_model.eval()
    dualcodec_model.cuda()
    dualcodec_with_dasheng = DualCodecWithDasheng(dualcodec_model, freeze_dasheng=True).cuda()

    ret = dualcodec_with_dasheng(torch.randn(3, 1, 24000).cuda())
    breakpoint()

def test_12hz():
    # test DualCodecWithDasheng
    dualcodec_model = DualCodec(
        encoder_rates=[4,5,6,8,2],
        decoder_rates=[2,8,6,5,4],
        convnext_dim=1024,
        is_causal=True,
        semantic_downsample_factor=2,
        ssl_dim=1280,
        encoder_dim=40,
    )
    # dualcodec_model.load_state_dict(torch.load('checkpoints/dualcodec.pt'))
    dualcodec_model.eval()
    dualcodec_model.cuda()
    dualcodec_with_dasheng = DualCodecWithDasheng(dualcodec_model, freeze_dasheng=True,
    semantic_downsample_factor=2,).cuda()

    # ret = dualcodec_with_dasheng(torch.randn(1, 1, 2400000).cuda())

    encoded = dualcodec_with_dasheng.encode(torch.randn(1,1,242040).cuda())

if __name__ == '__main__':
    test_12hz()