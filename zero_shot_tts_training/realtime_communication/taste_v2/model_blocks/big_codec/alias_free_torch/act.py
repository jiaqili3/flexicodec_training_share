# Adapted from https://github.com/junjun3518/alias-free-torch under the Apache License 2.0
#   LICENSE is in incl_licenses directory.

import torch.nn as nn
from .resample import UpSample1d, DownSample1d
from .resample import GeneralUpSample1d, GeneralDownSample1d


class Activation1d(nn.Module):
    def __init__(
        self,
        activation,
        up_ratio: int = 2,
        down_ratio: int = 2,
        up_kernel_size: int = 12,
        down_kernel_size: int = 12,
    ):
        super().__init__()
        self.up_ratio = up_ratio
        self.down_ratio = down_ratio
        self.act = activation
        self.upsample = UpSample1d(up_ratio, up_kernel_size)
        self.downsample = DownSample1d(down_ratio, down_kernel_size)

    # x: [B,C,T]
    def forward(self, x):
        x = self.upsample(x)
        x = self.act(x)
        x = self.downsample(x)

        return x


class GeneralActivation1d(nn.Module):
    def __init__(
        self,
        activation,
        up_ratio: int = 2,
        down_ratio: int = 2,
        up_kernel_size: int = 12,
        down_kernel_size: int = 12,
        causal: bool = False,
        filter_type: str = "updown",
    ):
        super().__init__()
        self.up_ratio = up_ratio
        self.down_ratio = down_ratio
        self.act = activation
        self.filter_type = filter_type

        if self.filter_type in ["updown", "causal_updown"]:
            self.upsample = GeneralUpSample1d(
                up_ratio, up_kernel_size, causal=causal, filter_type=filter_type
            )
            self.downsample = GeneralDownSample1d(
                down_ratio, down_kernel_size, causal=causal, filter_type=filter_type
            )

    # x: [B,C,T]
    def forward(self, x):
        if self.filter_type in ["updown", "causal_updown"]:
            x = self.upsample(x)

        x = self.act(x)

        if self.filter_type in ["updown", "causal_updown"]:
            x = self.downsample(x)

        return x
