# Adapted from https://github.com/junjun3518/alias-free-torch under the Apache License 2.0
#   LICENSE is in incl_licenses directory.

import torch.nn as nn
from torch.nn import functional as F
from .filter import LowPassFilter1d
from .filter import GeneralLowPassFilter1d
from .filter import kaiser_sinc_filter1d


class UpSample1d(nn.Module):
    def __init__(self, ratio=2, kernel_size=None):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        self.stride = ratio
        self.pad = self.kernel_size // ratio - 1
        self.pad_left = self.pad * self.stride + (self.kernel_size - self.stride) // 2
        self.pad_right = self.pad * self.stride + (self.kernel_size - self.stride + 1) // 2
        filter = kaiser_sinc_filter1d(
            cutoff=0.5 / ratio, half_width=0.6 / ratio, kernel_size=self.kernel_size
        )
        self.register_buffer("filter", filter)

    # x: [B, C, T]
    def forward(self, x):
        _, C, _ = x.shape

        x = F.pad(x, (self.pad, self.pad), mode="replicate")
        x = self.ratio * F.conv_transpose1d(
            x, self.filter.expand(C, -1, -1), stride=self.stride, groups=C
        )
        x = x[..., self.pad_left : -self.pad_right]

        return x


class DownSample1d(nn.Module):
    def __init__(self, ratio=2, kernel_size=None):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        self.lowpass = LowPassFilter1d(
            cutoff=0.5 / ratio, half_width=0.6 / ratio, stride=ratio, kernel_size=self.kernel_size
        )

    def forward(self, x):
        xx = self.lowpass(x)

        return xx


class GeneralUpSample1d(nn.Module):
    def __init__(self, ratio=2, kernel_size=None, causal=False, filter_type="updown"):
        super().__init__()
        assert filter_type in ["updown", "causal_updown"]

        self.ratio = ratio
        self.kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        self.stride = ratio
        self.pad = self.kernel_size // ratio - 1
        self.pad_left = self.pad * self.stride + (self.kernel_size - self.stride) // 2
        self.pad_right = self.pad * self.stride + (self.kernel_size - self.stride + 1) // 2
        filter = kaiser_sinc_filter1d(
            cutoff=0.5 / ratio, half_width=0.6 / ratio, kernel_size=self.kernel_size
        )
        if filter_type == "causal_updown":
            # NOTE: in case of conv_transpose1d, we need to use the last half
            filter = filter[:, :, kernel_size // 2 :]
            odd = kernel_size % 2 == 1
            self.kernel_size = kernel_size // 2 + odd

        self.register_buffer("filter", filter)
        self.causal = causal

    # x: [B, C, T]
    def forward(self, x):
        _, C, _ = x.shape

        if self.causal:
            x = self.ratio * F.conv_transpose1d(
                x, self.filter.expand(C, -1, -1), stride=self.stride, groups=C
            )
            x = x[..., : -(self.kernel_size - self.stride)]
        else:
            x = F.pad(x, (self.pad, self.pad), mode="replicate")
            x = self.ratio * F.conv_transpose1d(
                x, self.filter.expand(C, -1, -1), stride=self.stride, groups=C
            )
            x = x[..., self.pad_left : -self.pad_right]

        return x


class GeneralDownSample1d(nn.Module):
    def __init__(self, ratio=2, kernel_size=None, causal=False, filter_type="updown"):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        self.lowpass = GeneralLowPassFilter1d(
            cutoff=0.5 / ratio,
            half_width=0.6 / ratio,
            stride=ratio,
            kernel_size=self.kernel_size,
            causal=causal,
            filter_type=filter_type,
        )
        self.causal = causal

    def forward(self, x):
        xx = self.lowpass(x)

        return xx
