from typing import Tuple
import torch.nn as nn
from torch.nn import functional as F

from cosyvoice.utils.mask import make_pad_mask


class InterpolateRegulator(nn.Module):
    """
    InterpolateRegulator Module
    """

    def __init__(
        self,
        channels: int,
        sampling_ratios: Tuple,
        out_channels: int = None,
        groups: int = 1,
    ):
        """
        :param channels:
        :param sampling_ratios:
        :param out_channels:
        :param groups:
        """
        super().__init__()
        self.sampling_ratios = sampling_ratios
        out_channels = out_channels or channels
        model = nn.ModuleList([])
        if len(sampling_ratios) > 0:
            for _ in sampling_ratios:
                module = nn.Conv1d(channels, channels, 3, 1, 1)
                norm = nn.GroupNorm(groups, channels)
                act = nn.Mish()
                model.extend([module, norm, act])
        model.append(nn.Conv1d(channels, out_channels, 1, 1))
        self.model = nn.Sequential(*model)

    def forward(self, x, ylens=None):
        """
        :param x:
        :param ylens:
        :return:
        """
        # x in (B, T, D)
        mask = (~make_pad_mask(ylens)).to(x).unsqueeze(-1)
        x = F.interpolate(
            x.transpose(1, 2).contiguous(), size=ylens.max(), mode="nearest"
        )
        out = self.model(x).transpose(1, 2).contiguous()
        olens = ylens
        return out * mask, olens
