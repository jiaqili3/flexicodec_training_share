from vector_quantize_pytorch import GroupedResidualFSQ
import torch
import torch.nn as nn
from dataclasses import dataclass, field
import typing as tp

import torch
from torch import nn


@dataclass
class QuantizedResult:
    x: torch.Tensor
    codes: torch.Tensor
    bandwidth: torch.Tensor  # bandwidth in kb/s used, per batch item.
    penalty: tp.Optional[torch.Tensor] = None
    x_semantic: torch.Tensor = None
    metrics: dict = field(default_factory=dict)


class FSQVectorQuantizer(nn.Module):
    """Residual Vector Quantizer with separate projections for the first quantizer and the rest.

    Args:
        n_q (int): Number of residual vector quantizers used.
        n_semantic_q (int): Number of residual vector quantizers used for the semantic quantizer.
        no_quantization_mode (str): if 'true_skip', when doing no quantization, the input will not go
            through the sub quantizers. If `independent`, independent decisions are taken by
            the semantic and acoustic quantizers. If `same` (the default), the same decision is taken by both.
        **kwargs: Arguments to the constructor of `ResidualVectorQuantizer` that are shared between both.
    """

    def __init__(
        self,
        *,
        n_q: int = 8,
        no_quantization_rate: float = 0.0,
        no_quantization_mode: str = "same",
        n_q_semantic: int = 1,
        **kwargs,
    ):
        super().__init__()
        
        self.max_n_q = n_q
        self.num_codebooks = n_q
        self.n_q_semantic = n_q_semantic
        self.n_q_acoustic = n_q - n_q_semantic
        if no_quantization_mode == "true_skip":
            self.no_quantization_rate = no_quantization_rate
            # Setting to zero for the underlying RVQ.
            no_quantization_rate = 0.0
        else:
            self.no_quantization_rate = 0.0
        if no_quantization_mode == "same":
            kwargs["generator_seed"] = 1234
        kwargs["no_quantization_rate"] = no_quantization_rate
        
        self.rvq_rest = GroupedResidualFSQ(
            dim=512,
            groups=2,
            levels=[8,5,5,5],
            num_quantizers=4,
        )
        self.layer_norm = nn.LayerNorm(512)

    def forward(self, x: torch.Tensor, frame_rate: int):
        """
        Args:
            x (torch.Tensor): Input tensor of shape [B, C, T] with `C` number of channels.
            frame_rate (int): frame rate of the input (e.g `T = frame_rate * duration`), used to compute
                the bandwidth.

        Returns:
            QuantizedResult: Quantized result with the following attributes:
                - `x` (torch.Tensor): Quantized tensor of shape [B, C, T].
                - `codes` (torch.Tensor): Quantized codes of shape [B, K, T] with `K` number of codebooks.
                - `bw` (torch.Tensor): Bandwidth of the quantized tensor in kbits per second.
                - `penalty` (torch.Tensor): Commitment loss.
                - `metrics` (dict): RVQ metrics, in particular rate of dead code replacement, and entropy.
        """
        x = self.layer_norm(x.transpose(1,2))
        acoustic_result = self.rvq_rest(x)
        # l2_norm = x.norm(p=2)

        # randomly dropout
        if self.training and torch.rand(1) > 0.:
            full_quantized_emb = acoustic_result[0].transpose(1,2) # [B, C, T]
            full_quantized_codes = acoustic_result[1]
        else:
            full_quantized_emb = x.transpose(1,2) # [B, C, T]
            full_quantized_codes = None

        return QuantizedResult(
            full_quantized_emb,
            full_quantized_codes,
            penalty=torch.tensor(0.0),
            metrics={},
            bandwidth=0.0,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a given input tensor with the specified frame rate at the given bandwidth.
        The RVQ encode method sets the appropriate number of quantizer to use
        and returns indices for each quantizer.
        """
        codes = self.rvq_rest(x.transpose(1,2))[1]
        return codes

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode the given codes to the quantized representation."""
        return self.rvq_rest.get_output_from_indices(codes).transpose(1,2)

    @property
    def acoustic_quantizer(self):
        """This returns the quantizer that models the higher levels of the hierarchy (typically acoustic)."""
        return self.rvq_rest




# residual_fsq = GroupedResidualFSQ(
#     dim=256,
#     groups=2,
#     levels=[8,5,5,5],
#     num_quantizers=4,
# )
# x = torch.randn(1, 1024, 256)
# quantized, indices = residual_fsq(x)
# quantized_out = residual_fsq.get_output_from_indices(indices)
# breakpoint()