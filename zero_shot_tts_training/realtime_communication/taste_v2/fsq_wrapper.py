from einops import rearrange
import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F
from typing import List

from .fsq_quantizer import FSQ

class FSQWrapper(nn.Module):
    def __init__(self, input_dim, levels=[8,8,8,8,8], num_codebooks=1, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        
        self.fsq = FSQ(levels=levels, dim=input_dim, num_codebooks=num_codebooks, **kwargs)

    def forward(self, x, n_quantizers=None, possibly_no_quantizer=False):
        # x is (B, D, T) where D is input_dim
        x_transposed = x.transpose(1, 2) # (B, T, D)
        
        zq_transposed, indices = self.fsq(x_transposed)
        
        zq = zq_transposed.transpose(1, 2)
        
        if self.fsq.num_codebooks > 1:
            indices = indices.transpose(1, 2)
        else:
            indices = indices.unsqueeze(1)

        latents = None

        codebook_loss = torch.tensor(0.0, device=x.device)
        loss = torch.tensor(0.0, device=x.device)

        first_layer_quantized = zq
        
        return zq, indices, latents, loss, codebook_loss, first_layer_quantized

    def from_codes(self, codes):
        # codes: (B, n_q, T)
        if self.fsq.num_codebooks > 1:
            indices = codes.transpose(1, 2)
        else:
            indices = codes.squeeze(1)
        
        quantized_transposed = self.fsq.indices_to_codes(indices) 
        
        quantized = quantized_transposed.transpose(1, 2)
        
        return quantized, None 

# --- Invertible LayerNorm for Sequences (as defined above) ---
class InvertibleLayerNorm1D(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer('current_mean', None, persistent=False)
        self.register_buffer('current_std', None, persistent=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.current_mean = x.mean(dim=-1, keepdim=True)
        self.current_std = x.std(dim=-1, keepdim=True, unbiased=False) + self.eps
        normalized = (x - self.current_mean) / self.current_std
        return self.weight * normalized + self.bias
    def inverse(self, normalized_x: torch.Tensor) -> torch.Tensor:
        if self.current_mean is None or self.current_std is None:
            raise RuntimeError("The forward pass must be called before the inverse pass.")
        denormalized = (normalized_x - self.bias) / self.weight
        return denormalized * self.current_std + self.current_mean


# --- Main RFSQ Wrapper ---
class RFSQWrapper(nn.Module):
    """
    A wrapper for Residual Finite Scalar Quantization (RFSQ) with multiple strategies.
    Args:
        input_dim (int): The feature dimension of the input tensor.
        levels (List[int]): Levels for the FSQ codebook.
        num_codebooks (int): Number of codebooks per stage. Defaults to 2.
        num_quantizers (int): Number of residual stages. Defaults to 4.
        strategy (str): Strategy for RFSQ. Can be 'none', 'scale', or 'layernorm'.
        initial_scale (float): Initial scale for the 'scale' strategy.
        **kwargs: Additional arguments for FSQ constructors.
    """
    def __init__(
        self, 
        input_dim, 
        levels=[8,8,8,8,8], 
        num_codebooks=2, 
        num_quantizers: int = 4,
        strategy: str = 'none',
        initial_scale: float = 1.0,
        **kwargs
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_codebooks = num_codebooks
        self.num_quantizers = num_quantizers
        self.strategy = strategy
        if self.strategy not in ['none', 'scale', 'layernorm']:
            raise ValueError("RFSQ strategy must be 'none', 'scale', or 'layernorm'")
        self.quantizers = nn.ModuleList([
            FSQ(levels=levels, dim=input_dim, num_codebooks=num_codebooks, **kwargs) 
            for _ in range(self.num_quantizers)
        ])
        if self.strategy == 'scale':
            self.log_scales = nn.ParameterList([
                nn.Parameter(torch.log(torch.tensor(initial_scale)))
                for _ in range(self.num_quantizers)
            ])
        elif self.strategy == 'layernorm':
            self.layernorms = nn.ModuleList([
                InvertibleLayerNorm1D(num_features=input_dim)
                for _ in range(self.num_quantizers)
            ])
    def forward(self, x, n_quantizers=None, possibly_no_quantizer=False):
        x_transposed = x.transpose(1, 2) # (B, T, D)
        residual = x_transposed
        all_quantized_vectors = []
        all_indices = []
        num_quantizers_to_use = n_quantizers if n_quantizers is not None else self.num_quantizers
        for i in range(num_quantizers_to_use):
            current_input = residual
            quantizer = self.quantizers[i]
            if self.strategy == 'scale':
                scale = F.softplus(self.log_scales[i])
                scaled_input = current_input * scale
                quantized_scaled, indices_stage = quantizer(scaled_input)
                quantized_true = quantized_scaled / scale
            elif self.strategy == 'layernorm':
                layernorm = self.layernorms[i]
                normalized_input = layernorm(current_input)
                quantized_normalized, indices_stage = quantizer(normalized_input)
                quantized_true = layernorm.inverse(quantized_normalized)
            else:  # 'none' strategy
                quantized_true, indices_stage = quantizer(current_input)
            residual = current_input - quantized_true
            all_quantized_vectors.append(quantized_true)
            all_indices.append(indices_stage)
        zq_transposed = sum(all_quantized_vectors)
        zq = zq_transposed.transpose(1, 2)
        indices_stacked = torch.stack(all_indices, dim=1)
        indices = rearrange(indices_stacked, 'b nq t nc -> b (nq nc) t')
        first_layer_quantized = all_quantized_vectors[0].transpose(1, 2)
        latents = None
        codebook_loss = torch.tensor(0.0, device=x.device)
        loss = torch.tensor(0.0, device=x.device)
        return zq, indices, latents, loss, codebook_loss, first_layer_quantized
    def from_codes(self, codes):
        if self.strategy == 'layernorm':
            raise NotImplementedError(
                "The 'from_codes' method is not supported for the 'layernorm' strategy. "
                "The inverse transformation is data-dependent (requires mean/std from the original signal) "
                "and cannot be performed from codes alone."
            )
        num_active_quantizers = codes.shape[1] // self.num_codebooks
        codes_reshaped = rearrange(codes, 'b (nq nc) t -> b nq nc t', nc=self.num_codebooks)
        indices_per_stage = codes_reshaped.transpose(2, 3)
        all_quantized_vectors = []
        for i in range(num_active_quantizers):
            stage_indices = indices_per_stage[:, i, :, :]
            quantizer = self.quantizers[i]
            quantized_output = quantizer.indices_to_codes(stage_indices)
            if self.strategy == 'scale':
                scale = F.softplus(self.log_scales[i])
                quantized_true = quantized_output / scale
            else:  # 'none' strategy
                quantized_true = quantized_output
            all_quantized_vectors.append(quantized_true)
        quantized_transposed = sum(all_quantized_vectors)
        quantized = quantized_transposed.transpose(1, 2)
        return quantized, None
    
