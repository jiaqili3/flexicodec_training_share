from einops import rearrange
import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F
from .quantize.bsq import BinarySphericalQuantizer

class BSQWrapper(nn.Module):
    def __init__(self, input_dim, embed_dim=14, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        
        self.projection = nn.Linear(input_dim, embed_dim)
        self.inverse_projection = nn.Linear(embed_dim, input_dim)

        self.bsq = BinarySphericalQuantizer(embed_dim=embed_dim, input_format='blc', **kwargs)

    def forward(self, x, n_quantizers=None, possibly_no_quantizer=False):
        # x is (B, D, T) where D is input_dim
        x = x.transpose(1, 2) # (B, T, D)
        
        # Project to BSQ's embedding dimension
        projected_x = self.projection(x)

        # Quantize in the lower-dimensional space
        zq_embed_dim, loss, info = self.bsq(projected_x)
        
        # Project back to the original input dimension
        zq_input_dim = self.inverse_projection(zq_embed_dim)

        # Transpose back to (B, D, T)
        zq = zq_input_dim.transpose(1, 2)
        
        # Match RVQ output format
        codes = info['indices'].unsqueeze(1) # (B, 1, T)
        latents = None # No equivalent in BSQ
        
        # The returned loss from BSQ is a weighted sum of commit loss and entropy penalties.
        # We'll report the total loss as the commitment_loss for simplicity in the training loop,
        # and codebook_loss as zero.
        
        # In RVQ, first_layer_quantized is the output from the first quantizer.
        # We return the full quantized output in the original dimension space.
        first_layer_quantized = zq
        
        return zq, codes, latents, loss, torch.tensor(0.0, device=x.device), first_layer_quantized

    def from_codes(self, codes):
        # codes: (B, n_q, T), where n_q is 1 for semantic VQ
        if codes.shape[1] > 1:
            codes = codes[:, :1, :]
            
        codes = codes.squeeze(1) # (B, T)
        
        # Get quantized vector in the bsq embedding space
        quantized_embed_dim = self.bsq.get_codebook_entry(codes) 
        
        # Project back to the original input dimension
        quantized_input_dim = self.inverse_projection(quantized_embed_dim)

        # Transpose to (B, D, T)
        quantized = quantized_input_dim.transpose(1, 2)
        
        # RVQ returns (quantized, band_proportions), so we return None for the second value.
        return quantized, None 