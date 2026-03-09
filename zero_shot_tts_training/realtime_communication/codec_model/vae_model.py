import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Union
import numpy as np
from audiotools.ml import BaseModel
from dac.nn.layers import WNConv1d
from dac.nn.layers import WNConvTranspose1d
from dac.nn.layers import Snake1d
from .dac_model import Encoder, Decoder
from easydict import EasyDict as edict
import math
from .dac_model import init_weights

def pad_to_length(x, length):
    if x.shape[-1] > length:
        x = x[..., :length]
    elif x.shape[-1] < length:
        x = F.pad(x, (0, length - x.shape[-1]))
    return x

def vae_sample(mean, scale):
    stdev = nn.functional.softplus(scale) + 1e-4
    var = stdev * stdev
    logvar = torch.log(var)
    latents = torch.randn_like(mean) * stdev + mean

    kl = (mean * mean + var - logvar - 1).sum(1).mean()

    return latents, kl, logvar, stdev

class VAEBottleneck(nn.Module):
    def __init__(self):
        super().__init__()
        
    def encode(self, x, return_info=False, **kwargs):
        info = {}

        mean, scale = x.chunk(2, dim=1)

        x, kl, logvar, std = vae_sample(mean, scale)

        info["kl"] = kl
        info['mean'] = mean
        info['std'] = std
        info['logvar'] = logvar

        if return_info:
            return x, info
        else:
            return x

    def decode(self, x):
        return x

class SigmaVAEBottleneck(nn.Module):
    def __init__(self, std=0.75):
        super().__init__()
        self.std = std
        
    def encode(self, x, return_info=False, **kwargs):
        info = {}
        
        # In SigmaVAE, we only use the mean and fix the variance
        mean = x  # x is already just the mean
        
        # Sample using fixed standard deviation
        batch_size = mean.size(0)
        value = self.std / 0.8  # Scale factor similar to the image implementation
        std = torch.randn(batch_size).to(device=mean.device) * value
        
        # Expand std to match mean dimensions
        while std.dim() < mean.dim():
            std = std.unsqueeze(-1)
            
        # Sample from Gaussian with fixed variance
        latents = mean + std * torch.randn_like(mean)
        
        # Calculate KL loss (simplified for fixed variance)
        kl = (mean * mean).sum(1).mean()
        
        info["kl"] = kl
        
        if return_info:
            return latents, info
        else:
            return latents
            
    def decode(self, x):
        return x

class VAE(BaseModel):
    def __init__(
        self,
        encoder_dim: int = 64,
        encoder_rates: List[int] = [2, 4, 8, 8],
        latent_dim: int = 32,
        decoder_dim: int = 1536,
        decoder_rates: List[int] = [8, 8, 4, 2],
        sample_rate: int = 44100,
        use_sigma_vae: bool = False,
        sigma_vae_std: float = 0.75,
    ):
        super().__init__()

        self.encoder_dim = encoder_dim
        self.encoder_rates = encoder_rates
        self.decoder_dim = decoder_dim
        self.decoder_rates = decoder_rates
        self.sample_rate = sample_rate
        self.use_sigma_vae = use_sigma_vae
        self.sigma_vae_std = sigma_vae_std

        if latent_dim is None:
            latent_dim = encoder_dim * (2 ** len(encoder_rates))

        self.latent_dim = latent_dim
        self.hop_length = np.prod(encoder_rates)
        
        # Encoder from DAC
        if use_sigma_vae:
            # For SigmaVAE, we only need to output the mean
            self.encoder = Encoder(encoder_dim, encoder_rates, latent_dim)
            # Use SigmaVAE bottleneck
            self.vae_bottleneck = SigmaVAEBottleneck(std=sigma_vae_std)
        else:
            # Standard VAE outputs both mean and scale
            self.encoder = Encoder(encoder_dim, encoder_rates, latent_dim * 2)
            # Use standard VAE bottleneck
            self.vae_bottleneck = VAEBottleneck()
        
        # Decoder from DAC
        self.decoder = Decoder(
            latent_dim,  # Use original latent dim for decoder input
            decoder_dim,
            decoder_rates,
        )
        
        self.sample_rate = sample_rate
        self.apply(init_weights)

    def preprocess(self, audio_data, sample_rate):
        if sample_rate is None:
            sample_rate = self.sample_rate
        assert sample_rate == self.sample_rate

        length = audio_data.shape[-1]
        right_pad = math.ceil(length / self.hop_length) * self.hop_length - length
        audio_data = nn.functional.pad(audio_data, (0, right_pad))

        return audio_data

    @torch.no_grad()
    def encode(self, audio_data, sample_rate=24000, return_info=False):
        """Encode given audio data and return latent representation with KL loss

        Parameters
        ----------
        audio_data : Tensor[B x 1 x T]
            Audio data to encode
        sample_rate : int, optional
            Sample rate of audio data in Hz, by default 24000
        return_info : bool, optional
            Whether to return additional information, by default False

        Returns
        -------
        dict or tuple
            If return_info is False:
                Tensor[B x D x T]: Latent representation
            If return_info is True:
                Tuple containing:
                - Tensor[B x D x T]: Latent representation
                - dict: Additional information including KL loss
        """
        assert not self.training
        audio_data = self.preprocess(audio_data, sample_rate)
        z = self.encoder(audio_data)
        z, info = self.vae_bottleneck.encode(z, return_info=True)
        
        if return_info:
            return z, info
        return z

    def decode(self, z):
        """Decode latent representation to audio

        Parameters
        ----------
        z : Tensor[B x D x T]
            Latent representation to decode

        Returns
        -------
        Tensor[B x 1 x T]
            Decoded audio
        """
        x = self.decoder(z)
        return x

    def forward(self, audio_data, sample_rate=None):
        """Model forward pass

        Parameters
        ----------
        audio_data : Tensor[B x 1 x T]
            Audio data to encode
        sample_rate : int, optional
            Sample rate of audio data in Hz, by default None
            If None, defaults to `self.sample_rate`

        Returns
        -------
        dict
            A dictionary with the following keys:
            "x" : Tensor[B x 1 x T]
                Decoded audio data
            "z" : Tensor[B x D x T]
                Latent representation
            "kl_loss" : Tensor[1]
                KL divergence loss
        """
        length = audio_data.shape[-1]
        audio_data = self.preprocess(audio_data, sample_rate)
        
        # Encode
        z = self.encoder(audio_data)
        z, info = self.vae_bottleneck.encode(z, return_info=True)
        
        # Decode
        x = self.decoder(z)
        x = pad_to_length(x, length)
        
        return edict({
            "x": x,
            "z": z,
            "kl": info["kl"],
            "metrics": {
                'kl_loss': info["kl"],
            },
        })
