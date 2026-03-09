import torch.nn as nn
import torch
import torch.nn.functional as F

class WaveNextHead(nn.Module):
    """
    WaveNext Head module for predicting waveform samples.

    Args:
        dim (int): Hidden dimension of the model.
        n_fft (int): Size of Fourier transform.
        hop_length (int): The distance between neighboring sliding window frames, which should align with
                          the resolution of the input features.
    """

    def __init__(self, dim: int=512, n_fft: int=1024, hop_length: int=400, padding: str = "same"):
        super().__init__()
        l_fft = n_fft + 2
        l_shift = hop_length
        self.linear_1 = torch.nn.Linear(dim, l_fft)
        self.linear_2 = torch.nn.Linear(l_fft, l_shift, bias=False)

        # W init
        nn.init.trunc_normal_(self.linear_1.weight, std=0.02)
        nn.init.trunc_normal_(self.linear_2.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the WaveNextHead module .

        Args:
            x (Tensor): Input tensor of shape (B, L, H), where B is the batch size,
                        L is the sequence length, and H denotes the model dimension.

        Returns:
            Tensor: Reconstructed time-domain audio signal of shape (B, T), where T is the length of the output signal.
        """
        B, C, T = x.shape
        x = self.linear_1(x)
        x = self.linear_2(x)
        audio = x.view(B,-1).unsqueeze(1)
        audio = torch.clip(audio, min=-1.0, max=1.0)
        return audio
    

class WaveNextHead_Inverse(nn.Module):
    """
    WaveNext Head module for encoding waveform samples.

    Args:
        dim (int): Hidden dimension of the model.
        n_fft (int): Size of Fourier transform.
        hop_length (int): The distance between neighboring sliding window frames, which should align with
                          the resolution of the input features.
    """

    def __init__(self, dim: int=512, n_fft: int=1024, hop_length: int=400):
        super().__init__()
        self.l_fft = n_fft + 2
        self.l_shift = hop_length
        self.linear_1 = torch.nn.Linear(self.l_shift, self.l_fft, bias=False)
        self.linear_2 = torch.nn.Linear(self.l_fft, dim)

        # W init
        nn.init.trunc_normal_(self.linear_1.weight, std=0.02)
        nn.init.trunc_normal_(self.linear_2.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the WaveNextHead module .

        Args:
            x (Tensor): Input tensor of shape (B, T), where B is the batch size,
                        T is the audio sample number.

        Returns:
            Tensor: Encoded feature.
        """
        if x.dim() == 3:
            x = x.squeeze(1)
        assert x.dim() == 2

        length = x.shape[1] % self.l_shift
        pad_length = self.l_shift - length if length else 0
        x = F.pad(x, (0, pad_length), 'constant', 0)

        B, T = x.shape
        x = x.view(B, -1, self.l_shift)
        x = self.linear_1(x)
        x = self.linear_2(x)
        return x