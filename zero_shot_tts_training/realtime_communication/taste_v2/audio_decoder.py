import torch.nn as nn
import torch
from einops import rearrange

from .model_blocks.transformer_codec.wavenext import WaveNextHead
from .model_blocks.transformer_codec.conv_decoder import ConvDecoder
from .model_blocks.mimi import transformer as Stransformer

get_num_params = lambda model: sum(p.numel() for p in model.parameters())

def init_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        nn.init.constant_(m.bias, 0)

class AudioDecoder(nn.Module):
    def __init__(
        self,
        type: str = "transformer",
        # Transformer args
        d_model: int = 1024,
        num_heads: int = 16,
        num_layers: int = 8,
        dim_feedforward: int = 4096,
        causal: bool = False,
        layer_scale: float = 0.01,
        context: int = None, # infinite context
        conv_layout: bool = False,
        max_period: int = 10000,
        gating: str = 'none',
        norm: str = 'layer_norm',
        positional_embedding: str = 'rope',
        input_dimension: int = 1024,
        output_dimensions: list = [1024],
        is_conv_decoder: bool = False,
        # For ConvDecoder
        # up_ratios: list = [2, 2, 2, 2],
        # upsample_initial_channel: int = 128,
        # For WaveNextHead
        n_fft: int = 2048,
        hop_length: int = 960, # 16khz input
    ):
        super().__init__()

        _transformer_decoder__kwargs = {
            'd_model': d_model, 'num_heads': num_heads, 'num_layers': num_layers, 
            'causal': causal, 'layer_scale': layer_scale, 'context': context, 'conv_layout': conv_layout, 
            'max_period': max_period, 'gating': gating, 'norm': norm, 'positional_embedding': positional_embedding, 
            'dim_feedforward': dim_feedforward, 'input_dimension': input_dimension, 'output_dimensions': output_dimensions
        }

        if is_conv_decoder:
            assert False
            self.decoder = ConvDecoder(in_channels=d_model, up_ratios=up_ratios, upsample_initial_channel=upsample_initial_channel, causal=causal)
        else:
            self.decoder = WaveNextHead(dim=d_model, n_fft=n_fft, hop_length=hop_length)
        
        self.decoder_transformer = Stransformer.ProjectedTransformer(**_transformer_decoder__kwargs)

        self.reset_parameters()

    def decode(self, z):
        # z is the quantized semantic features, with shape (B, T, D)
        decoded = self.decoder_transformer(z)
        y = self.decoder(decoded)
        return y
    
    def forward(self, semantic_features=None):
        decoded = self.decode(semantic_features)
        return {
            "audio": decoded,
        }

    def reset_parameters(self):
        self.apply(init_weights)

def test_audiodec():
    import torchaudio
    from .audio_encoder import SenseVoiceAudioEncoder
    # 1. Init encoder and decoder
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sensevoice_encoder = SenseVoiceAudioEncoder(
        model_card="/data1/lijiaqi/codebase/TASTE-SpokenLM/STAGE1_TRAIN/storage/pretrained_models/SenseVoiceSmall",
        model_code_dir="/data1/lijiaqi/codebase/CSDs/zero-shot-tts-training/zero_shot_tts_training/realtime_communication/taste_v2/customized_sensevoice/model.py",
        extract_hidden=True,
    ).to(device)
    sensevoice_encoder.eval()

    audio_decoder = AudioDecoder().to(device)
    audio_decoder.eval()

    # 2. Prepare inputs
    audio_fpaths = [
        "/data1/lijiaqi/codebase/TASTE-SpokenLM/STAGE1_TRAIN/storage/pretrained_models/SenseVoiceSmall/example/en.mp3",
    ]
    audio_features, audio_features_lengths = sensevoice_encoder.extract_feature(
        audio_fpaths,
        cache={},
        language="auto",
        use_itn=True,
        batch_size_s=60,
    )
    audio_features = audio_features.to(device)
    audio_features_lengths = audio_features_lengths.to(device)

    # 3. Get semantic features from encoder
    with torch.no_grad():
        encoder_results = sensevoice_encoder.forward(audio_features, audio_features_lengths, return_text=True)
        # We use the 'vq' output which contains the quantized features
        quantized_semantic_features = encoder_results['deaggregated_features']
        print(f"Quantized semantic features shape: {quantized_semantic_features.shape}")

        # 4. Decode to get audio
        decoder_results = audio_decoder(semantic_features=quantized_semantic_features)
        output_audio = decoder_results['audio']
        print(f"Output audio shape: {output_audio.shape}")

        # 5. Save audio
        torchaudio.save("decoded_audio.wav", output_audio.cpu().squeeze(0), 16000)
        print("Saved decoded audio to decoded_audio.wav")

if __name__ == "__main__":
    test_audiodec()
