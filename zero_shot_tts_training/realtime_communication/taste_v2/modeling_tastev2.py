import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict as edict

from .audio_encoder import SenseVoiceAudioEncoder
from .audio_decoder import AudioDecoder

class TasteV2Model(nn.Module):
    """
    End-to-end model for speech codec.
    Combines a SenseVoice-based audio encoder and a transformer-based audio decoder.
    """
    def __init__(self, **kwargs):
        super().__init__()
        
        encoder_kwargs = {
            k.replace('encoder_', ''): v for k, v in kwargs.items() if k.startswith('encoder_')
        }
        encoder_kwargs['prepend_inputs_before_encoding'] = True
        decoder_kwargs = {
            k.replace('decoder_', ''): v for k, v in kwargs.items() if k.startswith('decoder_')
        }

        self.encoder = SenseVoiceAudioEncoder(**encoder_kwargs)
        self.decoder = AudioDecoder(**decoder_kwargs)
        self.trainer_callbacks = None
    
    def forward(self, dl_output):
        audio_data = dl_output.get("audio", dl_output).float() # [B, T]
        audio_lens = dl_output.get("audio_lens", dl_output)
        audio_features = dl_output.get("x", dl_output).float()
        audio_features_lengths = dl_output.get("x_lens", dl_output)
        # NOTE make sure audio data is 16khz!
        # extract features 
        # meta_data, audio_features, audio_features_lengths = self.encoder.model.prepare_inputs_16khz_data(audio_data, frontend=self.encoder.frontend, device=audio_data.device, data_len=audio_lens)
        # del audio_data
        # forward features
        out_dict = self.forward_features(audio_features, audio_features_lengths, audio_data)
        return out_dict

    def forward_features(self, audio_features, audio_features_lengths, target_audio):
        """
        Forward pass for training.
        
        Args:
            audio_features (torch.Tensor): Padded fbank features (B, T_feat, D_feat).
            audio_features_lengths (torch.Tensor): Lengths of fbank features (B,).
            target_audio (torch.Tensor): Ground truth audio waveform (B, 1, T_audio).
            
        Returns:
            dict: A dictionary containing the reconstructed audio and the training loss.
        """
        # Encode audio features to get semantic tokens
        encoder_results = self.encoder(
            audio_features, 
            audio_features_lengths, 
            return_text=True  # Not needed for training
        )
        
        # Use de-aggregated features which are at frame-level
        semantic_features = encoder_results.get('deaggregated_features')
        if semantic_features is None:
            raise ValueError("Could not find 'deaggregated_features' in encoder output. "
                             "Ensure VQ and deaggregation are enabled in the encoder.")
            
        # Decode semantic features to reconstruct audio
        decoder_results = self.decoder(semantic_features=semantic_features)
        reconstructed_audio = decoder_results['audio']
            
        # Get VQ losses from encoder
        vq_results = encoder_results.get('vq')
        if vq_results is not None:
            commitment_loss = vq_results.get('penalty', torch.tensor(0.0, device=target_audio.device))
            codebook_loss = vq_results.get('vq/codebook_loss', torch.tensor(0.0, device=target_audio.device))
        else:
            assert False

        return edict({
            "x": reconstructed_audio,
            "codes": encoder_results['vq']['codes'],
            "latents": encoder_results['vq']['latents'],
            "penalty": commitment_loss,
            "vq/codebook_loss": codebook_loss,
        })

def test_taste_v2_model():
    import torchaudio
    import librosa

    # 1. Init model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Use prefixed keyword arguments for sub-modules
    model_kwargs = {
        "encoder_model_card": "/data1/lijiaqi/codebase/TASTE-SpokenLM/STAGE1_TRAIN/storage/pretrained_models/SenseVoiceSmall",
        # "encoder_model_code_dir": "/data1/lijiaqi/codebase/CSDs/zero-shot-tts-training/zero_shot_tts_training/realtime_communication/taste_v2/customized_sensevoice/model.py",
        # Add any decoder-specific arguments here with 'decoder_' prefix if needed
    }

    model = TasteV2Model(**model_kwargs).to(device)
    model.eval()

    # 2. Prepare inputs from a real audio file
    audio_fpath = "/data1/lijiaqi/codebase/TASTE-SpokenLM/STAGE1_TRAIN/storage/pretrained_models/SenseVoiceSmall/example/en.mp3"
    
    # Use encoder's extract_feature method to get correct fbank features
    audio_features, audio_features_lengths = model.encoder.extract_feature([audio_fpath])
    audio_features = audio_features.to(device)
    audio_features_lengths = audio_features_lengths.to(device)

    # Load the ground truth audio
    target_audio, sr = librosa.load(audio_fpath, sr=16000, mono=True)
    target_audio = torch.from_numpy(target_audio).float().unsqueeze(0).unsqueeze(0).to(device)

    # 3. Run forward pass
    with torch.no_grad():
        results = model.forward_features(audio_features, audio_features_lengths, target_audio)
    
    # 4. Print results
    print("Test Results:")
    print(f"  - Commitment Loss: {results.penalty.item():.4f}")
    print(f"  - Codebook Loss: {results['vq/codebook_loss'].item():.4f}")
    print(f"  - Reconstructed Audio Shape: {results.x.shape}")
    
    # 5. Save reconstructed audio for inspection
    torchaudio.save("test_reconstructed.wav", results.x.cpu(), 16000)
    print("\nSaved reconstructed audio to test_reconstructed.wav")


if __name__ == "__main__":
    test_taste_v2_model()
