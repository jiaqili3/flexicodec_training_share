#!/usr/bin/env python3
# Copyright (c) 2025 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import sys
import yaml
import torch
import torchaudio
import soundfile as sf
from pathlib import Path
from datetime import datetime
from functools import partial
import torch.nn.functional as F

# Add paths for imports
# This allows importing from the project root
sys.path.append(Path(__file__).resolve().parent.parent.parent.parent.as_posix())
# Add path for dualcodec and its dependencies
sys.path.append('/data1/lijiaqi/codebase/TS3Codec')
sys.path.append('/data1/lijiaqi/codebase/CSDs/zero-shot-tts-training')
from zero_shot_tts_training.voicebox.d2codec_voicebox import VoiceboxWrapper
from dualcodec.model_tts.voicebox.vocoder_model import get_vocos_model_spectrogram, mel_to_wav_vocos
from audio_codec.train.feature_extractors import FBankGen

# Global feature extractor for dualcodec
feature_extractor_for_dualcodec = FBankGen(sr=16000)


def load_model_from_config(config_path: str, checkpoint_path: str, device: str = 'cuda'):
    """Load VoiceboxWrapper model from yaml config"""
    print(f"Loading model config from: {config_path}")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model_config = config['model']['config']
    print("Creating VoiceboxWrapper model...")
    # Replace /modelblob paths with machine-specific paths
    for key, value in model_config.items():
        if isinstance(value, str) and '/modelblob' in value:
            model_config[key] = value.replace('/modelblob', '/mnt/wus2/models')
    
    model = VoiceboxWrapper(**model_config)
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))
    
    model.load_state_dict(state_dict, strict=False)
    
    model.eval()
    model = model.to(device)
    
    return model


def load_vocoder(device='cuda'):
    """Load Vocos vocoder and mel extractor"""
    print("Loading Vocos model...")
    vocos_model, mel_model = get_vocos_model_spectrogram()
    vocos_model = vocos_model.to(device)
    infer_vocos = partial(mel_to_wav_vocos, vocos_model)
    return infer_vocos, mel_model


@torch.inference_mode()
def infer_voicebox(
    model: VoiceboxWrapper,
    vocoder_decode_func,
    input_audio_path: str,
    device: str = 'cuda',
    n_timesteps: int = 10,
    cfg: float = 2.0,
    rescale_cfg: float = 0.75,
):
    """Perform inference using Voicebox model"""
    # 1. Load audio and prepare for model
    audio, sr = torchaudio.load(input_audio_path)
    if sr != 16000:
        audio = torchaudio.transforms.Resample(sr, 16000)(audio)
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    audio = audio.to(device)

    # 2. Extract semantic codes for condition from full audio
    print("Extracting semantic codes from ground truth audio...")
    mel_features, _ = feature_extractor_for_dualcodec.extract_fbank(audio.cpu())
    mel_features = mel_features.to(device)
    x_lens = torch.tensor([mel_features.shape[1]], dtype=torch.long, device=device)

    dualcodec_output = model._extract_dualcodec_features(audio, mel=mel_features, x_lens=x_lens)
    cond_codes = dualcodec_output['semantic_codes'].squeeze(1)

    # 3. Create prompt from first 3s of audio
    print("Creating prompt from first 3s of audio...")
    prompt_duration_samples = 3 * 16000
    prompt_audio = audio[:, :prompt_duration_samples]
    
    # 4. Perform inference
    print("Running reverse diffusion...")
    voicebox_model = model.voicebox_model
    
    cond_feature = voicebox_model.cond_emb(cond_codes)
    cond_feature = F.interpolate(
        cond_feature.transpose(1, 2),
        scale_factor=voicebox_model.cond_scale_factor,
    ).transpose(1, 2)
    
    prompt_mel = model._extract_mel_features(prompt_audio)

    predicted_mel = voicebox_model.reverse_diffusion(
        cond=cond_feature,
        prompt=prompt_mel,
        n_timesteps=n_timesteps,
        cfg=cfg,
        rescale_cfg=rescale_cfg,
    )

    # 5. Vocode mel to wav
    print("Vocoding generated mel spectrogram...")
    predicted_audio = vocoder_decode_func(predicted_mel.transpose(1, 2))
    
    return predicted_audio.cpu().squeeze()


def main():
    parser = argparse.ArgumentParser(description="Inference for D2Codec-Voicebox")
    
    # Default paths
    default_config = '/data1/lijiaqi/codebase/CSDs/application/d2codec_voicebox/voicebox_300M_b2600_ga1.yaml'
    default_ckpt = '/mnt/wus2/models/projects/lijiaqi_csd/default/e2tts-dialogue/csd-job-voicebox_300M_b2600_ga1/e8cb9859-70a1-46be-b361-53a034502f72/models/checkpoint5/model.checkpoint'
    
    parser.add_argument("--config", type=str, default=default_config, help="Path to model config YAML file")
    parser.add_argument("--checkpoint", type=str, default=default_ckpt, help="Path to model checkpoint")
    parser.add_argument("--input_audio", type=str, default='/data1/lijiaqi/codebase/CSDs/2486365921931244890.wav', help="Path to ground truth audio file")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device for inference")
    
    # Inference parameters
    parser.add_argument("--n_timesteps", type=int, default=30, help="Number of diffusion timesteps")
    parser.add_argument("--cfg", type=float, default=2.0, help="Classifier-free guidance scale")
    parser.add_argument("--rescale_cfg", type=float, default=0.75, help="Rescaling factor for CFG")

    args = parser.parse_args()
    
    # Adjust config path to be relative to the project root
    if not args.config.startswith('/'):
        args.config = os.path.join(Path(__file__).resolve().parent.parent.parent.parent, args.config)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_filename = f"voicebox_output_{Path(args.input_audio).stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
    output_path = output_dir / output_filename
    
    print("=" * 50)
    print("D2Codec-Voicebox Inference")
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print("=" * 50)
    
    model = load_model_from_config(args.config, args.checkpoint, args.device)
    vocoder_decode_func, _ = load_vocoder(args.device)
    
    audio = infer_voicebox(
        model=model,
        vocoder_decode_func=vocoder_decode_func,
        input_audio_path=args.input_audio,
        device=args.device,
        n_timesteps=args.n_timesteps,
        cfg=args.cfg,
        rescale_cfg=args.rescale_cfg
    )
    
    print(f"Saving audio to: {output_path}")
    sf.write(str(output_path), audio.numpy(), 24000) # Vocos output is 24kHz
    print(f"✅ Voicebox inference completed! Audio saved to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main()) 