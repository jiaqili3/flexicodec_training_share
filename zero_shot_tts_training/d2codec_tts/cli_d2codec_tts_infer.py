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
import numpy as np
from typing import Dict, Optional, Tuple
from functools import partial
import torch.nn.functional as F

# Add paths for imports
# This allows importing from the project root
# sys.path.append(Path(__file__).resolve().parent.parent.parent.parent.as_posix())

# Import the model classes and tokenizer
sys.path.append('/data1/lijiaqi/codebase/CSDs/zero-shot-tts-training')
sys.path.append('/data1/lijiaqi/codebase/CSDs/zero-shot-tts-training/zero_shot_tts_training')
from zero_shot_tts_training.d2codec_tts.llm_duration import TransformerLMWrapper
from zero_shot_tts_training.tools.whisper_tokenize import text2idx
from zero_shot_tts_training.voicebox.d2codec_voicebox import VoiceboxWrapper
from dualcodec.model_tts.voicebox.vocoder_model import get_vocos_model_spectrogram, mel_to_wav_vocos


import sys
sys.path.append('/data1/lijiaqi/codebase/TS3Codec')

from audio_codec.train.feature_extractors import FBankGen
feature_extractor = FBankGen(sr=16000)

def load_model_from_config(config_path: str, checkpoint_path: Optional[str] = None, device: str = 'cuda'):
    """Load TransformerLMWrapper model from yaml config"""
    print(f"Loading model config from: {config_path}")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        raise ValueError(f"Error loading config file: {e}")
    
    if 'model' not in config or 'config' not in config['model']:
        raise ValueError("Config file must contain 'model.config' section")
    
    model_config = config['model']['config']
    print("Creating TransformerLMWrapper model...")
    # Replace /modelblob paths with /mnt/wus2/models in config
    for key, value in model_config.items():
        if isinstance(value, str) and '/modelblob' in value:
            model_config[key] = value.replace('/modelblob', '/mnt/wus2/models')
    
    model = TransformerLMWrapper(**model_config)
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))
    
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=True)
    if missing_keys:
        print(f"Warning: Missing keys in checkpoint: {missing_keys[:5]}{'...' if len(missing_keys) > 5 else ''}")
    if unexpected_keys:
        print(f"Warning: Unexpected keys in checkpoint: {unexpected_keys[:5]}{'...' if len(unexpected_keys) > 5 else ''}")
    
    model.eval()
    try:
        model = model.to(device)
    except Exception as e:
        print(f"Error moving model to device {device}: {e}. Falling back to CPU.")
        device = 'cpu'
        model = model.to(device)
    
    return model


def load_voicebox_model_from_config(config_path: str, checkpoint_path: str, device: str = 'cuda'):
    """Load VoiceboxWrapper model from yaml config"""
    print(f"Loading Voicebox model config from: {config_path}")
    
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
    
    print(f"Loading Voicebox checkpoint from: {checkpoint_path}")
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


def prepare_text_tokens(prompt_text: str, target_text: str, language: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Prepare text tokens for inference using Whisper tokenizer.
    Concatenates prompt_text and target_text before tokenization."""
    combined_text = prompt_text + " " + target_text
    print(combined_text)
    tokens = text2idx(combined_text, language=language)
    text_tokens = torch.tensor([tokens], dtype=torch.long)
    text_lengths = torch.tensor([len(tokens)])
    return text_tokens, text_lengths


def extract_reference_features(
    model: TransformerLMWrapper, 
    ref_audio_path: str, 
    device: str = 'cuda'
) -> Dict[str, torch.Tensor]:
    """Extract reference audio features using dualcodec"""
    print(f"Processing reference audio: {ref_audio_path}")
    
    ref_audio, sr = torchaudio.load(ref_audio_path)
    
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        ref_audio = resampler(ref_audio)
    
    if ref_audio.shape[0] > 1:
        ref_audio = ref_audio.mean(dim=0, keepdim=True)
    
    ref_audio = ref_audio.to(device)
    
    # Extract mel features using FBankExtractor
    mel_features, _ = feature_extractor.extract_fbank(ref_audio.cpu())
    mel_features = mel_features.to(device)  # Add batch dimension: [1, T, D]
    
    # Calculate x_lens 
    x_lens = torch.tensor([mel_features.shape[1]], dtype=torch.long).to(device)
    
    with torch.no_grad():
        dualcodec_output = model._extract_dualcodec_features(
            ref_audio, mel=mel_features, x_lens=x_lens, sample_rate=16000
        )
    
    return {
        'semantic_codes': dualcodec_output['semantic_codes'],
        'speech_token_len': dualcodec_output['speech_token_len'],
        'token_lengths': dualcodec_output['token_lengths']
    }


@torch.inference_mode()
def infer_tts(
    model: TransformerLMWrapper,
    text: str,
    language: str,
    ref_audio_path: Optional[str] = None,
    ref_text: Optional[str] = None,
    device: str = 'cuda',
    beam_size: int = 1,
    top_k: int = 25,
    temperature: float = 1.0,
    max_token_text_ratio: float = 20.0,
    min_token_text_ratio: float = 0.0,
    predict_duration: bool = True,
    voicebox_model: Optional[VoiceboxWrapper] = None,
    vocoder_decode_func=None,
    n_timesteps: int = 30,
    cfg: float = 2.0,
    rescale_cfg: float = 0.75,
) -> Dict[str, Tuple[torch.Tensor, int]]:
    """Perform TTS inference"""
    
    # Always assume prompt text is not None, use empty string if ref_text is None
    prompt_text = ref_text if ref_text else ""
    text_tokens, text_lengths = prepare_text_tokens(prompt_text, text, language)
    text_tokens = text_tokens.to(device)
    text_lengths = text_lengths.to(device)
    
    prompt_text_tokens = None
    prompt_text_lengths = torch.tensor([0]).to(device)
    # prompt_speech_token = torch.zeros(1, 0, dtype=torch.long).to(device)
    # prompt_speech_token_len = torch.tensor([0]).to(device)
    # prompt_token_lengths = None

    prompt_mel = None
    assert os.path.exists(ref_audio_path)
    ref_features = extract_reference_features(model, ref_audio_path, device)
    prompt_speech_token = ref_features['semantic_codes'].squeeze(1)
    prompt_speech_token_len = ref_features['speech_token_len']
    prompt_token_lengths = ref_features.get('token_lengths')
        
    if voicebox_model:
        print("Creating prompt from reference audio for Voicebox...")
        prompt_audio, sr = torchaudio.load(ref_audio_path)
        if sr != 16000:
            prompt_audio = torchaudio.transforms.Resample(sr, 16000)(prompt_audio)
        if prompt_audio.shape[0] > 1:
            prompt_audio = prompt_audio.mean(dim=0, keepdim=True)
        prompt_audio = prompt_audio.to(device)
        prompt_mel = voicebox_model._extract_mel_features(prompt_audio)
    
    print("Generating speech tokens...")
    
    result = model.transformer_lm.inference(
        text=text_tokens,
        text_len=text_lengths,
        prompt_text=None,
        prompt_text_len=None,
        prompt_speech_token=prompt_speech_token,
        prompt_speech_token_len=prompt_speech_token_len,
        prompt_token_lengths=prompt_token_lengths,
        embedding=None,
        beam_size=beam_size,
        top_k=top_k,
        temperature=temperature,
        max_token_text_ratio=max_token_text_ratio,
        min_token_text_ratio=min_token_text_ratio,
        predict_duration=predict_duration,
    )
    
    speech_tokens = result['speech_tokens']
    duration_classes = result.get('duration_classes', None)
    
    print(f"Generated {speech_tokens.shape[1]} speech tokens")

    outputs = {}

    # 1. Decode with DualCodec
    print("Decoding speech tokens to audio using DualCodec...")
    if duration_classes is None:
        print("Warning: Duration classes not predicted, skipping DualCodec decoding. Try running with --predict_duration.")
    else:
        assert duration_classes.shape == speech_tokens.shape, "Duration classes and speech tokens have different shapes"

        if hasattr(model.dualcodec_model, 'decode_from_codes'):
            decoded_audio_dc = model.dualcodec_model.decode_from_codes(
                semantic_codes=speech_tokens.unsqueeze(1),
                acoustic_codes=None,
                token_lengths=duration_classes,
            )
            outputs['dualcodec'] = (decoded_audio_dc.cpu().squeeze(), 16000)
        else:
            print("Warning: The dualcodec model does not have a `decode_from_codes` method. Skipping DualCodec decoding.")
    
    # 2. Decode with Voicebox if available
    if voicebox_model and vocoder_decode_func:
        print("Decoding speech tokens to audio using Voicebox...")

        expanded_gen_tokens = speech_tokens
        if duration_classes is not None:
            print(f"Expanding generated speech tokens using predicted durations... Original length: {speech_tokens.shape[1]}")
            assert speech_tokens.shape == duration_classes.shape, \
                f"Shape mismatch: speech_tokens {speech_tokens.shape}, duration_classes {duration_classes.shape}"
            
            # De-aggregate/expand speech tokens based on duration classes
            expanded_gen_tokens = torch.repeat_interleave(speech_tokens[0], duration_classes[0]).unsqueeze(0)
            print(f"Expanded generated speech tokens length: {expanded_gen_tokens.shape[1]}")

        # Expand prompt tokens if their durations are available
        expanded_prompt_tokens = prompt_speech_token
        if prompt_token_lengths is not None and prompt_speech_token.shape[1] > 0:
            print(f"Expanding prompt speech tokens using their durations... Original length: {prompt_speech_token.shape[1]}")
            assert prompt_speech_token.shape == prompt_token_lengths.shape, \
                f"Shape mismatch: prompt_speech_token {prompt_speech_token.shape}, prompt_token_lengths {prompt_token_lengths.shape}"
            expanded_prompt_tokens = torch.repeat_interleave(prompt_speech_token[0], prompt_token_lengths[0]).unsqueeze(0)
            print(f"Expanded prompt speech tokens length: {expanded_prompt_tokens.shape[1]}")

        # Concatenate prompt tokens with expanded generated tokens for Voicebox conditioning
        full_speech_tokens = torch.cat([expanded_prompt_tokens, expanded_gen_tokens], dim=1)
        print(f"Total tokens for Voicebox (prompt + generated): {full_speech_tokens.shape[1]}")

        voicebox_internal_model = voicebox_model.voicebox_model
        
        cond_feature = voicebox_internal_model.cond_emb(full_speech_tokens)
        cond_feature = F.interpolate(
            cond_feature.transpose(1, 2),
            scale_factor=voicebox_internal_model.cond_scale_factor,
        ).transpose(1, 2)
        
        predicted_mel = voicebox_internal_model.reverse_diffusion(
            cond=cond_feature,
            prompt=prompt_mel,
            n_timesteps=n_timesteps,
            cfg=cfg,
            rescale_cfg=rescale_cfg,
        )

        # 5. Vocode mel to wav
        print("Vocoding generated mel spectrogram...")
        decoded_audio_vb = vocoder_decode_func(predicted_mel.transpose(1, 2))
        outputs['voicebox'] = (decoded_audio_vb.cpu().squeeze(), 24000)
        
    return outputs


def main():
    parser = argparse.ArgumentParser(description="TTS Inference using DualCodec and TransformerLM")
    

    # model_config_path = '/data1/lijiaqi/codebase/CSDs/application/d2codec_tts/d2codec_tts_b2600_ga1.yaml'
    # ckpt_path = '/mnt/wus2/models/projects/lijiaqi_csd/default/e2tts-dialogue/csd-job-d2codec_tts_b2600_ga1/f12b91f7-d72a-4b9d-b803-e34b49c6a970/models/latest_checkpoint/model.checkpoint'
    # voicebox_config_path = '/data1/lijiaqi/codebase/CSDs/application/d2codec_voicebox/voicebox_300M_b2600_ga1.yaml'
    # voicebox_ckpt_path = '/mnt/wus2/models/projects/lijiaqi_csd/default/e2tts-dialogue/csd-job-voicebox_300M_b2600_ga1/e8cb9859-70a1-46be-b361-53a034502f72/models/checkpoint5/model.checkpoint'
    # model_config_path = '/data1/lijiaqi/codebase/CSDs/application/d2codec_tts/d2codec_tts_b2600_ga1_duration_cond_fsq.yaml'
    # ckpt_path = '/mnt/wus2/models/projects/lijiaqi_csd/default/e2tts-dialogue/csd-job-d2codec_tts_b2600_ga1_duration_cond_fsq/8505e22d-e667-4f1f-a691-30ee7eb044c3/models/checkpoint26/model.checkpoint'
    # voicebox_config_path = '/data1/lijiaqi/codebase/CSDs/application/d2codec_voicebox/voicebox_300M_b1200_ga3_fsq.yaml'
    # voicebox_ckpt_path = '/mnt/wus2/models/projects/lijiaqi_csd/default/e2tts-dialogue/csd-job-voicebox_300M_b1200_ga3_fsq/5c602a67-39b5-4b48-8e3d-b99e769acba8/models/latest_checkpoint/model.checkpoint'

    model_config_path = '/data1/lijiaqi/codebase/CSDs/application/d2codec_tts/d2codec_tts_b2600_ga1_duration_cond_12hz_d2codec.yaml'
    ckpt_path = '/mnt/wus2/models/projects/lijiaqi_csd/default/e2tts-dialogue/csd-job-d2codec_tts_b2600_ga1_duration_cond_12hz_d2codec/d68e4c3d-7d3d-4ee2-bd0b-0c8790feba78/models/latest_checkpoint/model.checkpoint'
    voicebox_config_path = '/data1/lijiaqi/codebase/CSDs/application/d2codec_voicebox/voicebox_300M_b1200_ga3_12hz_d2codec.yaml'
    voicebox_ckpt_path = '/mnt/wus2/models/projects/lijiaqi_csd/default/e2tts-dialogue/csd-job-voicebox_300M_b1200_ga3_12hz_d2codec/8e3c929e-d43f-4b9b-915d-2ea9769cdd27/models/latest_checkpoint/model.checkpoint'

    # Ported defaults from ditar_inference.py
    default_text = "However, these models are typically trained with high temporal feature resolution and short audio windows, which limits their efficiency and introduces bias when applied to long form audio!"
    default_ref_audio = '/data1/lijiaqi/codebase/CSDs/zero-shot-tts-training/example_prompts/3.wav'
    default_ref_text = 'produced the block books, which were the immediate predecessors of the true printed book.'

    parser.add_argument("--config", type=str, default=model_config_path, help="Path to model config YAML file")
    parser.add_argument("--checkpoint", type=str, default=ckpt_path, help="Path to model checkpoint")
    parser.add_argument("--voicebox_config", type=str, default=voicebox_config_path, help="Path to voicebox model config YAML file")
    parser.add_argument("--voicebox_checkpoint", type=str, default=voicebox_ckpt_path, help="Path to voicebox model checkpoint")
    parser.add_argument("--text", type=str, default=default_text, help="Text to synthesize")
    parser.add_argument("--language", type=str, default="en", help="Language for whisper tokenization")
    parser.add_argument("--ref_audio", type=str, default=default_ref_audio, help="Reference audio file for voice cloning")
    parser.add_argument("--ref_text", type=str, default=default_ref_text, help="Reference text for reference audio")
    parser.add_argument("--output", type=str, default=f"d2codec_tts_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav", help="Output audio file path")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device for inference")
    # Add other inference parameters from infer_tts
    parser.add_argument("--beam_size", type=int, default=1, help="Beam size for decoding")
    parser.add_argument("--top_k", type=int, default=25, help="Top-k sampling parameter")
    parser.add_argument("--temperature", type=float, default=0.9, help="Sampling temperature")
    parser.add_argument("--max_token_text_ratio", type=float, default=20.0)
    parser.add_argument("--min_token_text_ratio", type=float, default=0.0)
    parser.add_argument("--no_predict_duration", action="store_false", dest="predict_duration", help="Disable duration prediction")
    # Voicebox inference parameters
    parser.add_argument("--n_timesteps", type=int, default=30, help="Number of diffusion timesteps")
    parser.add_argument("--cfg", type=float, default=2.0, help="Classifier-free guidance scale")
    parser.add_argument("--rescale_cfg", type=float, default=0.75, help="Rescaling factor for CFG")
    parser.add_argument("--disable_second_stage", action="store_true", default=False, help="Disable second stage (Voicebox) and use DualCodec decoder directly.")

    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 50)
    print("DualCodec TTS Inference")
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print("=" * 50)
    
    model = load_model_from_config(args.config, args.checkpoint, args.device)
    print("Stage 1 model loaded successfully!")
    
    if not args.disable_second_stage:
        voicebox_model = load_voicebox_model_from_config(args.voicebox_config, args.voicebox_checkpoint, args.device)
        print("Stage 2 (Voicebox) model loaded successfully!")

        vocoder_decode_func, _ = load_vocoder(args.device)
        print("Vocoder loaded successfully!")
    else:
        voicebox_model = None
        vocoder_decode_func = None
        print("Second stage (Voicebox) is disabled. Using DualCodec decoder.")
    
    output_audios = infer_tts(
        model=model,
        text=args.text,
        language=args.language,
        ref_audio_path=args.ref_audio,
        ref_text=args.ref_text,
        device=args.device,
        beam_size=args.beam_size,
        top_k=args.top_k,
        temperature=args.temperature,
        max_token_text_ratio=args.max_token_text_ratio,
        min_token_text_ratio=args.min_token_text_ratio,
        predict_duration=args.predict_duration,
        voicebox_model=voicebox_model,
        vocoder_decode_func=vocoder_decode_func,
        n_timesteps=args.n_timesteps,
        cfg=args.cfg,
        rescale_cfg=args.rescale_cfg,
    )
    
    base_output_path = Path(args.output)
    saved_files = 0
    for decoder_type, (audio, sr) in output_audios.items():
        output_filename = f"{base_output_path.stem}_{decoder_type}{base_output_path.suffix}"
        output_path = output_dir / output_filename
        print(f"Saving {decoder_type} audio to: {output_path}")
        sf.write(str(output_path), audio.numpy(), sr)
        saved_files += 1

    if saved_files > 0:
        print("✅ TTS synthesis completed!")
    else:
        print("Could not generate any audio. Please check the logs for warnings.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())