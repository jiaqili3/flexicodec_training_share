#!/usr/bin/env python3
# Copyright (c) 2025 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import sys
import yaml
import json
import torch
import torchaudio
import soundfile as sf
from pathlib import Path
from datetime import datetime
import numpy as np
from typing import Dict, Optional, Tuple, List
from functools import partial
import torch.nn.functional as F
import librosa
import shutil
from tqdm import tqdm
import random
import time
import torch.multiprocessing as mp
from queue import Queue
import threading

# Add paths for imports
sys.path.append('/data1/lijiaqi/codebase/CSDs/zero-shot-tts-training')
sys.path.append('/data1/lijiaqi/codebase/CSDs/zero-shot-tts-training/zero_shot_tts_training')
from zero_shot_tts_training.d2codec_tts.llm_duration import TransformerLMWrapper
from zero_shot_tts_training.tools.whisper_tokenize import text2idx
from zero_shot_tts_training.voicebox.d2codec_voicebox import VoiceboxWrapper
from dualcodec.model_tts.voicebox.vocoder_model import get_vocos_model_spectrogram, mel_to_wav_vocos

# Import G2P phonemizer
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'tools'))
from g2p_phonemizer import g2p_phonemizer_en

sys.path.append('/data1/lijiaqi/codebase/CSDs/application/ts3codec/TS3Codec/')
from audio_codec.train.feature_extractors import FBankGen

# Global feature extractor for dualcodec
feature_extractor_for_dualcodec = FBankGen(sr=16000)


USE_G2P = False  # Set to True if you want to use G2P phonemization for English text
PARTIAL_G2P = False  # Set to True if you want to use partial G2P phonemization

INFERENCE_FRAMERATE = 0.91
print(f'Inference framerate set to: {INFERENCE_FRAMERATE}')

# Global feature extractor for dualcodec
feature_extractor_for_dualcodec = FBankGen(sr=16000)

def write_text_to_file(text, path):
    """Write text to file"""
    try:
        with open(path, 'w') as file:
            file.write(text)
        print(f"Text successfully written to {path}")
    except Exception as e:
        print(f"An error occurred: {e}")

def load_dialog_test_data(json_path: str) -> List[Dict]:
    """Load dialog test data from JSON file"""
    print(f"Loading dialog test data from: {json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    print(f"Loaded {len(test_data)} test cases")
    
    # Validate and prepare test cases
    valid_test_cases = []
    for item in test_data:
        # Check required fields
        if not all(key in item for key in ['key', 'text', 'audio_prompt_spk1', 'audio_prompt_spk2']):
            print(f"Skipping item missing required fields: {item}")
            continue
        
        # Validate text file path
        text_file_path = item['text']
        if not os.path.exists(text_file_path):
            print(f"Warning: Text file not found: {text_file_path}")
            continue
        
        # Load and parse the dialog text file
        try:
            with open(text_file_path, 'r', encoding='utf-8') as f:
                dialog_content = f.read().strip()
            
            # Parse dialog content
            dialog_turns = []
            current_speaker = None
            current_text = ""
            
            for line in dialog_content.split('\n'):
                if not line.strip():
                    continue
                
                # Parse format: speaker|text|start_time|end_time
                parts = line.split('|')
                if len(parts) >= 2:
                    speaker = parts[0]
                    text = parts[1]
                    
                    if speaker in ['1', '2']:
                        dialog_turns.append({
                            'speaker': speaker,
                            'text': text.strip()
                        })
            
            if not dialog_turns:
                print(f"Warning: No valid dialog turns found in {text_file_path}")
                continue
            
            # Convert dialog to α/β format
            formatted_dialog = ""
            for turn in dialog_turns:
                if turn['speaker'] == '1':
                    formatted_dialog += f"α {turn['text']} "
                elif turn['speaker'] == '2':
                    formatted_dialog += f"β {turn['text']} "
            
            formatted_dialog = formatted_dialog.strip()
            
        except Exception as e:
            print(f"Error reading dialog file {text_file_path}: {e}")
            continue
        
        # Validate audio prompt files
        audio_prompt_spk1 = item['audio_prompt_spk1']
        audio_prompt_spk2 = item['audio_prompt_spk2']
        
        if not os.path.exists(audio_prompt_spk1):
            print(f"Warning: Audio prompt for speaker 1 not found: {audio_prompt_spk1}")
            continue
            
        if not os.path.exists(audio_prompt_spk2):
            print(f"Warning: Audio prompt for speaker 2 not found: {audio_prompt_spk2}")
            continue
        
        valid_test_cases.append({
            'key': item['key'],
            'dialog_text': formatted_dialog,
            'dialog_turns': dialog_turns,
            'audio_prompt_spk1': audio_prompt_spk1,
            'audio_prompt_spk2': audio_prompt_spk2,
            'original_text_file': text_file_path
        })
    
    print(f"Found {len(valid_test_cases)} valid test cases")
    return valid_test_cases

def load_librispeech_test_data(json_path: str, testset_root: str, gt_audio_root: str) -> List[Dict]:
    """Load LibriSpeech test data from JSON file"""
    print(f"Loading test data from: {json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    print(f"Loaded {len(test_data)} test cases")
    
    # Validate and prepare test cases
    valid_test_cases = []
    for item in test_data:
        # Check required fields
        if not all(key in item for key in ['key', 'text', 'audio', 'text_ref']):
            print(f"Skipping item missing required fields: {item}")
            continue
            
        # Construct full reference audio path
        ref_audio_path = os.path.join(testset_root, item['audio'])
        if not os.path.exists(ref_audio_path):
            print(f"Warning: Reference audio file not found: {ref_audio_path}")
            continue
        
        # Construct ground truth audio path
        # For key "1188-133604-0001", GT audio should be at clean_original/1188/133604/1188-133604-0001.wav
        key_parts = item['key'].split('-')
        if len(key_parts) != 3:
            print(f"Warning: Invalid key format: {item['key']}")
            continue
        
        speaker_id, chapter_id, utterance_id = key_parts
        gt_audio_path = os.path.join(gt_audio_root, 'clean_original', speaker_id, chapter_id, f"{item['key']}.wav")
        
        if not os.path.exists(gt_audio_path):
            print(f"Warning: Ground truth audio file not found: {gt_audio_path}")
            continue
            
        valid_test_cases.append({
            'key': item['key'],
            'text': item['text'],
            'ref_audio_path': ref_audio_path,
            'gt_audio_path': gt_audio_path,
            'text_ref': item['text_ref']
        })
    
    print(f"Found {len(valid_test_cases)} valid test cases")
    return valid_test_cases

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
            model_config[key] = value.replace('/modelblob', '/mnt/scus/models')
    
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
            model_config[key] = value.replace('/modelblob', '/mnt/scus/models')
    
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
    vocos_model, mel_model = get_vocos_model_spectrogram('/data1/lijiaqi/data/vocos_emilia.safetensors')
    vocos_model = vocos_model.to(device)
    infer_vocos = partial(mel_to_wav_vocos, vocos_model)
    return infer_vocos, mel_model

def prepare_text_tokens(prompt_text: str, target_text: str, language: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Prepare text tokens for inference using Whisper tokenizer."""
    combined_text = 'α ' + prompt_text + ". β " + target_text + ' α ' + 'Wow.' + " β " + prompt_text
    print(f"Preparing text tokens for: {combined_text}")
    if USE_G2P:
        if PARTIAL_G2P:
            tokens = text2idx(prompt_text, language=language, g2p_prob=0.0) + text2idx(target_text, language=language, g2p_prob=1.0)[1:]
        else:
            tokens = text2idx(combined_text, language=language, g2p_prob=1.0)
    else:
        tokens = text2idx(combined_text, language=language, g2p_prob=0.0)
    text_tokens = torch.tensor([tokens], dtype=torch.long)
    text_lengths = torch.tensor([len(tokens)])
    return text_tokens, text_lengths

def extract_reference_features(
    model: TransformerLMWrapper, 
    ref_audio_path: str, 
    device: str = 'cuda'
) -> Dict[str, torch.Tensor]:
    """Extract reference audio features using dualcodec"""
    ref_audio, sr = torchaudio.load(ref_audio_path)
    
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        ref_audio = resampler(ref_audio)
    
    if ref_audio.shape[0] > 1:
        ref_audio = ref_audio.mean(dim=0, keepdim=True)
    
    ref_audio = ref_audio.to(device)
    
    # Extract mel features using FBankExtractor
    mel_features, _ = feature_extractor_for_dualcodec.extract_fbank(ref_audio.cpu())
    mel_features = mel_features.to(device)
    
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
def infer_dialog_tts(
    model: TransformerLMWrapper,
    dialog_turns: List[Dict],
    dialog_text: str,
    language: str,
    audio_prompt_spk1: str,
    audio_prompt_spk2: str,
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
    decoder_type: str = 'voicebox',
    duration_top_k: int = 1,
) -> Tuple[torch.Tensor, int]:
    """Perform TTS inference for dialog with multiple speakers"""
    
    # Process dialog by turns
    generated_audio_segments = []
    
    for i, turn in enumerate(dialog_turns):
        speaker = turn['speaker']
        text = turn['text']
        
        # Select audio prompt based on speaker
        if speaker == '1':
            ref_audio_path = audio_prompt_spk1
            # Create simple prompt text for speaker 1
            prompt_text = "Hello."
        else:  # speaker == '2'
            ref_audio_path = audio_prompt_spk2
            # Create simple prompt text for speaker 2
            prompt_text = "Hi there."
        
        print(f"Processing turn {i+1}/{len(dialog_turns)}: Speaker {speaker} - {text[:50]}...")
        
        # Generate audio for this turn
        audio_segment, sample_rate = infer_tts_single(
            model=model,
            text=text,
            language=language,
            ref_audio_path=ref_audio_path,
            ref_text=prompt_text,
            device=device,
            beam_size=beam_size,
            top_k=top_k,
            temperature=temperature,
            max_token_text_ratio=max_token_text_ratio,
            min_token_text_ratio=min_token_text_ratio,
            predict_duration=predict_duration,
            voicebox_model=voicebox_model,
            vocoder_decode_func=vocoder_decode_func,
            n_timesteps=n_timesteps,
            cfg=cfg,
            rescale_cfg=rescale_cfg,
            decoder_type=decoder_type,
            duration_top_k=duration_top_k,
        )
        
        generated_audio_segments.append(audio_segment)
    
    # Concatenate all audio segments with small pauses
    if len(generated_audio_segments) > 0:
        pause_duration = int(0.5 * sample_rate)  # 0.5 second pause
        pause_audio = torch.zeros(pause_duration)
        
        final_audio = generated_audio_segments[0]
        for segment in generated_audio_segments[1:]:
            final_audio = torch.cat([final_audio, pause_audio, segment])
        
        return final_audio, sample_rate
    else:
        # Return empty audio if no segments
        return torch.zeros(sample_rate), sample_rate

@torch.inference_mode()
def infer_tts_single(
    model: TransformerLMWrapper,
    text: str,
    language: str,
    ref_audio_path: str,
    ref_text: str,
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
    decoder_type: str = 'voicebox',
    duration_top_k: int = 1,
) -> Tuple[torch.Tensor, int]:
    min_token_text_ratio: float = 0.0,
    predict_duration: bool = True,
    voicebox_model: Optional[VoiceboxWrapper] = None,
    vocoder_decode_func=None,
    n_timesteps: int = 30,
    cfg: float = 2.0,
    rescale_cfg: float = 0.75,
    decoder_type: str = 'voicebox',
    duration_top_k: int = 1,
) -> Tuple[torch.Tensor, int]:
    """Perform TTS inference for a single test case"""
    
    # Prepare text tokens
    prompt_text = ref_text if ref_text else ""
    text_tokens, text_lengths = prepare_text_tokens(prompt_text, text, language)
    text_tokens = text_tokens.to(device)
    text_lengths = text_lengths.to(device)
    
    # Extract reference features
    ref_features = extract_reference_features(model, ref_audio_path, device)
    prompt_speech_token = ref_features['semantic_codes'].squeeze(1)
    prompt_speech_token_len = ref_features['speech_token_len']
    prompt_token_lengths = ref_features.get('token_lengths')

    if prompt_token_lengths is None:
        predict_duration = False
        print("Warning: No token lengths provided for prompt, duration prediction will be disabled.")
    
    # Prepare prompt mel for Voicebox if needed
    prompt_mel = None
    if voicebox_model and decoder_type == 'voicebox':
        prompt_audio, sr = torchaudio.load(ref_audio_path)
        if sr != 16000:
            prompt_audio = torchaudio.transforms.Resample(sr, 16000)(prompt_audio)
        if prompt_audio.shape[0] > 1:
            prompt_audio = prompt_audio.mean(dim=0, keepdim=True)
        prompt_audio = prompt_audio.to(device)
        prompt_mel = voicebox_model._extract_mel_features(prompt_audio)
        
        use_decoder_latent = voicebox_model.use_decoder_latent
        use_decoder_latent_before_agg = voicebox_model.use_decoder_latent_before_agg
        decoder_latent_pass_transformer = voicebox_model.decoder_latent_pass_transformer

        if use_decoder_latent_before_agg:

            ref_mel_features, _ = feature_extractor_for_dualcodec.extract_fbank(prompt_audio.cpu())
            ref_mel_features = ref_mel_features.to(device)
            ref_x_lens = torch.tensor([ref_mel_features.shape[1]], dtype=torch.long, device=device)

            model.dualcodec_model.similarity_threshold = INFERENCE_FRAMERATE

            ref_dualcodec_output = voicebox_model._extract_dualcodec_features(prompt_audio, mel=ref_mel_features, x_lens=ref_x_lens, manual_threshold=INFERENCE_FRAMERATE)
            prompt_mel = ref_dualcodec_output['decoder_latent_before_agg'].transpose(1,2)
            if decoder_latent_pass_transformer:
                prompt_mel = model.dualcodec_model.bottleneck_transformer(ref_dualcodec_output['decoder_latent_before_agg']).transpose(1,2)



    # Generate speech tokens
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
        duration_top_k=duration_top_k,
        inference_framerate=INFERENCE_FRAMERATE,
    )
    
    speech_tokens = result['speech_tokens']
    duration_classes = result.get('duration_classes', None)
    
    # Decode to audio
    if decoder_type == 'dualcodec':
        # if duration_classes is None:
        #     raise ValueError("Duration classes not predicted, cannot use DualCodec decoding")
        
        if hasattr(model.dualcodec_model, 'decode_from_codes'):
            decoded_audio = model.dualcodec_model.decode_from_codes(
                semantic_codes=speech_tokens.unsqueeze(1),
                acoustic_codes=None,
                token_lengths=duration_classes,
            )
            return decoded_audio.cpu().squeeze(), 16000
        else:
            raise ValueError("DualCodec model does not have decode_from_codes method")
    
    elif decoder_type == 'voicebox' and voicebox_model and vocoder_decode_func:
        # Expand tokens using duration classes
        expanded_gen_tokens = speech_tokens
        if duration_classes is not None:
            expanded_gen_tokens = torch.repeat_interleave(speech_tokens[0], duration_classes[0]).unsqueeze(0)
        
        # Expand prompt tokens
        expanded_prompt_tokens = prompt_speech_token
        if prompt_token_lengths is not None and prompt_speech_token.shape[1] > 0:
            expanded_prompt_tokens = torch.repeat_interleave(prompt_speech_token[0], prompt_token_lengths[0]).unsqueeze(0)
        
        # Concatenate for Voicebox conditioning
        full_speech_tokens = torch.cat([expanded_prompt_tokens, expanded_gen_tokens], dim=1)
        
        voicebox_internal_model = voicebox_model.voicebox_model
        
        cond_feature = voicebox_internal_model.cond_emb(full_speech_tokens)
        cond_feature = F.interpolate(
            cond_feature.transpose(1, 2),
            scale_factor=voicebox_internal_model.cond_scale_factor,
        ).transpose(1, 2)
        if voicebox_model.add_framerate_embedding:
            framerate_idx = voicebox_internal_model.flex_framerate_options.index(INFERENCE_FRAMERATE)
            framerate_emb = voicebox_internal_model.framerate_embedding(torch.tensor(framerate_idx, device=device))
            cond_feature = cond_feature + framerate_emb.unsqueeze(0).unsqueeze(0)
    
        predicted_mel = voicebox_internal_model.reverse_diffusion(
            cond=cond_feature,
            prompt=prompt_mel,
            n_timesteps=n_timesteps,
            cfg=cfg,
            rescale_cfg=rescale_cfg,
        )
        
        if predicted_mel.shape[-1] != 128:
            decoded_audio = model.dualcodec_model.dac.decoder(torch.cat([prompt_mel, predicted_mel], dim=1).transpose(1,2))[...,prompt_audio.shape[-1]:]
            return decoded_audio.cpu().squeeze(0), 16000
        else:
            # Vocode mel to wav
            decoded_audio = vocoder_decode_func(predicted_mel.transpose(1, 2))
            return decoded_audio.cpu().squeeze(), 24000
    
    else:
        raise ValueError(f"Invalid decoder_type: {decoder_type}")

def worker_process(
    gpu_id: int,
    test_cases: List[Dict],
    args,
    save_path: str,
    results_queue: mp.Queue
):
    """Worker process for a single GPU"""
    device = f"cuda:{gpu_id}"
    print(f"Worker {gpu_id}: Starting with {len(test_cases)} test cases on {device}")
    
    try:
        # Load models
        model = load_model_from_config(args.config, args.checkpoint, device)
        
        voicebox_model = None
        vocoder_decode_func = None
        if not args.disable_second_stage:
            voicebox_model = load_voicebox_model_from_config(args.voicebox_config, args.voicebox_checkpoint, device)
            vocoder_decode_func, _ = load_vocoder(device)
        
        # Process test cases
        for i, test_case in enumerate(test_cases):
            try:
                key = test_case['key']
                
                # Check if output already exists and skip if needed
                output_file = os.path.join(save_path, f"{key}.wav")
                if os.path.exists(output_file) and args.skip:
                    print(f"Worker {gpu_id}: Skipping existing file: {output_file}")
                    continue
                
                # Determine if this is dialog or LibriSpeech format
                is_dialog_format = 'dialog_text' in test_case
                
                # Perform TTS inference
                start_time = time.time()
                decoder_type = 'dualcodec' if args.disable_second_stage else 'voicebox'
                
                if is_dialog_format:
                    # Handle dialog format
                    dialog_turns = test_case['dialog_turns']
                    dialog_text = test_case['dialog_text']
                    audio_prompt_spk1 = test_case['audio_prompt_spk1']
                    audio_prompt_spk2 = test_case['audio_prompt_spk2']
                    
                    print(f"Worker {gpu_id}: Processing dialog with {len(dialog_turns)} turns")
                    
                    audio, sample_rate = infer_dialog_tts(
                        model=model,
                        dialog_turns=dialog_turns,
                        dialog_text=dialog_text,
                        language='en',
                        audio_prompt_spk1=audio_prompt_spk1,
                        audio_prompt_spk2=audio_prompt_spk2,
                        device=device,
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
                        decoder_type=decoder_type,
                        duration_top_k=1,
                    )
                    
                    # Save reference audio prompts
                    ref_spk1_dest = os.path.join(save_path, f"{key}_ref_spk1.wav")
                    ref_spk2_dest = os.path.join(save_path, f"{key}_ref_spk2.wav")
                    shutil.copyfile(audio_prompt_spk1, ref_spk1_dest)
                    shutil.copyfile(audio_prompt_spk2, ref_spk2_dest)
                    
                    # Save text information
                    text_content = f"Dialog text: {dialog_text}\n"
                    text_content += f"Original text file: {test_case['original_text_file']}\n"
                    text_content += f"Audio prompt spk1: {audio_prompt_spk1}\n"
                    text_content += f"Audio prompt spk2: {audio_prompt_spk2}\n"
                    text_content += f"Number of turns: {len(dialog_turns)}\n\n"
                    text_content += "Dialog turns:\n"
                    for j, turn in enumerate(dialog_turns):
                        text_content += f"Turn {j+1} (Speaker {turn['speaker']}): {turn['text']}\n"
                    
                else:
                    # Handle LibriSpeech format
                    text = test_case['text']
                    ref_audio_path = test_case['ref_audio_path']
                    gt_audio_path = test_case['gt_audio_path']
                    text_ref = test_case['text_ref']
                    
                    # Process text inputs
                    text_processed = text.strip().replace('"', '')
                    text_ref_processed = text_ref.strip().rstrip('!').rstrip('.').replace('"', '')
                    
                    text_final = text_processed
                    text_ref_final = text_ref_processed
                    
                    audio, sample_rate = infer_tts_single(
                        model=model,
                        text=text_final,
                        language='en',
                        ref_audio_path=ref_audio_path,
                        ref_text=text_ref_final.replace('.', ','),
                        device=device,
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
                        decoder_type=decoder_type,
                        duration_top_k=1,
                    )
                    
                    # Save reference audio to destination with _ref suffix
                    ref_audio_dest = os.path.join(save_path, f"{key}_ref.wav")
                    shutil.copyfile(ref_audio_path, ref_audio_dest)
                    
                    # Save ground truth audio with _gt suffix
                    gt_audio_dest = os.path.join(save_path, f"{key}_gt.wav")
                    shutil.copyfile(gt_audio_path, gt_audio_dest)
                    
                    # Save text information
                    text_content = f"Target: {text}\nReference: {text_ref}\nGT Audio: {gt_audio_path}\nRef Audio: {ref_audio_path}"
                
                inference_time = time.time() - start_time
                
                # Save audio
                torchaudio.save(output_file, audio.reshape(1,-1), sample_rate=sample_rate)
                print(f"Worker {gpu_id}: Saved audio to {output_file}")

                # Calculate metrics
                audio_duration = len(audio) / sample_rate
                rtf = inference_time / audio_duration

                print(f"Worker {gpu_id}: [{i+1}/{len(test_cases)}] {key} - Duration: {audio_duration:.2f}s, Inference: {inference_time:.2f}s, RTF: {rtf:.3f}")

                # Save text information
                with open(os.path.join(save_path, f'{key}.txt'), 'w', encoding='utf-8') as f:
                    f.write(text_content)

                # Report progress
                results_queue.put({
                    'worker_id': gpu_id,
                    'key': key,
                    'duration': audio_duration,
                    'inference_time': inference_time,
                    'rtf': rtf,
                    'status': 'success'
                })
                
            except Exception as e:
                print(f"Worker {gpu_id}: Error processing {key}: {e}")
                results_queue.put({
                    'worker_id': gpu_id,
                    'key': key if 'key' in locals() else 'unknown',
                    'status': 'error',
                    'error': str(e)
                })
    
    except Exception as e:
        print(f"Worker {gpu_id}: Fatal error: {e}")
        results_queue.put({
            'worker_id': gpu_id,
            'status': 'fatal_error',
            'error': str(e)
        })

def main():
    parser = argparse.ArgumentParser(description="SeedTTS Evaluation using DualCodec TTS")
    
    # Model configuration
    # model_config_path = '/data1/lijiaqi/codebase/CSDs/application/d2codec_tts/d2codec_tts_b2600_ga1_duration_cond_fsq.yaml'
    # ckpt_path = '/mnt/wus2/models/projects/lijiaqi_csd/default/e2tts-dialogue/csd-job-d2codec_tts_b2600_ga1_duration_cond_fsq/8505e22d-e667-4f1f-a691-30ee7eb044c3/models/checkpoint26/model.checkpoint'
    # model_config_path = '/data1/lijiaqi/codebase/CSDs/application/d2codec_tts/d2codec_tts_b2600_ga1_duration_cond_fsq_v2.yaml'
    # ckpt_path = '/mnt/wus2/models/projects/lijiaqi_csd/default/e2tts-dialogue/csd-job-d2codec_tts_b2600_ga1_duration_cond_fsq_v2/3be030f1-b8d7-4978-bd15-aad834ed4d3b/models/checkpoint61/model.checkpoint'
    # voicebox_config_path = '/data1/lijiaqi/codebase/CSDs/application/d2codec_voicebox/voicebox_300M_b1200_ga3_fsq.yaml'
    # voicebox_ckpt_path = '/mnt/wus2/models/projects/lijiaqi_csd/default/e2tts-dialogue/csd-job-voicebox_300M_b1200_ga3_fsq/5c602a67-39b5-4b48-8e3d-b99e769acba8/models/checkpoint34/model.checkpoint'
    
    # model_config_path = '/data1/lijiaqi/codebase/CSDs/application/d2codec_tts/d2codec_tts_b2600_ga1_duration_cond_fsq_v2_500M.yaml'
    # ckpt_path = '/mnt/wus2/models/projects/lijiaqi_csd/default/e2tts-dialogue/csd-job-d2codec_tts_b2600_ga1_duration_cond_fsq_v2_500M/a723edf9-bf60-40ad-a0b2-230208f04ebb/models/latest_checkpoint/model.checkpoint'
    # voicebox_config_path = '/data1/lijiaqi/codebase/CSDs/application/d2codec_voicebox/voicebox_300M_b1200_ga3_fsq_v2.yaml'
    # voicebox_ckpt_path = '/mnt/wus2/models/projects/lijiaqi_csd/default/e2tts-dialogue/csd-job-voicebox_300M_b1200_ga3_fsq_v2/4ba2ce4d-b4f4-4c5a-b19a-cf3352a8a560/models/latest_checkpoint/model.checkpoint'
    
    # model_config_path = '/data1/lijiaqi/codebase/CSDs/application/d2codec_tts/d2codec_tts_b2600_ga1_duration_cond_12hz_v3.yaml'
    # ckpt_path = '/mnt/wus2/models/projects/lijiaqi_csd/default/e2tts-dialogue/csd-job-d2codec_tts_b2600_ga1_duration_cond_12hz_v3/01a1a505-3a77-43fe-ad8b-4394b53c707b/models/latest_checkpoint/model.checkpoint'
    # voicebox_config_path = '/data1/lijiaqi/codebase/CSDs/application/d2codec_voicebox/voicebox_300M_b1200_ga3_12hz_v2.yaml'
    # voicebox_ckpt_path = '/mnt/wus2/models/projects/lijiaqi_csd/default/e2tts-dialogue/csd-job-voicebox_300M_b1200_ga3_12hz_v2/42cd59ce-b7b0-4aed-9f5d-d8d1130284b1/models/latest_checkpoint/model.checkpoint'

    # model_config_path = '/data1/lijiaqi/codebase/CSDs/application/d2codec_tts/d2codec_tts_b2600_ga1_duration_cond_6hz_static.yaml'
    # ckpt_path = '/mnt/wus2/models/projects/lijiaqi_csd/default/e2tts-dialogue/csd-job-d2codec_tts_b2600_ga1_duration_cond_6hz_static/58d842b3-695c-4500-9a10-1737e0d0b26f/models/latest_checkpoint/model.checkpoint'
    # voicebox_config_path = '/data1/lijiaqi/codebase/CSDs/application/d2codec_voicebox/voicebox_300M_b1200_ga3_12hz_v2.yaml'
    # voicebox_ckpt_path = '/mnt/wus2/models/projects/lijiaqi_csd/default/e2tts-dialogue/csd-job-voicebox_300M_b1200_ga3_12hz_v2/42cd59ce-b7b0-4aed-9f5d-d8d1130284b1/models/latest_checkpoint/model.checkpoint'
    
    # model_config_path = '/data1/lijiaqi/codebase/CSDs/application/d2codec_tts/d2codec_tts_b2600_ga1_duration_cond_fsq_v2_500M_g2p.yaml'
    # ckpt_path = '/mnt/wus2/models/projects/lijiaqi_csd/default/e2tts-dialogue/csd-job-d2codec_tts_b2600_ga1_duration_cond_fsq_v2_500M_g2p/9410a524-9830-4977-8f5b-8294fc152834/models/latest_checkpoint/model.checkpoint'
    # voicebox_config_path = '/data1/lijiaqi/codebase/CSDs/application/d2codec_voicebox/voicebox_300M_b1200_ga3_fsq_v2.yaml'
    # voicebox_ckpt_path = '/mnt/wus2/models/projects/lijiaqi_csd/default/e2tts-dialogue/csd-job-voicebox_300M_b1200_ga3_fsq_v2/4ba2ce4d-b4f4-4c5a-b19a-cf3352a8a560/models/latest_checkpoint/model.checkpoint'
    
    # model_config_path = '/data1/lijiaqi/codebase/CSDs/application/d2codec_tts/d2codec_tts_b2600_ga1_duration_cond_12hz_d2codec780ksteps.yaml'
    # # ckpt_path = '/mnt/scus/models/projects/lijiaqi_csd/default/e2tts-dialogue/csd-job-d2codec_tts_b2600_ga1_duration_cond_12hz_d2codec/d68e4c3d-7d3d-4ee2-bd0b-0c8790feba78/models/latest_checkpoint/model.checkpoint'
    # ckpt_path = '/mnt/scus/models/projects/lijiaqi_csd/default/e2tts-dialogue/csd-job-d2codec_tts_b2600_ga1_duration_cond_12hz_d2codec780ksteps/2d908ead-4df2-4884-b894-b4e804f773c5/models/latest_checkpoint/model.checkpoint'
    # voicebox_config_path = '/data1/lijiaqi/codebase/CSDs/application/d2codec_voicebox/voicebox_300M_b1200_ga3_12hz_d2codec780ksteps.yaml'
    # voicebox_ckpt_path = '/mnt/scus/models/projects/lijiaqi_csd/default/e2tts-dialogue/csd-job-voicebox_300M_b1200_ga3_12hz_d2codec780ksteps/bbefd476-d766-4862-921f-32c02977d191/models/latest_checkpoint/model.checkpoint'
    
    # model_config_path = '/data1/lijiaqi/codebase/CSDs/application/d2codec_tts/d2codec_tts_b2600_ga1_duration_cond_8hz_d2codec.yaml'
    # ckpt_path = '/mnt/wus2/models/projects/lijiaqi_csd/default/e2tts-dialogue/csd-job-d2codec_tts_b2600_ga1_duration_cond_8hz_d2codec/edcadaf1-fce8-4495-86e8-f84c3375da1a/models/latest_checkpoint/model.checkpoint'
    # voicebox_config_path = '/data1/lijiaqi/codebase/CSDs/application/d2codec_voicebox/voicebox_300M_b1200_ga3_12hz_d2codec780ksteps.yaml'
    # voicebox_ckpt_path = '/mnt/wus2/models/projects/lijiaqi_csd/default/e2tts-dialogue/csd-job-voicebox_300M_b1200_ga3_12hz_d2codec780ksteps/bbefd476-d766-4862-921f-32c02977d191/models/latest_checkpoint/model.checkpoint'


    # model_config_path = '/data1/lijiaqi/codebase/CSDs/application/d2codec_tts/flextts_b2600_ga1_duration_cond_12hz8hz6hz_d2codec780ksteps.yaml'
    # ckpt_path = '/mnt/scus/models/projects/lijiaqi_csd/default/e2tts-dialogue/csd-job-flextts_b2600_ga1_duration_cond_12hz8hz6hz_d2codec780ksteps/749fcb40-b4bf-408a-8322-a476a3996fca/models/latest_checkpoint/model.checkpoint'
    voicebox_config_path = '/data1/lijiaqi/codebase/CSDs/application/d2codec_voicebox/voicebox_300M_b1200_ga3_12hz_d2codec780ksteps.yaml'
    voicebox_ckpt_path = '/mnt/scus/models/projects/lijiaqi_csd/default/e2tts-dialogue/csd-job-voicebox_300M_b1200_ga3_12hz_d2codec780ksteps/bbefd476-d766-4862-921f-32c02977d191/models/latest_checkpoint/model.checkpoint'

    # model_config_path = '/data1/lijiaqi/codebase/CSDs/application/d2codec_tts/dialog_d2codec_tts_b2600_ga1_duration_cond_8hz_d2codec_span.yaml'
    # ckpt_path = '/mnt/scus/models/projects/lijiaqi_csd/default/e2tts-dialogue/csd-job-dialog_d2codec_tts_b2600_ga1_duration_cond_8hz_d2codec_span/bd687557-cb5c-4918-a00c-243baa078601/models/latest_checkpoint/model.checkpoint'
    model_config_path = '/data1/lijiaqi/codebase/CSDs/application/d2codec_tts/dialog_d2codec_tts_b2600_ga1_duration_cond_8hz_d2codec.yaml'
    # ckpt_path = '/mnt/scus/models/projects/lijiaqi_csd/default/e2tts-dialogue/csd-job-dialog_d2codec_tts_b2600_ga1_duration_cond_8hz_d2codec_span/253926a0-e2ca-471c-8838-ded0bc791ed3/models/latest_checkpoint/model.checkpoint'
    ckpt_path = '/mnt/scus/models/projects/lijiaqi_csd/default/e2tts-dialogue/csd-job-dialog_d2codec_tts_b2600_ga1_duration_cond_8hz_d2codec/ff49a9c3-9948-484e-8c02-f8aacbc79dd8/models/latest_checkpoint/model.checkpoint'
    # ckpt_path = '/mnt/scus/models/projects/lijiaqi_csd/default/e2tts-dialogue/csd-job-flextts_b2600_ga1_duration_cond_12hz8hz6hz_d2codec780ksteps/749fcb40-b4bf-408a-8322-a476a3996fca/models/latest_checkpoint/model.checkpoint'
    # voicebox_config_path = '/data1/lijiaqi/codebase/CSDs/application/d2codec_voicebox/flexvoicebox_300M_b1200_ga3_fsq_v3_continuous_12hz8hz6hz_largebatch.yaml'
    # voicebox_ckpt_path = '/mnt/scus/models/projects/lijiaqi_csd/default/e2tts-dialogue/csd-job-flexvoicebox_300M_b1200_ga3_fsq_v3_continuous_12hz8hz6hz_largebatch/5e3bca91-fcb1-4e93-bcd7-2afbb90ee9dd/models/latest_checkpoint/model.checkpoint'


    # Model arguments
    parser.add_argument("--config", type=str, default=model_config_path, help="Path to model config YAML file")
    parser.add_argument("--checkpoint", type=str, default=ckpt_path, help="Path to model checkpoint")
    parser.add_argument("--voicebox_config", type=str, default=voicebox_config_path, help="Path to voicebox model config YAML file")
    parser.add_argument("--voicebox_checkpoint", type=str, default=voicebox_ckpt_path, help="Path to voicebox model checkpoint")
    parser.add_argument("--disable_second_stage", action="store_true", default=True, help="Disable second stage (Voicebox) and use DualCodec decoder directly")
    
    # Data format arguments
    parser.add_argument('--data_format', type=str, choices=['librispeech', 'dialog'], default='dialog', help='Data format type')
    
    # LibriSpeech data arguments (used when data_format='librispeech')
    parser.add_argument('--json_path', type=str, default='/data1/lijiaqi/data/test_librispeech/test_clean_pc_with_ref_text.json', help='path to the test data JSON file')
    parser.add_argument('--testset_root', type=str, default='/data1/lijiaqi/data/test_librispeech/', help='root path to the LibriSpeech test dataset (contains clean_original_last_3s)')
    parser.add_argument('--gt_audio_root', type=str, default='/data1/lijiaqi/data/test_librispeech/', help='root path to the ground truth LibriSpeech audio (contains clean_original)')
    
    # Dialog data arguments (used when data_format='dialog')
    parser.add_argument('--dialog_json_path', type=str, default='/mnt/wus2/models/users/v-leyzhang/data/podcast-testsets/test.json', help='path to the dialog test data JSON file')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='/data1/lijiaqi/codebase/lab/audio_outputs/dialog_0826', help='output directory')
    parser.add_argument('--skip', action='store_true', help='skip existing files', default=False)
    parser.add_argument('--rm', action='store_true', help='remove all existing files in target directory before starting', default=False)
    
    # Device arguments
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device for inference")
    parser.add_argument("--num_gpus", "-g", type=int, default=1, help="Number of GPUs to use (default: all available)")
    
    # Inference parameters
    parser.add_argument("--beam_size", type=int, default=1, help="Beam size for decoding")
    parser.add_argument("--top_k", type=int, default=25, help="Top-k sampling parameter")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--max_token_text_ratio", type=float, default=20.0)
    parser.add_argument("--min_token_text_ratio", type=float, default=0.0)
    parser.add_argument("--no_predict_duration", action="store_false", dest="predict_duration", help="Disable duration prediction")
    parser.add_argument("--n_timesteps", type=int, default=40, help="Number of diffusion timesteps")
    parser.add_argument("--cfg", type=float, default=2.0, help="Classifier-free guidance scale")
    parser.add_argument("--rescale_cfg", type=float, default=0.75, help="Rescaling factor for CFG")
    
    args = parser.parse_args()
    
    # Determine number of GPUs
    if args.num_gpus is None:
        args.num_gpus = torch.cuda.device_count()
    else:
        args.num_gpus = min(args.num_gpus, torch.cuda.device_count())
    
    if args.num_gpus == 0:
        raise ValueError("No CUDA devices available")
    
    print("=" * 50)
    print(f"DualCodec TTS {args.data_format.title()} Evaluation")
    print(f"Using {args.num_gpus} GPUs")
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print("=" * 50)
    
    # Load test data based on format
    if args.data_format == 'librispeech':
        test_cases = load_librispeech_test_data(args.json_path, args.testset_root, args.gt_audio_root)
    elif args.data_format == 'dialog':
        test_cases = load_dialog_test_data(args.dialog_json_path)
    else:
        raise ValueError(f"Unsupported data format: {args.data_format}")
    
    if not test_cases:
        print("No valid test cases found!")
        return
    
    # Create save path
    save_path = args.output_dir
    
    # Remove existing files if --rm is specified
    if args.rm and os.path.exists(save_path):
        print(f"Removing existing files in: {save_path}")
        shutil.rmtree(save_path)
    
    os.makedirs(save_path, exist_ok=True)
    
    print(f"Processing {len(test_cases)} test cases")
    print(f"Output directory: {save_path}")
    
    # Distribute test cases across GPUs
    test_cases_per_gpu = len(test_cases) // args.num_gpus
    remainder = len(test_cases) % args.num_gpus
    
    test_case_chunks = []
    start_idx = 0
    for i in range(args.num_gpus):
        end_idx = start_idx + test_cases_per_gpu + (1 if i < remainder else 0)
        test_case_chunks.append(test_cases[start_idx:end_idx])
        start_idx = end_idx
    
    # Create result queue for progress tracking
    results_queue = mp.Queue()
    
    # Start worker processes
    processes = []
    for gpu_id in range(args.num_gpus):
        if len(test_case_chunks[gpu_id]) > 0:
            p = mp.Process(
                target=worker_process,
                args=(gpu_id, test_case_chunks[gpu_id], args, save_path, results_queue)
            )
            p.start()
            processes.append(p)
            print(f"Started worker {gpu_id} with {len(test_case_chunks[gpu_id])} test cases")
    
    # Monitor progress
    total_cases = len(test_cases)
    completed_cases = 0
    success_cases = 0
    error_cases = 0
    total_duration = 0
    total_inference_time = 0
    
    # Progress monitoring
    while completed_cases < total_cases:
        try:
            result = results_queue.get(timeout=1)
            completed_cases += 1
            
            if result['status'] == 'success':
                success_cases += 1
                total_duration += result['duration']
                total_inference_time += result['inference_time']
                avg_rtf = total_inference_time / total_duration if total_duration > 0 else 0
                print(f"Progress: {completed_cases}/{total_cases} - Success: {success_cases}, Errors: {error_cases}, Avg RTF: {avg_rtf:.3f}")
            elif result['status'] == 'error':
                error_cases += 1
                print(f"Progress: {completed_cases}/{total_cases} - Error in {result['key']}: {result['error']}")
            elif result['status'] == 'fatal_error':
                print(f"Fatal error in worker {result['worker_id']}: {result['error']}")
                
        except:
            # Check if all processes are still alive
            alive_processes = [p for p in processes if p.is_alive()]
            if not alive_processes:
                break
    
    # Wait for all processes to complete
    for p in processes:
        p.join()
    
    print(f"\nCompleted {args.data_format.title()} TTS inference:")
    print(f"  Total cases: {total_cases}")
    print(f"  Successful: {success_cases}")
    print(f"  Errors: {error_cases}")
    if total_duration > 0:
        print(f"  Average RTF: {total_inference_time / total_duration:.3f}")
        print(f"  Total audio duration: {total_duration:.2f}s")
        print(f"  Total inference time: {total_inference_time:.2f}s")
    
    print(f"🎉 {args.data_format.title()} TTS evaluation completed!")

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main() 