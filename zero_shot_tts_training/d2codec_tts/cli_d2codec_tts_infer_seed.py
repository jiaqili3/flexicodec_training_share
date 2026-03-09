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

sys.path.append('/data1/lijiaqi/codebase/TS3Codec')
from audio_codec.train.feature_extractors import FBankGen

USE_G2P = True  # Set to True if you want to use G2P phonemization for English text
PARTIAL_G2P = True  # Set to True if you want to use partial G2P phonemization

def write_text_to_file(text, path):
    """Write text to file"""
    try:
        with open(path, 'w') as file:
            file.write(text)
        print(f"Text successfully written to {path}")
    except Exception as e:
        print(f"An error occurred: {e}")

def make_test_list(testset_folder, prompt_type, language_type):
    """Create test list from SeedTTS testset - copied from seed_eval_api.py"""
    if language_type == 'zh_hard':
        language_type = 'zhhard'
    test_list = [] 
    language = ""
    if language_type == "en":
        language = "en"
    elif language_type == "zh" or language_type == "zhhard":
        language = "zh"
    else:
        raise ValueError("language_type must be 'en' or 'zh' or 'zh_hard'")
    
    if prompt_type == "all":
        if language_type == 'zhhard':
            test_path = os.path.join(testset_folder, "{}/hardcase.lst".format(language))
        else:
            test_path = os.path.join(testset_folder, "{}/meta.lst".format(language))
        with open(test_path, "r") as f:
            test_list = f.readlines()
    elif prompt_type.startswith("within"):
        limit = float(prompt_type.split(" ")[-1])
        all_test_list = os.path.join(testset_folder, "{}/meta.lst".format(language))
        with open(all_test_list, "r") as f:
            test_info = f.readlines()
        for info in tqdm(test_info):
            _, _, prompt_wav, _ = info.strip().split("|")
            prompt_wav = os.path.join(testset_folder, os.path.join(language, prompt_wav))
            y, sr = librosa.load(prompt_wav, sr=None)
            duration = librosa.get_duration(y=y, sr=sr)
            if duration <= limit:
                test_list.append(info.strip())
        test_path = os.path.join(testset_folder, "{}/within_{}.lst".format(language, limit))
        with open(test_path, "w") as f:
            for line in test_list:
                f.write(line + "\n")
    else:
        raise ValueError("prompt_type must be 'all' or 'within n(s)'")
    
    random.shuffle(test_list)
    return test_list, language

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
    """Prepare text tokens for inference using Whisper tokenizer."""
    combined_text = prompt_text + " " + target_text
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
    feature_extractor = FBankGen(sr=16000)
    mel_features, _ = feature_extractor.extract_fbank(ref_audio.cpu())
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
        
        predicted_mel = voicebox_internal_model.reverse_diffusion(
            cond=cond_feature,
            prompt=prompt_mel,
            n_timesteps=n_timesteps,
            cfg=cfg,
            rescale_cfg=rescale_cfg,
        )
        
        # Vocode mel to wav
        decoded_audio = vocoder_decode_func(predicted_mel.transpose(1, 2))
        return decoded_audio.cpu().squeeze(), 24000
    
    else:
        raise ValueError(f"Invalid decoder_type: {decoder_type}")

def worker_process(
    gpu_id: int,
    test_cases: List[str],
    args,
    testset_folder: str,
    language: str,
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
        for i, info in enumerate(test_cases):
            try:
                target_name, prompt_text, prompt_wav, target_text = info.strip().split("|")
                
                # Check if output already exists and skip if needed
                output_file = os.path.join(save_path, f"{target_name}.wav")
                if os.path.exists(output_file) and args.skip:
                    print(f'Worker {gpu_id}: Skipping {output_file}')
                    continue
                
                # Construct full prompt wav path
                prompt_wav_path = os.path.join(testset_folder, language, prompt_wav)
                
                # Get ground truth path and duration
                gt_path = os.path.abspath(os.path.join(testset_folder, language, 'wavs', f'{target_name}.wav'))
                if not os.path.exists(gt_path):
                    print(f'Worker {gpu_id}: Ground truth file not found: {gt_path}')
                    continue
                
                # Calculate GT duration
                gt_audio, gt_sr = librosa.load(gt_path, sr=None)
                gt_duration = librosa.get_duration(y=gt_audio, sr=gt_sr)
                
                # Process text inputs
                prompt_text_processed = prompt_text.strip().lower().rstrip('.').replace('"', '')
                target_text_processed = target_text.strip().lower().replace('"', '')
                
                # Apply G2P if enabled and language is English
                # if args.use_g2p and language == 'en':
                #     prompt_text_final = prompt_text_processed
                #     target_text_final = g2p_phonemizer_en(target_text_processed, shorten_g2p_sequence=True)
                #     print(f"Worker {gpu_id}: Original target text: {target_text_processed}")
                #     print(f"Worker {gpu_id}: G2P target text: {target_text_final}")
                # else:
                prompt_text_final = prompt_text_processed
                target_text_final = target_text_processed
                
                # Perform TTS inference
                start_time = time.time()
                
                decoder_type = 'dualcodec' if args.disable_second_stage else 'voicebox'
                audio, sample_rate = infer_tts_single(
                    model=model,
                    text=target_text_final.lower().replace('.', ','),
                    language=language,
                    ref_audio_path=prompt_wav_path,
                    ref_text=prompt_text_final.lower().replace('.', ','),
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
                
                inference_time = time.time() - start_time
                
                # Save audio
                sf.write(output_file, audio.numpy(), sample_rate)
                print(f"Worker {gpu_id}: Saved audio to {output_file}")

                # Save prompt wav to destination with _prompt suffix
                prompt_wav_dest = os.path.join(save_path, f"{target_name}_prompt.wav")
                shutil.copyfile(prompt_wav_path, prompt_wav_dest)

                # Calculate metrics
                audio_duration = len(audio) / sample_rate
                rtf = inference_time / audio_duration

                print(f"Worker {gpu_id}: [{i+1}/{len(test_cases)}] {target_name} - Duration: {audio_duration:.2f}s, Inference: {inference_time:.2f}s, RTF: {rtf:.3f}")

                # Save text
                if args.use_g2p and language == 'en':
                    text_content = f"Original: {target_text_processed}\nG2P: {target_text_final}"
                    write_text_to_file(text_content, os.path.join(save_path, f'{target_name}.txt'))
                else:
                    write_text_to_file(target_text, os.path.join(save_path, f'{target_name}.txt'))

                # Report progress
                results_queue.put({
                    'worker_id': gpu_id,
                    'target_name': target_name,
                    'duration': audio_duration,
                    'inference_time': inference_time,
                    'rtf': rtf,
                    'status': 'success'
                })
                
            except Exception as e:
                print(f"Worker {gpu_id}: Error processing {target_name}: {e}")
                results_queue.put({
                    'worker_id': gpu_id,
                    'target_name': target_name if 'target_name' in locals() else 'unknown',
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
    
    model_config_path = '/data1/lijiaqi/codebase/CSDs/application/d2codec_tts/d2codec_tts_b2600_ga1_duration_cond_fsq_v2_500M_g2p.yaml'
    ckpt_path = '/mnt/wus2/models/projects/lijiaqi_csd/default/e2tts-dialogue/csd-job-d2codec_tts_b2600_ga1_duration_cond_fsq_v2_500M_g2p/9410a524-9830-4977-8f5b-8294fc152834/models/latest_checkpoint/model.checkpoint'
    voicebox_config_path = '/data1/lijiaqi/codebase/CSDs/application/d2codec_voicebox/voicebox_300M_b1200_ga3_fsq_v2.yaml'
    voicebox_ckpt_path = '/mnt/wus2/models/projects/lijiaqi_csd/default/e2tts-dialogue/csd-job-voicebox_300M_b1200_ga3_fsq_v2/4ba2ce4d-b4f4-4c5a-b19a-cf3352a8a560/models/latest_checkpoint/model.checkpoint'
    
    
    # Model arguments
    parser.add_argument("--config", type=str, default=model_config_path, help="Path to model config YAML file")
    parser.add_argument("--checkpoint", type=str, default=ckpt_path, help="Path to model checkpoint")
    parser.add_argument("--voicebox_config", type=str, default=voicebox_config_path, help="Path to voicebox model config YAML file")
    parser.add_argument("--voicebox_checkpoint", type=str, default=voicebox_ckpt_path, help="Path to voicebox model checkpoint")
    parser.add_argument("--disable_second_stage", action="store_true", default=False, help="Disable second stage (Voicebox) and use DualCodec decoder directly")
    
    # SeedTTS evaluation arguments (from seed_eval_api.py)
    parser.add_argument('--name', type=str, default='/data1/lijiaqi/codebase/lab/audio_outputs/d2codec_tts0805_seed_eval', help='name of output folder')
    parser.add_argument('--language', type=str, default=['en'], nargs='+', help='the language')
    parser.add_argument('--skip', action='store_true', help='skip existing files', default=False)
    parser.add_argument('--rm', action='store_true', help='remove all existing files in target directory before starting', default=False)
    parser.add_argument('--testset_folder', type=str, default='/data1/lijiaqi/codebase/lab/seedtts_testset/', help='path to the SeedTTS testset folder')
    parser.add_argument('--use_g2p', action='store_true', help='use G2P phonemization for English text', default=False)
    
    # Inference parameters
    parser.add_argument("--beam_size", type=int, default=1, help="Beam size for decoding")
    parser.add_argument("--top_k", type=int, default=25, help="Top-k sampling parameter")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--max_token_text_ratio", type=float, default=20.0)
    parser.add_argument("--min_token_text_ratio", type=float, default=0.0)
    parser.add_argument("--no_predict_duration", action="store_false", dest="predict_duration", help="Disable duration prediction")
    parser.add_argument("--n_timesteps", type=int, default=30, help="Number of diffusion timesteps")
    parser.add_argument("--cfg", type=float, default=2.0, help="Classifier-free guidance scale")
    parser.add_argument("--rescale_cfg", type=float, default=0.75, help="Rescaling factor for CFG")
    
    # Multi-GPU arguments
    parser.add_argument("--num_gpus", "-g", type=int, default=None, help="Number of GPUs to use (default: all available)")
    
    args = parser.parse_args()
    
    # Determine number of GPUs
    if args.num_gpus is None:
        args.num_gpus = torch.cuda.device_count()
    else:
        args.num_gpus = min(args.num_gpus, torch.cuda.device_count())
    
    if args.num_gpus == 0:
        raise ValueError("No CUDA devices available")
    
    print("=" * 50)
    print("DualCodec TTS SeedTTS Evaluation")
    print(f"Using {args.num_gpus} GPUs")
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print("=" * 50)
    
    save_folder = args.name
    top_k_settings = [args.top_k]
    language_settings = args.language
    
    # Process each language
    for language_type in language_settings:
        for top_k in top_k_settings:
            # AR setting
            temp = args.temperature
            top_p = 1.0
            
            # Prompt type setting
            prompt_type = "all"  # "all" or "within 5"(within n(s))
            
            # Create test list
            print(f"Creating test list for {language_type}...")
            test_list, language = make_test_list(args.testset_folder, prompt_type, language_type)
            if language_type == 'zh_hard':
                language_type = 'zhhard' 
            prompt_type = prompt_type.replace(" ", "_")
            
            # Create save path
            g2p_suffix = "_g2p" if args.use_g2p and language_type == 'en' else ""
            decoder_suffix = "_dualcodec" if args.disable_second_stage else "_voicebox"
            save_path = os.path.join(
                save_folder, 
                f"seedtts_{language_type}_{temp}_top{top_k}_top{top_p}{g2p_suffix}{decoder_suffix}/rerank_1"
            )
            
            # Remove existing files if --rm is specified
            if args.rm and os.path.exists(save_path):
                print(f"Removing existing files in: {save_path}")
                shutil.rmtree(save_path)
            
            os.makedirs(save_path, exist_ok=True)
            
            print(f"Processing {len(test_list)} test cases for {language_type}")
            print(f"Output directory: {save_path}")
            
            # Distribute test cases across GPUs
            test_cases_per_gpu = len(test_list) // args.num_gpus
            remainder = len(test_list) % args.num_gpus
            
            test_case_chunks = []
            start_idx = 0
            for i in range(args.num_gpus):
                end_idx = start_idx + test_cases_per_gpu + (1 if i < remainder else 0)
                test_case_chunks.append(test_list[start_idx:end_idx])
                start_idx = end_idx
            
            # Create result queue for progress tracking
            results_queue = mp.Queue()
            
            # Start worker processes
            processes = []
            for gpu_id in range(args.num_gpus):
                if len(test_case_chunks[gpu_id]) > 0:
                    p = mp.Process(
                        target=worker_process,
                        args=(gpu_id, test_case_chunks[gpu_id], args, args.testset_folder, language, save_path, results_queue)
                    )
                    p.start()
                    processes.append(p)
                    print(f"Started worker {gpu_id} with {len(test_case_chunks[gpu_id])} test cases")
            
            # Monitor progress
            total_cases = len(test_list)
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
                        print(f"Progress: {completed_cases}/{total_cases} - Error in {result['target_name']}: {result['error']}")
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
            
            print(f"\nCompleted {language_type}:")
            print(f"  Total cases: {total_cases}")
            print(f"  Successful: {success_cases}")
            print(f"  Errors: {error_cases}")
            if total_duration > 0:
                print(f"  Average RTF: {total_inference_time / total_duration:.3f}")
                print(f"  Total audio duration: {total_duration:.2f}s")
                print(f"  Total inference time: {total_inference_time:.2f}s")
    
    print("🎉 SeedTTS evaluation completed!")

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main() 