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
import multiprocessing as mp
from pathlib import Path
from datetime import datetime
from functools import partial
import torch.nn.functional as F
from tqdm import tqdm
import shutil
import time
from typing import List, Dict, Optional

# Add paths for imports
# This allows importing from the project root
sys.path.append(Path(__file__).resolve().parent.parent.parent.parent.as_posix())
# Add path for dualcodec and its dependencies
# sys.path.append('/data1/lijiaqi/codebase/CSDs/application/ts3codec/TS3Codec/zero-shot-tts-training')
sys.path.append('/data1/lijiaqi/codebase/CSDs/')
sys.path.append('/data1/lijiaqi/codebase/CSDs//zero-shot-tts-training')
from zero_shot_tts_training.voicebox.d2codec_voicebox import VoiceboxWrapper
from dualcodec.model_tts.voicebox.vocoder_model import get_vocos_model_spectrogram, mel_to_wav_vocos
sys.path.append('/data1/lijiaqi/codebase/CSDs/application/ts3codec/TS3Codec')

from audio_codec.train.feature_extractors import FBankGen

# Global feature extractor for dualcodec
feature_extractor_for_dualcodec = FBankGen(sr=16000)

threshold = 1.0

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
            model_config[key] = value.replace('/modelblob', '/mnt/scus/models')
    
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
def infer_voicebox_librispeech(
    model: VoiceboxWrapper,
    vocoder_decode_func,
    gt_audio_path: str,
    ref_audio_path: str,
    device: str = 'cuda',
    n_timesteps: int = 10,
    cfg: float = 2.0,
    rescale_cfg: float = 0.75,
):
    """Perform inference using Voicebox model with LibriSpeech data"""
    use_decoder_latent = model.use_decoder_latent
    use_decoder_latent_before_agg = model.use_decoder_latent_before_agg
    decoder_latent_pass_transformer = model.decoder_latent_pass_transformer
    
    # 1. Load ground truth audio and extract semantic codes
    print(f"Loading ground truth audio: {gt_audio_path}")
    gt_audio, sr = torchaudio.load(gt_audio_path)
    return gt_audio, sr
    if sr != 16000:
        gt_audio = torchaudio.transforms.Resample(sr, 16000)(gt_audio)
    if gt_audio.shape[0] > 1:
        gt_audio = gt_audio.mean(dim=0, keepdim=True)
    gt_audio = gt_audio.to(device)

    # 2. Load reference audio for prompt
    print(f"Loading reference audio for prompt: {ref_audio_path}")
    ref_audio, sr = torchaudio.load(ref_audio_path)
    if sr != 16000:
        ref_audio = torchaudio.transforms.Resample(sr, 16000)(ref_audio)
    if ref_audio.shape[0] > 1:
        ref_audio = ref_audio.mean(dim=0, keepdim=True)
    ref_audio = ref_audio.to(device)

    # 3. Extract semantic codes from reference audio
    print("Extracting semantic codes from reference audio...")
    ref_mel_features, _ = feature_extractor_for_dualcodec.extract_fbank(ref_audio.cpu())
    ref_mel_features = ref_mel_features.to(device)
    ref_x_lens = torch.tensor([ref_mel_features.shape[1]], dtype=torch.long, device=device)

    model.dualcodec_model.similarity_threshold = threshold

    ref_dualcodec_output = model._extract_dualcodec_features(ref_audio, mel=ref_mel_features, x_lens=ref_x_lens, manual_threshold=threshold)
    if not use_decoder_latent:
        ref_cond_codes = ref_dualcodec_output['semantic_codes'].squeeze(1)
    else:
        ref_cond_codes = ref_dualcodec_output['semantic_codes_aggregated'].squeeze(1)

    # 4. Extract semantic codes from ground truth audio for conditioning
    print("Extracting semantic codes from ground truth audio...")
    gt_mel_features, _ = feature_extractor_for_dualcodec.extract_fbank(gt_audio.cpu())
    gt_mel_features = gt_mel_features.to(device)
    gt_x_lens = torch.tensor([gt_mel_features.shape[1]], dtype=torch.long, device=device)

    gt_dualcodec_output = model._extract_dualcodec_features(gt_audio, mel=gt_mel_features, x_lens=gt_x_lens)
    gt_cond_codes = gt_dualcodec_output['semantic_codes'].squeeze(1) if not use_decoder_latent else gt_dualcodec_output['semantic_codes_aggregated'].squeeze(1)

    gt_token_lengths = gt_dualcodec_output['token_lengths']

    # use GT speech
    # reconstructed mel
    # gt_mel = model._extract_mel_features(gt_audio)
    # predicted_audio = vocoder_decode_func(gt_mel.transpose(1, 2))
    # return predicted_audio.cpu().squeeze(), 24000

    # return gt_audio.cpu().reshape(1,-1), 16000
    # return model.dualcodec_model.dac.decoder(model.dualcodec_model.bottleneck_transformer(gt_dualcodec_output['decoder_latent_before_agg'])).squeeze(0).cpu(), 16000

    # 5. Concatenate reference and GT semantic codes (ref first, then GT)
    print("Concatenating reference and GT semantic codes...")
    cond_codes = torch.cat([ref_cond_codes, gt_cond_codes], dim=1)
    
    # 6. Perform inference
    print("Running reverse diffusion...")
    voicebox_model = model.voicebox_model
    
    cond_feature = voicebox_model.cond_emb(cond_codes)
    cond_feature = F.interpolate(
        cond_feature.transpose(1, 2),
        scale_factor=voicebox_model.cond_scale_factor,
    ).transpose(1, 2)
    if model.add_framerate_embedding:
        framerate_idx = model.flex_framerate_options.index(threshold)
        framerate_emb = model.framerate_embedding(torch.tensor(framerate_idx, device=device))
        cond_feature = cond_feature + framerate_emb.unsqueeze(0).unsqueeze(0)
    
    # Use the reference audio as prompt
    if use_decoder_latent:
        prompt_mel = ref_dualcodec_output['decoder_latent'].transpose(1,2)
    elif use_decoder_latent_before_agg:
        prompt_mel = ref_dualcodec_output['decoder_latent_before_agg'].transpose(1,2)
        if decoder_latent_pass_transformer:
            prompt_mel = model.dualcodec_model.bottleneck_transformer(ref_dualcodec_output['decoder_latent_before_agg']).transpose(1,2)
    else:
        prompt_mel = model._extract_mel_features(ref_audio)

    if model.concat_speaker_embedding:
        speaker_embedding = model._extract_speaker_embedding(ref_audio, sample_rate=16000)
        speaker_emb_expanded = speaker_embedding.unsqueeze(0).unsqueeze(0)
        cond_feature = model.spk_linear(torch.cat([cond_feature, speaker_emb_expanded], dim=-1))

    with torch.autocast(device_type='cuda', dtype=torch.float32):
        predicted_mel = voicebox_model.reverse_diffusion(
            cond=cond_feature,
            prompt=prompt_mel,
            n_timesteps=n_timesteps,
            cfg=cfg,
            rescale_cfg=rescale_cfg,
        )

        # 7. Vocode mel to wav
        if use_decoder_latent:
            predicted_audio = model.dualcodec_model.decode_from_latent(predicted_mel.transpose(1,2), gt_token_lengths)
            return predicted_audio.cpu().squeeze(), 16000
        elif use_decoder_latent_before_agg:
            if decoder_latent_pass_transformer:
                predicted_audio = model.dualcodec_model.dac.decoder(torch.cat([prompt_mel, predicted_mel], dim=1).transpose(1,2))
                predicted_audio = predicted_audio[...,ref_audio.shape[-1]:]
                return predicted_audio.cpu().squeeze(0), 16000
                # return model.dualcodec_model.dac.decoder(predicted_mel.transpose(1,2)).cpu().squeeze(0), 16000
            else:
                predicted_audio = model.dualcodec_model.dac.decoder(model.dualcodec_model.bottleneck_transformer(torch.cat([prompt_mel, predicted_mel], dim=1).transpose(1,2)))
                predicted_audio = predicted_audio[...,ref_audio.shape[-1]:]
                return predicted_audio.cpu().squeeze(0), 16000
        else:
            print("Vocoding generated mel spectrogram...")
            predicted_audio = vocoder_decode_func(predicted_mel.transpose(1, 2))
            return predicted_audio.cpu().squeeze(), 24000


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
        vocoder_decode_func, _ = load_vocoder(device)
        
        # Process test cases
        for i, test_case in enumerate(test_cases):
            try:
                key = test_case['key']
                gt_audio_path = test_case['gt_audio_path']
                ref_audio_path = test_case['ref_audio_path']
                text = test_case['text']
                text_ref = test_case['text_ref']
                
                # Check if output already exists and skip if needed
                output_file = os.path.join(save_path, f"{key}.wav")
                if os.path.exists(output_file) and args.skip:
                    print(f"Worker {gpu_id}: Skipping existing file: {output_file}")
                    continue
                
                # Perform inference
                start_time = time.time()
                
                audio, sr = infer_voicebox_librispeech(
                    model=model,
                    vocoder_decode_func=vocoder_decode_func,
                    gt_audio_path=gt_audio_path,
                    ref_audio_path=ref_audio_path,
                    device=device,
                    n_timesteps=args.n_timesteps,
                    cfg=args.cfg,
                    rescale_cfg=args.rescale_cfg
                )
                
                inference_time = time.time() - start_time
                
                # Save audio
                if sr == 16000:
                    torchaudio.save(output_file, audio, 16000)
                    # sf.write(output_file, audio.numpy(), 16000)  # Vocos output is 24kHz
                else:
                    sf.write(output_file, audio.numpy(), 24000)  # Vocos output is 24kHz
                print(f"Worker {gpu_id}: Saved audio to {output_file}")

                # Save reference audio to destination with _ref suffix
                ref_audio_dest = os.path.join(save_path, f"{key}_ref.wav")
                shutil.copyfile(ref_audio_path, ref_audio_dest)
                
                # Save ground truth audio with _gt suffix
                gt_audio_dest = os.path.join(save_path, f"{key}_gt.wav")
                shutil.copyfile(gt_audio_path, gt_audio_dest)

                # Calculate metrics
                audio_duration = audio.shape[-1] / sr
                rtf = inference_time / audio_duration

                print(f"Worker {gpu_id}: [{i+1}/{len(test_cases)}] {key} - Duration: {audio_duration:.2f}s, Inference: {inference_time:.2f}s, RTF: {rtf:.3f}")

                # Save text information
                text_content = f"Target: {text}\nReference: {text_ref}\nGT Audio: {gt_audio_path}\nRef Audio: {ref_audio_path}"
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
    parser = argparse.ArgumentParser(description="LibriSpeech Inference for D2Codec-Voicebox using GT semantic tokens")
    
    # Default paths
    # voicebox_config_path = '/data1/lijiaqi/codebase/CSDs/application/d2codec_voicebox/voicebox_300M_b1200_ga3_fsq_v2.yaml'
    # voicebox_ckpt_path = '/mnt/wus2/models/projects/lijiaqi_csd/default/e2tts-dialogue/csd-job-voicebox_300M_b1200_ga3_fsq_v2/4ba2ce4d-b4f4-4c5a-b19a-cf3352a8a560/models/latest_checkpoint/model.checkpoint'
    # voicebox_config_path = '/data1/lijiaqi/codebase/CSDs/application/d2codec_voicebox/voicebox_300M_b1200_ga3_12hz_v2.yaml'
    # voicebox_ckpt_path = '/mnt/wus2/models/projects/lijiaqi_csd/default/e2tts-dialogue/csd-job-voicebox_300M_b1200_ga3_12hz_v2/42cd59ce-b7b0-4aed-9f5d-d8d1130284b1/models/latest_checkpoint/model.checkpoint'
    # voicebox_config_path = '/data1/lijiaqi/codebase/CSDs/application/d2codec_voicebox/voicebox_300M_b1200_ga3_fsq.yaml'
    # voicebox_ckpt_path = '/mnt/wus2/models/projects/lijiaqi_csd/default/e2tts-dialogue/csd-job-voicebox_300M_b1200_ga3_fsq/5c602a67-39b5-4b48-8e3d-b99e769acba8/models/checkpoint34/model.checkpoint'
    # voicebox_config_path = '/data1/lijiaqi/codebase/CSDs/application/d2codec_voicebox/voicebox_300M_b1200_ga3_12hz_s2codec.yaml'
    # voicebox_ckpt_path = '/mnt/wus2/models/projects/lijiaqi_csd/default/e2tts-dialogue/csd-job-voicebox_300M_b1200_ga3_12hz_s2codec/b10d550e-008d-4801-8f6e-8c74365b2e52/models/latest_checkpoint/model.checkpoint'
    # voicebox_config_path = '/data1/lijiaqi/codebase/CSDs/application/d2codec_voicebox/voicebox_300M_b1200_ga3_12hz_d2codec.yaml'
    # voicebox_ckpt_path = '/mnt/wus2/models/projects/lijiaqi_csd/default/e2tts-dialogue/csd-job-voicebox_300M_b1200_ga3_12hz_d2codec/36487f98-c1a1-495c-85c0-fcb1dd596ee1/models/latest_checkpoint/model.checkpoint'
    # voicebox_config_path = '/data1/lijiaqi/codebase/CSDs/application/d2codec_voicebox/voicebox_300M_b1200_ga3_fsq_v3_continuous_8hz6hz.yaml'
    # voicebox_ckpt_path = '/mnt/wus2/models/projects/lijiaqi_csd/default/e2tts-dialogue/csd-job-voicebox_300M_b1200_ga3_fsq_v3_continuous_8hz6hz/e841edc4-9925-424a-bdb4-03c34443eb6a/models/latest_checkpoint/model.checkpoint'
    
    # voicebox_config_path = '/data1/lijiaqi/codebase/CSDs/application/d2codec_voicebox/voicebox_300M_b1200_ga3_fsq_v3_continuous_12hz8hz6hz_largebatch.yaml'
    # voicebox_ckpt_path = '/mnt/scus/models/projects/lijiaqi_csd/default/e2tts-dialogue/csd-job-voicebox_300M_b1200_ga3_fsq_v3_continuous_12hz8hz6hz_largebatch/fa7aba3c-954a-4b4b-a59a-a1027360cde0/models/latest_checkpoint/model.checkpoint'
    # voicebox_ckpt_path = '/mnt/wus2/models/projects/lijiaqi_csd/default/e2tts-dialogue/csd-job-voicebox_300M_b1200_ga3_fsq_v3_continuous_12hz8hz6hz/bb622c50-689b-437e-8764-da49d73e5e6f/models/latest_checkpoint/model.checkpoint'
    # voicebox_ckpt_path = '/mnt/scus/models/projects/lijiaqi_csd/default/e2tts-dialogue/csd-job-voicebox_300M_b1200_ga3_fsq_v3_continuous_12hz8hz6hz/b1d28891-b70c-4712-96a9-90010e5f37e4/models/latest_checkpoint/model.checkpoint'

    # voicebox_config_path = '/data1/lijiaqi/codebase/CSDs/application/d2codec_voicebox/flexvoicebox_300M_b1200_ga3_fsq_v3_continuous_12hz8hz6hz_largebatch.yaml'
    # voicebox_ckpt_path = '/mnt/scus/models/projects/lijiaqi_csd/default/e2tts-dialogue/csd-job-flexvoicebox_300M_b1200_ga3_fsq_v3_continuous_12hz8hz6hz_largebatch/5e3bca91-fcb1-4e93-bcd7-2afbb90ee9dd/models/latest_checkpoint/model.checkpoint'
    # voicebox_config_path = '/data1/lijiaqi/codebase/CSDs/application/d2codec_voicebox/voicebox_300M_b1200_ga3_fsq_v3_continuous_12hz8hz6hz_largebatch.yaml'
    # voicebox_ckpt_path = '/mnt/scus/models/projects/lijiaqi_csd/default/e2tts-dialogue/csd-job-voicebox_300M_b1200_ga3_fsq_v3_continuous_12hz8hz6hz_largebatch/fa7aba3c-954a-4b4b-a59a-a1027360cde0/models/checkpoint13/model.checkpoint'
    
    # voicebox_config_path = '/data1/lijiaqi/codebase/CSDs/application/d2codec_voicebox/flexvoicebox_300M_b1200_ga3_fsq_v3_continuous_12hz8hz6hz_largebatch_1200ksteps.yaml'
    # voicebox_ckpt_path = '/mnt/scus/models/projects/lijiaqi_csd/default/e2tts-dialogue/csd-job-voicebox_300M_b1200_ga3_fsq_v3_continuous_12hz8hz6hz_largebatch/fa7aba3c-954a-4b4b-a59a-a1027360cde0/models/checkpoint13/model.checkpoint'
    
    voicebox_config_path = '/data1/lijiaqi/codebase/CSDs/application/d2codec_voicebox/flexvoicebox_300M_b1200_ga3_fsq_v3_continuous_12hz8hz6hz_largebatch.yaml'
    voicebox_ckpt_path = '/mnt/scus/models/projects/lijiaqi_csd/default/e2tts-dialogue/csd-job-flexvoicebox_300M_b1200_ga3_fsq_v3_continuous_12hz8hz6hz_largebatch/70ac5f28-c624-42d0-b3ee-5b330d94689e/models/latest_checkpoint/model.checkpoint'
    
    
    # voicebox_config_path = '/data1/lijiaqi/codebase/CSDs/application/d2codec_voicebox/flexvoicebox_300M_b1200_ga3_fsq_v3_continuous_12hz8hz6hz_largebatch_spk.yaml'
    # voicebox_ckpt_path = '/mnt/scus/models/projects/lijiaqi_csd/default/e2tts-dialogue/csd-job-flexvoicebox_300M_b1200_ga3_fsq_v3_continuous_12hz8hz6hz_largebatch_spk/3b32262a-6267-45fc-be38-0f301412e9cd/models/latest_checkpoint/model.checkpoint'


    # voicebox_config_path = '/data1/lijiaqi/codebase/CSDs/application/d2codec_voicebox/flexvoicebox_300M_b1200_ga3_fsq_v3_continuous_12hz8hz6hz_largebatch_spk_notransformer.yaml'
    # voicebox_ckpt_path = '/mnt/scus/models/projects/lijiaqi_csd/default/e2tts-dialogue/csd-job-flexvoicebox_300M_b1200_ga3_fsq_v3_continuous_12hz8hz6hz_largebatch_spk_notransformer/e216d4a5-4e1b-4767-9245-5f8ed3ae6095/models/latest_checkpoint/model.checkpoint'






    parser.add_argument("--config", type=str, default=voicebox_config_path, help="Path to model config YAML file")
    parser.add_argument("--checkpoint", type=str, default=voicebox_ckpt_path, help="Path to model checkpoint")
    
    # LibriSpeech data arguments
    parser.add_argument('--json_path', type=str, default='/data1/lijiaqi/data/test_librispeech/test_clean_pc_with_ref_text.json', help='path to the test data JSON file')
    parser.add_argument('--testset_root', type=str, default='/data1/lijiaqi/data/test_librispeech/', help='root path to the LibriSpeech test dataset (contains clean_original_last_3s)')
    parser.add_argument('--gt_audio_root', type=str, default='/data1/lijiaqi/data/test_librispeech/', help='root path to the ground truth LibriSpeech audio (contains clean_original)')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='/data1/lijiaqi/codebase/lab/audio_outputs/0904test_12hz_continuous_gt', help='output directory')
    # parser.add_argument('--output_dir', type=str, default='/data1/lijiaqi/codebase/lab/audio_outputs/d2codec_voicebox_librispeech_gt_tokens_flexvoicebox_300M_b1200_ga3_fsq_v3_continuous_12hz8hz6hz_gtmelaudio0904', help='output directory')
    parser.add_argument('--skip', action='store_true', help='skip existing files', default=False)
    parser.add_argument('--rm', action='store_true', help='remove all existing files in target directory before starting', default=False)
    
    # Device arguments
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device for inference")
    parser.add_argument("--num_gpus", "-g", type=int, default=1, help="Number of GPUs to use (default: all available)")
    
    # Inference parameters
    parser.add_argument("--n_timesteps", type=int, default=15, help="Number of diffusion timesteps")
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
    print("D2Codec-Voicebox LibriSpeech Inference with GT Semantic Tokens")
    print(f"Using {args.num_gpus} GPUs")
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print("=" * 50)
    
    # Load test data
    test_cases = load_librispeech_test_data(args.json_path, args.testset_root, args.gt_audio_root)
    
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
    
    print(f"\nCompleted LibriSpeech Voicebox inference:")
    print(f"  Total cases: {total_cases}")
    print(f"  Successful: {success_cases}")
    print(f"  Errors: {error_cases}")
    if total_duration > 0:
        print(f"  Average RTF: {total_inference_time / total_duration:.3f}")
        print(f"  Total audio duration: {total_duration:.2f}s")
        print(f"  Total inference time: {total_inference_time:.2f}s")
    
    print("🎉 LibriSpeech Voicebox inference completed!")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
