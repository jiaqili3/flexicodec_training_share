#!/usr/bin/env python3
# Copyright (c) 2025 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


# CosyVoice imports and setup
import sys
# sys.path.append('/data1/lijiaqi/codebase/CSDs/zero-shot-tts-training/zero_shot_tts_training/')
sys.path.append('/data1/lijiaqi/codebase/CosyVoice')
sys.path.append('/data1/lijiaqi/codebase/CosyVoice/third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice
from cosyvoice.utils.file_utils import load_wav
import torchaudio





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

# Add paths for imports (commented out for CosyVoice)
# sys.path.append('/data1/lijiaqi/codebase/CSDs/zero-shot-tts-training')
# sys.path.append('/data1/lijiaqi/codebase/CSDs/zero-shot-tts-training/zero_shot_tts_training')
# from zero_shot_tts_training.d2codec_tts.llm_duration import TransformerLMWrapper
# from zero_shot_tts_training.tools.whisper_tokenize import text2idx
# from zero_shot_tts_training.voicebox.d2codec_voicebox import VoiceboxWrapper
# from dualcodec.model_tts.voicebox.vocoder_model import get_vocos_model_spectrogram, mel_to_wav_vocos

# Import G2P phonemizer (commented out for CosyVoice)
# sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'tools'))
# from g2p_phonemizer import g2p_phonemizer_en

# sys.path.append('/data1/lijiaqi/codebase/CSDs/application/ts3codec/TS3Codec/')
# from audio_codec.train.feature_extractors import FBankGen

# Global feature extractor for dualcodec (commented out for CosyVoice)
# feature_extractor_for_dualcodec = FBankGen(sr=16000)


# USE_G2P = True  # Set to True if you want to use G2P phonemization for English text (commented out for CosyVoice)
# PARTIAL_G2P = False  # Set to True if you want to use partial G2P phonemization (commented out for CosyVoice)

# INFERENCE_FRAMERATE = 0.91
# print(f'Inference framerate set to: {INFERENCE_FRAMERATE}')

# Global feature extractor for dualcodec (commented out for CosyVoice)
# feature_extractor_for_dualcodec = FBankGen(sr=16000)

def write_text_to_file(text, path):
    """Write text to file"""
    try:
        with open(path, 'w') as file:
            file.write(text)
        print(f"Text successfully written to {path}")
    except Exception as e:
        print(f"An error occurred: {e}")

def load_cosyvoice_model(model_path: str):
    """Load CosyVoice model"""
    print(f"Loading CosyVoice model from: {model_path}")
    try:
        cosyvoice = CosyVoice(model_path)
        print("CosyVoice model loaded successfully")
        return cosyvoice
    except Exception as e:
        print(f"Error loading CosyVoice model: {e}")
        raise

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
    """Load CosyVoice model (replacing D2Codec model loading)"""
    print(f"Loading CosyVoice model instead of D2Codec model")
    # For CosyVoice, we don't need config files, just the model path
    # This function is kept for compatibility but redirects to CosyVoice loading
    return load_cosyvoice_model(checkpoint_path if checkpoint_path else config_path)

def load_voicebox_model_from_config(config_path: str, checkpoint_path: str, device: str = 'cuda'):
    """Voicebox model loading (disabled for CosyVoice)"""
    print(f"Voicebox model loading disabled for CosyVoice inference")
    return None

def load_vocoder(device='cuda'):
    """Load vocoder (disabled for CosyVoice)"""
    print("Vocoder loading disabled for CosyVoice inference")
    return None, None

def prepare_text_tokens(prompt_text: str, target_text: str, language: str) -> Tuple[str, str]:
    """Prepare text for CosyVoice inference (simplified)."""
    combined_text = target_text  # CosyVoice handles text directly
    print(f"Preparing text for CosyVoice: {combined_text}")
    return prompt_text, combined_text

def extract_reference_features(
    model, 
    ref_audio_path: str, 
    device: str = 'cuda'
) -> Dict[str, torch.Tensor]:
    """Load reference audio for CosyVoice (simplified)"""
    print(f"Loading reference audio: {ref_audio_path}")
    prompt_speech_16k = load_wav(ref_audio_path, 16000)
    return {
        'prompt_speech': prompt_speech_16k,
        'path': ref_audio_path
    }

@torch.inference_mode()
def infer_tts_single(
    args,
    model,  # CosyVoice model
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
    voicebox_model = None,
    vocoder_decode_func=None,
    n_timesteps: int = 30,
    cfg: float = 2.0,
    rescale_cfg: float = 0.75,
    decoder_type: str = 'cosyvoice',
    duration_top_k: int = 1,
    duration_temperature: float = 1.0
) -> Tuple[torch.Tensor, int, float, float]:
    """Perform TTS inference using CosyVoice"""
    print(f"Performing CosyVoice inference for text: {text}")
    
    # Load reference audio using CosyVoice utility
    prompt_speech_16k = load_wav(ref_audio_path, 16000)
    
    # Prepare language tag for CosyVoice
    if language == 'en':
        language_tag = '<|en|>'
    elif language == 'zh':
        language_tag = '<|zh|>'
    else:
        language_tag = '<|en|>'  # Default to English
    
    # Combine language tag with text
    full_text = language_tag + text
    full_ref_text = language_tag + ref_text if ref_text else language_tag
    
    print(f"CosyVoice input text: {full_text}")
    print(f"CosyVoice reference text: {full_ref_text}")
    
    ar_start_time = time.time()
    
    # Generate audio using CosyVoice zero-shot inference
    try:
        # Use zero_shot inference with English language tag
        results = list(model.inference_zero_shot(full_text, full_ref_text, prompt_speech_16k, stream=False))
        
        if not results:
            raise ValueError("No audio generated by CosyVoice")
        
        # Get the first result
        result = results[0]
        generated_audio = result['tts_speech']
        sample_rate = model.sample_rate
        
        ar_inference_time = time.time() - ar_start_time
        
        # Calculate metrics
        audio_duration = generated_audio.shape[-1] / sample_rate
        avg_framerate = 0.0  # Not applicable for CosyVoice
        
        print(f"CosyVoice generated audio: {generated_audio.shape}, sample_rate: {sample_rate}")
        
        return generated_audio.cpu().squeeze(), sample_rate, avg_framerate, ar_inference_time
        
    except RuntimeError as e:
        print(f"Error in CosyVoice inference: {e}")
        # Return a dummy audio tensor in case of error
        dummy_audio = torch.zeros(16000)  # 1 second of silence
        return dummy_audio, 16000, 0.0, time.time() - ar_start_time

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
        # Load CosyVoice model
        print(f"Worker {gpu_id}: Loading CosyVoice model from {args.cosyvoice_model_path}")
        model = load_cosyvoice_model(args.cosyvoice_model_path)
        
        # CosyVoice doesn't need second stage models
        voicebox_model = None
        vocoder_decode_func = None
        
        # Process test cases
        for i, test_case in enumerate(test_cases):
            try:
                key = test_case['key']
                text = test_case['text']
                ref_audio_path = test_case['ref_audio_path']
                gt_audio_path = test_case['gt_audio_path']
                text_ref = test_case['text_ref']
                
                # Check if output already exists and skip if needed
                output_file = os.path.join(save_path, f"{key}.wav")
                if os.path.exists(output_file) and args.skip:
                    print(f"Worker {gpu_id}: Skipping existing file: {output_file}")
                    continue
                
                # Process text inputs
                text_processed = text.strip().lower().replace('"', '')
                text_ref_processed = text_ref.strip().lower().rstrip('!').rstrip('.').replace('"', '')
                
                # Apply G2P if enabled (disabled for CosyVoice)
                # if USE_G2P:
                #     if PARTIAL_G2P:
                #         text_final = g2p_phonemizer_en(text_processed, shorten_g2p_sequence=True)
                #         text_ref_final = text_ref_processed  # Keep reference text as is
                #     else:
                #         text_final = g2p_phonemizer_en(text_processed, shorten_g2p_sequence=True)
                #         text_ref_final = g2p_phonemizer_en(text_ref_processed, shorten_g2p_sequence=True)
                #     print(f"Worker {gpu_id}: Original target text: {text_processed}")
                #     print(f"Worker {gpu_id}: G2P target text: {text_final}")
                # else:
                text_final = text_processed
                text_ref_final = text_ref_processed
                
                # Perform TTS inference
                start_time = time.time()
                
                decoder_type = 'cosyvoice'  # Use CosyVoice
                audio, sample_rate, avg_framerate, ar_inference_time = infer_tts_single(
                    args=args,
                    model=model,
                    text=text_final.lower(),  #.replace('.', ','),
                    language='en',  # LibriSpeech is English
                    ref_audio_path=ref_audio_path,
                    ref_text=text_ref_final.lower().replace('.', ','),
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
                    duration_top_k=args.duration_top_k,
                    duration_temperature=args.duration_temperature
                )
                
                inference_time = time.time() - start_time
                
                # Save audio
                torchaudio.save(output_file, audio.reshape(1,-1), sample_rate=sample_rate)
                # sf.write(output_file, audio.numpy(), sample_rate)
                print(f"Worker {gpu_id}: Saved audio to {output_file}")

                # Save reference audio to destination with _ref suffix
                ref_audio_dest = os.path.join(save_path, f"{key}_ref.wav")
                shutil.copyfile(ref_audio_path, ref_audio_dest)
                
                # Save ground truth audio with _gt suffix
                gt_audio_dest = os.path.join(save_path, f"{key}_gt.wav")
                shutil.copyfile(gt_audio_path, gt_audio_dest)

                # Calculate metrics
                audio_duration = audio.shape[-1] / sample_rate
                rtf = inference_time / audio_duration
                ar_rtf = ar_inference_time / audio_duration

                print(f"Worker {gpu_id}: [{i+1}/{len(test_cases)}] {key} - Duration: {audio_duration:.2f}s, Inference: {inference_time:.2f}s, RTF: {rtf:.3f}, AR_RTF: {ar_rtf:.3f}")

                # Save text information
                text_content = f"Target: {text}\nReference: {text_ref}\nGT Audio: {gt_audio_path}\nRef Audio: {ref_audio_path}"
                # Removed G2P text info for CosyVoice
                # if USE_G2P:
                #     text_content += f"\nG2P Target: {text_final}"
                #     if not PARTIAL_G2P:
                #         text_content += f"\nG2P Reference: {text_ref_final}"
                with open(os.path.join(save_path, f'{key}.txt'), 'w', encoding='utf-8') as f:
                    f.write(text_content)

                # Report progress
                results_queue.put({
                    'worker_id': gpu_id,
                    'key': key,
                    'duration': audio_duration,
                    'inference_time': inference_time,
                    'rtf': rtf,
                    'ar_rtf': ar_rtf,
                    'avg_framerate': avg_framerate,
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
    parser = argparse.ArgumentParser(description="LibriSpeech Evaluation using CosyVoice")
    
    # CosyVoice Model configuration (replace all D2Codec paths)
    cosyvoice_model_path = 'pretrained_models/CosyVoice-300M'  # Default CosyVoice model path
    
    # Model arguments (simplified for CosyVoice)
    parser.add_argument("--cosyvoice_model_path", type=str, default="/data1/lijiaqi/codebase/CosyVoice/pretrained_models/CosyVoice-300M", help="Path to CosyVoice model directory")
    # Keep these for compatibility but they won't be used
    parser.add_argument("--config", type=str, default="", help="Not used for CosyVoice")
    parser.add_argument("--checkpoint", type=str, default="", help="Not used for CosyVoice")
    parser.add_argument("--voicebox_config", type=str, default="", help="Not used for CosyVoice")
    parser.add_argument("--voicebox_checkpoint", type=str, default="", help="Not used for CosyVoice")
    parser.add_argument("--disable_second_stage", action="store_true", default=True, help="Always true for CosyVoice")
    
    # LibriSpeech data arguments
    parser.add_argument('--json_path', type=str, default='/data1/lijiaqi/data/test_librispeech/test_clean_pc_with_ref_text.json', help='path to the test data JSON file')
    parser.add_argument('--testset_root', type=str, default='/data1/lijiaqi/data/test_librispeech/', help='root path to the LibriSpeech test dataset (contains clean_original_last_3s)')
    parser.add_argument('--gt_audio_root', type=str, default='/data1/lijiaqi/data/test_librispeech/', help='root path to the ground truth LibriSpeech audio (contains clean_original)')
    
    # Output arguments
    parser.add_argument('--inference_framerate', type=float, default=1.0, help='Not used for CosyVoice')
    parser.add_argument('--output_dir_base', type=str, default='/data1/lijiaqi/codebase/lab/audio_outputs/', help='output base directory')
    parser.add_argument('--output_dir', type=str, default='cosyvoice_librispeech_eval', help='output directory')
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
    parser.add_argument("--n_timesteps", type=int, default=15, help="Number of diffusion timesteps")
    parser.add_argument("--cfg", type=float, default=2.0, help="Classifier-free guidance scale")
    parser.add_argument("--duration_top_k", type=int, default=2, help="Top-k sampling for duration prediction")
    parser.add_argument("--duration_temperature", type=float, default=0.8, help="Temperature for duration prediction")
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
    print("CosyVoice LibriSpeech Evaluation")
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
    save_path = os.path.join(args.output_dir_base, args.output_dir)
    
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
    total_ar_inference_time = 0
    total_avg_framerate = 0.0
    
    # Progress monitoring
    while completed_cases < total_cases:
        try:
            result = results_queue.get(timeout=1)
            completed_cases += 1
            
            if result['status'] == 'success':
                success_cases += 1
                total_duration += result['duration']
                total_inference_time += result['inference_time']
                total_ar_inference_time += result.get('ar_rtf', 0) * result['duration']
                total_avg_framerate += result['avg_framerate']
                avg_rtf = total_inference_time / total_duration if total_duration > 0 else 0
                avg_ar_rtf = total_ar_inference_time / total_duration if total_duration > 0 else 0
                avg_framerate_all = total_avg_framerate / success_cases if success_cases > 0 else 0
                print(f"Progress: {completed_cases}/{total_cases} - Success: {success_cases}, Errors: {error_cases}, Avg RTF: {avg_rtf:.3f}, Avg AR_RTF: {avg_ar_rtf:.3f}, Avg Framerate: {avg_framerate_all:.3f}")
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
    
    print(f"\nCompleted LibriSpeech CosyVoice TTS inference:")
    print(f"  Total cases: {total_cases}")
    print(f"  Successful: {success_cases}")
    print(f"  Errors: {error_cases}")
    if total_duration > 0:
        print(f"  Average RTF: {total_inference_time / total_duration:.3f}")
        print(f"  Average AR_RTF: {total_ar_inference_time / total_duration:.3f}")
        print(f"  Total audio duration: {total_duration:.2f}s")
        print(f"  Total inference time: {total_inference_time:.2f}s")
    
    print("🎉 LibriSpeech CosyVoice TTS evaluation completed!")

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()