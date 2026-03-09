import whisper
import torch
import numpy as np
import soundfile as sf

# Initialize whisper_model globally
whisper_model = None

def load_audio(file_path, sample_rate=16000):
    """
    Load audio file and resample it to 16kHz if necessary.

    Parameters:
    - file_path: Path to the audio file
    - sample_rate: Target sample rate for Whisper, 16kHz

    Returns:
    - audio: Audio data array sampled at 16kHz
    """
    audio, original_sample_rate = sf.read(file_path)
    
    # Resample if needed
    if original_sample_rate != sample_rate:
        from scipy.signal import resample
        num_samples = int(len(audio) * sample_rate / original_sample_rate)
        audio = resample(audio, num_samples)

    return np.float32(audio)

import whisper

@torch.no_grad()
def get_prompt_text(file_path):
    """
    Transcribe audio from a file using Whisper and return both the full transcription 
    and a shorter prompt based on the first few seconds.

    Parameters:
    - file_path: Path to the audio file

    Returns:
    - full_prompt_text: Full transcribed text
    - detected_language: Language detected by Whisper
    - shot_prompt_text: Text transcribed from the initial segment up to 4 seconds
    - short_prompt_end_ts: End timestamp of the short prompt
    """
    global whisper_model
    # Load Whisper model if not already loaded
    if whisper_model is None:
        whisper_model = whisper.load_model("turbo", device='cuda')

    # Load audio from file and convert to 16kHz for Whisper
    speech_16k = load_audio(file_path, sample_rate=16000)

    # Transcribe the audio
    asr_result = whisper_model.transcribe(speech_16k)
    full_prompt_text = asr_result["text"]
    detected_language = asr_result.get("language", "unknown")  # Get language if available

    # Extract short prompt based on initial segments up to 4 seconds
    segment_texts = []
    short_prompt_end_ts = 0.0
    for segment in asr_result["segments"]:
        segment_texts.append(segment['text'])
        short_prompt_end_ts = segment['end']
        if short_prompt_end_ts >= 4:
            break

    shot_prompt_text = "".join(segment_texts)
    return full_prompt_text, detected_language, shot_prompt_text, short_prompt_end_ts

# Example Usage
if __name__ == "__main__":
    # Replace 'path/to/audio.wav' with the actual audio file path
    file_path = "path/to/audio.wav"
    
    full_text, short_text, end_ts = get_prompt_text(file_path)
    
    print("Full Transcription:", full_text)
    print("Short Prompt:", short_text)
    print("Short Prompt End Timestamp:", end_ts)