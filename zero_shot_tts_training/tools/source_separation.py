import librosa
import os
import numpy as np
import torch
import torchaudio.transforms as T
from .models import separate_fast
import torchaudio
import soundfile as sf

def preprocess_audio_torch(waveform, original_sample_rate, target_sample_rate=16000) -> tuple[np.ndarray, int]:
    # Resample the waveform to the target sample rate (16kHz)
    resample = T.Resample(orig_freq=original_sample_rate, new_freq=target_sample_rate)
    waveform = resample(waveform)

    # Ensure the audio is mono (downmix stereo if needed)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Convert waveform to NumPy array for further processing
    waveform_np = waveform.numpy().flatten()

    # Convert to 16-bit PCM (like setting sample width to 2 bytes)
    # In PyTorch, we already work with float32 by default
    waveform_np = waveform_np.astype(np.float32)

    # Calculate dBFS (decibel relative to full scale) of the audio
    rms = np.sqrt(np.mean(waveform_np ** 2))
    dBFS = 20 * np.log10(rms) if rms > 0 else -float('inf')

    # Calculate the gain to be applied (target dBFS is -20)
    target_dBFS = -20
    gain = target_dBFS - dBFS
    print(f"Calculating the gain needed for the audio: {gain} dB")

    # Apply gain, limiting it between -3 and 3 dB
    gain = min(max(gain, -3), 3)
    waveform_np = waveform_np * (10 ** (gain / 20))

    # Normalize waveform (max absolute amplitude should be 1.0)
    max_amplitude = np.max(np.abs(waveform_np))
    if max_amplitude > 0:
        waveform_np /= max_amplitude

    print(f"waveform shape: {waveform_np.shape}")
    print(f"waveform dtype: {waveform_np.dtype}")

    return waveform_np, target_sample_rate

def source_separation(predictor, waveform_np, sample_rate):
    """
    Separate the audio into vocals and non-vocals using the given predictor.

    Args:
        predictor: The separation model predictor.
        audio (str or dict): The audio file path or a dictionary containing audio waveform and sample rate.

    Returns:
        dict: A waveform with 441000 Hz sample rate.
    """

    mix, rate = None, None

    # if isinstance(audio, str):
    #     mix, rate = librosa.load(audio, mono=False, sr=44100)
    # else:
    #     # resample to 44100
    rate = sample_rate
    mix = librosa.resample(waveform_np, orig_sr=rate, target_sr=44100)

    vocals, no_vocals = predictor.predict(mix)

    # convert vocals back to previous sample rate
    print(f"vocals shape before resample: {vocals.shape}")
    # vocals = librosa.resample(vocals.T, orig_sr=44100, target_sr=rate).T
    # no_vocals = librosa.resample(no_vocals.T, orig_sr=44100, target_sr=rate).T
    print(f"vocals shape after resample: {vocals.shape}")
    return vocals # , no_vocals

def source_separation_all(predictor, waveform_np, sample_rate):
    """
    Separate the audio into vocals and non-vocals using the given predictor.

    Args:
        predictor: The separation model predictor.
        audio (str or dict): The audio file path or a dictionary containing audio waveform and sample rate.

    Returns:
        dict: A waveform with 441000 Hz sample rate.
    """

    mix, rate = None, None

    # if isinstance(audio, str):
    #     mix, rate = librosa.load(audio, mono=False, sr=44100)
    # else:
    #     # resample to 44100
    rate = sample_rate
    mix = librosa.resample(waveform_np, orig_sr=rate, target_sr=44100)

    vocals, no_vocals = predictor.predict(mix)

    # convert vocals back to previous sample rate
    print(f"vocals shape before resample: {vocals.shape}")
    # vocals = librosa.resample(vocals.T, orig_sr=44100, target_sr=rate).T
    # no_vocals = librosa.resample(no_vocals.T, orig_sr=44100, target_sr=rate).T
    print(f"vocals shape after resample: {vocals.shape}")
    return vocals, no_vocals


def init_separate_model():
    print(" * Loading Background Noise Model")
    step1= {
        "model_path": "models/separate_model/UVR-MDX-NET-Inst_HQ_3.onnx",
        "denoise": True,
        "margin": 44100,
        "chunks": 15,
        "n_fft": 6144,
        "dim_t": 8,
        "dim_f": 3072
    }
    separate_predictor = separate_fast.Predictor(
        os.path.join("/ssd2/lizhekai/code/AudioPipeline/models/", "separate_model/UVR-MDX-NET-Inst_HQ_3.onnx"),
        args=step1, 
        device="cuda",
    )
    return separate_predictor

separate_predictor = init_separate_model()

def denoise(waveform, original_sample_rate, target_sample_rate = 24000) -> np.ndarray:

#     waveform_np, sample_rate = preprocess_audio_torch(waveform, original_sample_rate, target_sample_rate)
    vocal = source_separation(separate_predictor, waveform, original_sample_rate)
    return vocal

if __name__ == "__main__":
    speech_path = "/root/code/DCA_LC_3_1.wav"
    print(" * Preprocessing Audio")
    waveform, original_sample_rate = torchaudio.load(speech_path)
    # Tensor to ndarray
    waveform_np = waveform.squeeze(0).numpy()
    # breakpoint()
    vocal = denoise(waveform_np, original_sample_rate)
    # save audio
    print(" * Saving Audio")
    sf.write("vocal.wav", vocal, samplerate=44100)