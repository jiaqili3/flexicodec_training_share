import logging
import random
import pyarrow.parquet as pq
from io import BytesIO
import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import copy
import torch.distributed as dist
import os
from pathlib import Path
import pyloudnorm as pyln
import string
import time
import transformers

torchaudio.set_audio_backend("soundfile")
AUDIO_FORMAT_SETS = set(["flac", "mp3", "m4a", "ogg", "opus", "wav", "wma"])


def _build_semantic_model(semantic_model, mean_var_path, repcodec_model, repcodec_path):
    """Build the w2v semantic model and load pretrained weights."""
    import safetensors

    if semantic_model is not None:
        semantic_model = semantic_model.eval()
        layer_idx = 15
        output_idx = layer_idx + 2
        stat_mean_var = torch.load(mean_var_path)
        semantic_mean = stat_mean_var["mean"]
        semantic_std = torch.sqrt(stat_mean_var["var"])
        semantic_mean = semantic_mean
        semantic_std = semantic_std
    else:
        semantic_mean = semantic_std = layer_idx = output_idx = 0.0
    safetensors.torch.load_model(repcodec_model, repcodec_path)
    repcodec_model = repcodec_model.eval()
    # print("semantic mean: ", semantic_mean.cpu(), "semantic std: ", semantic_std.cpu())
    return {
        "model": semantic_model,
        "layer_idx": layer_idx,
        "output_idx": output_idx,
        "mean": semantic_mean,
        "std": semantic_std,
        "repcodec_model": repcodec_model,
    }



def extract_mel_vocos(data, sample_rate=24000, n_fft=1024, hop_length=256, n_mels=100, mode='train'):
    from vocoder_vocos.vocos.feature_extractors import MelSpectrogramFeatures
    extractor = MelSpectrogramFeatures(sample_rate, n_fft, hop_length, n_mels)
    """mel feature extraction using vocos impl"""
    for sample in data:
        assert sample['sample_rate'] == sample_rate
        sample['mel_feat'] = extractor(sample['speech'])
        yield sample


def segment_w2v(data, segment_length=5 * 50, mode="train"):
    """segmentation (for training codecs)"""
    print("segment length:", segment_length)
    for sample in data:
        if sample["speech_feat"].shape[0] <= segment_length:
            yield sample
        else:
            st = random.randint(0, sample["speech_feat"].shape[0] - segment_length - 1)
            ed = st + segment_length
            sample["speech_feat"] = sample["speech_feat"][st:ed]
            sample["speech_feat_mask"] = sample["speech_feat_mask"][st:ed]
            yield sample

def segment_speech(data, segment_length=5 * 24000, mode="train"):
    """Segment and pad speech data for training codecs."""
    print("Segment speech length:", segment_length)
    for sample in data:
        speech_length = sample["speech"].shape[-1]

        if speech_length <= segment_length:
            # Pad speech to match the segment length
            pad_width = segment_length - speech_length
            sample["speech"] = torch.nn.functional.pad(
                sample["speech"], 
                (0, pad_width),  # Pad at the end
                mode='constant', 
                value=0  # Pad with zeros
            )
            yield sample
        else:
            # Randomly crop the speech segment
            st = random.randint(0, speech_length - segment_length - 1)
            ed = st + segment_length
            sample["speech"] = sample["speech"][..., st:ed]
            yield sample

# def loudness_norm(
#     audio: torch.Tensor, rate: int, peak=-1.0, loudness=-23.0, block_size=0.400
# ) -> torch.Tensor:
#     """
#     Perform loudness normalization (ITU-R BS.1770-4) on audio files.

#     Args:
#         audio: audio data
#         rate: sample rate
#         peak: peak normalize audio to N dB. Defaults to -1.0.
#         loudness: loudness normalize audio to N dB LUFS. Defaults to -23.0.
#         block_size: block size for loudness measurement. Defaults to 0.400. (400 ms)

#     Returns:
#         loudness normalized audio
#     """
#     audio = audio.numpy()
#     # peak normalize audio to [peak] dB
#     audio = pyln.normalize.peak(audio, peak)

#     # measure the loudness first
#     meter = pyln.Meter(rate, block_size=block_size)  # create BS.1770 meter
#     _loudness = meter.integrated_loudness(audio)

#     return pyln.normalize.loudness(audio, _loudness, loudness)


def normalize(data, mode="train", en_punct=True, use_kana=False):
    """multilingual text normalize"""
    from soundstorm_ar.normalization import en, zh, ja

    for value in data:
        try:
            if value["language"] == "en":
                norm_text = en.normalize_en(value["text"])
            elif value["language"] == "zh":
                norm_text = zh.normalize_zh(value["text"], en_punct=en_punct)
            elif value["language"] == "ja":
                norm_text = ja.normalize_ja(
                    value["text"], en_punct=en_punct, use_kana=use_kana
                )  # has cn characters if use_kana=False
            else:
                norm_text = value["text"]
            # print(norm_text)
            value["text"] = norm_text
        except Exception as e:
            print(e)
            print('error raised from:', value)
            continue
        yield value


def w2v_feature(data, feature_extractor, mode="train", make_multiple_of=1):
    """
    Args:
        data(Iterable[str]): url or local file list
        w2v_path(str): wav2vec2.0 model path
        keep_speech(bool): whether to keep the speech waveform
    """
    for sample in data:
        if sample["sample_rate"] != 16000:
            resampler = torchaudio.transforms.Resample(sample["sample_rate"], 16000)
            tmp_speech = resampler(sample["speech"])
            feats = feature_extractor(tmp_speech, sampling_rate=16000, return_tensors='pt')
        else:
            feats = feature_extractor(sample["speech"], sampling_rate=16000, return_tensors='pt')
        
        if 'input_values' in feats: # hubert extractor
            sample["speech_feat"] = feats["input_values"].squeeze(0).squeeze(0) # (1,T)
            sample["speech_feat_mask"] = torch.ones(1,1)
            yield sample
        else: # wav2vec2.0 extractor
            sample["speech_feat"] = feats["input_features"][0]
            
            sample["speech_feat_mask"] = feats["attention_mask"][0]
            start_idx = sample["speech_feat"].shape[0] % make_multiple_of
            sample["speech_feat"] = sample["speech_feat"][start_idx:]
            sample["speech_feat_mask"] = sample["speech_feat_mask"][start_idx:]
            yield sample


def gluster_opener(data, mode="train", num_epochs=1, manual_dist_sampler=False, min_seconds=3.0, max_seconds=45.0):
    """
    WARNING: should set `manual_dist_sampler=False` if the datalist on each process is already disjoint.
    Set it to True if the datalist is the same across all procs, and you want distributed sampler.
    Skip the data that does not belong to this proc.
    """
    # Check if DEBUG mode is enabled
    debug_mode = os.environ.get("DEBUG", "").lower() in ("1", "true", "yes")
    debug_batches = []
    debug_batch_count = 0
    max_debug_batches = 10

    if manual_dist_sampler:
        assert dist.is_initialized(), "Distributed mode requires initialized process group"
        rank = dist.get_rank()  # Get the current process rank
        world_size = dist.get_world_size()  # Total number of processes
        print(f"[Rank {rank}] Initialized with manual_dist_sampler=True. Total processes: {world_size}.")
    else:
        rank = 0  # Default to rank 0 when not in distributed mode
        world_size = 1  # Treat as single process

    # Iterate through epochs
    for epoch in range(num_epochs):
        # Iterate through samples in the dataset
        for i, sample in enumerate(data):
            # Distributed sampling: only handle samples that match the current process
            if manual_dist_sampler and (i % world_size != rank):
                print(f"[Rank {rank}] Skipping sample {i} in epoch {epoch} (not assigned to this process).")
                continue

            # Create a new sample dictionary with the necessary modifications
            new_sample = copy.deepcopy(sample["src"])
            new_sample['epoch'] = epoch  # Add epoch information

            if new_sample['duration'] < min_seconds:
                continue
            if new_sample['duration'] > max_seconds:
                continue

            # In debug mode, store the first 10 batches
            if debug_mode:
                if debug_batch_count < max_debug_batches:
                    # Load and cache the audio data immediately
                    try:
                        from .gluster_dataset import load_audio_from_tar
                        new_sample["speech"], new_sample["sample_rate"] = load_audio_from_tar(
                            new_sample["wav"]
                        )
                        # Store the loaded audio data in the sample
                        new_sample["cached_audio"] = True
                        debug_batches.append(new_sample)
                        debug_batch_count += 1
                        if debug_batch_count == max_debug_batches:
                            print(f"[DEBUG] Collected and cached {max_debug_batches} batches for debug mode")
                    except Exception as e:
                        print(f"[DEBUG] Failed to load audio for sample: {e}")
                        continue
                # After collecting 10 batches, keep yielding them
                if debug_batch_count == max_debug_batches:
                    while True:
                        for batch in debug_batches:
                            # Return a copy of the cached batch
                            yield copy.deepcopy(batch)

            # Normal mode: yield the sample
            yield new_sample
        


def gluster_filter(
    data,
    max_length=10240,
    min_length=10,
    token_max_length=400,
    token_min_length=3,
    min_output_input_ratio=0.0005,
    max_output_input_ratio=0.8,
    ignore_text=False,  # if False, no filtering 'text_token' entry in sample
    mode="train",
    load_from_tar=True,
    make_multiple_of=1,
):
    """Filter sample according to feature and label length
    Inplace operation.

    Args::
        data: Iterable[{key, wav, label, sample_rate}]
        max_length: drop utterance which is greater than max_length(10ms)
        min_length: drop utterance which is less than min_length(10ms)
        token_max_length: drop utterance which is greater than
            token_max_length, especially when use char unit for
            english modeling
        token_min_length: drop utterance which is
            less than token_max_length
        min_output_input_ratio: minimal ration of
            token_length / feats_length(10ms)
        max_output_input_ratio: maximum ration of
            token_length / feats_length(10ms)

    Returns:
        Iterable[{key, wav, label, sample_rate}]
    """
    
    from .gluster_dataset import load_audio_from_tar
    for sample in data:
        # sample['speech'] = torch.randn(100000)
        new_sample = copy.deepcopy(sample)
        start_time = time.time()
        try:
            # Check if audio is already cached
            if "cached_audio" in new_sample and new_sample["cached_audio"]:
                # Use cached audio data
                new_sample["speech"] = new_sample["speech"]
                new_sample["sample_rate"] = new_sample["sample_rate"]
            else:
                # Load audio from disk
                if load_from_tar:
                    new_sample["speech"], new_sample["sample_rate"] = load_audio_from_tar(
                        new_sample["wav"]
                    )
                else:
                    new_sample["speech"], new_sample["sample_rate"] = torchaudio.load(
                        new_sample["wav"]
                    )

        except Exception as e:
            print(e)
            continue
        end_time = time.time()
        if (new_sample["speech"].shape[-1] // new_sample["sample_rate"]) > 35.0:
            print('too long audio, skipped')
            continue
        new_sample["speech"] = new_sample["speech"][..., new_sample["speech"].shape[-1]%make_multiple_of:]
        new_sample["load_audio_time"] = end_time - start_time
        if not ignore_text:
            num_frames = new_sample["speech"].size(1) / new_sample["sample_rate"] * 50
            text_token_ratio = len(new_sample["text_token"]) / num_frames
            if (
                text_token_ratio < min_output_input_ratio
                or text_token_ratio > max_output_input_ratio
            ):
                print('token ratio, skipped')
                continue
            if (
                len(new_sample["text_token"]) < token_min_length
                or len(new_sample["text_token"]) > token_max_length
            ):
                print('token max length, skipped')
                continue
        yield new_sample
        continue


def parquet_opener(data, mode="train", tts_data={}):
    """Give url or local file, return file descriptor
    Inplace operation.

    Args:
        data(Iterable[str]): url or local file list

    Returns:
        Iterable[{src, stream}]
    """
    for sample in data:
        assert "src" in sample
        url = sample["src"]
        try:
            df = pq.read_table(url).to_pandas()
            for i in range(len(df)):
                if mode == "inference" and df.loc[i, "utt"] not in tts_data:
                    continue
                sample.update(dict(df.loc[i]))
                if mode == "train":
                    # NOTE do not return sample directly, must initialize a new dict
                    yield {**sample}
                else:
                    for index, text in enumerate(tts_data[df.loc[i, "utt"]]):
                        yield {**sample, "tts_index": index, "tts_text": text}
        except Exception as ex:
            logging.warning("Failed to open {}, ex info {}".format(url, ex))


def filter(
    data,
    max_length=10240,
    min_length=10,
    token_max_length=200,
    token_min_length=1,
    min_output_input_ratio=0.0005,
    max_output_input_ratio=1,
    mode="train",
):
    """Filter sample according to feature and label length
    Inplace operation.

    Args::
        data: Iterable[{key, wav, label, sample_rate}]
        max_length: drop utterance which is greater than max_length(10ms)
        min_length: drop utterance which is less than min_length(10ms)
        token_max_length: drop utterance which is greater than
            token_max_length, especially when use char unit for
            english modeling
        token_min_length: drop utterance which is
            less than token_max_length
        min_output_input_ratio: minimal ration of
            token_length / feats_length(10ms)
        max_output_input_ratio: maximum ration of
            token_length / feats_length(10ms)

    Returns:
        Iterable[{key, wav, label, sample_rate}]
    """
    for sample in data:
        sample["speech"], sample["sample_rate"] = torchaudio.load(
            BytesIO(sample["audio_data"])
        )
        del sample["audio_data"]
        # sample['wav'] is torch.Tensor, we have 100 frames every second
        num_frames = sample["speech"].size(1) / sample["sample_rate"] * 100
        if num_frames < min_length:
            continue
        if num_frames > max_length:
            continue
        if len(sample["text_token"]) < token_min_length:
            continue
        if len(sample["text_token"]) > token_max_length:
            continue
        if len(sample["speech_token"]) == 0:
            continue
        if num_frames != 0:
            if len(sample["text_token"]) / num_frames < min_output_input_ratio:
                continue
            if len(sample["text_token"]) / num_frames > max_output_input_ratio:
                continue
        yield sample


def resample(data, resample_rate=22050, min_sample_rate=16000, mode="train"):
    """Resample data.
    Inplace operation.

    Args:
        data: Iterable[{key, wav, label, sample_rate}]
        resample_rate: target resample rate

    Returns:
        Iterable[{key, wav, label, sample_rate}]
    """
    for sample in data:
        assert "sample_rate" in sample
        assert "speech" in sample
        sample_rate = sample["sample_rate"]
        waveform = sample["speech"]
        if sample_rate != resample_rate:
            if sample_rate < min_sample_rate:
                continue
            sample["sample_rate"] = resample_rate
            sample["speech"] = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=resample_rate
            )(waveform)
        max_val = sample["speech"].abs().max()
        if max_val > 1:
            sample["speech"] /= max_val
        yield sample


def compute_fbank(data, feat_extractor, mode="train"):
    """Extract fbank

    Args:
        data: Iterable[{key, wav, label, sample_rate}]

    Returns:
        Iterable[{key, feat, label}]
    """
    for sample in data:
        assert "sample_rate" in sample
        assert "speech" in sample
        assert "utt" in sample
        assert "text_token" in sample
        waveform = sample["speech"]
        mat = feat_extractor(waveform).squeeze(dim=0).transpose(0, 1)
        sample["speech_feat"] = mat
        del sample["speech"]
        yield sample


def parse_embedding(data, normalize, mode="train"):
    """Parse utt_embedding/spk_embedding

    Args:
        data: Iterable[{key, wav, label, sample_rate}]

    Returns:
        Iterable[{key, feat, label}]
    """
    for sample in data:
        sample["utt_embedding"] = torch.tensor(
            sample["utt_embedding"], dtype=torch.float32
        )
        sample["spk_embedding"] = torch.tensor(
            sample["spk_embedding"], dtype=torch.float32
        )
        if normalize:
            sample["utt_embedding"] = F.normalize(sample["utt_embedding"], dim=0)
            sample["spk_embedding"] = F.normalize(sample["spk_embedding"], dim=0)
        yield sample

import random
import numpy as np

def add_white_noise(
    data, 
    mode="train", 
    min_snr_db=5, 
    max_snr_db=30, 
    noise_prob=0.5,
    full_band_prob=0.3,  # Probability of using full-band noise
    min_freq_range=[100, 3000],  # Range for randomly selecting min_freq
    max_freq_range=[3100, 12000]  # Range for randomly selecting max_freq
):
    """Add white noise to the audio signal with controllable SNR and frequency bands.
    
    Args:
        data: Iterable of samples containing audio data
        mode: 'train' or 'eval' mode
        min_snr_db: Minimum SNR in dB (higher SNR = less noise)
        max_snr_db: Maximum SNR in dB
        noise_prob: Probability of adding noise to each sample
        full_band_prob: Probability of using full-band noise
        min_freq_range: Range for randomly selecting minimum frequency
        max_freq_range: Range for randomly selecting maximum frequency
    """
    print('white noise module is ready.')
    for sample in data:
        if mode == "train" and random.random() < noise_prob:
            speech = sample["speech"]
            signal_power = torch.mean(speech ** 2)
            snr_db = random.uniform(min_snr_db, max_snr_db)
            snr = 10 ** (snr_db / 10)
            noise_power = signal_power / snr
            
            # Generate initial white noise matching speech shape
            noise = torch.randn_like(speech) * torch.sqrt(noise_power)
            
            # Decide whether to use full-band or frequency-selective noise
            if random.random() >= full_band_prob:
                # Apply frequency band filtering
                n_fft = 2048
                noise_spec = torch.stft(
                    noise.squeeze(0),
                    n_fft=n_fft,
                    hop_length=n_fft//4,
                    win_length=n_fft,
                    window=torch.hann_window(n_fft).to(noise.device),
                    return_complex=True
                )
                
                # Create frequency mask with random bands
                freqs = torch.fft.fftfreq(n_fft, d=1/sample["sample_rate"])[:n_fft//2 + 1]
                mask = torch.zeros_like(freqs)
                
                # Randomly select frequency bands
                min_freq = random.uniform(min_freq_range[0], min_freq_range[1])
                max_freq = random.uniform(max_freq_range[0], max_freq_range[1])
                
                # Ensure max_freq > min_freq
                if max_freq < min_freq:
                    min_freq, max_freq = max_freq, min_freq
                
                mask[(freqs >= min_freq) & (freqs <= max_freq)] = 1.0
                
                # Apply mask to noise spectrum
                mask = mask.unsqueeze(-1).to(noise_spec.device)
                noise_spec = noise_spec * mask
                
                # Convert back to time domain
                noise = torch.istft(
                    noise_spec,
                    n_fft=n_fft,
                    hop_length=n_fft//4,
                    win_length=n_fft,
                    window=torch.hann_window(n_fft).to(noise.device)
                )
                noise = noise.unsqueeze(0)
                
                # Handle length mismatch by repeating or trimming
                if noise.shape[-1] < speech.shape[-1]:
                    # Repeat noise to match length
                    repeats = (speech.shape[-1] + noise.shape[-1] - 1) // noise.shape[-1]
                    noise = noise.repeat(1, repeats)
                    noise = noise[..., :speech.shape[-1]]
                elif noise.shape[-1] > speech.shape[-1]:
                    # Trim excess noise
                    noise = noise[..., :speech.shape[-1]]
            
            # Add noise to signal
            sample["speech"] = speech + noise
            
            # Normalize if needed
            max_val = sample["speech"].abs().max()
            if max_val > 1:
                sample["speech"] = sample["speech"] / max_val
                
        yield sample


def g2p_phonemizer_en(
    data, 
    mode='train',
    mean=0.0,
    std=0.5,
    full_g2p_prob=0.2,
    shorten_g2p_sequence=True,
):
    """
    Transform sample['text'] into phonemes.
    Only applies to English or French text samples.
    This module should be placed after normalization for better accuracy.
    
    Parameters:
    - data: List of dictionaries containing the sample data, e.g., [{'text': 'Hello world', 'language': 'en'}, ...]
    - mode: Mode for processing ('train' or 'eval'). 
    - mean, std: the mean and std of the probability that some text are transformed to phone
    """
    from tools.g2p_phonemizer import g2p_phonemizer_en as processor

    for sample in data:
        if sample['language'] == 'en' or sample['language'] == 'fr':
            text = sample['text']
            words = text.split()  # Split the text into words
            
            if mode == 'train':
                # Generate a normal distribution to sample phone_ratio from it
                norm_phone_ratio = np.abs(np.random.normal(mean, std))

                # for a prob of `full_g2p_prob` prob, we transform all words
                if np.random.random() < full_g2p_prob:
                    norm_phone_ratio = 1.0

                if norm_phone_ratio == 1.0:
                    # transform the text together
                    transformed_words = processor(text, shorten_g2p_sequence=shorten_g2p_sequence)
                else:
                    # Ensure phone_ratio stays within [0, 1]
                    norm_phone_ratio = min(max(norm_phone_ratio, 0), 1)

                    # Determine how many words should be transformed based on the normalized phone_ratio
                    num_to_transform = round(len(words) * norm_phone_ratio)

                    # Ensure we do not sample more words than available
                    num_to_transform = min(num_to_transform, len(words))

                    words_to_transform = random.sample(words, num_to_transform)

                    try:
                        # Process the words to phonemes
                        transformed_words = [
                            processor(word, shorten_g2p_sequence=shorten_g2p_sequence) if word in words_to_transform and word not in string.punctuation else word 
                            for word in words
                        ]
                        transformed_words = ' '.join(transformed_words)
                    except Exception as e:
                        print('exception', e)
                        continue

                # Join the transformed words back into text
                sample['text_original'] = copy.deepcopy(sample['text'])
                sample['text'] = transformed_words

            elif mode == 'eval':
                # In eval mode, just apply phonemizer to the entire text
                sample['text'] = ' '.join([processor(word) if word not in string.punctuation else word for word in words])
        yield sample
    
    return data


def tokenize(
    data,
    get_tokenizer,
    allowed_special="all",
    mode="train",
    prepend_language_token=True,
):
    """Decode text to chars or BPE
    Inplace operation

    Args:
        data: Iterable[{key, wav, txt, sample_rate}]

    Returns:
        Iterable[{key, wav, txt, tokens, label, sample_rate}]
    """
    def is_english_string(s):
        # Define the set of allowed characters: all English letters and punctuation
        allowed_chars = set(
            string.ascii_letters + string.punctuation + string.whitespace
        )

        # Check if all characters in the string are within the allowed set
        return all(char in allowed_chars for char in s)

    tokenizer = get_tokenizer()
    for sample in data:
        assert "text" in sample
        sample["text"] = sample["text"].strip()

        # print(sample["text"])

        # for i in sample['text']:
        #     if tokenizer.encode(i).__len__() != 1:
        #         breakpoint()
        sample["text_token"] = tokenizer.encode(
            sample["text"], allowed_special=allowed_special
        )
        # if sample["language"] != "en" and is_english_string(sample["text"]):
        #     continue
        if prepend_language_token:
            sample["text_token"] = [
                tokenizer.to_language_token(sample["language"])
            ] + sample["text_token"]
        yield sample


def shuffle(data, shuffle_size=10000, mode="train"):
    """Local shuffle the data

    Args:
        data: Iterable[{key, feat, label}]
        shuffle_size: buffer size for shuffle

    Returns:
        Iterable[{key, feat, label}]
    """
    buf = []
    if dist.is_initialized():
        rank = dist.get_rank()
        print("RANK {} sort init".format(rank))
    else:
        rank = 0
    # shuffle_size += int(rank) * 1000

    for sample in data:
        buf.append(sample)
        if len(buf) >= shuffle_size:
            random.shuffle(buf)
            for x in buf:
                yield x
            buf = []
    # The sample left over
    random.shuffle(buf)
    for x in buf:
        yield x


def sort(data, sort_size=500, ignore_text=True, mode="train"):
    """Sort the data by feature length.
    Sort is used after shuffle and before batch, so we can group
    utts with similar lengths into a batch, and `sort_size` should
    be less than `shuffle_size`

    Args:
        data: Iterable[{key, feat, label}]
        sort_size: buffer size for sort
        ignore text: not calculate text token when sorting

    Returns:
        Iterable[{key, feat, label}]
    """
    if dist.is_initialized():
        rank = dist.get_rank()
        print("RANK {} sort init".format(rank))
    else:
        rank = 0
    buf = []

    if dist.is_initialized():
        rank = dist.get_rank()
        print("RANK {} sort init".format(rank))
    else:
        rank = 0
    # sort_size += int(rank) * 500

    for sample in data:
        buf.append(sample)
        if len(buf) >= sort_size:
            if not ignore_text:
                buf.sort(
                    key=lambda x: x["duration"] + len(x["text_token"]),
                    reverse=bool(rank % 2),
                )
            else:
                buf.sort(key=lambda x: x["duration"], reverse=bool(rank % 2))
            for x in buf:
                yield x
            buf = []
    # The sample left over
    buf.sort(key=lambda x: x["duration"])
    for x in buf:
        yield x


def static_batch(data, batch_size=16):
    """Static batch the data by `batch_size`

    Args:
        data: Iterable[{key, feat, label}]
        batch_size: batch size

    Returns:
        Iterable[List[{key, feat, label}]]
    """
    buf = []
    for sample in data:
        buf.append(sample)
        if len(buf) >= batch_size:
            yield buf
            buf = []
    if len(buf) > 0:
        yield buf


def dynamic_batch(data, max_frames_in_batch=12000, max_batch_size=50, mode="train", ignore_text=False, min_factor=0.95, max_factor=1.5):
    """Dynamic batch data with a quadratic exponent that scales based on sequence length."""
    buf = []
    longest_frames = 0

    for sample in data:
        new_sample_frames = sample["speech_feat"].shape[0]
        if not ignore_text:
            new_sample_frames += len(sample["text_token"])

        # Dynamically adjust the quadratic factor based on sequence length
        length_ratio = new_sample_frames / max_frames_in_batch
        quadratic_factor = min_factor + (max_factor - min_factor) * length_ratio  # Scales within [min_factor, max_factor]

        # Apply the dynamic quadratic factor to `new_sample_frames`
        adjusted_frames = int(new_sample_frames ** quadratic_factor)
        longest_frames = max(longest_frames, adjusted_frames)
        
        frames_after_padding = longest_frames * (len(buf) + 1)

        # Check batch size and frame constraints
        if frames_after_padding > max_frames_in_batch or len(buf) >= max_batch_size:
            if buf == []:
                yield [sample]
                longest_frames = 0
            else:
                yield buf
                buf = [sample]
                longest_frames = adjusted_frames
        else:
            buf.append(sample)
    
    if len(buf) > 0:
        yield buf


def batch(
    data,
    batch_type="static",
    batch_size=16,
    max_frames_in_batch=12000,
    mode="train",
    ignore_text=False,
):
    """Wrapper for static/dynamic batch"""
    if mode == "inference":
        return static_batch(data, 1)
    else:
        if batch_type == "static":
            return static_batch(data, batch_size)
        elif batch_type == "dynamic":
            return dynamic_batch(data, max_frames_in_batch, ignore_text=ignore_text)
        else:
            logging.fatal("Unsupported batch type {}".format(batch_type))


def gluster_padding(
    data, use_spk_embedding=False, ignore_text=False, return_speech=False, extract_spec=False, mode="train"
):
    """Padding the data into training data

    Args:
        data: Iterable[List[{key, feat, label}]]

    Returns:
        Iterable[Tuple(keys, feats, labels, feats lengths, label lengths)]
    """
    for sample in data:
        packed_batch_features = {}
        try:
            speech_feat = [i["speech_feat"] for i in sample]
            speech_feat = pad_sequence(speech_feat, batch_first=True, padding_value=0)
            packed_batch_features["input_features"] = speech_feat.contiguous()  # w2v features

            packed_batch_features["attention_mask"] = pad_sequence(
                [utt["speech_feat_mask"].float() for utt in sample], batch_first=True
            ).contiguous()

            packed_batch_features["speech_token_len"] = torch.tensor(
                [len(utt["speech_feat_mask"]) for utt in sample]
            ).contiguous()
        except Exception as e:
            print(e)

        if not ignore_text:
            text_token = [torch.tensor(i["text_token"]) for i in sample]
            text_token_len = torch.tensor(
                [i.size(0) for i in text_token], dtype=torch.int32
            )
            text_token = pad_sequence(text_token, batch_first=True, padding_value=0)
            packed_batch_features["text_token"] = text_token.contiguous()
            packed_batch_features["text_token_len"] = text_token_len.contiguous()

        if return_speech:
            packed_batch_features["speech"] = pad_sequence(
                [utt["speech"].squeeze(0) for utt in sample], batch_first=True
            ).contiguous()
            packed_batch_features["speech_lens"] = torch.tensor(
                [utt["speech"].squeeze(0).__len__() for utt in sample]
            ).contiguous()

        if extract_spec:
            import s3tokenizer
            mels = []
            for b in sample:
                # s3tokenizer uses 16khz mel
                if b['sample_rate'] != 16000:
                    b['speech'] = torchaudio.functional.resample(b['speech'], b['sample_rate'], 16000)
                
                audio = b['speech'][0]  # get the first channel
                mels.append(s3tokenizer.log_mel_spectrogram(audio))
            packed_batch_features['mels'], packed_batch_features['mels_lens'] = s3tokenizer.padding(mels)


        if not use_spk_embedding:
            packed_batch_features["embedding"] = None  # no speaker embedding

        for key in packed_batch_features.keys():
            if isinstance(packed_batch_features[key], torch.Tensor):
                if torch.isnan(packed_batch_features[key]).any():
                    print('NaN found in preprocessor, key', key)
                    continue
        packed_batch_features['epoch'] = sample[0]['epoch']
        packed_batch_features['duration'] = sum(s['duration'] for s in sample)
        packed_batch_features['load_audio_time'] = max(s['load_audio_time'] for s in sample)
        packed_batch_features['sample_rate'] = sample[0]['sample_rate']
        yield packed_batch_features


def padding(data, use_spk_embedding, mode="train"):
    """Padding the data into training data

    Args:
        data: Iterable[List[{key, feat, label}]]

    Returns:
        Iterable[Tuple(keys, feats, labels, feats lengths, label lengths)]
    """
    for sample in data:
        assert isinstance(sample, list)
        speech_feat_len = torch.tensor(
            [x["speech_feat"].size(1) for x in sample], dtype=torch.int32
        )
        order = torch.argsort(speech_feat_len, descending=True)

        utts = [sample[i]["utt"] for i in order]
        speech_token = [torch.tensor(sample[i]["speech_token"]) for i in order]
        speech_token_len = torch.tensor(
            [i.size(0) for i in speech_token], dtype=torch.int32
        )
        speech_token = pad_sequence(speech_token, batch_first=True, padding_value=0)
        speech_feat = [sample[i]["speech_feat"] for i in order]
        speech_feat_len = torch.tensor(
            [i.size(0) for i in speech_feat], dtype=torch.int32
        )
        speech_feat = pad_sequence(speech_feat, batch_first=True, padding_value=0)
        text = [sample[i]["text"] for i in order]
        text_token = [torch.tensor(sample[i]["text_token"]) for i in order]
        text_token_len = torch.tensor(
            [i.size(0) for i in text_token], dtype=torch.int32
        )
        text_token = pad_sequence(text_token, batch_first=True, padding_value=0)
        utt_embedding = torch.stack([sample[i]["utt_embedding"] for i in order], dim=0)
        spk_embedding = torch.stack([sample[i]["spk_embedding"] for i in order], dim=0)
        batch = {
            "utts": utts,
            "speech_token": speech_token,
            "speech_token_len": speech_token_len,
            "speech_feat": speech_feat,
            "speech_feat_len": speech_feat_len,
            "text": text,
            "text_token": text_token,
            "text_token_len": text_token_len,
            "utt_embedding": utt_embedding,
            "spk_embedding": spk_embedding,
        }
        if mode == "inference":
            tts_text = [sample[i]["tts_text"] for i in order]
            tts_index = [sample[i]["tts_index"] for i in order]
            tts_text_token = [torch.tensor(sample[i]["tts_text_token"]) for i in order]
            tts_text_token_len = torch.tensor(
                [i.size(0) for i in tts_text_token], dtype=torch.int32
            )
            tts_text_token = pad_sequence(
                tts_text_token, batch_first=True, padding_value=-1
            )
            batch.update(
                {
                    "tts_text": tts_text,
                    "tts_index": tts_index,
                    "tts_text_token": tts_text_token,
                    "tts_text_token_len": tts_text_token_len,
                }
            )
        if use_spk_embedding is True:
            batch["embedding"] = batch["spk_embedding"]
        else:
            batch["embedding"] = batch["utt_embedding"]
        yield batch
