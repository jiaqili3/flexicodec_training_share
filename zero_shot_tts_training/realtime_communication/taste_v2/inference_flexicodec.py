
import torch
import time
import re
from zero_shot_tts_training.realtime_communication.taste_v2.modeling_flexicodec import FlexiCodec
get_params = lambda model: sum(p.numel() for p in model.parameters()) / 1e6
from pathlib import Path
def prepare_model():
    # model_path = '/mnt/lijiaqi/exp/v0/dualcodecsensevoice_12hzv2/dualcodecsensevoice_12hzv2_20251027/checkpoint/epoch-0001_step-0810000_loss-0.000000-dualcodecsensevoice_12hzv2'
    
    # model_path = '/mnt/lijiaqi/exp/v0/dualcodecsensevoice_12hzv2_flexi/dualcodecsensevoice_12hzv2_flexi_20251028/checkpoint/epoch-0000_step-0240000_loss-0.000000-dualcodecsensevoice_12hzv2_flexi'
    # model_path = '/mnt/lijiaqi/exp/v0/dualcodecsensevoice_12hzv2_large/dualcodecsensevoice_12hzv2_large_20251031/checkpoint/epoch-0000_step-0360000_loss-0.000000-dualcodecsensevoice_12hzv2_large'
    model_path = '/mnt/lijiaqi/exp/v0/dualcodecsensevoice_12hzv2_large_learnable/dualcodecsensevoice_12hzv2_large_learnable_20251031/checkpoint/epoch-0000_step-0330000_loss-0.000000-dualcodecsensevoice_12hzv2_large_learnable'


    model_name = Path(model_path).parent.parent.name
    match = re.search(r'_step-(\d+)', model_path)
    step_info = "unknown"
    if match:
        # Extract the numeric part (e.g., '0060000')
        step_number_str = match.group(1)
        # Convert to an integer to perform math (removes leading zeros)
        step_number_int = int(step_number_str)
        # --- 3. Format the number to 'k' notation ---
        # Check if the number is a multiple of 1000 and not zero
        if step_number_int > 0 and step_number_int % 1000 == 0:
            # Divide by 1000 and format as an integer string with 'k'
            step_info_formatted = f"{step_number_int // 1000}k"
        else:
            # If not a clean multiple of 1000, just use the number
            step_info_formatted = str(step_number_int)
    # --- 3. Append the step info to the model_name ---
    new_model_name = f"{model_name}_step{step_info_formatted}"

    # model_path = '/mnt/lijiaqi/exp/v0/dualcodecsensevoice_8hz/dualcodecsensevoice_8hz_20251022/checkpoint/epoch-0000_step-0800000_loss-0.000000-dualcodecsensevoice_8hz'
    return FlexiCodec.from_pretrained_custom(model_path).cuda(), new_model_name

@torch.no_grad()
def infer_sensevoice(audio, model):
    pass
@torch.no_grad()
def infer(audio, model=None, num_quantizers=8):
    # Calculate the duration of the audio in seconds
    sample_rate = 24000  # Assuming the sample rate is 24 kHz
    duration = audio.shape[-1] / sample_rate  # Duration = number of samples / sample rate
    print(f"Input audio duration: {duration:.2f} seconds") # <<< Added for context
    dl_output = {
        "audio": audio.cuda(),
        "num_quantizers": num_quantizers,
        "manual_threshold": 0.91,
    }
    # --- Timing the Encoding Step ---
    encode_start_time = time.time() # <<< Start timer for encoding
    # Encode the audio to get semantic and acoustic codes
    encoded_output = model(
        dl_output,
        encode_only=False, # This seems to control both encoding and potentially decoding path
    )
    encode_end_time = time.time() # <<< End timer for encoding
    encode_duration = encode_end_time - encode_start_time # <<< Calculate encoding time
    print(f"Time used for encoding: {encode_duration:.4f} seconds") # <<< Print encoding time
    # Initialize decode_duration in case the decoding step is skipped
    decode_duration = 0.0 # <<< Initialize decode time
    if "x" in encoded_output:
        # This block seems to be a combined encode/decode or just an encode path
        # The model call above already produced the final audio 'x'
        print("Decoding was likely performed inside the main model call. Total time measured as 'encoding'.")
        reconstructed_audio = encoded_output["x"]
        token_ratio = 1.0
        semantic_features = encoded_output.get("semantic_features", None)
        semantic_codes = encoded_output.get("semantic_codes", None)
        token_lengths = encoded_output['token_lengths']
        sim = encoded_output.get('sim', None)
    else:
        # This block has separate encoding and decoding steps
        # Extract the codes and token lengths from the encoding step
        semantic_codes = encoded_output['semantic_codes']
        acoustic_codes = encoded_output['acoustic_codes']
        token_lengths = encoded_output['token_lengths']
        alignmnet_matrix = encoded_output['alignment_matrix']
        sim = encoded_output.get('sim', None)
        # --- Timing the Decoding Step ---
        decode_start_time = time.time() # <<< Start timer for decoding
        # Decode from codes to reconstruct the audio
        reconstructed_audio = model.decode_from_codes(
            semantic_codes=semantic_codes,
            acoustic_codes=acoustic_codes,
            token_lengths=token_lengths,
        )
        decode_end_time = time.time() # <<< End timer for decoding
        decode_duration = decode_end_time - decode_start_time # <<< Calculate decoding time
        # print(f"Time used for decoding: {decode_duration:.4f} seconds") # <<< Print decoding time
        token_ratio = (alignmnet_matrix.shape[1] / alignmnet_matrix.shape[2])
        semantic_features = encoded_output.get("semantic_features", None)
    # You might want to return these values for more systematic logging
    return {
        "out": reconstructed_audio.cpu().to(torch.float32),
        "compressed": semantic_codes,
        # "compressed": encoded_output['acoustic_codes'][0],
        "encode_rtf": token_ratio, # Note: This variable name might be misleading now
        "decode_rtf": token_ratio, # Note: This variable name might be misleading now
        "semantic_features": semantic_features,
        "token_lengths": token_lengths,
        "sim": sim,
        "encode_time_sec": encode_duration, # <<< Added encode time to the output dictionary
        "decode_time_sec": decode_duration, # <<< Added decode time to the output dictionary
    }


# Example usage
if __name__ == '__main__':
    import torch
    import torchaudio
    model = prepare_model()
    # audio, sr = torchaudio.load("/mnt/lijiaqi/LibriSpeech/test-clean/121/123852/121-123852-0001.flac")
    audio, sr = torchaudio.load("/mnt/lijiaqi/seedtts_eval/seedtts_testset/zh/wavs/10004423-00000048.wav")
    audio = torchaudio.functional.resample(audio, sr, 24000)
    out = infer(audio=audio, model=model, num_quantizers=6)
    torchaudio.save("out.wav", out["out"].squeeze(1).cpu(), 24000)
    breakpoint()