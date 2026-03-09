import torch
from functools import lru_cache
import torchaudio
import torch.nn.functional as F
import time
from einops import rearrange

class Inference:
    def __init__(
        self, model, ckpt_path, cfg, device="cuda", normalize=False, half=False, split_paragraph=True, **kwargs
    ) -> None:
        self.model = model
        import safetensors.torch

        self.model.to(device)
        self.model.eval()
        safetensors.torch.load_model(self.model, ckpt_path, device=device)
        self.cfg = cfg

        for key in self.cfg.semantic_model:
            if isinstance(self.cfg.semantic_model[key], torch.nn.Module) or isinstance(
                self.cfg.semantic_model[key], torch.Tensor
            ):
                self.cfg.semantic_model[key] = self.cfg.semantic_model[key].to(device)
        self.device = device

        if (
            hasattr(self.cfg, "skip_semantic_normalize")
            and self.cfg.skip_semantic_normalize
        ):
            print("skip semantic normalize")

        # self.model = self.model.half()

    @torch.no_grad()
    def encode(
        self,
        audio,
        n_quantizers=8,
    ):
        """
            Generate text given speech and text prompts.

        Args:
            audio (str or Tensor): Speech file path or a tensor with shape (n_samples,).
            prompt_text (str): Text prompt.
            prompt_language (str): Language of the prompt.
            target_text (str): Target text to be completed.
            target_language (str): Language of the target text.
            use_prompt_text (bool, optional): Whether to use the prompt text as input. Defaults to True.
            temp (float, optional): Temperature parameter for the distribution. Defaults to 1.0.
            top_k (int, optional): Number of tokens to keep before applying `top_p`. Defaults to 1000.
            top_p (float, optional): Probability threshold to use for filtering tokens. Defaults to 0.85.

        Returns:
            codes: [b t q]
        """
        self.model.eval()

        audio_16k = torchaudio.functional.resample(audio, 24000, 16000)

        feature_extractor = self.cfg.feature_extractor
        inputs = feature_extractor(
            audio_16k, sampling_rate=16000, return_tensors="pt"
        )
        input_features = inputs["input_features"][0]
        attention_mask = inputs["attention_mask"][0]

        input_features = input_features.unsqueeze(0).to(self.device)
        attention_mask = attention_mask.unsqueeze(0).to(self.device)
        audio = audio.to(self.device)
        feat = self._extract_semantic_code(
            input_features, attention_mask
        ).transpose(1,2)

        feat = torch.nn.functional.avg_pool1d(feat, self.model.semantic_downsample_factor, self.model.semantic_downsample_factor)
        # out_dict, semantic_edict = self.model(audio, semantic_repr=feat,
        #                                         bypass_quantize_rate=0.,
        #                                         # possibly_no_quantizer=True, # internal dropout
        #                                         n_quantizers=n_quantizers,
        # )
        # return out_dict['x'], semantic_edict['codes']

        if self.model.override_dac_encoder:
            semantic_codes, acoustic_codes = self.model.encode(input_features, num_quantizers=n_quantizers, semantic_repr=feat)
        else:
            semantic_codes, acoustic_codes = self.model.encode(audio, num_quantizers=n_quantizers, semantic_repr=feat)
        
        return torch.cat([semantic_codes, acoustic_codes], dim=1).transpose(1,2) # [b t q]
    
    @torch.no_grad()
    def decode(self, codes):
        """
        input: codes [b t q]
        output: wav 
        """
        semantic_codes = codes[..., :1].transpose(1,2)
        acoustic_codes = codes[..., 1:].transpose(1,2)
        return self.model.decode_from_codes(semantic_codes, acoustic_codes)
    
    @torch.no_grad()
    def inference(
        self,
        audio,
        n_quantizers=8,
    ):
        """
        Generate text given speech and text prompts.

        Args:
            audio (str or Tensor): Speech file path or a tensor with shape (n_samples,).
            n_quantizers (int, optional): Number of quantizers to use. Defaults to 8.

        Returns:
            tuple: A tuple containing:
                - audio (Tensor): Decoded audio tensor.
                - semantic_codes (Tensor): Semantic codes.
                - encode_rtf (float): Encoding real-time factor.
                - decode_rtf (float): Decoding real-time factor.
        """
        self.model.eval()

        # Calculate the duration of the audio in seconds
        sample_rate = 24000  # Assuming the sample rate is 24 kHz
        duration = audio.shape[-1] / sample_rate  # Duration = number of samples / sample rate

        # Resample audio to 16 kHz
        audio_16k = torchaudio.functional.resample(audio, 24000, 16000)

        # Start encoding timer
        encode_start_time = time.time()

        # Extract features
        feature_extractor = self.cfg.feature_extractor
        inputs = feature_extractor(
            audio_16k, sampling_rate=16000, return_tensors="pt"
        )
        input_features = inputs["input_features"][0]
        attention_mask = inputs["attention_mask"][0]

        input_features = input_features.unsqueeze(0).to(self.device)
        attention_mask = attention_mask.unsqueeze(0).to(self.device)
        audio = audio.to(self.device)

        # Extract semantic codes
        feat = self._extract_semantic_code(
            input_features, attention_mask
        ).transpose(1, 2)

        feat = torch.nn.functional.avg_pool1d(
            feat, self.model.semantic_downsample_factor, self.model.semantic_downsample_factor
        )

        # Encode audio
        if self.model.override_dac_encoder:
            semantic_codes, acoustic_codes = self.model.encode(input_features, num_quantizers=n_quantizers, semantic_repr=feat)
        else:
            semantic_codes, acoustic_codes = self.model.encode(audio, num_quantizers=n_quantizers, semantic_repr=feat)

        # End encoding timer
        encode_end_time = time.time()
        encode_time = encode_end_time - encode_start_time
        encode_rtf = encode_time / duration  # RTF = encode_time / audio_duration

        # Start decoding timer
        decode_start_time = time.time()

        # Decode audio
        audio = self.model.decode_from_codes(semantic_codes, acoustic_codes)

        # End decoding timer
        decode_end_time = time.time()
        decode_time = decode_end_time - decode_start_time
        decode_rtf = decode_time / duration  # RTF = decode_time / audio_duration

        try:
            semantic_feature = self.model.semantic_vq.from_codes(semantic_codes)[0]
            semantic_feature = self.model.convnext_decoder(semantic_feature)
            semantic_feature = rearrange(semantic_feature, '1 h t -> t h')
        except:
            try:
                semantic_feature = self.model.quantizer.from_codes(semantic_codes)[0]
                semantic_feature = self.model.transform_model(semantic_feature)
                semantic_feature = rearrange(semantic_feature, '1 h t -> t h')
            except: 
                semantic_feature = None

        return audio, semantic_codes, encode_rtf, decode_rtf, semantic_feature
        

    @torch.no_grad()
    @torch.cuda.amp.autocast()
    def _extract_semantic_code(self, input_features, attention_mask):
        """
            从输入特征中提取语义编码。
        该函数不需要梯度，因此被标记为@torch.no_grad().

        Args:
            input_features (torch.Tensor, shape=(B, T, C)): 输入特征，其中B是batch size，T是时间维度，C是通道维度。
            attention_mask (torch.Tensor, shape=(B, T)): 注意力掩码，其中元素为0表示对应位置的特征无效，非0表示有效。

        Returns:
            tuple (torch.Tensor, shape=(B, T)): 返回一个元组，包含语义编码和对应的量化索引（可选）。
                - semantic_code (torch.Tensor, shape=(B, T)): 语义编码，其中B是batch size，T是时间维度。
                - rep_index (Optional, torch.Tensor, shape=(B, T)): 对于每个时间步骤，如果存在对应的量化索引，则返回这些索引；否则返回None。
        """
        vq_emb = self.cfg.semantic_model["model"](
            input_features=input_features,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        feat = vq_emb.hidden_states[self.cfg.semantic_model["output_idx"]]  # (B, T, C)

        if (
            hasattr(self.cfg, "skip_semantic_normalize")
            and self.cfg.skip_semantic_normalize
        ):
            pass
        else:
            feat = (feat - self.cfg.semantic_model["mean"]) / self.cfg.semantic_model[
                "std"
            ]
        return feat

def prepare_model():
    import hydra
    from hydra import initialize, initialize_config_module, initialize_config_dir, compose

    with initialize(version_base="1.3", config_path="../../../conf/"):
        cfg = compose(config_name="codec_infer_with_semantic.yaml", overrides=[])
        print(cfg)
    inference = hydra.utils.instantiate(cfg.inference)
    return inference
@torch.no_grad()
def infer(audio, model=None, num_quantizers=8):
    audio = audio.reshape(1,1,-1).cpu()
    out, codes, encode_rtf_, decode_rtf, semantic_feature = model.inference(audio, n_quantizers=num_quantizers)
    out = pad_to_length(out, audio.shape[-1])
    return out, codes, encode_rtf_, decode_rtf, semantic_feature

@torch.no_grad()
def infer_vae(audio, model=None, num_quantizers=8):
    audio = audio.reshape(1,1,-1).cuda()
    out_dict = model.model(audio)
    x = pad_to_length(out_dict.x, audio.shape[-1])
    return {
        'out': x,
        'latent': out_dict.z,
        'kl': out_dict.kl.cpu().item()
    }


def pad_to_length(x, length, pad_value=0):
    # Get the current size along the last dimension
    current_length = x.shape[-1]

    # If the length is greater than current_length, we need to pad
    if length > current_length:
        pad_amount = length - current_length
        # Pad on the last dimension (right side), keeping all other dimensions the same
        x_padded = F.pad(x, (0, pad_amount), value=pad_value)
    else:
        # If no padding is required, simply slice the tensor
        x_padded = x[..., :length]

    return x_padded
