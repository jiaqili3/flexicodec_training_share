import numpy as np
import torch
from typing import Dict, Optional, Union, List, Tuple
from torch import nn
import torch.nn.functional as F
import torchaudio
from torch.nn.utils.rnn import pad_sequence, unpad_sequence
from funasr import AutoModel
from funasr.utils.misc import deep_update
import librosa
import logging
import os
from einops import rearrange
from transformers import WhisperModel, WhisperProcessor
from easydict import EasyDict as edict
from .model_utils import load_whisper_whole_model, get_s3_encoder_dict
RTSLM_WORK_DIR = os.getenv(
    'RTSLM_WORK_DIR'
)
from .quantize.bsq import BinarySphericalQuantizer, SimpleQuantizer
from zero_shot_tts_training.realtime_communication.codec_model.dac_quantize import ResidualVectorQuantize

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseAudioEncoder(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    # process that should be done during data collation
    # TODO: should be moved to other place

    def to(self, device):
        self.device = device
        return super().to(device)
    
    def get_device(self):
        if next(self.parameters(), None) is not None:
            return next(self.parameters()).device
        elif next(self.buffers(), None) is not None:
            return next(self.buffers()).device
        else:
            return 'cpu'
        
    def extract_feature(
        self,
        audio_fpaths: List[str],
    ):
        raise NotImplementedError

    # Parameter-related functions
    def forward(
        self,
        audio_features: torch.Tensor,
        audio_features_lengths: torch.Tensor,
        **kwargs,
    ):
        raise NotImplementedError

class SenseVoiceAudioEncoder(BaseAudioEncoder):
    def __init__(
        self,
        model_card: str = "iic/SenseVoiceSmall",
        model_code_dir: str = "customized_sensevoice/model.py",
        dither: float = 1.0, # If you don't want to use dither in kaldi.fbank, set it to 0.0 for reproducability
        hub: str = "ms",
        prepend_inputs_before_encoding: bool = False, 
        extract_hidden: bool = True,
        transform_type: str = 'linear',
        transform_out_dim: int = 1024,
        use_vq: bool = True,
        vq_type: str = "rvq",
        codebook_size: int = 16384,
        codebook_dim: int = 8,
        n_codebooks: int = 16,
        quantizer_dropout: float = 1.0,
        alignment_mode: str = 'ctc', # 'ctc' or 'similarity'
        similarity_threshold: float = 0.9,
    ):
        # override model_code_dir
        from pathlib import Path
        model_code_dir = f'{str(Path(__file__).parent)}/customized_sensevoice/model.py'
        super().__init__()

        self.funasr_model = AutoModel(
            model=model_card,
            trust_remote_code=True,
            remote_code=model_code_dir,
            hub=hub,
            device="cpu",
            disable_update=True
        )

        self.kwargs = self.funasr_model.kwargs
        logger.info(self.kwargs)
        # separate components for flexible usage. 
        self.frontend = self.kwargs["frontend"] # fbank
        self.frontend.dither = dither
        # I examined and it is a SentencepiecesTokenizer (built by themselves)
        self.text_tokenizer = self.kwargs["tokenizer"]
        self.model = self.funasr_model.model
        self.feature_dim = self.kwargs['encoder_conf']['output_size']
        self.prepend_inputs_before_encoding  = prepend_inputs_before_encoding
        if self.prepend_inputs_before_encoding:
            logger.info("Will automatically prepend SenseVoice input tokens before encoding!")
        self.extract_hidden = extract_hidden
        if self.extract_hidden:
            logger.info("Will extract hidden repr from SenseVoice-Small (before tp_encoders, after encoders)")

        self.transform_type = transform_type
        if self.transform_type == 'linear':
            self.transform = nn.Linear(self.feature_dim, transform_out_dim)
        else:
            raise ValueError(f"Invalid transform type: {self.transform_type}")

        self.use_vq = use_vq
        self.vq_type = vq_type
        if self.use_vq:
            if self.vq_type == "rvq":
                self.rvq = ResidualVectorQuantize(
                    input_dim=transform_out_dim,
                    n_codebooks=n_codebooks,
                    codebook_size=codebook_size,
                    codebook_dim=codebook_dim,
                    quantizer_dropout=quantizer_dropout
                )
            elif self.vq_type == "bsq":
                # codebook size 2^vq_emb_dim, 2^14 = 16384
                vq_emb_dim = int(np.log2(codebook_size))
                self.bsq = BinarySphericalQuantizer(
                    embed_dim=vq_emb_dim,
                )
            else:
                raise ValueError(f"Invalid VQ type: {self.vq_type}")

        self.alignment_mode = alignment_mode
        self.similarity_threshold = similarity_threshold
        # freeze funasr
        self.freeze_funasr_model()

    def freeze_funasr_model(self):
        """Freeze all parameters in the FunASR model."""
        for param in self.model.parameters():
            param.requires_grad = False

    def extract_feature(
        self,
        audio_fpaths: List[str],
        **cfg,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cfg['device'] = cfg.get("device", self.get_device())
        meta_data, audio_features, audio_features_lengths = self.model.prepare_inputs(
            audio_fpaths,
            data_lengths=None,
            tokenizer=self.text_tokenizer,
            frontend=self.frontend,
            **cfg,
        )
        # print(meta_data)
        return audio_features, audio_features_lengths

    def _perform_ctc_alignment(self, raw_yseq):
        yseq_unique_consecutive = torch.unique_consecutive(raw_yseq)
        is_change_raw = torch.cat([torch.tensor([True], device=raw_yseq.device), raw_yseq[1:] != raw_yseq[:-1]])
        frame_to_segment_map = torch.cumsum(is_change_raw, dim=0) - 1
        return yseq_unique_consecutive, frame_to_segment_map

    def _perform_similarity_alignment(self, h_frames, raw_yseq, ctc_logits):
        speech_feat_len = h_frames.shape[0]
        if speech_feat_len <= 1:
            return torch.unique_consecutive(raw_yseq), torch.zeros(speech_feat_len, dtype=torch.long, device=h_frames.device)

        # 1. Vectorized similarity calculation
        sim = F.cosine_similarity(h_frames[:-1], h_frames[1:], dim=1)
        # ensure first four special tokens are not grouped
        sim[:3] = 0.0

        # 2. Vectorized grouping
        is_new_group_boundary = sim <= self.similarity_threshold
        is_new_group_padded = torch.cat([
            torch.tensor([True], device=h_frames.device), 
            is_new_group_boundary
        ])
        frame_to_segment_map = torch.cumsum(is_new_group_padded.long(), dim=0) - 1
        num_segments = frame_to_segment_map[-1].item() + 1
        
        # 3. Create pseudo-labels for segments
        yseq_unique_consecutive = torch.arange(num_segments, device=h_frames.device, dtype=torch.long)
        
        return yseq_unique_consecutive, frame_to_segment_map

    @torch.no_grad()
    def forward_encoder(
        self,
        audio_features: torch.Tensor,
        audio_features_lengths: torch.Tensor,
        return_text: bool = False,
        **kwargs,
    ) -> Dict[str, Optional[torch.Tensor]]:
        '''
        Args:
            audio_features: torch.Tensor, shape (batch_size, max_audio_len, audio_feat_dim)
            audio_features_lengths: torch.Tensor, shape (batch_size,)
        '''
        if self.prepend_inputs_before_encoding:
            audio_features, audio_features_lengths = self.model.prepend_inputs(audio_features, audio_features_lengths, **kwargs)
        encoder_out, encoder_out_lengths, hidden_out, hiddens = self.model.encoder(audio_features, audio_features_lengths, extract_hidden=self.extract_hidden)

        if isinstance(encoder_out, tuple):
            encoder_out = encoder_out[0]
        
        if isinstance(hidden_out, torch.Tensor):
            # print(f"Use hidden_out!, enc_out_shape={encoder_out.shape}, hidden_out_shape={hidden_out.shape}")
            assert encoder_out.shape[1] == hidden_out.shape[1], f"length between endoer out and hidden out does not match ({encoder_out.shape}, {hidden_out.shape})."
        # generate ctc output

        ctc_logits = self.model.ctc.log_softmax(encoder_out)
        texts = []
        alignments = []
        all_decoded_tokens = []
        for i in range(ctc_logits.size(0)):
            x = ctc_logits[i, : encoder_out_lengths[i].item(), :]
            raw_yseq = x.argmax(dim=-1)
            speech_feat_len = len(raw_yseq)

            if self.alignment_mode == 'similarity':
                h_frames = hidden_out[i, :encoder_out_lengths[i].item(), :]
                yseq_unique_consecutive, frame_to_segment_map = self._perform_similarity_alignment(h_frames, raw_yseq, ctc_logits)
            
            else: # ctc mode
                yseq_unique_consecutive, frame_to_segment_map = self._perform_ctc_alignment(raw_yseq)

            # The following logic is common for both modes
            if self.alignment_mode == 'ctc':
                is_segment_blank = (yseq_unique_consecutive == self.model.blank_id)
                segment_tokens = yseq_unique_consecutive[~is_segment_blank]
            else: # similarity mode
                is_segment_blank = torch.zeros_like(yseq_unique_consecutive, dtype=torch.bool)
                segment_tokens = yseq_unique_consecutive

            if len(segment_tokens) == 0:
                # Handle case of empty transcription
                texts.append("")
                alignments.append(torch.empty(0, speech_feat_len, device=raw_yseq.device, dtype=torch.int))
                continue

            is_change_in_segment_tokens = torch.cat([torch.tensor([True], device=raw_yseq.device), segment_tokens[1:] != segment_tokens[:-1]])
            text_tokens = segment_tokens[is_change_in_segment_tokens]
            text_token_len = len(text_tokens)

            # Map segments to final text tokens
            if self.alignment_mode == 'ctc':
                text_token_indices_for_segments = torch.cumsum(is_change_in_segment_tokens, dim=0) - 1
                segment_to_text_map = torch.full((len(yseq_unique_consecutive),), -1, device=raw_yseq.device, dtype=torch.long)
                segment_to_text_map[~is_segment_blank] = text_token_indices_for_segments

                # Handle blank segments by assigning them to nearest non-blank token
                for seg_idx in range(len(yseq_unique_consecutive)):
                    if segment_to_text_map[seg_idx] == -1:  # This is a blank segment
                        # Find nearest non-blank segment
                        left_idx = seg_idx - 1
                        right_idx = seg_idx + 1
                        
                        # Look for nearest non-blank to the left
                        while left_idx >= 0 and segment_to_text_map[left_idx] == -1:
                            left_idx -= 1
                        
                        # Look for nearest non-blank to the right
                        while right_idx < len(yseq_unique_consecutive) and segment_to_text_map[right_idx] == -1:
                            right_idx += 1
                        
                        # Assign to nearest non-blank token
                        if left_idx >= 0 and right_idx < len(yseq_unique_consecutive):
                            # Both sides have non-blanks, choose the closer one (prefer right)
                            segment_to_text_map[seg_idx] = segment_to_text_map[right_idx]
                        elif left_idx >= 0:
                            # Only left side has non-blank
                            segment_to_text_map[seg_idx] = segment_to_text_map[left_idx]
                        elif right_idx < len(yseq_unique_consecutive):
                            # Only right side has non-blank
                            segment_to_text_map[seg_idx] = segment_to_text_map[right_idx]
            else: # similarity mode
                segment_to_text_map = torch.arange(len(yseq_unique_consecutive), device=raw_yseq.device)

            # Combine maps to get per-frame text token index
            frame_to_text_token_map = segment_to_text_map[frame_to_segment_map]

            # Create the alignment matrix
            alignment_matrix = torch.zeros((text_token_len, speech_feat_len), device=raw_yseq.device, dtype=torch.int)
            # Now all frames should have valid token assignments
            frame_indices = torch.arange(speech_feat_len, device=raw_yseq.device)
            valid_token_mask = frame_to_text_token_map >= 0
            alignment_matrix[frame_to_text_token_map[valid_token_mask], frame_indices[valid_token_mask]] = 1
            alignments.append(alignment_matrix)

            # Step 5: Decode tokens for plotting and text for verification
            if return_text:
                if self.alignment_mode == 'ctc':
                    decoded_tokens = [self.text_tokenizer.decode([t]) for t in text_tokens.tolist()]
                    text = self.text_tokenizer.decode(text_tokens.tolist())
                else: # similarity mode
                    decoded_tokens = [f"group_{t.item()}" for t in text_tokens]
                    text = " ".join(decoded_tokens)
                    
                all_decoded_tokens.append(decoded_tokens)
                texts.append(text)

        if self.extract_hidden:
            return {
                'ctc_logits': ctc_logits,
                'encoded_feats': encoder_out,
                'half_hidden_feats': hidden_out,
                'encoded_feats_lengths': encoder_out_lengths,
                "text": texts,
                "alignments": alignments,
                "decoded_tokens": all_decoded_tokens,
            }
        else:
            return {
                'ctc_logits': ctc_logits,
                'encoded_feats': encoder_out,
                'encoded_feats_lengths': encoder_out_lengths,
                "text": texts,
                "alignments": alignments,
                "decoded_tokens": all_decoded_tokens,
            }

    

    def aggregate_semantic(
        self,
        hidden_features: torch.Tensor,
        alignment_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """
        Aggregate hidden features from speech frames to text tokens using alignment matrix.
        This is a vectorized implementation for a batch.
        
        Args:
            hidden_features: torch.Tensor, shape (batch_size, feat_len, hidden_dim)
                Hidden features from the encoder (e.g., from half-layer depth)
            alignment_matrix: torch.Tensor, shape (batch_size, txt_len, feat_len)
                Padded binary alignment matrices for the batch where 1 indicates
                token-frame correspondence.
                
        Returns:
            torch.Tensor, shape (batch_size, txt_len, hidden_dim)
                Aggregated semantic features for each text token
        """
        # transform the hidden features
        hidden_features = self.transform(hidden_features)

        # Ensure alignment matrix is float and on the correct device
        alignment_float = alignment_matrix.to(hidden_features.device, dtype=hidden_features.dtype)
        
        # Calculate the sum of features for each token via vectorized operation.
        # einsum `btf,bfh->bth` performs a batch matrix multiplication.
        # It sums the features in hidden_features (h) for each frame (f) that belongs
        # to a given token (t), for each item in the batch (b).
        summed_features = torch.einsum('btf,bfh->bth', alignment_float, hidden_features)
        
        # Calculate the number of frames assigned to each token for each batch item.
        # Shape: (batch_size, txt_len)
        token_frame_counts = alignment_float.sum(dim=2)
        
        # To avoid division by zero for tokens with no frames, clamp counts to a minimum of 1.
        # The numerator (summed_features) will be zero for these tokens anyway.
        token_frame_counts = token_frame_counts.clamp(min=1)
        
        # Reshape counts for broadcasting over the hidden dimension.
        # Shape becomes (batch_size, txt_len, 1)
        counts_reshaped = token_frame_counts.unsqueeze(-1)
        
        # Compute the average by dividing the summed features by the counts.
        aggregated_features = summed_features / counts_reshaped
        
        return aggregated_features

    def deaggregate_semantic(
        self,
        semantic_features: torch.Tensor,
        alignment_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """
        De-aggregate semantic features from text tokens back to speech frames using alignment matrix.
        This is the inverse operation of aggregate_semantic.
        
        Args:
            semantic_features: torch.Tensor, shape (batch_size, txt_len, hidden_dim)
                Semantic features for each text token (e.g., from VQ).
            alignment_matrix: torch.Tensor, shape (batch_size, txt_len, feat_len)
                Padded binary alignment matrices for the batch where 1 indicates
                token-frame correspondence.
                
        Returns:
            torch.Tensor, shape (batch_size, feat_len, hidden_dim)
                Expanded features for each speech frame.
        """
        # Ensure alignment matrix is float and on the correct device
        alignment_float = alignment_matrix.to(semantic_features.device, dtype=semantic_features.dtype)
        
        # Use einsum to perform the de-aggregation.
        # 'bth,btf->bfh' means for each batch item (b), it multiplies the token-level
        # hidden features (h) with the alignment matrix (which maps tokens 't' to frames 'f')
        # and sums over the token dimension 't'. Since the alignment is one-hot for each frame,
        # this effectively selects the correct token feature for each frame.
        expanded_features = torch.einsum('bth,btf->bfh', semantic_features, alignment_float)
        
        # remove the first four label tokens
        expanded_features = expanded_features[:, 4:, :]
        return expanded_features

    def forward(
        self,
        audio_features: torch.Tensor,
        audio_features_lengths: torch.Tensor,
        return_text: bool = False,
        **kwargs,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Full forward pass: encoding -> semantic aggregation -> vector quantization
        
        Args:
            audio_features: torch.Tensor, shape (batch_size, max_audio_len, audio_feat_dim)
            audio_features_lengths: torch.Tensor, shape (batch_size,)
            return_text: bool, whether to return decoded text
            
        Returns:
            Dict containing:
                - All outputs from forward_encoder
                - aggregated_semantic_features: aggregated semantic features
                - If use_vq=True: quantized_features, vq_codes, vq_losses
        """
        # Step 1: Forward through encoder
        encoder_results = self.forward_encoder(
            audio_features, 
            audio_features_lengths, 
            return_text=return_text,
            **kwargs
        )
        
        # Step 2: Aggregate semantic features using alignments
        # Use half_hidden_feats if available, otherwise use encoded_feats
        if self.extract_hidden and 'half_hidden_feats' in encoder_results:
            features_for_aggregation = encoder_results['half_hidden_feats']
        else:
            features_for_aggregation = encoder_results['encoded_feats']
            
        batch_size = features_for_aggregation.shape[0]
        alignments = encoder_results.get('alignments')

        # To use pad_sequence, we first need to pad each 2D alignment
        # matrix to the same feature length.
        max_feat_len = features_for_aggregation.shape[1]
        padded_feat_alignments = [
            F.pad(a, (0, max_feat_len - a.shape[1]), "constant", 0) for a in alignments
        ]

        # Now that they only vary in the first dimension (text length), 
        # we can use pad_sequence.
        padded_alignments = pad_sequence(padded_feat_alignments, batch_first=True, padding_value=0.0)

        # Aggregate semantic features for the whole batch
        aggregated_semantic_features = self.aggregate_semantic(
            features_for_aggregation,
            padded_alignments
        )
        
        # Add to results
        encoder_results['aggregated_semantic_features'] = aggregated_semantic_features
        
        # Step 4: Vector Quantization (if enabled)
        if self.use_vq and aggregated_semantic_features.numel() > 0:
            # Transpose for VQ: (B, T, D) -> (B, D, T)
            vq_input = aggregated_semantic_features.transpose(1, 2)
            
            quantized_semantic_features = None

            if self.vq_type == "rvq":
                z_q, codes, latents, commitment_loss, codebook_loss, _ = self.rvq(vq_input)
                # Transpose back: (B, D, T) -> (B, T, D)
                quantized_semantic_features = z_q.transpose(1, 2)
                
                encoder_results['vq'] = edict({
                    'x': quantized_semantic_features,
                    'codes': codes,
                    'latents': latents,
                    'penalty': commitment_loss,
                    'vq/codebook_loss': codebook_loss,
                })
                
            elif self.vq_type == "bsq":
                # BSQ expects different input format, need to check the actual implementation
                # For now, assuming it works similarly to RVQ
                quantized_output = self.bsq(vq_input)
                if isinstance(quantized_output, tuple):
                    quantized_semantic_features = quantized_output[0].transpose(1, 2)
                    encoder_results['quantized_features'] = quantized_semantic_features
                    if len(quantized_output) > 1:
                        encoder_results['vq_codes'] = quantized_output[1]
                else:
                    quantized_semantic_features = quantized_output.transpose(1, 2)
                    encoder_results['quantized_features'] = quantized_semantic_features
                    

            if quantized_semantic_features is not None:
                deaggregated_features = self.deaggregate_semantic(
                    quantized_semantic_features,
                    padded_alignments
                )
                encoder_results['deaggregated_features'] = deaggregated_features
            
        return encoder_results

class WhisperAudioEncoder(BaseAudioEncoder):
    def __init__(
        self, 
        model_name_or_path: str,
        s3_encoder_ckpt: str = None, # checkpoint path of s3_encoder model 
        target_hidden_layer: int = 6, # specify which layer to extract. NOTE: zero means to extract the embed feature. Set to -1 to extract all hidden
        encoder_model: nn.Module = None, # allow passing a WhisperEncoder down for usage.
        attn_implementation: str = "eager", # possible choices: [eager, sdpa, flash_attention_2]
        dtype: str = "float32",
    ):
        super().__init__()
        if encoder_model == None:
            whole_model, _torch_dtype = load_whisper_whole_model(
                model_name_or_path,
                attn_implementation = attn_implementation,
                dtype = dtype,
            )
            self.encoder = whole_model.get_encoder()
        else:
            self.encoder = encoder_model

        self.processor = WhisperProcessor.from_pretrained(model_name_or_path)
        # print(self.encoder)
        return_last_hidden = False
        if target_hidden_layer != -1 and not return_last_hidden: # -1 means extract all hidden layers. 
            for i, layer in enumerate(self.encoder.layers):
                if i > target_hidden_layer:
                    print(f"Delete layer {i}")
                    self.encoder.layers[i] = None
                
        # check load s3_encoder_ckpt or no
        if s3_encoder_ckpt != None:
            s3_encoder_dict = get_s3_encoder_dict(self.state_dict(), s3_encoder_ckpt)
            # print(s3_encoder_dict)
            self.load_state_dict(s3_encoder_dict)
        
        self.extractor_hop_length = self.processor.feature_extractor.hop_length  # This is basically 160 for whisper extractor
        self.extractor_max_frames = self.processor.feature_extractor.nb_max_frames  # This is basically 30 * 16000 // 160  
        self.expected_seq_length = self.encoder.max_source_positions * self.encoder.conv1.stride[0] * self.encoder.conv2.stride[0]
        print(f"WhisperAudioEncoder | expected sequence lengths: {self.expected_seq_length}")
        self.target_hidden_layer = target_hidden_layer
        print(f"WhisperAudioEncoder | target layer: {self.target_hidden_layer}")
    
    def extract_feature(
        self,
        audio_fpaths: List[str],
        permute: bool = True,
        # pad_to_whisper_input_size: Optional[bool] = None, NOTE: deprecated. This behavior does not align with the original whisper designation
        **cfg,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        audio_feat_list, audio_feat_len_list = [], []
        waveforms = []
        for audio_fpath in audio_fpaths:
            waveform, sr = librosa.load(audio_fpath, sr=16_000, mono=True) # target sr is 16_000 for whisper
            assert sr==16_000, "Something went wrong"
            waveforms.append(waveform)
            audio_feat_len_list.append(len(waveform) // self.extractor_hop_length)

        inputs = self.processor(waveforms, sampling_rate=16000, return_tensors='pt', max_length=None) # will automatically pad to 
        audio_feat = inputs["input_features"]
        if permute:
            audio_feat = audio_feat.transpose(-1, -2) # (B, C, T) -> (B, T, C)
        audio_feat_len = torch.tensor(audio_feat_len_list, dtype=torch.int32) # This is for reference only. Whisper encoder accepts same input sizes (=3000) only. 
        return audio_feat, audio_feat_len

    def pad_to_whisper_input_size(self, audio_feat: List[torch.Tensor], padding_value=0.0):
        b, t = len(audio_feat), self.expected_seq_length
        c = audio_feat[0].shape[-1] # each tensor is with shape (T, C)
        padded_tensors = torch.full(
            (b, t, c),
            padding_value,
        )
        for i, tensor in enumerate(audio_feat):
            length = tensor.size(0)
            padded_tensors[i, :length] = tensor
        
        return padded_tensors

    # @torch.cuda.amp.autocast()
    # @torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)
    @torch.amp.autocast('cuda')
    def forward(
        self,
        audio_features: torch.Tensor,
        audio_features_lengths: torch.Tensor,
        attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ) -> Dict[str, Optional[torch.Tensor]]:
        r"""
        Args:
            input_features (`torch.LongTensor` of shape `(batch_size, feature_size, sequence_length)`):
                Float values of mel features extracted from the raw speech waveform. Raw speech waveform can be
                obtained by loading a `.flac` or `.wav` audio file into an array of type `List[float]` or a
                `numpy.ndarray`, *e.g.* via the soundfile library (`pip install soundfile`). To prepare the array into
                `input_features`, the [`AutoFeatureExtractor`] should be used for extracting the mel features, padding
                and conversion into a tensor of type `torch.FloatTensor`. See [`~WhisperFeatureExtractor.__call__`]
            attention_mask (`torch.Tensor`)`, *optional*):
                Whisper does not support masking of the `input_features`, this argument is preserved for compatibility,
                but it is not used. By default the silence in the input log mel spectrogram are ignored.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        # print(audio_features.shape)
        input_features = audio_features.transpose(1, 2) # (B, T, C) -> (B, C, T)

        if input_features.shape[-1] != self.expected_seq_length:
            if input_features.shape[-1] < self.expected_seq_length:
                # already padded but should be extended to fit whisper's input seq length
                p1d = (0, self.expected_seq_length - input_features.shape[-1])
                input_features = F.pad(input_features, p1d, 'constant', 0.0)
            else:
                raise ValueError(
                )
        # print(input_features.dtype)
        inputs_embeds = nn.functional.gelu(self.encoder.conv1(input_features))
        inputs_embeds = nn.functional.gelu(self.encoder.conv2(inputs_embeds))

        inputs_embeds = inputs_embeds.permute(0, 2, 1) # (B, T, C)
        embed_pos = self.encoder.embed_positions.weight

        hidden_states = inputs_embeds + embed_pos
        hidden_states = nn.functional.dropout(hidden_states, p=self.encoder.dropout, training=self.encoder.training)

        encoder_states = {} if output_hidden_states else None

        for idx, encoder_layer in enumerate(self.encoder.layers):
            if idx == self.target_hidden_layer:
                results = {
                    'encoded_feats': hidden_states,
                    'encoded_feats_lengths': audio_features_lengths // 2, # whisper encoder will down-sample by 2 
                }
                return results
            elif idx == output_hidden_states:
                encoder_states[f'{idx}'] = hidden_states
            # if self.target_hidden_layer < 0:
            #     encoder_states = encoder_states + (hidden_states,) # forbidden returning all hidden states
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            # else:
            layer_outputs = encoder_layer(
                hidden_states,
                None,
                layer_head_mask=None,
                output_attentions=output_attentions,
            )

            hidden_states = layer_outputs[0]
            if output_attentions:
                raise NotImplementedError
                # attentions = layer_outputs[1]

        if self.target_hidden_layer < 0:
            hidden_states = self.encoder.layer_norm(hidden_states)
            encoder_states['last_hidden'] = hidden_states
            results = {
                'encoded_feats': encoder_states,
                'encoded_feats_lengths': audio_features_lengths // 2, 
            } # return all encoder states
        else: 
            # return last hidden
            results = {
                'encoded_feats': hidden_states,
                'encoded_feats_lengths': audio_features_lengths // 2, 
            }
        return results

torch.manual_seed(42)
def test_sensevoice_enc(): # testing audio extraction
    # import os
    # RTSLM_WORK_DIR = os.getenv('RTSLM_WORK_DIR')
    print(RTSLM_WORK_DIR)
    sensevoice_encoder = SenseVoiceAudioEncoder(
        model_card="/data1/lijiaqi/codebase/TASTE-SpokenLM/STAGE1_TRAIN/storage/pretrained_models/SenseVoiceSmall",
        model_code_dir="/data1/lijiaqi/codebase/CSDs/zero-shot-tts-training/zero_shot_tts_training/realtime_communication/taste_v2/customized_sensevoice/model.py",
        extract_hidden=True,
        prepend_inputs_before_encoding=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sensevoice_encoder.to(device)

    audio_fpaths = [
        # "/root/rtslm/CosyVoice/cross_lingual_prompt.wav",
        # "/root/rtslm/CosyVoice/cross_lingual.wav",
        # f"{RTSLM_WORK_DIR}/CosyVoice/instruct4.wav",
        # f"/data1/lijiaqi/codebase/TASTE-SpokenLM/STAGE1_TRAIN/storage/pretrained_models/SenseVoiceSmall/example/en.mp3",
        '/data1/lijiaqi/codebase/CSDs/out1.wav',
    ]
    print(sensevoice_encoder)

    audio_features, audio_features_lengths = sensevoice_encoder.extract_feature(
        audio_fpaths,
        cache={},
        language="auto",  # "zn", "en", "yue", "ja", "ko", "nospeech"
        use_itn=True,
        batch_size_s=60,
    )
    def extract_fbank(self, data, data_len=None, data_type: str = "sound", **kwargs):
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
            if len(data.shape) < 2:
                data = data[None, :]  # data: [batch, N]
            data_len = [data.shape[1]] if data_len is None else data_len
        elif isinstance(data, torch.Tensor):
            if len(data.shape) < 2:
                data = data[None, :]  # data: [batch, N]
            data_len = [data.shape[1]] if data_len is None else data_len
        elif isinstance(data, (list, tuple)):
            data_list, data_len = [], []
            for data_i in data:
                if isinstance(data_i, np.ndarray):
                    data_i = torch.from_numpy(data_i)
                data_list.append(data_i)
                data_len.append(data_i.shape[0])

        data, data_len = self.frontend(data, data_len, **kwargs)


        if isinstance(data_len, (list, tuple)):
            data_len = torch.tensor([data_len])
        return data.to(torch.float32), data_len.to(torch.int32)
    # override feature
    import torchaudio
    audio_data, _ = torchaudio.load('/data1/lijiaqi/codebase/CSDs/out.wav')
    # another file: /data1/lijiaqi/codebase/CSDs/out1.wav

    # override frontend
    import funasr
    cmvn_file = '/data1/lijiaqi/codebase/CSDs/zero-shot-tts-training/zero_shot_tts_training/realtime_communication/taste_v2/customized_sensevoice/am.mvn'
    sensevoice_encoder.frontend = funasr.frontends.wav_frontend.WavFrontend(
                cmvn_file=cmvn_file,
                n_mels=80,
                frame_length=25,
                frame_shift=10,
                lfr_m=7,
                lfr_n=6,
            )


    audio_features, audio_features_lengths = extract_fbank(sensevoice_encoder, audio_data.to(device), None)
    print(f'audio_feature.shape: {audio_features.shape}')
    audio_features = audio_features.to(device)
    audio_features_lengths = audio_features_lengths.to(device)

    results = sensevoice_encoder.forward(audio_features, audio_features_lengths, return_text=True, use_itn=True)
    for feat in results['encoded_feats']:
        print(feat.shape)
    for feat_length in results['encoded_feats_lengths']:
        print(feat_length)
    for ctc in results['ctc_logits']:
        print(ctc.shape)
    print(results)
    
    # Test the new aggregate_semantic method
    for i, alignment in enumerate(results['alignments']):
        print("Alignment matrix shape:", alignment.shape)
        print("Each frame assigned to exactly one token:", (alignment.sum(0) == 1).all().item())
        
        # Test semantic aggregation using the half-layer features if available
        if 'half_hidden_feats' in results and results['half_hidden_feats'] is not None:
            features_to_agg = results['half_hidden_feats'][i:i+1] # Keep batch dimension
            print("\nUsing half-layer hidden features for aggregation.")
        else:
            features_to_agg = results['encoded_feats'][i:i+1] # Fallback
            print("\nHalf-layer features not found, using final encoder output for aggregation.")

        aggregated_semantic = sensevoice_encoder.aggregate_semantic(
            features_to_agg, 
            alignment.unsqueeze(0)
        )
        print(f"Original hidden features shape: {features_to_agg.shape}")
        print(f"Aggregated semantic features shape: {aggregated_semantic.shape}")
        # print(f"Number of text tokens: {len(results['decoded_tokens'][i])}")
        
        # Plotting for validation
        # import matplotlib.pyplot as plt
        # import numpy as np
        
        # Get data for the current sample
        audio_feat = audio_features[i, :alignment.shape[1]].cpu().detach().numpy()
        alignment_matrix = alignment.cpu().numpy()
        decoded_tokens = results['decoded_tokens'][i]
        
        # fig, ax = plt.subplots(1, 1, figsize=(20, 8))
        
        # # Plot F-Bank features as background
        # im_fbank = ax.imshow(audio_feat.T, aspect='auto', origin='lower', cmap='viridis', 
        #                     extent=[0, alignment.shape[1], 0, audio_feat.shape[1]], alpha=0.8)
        
        # # Overlay alignment boundaries
        # current_frame = 0
        # colors = plt.cm.Set3(np.linspace(0, 1, len(decoded_tokens)))
        
        # for token_idx, token in enumerate(decoded_tokens):
        #     # Find the frames assigned to this token
        #     token_frames = np.where(alignment_matrix[token_idx] == 1)[0]
        #     if len(token_frames) > 0:
        #         start_frame = token_frames[0]
        #         end_frame = token_frames[-1] + 1
                
        #         # Draw token boundary as vertical lines
        #         ax.axvline(x=start_frame, color='red', linestyle='-', linewidth=2, alpha=0.7)
        #         if token_idx == len(decoded_tokens) - 1:  # Last token
        #             ax.axvline(x=end_frame, color='red', linestyle='-', linewidth=2, alpha=0.7)
                
        #         # Add token label at the center of its span
        #         center_frame = (start_frame + end_frame) / 2
        #         ax.text(center_frame, audio_feat.shape[1] * 0.95, token, 
        #                ha='center', va='top', fontsize=12, fontweight='bold',
        #                bbox=dict(boxstyle="round,pad=0.3", facecolor=colors[token_idx], alpha=0.7))
                
        #         # Optional: Add colored regions for each token
        #         ax.axvspan(start_frame, end_frame, alpha=0.2, color=colors[token_idx])
        
        # ax.set_title("F-Bank Features with CTC Forced Alignment Overlay")
        # ax.set_xlabel("Frames")
        # ax.set_ylabel("F-Bank bin")
        
        # # Add colorbar for F-Bank features
        # cbar = fig.colorbar(im_fbank, ax=ax, shrink=0.8)
        # cbar.set_label('F-Bank magnitude')
        
        # plt.tight_layout()
        # plt.savefig("alignment_visualization.png", dpi=150, bbox_inches='tight')
        # print("Saved alignment plot to alignment_visualization.png")
        # plt.close(fig)
        
        # Generate JSON alignment output
        import json
        
        output_dir = "aligned_audio_segments"
        os.makedirs(output_dir, exist_ok=True)
        # hop_samples = (sensevoice_encoder.frontend.opts.frame_shift * sr / 1000) * sensevoice_encoder.frontend.opts.lfr_n
        hop_samples = (10 * 16000 / 1000) * 6
        print(f"Hop samples: {hop_samples}")
        
        # Calculate frame rate (frames per second)
        # SenseVoice uses 25ms frame shift by default, so frame_rate = 1000/25 = 40 fps
        frame_rate = 16.6667  # frames per second for SenseVoice
        
        alignment_json = {"words": []}
        
        # Skip first 4 tokens (control tokens) and process actual speech tokens
        alignment_matrix = alignment_matrix[4:]
        decoded_tokens = decoded_tokens[4:]
        speech_tokens_start = 0
        if len(decoded_tokens) > speech_tokens_start:
            for token_idx in range(speech_tokens_start, len(decoded_tokens)):
                token = decoded_tokens[token_idx]
                # Find the frames assigned to this token
                token_frames = np.where(alignment_matrix[token_idx] == 1)[0]
                if len(token_frames) > 0:
                    start_frame = max(0,token_frames[0]-1)
                    end_frame = min(len(audio_data[0]),token_frames[-1]-1)
                    
                    # Convert frames to seconds
                    start_time = start_frame / frame_rate
                    end_time = end_frame / frame_rate
                    
                    start_sample = int(start_time * 16000)
                    end_sample = int(end_time * 16000)
                    segment_waveform = audio_data[:, start_sample:end_sample]
                    
                    # Sanitize filename
                    safe_token = "".join([c for c in token if c.isalpha() or c.isdigit() or c.isspace()]).rstrip().replace(" ", "_")
                    if not safe_token:
                        safe_token = f"token_{token_idx}"
                    
                    segment_filename = os.path.join(output_dir, f"{token_idx}_{safe_token}.wav")
                    torchaudio.save(segment_filename, segment_waveform, 16000)
                    print(f"Saved segment to {segment_filename}")

                    # Calculate confidence as the maximum CTC probability for this token across its frames
                    # ctc_probs = torch.softmax(results['ctc_logits'][i], dim=-1)
                    # token_id = text_tokens[token_idx].item()
                    alignment_json["words"].append({
                        "text": token,
                        "start": round(start_time, 3),
                        "end": round(end_time, 3),
                        # "confidence": round(confidence, 3)
                    })
        
        # Save JSON file
        with open("alignment_output.json", "w", encoding="utf-8") as f:
            json.dump(alignment_json, f, indent=2, ensure_ascii=False)
        print("Saved alignment JSON to alignment_output.json")

def test_sensevoice_enc_similarity(): # testing audio extraction
    # import os
    # RTSLM_WORK_DIR = os.getenv('RTSLM_WORK_DIR')
    print(RTSLM_WORK_DIR)
    sensevoice_encoder = SenseVoiceAudioEncoder(
        model_card="/data1/lijiaqi/codebase/TASTE-SpokenLM/STAGE1_TRAIN/storage/pretrained_models/SenseVoiceSmall",
        model_code_dir="/data1/lijiaqi/codebase/CSDs/zero-shot-tts-training/zero_shot_tts_training/realtime_communication/taste_v2/customized_sensevoice/model.py",
        extract_hidden=True,
        prepend_inputs_before_encoding=True,
        alignment_mode='similarity',
        similarity_threshold=0.9,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sensevoice_encoder.to(device)

    audio_fpaths = [
        # "/root/rtslm/CosyVoice/cross_lingual_prompt.wav",
        # "/root/rtslm/CosyVoice/cross_lingual.wav",
        # f"{RTSLM_WORK_DIR}/CosyVoice/instruct4.wav",
        # f"/data1/lijiaqi/codebase/TASTE-SpokenLM/STAGE1_TRAIN/storage/pretrained_models/SenseVoiceSmall/example/en.mp3",
        '/data1/lijiaqi/codebase/CSDs/out1.wav',
    ]
    print(sensevoice_encoder)

    audio_features, audio_features_lengths = sensevoice_encoder.extract_feature(
        audio_fpaths,
        cache={},
        language="auto",  # "zn", "en", "yue", "ja", "ko", "nospeech"
        use_itn=True,
        batch_size_s=60,
    )
    def extract_fbank(self, data, data_len=None, data_type: str = "sound", **kwargs):
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
            if len(data.shape) < 2:
                data = data[None, :]  # data: [batch, N]
            data_len = [data.shape[1]] if data_len is None else data_len
        elif isinstance(data, torch.Tensor):
            if len(data.shape) < 2:
                data = data[None, :]  # data: [batch, N]
            data_len = [data.shape[1]] if data_len is None else data_len
        elif isinstance(data, (list, tuple)):
            data_list, data_len = [], []
            for data_i in data:
                if isinstance(data_i, np.ndarray):
                    data_i = torch.from_numpy(data_i)
                data_list.append(data_i)
                data_len.append(data_i.shape[0])

        data, data_len = self.frontend(data, data_len, **kwargs)


        if isinstance(data_len, (list, tuple)):
            data_len = torch.tensor([data_len])
        return data.to(torch.float32), data_len.to(torch.int32)
    # override feature
    import torchaudio
    audio_data, _ = torchaudio.load('/data1/lijiaqi/codebase/CSDs/out.wav')
    # another file: /data1/lijiaqi/codebase/CSDs/out1.wav

    # override frontend
    import funasr
    cmvn_file = '/data1/lijiaqi/codebase/CSDs/zero-shot-tts-training/zero_shot_tts_training/realtime_communication/taste_v2/customized_sensevoice/am.mvn'
    sensevoice_encoder.frontend = funasr.frontends.wav_frontend.WavFrontend(
                cmvn_file=cmvn_file,
                n_mels=80,
                frame_length=25,
                frame_shift=10,
                lfr_m=7,
                lfr_n=6,
            )


    audio_features, audio_features_lengths = extract_fbank(sensevoice_encoder, audio_data.to(device), None)
    audio_features = audio_features.to(device)
    audio_features_lengths = audio_features_lengths.to(device)

    results = sensevoice_encoder.forward(audio_features, audio_features_lengths, return_text=True, use_itn=False)
    for feat in results['encoded_feats']:
        print(feat.shape)
    for feat_length in results['encoded_feats_lengths']:
        print(feat_length)
    for ctc in results['ctc_logits']:
        print(ctc.shape)
    print(results)
    
    # Test the new aggregate_semantic method
    for i, alignment in enumerate(results['alignments']):
        print("Alignment matrix shape:", alignment.shape)
        print("Each frame assigned to exactly one token:", (alignment.sum(0) == 1).all().item())
        
        # Test semantic aggregation using the half-layer features if available
        if 'half_hidden_feats' in results and results['half_hidden_feats'] is not None:
            features_to_agg = results['half_hidden_feats'][i:i+1] # Keep batch dimension
            print("\nUsing half-layer hidden features for aggregation.")
        else:
            features_to_agg = results['encoded_feats'][i:i+1] # Fallback
            print("\nHalf-layer features not found, using final encoder output for aggregation.")

        aggregated_semantic = sensevoice_encoder.aggregate_semantic(
            features_to_agg, 
            alignment.unsqueeze(0)
        )
        print(f"Original hidden features shape: {features_to_agg.shape}")
        print(f"Aggregated semantic features shape: {aggregated_semantic.shape}")
        # print(f"Number of text tokens: {len(results['decoded_tokens'][i])}")
        
        # Get data for the current sample
        audio_feat = audio_features[i, :alignment.shape[1]].cpu().detach().numpy()
        alignment_matrix = alignment.cpu().numpy()
        decoded_tokens = results['decoded_tokens'][i]
        
        # Generate JSON alignment output
        import json
        
        output_dir = "aligned_audio_segments_similarity"
        os.makedirs(output_dir, exist_ok=True)
        # hop_samples = (sensevoice_encoder.frontend.opts.frame_shift * sr / 1000) * sensevoice_encoder.frontend.opts.lfr_n
        hop_samples = (10 * 16000 / 1000) * 6
        print(f"Hop samples: {hop_samples}")
        
        # Calculate frame rate (frames per second)
        # SenseVoice uses 25ms frame shift by default, so frame_rate = 1000/25 = 40 fps
        frame_rate = 16.6667  # frames per second for SenseVoice
        
        alignment_json = {"words": []}
        
        # Skip first 4 tokens (control tokens) and process actual speech tokens
        alignment_matrix = alignment_matrix[4:]
        decoded_tokens = decoded_tokens[4:]
        speech_tokens_start = 0
        if len(decoded_tokens) > speech_tokens_start:
            for token_idx in range(speech_tokens_start, len(decoded_tokens)):
                token = decoded_tokens[token_idx]
                # Find the frames assigned to this token
                token_frames = np.where(alignment_matrix[token_idx] == 1)[0]
                if len(token_frames) > 0:
                    start_frame = max(0,token_frames[0])
                    end_frame = min(len(audio_data[0]),token_frames[-1])
                    
                    # Convert frames to seconds
                    start_time = start_frame / frame_rate
                    end_time = end_frame / frame_rate
                    
                    start_sample = int(start_frame * hop_samples)
                    end_sample = int(end_frame * hop_samples)
                    segment_waveform = audio_data[:, start_sample:end_sample]
                    
                    # Sanitize filename
                    safe_token = "".join([c for c in token if c.isalpha() or c.isdigit() or c.isspace()]).rstrip().replace(" ", "_")
                    if not safe_token:
                        safe_token = f"token_{token_idx}"
                    
                    segment_filename = os.path.join(output_dir, f"{token_idx}_{safe_token}.wav")
                    torchaudio.save(segment_filename, segment_waveform, 16000)
                    print(f"Saved segment to {segment_filename}")

                    alignment_json["words"].append({
                        "text": token,
                        "start": round(start_time, 3),
                        "end": round(end_time, 3),
                    })
        
        # Save JSON file
        with open("alignment_output_similarity.json", "w", encoding="utf-8") as f:
            json.dump(alignment_json, f, indent=2, ensure_ascii=False)
        print("Saved alignment JSON to alignment_output_similarity.json")

def test_whisper_enc():
    model_fpath = "/proj/mtklmadm/dev/mtk53678/rtslm_storage/pretrained_models/whisper-large-v3"
    s3_encoder_ckpt = '/proj/mtklmadm/dev/mtk53684/new_model.pth'
    whisper_encoder = WhisperAudioEncoder(model_fpath, s3_encoder_ckpt = None)
    
    audio_fpaths = [
        # "/root/rtslm/CosyVoice/cross_lingual_prompt.wav",
        # "/root/rtslm/CosyVoice/cross_lingual.wav",
        # "/root/rtslm/CosyVoice/instruct.wav",
        f"{RTSLM_WORK_DIR}/CosyVoice/en.mp3",
        # f"{RTSLM_WORK_DIR}/CosyVoice/instruct4.wav"
    ]
    audio_feat, audio_feat_len = whisper_encoder.extract_feature(
        audio_fpaths,
        pad_to_whisper_input_size=True
    )
    # print(rearrange(audio_feat, 'b t c -> b c t').is_contiguous())
    print(audio_feat_len)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    whisper_encoder.to(device)
    whisper_encoder.eval()
    with torch.no_grad():
        hidden_state = whisper_encoder(
            audio_feat.to(device),
            audio_feat_len.to(device)
        )
        print(hidden_state)


if __name__ == "__main__":
    # test_whisper_enc()
    test_sensevoice_enc()
    # test_sensevoice_enc_similarity()