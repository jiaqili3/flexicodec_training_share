from functools import partial
from pathlib import Path
import sys
sys.path.append(f'{str(Path(__file__).parent.parent.parent.parent)}')
from zero_shot_tts_training.realtime_communication.codec_model.cnn import ConvNeXtBlock
import torch.nn as nn
import math
from typing import List
from typing import Union
import torchaudio
import numpy as np
import torch
from audiotools import AudioSignal
from audiotools.ml import BaseModel
from torch import nn
from transformers import Wav2Vec2BertModel
# from .base import CodecMixin
from dac.nn.layers import Snake1d
from dac.nn.layers import WNConv1d
from dac.nn.layers import WNConvTranspose1d
from zero_shot_tts_training.realtime_communication.codec_model.dac_quantize import ResidualVectorQuantize
from .bsq_quantizer import BSQWrapper
from .fsq_wrapper import FSQWrapper
from easydict import EasyDict as edict
import torch.nn.functional as F
import random
from einops import rearrange
from torch.nn.utils.rnn import pad_sequence
# Import FunASR for direct model usage
from funasr import AutoModel
# Import transformer components
import zero_shot_tts_training.realtime_communication.taste_v2.model_blocks.mimi.transformer as Stransformer

params = lambda model: sum(p.numel() for p in model.parameters())
class UpsampleConvNeXtBlock(nn.Module):
    def __init__(self, in_dim, out_dim, upsample_factor=2, intermediate_dim=2048, is_causal=False, n_layers=4):
        super().__init__()
        
        # Upsampling layer (transposed convolution-based)
        self.upsample = WNConvTranspose1d(in_dim, out_dim, kernel_size=upsample_factor, stride=upsample_factor)
        
        # ConvNeXt block sequence
        self.convnext_blocks = nn.Sequential(
            *[ConvNeXtBlock(dim=out_dim, intermediate_dim=intermediate_dim, is_causal=is_causal) for _ in range(n_layers)]
        )

    def forward(self, x):
        # Upsample then apply ConvNeXt blocks
        x = self.upsample(x)
        return self.convnext_blocks(x)

class DownsampleConvNeXtBlock(nn.Module):
    def __init__(self, in_dim, out_dim, downsample_factor=2, intermediate_dim=2048, is_causal=False, n_layers=4):
        super().__init__()
        # Downsampling layer (stride-based)
        self.downsample = WNConv1d(in_dim, out_dim, kernel_size=downsample_factor, stride=downsample_factor)
        
        # ConvNeXt block sequence
        self.convnext_blocks = nn.Sequential(
            *[ConvNeXtBlock(dim=out_dim, intermediate_dim=intermediate_dim, is_causal=is_causal) for _ in range(n_layers)]
        )

    def forward(self, x):
        # Downsample then apply ConvNeXt blocks
        x = self.downsample(x)
        return self.convnext_blocks(x)
def get_vocoder_decode_func_and_mel_spec():
    from dualcodec.model_tts.voicebox.vocoder_model import get_vocos_model_spectrogram, mel_to_wav_vocos
    vocos_model, mel_model = get_vocos_model_spectrogram()
    infer_vocos = partial(mel_to_wav_vocos, vocos_model)
    return infer_vocos, mel_model
    
@torch.no_grad()
def _extract_semantic_code(
    semantic_model,
    input_features,
    attention_mask,
    mean,
    std,
    *,
    skip_normalize: bool = False,
    sensevoice_prepend_inputs: bool = True,
    sim_layer_idx=None,
    semantic_layer_idx=None,
):
    """Return `(semantic_repr, sim_repr)` in (B, T, C) format.

    * For Wav2Vec2-BERT both outputs are the same.
    * For SenseVoice we select hidden layers according to the supplied indices.
    """
    
    # Check if using FunASR model (SenseVoice)
    if isinstance(semantic_model, Wav2Vec2BertModel):
        # Original Wav2Vec2BertModel logic with mixed precision
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            vq_emb = semantic_model(
                input_features=input_features,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            output_idx = 17
            feat = vq_emb.hidden_states[output_idx]  # (B, T, C)
            if not skip_normalize:
                feat = (feat - mean) / std
        return feat, feat # layer selection not implemented
    
    elif hasattr(semantic_model, 'encoder'):
        # For FunASR model, we need to pass audio_features_lengths
        # Create dummy lengths based on attention_mask or input_features shape
        audio_features_lengths = torch.tensor([input_features.shape[1]] * input_features.shape[0], 
                                            device=input_features.device, dtype=torch.long)
        
        # Check if we need to prepend inputs (similar to SenseVoiceAudioEncoder.forward_encoder)
        # For SenseVoice, we typically want to prepend inputs
        if sensevoice_prepend_inputs:
            input_features, audio_features_lengths = semantic_model.prepend_inputs(
                input_features, audio_features_lengths
            ) # [b,t+4,c]
        
        # Call FunASR model encoder directly (no mixed precision for FunASR)
        encoder_out, encoder_out_lengths, hidden_out, hiddens = semantic_model.encoder(
            input_features, audio_features_lengths, extract_hidden=True
        )
        
        if semantic_layer_idx is None:
            semantic_feat = hidden_out[:, 4:]
            return semantic_feat, semantic_feat
        else:
            # Support range/list for sim_layer_idx
            if isinstance(sim_layer_idx, (list, tuple, range)):
                sim_feat = torch.stack([hiddens[idx] for idx in sim_layer_idx], dim=0).mean(dim=0)
            elif sim_layer_idx is None:
                sim_feat = hidden_out
            else:
                sim_feat = hiddens[sim_layer_idx]
            if isinstance(semantic_layer_idx, (list, tuple, range)):
                if semantic_layer_idx[1] == -1:
                    semantic_layer_idx[1] = len(hiddens)
                semantic_feat = torch.stack([hiddens[idx] for idx in range(semantic_layer_idx[0], semantic_layer_idx[1])], dim=0).mean(dim=0)
            else:
                semantic_feat = hiddens[semantic_layer_idx]
            sim_feat = sim_feat[:, 4:]
            semantic_feat = semantic_feat[:, 4:]
            return semantic_feat, sim_feat

    else:
        raise ValueError(f"Unsupported semantic model type: {type(semantic_model)}")

class QueryTokenAggregator(nn.Module):
    """
    Aggregates features based on similarity grouping using a query-based transformer.
    1. Initializes a query for each group (either via mean-pooling or a shared learnable token).
    2. Interleaves the query token after each corresponding group of input features.
    3. Processes the interleaved sequence through a transformer.
    4. Retrieves the transformer's output for the query token positions, which
       serve as the aggregated representation for each group.
    """

    def __init__(self, dim: int, in_out_dim: int, num_heads: int, num_layers: int, dim_feedforward: int, causal: bool = False, context_frames: int = 125, use_mean_pooling_init: bool = True, add_query_embedding: bool = False):
        super().__init__()
        self.use_mean_pooling_init = use_mean_pooling_init
        self.add_query_embedding = add_query_embedding
        
        if not self.use_mean_pooling_init:
            self.query_token = nn.Parameter(torch.randn(1, dim, 1))
        
        # Add learnable query embedding if enabled
        if self.add_query_embedding:
            self.query_embedding = nn.Parameter(torch.randn(1, in_out_dim, 1))

        transformer_kwargs = {
            'd_model': dim,
            'num_heads': num_heads,
            'num_layers': num_layers,
            'causal': causal,
            'layer_scale': 0.01,
            'context': context_frames,
            'conv_layout': True,
            'max_period': 10000,
            'gating': 'none',
            'norm': 'layer_norm',
            'positional_embedding': 'rope',
            'dim_feedforward': dim_feedforward,
            'input_dimension': in_out_dim,
            'output_dimensions': [in_out_dim],
        }
        self.transformer = Stransformer.ProjectedTransformer(**transformer_kwargs)

    def forward(self, features: torch.Tensor, alignment_matrix: torch.Tensor, num_segments_per_item: torch.Tensor):
        """
        Args:
            features (torch.Tensor): Input features of shape (B, D, T).
            alignment_matrix (torch.Tensor): Binary matrix of shape (B, G, T) indicating group membership.
            num_segments_per_item (torch.Tensor): Tensor of shape (B,) with the number of groups for each batch item.
        Returns:
            torch.Tensor: Aggregated features of shape (B, D, G_max).
        """
        B, D, T = features.shape
        _B, G, _T = alignment_matrix.shape
        device = features.device

        # If features are longer than the alignment matrix, trim them to match.
        if T > _T:
            features = features[..., :_T]
            T = _T

        # The time dimension of features and alignment_matrix must now match.
        assert T == _T, f"Feature time dimension {T} must match alignment matrix time dimension {_T}"

        # 1. Create masks for valid groups and frames for each item in the batch
        group_mask = torch.arange(G, device=device).unsqueeze(0) < num_segments_per_item.unsqueeze(1)  # (B, G)

        group_last_frame_indices = (alignment_matrix * torch.arange(T, device=device)).max(dim=2).values  # (B, G)

        # Infer true frame lengths from the alignment matrix
        valid_last_indices = group_last_frame_indices.masked_fill(~group_mask, -1)
        frame_lengths = valid_last_indices.max(dim=1).values + 1  # (B,)
        frame_mask = torch.arange(T, device=device).unsqueeze(0) < frame_lengths.unsqueeze(1)  # (B, T)

        # 2. Calculate destination indices for interleaving
        # For frames: its original index + number of groups that end before it.
        last_indices_for_count = group_last_frame_indices.clone()
        last_indices_for_count[~group_mask] = T + 1  # Use a large value for padded groups
        num_queries_before = (last_indices_for_count.unsqueeze(2) < torch.arange(T, device=device)).sum(dim=1)
        frame_dest = torch.arange(T, device=device) + num_queries_before  # (B, T)

        # For queries: last frame of its group + its own index in the group sequence + 1.
        query_dest = group_last_frame_indices + torch.arange(G, device=device) + 1  # (B, G)

        # 3. Create the source sequence by concatenating features and query tokens
        if self.use_mean_pooling_init:
            # Dynamically create queries by mean-pooling features within each group
            alignment_float = alignment_matrix.to(features.dtype)
            summed_features = torch.einsum('bgt,bdt->bgd', alignment_float, features)
            group_frame_counts = alignment_float.sum(dim=2).clamp(min=1)
            queries = (summed_features / group_frame_counts.unsqueeze(-1)).transpose(1, 2)
            
            # Add learnable query embedding if enabled
            if self.add_query_embedding:
                queries = queries + self.query_embedding.expand(B, -1, G)
        else:
            # Use the single learnable query token for all groups
            queries = self.query_token.expand(B, -1, G)
        
        source_seq = torch.cat([features, queries], dim=2)  # (B, D, T+G)

        # 4. Create the interleaved sequence using a permutation derived from destination indices
        dest_indices = torch.cat([frame_dest, query_dest], dim=1)  # (B, T+G)
        source_mask = torch.cat([frame_mask, group_mask], dim=1)  # (B, T+G)

        # Invalidate destination indices for padded elements by pushing them to the end
        max_len = T + G
        dest_indices_masked = dest_indices.masked_fill(~source_mask, max_len)

        # Get permutation for interleaving
        perm = dest_indices_masked.argsort(dim=1)  # (B, T+G)
        perm_expanded = perm.unsqueeze(1).expand(-1, D, -1)
        interleaved_features = torch.gather(source_seq, 2, perm_expanded)  # (B, D, T+G)

        # 5. Pass through transformer. Note: Assumes transformer handles zero-padding.
        transformer_output = self.transformer(interleaved_features)  # (B, D, T+G)
        
        # 6. Retrieve the transformer outputs corresponding to the query positions
        # We need the inverse permutation to find where the original queries landed.
        inverse_perm = perm.argsort(dim=1)
        query_pos_in_interleaved = inverse_perm[:, T:]  # (B, G)
        
        # Gather from transformer output using the final positions of queries
        query_pos_expanded = query_pos_in_interleaved.unsqueeze(1).expand(-1, D, -1)
        aggregated_features = torch.gather(transformer_output, 2, query_pos_expanded)  # (B, D, G)
        
        # Mask the final output for any padded groups
        aggregated_features = aggregated_features.masked_fill(~group_mask.unsqueeze(1), 0.0)

        return aggregated_features

class DualCodec(nn.Module):
    def __init__(
        self,
        encoder_dim: int = 64,
        encoder_rates: List[int] = [2, 4, 8, 8],
        latent_dim: int = None,
        decoder_dim: int = 1536,
        decoder_rates: List[int] = [8, 8, 4, 2],
        n_codebooks: int = 9,
        codebook_size: int = 1024,
        semantic_codebook_size: int = 16384,
        codebook_dim: Union[int, list] = 8,
        semantic_codebook_dim=8,
        use_bsq_for_semantic_vq: bool = False,
        bsq_config: dict = None,
        use_fsq_for_semantic_vq: bool = False,
        fsq_config: dict = None,
        quantizer_dropout: bool = True,
        sample_rate: int = 24000,
        distill_projection_out_dim=1024,
        convnext_dim=768,
        convnext_layers=4,
        decode_semantic_for_codec=True,
        is_causal=True,
        semantic_downsample_factor=2,
        use_concat_downsampling=False,
        use_conv_downsampling=False,
        override_dac_encoder=None, # torch.nn.Module
        override_vocos_decoder=None,
        semantic_encoder=None,
        semantic_decoder=None,
        ssl_dim=1024,
        semantic_model_path="/mnt//models/projects/lijiaqi_csd/w2v-bert-2.0",  # "/data1/lijiaqi/codebase/TASTE-SpokenLM/STAGE1_TRAIN/storage/pretrained_models/SenseVoiceSmall"
        lambda_distill_loss=15.0,
        # New parameters for similarity-based alignment
        use_similarity_alignment: bool = False, # if false, is original dualcodec
        similarity_threshold=None,
        # Dynamic similarity threshold parameters
        use_dynamic_similarity_threshold: bool = False,
        similarity_threshold_lower: float = 0.7,
        similarity_threshold_upper: float = 1.0,
        # Flex framerate parameters
        flex_framerate: bool = False,
        flex_framerate_options: list = [0.86, 0.90, 1.0],
        skip_normalize=False,
        half_semantic_model=True,
        # Maximum number of tokens (frames) allowed in one similarity group.
        # Set to ``None`` to disable this constraint. When ``use_similarity_alignment``
        # is ``True`` and this value is an ``int`` > 0, any group with more than
        # ``max_tokens_per_group`` frames will be split into smaller groups so that
        # each contains at most this many frames.
        max_tokens_per_group: int | None = 6,
        # New parameter to choose semantic model type
        semantic_model_type: str = "w2vbert",  # "w2vbert" or "sensevoice"
        # SenseVoice specific parameters (only essential ones for FunASR model)
        sensevoice_prepend_inputs: bool = True,  # Whether to prepend inputs before encoding
        # DAC-specific DC-AE parameter
        residual_autoencode: bool = False,
        # Bottleneck transformer parameters
        use_bottleneck_transformer: bool = False, # if true, deagg = repeat + repa transformer
        bottleneck_transformer_config: dict = None,
        transformer_num_layers: int = 6,
        transformer_dim: int = 512,
        transformer_dim_feedforward: int = 2048,
        transformer_num_heads: int = 8,
        transformer_causal: bool = False,
        transformer_context_frames: int = 24,
        # Second decoder transformer
        use_second_decoder_transformer: bool = False,
        transformer_2_num_layers: int = None,
        # Aggregator transformer parameters
        use_query_token_aggregator: bool = False,
        agg_transformer_num_layers: int = 6,
        agg_transformer_dim: int = 512,
        agg_transformer_num_heads: int = 8,
        agg_transformer_dim_feedforward: int = 2048,
        agg_transformer_causal: bool = False,
        agg_use_mean_pooling_init: bool = True,
        agg_add_query_embedding: bool = False,
        agg_transformer_context_frames: int = None,
        # Fixed-rate aggregation parameters
        use_fixed_rate_aggregator: bool = False,
        aggregator_downsample_ratio: int = 4,
        aggregator_downsample_ratio_options: list = None,
        # New: control which hidden states to use for sim/semantic
        sensevoice_sim_layer_idx=None,
        sensevoice_semantic_layer_idx=None,
        # Flow matching decoder parameters
        use_flow_matching_decoder: bool = False,
        flow_matching_mel_dim: int = 128,
        flow_matching_hidden_size: int = 1024,
        flow_matching_num_layers: int = 12,
        flow_matching_num_heads: int = 16,
        flow_matching_cfg_scale: float = 0.2,
        flow_matching_use_cond_code: bool = False,
        flow_matching_cond_codebook_size: int = 1,
        flow_matching_cond_dim: int = 512,
        flow_matching_cond_scale_factor: int = 6,
        flow_matching_time_scheduler: str = "cos",
        # New streaming transformer parameters for VoiceBox
        flow_matching_context: int = 150,
        flow_matching_causal: bool = False,
        flow_matching_has_prompt: bool = False, # if true, will use vanilla voicebox model with speech prompt
        # REPA loss parameters for flow matching decoder
        use_repa_loss: bool = True,
        repa_loss_weight: float = 1.0,
        repa_loss_type: str = "cosine",  # "cosine", "l2", or "l1"
        repa_layer_idx: int = 9,  # Layer index for REPA loss (0-indexed, default is 10th layer)
        repa_projection_dim: int = 512,  # Output dimension for REPA projection MLP
        insert_query_before_downsample: bool = False,
        no_acoustic_aggregator: bool = False, # if true,  semantic aggregator will be acoustic aggregator
        sometimes_skip_repa: bool = False,
        distill_with_avg: bool = False,
    ):
        super().__init__()
        self.insert_query_before_downsample = insert_query_before_downsample
        self.semantic_downsample_factor = semantic_downsample_factor
        self.concat_downsample_factor = 1
        self.use_concat_downsampling = use_concat_downsampling
        self.use_conv_downsampling = use_conv_downsampling
        self.lambda_distill_loss = lambda_distill_loss
        self.semantic_encoder = semantic_encoder
        self.semantic_decoder = semantic_decoder
        self.use_bsq_for_semantic_vq = use_bsq_for_semantic_vq
        self.use_fsq_for_semantic_vq = use_fsq_for_semantic_vq
        self.use_repa_loss = use_repa_loss
        self.repa_loss_weight = repa_loss_weight
        self.repa_loss_type = repa_loss_type
        self.repa_layer_idx = repa_layer_idx
        self.repa_projection_dim = repa_projection_dim
        self.sometimes_skip_repa = sometimes_skip_repa
        if agg_transformer_context_frames is None:
            agg_transformer_context_frames = transformer_context_frames
        
        # Similarity alignment parameters
        self.use_similarity_alignment = use_similarity_alignment
        self.similarity_threshold = similarity_threshold
        self.use_dynamic_similarity_threshold = use_dynamic_similarity_threshold
        self.similarity_threshold_lower = similarity_threshold_lower
        self.similarity_threshold_upper = similarity_threshold_upper
        
        # Flex framerate parameters
        self.flex_framerate = flex_framerate
        self.flex_framerate_options = flex_framerate_options
        
        if self.use_similarity_alignment:
            if not self.use_dynamic_similarity_threshold:
                assert self.similarity_threshold is not None, "similarity_threshold must be set when use_similarity_alignment=True and use_dynamic_similarity_threshold=False"
            else:
                assert self.similarity_threshold_lower < self.similarity_threshold_upper, "similarity_threshold_lower must be less than similarity_threshold_upper"
        self.skip_normalize = skip_normalize
        self.max_tokens_per_group = max_tokens_per_group if (max_tokens_per_group is None or max_tokens_per_group > 0) else None
        self.use_query_token_aggregator = use_query_token_aggregator
        self.no_acoustic_aggregator = no_acoustic_aggregator
        # Fixed-rate aggregation parameters
        self.use_fixed_rate_aggregator = use_fixed_rate_aggregator
        self.aggregator_downsample_ratio = aggregator_downsample_ratio
        self.aggregator_downsample_ratio_options = aggregator_downsample_ratio_options
        if self.use_fixed_rate_aggregator:
            assert not self.use_similarity_alignment, "Cannot use both fixed-rate aggregation and similarity-based alignment."
            assert self.use_query_token_aggregator, "Fixed-rate aggregation requires use_query_token_aggregator to be True."
        
        # Bottleneck transformer parameters
        self.use_bottleneck_transformer = use_bottleneck_transformer
        self.use_second_decoder_transformer = use_second_decoder_transformer
        if self.use_bottleneck_transformer:
            transformer_kwargs = {
                'd_model': transformer_dim,
                'num_heads': transformer_num_heads,
                'num_layers': transformer_num_layers,
                'causal': transformer_causal,
                'layer_scale': 0.01,
                'context': transformer_context_frames,  # Use calculated context window
                'conv_layout': True,
                'max_period': 10000,
                'gating': 'none',
                'norm': 'layer_norm',
                'positional_embedding': 'rope',
                'dim_feedforward': transformer_dim_feedforward,
                'input_dimension': latent_dim,
                'output_dimensions': [latent_dim],
            }
            if transformer_num_layers == 0:
                self.bottleneck_transformer = nn.Identity()
            else:
                self.bottleneck_transformer = Stransformer.ProjectedTransformer(**transformer_kwargs)
            
            if self.use_second_decoder_transformer:
                transformer_2_kwargs = transformer_kwargs.copy()
                if transformer_2_num_layers is not None:
                    transformer_2_kwargs['num_layers'] = transformer_2_num_layers
                self.bottleneck_transformer_2 = Stransformer.ProjectedTransformer(**transformer_2_kwargs)
            else:
                self.bottleneck_transformer_2 = nn.Identity()

            transformer_kwargs_repa = {
                'd_model': latent_dim,
                'num_heads': transformer_num_heads,
                'num_layers': 4,
                'causal': transformer_causal,
                'layer_scale': 0.01,
                'context': transformer_context_frames,  # Use calculated context window
                'conv_layout': True,
                'max_period': 10000,
                'gating': 'none',
                'norm': 'layer_norm',
                'positional_embedding': 'rope',
                'dim_feedforward': transformer_dim_feedforward,
                'input_dimension': latent_dim,
                'output_dimensions': [latent_dim],
            }
            self.repa_mlp = nn.Sequential(
                nn.Linear(latent_dim, latent_dim*3),
                nn.GELU(),
                nn.Linear(latent_dim*3, latent_dim),
            )
            # self.bottleneck_transformer_repa = Stransformer.ProjectedTransformer(**transformer_kwargs_repa)
        else:
            self.bottleneck_transformer = nn.Identity()
            self.bottleneck_transformer_2 = nn.Identity()
        
        # Initialize semantic model based on type
        self.semantic_model_type = semantic_model_type
        self.sensevoice_prepend_inputs = sensevoice_prepend_inputs
        
        if semantic_model_type == "sensevoice":
            # reset semantic downsample factor
            ssl_dim=512
            # Store SenseVoice specific parameters
            
            # override model_code_dir
            from pathlib import Path
            sensevoice_model_code_dir = f'{str(Path(__file__).parent)}/customized_sensevoice/model.py'
            # Initialize FunASR model directly
            funasr_model = AutoModel(
                model=semantic_model_path,
                trust_remote_code=True,
                remote_code=sensevoice_model_code_dir,
                device="cpu",
                disable_update=True
            )
            # Set semantic_model to the model directly, similar to audio_encoder.py
            self.semantic_model = funasr_model.model
            # For FunASR model, we don't need mean/var stats as normalization is handled internally
            self.register_buffer("semantic_mean", torch.zeros(1))
            self.register_buffer("semantic_std", torch.ones(1))
            self.semantic_model_path = semantic_model_path
        else:
            from pathlib import Path
            # Default Wav2Vec2BertModel initialization
            mean_var_path = f'{str(Path(__file__).parent)}/w2vbert2_mean_var_stats_emilia.pt'
            stat_mean_var = torch.load(mean_var_path)
            self.register_buffer("semantic_mean", stat_mean_var["mean"])
            self.register_buffer("semantic_std", stat_mean_var["var"])
            self.semantic_model_path = semantic_model_path
            self.semantic_model = Wav2Vec2BertModel.from_pretrained(self.semantic_model_path).eval()
            if half_semantic_model:
                self.semantic_model = self.semantic_model.half()
        
        self.freeze_semantic_model()
        self.trainer_callbacks = None
        self.sample_rate = sample_rate
        self.residual_autoencode = residual_autoencode
        self.infer_using_dynamic_threshold = False
        if not residual_autoencode:
            from zero_shot_tts_training.realtime_communication.codec_model.dac_model import DAC
            self.dac = DAC(
                encoder_dim,
                encoder_rates,
                latent_dim,
                decoder_dim,
                decoder_rates,
                n_codebooks,
                codebook_size,
                codebook_dim,
                quantizer_dropout,
                sample_rate,
                distill_projection_out_dim,
                distill=False,
            )
        else:
            from zero_shot_tts_training.realtime_communication.codec_model.dac_model_dcae import DAC
            self.dac = DAC(
                encoder_dim,
                encoder_rates,
                latent_dim,
                decoder_dim,
                decoder_rates,
                n_codebooks,
                codebook_size,
                codebook_dim,
                quantizer_dropout,
                sample_rate,
                distill_projection_out_dim,
                distill=False,
                residual_autoencode=residual_autoencode,
            )

        if override_dac_encoder is not None:
            self.dac.encoder = override_dac_encoder
            self.override_dac_encoder = True
        else:
            self.override_dac_encoder = False
        if override_vocos_decoder is not None:
            self.dac.decoder = override_vocos_decoder
            self.override_vocos_decoder = True
        else:
            self.override_vocos_decoder = False

        self.decode_semantic_for_codec = decode_semantic_for_codec
        self.encoder_rates = encoder_rates
        self.ssl_dim = ssl_dim
        self.dac_bn_dim = self.dac.latent_dim
        self.manual_threshold = None

        # get the dim after downsampling
        if self.use_concat_downsampling:
            assert not self.use_conv_downsampling
            print('using concat downsampling')
            # reset semantic_downsample_factor so that it will not perform avg pooling
            self.concat_downsample_factor = semantic_downsample_factor
            self.semantic_downsample_factor = 1

        self.convnext_encoder = nn.Sequential(
            WNConv1d(
                self.ssl_dim, convnext_dim, kernel_size=1,
            ),
            *[ConvNeXtBlock(
                dim=convnext_dim,
                intermediate_dim=2048,
                is_causal=is_causal
            ) for _ in range(convnext_layers)],  # Unpack the list directly into nn.Sequential
        )
        if semantic_encoder is not None:
            self.convnext_encoder = semantic_encoder

        if self.use_bsq_for_semantic_vq:
            bsq_params = (bsq_config or {}).copy()
            bsq_embed_dim = bsq_params.pop('embed_dim', 14)
            self.semantic_vq = BSQWrapper(
                input_dim=convnext_dim,
                embed_dim=bsq_embed_dim,
                **bsq_params
            )
        elif self.use_fsq_for_semantic_vq:
            fsq_params = (fsq_config or {}).copy()
            self.semantic_vq = FSQWrapper(
                input_dim=convnext_dim,
                **fsq_params
            )
        else:
            self.semantic_vq = ResidualVectorQuantize(
                convnext_dim, n_codebooks=1, codebook_size=semantic_codebook_size,
                codebook_dim=semantic_codebook_dim,
            )

        self.convnext_decoder = nn.Sequential(
            *[ConvNeXtBlock(
                dim=convnext_dim,
                intermediate_dim=2048,
                is_causal=is_causal,
            ) for _ in range(convnext_layers)],  # Unpack the list directly into nn.Sequential
            WNConv1d(
                convnext_dim, self.dac_bn_dim, kernel_size=1,
            ),
        )
        if semantic_decoder is not None:
            self.convnext_decoder = semantic_decoder
            if self.use_bsq_for_semantic_vq:
                bsq_params = (bsq_config or {}).copy()
                bsq_embed_dim = bsq_params.pop('embed_dim', 14)
                self.semantic_vq = BSQWrapper(
                    input_dim=1024,
                    embed_dim=bsq_embed_dim,
                    **bsq_params
                )
            elif self.use_fsq_for_semantic_vq:
                fsq_params = (fsq_config or {}).copy()
                self.semantic_vq = FSQWrapper(
                    input_dim=1024,
                    **fsq_params
                )
            else:
                self.semantic_vq = ResidualVectorQuantize(
                    1024, n_codebooks=1, codebook_size=semantic_codebook_size,
                    codebook_dim=semantic_codebook_dim,
                )
        # if not self.decode_semantic_for_codec:
        #     assert convnext_dim == 1024
        self.sensevoice_sim_layer_idx = sensevoice_sim_layer_idx
        self.sensevoice_semantic_layer_idx = sensevoice_semantic_layer_idx
        self.distill_with_avg = distill_with_avg

        self.use_flow_matching_decoder = use_flow_matching_decoder
        # Import VoiceBox for flow matching decoder
        if self.use_flow_matching_decoder:
            if flow_matching_has_prompt:
                from zero_shot_tts_training.voicebox.voicebox_model import VoiceBox
            else:
                from zero_shot_tts_training.voicebox.voicebox_models_codec import VoiceBox
            # Flow matching decoder initialization
            # override dac decoder
            self.dac.decoder = None
            # Create a configuration object for VoiceBox with streaming transformer
            flow_matching_cfg = edict({
                'mel_dim': flow_matching_mel_dim,
                'hidden_size': flow_matching_hidden_size,
                'num_layers': flow_matching_num_layers,
                'num_heads': flow_matching_num_heads,
                'cfg_scale': flow_matching_cfg_scale,
                'use_cond_code': flow_matching_use_cond_code,
                'cond_codebook_size': flow_matching_cond_codebook_size,
                'cond_dim': flow_matching_cond_dim,
                'cond_scale_factor': flow_matching_cond_scale_factor,
                'time_scheduler': flow_matching_time_scheduler,
                'context': flow_matching_context,  # Context window for variable length generalization
                'causal': flow_matching_causal,  # Whether to use causal attention
                'repa_layer_idx': repa_layer_idx,  # Layer index for REPA loss
            })
            
            self.flow_matching_decoder = VoiceBox(**flow_matching_cfg)
            
            # Create REPA projection MLP only when use_repa_loss is True
            if self.use_repa_loss:
                self.repa_projection = nn.Sequential(
                    nn.Linear(flow_matching_hidden_size, flow_matching_hidden_size * 4),
                    nn.SiLU(),
                    nn.Linear(flow_matching_hidden_size * 4, repa_projection_dim),
                )
            else:
                self.repa_projection = None
        else:
            self.flow_matching_decoder = None
            self.repa_projection = None

        if self.use_query_token_aggregator:
            self.semantic_aggregator = QueryTokenAggregator(
                dim=agg_transformer_dim,
                in_out_dim=ssl_dim,
                num_heads=agg_transformer_num_heads,
                num_layers=agg_transformer_num_layers,
                dim_feedforward=agg_transformer_dim_feedforward,
                causal=agg_transformer_causal,
                use_mean_pooling_init=agg_use_mean_pooling_init,
                add_query_embedding=agg_add_query_embedding,
                context_frames=transformer_context_frames,
            )
            if not self.no_acoustic_aggregator:
                self.acoustic_aggregator = QueryTokenAggregator(
                    dim=self.dac_bn_dim, # latent_dim
                    in_out_dim=self.dac_bn_dim,
                    num_heads=agg_transformer_num_heads,
                    num_layers=agg_transformer_num_layers,
                    dim_feedforward=agg_transformer_dim_feedforward,
                    causal=agg_transformer_causal,
                    use_mean_pooling_init=agg_use_mean_pooling_init,
                    add_query_embedding=agg_add_query_embedding,
                    context_frames=agg_transformer_context_frames,
                )
            else:
                # alias the semantic_aggregator to acoustic_aggregator
                self.acoustic_aggregator = self.semantic_aggregator
        if flow_matching_has_prompt:
            print("Freezing all parameters except for flow matching decoder and repa_projection.")
            for name, param in self.named_parameters():
                if 'flow_matching_decoder' not in name and 'repa_projection' not in name:
                    param.requires_grad = False
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"DualCodec - Total Parameters: {total_params / 1e6:.2f}M")
        print(f"DualCodec - Trainable Parameters: {trainable_params / 1e6:.2f}M")
        
        # Print detailed submodule analysis
        self.print_submodule_params()
    def _get_current_similarity_threshold(self) -> float:
        """
        Get the current similarity threshold for alignment.
        If using flex framerate, randomly selects from flex_framerate_options during training.
        If using dynamic threshold, returns a random value between lower and upper bounds.
        Otherwise, returns the fixed threshold.
        
        Returns:
            float: Current similarity threshold value
        """
        if self.manual_threshold is not None:
            return float(self.manual_threshold)
        elif self.flex_framerate and self.training:
            # Select randomly from flex_framerate_options during training
            threshold = random.choice(self.flex_framerate_options)
            return threshold
        elif (self.use_dynamic_similarity_threshold and self.training) or self.infer_using_dynamic_threshold:
            # Sample a random threshold between lower and upper bounds
            threshold = random.uniform(self.similarity_threshold_lower, self.similarity_threshold_upper)
            return threshold
        else:
            return self.similarity_threshold

    def _get_current_aggregator_downsample_ratio(self) -> int:
        """
        Get the current aggregator downsample ratio.
        If aggregator_downsample_ratio_options is provided, randomly selects from the options during training.
        Otherwise, returns the fixed aggregator_downsample_ratio.
        
        Returns:
            int: Current aggregator downsample ratio value
        """
        if self.aggregator_downsample_ratio_options is not None and self.training:
            # Select randomly from aggregator_downsample_ratio_options during training
            ratio = random.choice(self.aggregator_downsample_ratio_options)
            return int(ratio)
        else:
            return int(self.aggregator_downsample_ratio)

    def _downsample_semantic_features(self, features: torch.Tensor) -> torch.Tensor:
        """
        Downsample semantic features using either avg_pool1d (for integer factors) 
        or interpolate (for fractional factors).
        
        Args:
            features: torch.Tensor, shape (B, C, T) - semantic features to downsample
            
        Returns:
            torch.Tensor: downsampled features
        """
        if self.semantic_downsample_factor == 1:
            return features
        
        # Check if downsample factor is an integer
        if self.semantic_downsample_factor == int(self.semantic_downsample_factor):
            # Use avg_pool1d for integer factors
            return torch.nn.functional.avg_pool1d(
                features,
                self.semantic_downsample_factor,
                self.semantic_downsample_factor,
            )
        else:
            # Use interpolate for fractional factors
            target_length = int(features.shape[-1] / self.semantic_downsample_factor)
            return torch.nn.functional.interpolate(
                features,
                size=target_length,
                mode='linear',
                align_corners=False
            )

    def _downsample_x_lens(self, x_lens: torch.Tensor) -> torch.Tensor:
        """
        Downsample x_lens by the same factor as semantic features.
        
        Args:
            x_lens: torch.Tensor, shape (B,) - original feature lengths
            
        Returns:
            torch.Tensor: downsampled lengths
        """
        if self.semantic_downsample_factor == 1 or x_lens is None:
            return x_lens
        
        # Check if downsample factor is an integer
        if self.semantic_downsample_factor == int(self.semantic_downsample_factor):
            # For integer factors, divide and round up to ensure we don't lose valid frames
            downsampled_lens = torch.div(x_lens, self.semantic_downsample_factor, rounding_mode='floor')
        else:
            # For fractional factors, apply the same scaling
            downsampled_lens = (x_lens.float() / self.semantic_downsample_factor).long()
        
        # Ensure we have at least 1 frame if original length > 0
        downsampled_lens = torch.where(x_lens > 0, torch.clamp(downsampled_lens, min=1), downsampled_lens)
        
        return downsampled_lens

    def freeze_semantic_model(self):
        """Freeze all parameters in the semantic model."""
        if self.semantic_model_type == "sensevoice":
            # Freeze FunASR model parameters
            for param in self.semantic_model.parameters():
                param.requires_grad = False
        else:
            # Freeze Wav2Vec2BertModel parameters
            for param in self.semantic_model.parameters():
                param.requires_grad = False
    def semantic_quantize(self, semantic_repr):
        if self.override_dac_encoder:
            pad_amount = audio_data.shape[1] % self.concat_downsample_factor
            audio_data = audio_data[:, :audio_data.shape[1] - pad_amount]
            semantic_repr = semantic_repr[..., :semantic_repr.shape[-1] - pad_amount]
            # audio_data = torch.nn.functional.pad(audio_data, (0, pad_amount))
            # semantic_repr = torch.nn.functional.pad(semantic_repr, (0, pad_amount))
            semantic_repr = rearrange(semantic_repr, 'b c (t k) -> b (k c) t', k=self.concat_downsample_factor)
            # encoded_feature = self.dac.encoder(audio_data)
        elif self.use_concat_downsampling:
            # semantic_repr = semantic_repr[..., 1:]
            # left pad the same as first frame
            semantic_repr = torch.nn.functional.pad(semantic_repr, (1,0), mode='reflect')
            semantic_repr = rearrange(semantic_repr, 'b c (t k) -> b (k c) t', k=self.concat_downsample_factor)
        semantic = self.convnext_encoder(semantic_repr)
            
        semantic, codes, latents, commitment_loss, codebook_loss, first_layer_quantized = self.semantic_vq(semantic)
        codes = rearrange(codes, 'b 1 t -> b t')
        return codes

    def decode_from_codes(self, semantic_codes, acoustic_codes, token_lengths=None):
        """
        Decodes from semantic and acoustic codes. If token_lengths are provided,
        it assumes features are aggregated and will de-aggregate them.
        
        Args:
            semantic_codes (torch.Tensor): semantic codes of shape [B, n_q_s, G]
            acoustic_codes (torch.Tensor): acoustic codes of shape [B, n_q_a, G] or None
            token_lengths: Optional[torch.Tensor], shape (B, G)
                If provided, will de-aggregate the features after VQ decoding.
                Each value is the number of frames in the corresponding group.
        """
        semantic = self.semantic_vq.from_codes(semantic_codes)[0]
        if self.decode_semantic_for_codec:
            semantic_decoded = self.convnext_decoder(semantic)
        else:
            semantic_decoded = semantic

        # Handle alignment-based decoding (de-aggregation)
        is_aggregated = (self.use_similarity_alignment or self.use_fixed_rate_aggregator)
        if is_aggregated and token_lengths is not None:
            # De-aggregate semantic features to match acoustic dimensions
            semantic_expanded = self._deaggregate_features_from_token_lengths(semantic_decoded, token_lengths)
            
            # Decode acoustic codes normally first
            if acoustic_codes is not None:
                acoustic_vq = self.dac.quantizer.from_codes(acoustic_codes)[0]
                
                # De-aggregate acoustic features
                acoustic_expanded = self._deaggregate_features_from_token_lengths(acoustic_vq, token_lengths)
            else:
                acoustic_expanded = 0.0
            
            # Add semantic contribution to acoustic features
            acoustic_final = acoustic_expanded + semantic_expanded
            
            # Apply bottleneck transformers
            acoustic_final = self.bottleneck_transformer(acoustic_final)
            acoustic_final = self.bottleneck_transformer_2(acoustic_final)
            
            # Use flow matching decoder if enabled
            if self.use_flow_matching_decoder:
                if not hasattr(self, 'infer_vocos'):
                    from zero_shot_tts_training.realtime_communication.taste_v2.modeling_dualcodec import get_vocoder_decode_func_and_mel_spec
                    self.infer_vocos, _ = get_vocoder_decode_func_and_mel_spec()
                
                # Convert acoustic features for flow matching inference
                cond_feature = self.flow_matching_decoder.cond_emb(acoustic_final.transpose(1,2))
                mel_spec_rev = self.flow_matching_decoder.reverse_diffusion(cond_feature, mel_mask=None, n_timesteps=30, cfg=1.0)
                audio = self.infer_vocos(mel_spec_rev.transpose(1,2)).cpu()
                # Resample to 16k if needed
                import torchaudio
                audio = torchaudio.functional.resample(audio, 24000, 16000)
            else:
                # Use original DAC decoder
                audio = self.dac.decoder(acoustic_final)
        else:
            # Original decoding without alignment/aggregation
            audio = self.dac.decode_from_codes(acoustic_codes, semantic_latent=semantic_decoded)
        
        return audio
    
    def decode_from_latent(self, latent, token_lengths):
        acoustic_final = self._deaggregate_features_from_token_lengths(latent, token_lengths)
        acoustic_final = self.bottleneck_transformer(acoustic_final) # TODO match the shape of acoustic_final
        acoustic_final = self.bottleneck_transformer_2(acoustic_final)
        audio_output = self.dac.decoder(acoustic_final)
        return audio_output
    
    def forward(self, dl_output, encode_only=False, infer_using_dynamic_threshold=False):
        audio_data = dl_output.get("audio", dl_output).float()
        if len(audio_data.shape) == 2:
            audio_data = audio_data.unsqueeze(1) # [B, 1, T]
        audio_features = dl_output.get("x", dl_output).float()
        x_lens = dl_output.get("x_lens", None)  # Optional input lengths [batch_size]
        mel_mask = dl_output.get("mel_mask", None)  # Optional mask for flow matching
        manual_threshold = dl_output.get("manual_threshold", None)  # Optional mask for flow matching
        audio_features_masks = torch.ones_like(audio_features[:,:,0])
        semantic_repr, sim_repr = _extract_semantic_code(
            self.semantic_model,
            audio_features,
            audio_features_masks,
            self.semantic_mean,
            self.semantic_std,
            skip_normalize=self.skip_normalize,
            sensevoice_prepend_inputs=self.sensevoice_prepend_inputs,
            sim_layer_idx=self.sensevoice_sim_layer_idx,
            semantic_layer_idx=self.sensevoice_semantic_layer_idx,
        )
        semantic_repr = semantic_repr.transpose(1,2)
        sim_repr = sim_repr.transpose(1,2)
        out_dict = self.forward_features(
            audio_data,
            self.sample_rate,
            semantic_repr=semantic_repr,
            alignment_hidden=sim_repr,
            n_quantizers=dl_output.get("num_quantizers", None),
            possibly_no_quantizer=False,
            mel=dl_output.get("mel", None),
            encode_only=encode_only,
            x_lens=x_lens,
            mel_mask=mel_mask,
            infer_using_dynamic_threshold=infer_using_dynamic_threshold,
            manual_threshold=manual_threshold,
        )
        return out_dict
    def forward_features(self, 
            audio_data: torch.Tensor,
            sample_rate: int = 24000,
            n_quantizers: int = None,
            semantic_repr=None,
            alignment_hidden=None,
            bypass_quantize_rate=0.125,
            possibly_no_quantizer=False,
            mel=None,
            encode_only: bool = False,
            x_lens=None,
            mel_mask=None,
            infer_using_dynamic_threshold: bool = False,
            manual_threshold=None,
        ):
        """
        semantic_repr: [B, C, T] at same frame rate as acoustic codes
        alignment_hidden: Optional representation [B, C, T] to compute similarity alignment; if None, use semantic_repr
        """
        if manual_threshold is not None:
            self.manual_threshold = manual_threshold
            assert not infer_using_dynamic_threshold
            self.training = False
        if infer_using_dynamic_threshold:
            self.training = False
            n_quantizers = None
            self.infer_using_dynamic_threshold = True
        else:
            if encode_only:
                self.training = False
        if mel is not None:
            mel = mel.transpose(1,2)
        semantic_repr_before_downsample = semantic_repr.clone().detach()
        semantic_repr = self._downsample_semantic_features(semantic_repr)

        if alignment_hidden is not None:
            alignment_hidden = self._downsample_semantic_features(alignment_hidden)
        else:
            alignment_hidden = semantic_repr
        semantic_repr_ret = semantic_repr.clone().detach()
        
        # Downsample x_lens by the same factor as semantic features
        x_lens_downsampled = self._downsample_x_lens(x_lens)
        
        # Regular audio processing
        audio_data_preprocessed = self.dac.preprocess(audio_data, sample_rate)
        acoustic_features = self.dac.encoder(audio_data_preprocessed)
        
        # Generate alignment matrix if using similarity-based alignment
        alignment_matrices = None
        num_segments_per_item = None
        is_aggregated = self.use_similarity_alignment or self.use_fixed_rate_aggregator
        sim = None

        # Ensure time dimensions match
        if acoustic_features.shape[-1] != semantic_repr.shape[-1]:
            # assert the shape difference is at most 2
            assert abs(acoustic_features.shape[-1] - semantic_repr.shape[-1]) <= 2
            min_len = min(acoustic_features.shape[-1], semantic_repr.shape[-1])
            acoustic_features = acoustic_features[..., :min_len]
            semantic_repr = semantic_repr[..., :min_len]
            semantic_repr_ret = semantic_repr_ret[..., :min_len]
        if self.use_similarity_alignment:
            # Vectorized alignment computation for the whole batch, based on semantic_repr
            h_frames_batch = alignment_hidden.transpose(1, 2)  # (B, T, D)
            alignment_matrices, sim, num_segments_per_item = self._perform_similarity_alignment_vectorized(h_frames_batch, x_lens=x_lens_downsampled)
        elif self.use_fixed_rate_aggregator:
            alignment_matrices, num_segments_per_item = self._generate_fixed_rate_alignment(acoustic_features, x_lens=x_lens_downsampled)

        if is_aggregated:

            if self.use_query_token_aggregator:
                if self.insert_query_before_downsample and self.semantic_downsample_factor != 1:
                    # Upsample alignment matrix for semantic features before downsampling
                    # by scale factor, then crop if there is a mismatch.
                    alignment_matrices_for_semantic = F.interpolate(
                        alignment_matrices.float(),
                        scale_factor=self.semantic_downsample_factor,
                        mode='nearest',
                    )
                    assert abs(alignment_matrices_for_semantic.shape[-1] - semantic_repr_before_downsample.shape[-1]) <= 2, f"alignment_matrices_for_semantic.shape[-1]: {alignment_matrices_for_semantic.shape[-1]}, semantic_repr_before_downsample.shape[-1]: {semantic_repr_before_downsample.shape[-1]}"
                    alignment_matrices_for_semantic = alignment_matrices_for_semantic[..., :semantic_repr_before_downsample.shape[-1]]
                    semantic_repr = self.semantic_aggregator(semantic_repr_before_downsample, alignment_matrices_for_semantic, num_segments_per_item)
                else:
                    # Original logic: aggregate semantic features after downsampling
                    semantic_repr = self.semantic_aggregator(semantic_repr, alignment_matrices, num_segments_per_item)

                acoustic_aggregated = self.acoustic_aggregator(acoustic_features, alignment_matrices, num_segments_per_item)
                # For distillation loss, aggregate the GT features as well
                if not self.distill_with_avg:
                    semantic_repr_gt_agg = self.semantic_aggregator(semantic_repr_ret, alignment_matrices, num_segments_per_item)
                else:
                    semantic_repr_gt_agg = self.aggregate_features(semantic_repr_ret, alignment_matrices)
            else:
                # Aggregate `semantic_repr` BEFORE convnext using simple mean-pooling
                semantic_repr = self.aggregate_features(semantic_repr, alignment_matrices)
                # Aggregate acoustic features
                acoustic_aggregated = self.aggregate_features(acoustic_features, alignment_matrices)
                # Aggregate ground truth representation for distillation loss
                semantic_repr_gt_agg = self.aggregate_features(semantic_repr_ret, alignment_matrices)
            
            # Process aggregated semantic stream
            semantic_aggregated = self.convnext_encoder(semantic_repr)

        else:
            # No alignment - process semantic features directly
            semantic_aggregated = self.convnext_encoder(semantic_repr)
            acoustic_aggregated = acoustic_features
            semantic_repr_gt_agg = semantic_repr_ret  # already match shape
        
        # Quantize semantic stream
        semantic_vq, semantic_codes, latents, commitment_loss, codebook_loss, first_layer_quantized = self.semantic_vq(semantic_aggregated)
        
        if self.decode_semantic_for_codec:
            semantic_decoded = self.convnext_decoder(semantic_vq)
        else:
            semantic_decoded = semantic_vq

        # Prepare for acoustic quantization
        bypass_quantize = random.random() < bypass_quantize_rate
        if not self.training:
            bypass_quantize = False
        if n_quantizers == 1:
            bypass_quantize = True
        if n_quantizers is not None:
            n_quantizers = n_quantizers - 1
        
        # Use aggregated semantic latent for subtraction
        subtracted_latent_agg = semantic_decoded  # already aggregated if alignment enabled, else passthrough
        
        # Quantize acoustic stream (with aggregated features if using alignment)
        if is_aggregated:
            # For aggregated acoustic features, we need to modify DAC's quantization
            acoustic_vq_input = acoustic_aggregated - subtracted_latent_agg
            
            if bypass_quantize:
                acoustic_codes, acoustic_latents, acoustic_commitment_loss, acoustic_codebook_loss = \
                    None, None, 0.0, 0.0
                acoustic_vq = 0.0
            else:
                acoustic_vq, acoustic_codes, acoustic_latents, acoustic_commitment_loss, acoustic_codebook_loss, _ = \
                    self.dac.quantizer(acoustic_vq_input, n_quantizers, possibly_no_quantizer=possibly_no_quantizer)

            
            if not bypass_quantize:
                # De-aggregate acoustic and semantic features separately, then sum them.
                # This keeps the computation graph cleaner for DDP to trace.
                acoustic_expanded = self.deaggregate_features(acoustic_vq, alignment_matrices)
                semantic_expanded = self.deaggregate_features(semantic_decoded, alignment_matrices)
                acoustic_final = acoustic_expanded + semantic_expanded
            else:
                # Bypassed quantization, directly expand semantic only
                acoustic_final = self.deaggregate_features(semantic_decoded, alignment_matrices)
                semantic_expanded = acoustic_final
            
            if encode_only:
                assert not self.training
                token_lengths = alignment_matrices.sum(dim=2).long()
                
                # Deaggregate semantic codes to match original frame rate
                semantic_codes_deaggregated = self._deaggregate_features_from_token_lengths(
                    semantic_codes.float(), token_lengths
                ).long()
                
                return_dict = edict({
                    "semantic_codes": semantic_codes,  # Aggregated codes [B, 1, G]
                    "semantic_codes_deaggregated": semantic_codes_deaggregated,  # Deaggregated codes [B, 1, T]
                    "acoustic_codes": acoustic_codes,
                    "token_lengths": token_lengths,
                    "alignment_matrix": alignment_matrices,
                    # "semantic_features": semantic_expanded.squeeze(0).cpu().detach().transpose(0,1) if not self.training else None,
                    "semantic_features": semantic_expanded.cpu().detach() if not self.training else None,
                    "speech_token_len": num_segments_per_item,  # Valid speech token lengths after aggregation
                    "semantic_repr_ret": semantic_repr_ret,
                    "decoder_latent": acoustic_vq+semantic_decoded,
                    "decoder_latent_before_agg": acoustic_final,
                    "semantic_repr_before_downsample": semantic_repr_before_downsample,
                    "sim": None if sim is None else sim
                })
                return return_dict
            # Decode to audio
            # [b t c]
            # num_params = sum(p.numel() for p in self.bottleneck_transformer.parameters())
            acoustic_final = self.bottleneck_transformer(acoustic_final) # TODO match the shape of acoustic_final

            if self.use_bottleneck_transformer:
                if self.no_acoustic_aggregator:
                    repa_gt = semantic_repr_ret
                    repa_output = self.repa_mlp(acoustic_final.transpose(1,2))
                    repa_loss = F.mse_loss(repa_gt, repa_output.transpose(1,2)) * 15.0
                    if self.sometimes_skip_repa:
                        # repa_loss = torch.tensor(0.0, device=acoustic_final.device) v4
                        if not bypass_quantize:
                            repa_loss = torch.tensor(0.0, device=acoustic_final.device) # v5
                else:
                    repa_gt = semantic_repr_ret + acoustic_features if not bypass_quantize else semantic_repr_ret
                    repa_output = self.repa_mlp(acoustic_final.transpose(1,2))
                    repa_loss = F.mse_loss(repa_gt, repa_output.transpose(1,2))
            else:
                repa_loss = torch.tensor(0.0, device=acoustic_final.device)
            acoustic_final = self.bottleneck_transformer_2(acoustic_final)

            # Use flow matching decoder if enabled
            if self.use_flow_matching_decoder:
                if self.training:
                    # Convert acoustic features to mel spectrogram format for flow matching
                    # acoustic_final shape: [B, D, T] -> need to convert to mel format
                    mel_features = mel  # [B, T, D]
                    assert mel is not None
                    
                    # Use mel_mask if provided, otherwise create default mask
                    if mel_mask is not None:
                        # mel_mask is [B, 1, T], convert to [B, T] for flow matching
                        if len(mel_mask.shape) == 3:
                            mel_mask = mel_mask.squeeze(1)  # [B, T]
                        else:
                            mel_mask = mel_mask  # [B, T]
                        # Ensure mel_mask has the same length as mel_features
                        print(f"mel_mask.shape: {mel_mask.shape}, mel_features.shape: {mel_features.shape}")
                        if mel_mask.shape[1] != mel_features.shape[1]:
                            # Pad or crop mel_mask to match mel_features length
                            if mel_mask.shape[1] < mel_features.shape[1]:
                                mel_mask = F.pad(mel_mask, (0, mel_features.shape[1] - mel_mask.shape[1]), value=False)
                            else:
                                mel_mask = mel_mask[:, :mel_features.shape[1]]
                    else:
                        # Create default mask for flow matching
                        mel_mask = torch.ones(mel_features.shape[0], mel_features.shape[1], 
                                            dtype=torch.bool, device=mel_features.device)
                    
                    # Always use the method that returns hidden features to ensure REPA projection parameters are used
                    noise, x, flow_pred, final_mask, prompt_len, hidden_features = self.flow_matching_decoder.forward_with_hidden_features(
                        mel_features, mel_mask, cond_code=None, cond_feature=acoustic_final.transpose(1,2)
                    )
                    
                    # For flow matching, we need to compute the flow matching loss
                    # The flow_pred is the predicted flow, and we need to compute the ground truth flow
                    flow_gt = x - (1 - self.flow_matching_decoder.sigma) * noise
                    
                    # Compute flow matching loss
                    flow_matching_loss = F.l1_loss(
                        flow_pred, flow_gt, reduction="mean"
                    ).float()
                    
                    # Compute REPA loss
                    repa_loss_value = torch.tensor(0.0, device=flow_matching_loss.device)
                    if self.use_repa_loss and self.repa_projection is not None:
                        # Get SenseVoice representations for REPA loss
                        target_repr = semantic_repr_before_downsample  # [B, D, T] - SenseVoice representation
                        
                        # Apply REPA projection to hidden features
                        # hidden_features shape: [B, T, hidden_size] -> [B, T, repa_projection_dim]
                        projected_features = hidden_features
                        
                        # Apply 3x downsampling to projected features to match SenseVoice frame rate
                        B, T, D = projected_features.shape
                        target_length = T // 3  # 3x downsampling
                        
                        # Use average pooling with stride=3 to downsample
                        projected_features_downsampled = projected_features.transpose(1, 2)  # [B, D, T]
                        projected_features_downsampled = F.avg_pool1d(
                            projected_features_downsampled,
                            kernel_size=3,
                            stride=3
                        )  # [B, D, T//3]
                        projected_features_downsampled = self.repa_projection(projected_features_downsampled.transpose(1, 2)).transpose(1, 2)
                        
                        # Use the helper method to compute REPA loss with downsampled projected features
                        repa_loss_value = self.compute_repa_loss(projected_features_downsampled, target_repr, mask=None)
                    else:
                        repa_loss_value = torch.tensor(0.0)
                    
                    # Store flow matching loss for later use
                    flow_matching_loss_value = flow_matching_loss
                    audio_output = None
                else:
                    cond_feature = self.flow_matching_decoder.cond_emb(acoustic_final.transpose(1,2))
                    mel_spec_rev = self.flow_matching_decoder.reverse_diffusion(cond_feature, mel_mask=None, n_timesteps=30, cfg=1.0)
                    if not hasattr(self, 'infer_vocos'):
                        self.infer_vocos, _ = get_vocoder_decode_func_and_mel_spec()
                    audio_output = self.infer_vocos(mel_spec_rev.transpose(1,2)).cpu()
                    # resample to 16k
                    audio_output = torchaudio.functional.resample(audio_output, 24000, 16000)
                    flow_matching_loss_value = torch.tensor(0.0, device=acoustic_final.device)
                    repa_loss_value = torch.tensor(0.0, device=acoustic_final.device)
            else:
                # Use original DAC decoder
                audio_output = self.dac.decoder(acoustic_final)
                flow_matching_loss_value = torch.tensor(0.0)
                repa_loss_value = torch.tensor(0.0)
            
            acoustic_edict = edict({
                "x": audio_output,
                "z": acoustic_final,
                "codes": acoustic_codes,
                "latents": acoustic_latents,
                "penalty": acoustic_commitment_loss,
                "vq/codebook_loss": acoustic_codebook_loss,
                "metrics": {},
            })
        else:
            # Original DAC processing (non-dynamic frame rate codec)
            if encode_only:
                # For non-aggregated case, we need to get acoustic codes without full decoding
                acoustic_vq_input = acoustic_features - subtracted_latent_agg
                
                if bypass_quantize:
                    acoustic_codes = None
                    acoustic_latents = None
                    acoustic_commitment_loss = 0.0
                    acoustic_codebook_loss = 0.0
                else:
                    _, acoustic_codes, acoustic_latents, acoustic_commitment_loss, acoustic_codebook_loss, _ = \
                        self.dac.quantizer(acoustic_vq_input, n_quantizers, possibly_no_quantizer=possibly_no_quantizer)
                
                # Calculate speech_token_len for non-aggregated case
                if x_lens_downsampled is not None:
                    speech_token_len = x_lens_downsampled
                else:
                    # Fallback to semantic_codes length if no x_lens provided
                    speech_token_len = torch.tensor([semantic_codes.shape[-1]] * semantic_codes.shape[0], 
                                                  device=semantic_codes.device, dtype=torch.long)
                return_dict = edict({
                    "semantic_codes": semantic_codes,
                    "semantic_codes_deaggregated": semantic_codes,
                    "acoustic_codes": acoustic_codes,
                    "token_lengths": None,  # No aggregation for non-dynamic frame rate
                    "alignment_matrix": torch.zeros(1,1,1),  # No alignment for non-dynamic frame rate
                    "semantic_features": semantic_decoded.cpu().detach() if not self.training else None,
                    "speech_token_len": speech_token_len,
                    "semantic_repr_ret": semantic_repr_ret,
                    # "decoder_latent": semantic_decoded+self.dac.quantizer.from_codes(acoustic_codes)[0],
                })
                if acoustic_codes is not None:
                    return_dict['decoder_latent_before_agg'] = semantic_decoded+self.dac.quantizer.from_codes(acoustic_codes)[0]
                return return_dict
            
            # Continue with full DAC processing for non-encode_only case
            acoustic_edict = self.dac(
                encoded_feature=acoustic_features, 
                sample_rate=sample_rate, 
                n_quantizers=n_quantizers, 
                subtracted_latent=subtracted_latent_agg, 
                bypass_quantize=bypass_quantize, 
                possibly_no_quantizer=possibly_no_quantizer,
                cut_from_front=False
            )
            repa_loss = torch.tensor(0.0)
            flow_matching_loss_value = torch.tensor(0.0)
            repa_loss_value = torch.tensor(0.0, device=acoustic_edict['x'].device if acoustic_edict['x'] is not None else acoustic_edict['z'].device)
            semantic_expanded = semantic_decoded
        
        if not self.decode_semantic_for_codec:
            semantic_decoded = self.convnext_decoder(semantic_vq)
            semantic_repr_ret = semantic_repr_ret[..., :semantic_decoded.shape[-1]]

        # Distillation loss uses aggregated ground truth and quantized semantic
        if not self.no_acoustic_aggregator:
            distill_loss = F.mse_loss(semantic_repr_gt_agg.detach(), semantic_decoded) * self.lambda_distill_loss
        else:
            distill_loss = torch.tensor(0.0)

        length = audio_data.shape[-1]
        if acoustic_edict['x'] is not None:
            if acoustic_edict['x'].shape[-1] < length:
                acoustic_edict['x'] = nn.functional.pad(acoustic_edict['x'], (0, length - acoustic_edict['x'].shape[-1]))
            else:
                acoustic_edict['x'] = acoustic_edict['x'][..., :length]

        # Prepare return dict
        if self.use_flow_matching_decoder:
            merged_edict = edict({
                # "repa_loss": repa_loss,
                "audio": acoustic_edict['x'],
                "flow_matching_loss": flow_matching_loss_value,
                "flow_matching_repa_loss": repa_loss_value,
                "vq/commitment_loss": acoustic_edict['penalty'] + commitment_loss,
                "vq/codebook_loss": acoustic_edict['vq/codebook_loss'] + codebook_loss,
                "distill_loss": distill_loss,
            })
        else:
            merged_edict = edict({
                "audio": acoustic_edict['x'],
                # "x": acoustic_edict['x'],
                "acoustic_codes": acoustic_edict['codes'],
                "semantic_codes": semantic_codes,
                "latents": acoustic_edict['latents'],
                # "penalty": acoustic_edict['penalty'] + commitment_loss,
                "vq/commitment_loss": acoustic_edict['penalty'] + commitment_loss,
                "vq/codebook_loss": acoustic_edict['vq/codebook_loss'] + codebook_loss,
                "metrics": acoustic_edict['metrics'],
                "semantic_repr": semantic_repr_ret,
                "distill_loss": distill_loss,
                "bypassed_quantize": bypass_quantize,
                "repa_loss": repa_loss,
                "flow_matching_loss": flow_matching_loss_value,
                "flow_matching_repa_loss": repa_loss_value,
                "semantic_features": semantic_expanded.squeeze(0).cpu().detach().transpose(0,1) if not self.training else None,
                # "semantic_features": semantic_vq.squeeze(0).cpu().detach().transpose(0,1) if not self.training else None,
                "token_lengths": None,
            })
        # Add alignment information if used
        # if self.use_similarity_alignment and alignment_matrices is not None:
        #     merged_edict["alignment_matrices"] = alignment_matrices
        #     merged_edict["num_segments_per_item"] = num_segments_per_item
        #     merged_edict["semantic_aggregated"] = semantic_aggregated
        #     merged_edict["acoustic_aggregated"] = acoustic_aggregated
        
        # Add compression ratio and speech token lengths if alignment is used
        if self.use_similarity_alignment or self.use_fixed_rate_aggregator:
            original_frames_lengths = alignment_matrices.shape[-1]
            num_segments = num_segments_per_item.float().mean().item()
            compression_ratio = (num_segments / original_frames_lengths)
            merged_edict["token_ratio"] = compression_ratio
            merged_edict["speech_token_len"] = num_segments_per_item  # Valid speech token lengths after aggregation
        else:
            # If not using alignment, speech_token_len is None (indicating no aggregation)
            merged_edict["speech_token_len"] = None
        return merged_edict

    def _generate_fixed_rate_alignment(self, features: torch.Tensor, x_lens=None):
        """
        Generate a fixed-rate alignment matrix for downsampling.
        
        Args:
            features: torch.Tensor, shape (B, D, T)
                The features to be downsampled. Used to determine shape and device.
            x_lens: torch.Tensor, shape (B,), optional
                Valid lengths for each item in the batch.
                
        Returns:
            torch.Tensor, shape (B, G, T) - The alignment matrix.
            torch.Tensor, shape (B,) - The number of segments per item.
        """
        B, _, T = features.shape
        device = features.device
        
        downsample_ratio = int(self._get_current_aggregator_downsample_ratio())
        
        if T == 0:
            return torch.zeros(B, 0, 0, device=device, dtype=torch.float), torch.zeros(B, device=device, dtype=torch.long)
            
        # 1. Calculate number of groups
        num_groups = (T + downsample_ratio - 1) // downsample_ratio

        # 2. Assign each frame to a group
        frame_indices = torch.arange(T, device=device)
        group_assignments = torch.div(frame_indices, downsample_ratio, rounding_mode='floor') # (T,)

        # 3. Create one-hot alignment matrix
        alignment_matrix = F.one_hot(group_assignments, num_classes=num_groups).float() # (T, G)
        alignment_matrix = alignment_matrix.transpose(0, 1).unsqueeze(0).expand(B, -1, -1) # (B, G, T)
        
        # 4. Mask out padding frames if x_lens is provided
        if x_lens is not None:
            valid_frame_mask = torch.arange(T, device=device).unsqueeze(0) < x_lens.unsqueeze(1)  # (B, T)
            alignment_matrix = alignment_matrix * valid_frame_mask.unsqueeze(1).float()  # Zero out padded frames
            
            # Calculate number of valid segments per item
            num_segments_per_item = torch.zeros(B, device=device, dtype=torch.long)
            for b in range(B):
                valid_length = x_lens[b]
                if valid_length > 0:
                    num_segments_per_item[b] = (valid_length + downsample_ratio - 1) // downsample_ratio
                else:
                    num_segments_per_item[b] = 0
        else:
            # Number of segments is the same for all batch items
            num_segments_per_item = torch.tensor([num_groups] * B, device=device, dtype=torch.long)
        
        return alignment_matrix, num_segments_per_item

    def _perform_similarity_alignment_vectorized(self, h_frames_v: torch.Tensor, x_lens=None):
        """
        Perform similarity-based alignment for an entire batch in a fully vectorized manner.
        
        Args:
            h_frames_v: torch.Tensor, shape (B, T, D)
                Hidden features for similarity calculation where B is the batch size,
                T is the number of frames and D is the hidden dimension.
            x_lens: torch.Tensor, shape (B,), optional
                Valid lengths for each item in the batch. If provided, only these frames
                will be considered for alignment computation.
                
        Returns:
            torch.Tensor, shape (B, max_groups, T)
                Padded binary alignment matrices for the batch where 1 indicates
                that the frame (column) belongs to the group (row). `max_groups`
                is the maximum number of groups among all items in the batch.
        """
        B, T, D = h_frames_v.shape

        if T <= 1:
            # All sequences are length 1 – return identity matrices with a single group
            return torch.ones(B, 1, T, device=h_frames_v.device, dtype=h_frames_v.dtype), \
                   torch.ones(B, T-1, device=h_frames_v.device, dtype=h_frames_v.dtype), \
                   torch.ones(B, device=h_frames_v.device, dtype=torch.long)

        # Create valid frame mask if x_lens is provided
        if x_lens is not None:
            # Create mask for valid frames
            valid_frame_mask = torch.arange(T, device=h_frames_v.device).unsqueeze(0) < x_lens.unsqueeze(1)  # (B, T)
        else:
            valid_frame_mask = torch.ones(B, T, device=h_frames_v.device, dtype=torch.bool)

        # 1. Cosine similarity between consecutive frames -> (B, T-1)
        sim = F.cosine_similarity(h_frames_v[:, :-1, :], h_frames_v[:, 1:, :], dim=2)

        # Mask out similarities for invalid transitions (where either frame is padding)
        if x_lens is not None:
            valid_transition_mask = valid_frame_mask[:, :-1] & valid_frame_mask[:, 1:]  # (B, T-1)
            # Set similarity to 0.0 (low similarity) for invalid transitions to force new segments
            # This prevents padding frames from being grouped with valid content
            sim = torch.where(valid_transition_mask, sim, torch.zeros_like(sim))

        # 2. Determine new segment boundaries (B, T-1)
        current_threshold = self._get_current_similarity_threshold()
        is_new_group_boundary = sim <= current_threshold

        # Pad a leading True to mark the start of the first segment -> (B, T)
        is_new_group_padded = torch.cat(
            [torch.ones(B, 1, dtype=torch.bool, device=h_frames_v.device), is_new_group_boundary], dim=1
        )

        # If max_tokens_per_group is set, we must also split segments that
        # are too long. This can be done efficiently in a vectorized way.
        if self.max_tokens_per_group is not None:
            # Find the index of each frame within its original segment
            arange_t = torch.arange(T, device=h_frames_v.device, dtype=torch.long).unsqueeze(0)
            segment_start_markers = arange_t * is_new_group_padded.long()
            last_segment_start_indices = torch.cummax(segment_start_markers, dim=1).values
            frame_indices_in_segment = arange_t - last_segment_start_indices
            
            # A new boundary is formed either by the original similarity score
            # or by reaching the maximum token limit.
            is_split_boundary = (frame_indices_in_segment % self.max_tokens_per_group) == 0
            
            # The final segment map is the cumulative sum of all boundaries.
            frame_to_segment_map = torch.cumsum(is_split_boundary.long(), dim=1) - 1
        else:
            # Original logic: only split based on similarity.
            frame_to_segment_map = torch.cumsum(is_new_group_padded.long(), dim=1) - 1
        
        # 4. Determine number of segments per item and global max
        # Only count segments that contain valid frames
        if x_lens is not None:
            # For each batch item, find the maximum segment index for valid frames
            num_segments_per_item = torch.zeros(B, device=h_frames_v.device, dtype=torch.long)
            for b in range(B):
                valid_length = x_lens[b]
                num_segments_per_item[b] = frame_to_segment_map[b, valid_length - 1] + 1
        else:
            num_segments_per_item = frame_to_segment_map.max(dim=1).values + 1  # (B,)
        # max_segments = int(num_segments_per_item.max().item())
        max_segments = int((frame_to_segment_map.max(dim=1).values + 1).max().item())

        # 5. Build alignment matrices using advanced indexing / scatter
        alignment_matrix = torch.zeros(B, max_segments, T, device=h_frames_v.device, dtype=torch.bool)

        # Prepare indices for scatter
        batch_indices = torch.arange(B, device=h_frames_v.device).unsqueeze(1).expand(B, T)  # (B, T)
        frame_indices = torch.arange(T, device=h_frames_v.device).unsqueeze(0).expand(B, T)  # (B, T)

        alignment_matrix[batch_indices, frame_to_segment_map, frame_indices] = True

        return alignment_matrix.float(), sim, num_segments_per_item

    def aggregate_features(
        self,
        features: torch.Tensor,
        alignment_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """
        Aggregate features using alignment matrix.
        
        Args:
            features: torch.Tensor, shape (batch_size, feat_len, feature_dim) or (batch_size, feature_dim, feat_len)
            alignment_matrix: torch.Tensor, shape (batch_size, num_groups, feat_len)
                
        Returns:
            torch.Tensor, shape (batch_size, num_groups, feature_dim) or (batch_size, feature_dim, num_groups)
        """
        # Handle both (B, T, D) and (B, D, T) formats
        is_channel_last = features.dim() == 3 and features.shape[-1] != alignment_matrix.shape[-1]
        
        if not is_channel_last:
            # Features are (B, D, T), need to transpose for aggregation
            features = features.transpose(1, 2)  # (B, T, D)
        
        # Ensure alignment matrix is float and on the correct device
        alignment_float = alignment_matrix.to(features.device, dtype=features.dtype)
        
        # Calculate the sum of features for each group via vectorized operation
        summed_features = torch.einsum('bgt,btd->bgd', alignment_float, features)
        
        # Calculate the number of frames assigned to each group
        group_frame_counts = alignment_float.sum(dim=2)  # (batch_size, num_groups)
        
        # To avoid division by zero, clamp counts to a minimum of 1
        group_frame_counts = group_frame_counts.clamp(min=1)
        
        # Reshape counts for broadcasting
        counts_reshaped = group_frame_counts.unsqueeze(-1)  # (batch_size, num_groups, 1)
        
        # Compute the average
        aggregated_features = summed_features / counts_reshaped
        
        if not is_channel_last:
            # Transpose back to (B, D, T) format
            aggregated_features = aggregated_features.transpose(1, 2)
        
        return aggregated_features

    def compute_repa_loss(self, hidden_features_downsampled, target_repr, mask=None):
        """
        Compute REPA (Representation Alignment) loss between VoiceBox hidden features and SenseVoice representations.
        
        Args:
            hidden_features_downsampled: torch.Tensor, shape (B, D, T) - Hidden features from VoiceBox (3x downsampled)
            target_repr: torch.Tensor, shape (B, D, T) - Target SenseVoice representation
            mask: torch.Tensor, shape (B, T) - Optional mask for valid tokens (original length)
            
        Returns:
            torch.Tensor: REPA loss value
        """
        # Ensure both tensors have the same shape
        if hidden_features_downsampled.shape[-1] != target_repr.shape[-1]:
            # Assert shape difference is at most 2
            assert abs(hidden_features_downsampled.shape[-1] - target_repr.shape[-1]) <= 2
            min_len = min(hidden_features_downsampled.shape[-1], target_repr.shape[-1])
            hidden_features_downsampled = hidden_features_downsampled[..., :min_len]
            target_repr = target_repr[..., :min_len]
        
        # Compute REPA loss based on the specified type
        if self.repa_loss_type == "cosine":
            # Cosine similarity loss (maximize similarity)
            # Normalize along the feature dimension (dim=1 for B, D, T format)
            hidden_norm = F.normalize(hidden_features_downsampled, dim=1)
            target_norm = F.normalize(target_repr, dim=1)
            cosine_sim = F.cosine_similarity(hidden_norm, target_norm, dim=1)  # (B, T)
            
            if mask is not None:
                # Downsample mask to match the downsampled features
                if mask.shape[-1] != hidden_features_downsampled.shape[-1]:
                    mask_downsampled = F.interpolate(
                        mask.unsqueeze(1).float(), 
                        size=hidden_features_downsampled.shape[-1], 
                        mode='linear', 
                        align_corners=False
                    ).squeeze(1).bool()
                else:
                    mask_downsampled = mask
                
                repa_loss = (1 - cosine_sim) * mask_downsampled.float()
                repa_loss = repa_loss.sum() / mask_downsampled.sum().clamp(min=1)
            else:
                repa_loss = (1 - cosine_sim).mean()
        elif self.repa_loss_type == "l2":
            # L2 distance loss
            repa_loss = F.mse_loss(hidden_features_downsampled, target_repr)
        elif self.repa_loss_type == "l1":
            # L1 distance loss
            repa_loss = F.l1_loss(hidden_features_downsampled, target_repr)
        else:
            raise ValueError(f"Unknown REPA loss type: {self.repa_loss_type}")
        
        return repa_loss * self.repa_loss_weight

    def deaggregate_features(
        self,
        grouped_features: torch.Tensor,
        alignment_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """
        De-aggregate features back to original frame structure.
        
        Args:
            grouped_features: torch.Tensor, shape (batch_size, num_groups, feature_dim) or (batch_size, feature_dim, num_groups)
            alignment_matrix: torch.Tensor, shape (batch_size, num_groups, feat_len)
                
        Returns:
            torch.Tensor, shape (batch_size, feat_len, feature_dim) or (batch_size, feature_dim, feat_len)
        """
        # Handle both (B, G, D) and (B, D, G) formats
        is_channel_last = grouped_features.dim() == 3 and grouped_features.shape[1] == alignment_matrix.shape[1]
        
        if not is_channel_last:
            # Features are (B, D, G), need to transpose for de-aggregation
            grouped_features = grouped_features.transpose(1, 2)  # (B, G, D)
        
        # Ensure alignment matrix is float and on the correct device
        alignment_float = alignment_matrix.to(grouped_features.device, dtype=grouped_features.dtype)
        
        # De-aggregate: expand group features to frame features
        expanded_features = torch.einsum('bgd,bgt->btd', grouped_features, alignment_float)
        
        if not is_channel_last:
            # Transpose back to (B, D, T) format
            expanded_features = expanded_features.transpose(1, 2)
        
        return expanded_features

    def _deaggregate_features_from_token_lengths(
        self,
        grouped_features: torch.Tensor,
        token_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        De-aggregate features back to original frame structure using token_lengths.
        This is a replacement for `deaggregate_features` when only token lengths are available.
        
        Args:
            grouped_features: torch.Tensor, shape (batch_size, feature_dim, num_groups)
            token_lengths: torch.Tensor, shape (batch_size, num_groups)
                
        Returns:
            torch.Tensor, shape (batch_size, feature_dim, feat_len)
        """
        B, D, G = grouped_features.shape
        assert G == token_lengths.shape[1], "Number of groups in features and token_lengths must match."
        
        # Permute features to be (B, G, D) for repeat_interleave
        grouped_features_permuted = grouped_features.permute(0, 2, 1)
        
        expanded_features_list = []
        for i in range(B):
            # For each item in the batch, repeat its features according to token_lengths
            # token_lengths contains the number of repetitions for each feature vector in the group
            expanded_item = torch.repeat_interleave(grouped_features_permuted[i], token_lengths[i], dim=0)
            expanded_features_list.append(expanded_item)
        
        # Pad the list of tensors to the same length and stack them
        expanded_features = pad_sequence(expanded_features_list, batch_first=True, padding_value=0.0)
        
        # Transpose back to (B, D, T) format
        expanded_features = expanded_features.transpose(1, 2)
        
        return expanded_features

    def print_submodule_params(self):
        """Print the number of parameters in each submodule of the DualCodec model."""
        print("\n" + "="*80)
        print("DUALCODEC SUBMODULE PARAMETER ANALYSIS")
        print("="*80)
        
        # Define submodules to analyze
        submodules = {
            'semantic_model': self.semantic_model,
            'convnext_encoder': self.convnext_encoder,
            'semantic_vq': self.semantic_vq,
            'convnext_decoder': self.convnext_decoder,
            'dac': self.dac,
        }
        
        # Add conditional submodules
        if hasattr(self, 'bottleneck_transformer') and self.bottleneck_transformer is not None:
            submodules['bottleneck_transformer'] = self.bottleneck_transformer
        if hasattr(self, 'bottleneck_transformer_2') and self.bottleneck_transformer_2 is not None:
            submodules['bottleneck_transformer_2'] = self.bottleneck_transformer_2
        if hasattr(self, 'repa_mlp') and self.repa_mlp is not None:
            submodules['repa_mlp'] = self.repa_mlp
        if hasattr(self, 'flow_matching_decoder') and self.flow_matching_decoder is not None:
            submodules['flow_matching_decoder'] = self.flow_matching_decoder
        if hasattr(self, 'repa_projection') and self.repa_projection is not None:
            submodules['repa_projection'] = self.repa_projection
        if hasattr(self, 'semantic_aggregator') and self.semantic_aggregator is not None:
            submodules['semantic_aggregator'] = self.semantic_aggregator
        if hasattr(self, 'acoustic_aggregator') and self.acoustic_aggregator is not None:
            submodules['acoustic_aggregator'] = self.acoustic_aggregator
        
        # Analyze each submodule
        total_params = 0
        trainable_params = 0
        
        for name, module in submodules.items():
            if module is None:
                continue
                
            module_params = sum(p.numel() for p in module.parameters())
            module_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            
            print(f"{name:25s}: {module_params:>10,} params ({module_params/1e6:>6.2f}M) | "
                  f"Trainable: {module_trainable:>10,} ({module_trainable/1e6:>6.2f}M)")
            
            total_params += module_params
            trainable_params += module_trainable
        
        # Print DAC submodules if available
        if hasattr(self.dac, 'encoder') and self.dac.encoder is not None:
            dac_encoder_params = sum(p.numel() for p in self.dac.encoder.parameters())
            dac_encoder_trainable = sum(p.numel() for p in self.dac.encoder.parameters() if p.requires_grad)
            print(f"{'dac.encoder':25s}: {dac_encoder_params:>10,} params ({dac_encoder_params/1e6:>6.2f}M) | "
                  f"Trainable: {dac_encoder_trainable:>10,} ({dac_encoder_trainable/1e6:>6.2f}M)")
            total_params += dac_encoder_params
            trainable_params += dac_encoder_trainable
            
        if hasattr(self.dac, 'decoder') and self.dac.decoder is not None:
            dac_decoder_params = sum(p.numel() for p in self.dac.decoder.parameters())
            dac_decoder_trainable = sum(p.numel() for p in self.dac.decoder.parameters() if p.requires_grad)
            print(f"{'dac.decoder':25s}: {dac_decoder_params:>10,} params ({dac_decoder_params/1e6:>6.2f}M) | "
                  f"Trainable: {dac_decoder_trainable:>10,} ({dac_decoder_trainable/1e6:>6.2f}M)")
            total_params += dac_decoder_params
            trainable_params += dac_decoder_trainable
            
        if hasattr(self.dac, 'quantizer') and self.dac.quantizer is not None:
            dac_quantizer_params = sum(p.numel() for p in self.dac.quantizer.parameters())
            dac_quantizer_trainable = sum(p.numel() for p in self.dac.quantizer.parameters() if p.requires_grad)
            print(f"{'dac.quantizer':25s}: {dac_quantizer_params:>10,} params ({dac_quantizer_params/1e6:>6.2f}M) | "
                  f"Trainable: {dac_quantizer_trainable:>10,} ({dac_quantizer_trainable/1e6:>6.2f}M)")
            total_params += dac_quantizer_params
            trainable_params += dac_quantizer_trainable
        
        print("-" * 80)
        print(f"{'TOTAL':25s}: {total_params:>10,} params ({total_params/1e6:>6.2f}M) | "
              f"Trainable: {trainable_params:>10,} ({trainable_params/1e6:>6.2f}M)")
        
        # Check for any remaining parameters not accounted for
        all_params = sum(p.numel() for p in self.parameters())
        all_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        if all_params != total_params:
            remaining_params = all_params - total_params
            remaining_trainable = all_trainable - trainable_params
            print(f"{'REMAINING':25s}: {remaining_params:>10,} params ({remaining_params/1e6:>6.2f}M) | "
                  f"Trainable: {remaining_trainable:>10,} ({remaining_trainable/1e6:>6.2f}M)")
        
        print("="*80 + "\n")

def test_dual_codec_comparison():
    """Compare DualCodec with and without similarity alignment"""
    import torchaudio
    import librosa
    from transformers import SeamlessM4TFeatureExtractor
    import time

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sr = 16000
    
    # Test audio
    audio_fpath = "/data1/lijiaqi/codebase/CSD/out.wav"

    # Load one audio file and create two versions with different lengths
    audio_16k, _ = librosa.load(audio_fpath, sr=sr, mono=True)

    # Full-length sample
    audio_full = torch.from_numpy(audio_16k).float()

    # Shorter sample (e.g., 60 % of the original)
    shorter_len = int(len(audio_16k) * 0.6)
    audio_short = torch.from_numpy(audio_16k[:shorter_len]).float()

    # Resample to 24 kHz for the codec branch
    resampler_24k = torchaudio.transforms.Resample(sr, 24000)
    audio_full_24k = resampler_24k(audio_full.unsqueeze(0))        # (1, T1_24k)
    audio_short_24k = resampler_24k(audio_short.unsqueeze(0))      # (1, T2_24k)

    # Stack and pad to the max length among the two for batching
    max_24k_len = max(audio_full_24k.shape[1], audio_short_24k.shape[1])
    pad_24k_full  = F.pad(audio_full_24k,  (0, max_24k_len - audio_full_24k.shape[1]))
    pad_24k_short = F.pad(audio_short_24k, (0, max_24k_len - audio_short_24k.shape[1]))
    audio_24k_batch = torch.cat([pad_24k_full, pad_24k_short], dim=0)  # (B=2, T_max_24k)

    # Extract features (16 kHz) separately for the two versions
    feature_extractor = SeamlessM4TFeatureExtractor.from_pretrained(
        "/mnt//models/projects/lijiaqi_csd/w2v-bert-2.0"
    )
    features_full  = feature_extractor(audio_full.unsqueeze(0),  return_tensors="pt", sampling_rate=sr).input_features
    features_short = feature_extractor(audio_short.unsqueeze(0), return_tensors="pt", sampling_rate=sr).input_features

    # Pad to common length for batching
    max_feat_len = max(features_full.shape[1], features_short.shape[1])
    pad_feat_full  = F.pad(features_full,  (0,0,0, max_feat_len - features_full.shape[1]))
    pad_feat_short = F.pad(features_short, (0,0,0, max_feat_len - features_short.shape[1]))
    audio_features = torch.cat([pad_feat_full, pad_feat_short], dim=0).to(device)  # (B, T_max, D)

    # Length tensors
    audio_lens_24k = torch.tensor([audio_full_24k.shape[1],  audio_short_24k.shape[1]],  device=device)
    audio_feat_lens = torch.tensor([features_full.shape[1],  features_short.shape[1]], device=device)

    dl_output = {
        "audio": audio_24k_batch.to(device),          # (B, T_max_24k)
        "audio_lens": audio_lens_24k,
        "x": audio_features,
        "x_lens": audio_feat_lens,
    }
    
    print("Testing DualCodec with Similarity Alignment Only:")
    print("=" * 70)

    # Instantiate and evaluate the aggregated DualCodec
    model_with_align = DualCodec(
        sample_rate=24000,
        encoder_rates=[4,5,6,8],
        decoder_rates=[8,6,5,4],
        n_codebooks=11,
        codebook_size=1024,
        semantic_codebook_size=16384,
        is_causal=True,
        use_similarity_alignment=True,  # Enable similarity-based alignment
        similarity_threshold=0.75,      # Threshold for grouping similar frames
        skip_normalize=True,
    ).to(device)

    model_with_align.eval()
    
    start_time = time.time()
    with torch.no_grad():
        result_with_align = model_with_align(dl_output)
    time_with_align = time.time() - start_time
    
    print(f"\nWith Similarity Alignment:")
    print(f"  Processing Time: {time_with_align:.4f}s")
    print(f"  Reconstructed Shape: {result_with_align['audio'].shape}")
    
    if "alignment_matrices" in result_with_align:
        alignment_matrices = result_with_align["alignment_matrices"]
        original_frames = alignment_matrices.shape[2]
        compressed_groups = alignment_matrices.shape[1]
        compression_ratio = original_frames / compressed_groups
        print(f"  Compression Ratio: {compression_ratio:.2f}x")
        print(f"  Original Frames: {original_frames}")
        print(f"  Compressed Groups: {compressed_groups}")
    
    # Save output audio for inspection
    torchaudio.save("dualcodec_with_alignment.wav", result_with_align['audio'][0].cpu(), 24000)
    print("\nSaved output audio to dualcodec_with_alignment.wav")


def test_dual_codec_with_sensevoice():
    """Test DualCodec with FunASR model instead of Wav2Vec2BertModel"""
    import torchaudio
    import librosa
    import time

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sr = 16000
    
    # Test audio
    audio_fpath = "/data1/lijiaqi/codebase/CSD/out.wav"

    # Load audio file
    audio_16k, _ = librosa.load(audio_fpath, sr=sr, mono=True)
    audio_full = torch.from_numpy(audio_16k).float()

    def extract_fbank(frontend, data, data_len=None, data_type: str = "sound", **kwargs):
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

        data, data_len = frontend(data, data_len, **kwargs)


        if isinstance(data_len, (list, tuple)):
            data_len = torch.tensor([data_len])
        return data.to(torch.float32), data_len.to(torch.int32)
    # override feature
    import torchaudio
    audio_data, _ = torchaudio.load('/data1/lijiaqi/codebase/CSD/out.wav')
    # another file: /data1/lijiaqi/codebase/CSD/out1.wav

    # override frontend
    import funasr
    cmvn_file = '/data1/lijiaqi/codebase/CSD/zero-shot-tts-training/zero_shot_tts_training/realtime_communication/taste_v2/customized_sensevoice/am.mvn'
    frontend = funasr.frontends.wav_frontend.WavFrontend(
                cmvn_file=cmvn_file,
                n_mels=80,
                frame_length=25,
                frame_shift=10,
                lfr_m=7,
                lfr_n=6,
            )


    audio_features, audio_features_lengths = extract_fbank(frontend, audio_data.to(device), None)
    audio_features = audio_features.to(device)
    audio_features_lengths = audio_features_lengths.to(device)

    dl_output = {
        "audio": audio_data.to(device),          # (B, T_24k)
        "x": audio_features.to(device),
        "x_lens": audio_features_lengths.to(device),
    }
    
    print("Testing DualCodec with FunASR model:")
    print("=" * 70)

    # Instantiate DualCodec with FunASR
    model_with_sensevoice = DualCodec(
        sample_rate=16000,
        encoder_rates=[4,5,6,8],
        decoder_rates=[8,6,5,4],
        n_codebooks=11,
        codebook_size=1024,
        semantic_codebook_size=32768,
        is_causal=False,
        use_similarity_alignment=True,
        similarity_threshold=0.85,
        skip_normalize=True,
        # FunASR specific parameters (simplified)
        semantic_model_type="sensevoice",
        semantic_model_path="/data1/lijiaqi/codebase/TASTE-SpokenLM/STAGE1_TRAIN/storage/pretrained_models/SenseVoiceSmall",
        sensevoice_prepend_inputs=True,
        latent_dim=512,
        ssl_dim=512,
    ).to(device)

    start_time = time.time()
    with torch.no_grad():
        result_with_sensevoice = model_with_sensevoice(dl_output)
    time_with_sensevoice = time.time() - start_time
    
    print(f"\nWith FunASR model:")
    print(f"  Processing Time: {time_with_sensevoice:.4f}s")
    print(f"  Reconstructed Shape: {result_with_sensevoice['audio'].shape}")
    print(f"  Semantic Model Type: {model_with_sensevoice.semantic_model_type}")
    
    if "token_ratio" in result_with_sensevoice:
        print(f"  Token Ratio: {result_with_sensevoice['token_ratio']:.4f}")
    
    # Save output audio for inspection
    torchaudio.save("dualcodec_with_sensevoice.wav", result_with_sensevoice['audio'][0].cpu(), 24000)
    print("\nSaved output audio to dualcodec_with_sensevoice.wav")


def test_dual_codec_with_dcae():
    """Test DualCodec with DC-AE (Deep Compression Autoencoder) functionality"""
    import torchaudio
    import librosa
    import time

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sr = 16000
    
    # Test audio
    audio_fpath = "/data1/lijiaqi/codebase/CSD/out.wav"

    # Load audio file
    audio_16k, _ = librosa.load(audio_fpath, sr=sr, mono=True)
    audio_full = torch.from_numpy(audio_16k).float()

    # Resample to 24 kHz for the codec branch
    resampler_24k = torchaudio.transforms.Resample(sr, 24000)
    audio_24k = resampler_24k(audio_full.unsqueeze(0))  # (1, T_24k)

    # Extract features (16 kHz)
    feature_extractor = SeamlessM4TFeatureExtractor.from_pretrained(
        "/mnt//models/projects/lijiaqi_csd/w2v-bert-2.0"
    )
    features = feature_extractor(audio_full.unsqueeze(0), return_tensors="pt", sampling_rate=sr).input_features

    dl_output = {
        "audio": audio_24k.to(device),          # (B, T_24k)
        "x": features.to(device),
    }
    
    print("Testing DualCodec with DC-AE (Deep Compression Autoencoder):")
    print("=" * 70)

    # Test 1: Regular DAC model (without DC-AE)
    print("\n1. Testing Regular DAC model (residual_autoencode=False):")
    model_regular = DualCodec(
        sample_rate=24000,
        encoder_rates=[4,5,6,8],
        decoder_rates=[8,6,5,4],
        n_codebooks=11,
        codebook_size=1024,
        semantic_codebook_size=16384,
        is_causal=False,
        use_similarity_alignment=False,
        skip_normalize=True,
        semantic_model_type="w2vbert",
        # DAC-specific DC-AE parameter
        residual_autoencode=False,
    ).to(device)

    model_regular.eval()
    
    start_time = time.time()
    with torch.no_grad():
        result_regular = model_regular(dl_output)
    time_regular = time.time() - start_time
    
    print(f"  Processing Time: {time_regular:.4f}s")
    print(f"  Reconstructed Shape: {result_regular['audio'].shape}")
    print(f"  VQ Commitment Loss: {result_regular['vq/commitment_loss']:.6f}")
    print(f"  VQ Codebook Loss: {result_regular['vq/codebook_loss']:.6f}")
    
    # Test 2: DAC model with DC-AE enabled
    print("\n2. Testing DAC model with DC-AE (residual_autoencode=True):")
    model_dcae = DualCodec(
        sample_rate=24000,
        encoder_rates=[4,5,6,8],
        decoder_rates=[8,6,5,4],
        n_codebooks=11,
        codebook_size=1024,
        semantic_codebook_size=16384,
        is_causal=False,
        use_similarity_alignment=False,
        skip_normalize=True,
        semantic_model_type="w2vbert",
        # DAC-specific DC-AE parameter
        residual_autoencode=True,
    ).to(device)

    model_dcae.eval()
    
    start_time = time.time()
    with torch.no_grad():
        result_dcae = model_dcae(dl_output)
    time_dcae = time.time() - start_time
    
    print(f"  Processing Time: {time_dcae:.4f}s")
    print(f"  Reconstructed Shape: {result_dcae['audio'].shape}")
    print(f"  VQ Commitment Loss: {result_dcae['vq/commitment_loss']:.6f}")
    print(f"  VQ Codebook Loss: {result_dcae['vq/codebook_loss']:.6f}")
    
    # Compare results
    print("\n3. Comparison:")
    print(f"  Time Difference: {abs(time_dcae - time_regular):.4f}s")
    print(f"  Commitment Loss Difference: {abs(result_dcae['vq/commitment_loss'] - result_regular['vq/commitment_loss']):.6f}")
    print(f"  Codebook Loss Difference: {abs(result_dcae['vq/codebook_loss'] - result_regular['vq/codebook_loss']):.6f}")
    
    # Check if outputs are different (DC-AE should produce different results)
    audio_diff = torch.abs(result_dcae['audio'] - result_regular['audio']).mean().item()
    print(f"  Audio Output Difference: {audio_diff:.6f}")
    
    # Save output audio for inspection
    torchaudio.save("dualcodec_regular.wav", result_regular['audio'][0].cpu(), 24000)
    torchaudio.save("dualcodec_dcae.wav", result_dcae['audio'][0].cpu(), 24000)
    print("\nSaved output audio to dualcodec_regular.wav and dualcodec_dcae.wav")
    
    # Test 3: DC-AE with SenseVoice
    print("\n4. Testing DC-AE with SenseVoice model:")
    
    # Prepare SenseVoice features
    import funasr
    cmvn_file = '/data1/lijiaqi/codebase/CSD/zero-shot-tts-training/zero_shot_tts_training/realtime_communication/taste_v2/customized_sensevoice/am.mvn'
    frontend = funasr.frontends.wav_frontend.WavFrontend(
                cmvn_file=cmvn_file,
                n_mels=80,
                frame_length=25,
                frame_shift=10,
                lfr_m=7,
                lfr_n=6,
            )

    def extract_fbank(frontend, data, data_len=None, data_type: str = "sound", **kwargs):
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
            if len(data.shape) < 2:
                data = data[None, :]
            data_len = [data.shape[1]] if data_len is None else data_len
        elif isinstance(data, torch.Tensor):
            if len(data.shape) < 2:
                data = data[None, :]
            data_len = [data.shape[1]] if data_len is None else data_len
        elif isinstance(data, (list, tuple)):
            data_list, data_len = [], []
            for data_i in data:
                if isinstance(data_i, np.ndarray):
                    data_i = torch.from_numpy(data_i)
                data_list.append(data_i)
                data_len.append(data_i.shape[0])

        data, data_len = frontend(data, data_len, **kwargs)

        if isinstance(data_len, (list, tuple)):
            data_len = torch.tensor([data_len])
        return data.to(torch.float32), data_len.to(torch.int32)

    audio_data, _ = torchaudio.load('/data1/lijiaqi/codebase/CSD/out.wav')
    audio_features, audio_features_lengths = extract_fbank(frontend, audio_data.to(device), None)
    
    dl_output_sensevoice = {
        "audio": audio_data.to(device),
        "x": audio_features.to(device),
        "x_lens": audio_features_lengths.to(device),
    }
    
    model_dcae_sensevoice = DualCodec(
        sample_rate=16000,
        encoder_rates=[4,5,6,8],
        decoder_rates=[8,6,5,4],
        n_codebooks=11,
        codebook_size=1024,
        semantic_codebook_size=32768,
        is_causal=False,
        use_similarity_alignment=True,
        similarity_threshold=0.85,
        skip_normalize=True,
        semantic_model_type="sensevoice",
        sensevoice_prepend_inputs=True,
        latent_dim=512,
        ssl_dim=512,
        # DAC-specific DC-AE parameter
        residual_autoencode=True,
    ).to(device)

    model_dcae_sensevoice.eval()
    
    start_time = time.time()
    with torch.no_grad():
        result_dcae_sensevoice = model_dcae_sensevoice(dl_output_sensevoice)
    time_dcae_sensevoice = time.time() - start_time
    
    print(f"  Processing Time: {time_dcae_sensevoice:.4f}s")
    print(f"  Reconstructed Shape: {result_dcae_sensevoice['audio'].shape}")
    print(f"  VQ Commitment Loss: {result_dcae_sensevoice['vq/commitment_loss']:.6f}")
    print(f"  VQ Codebook Loss: {result_dcae_sensevoice['vq/codebook_loss']:.6f}")
    
    if "token_ratio" in result_dcae_sensevoice:
        print(f"  Token Ratio: {result_dcae_sensevoice['token_ratio']:.4f}")
    
    # Save output audio for inspection
    torchaudio.save("dualcodec_dcae_sensevoice.wav", result_dcae_sensevoice['audio'][0].cpu(), 24000)
    print("\nSaved output audio to dualcodec_dcae_sensevoice.wav")


def test_dual_codec_with_repa_loss():
    """Test DualCodec with REPA loss using hidden features from VoiceBox"""
    import torchaudio
    import librosa
    import time

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sr = 16000
    
    # Test audio
    audio_fpath = "/data1/lijiaqi/codebase/CSD/out.wav"

    # Load audio file
    audio_16k, _ = librosa.load(audio_fpath, sr=sr, mono=True)
    audio_full = torch.from_numpy(audio_16k).float()

    # Resample to 24 kHz for the codec
    audio_24k = torchaudio.functional.resample(audio_full, sr, 24000)
    audio_24k = audio_24k.unsqueeze(0)  # Add batch dimension

    print(f"Audio shape: {audio_24k.shape}")

    # Create DualCodec with REPA loss enabled
    model = DualCodec(
        use_flow_matching_decoder=True,
        use_repa_loss=True,
        repa_loss_weight=1.0,
        repa_loss_type="cosine",
        repa_layer_idx=9,  # Extract features from 10th layer (0-indexed)
        repa_projection_dim=1024,  # Output dimension for REPA projection MLP
        flow_matching_context=150,
        flow_matching_causal=False,
    ).to(device)

    print(f"Model created with REPA loss enabled")
    print(f"REPA loss weight: {model.repa_loss_weight}")
    print(f"REPA loss type: {model.repa_loss_type}")
    print(f"REPA layer index: {model.repa_layer_idx} (extracting from layer {model.repa_layer_idx + 1})")
    print(f"REPA projection dimension: {model.repa_projection_dim}")
    print(f"REPA features are 3x downsampled in DualCodec to match SenseVoice frame rate")

    # Forward pass
    with torch.no_grad():
        start_time = time.time()
        output = model(audio_24k.to(device))
        end_time = time.time()

    print(f"Forward pass completed in {end_time - start_time:.4f} seconds")
    print(f"Output keys: {output.keys()}")
    
    # Check if REPA loss is computed
    if "flow_matching_repa_loss" in output:
        repa_loss_value = output["flow_matching_repa_loss"]
        print(f"REPA loss value: {repa_loss_value.item():.6f}")
    else:
        print("REPA loss not found in output")

    # Check other outputs
    if "audio" in output:
        print(f"Output audio shape: {output['audio'].shape}")
    if "flow_matching_loss" in output:
        print(f"Flow matching loss: {output['flow_matching_loss'].item():.6f}")

    print("REPA loss test completed successfully!")


def test_dual_codec_with_x_lens():
    """Test DualCodec with x_lens parameter to handle padded inputs"""
    import torchaudio
    import librosa
    import time

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sr = 16000
    
    # Create dummy audio data with different lengths in the batch
    B = 2
    T_audio = 48000  # 3 seconds at 16kHz
    T_features = 300  # Corresponding feature length
    
    # Create audio data with different actual lengths
    audio_lens = torch.tensor([T_audio, T_audio // 2], device=device)  # Second item is half length
    audio_data = torch.randn(B, T_audio, device=device)
    
    # Zero out the padding for the second item
    audio_data[1, audio_lens[1]:] = 0
    
    # Create feature data with padding
    x_lens = torch.tensor([T_features, T_features // 2], device=device)  # Second item is half length  
    audio_features = torch.randn(B, T_features, 80, device=device)
    
    # Zero out the padding for the second item in features
    audio_features[1, x_lens[1]:] = 0
    
    dl_output = {
        "audio": audio_data.unsqueeze(1),  # Add channel dimension
        "x": audio_features,
        "x_lens": x_lens,  # NEW: Input lengths indicating valid frames
    }
    
    print("Testing DualCodec with x_lens parameter:")
    print("=" * 60)
    print(f"Batch size: {B}")
    print(f"Audio lengths: {audio_lens.tolist()}")
    print(f"Feature lengths (x_lens): {x_lens.tolist()}")

    # Test with similarity alignment and semantic downsampling
    semantic_downsample_factor = 2  # Test with downsampling
    model = DualCodec(
        sample_rate=16000,
        encoder_rates=[4,5,6,8],
        decoder_rates=[8,6,5,4],
        n_codebooks=11,
        codebook_size=1024,
        semantic_codebook_size=16384,
        is_causal=False,
        use_similarity_alignment=True,  # Enable similarity-based alignment
        similarity_threshold=0.75,
        skip_normalize=True,
        semantic_model_type="w2vbert",
        use_query_token_aggregator=True,
        latent_dim=512,
        semantic_downsample_factor=semantic_downsample_factor,  # Test downsampling
    ).to(device)
    
    print(f"Semantic downsample factor: {semantic_downsample_factor}")
    print(f"Expected downsampled x_lens: {[x // semantic_downsample_factor for x in x_lens.tolist()]}")

    model.eval()
    
    start_time = time.time()
    with torch.no_grad():
        result = model(dl_output)
    time_taken = time.time() - start_time
    
    print(f"\nResults:")
    print(f"  Processing Time: {time_taken:.4f}s")
    print(f"  Output audio shape: {result['audio'].shape}")
    
    # Check speech_token_len
    if "speech_token_len" in result and result["speech_token_len"] is not None:
        speech_token_len = result["speech_token_len"]
        print(f"  Speech token lengths: {speech_token_len.tolist()}")
        print(f"  Token compression ratio: {result.get('token_ratio', 'N/A'):.4f}")
    else:
        print("  Speech token lengths: None (no aggregation)")
    
    # Test encode_only mode
    print(f"\nTesting encode_only mode:")
    with torch.no_grad():
        encoded_result = model(dl_output, encode_only=True)
    
    if "speech_token_len" in encoded_result:
        print(f"  Encoded speech token lengths: {encoded_result['speech_token_len'].tolist()}")
        print(f"  Semantic codes shape: {encoded_result['semantic_codes'].shape}")
        if encoded_result['acoustic_codes'] is not None:
            print(f"  Acoustic codes shape: {encoded_result['acoustic_codes'].shape}")
    
    print("\nTest completed successfully!")


def test_x_lens_downsampling():
    """Test the x_lens downsampling functionality specifically"""
    print("Testing x_lens downsampling:")
    print("=" * 50)
    
    device = torch.device("cpu")  # Use CPU for simple testing
    
    # Test with integer downsample factor
    model = DualCodec(
        sample_rate=16000,
        encoder_rates=[4],
        decoder_rates=[4],
        n_codebooks=4,
        codebook_size=1024,
        semantic_codebook_size=1024,
        semantic_downsample_factor=2,  # Integer factor
        latent_dim=256,
    ).to(device)
    
    original_x_lens = torch.tensor([100, 75, 50])
    downsampled_x_lens = model._downsample_x_lens(original_x_lens)
    print(f"Integer factor (2): {original_x_lens.tolist()} -> {downsampled_x_lens.tolist()}")
    
    # Test with fractional downsample factor
    model.semantic_downsample_factor = 1.5
    downsampled_x_lens = model._downsample_x_lens(original_x_lens)
    print(f"Fractional factor (1.5): {original_x_lens.tolist()} -> {downsampled_x_lens.tolist()}")
    
    # Test with no downsampling
    model.semantic_downsample_factor = 1
    downsampled_x_lens = model._downsample_x_lens(original_x_lens)
    print(f"No downsampling (1): {original_x_lens.tolist()} -> {downsampled_x_lens.tolist()}")
    
    # Test with None input
    downsampled_x_lens = model._downsample_x_lens(None)
    print(f"None input: None -> {downsampled_x_lens}")
    
    print("✓ x_lens downsampling test completed!")


if __name__ == '__main__':
    # test_dual_codec_model()
    # print("\n" + "="*70 + "\n")
    # test_dual_codec_comparison()
    # print("\n" + "="*70 + "\n")
    # test_dual_codec_with_sensevoice()
    # print("\n" + "="*70 + "\n")
    # test_dual_codec_with_dcae()
    # test_dual_codec_with_repa_loss()
    test_x_lens_downsampling()
    print("\n" + "="*70 + "\n")
    test_dual_codec_with_x_lens()