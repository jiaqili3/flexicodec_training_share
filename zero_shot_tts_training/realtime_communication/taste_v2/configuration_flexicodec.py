# 1. Import necessary libraries
from transformers import PretrainedConfig
from typing import List, Union, Optional

# 2. Define the configuration class
class FlexiCodecConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a FlexiCodec model.
    It is used to instantiate the model according to the specified arguments, defining the model architecture.
    """
    model_type = "flexicodec"  # A unique identifier for your model type

    def __init__(
        self,
        # ------------DAC Encoder config---------------
        encoder_dim: int = 64,
        encoder_rates: List[int] = [2, 4, 8, 8],
        latent_dim: int = None,
        decoder_dim: int = 1536,
        decoder_rates: List[int] = [8, 8, 4, 2],
        # ------------DAC Quantizer config---------------
        n_codebooks: int = 9,
        codebook_size: int = 1024,
        semantic_codebook_size: int = 16384,
        codebook_dim: Union[int, list] = 8,
        semantic_codebook_dim=8,
        use_bsq_for_semantic_vq: bool = False,
        bsq_config: dict = None,
        use_fsq_for_semantic_vq: bool = True,
        fsq_config: dict = None,
        quantizer_dropout: bool = True,
        sample_rate: int = 24000,
        bypass_quantize_rate: float = 0.125,
        # ------------Semantic Model config---------------
        convnext_dim=768,
        convnext_layers=4,
        decode_semantic_for_codec=True,
        is_causal=True,
        semantic_downsample_factor=2,
        use_concat_downsampling=False,
        use_conv_downsampling=False,
        override_dac_encoder=None,
        override_vocos_decoder=None,
        # Optional: use MiMo encoder instead of DAC encoder
        use_mimo_codec_encoder: bool = False,
        mimo_config: dict | None = None,
        ssl_dim=1024,
        lambda_distill_loss=15.0,
        # ------------Similarity-based merging config---------------
        use_similarity_alignment: bool = False,
        similarity_threshold=None,
        use_dynamic_similarity_threshold: bool = False,
        similarity_threshold_lower: float = 0.8,
        similarity_threshold_upper: float = 1.0,
        flex_framerate: bool = False,
        flex_framerate_options: list = [0.86, 0.90, 1.0],
        skip_normalize=True,
        half_semantic_model=False,
        max_tokens_per_group: Optional[int] = 6,
        semantic_model_type: str = "sensevoice",
        semantic_model_path="/mnt/lijiaqi//SenseVoiceSmall",
        cmvn_path: str = "/mnt/lijiaqi/zero-shot-tts-training/zero_shot_tts_training/realtime_communication/taste_v2/customized_sensevoice/am.mvn",
        sensevoice_prepend_inputs: bool = True,
        whisper_layer_idx: int = -1,
        # ------------Transformer config---------------
        use_bottleneck_transformer: bool = False,
        bottleneck_transformer_config: dict = None,
        transformer_num_layers: int = 6,
        transformer_dim: int = 512,
        transformer_dim_feedforward: int = 2048,
        transformer_num_heads: int = 8,
        transformer_causal: bool = False,
        transformer_context_frames: int = 24,
        # ------------Second decoder Transformer config (deprecated)---------------
        use_second_decoder_transformer: bool = False,
        transformer_2_num_layers: int = None,
        # ------------Aggregator Transformer config---------------
        use_query_token_aggregator: bool = False,
        agg_transformer_num_layers: int = 6,
        agg_transformer_dim: int = 512,
        agg_transformer_num_heads: int = 8,
        agg_transformer_dim_feedforward: int = 2048,
        agg_transformer_causal: bool = False,
        agg_use_mean_pooling_init: bool = True,
        agg_add_query_embedding: bool = False,
        agg_transformer_context_frames: int = None,
        use_fixed_rate_aggregator: bool = False,
        aggregator_downsample_ratio: int = 4,
        aggregator_downsample_ratio_options: list = None,
        sensevoice_sim_layer_idx=None,
        sensevoice_semantic_layer_idx=None,
        # ------------Flow Matching Decoder config---------------
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
        flow_matching_context: int = 150,
        flow_matching_causal: bool = False,
        flow_matching_has_prompt: bool = False,
        # ------------Loss config---------------
        use_repa_loss: bool = True,
        repa_loss_weight: float = 1.0,
        repa_loss_type: str = "cosine",
        repa_layer_idx: int = 9,
        repa_projection_dim: int = 512,
        insert_query_before_downsample: bool = False,
        no_acoustic_aggregator: bool = False,
        sometimes_skip_repa: bool = False,
        distill_with_avg: bool = False,
        add_semantic_spec_loss: bool = True,
        **kwargs
    ):
        
        # A more robust way to assign all parameters:
        for key, value in locals().items():
            if key != 'self' and not key.startswith('__') and key != 'kwargs':
                setattr(self, key, value)

        super().__init__(**kwargs)
if __name__ == "__main__":
    config = FlexiCodecConfig(
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
        semantic_model_path="/mnt/lijiaqi//SenseVoiceSmall",
        sensevoice_prepend_inputs=True,
        latent_dim=512,
        ssl_dim=512,
    )