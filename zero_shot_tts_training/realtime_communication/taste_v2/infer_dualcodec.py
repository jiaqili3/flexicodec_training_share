import sys
import os

# sys.path.append('/data1/lijiaqi/codebase/CSDs')
sys.path.append('/data1/lijiaqi/codebase/CSDs/application/ts3codec/TS3Codec')

import os
IS_CLUSTER = 'IS_CLUSTER' in os.environ and os.environ['IS_CLUSTER'] == '1'
if IS_CLUSTER:
    sys.path.append('/modelblob/users/lijiaqi/codebase/TS3Codec')

import torch
import time
import torchaudio
from transformers import SeamlessM4TFeatureExtractor
from audio_codec.train.feature_extractors import FBankGen, WhisperGen

get_params = lambda model: sum(p.numel() for p in model.parameters()) / 1e6

def prepare_model():
    """Loads the DualCodec model and its corresponding feature extractor."""
    import yaml
    
    # Load model from config and checkpoint
    # model_config_path = '/data1/lijiaqi/codebase/CSDs/application/ts3codec/TS3Codec/configs/codec/amlt/librilight_dualcodec25hz.yaml'
    # ckpt_path = '/mnt/wus2/models/users/lijiaqi/exp_codec/v3.8/librilight_dualcodec25hz/260000.pth'
    # model_config_path = '/data1/lijiaqi/codebase/CSDs/application/ts3codec/TS3Codec/configs/codec/amlt/librilight_dualcodec25hz_sim0_6.yaml'
    # ckpt_path = '/mnt/wus2/models/users/lijiaqi/exp_codec/v3.8/librilight_dualcodec25hz_sim0_6/180000.pth'
    # model_config_path = '/data1/lijiaqi/codebase/CSDs/application/ts3codec/TS3Codec/configs/codec/amlt/librilight_dualcodec25hz_sim0_75_nonorm.yaml'
    # ckpt_path = '/mnt/wus2/models/users/lijiaqi/exp_codec/v3.8/librilight_dualcodec25hz_sim0_75_nonorm/240000.pth'
    # model_config_path = '/data1/lijiaqi/codebase/CSDs/application/ts3codec/TS3Codec/configs/codec/amlt/librilight_dualcodecsensevoice_sim_1_0.yaml'
    # ckpt_path = '/mnt/wus2/models/users/lijiaqi/exp_codec/v3.8/librilight_dualcodecsensevoice_sim_1_0/400000.pth'
    # model_config_path = '/data1/lijiaqi/codebase/CSDs/application/ts3codec/TS3Codec/configs/codec/amlt/librilight_dualcodecsensevoice_sim_0_85.yaml'
    # ckpt_path = '/mnt/wus2/models/users/lijiaqi/exp_codec/v3.8/librilight_dualcodecsensevoice_sim_0_85/40000.pth'
    # model_config_path = '/data1/lijiaqi/codebase/CSDs/application/ts3codec/TS3Codec/configs/codec/amlt/librilight_dualcodecsensevoice_sim_0_9_dcae.yaml'
    # ckpt_path = '/mnt/wus2/models/users/lijiaqi/exp_codec/v3.8/librilight_dualcodecsensevoice_sim_0_9_dcae/60000.pth'
    # model_config_path = '/data1/lijiaqi/codebase/CSDs/application/ts3codec/TS3Codec/configs/codec/amlt/librilight_dualcodecsensevoice_sim_0_9_transformer.yaml'
    # ckpt_path = '/mnt/wus2/models/users/lijiaqi/exp_codec/v3.8/librilight_dualcodecsensevoice_sim_0_9_transformer/60000.pth'
    # model_config_path = '/data1/lijiaqi/codebase/CSDs/application/ts3codec/TS3Codec/configs/codec/amlt/librilight_dualcodecsensevoice_sim_1_0_8_3hz.yaml'
    # ckpt_path = '/mnt/wus2/models/users/lijiaqi/exp_codec/v3.9/librilight_dualcodecsensevoice_sim_1_0_8_3hz/60000.pth'
    # model_config_path = '/data1/lijiaqi/codebase/CSDs/application/ts3codec/TS3Codec/configs/codec/amlt/librilight_dualcodecsensevoice_sim_0_9_dcae.yaml'
    # ckpt_path = '/mnt/wus2/models/users/lijiaqi/exp_codec/v3.8/librilight_dualcodecsensevoice_sim_0_9_dcae/120000.pth'
    # model_config_path = '/data1/lijiaqi/codebase/CSDs/application/ts3codec/TS3Codec/configs/codec/amlt/librilight_dualcodecsensevoice_sim_0_9_transformer.yaml'
    # ckpt_path = '/mnt/wus2/models/users/lijiaqi/exp_codec/v3.8/librilight_dualcodecsensevoice_sim_0_9_transformer/100000.pth'
    # model_config_path = '/data1/lijiaqi/codebase/CSDs/application/ts3codec/TS3Codec/configs/codec/amlt/librilight_dualcodecsensevoice_sim_0_85.yaml'
    # ckpt_path = '/mnt/wus2/models/users/lijiaqi/exp_codec/v3.8/librilight_dualcodecsensevoice_sim_0_85/120000.pth'
    # model_config_path = '/data1/lijiaqi/codebase/CSDs/application/ts3codec/TS3Codec/configs/codec/amlt/librilight_dualcodecsensevoice_sim_1_0_4hz.yaml'
    # ckpt_path = '/mnt/wus2/models/users/lijiaqi/exp_codec/v3.9/librilight_dualcodecsensevoice_sim_1_0_4hz/280000.pth'
    # model_config_path = '/data1/lijiaqi/codebase/CSDs/application/ts3codec/TS3Codec/configs/codec/amlt/librilight_dualcodecsensevoice_sim_1_0_8_3hz_layer20_semantic.yaml'
    # ckpt_path = '/mnt/wus2/models/users/lijiaqi/exp_codec/v3.9/librilight_dualcodecsensevoice_sim_1_0_8_3hz_layer20_semantic/260000.pth'
    # model_config_path = '/data1/lijiaqi/codebase/CSDs/application/ts3codec/TS3Codec/configs/codec/amlt/librilight_dualcodecsensevoice_sim_1_0_8_3hz_avg_semantic.yaml'
    # ckpt_path = '/mnt/wus2/models/users/lijiaqi/exp_codec/v3.9/librilight_dualcodecsensevoice_sim_1_0_8_3hz_avg_semantic/260000.pth'
    # model_config_path = '/data1/lijiaqi/codebase/CSDs/application/ts3codec/TS3Codec/configs/codec/amlt/librilight_dualcodecsensevoice_sim_1_0_8_3hz.yaml'
    # ckpt_path = '/mnt/wus2/models/users/lijiaqi/exp_codec/v3.9/librilight_dualcodecsensevoice_sim_1_0_8_3hz/340000.pth'
    # model_config_path = '/data1/lijiaqi/codebase/CSDs/application/ts3codec/TS3Codec/configs/codec/amlt/librilight_dualcodecsensevoice_sim_1_0_8hz_sim0_8.yaml'
    # ckpt_path = '/mnt/wus2/models/users/lijiaqi/exp_codec/v3.9/librilight_dualcodecsensevoice_sim_1_0_8hz_sim0_8/220000.pth'
    # model_config_path = '/data1/lijiaqi/codebase/CSDs/application/ts3codec/TS3Codec/configs/codec/amlt/librilight_dualcodecsensevoice_sim_1_0_8hz_sim0_9_init.yaml'
    # ckpt_path = '/mnt/wus2/models/users/lijiaqi/exp_codec/v3.9/librilight_dualcodecsensevoice_sim_1_0_8hz_sim0_9_init/400000.pth'
    # model_config_path = '/data1/lijiaqi/codebase/CSDs/application/ts3codec/TS3Codec/configs/codec/amlt/librilight_dualcodecsensevoice_sim_1_0_8hz_sim0_85_init.yaml'
    # ckpt_path = '/mnt/wus2/models/users/lijiaqi/exp_codec/v3.9/librilight_dualcodecsensevoice_sim_1_0_8hz_sim0_85_init/400000.pth'
    # model_config_path = '/data1/lijiaqi/codebase/CSDs/application/ts3codec/TS3Codec/configs/codec/amlt/librilight_dualcodecsensevoice_sim_1_0_8hz_sim0_85_init_repa.yaml'
    # ckpt_path = '/mnt/wus2/models/users/lijiaqi/exp_codec/v3.9/librilight_dualcodecsensevoice_sim_1_0_8hz_sim0_85_init_repa/380000.pth'
    # model_config_path = '/data1/lijiaqi/codebase/CSDs/application/ts3codec/TS3Codec/configs/codec/amlt/librilight_dualcodecsensevoice_sim_0_85.yaml'
    # ckpt_path = '/mnt/wus2/models/users/lijiaqi/exp_codec/v3.8/librilight_dualcodecsensevoice_sim_0_85/120000.pth'
    # model_config_path = '/data1/lijiaqi/codebase/CSDs/application/ts3codec/TS3Codec/configs/codec/amlt/librilight_dualcodecsensevoice_sim_1_0_8hz_sim0_8_avg_semantic_last_sim.yaml'
    # ckpt_path = '/mnt/wus2/models/users/lijiaqi/exp_codec/v3.9/librilight_dualcodecsensevoice_sim_1_0_8hz_sim0_8_avg_semantic_last_sim/360000.pth'
    # model_config_path = '/data1/lijiaqi/codebase/CSDs/application/ts3codec/TS3Codec/configs/codec/amlt/librilight_dualcodecsensevoice_sim_1_0_8hz_sim0_8.yaml'
    # ckpt_path = '/mnt/wus2/models/users/lijiaqi/exp_codec/v3.9/librilight_dualcodecsensevoice_sim_1_0_8hz_sim0_8/360000.pth'
    # model_config_path = '/data1/lijiaqi/codebase/CSDs/application/ts3codec/TS3Codec/configs/codec/amlt/librilight_dualcodecsensevoice_sim_1_0_8hz_sim0_85.yaml'
    # ckpt_path = '/mnt/wus2/models/users/lijiaqi/exp_codec/v3.9/librilight_dualcodecsensevoice_sim_1_0_8hz_sim0_85/100000.pth'
    # model_config_path = '/data1/lijiaqi/codebase/CSDs/application/ts3codec/TS3Codec/configs/codec/amlt/librilight_dualcodecsensevoice_sim_1_0_6hz.yaml'
    # ckpt_path = '/mnt/wus2/models/users/lijiaqi/exp_codec/v3.9/librilight_dualcodecsensevoice_sim_1_0_6hz/320000.pth'
    # model_config_path = '/data1/lijiaqi/codebase/CSDs/application/ts3codec/TS3Codec/configs/codec/amlt/librilight_dualcodecsensevoice_sim_1_0_8hz_sim0_8_repa_flowmatching_repa.yaml'
    # ckpt_path = '/mnt/wus2/models/users/lijiaqi/exp_codec/v3.9/librilight_dualcodecsensevoice_sim_1_0_8hz_sim0_8_repa_flowmatching_repa/1200000.pth'
    # model_config_path = '/data1/lijiaqi/codebase/CSDs/application/ts3codec/TS3Codec/configs/codec/amlt/librilight_dualcodecsensevoice_sim_1_0_8hz_sim0_85_init_query_repa.yaml'
    # ckpt_path = '/mnt/wus2/models/users/lijiaqi/exp_codec/v3.9/librilight_dualcodecsensevoice_sim_1_0_8hz_sim0_85_init_query_repa/800000.pth'
    # model_config_path = '/data1/lijiaqi/codebase/CSDs/application/ts3codec/TS3Codec/configs/codec/amlt/librilight_dualcodecsensevoice_sim_1_0.yaml'
    # ckpt_path = '/mnt/wus2/models/users/lijiaqi/exp_codec/v3.8/librilight_dualcodecsensevoice_sim_1_0/400000.pth'
    # model_config_path = '/data1/lijiaqi/codebase/CSDs/application/ts3codec/TS3Codec/configs/codec/amlt/librilight_dualcodec12hz.yaml'
    # ckpt_path = '/mnt/wus2/models/users/lijiaqi/exp_codec/v3.9/librilight_dualcodec12hz/660000.pth'
    # model_config_path = '/data1/lijiaqi/codebase/CSDs/application/ts3codec/TS3Codec/configs/codec/amlt/librilight_dualcodecsensevoice_sim_1_0_8hz_sim0_85_init_query_repa_larger_transformer_insert_before_dynamic_fsq_v2.yaml'
    # ckpt_path = '/mnt/wus2/models/users/lijiaqi/exp_codec/v3.9/librilight_dualcodecsensevoice_sim_1_0_8hz_sim0_85_init_query_repa_larger_transformer_insert_before_dynamic_fsq_v2/920000.pth'
    # model_config_path = '/data1/lijiaqi/codebase/CSDs/application/ts3codec/TS3Codec/configs/codec/amlt/librilight_dualcodec6hz.yaml'
    # ckpt_path = '/mnt/wus2/models/users/lijiaqi/exp_codec/v3.9/librilight_dualcodec6hz/660000.pth'
    # model_config_path = '/data1/lijiaqi/codebase/CSDs/application/ts3codec/TS3Codec/configs/codec/amlt/librilight_dualcodecsensevoice_sim_1_0_6hz_query_repa_larger_transformer_fsq.yaml'
    # ckpt_path = '/mnt/wus2/models/users/lijiaqi/exp_codec/v3.9/librilight_dualcodecsensevoice_sim_1_0_6hz_query_repa_larger_transformer_fsq/800000.pth'
    # model_config_path = '/data1/lijiaqi/codebase/CSDs/application/ts3codec/TS3Codec/configs/codec/amlt/librilight_dualcodecsensevoice_sim_1_0_6hz_query_repa_larger_transformer_fsq.yaml'
    # ckpt_path = '/mnt/wus2/models/users/lijiaqi/exp_codec/v3.9/librilight_dualcodecsensevoice_sim_1_0_6hz_query_repa_larger_transformer_fsq/800000.pth'
    # model_config_path = '/data1/lijiaqi/codebase/CSDs/application/ts3codec/TS3Codec/configs/codec/amlt/librilight_dualcodecsensevoice_sim_1_0_8hz_sim0_85_init_query_repa_larger_transformer_insert_before_dynamic_fsq_v3.yaml'
    # ckpt_path = '/mnt/wus2/models/users/lijiaqi/exp_codec/v3.9/librilight_dualcodecsensevoice_sim_1_0_8hz_sim0_85_init_query_repa_larger_transformer_insert_before_dynamic_fsq_v3/580000.pth'
    # model_config_path = '/data1/lijiaqi/codebase/CSDs/application/ts3codec/TS3Codec/configs/codec/amlt/librilight_dualcodecsensevoice_sim_1_0_12hz_query_repa_larger_transformer_fsq.yaml'
    # ckpt_path = '/mnt/wus2/models/users/lijiaqi/exp_codec/v3.9/librilight_dualcodecsensevoice_sim_1_0_12hz_query_repa_larger_transformer_fsq/580000.pth'
    # model_config_path = '/data1/lijiaqi/codebase/CSDs/application/ts3codec/TS3Codec/configs/codec/amlt/librilight_dualcodecsensevoice_sim_1_0_6hz_query_repa_larger_transformer_fsq_v2.yaml'
    # ckpt_path = '/mnt/wus2/models/users/lijiaqi/exp_codec/v3.9/librilight_dualcodecsensevoice_sim_1_0_6hz_query_repa_larger_transformer_fsq_v2/580000.pth'
    # model_config_path = '/data1/lijiaqi/codebase/CSDs/application/ts3codec/TS3Codec/configs/codec/amlt/librilight_dualcodecsensevoice_sim_1_0_8hz_sim0_85_init_query_repa_larger_transformer_insert_before_dynamic_fsq_v2_scale_query.yaml'
    # ckpt_path = '/mnt/wus2/models/users/lijiaqi/exp_codec/v3.9/librilight_dualcodecsensevoice_sim_1_0_8hz_sim0_85_init_query_repa_larger_transformer_insert_before_dynamic_fsq_v2_scale_query/1200000.pth'
    # model_config_path = '/data1/lijiaqi/codebase/CSDs/application/ts3codec/TS3Codec/configs/codec/amlt/librilight_dualcodecsensevoice_sim_1_0_8hz_sim0_85_init_query_repa_larger_transformer_insert_before_dynamic_fsq_v3.yaml'
    # ckpt_path = '/mnt/wus2/models/users/lijiaqi/exp_codec/v3.9/librilight_dualcodecsensevoice_sim_1_0_8hz_sim0_85_init_query_repa_larger_transformer_insert_before_dynamic_fsq_v3/860000.pth'
    # model_config_path = '/data1/lijiaqi/codebase/CSDs/application/ts3codec/TS3Codec/configs/codec/amlt/librilight_dualcodec12hz.yaml'
    # ckpt_path = '/mnt/wus2/models/users/lijiaqi/exp_codec/v3.9/librilight_dualcodec12hz/680000.pth'
    # model_config_path = '/data1/lijiaqi/codebase/CSDs/application/ts3codec/TS3Codec/configs/codec/amlt/librilight_dac8hz.yaml'
    # ckpt_path = '/mnt/wus2/models/users/lijiaqi/exp_codec/v3.9/librilight_dac8hz/600000.pth'
    # model_config_path = '/data1/lijiaqi/codebase/CSDs/application/ts3codec/TS3Codec/configs/codec/amlt/librilight_dualcodecsensevoice_sim_1_0_6hz_query_repa_larger_transformer_fsq.yaml'
    # ckpt_path = '/mnt/wus2/models/users/lijiaqi/exp_codec/v3.9/librilight_dualcodecsensevoice_sim_1_0_6hz_query_repa_larger_transformer_fsq/800000.pth'
    # model_config_path = '/data1/lijiaqi/codebase/CSDs/application/ts3codec/TS3Codec/configs/codec/amlt/librilight_dualcodecsensevoice_sim_1_0_8hz_sim0_85_init_query_repa_larger_transformer_insert_before_dynamic_fsq_v3_scale_fsq.yaml'
    # ckpt_path = '/mnt/wus2/models/users/lijiaqi/exp_codec/v3.9/librilight_dualcodecsensevoice_sim_1_0_8hz_sim0_85_init_query_repa_larger_transformer_insert_before_dynamic_fsq_v3_scale_fsq/420000.pth'
    # model_config_path = '/data1/lijiaqi/codebase/CSDs/application/ts3codec/TS3Codec/configs/codec/amlt/librilight_dualcodecsensevoice_sim_1_0_8hz_sim0_85_init_query_repa_larger_transformer_insert_before_dynamic_fsq_v3_scale_fsq.yaml'
    # ckpt_path = '/mnt/wus2/models/users/lijiaqi/exp_codec/v3.9/librilight_dualcodecsensevoice_sim_1_0_8hz_sim0_85_init_query_repa_larger_transformer_insert_before_dynamic_fsq_v3_scale_fsq/420000.pth'
    # model_config_path = '/data1/lijiaqi/codebase/CSDs/application/ts3codec/TS3Codec/configs/codec/amlt/librilight_dualcodecsensevoice_sim_1_0_8hz_sim0_85_init_query_repa_larger_transformer_insert_before_dynamic_fsq_v3_noquery.yaml'
    # ckpt_path = '/mnt/scus/models/users/lijiaqi/exp_codec/v3.9/librilight_dualcodecsensevoice_sim_1_0_8hz_sim0_85_init_query_repa_larger_transformer_insert_before_dynamic_fsq_v3_noquery/600000.pth'
    # model_config_path = '/data1/lijiaqi/codebase/CSDs/application/ts3codec/TS3Codec/configs/codec/amlt/librilight_dualcodec12hzv2.yaml'
    # ckpt_path = '/mnt/scus/models/users/lijiaqi/exp_codec/v3.9/librilight_dualcodecsensevoice_sim_1_0_8hz_sim0_85_init_query_repa_larger_transformer_insert_before_dynamic_fsq_v3/360000.pth'
    # model_config_path = '/data1/lijiaqi/codebase/CSDs/application/ts3codec/TS3Codec/configs/codec/amlt/librilight_dualcodecsensevoice_sim_1_0_8hz_sim0_85_init_query_repa_larger_transformer_insert_before_dynamic_fsq_v3.yaml'
    # ckpt_path = '/mnt/scus/models/users/lijiaqi/exp_codec/v3.9/librilight_dualcodecsensevoice_sim_1_0_8hz_sim0_85_init_query_repa_larger_transformer_insert_before_dynamic_fsq_v3/1080000.pth'
    # model_config_path = '/data1/lijiaqi/codebase/CSDs/application/ts3codec/TS3Codec/configs/codec/amlt/librilight_dualcodecsensevoice_sim_1_0_8hz_sim0_85_init_query_repa_larger_transformer_insert_before_dynamic_fsq_v3_smallwindow.yaml'
    # ckpt_path = '/mnt/scus/models/users/lijiaqi/exp_codec/v3.9/librilight_dualcodecsensevoice_sim_1_0_8hz_sim0_85_init_query_repa_larger_transformer_insert_before_dynamic_fsq_v3_smallwindow/1500000.pth'
    model_config_path = '/data1/lijiaqi/codebase/CSDs/application/ts3codec/TS3Codec/configs/codec/amlt/librilight_dualcodecsensevoice_sim_1_0_8hz_sim0_85_init_query_repa_larger_transformer_insert_before_dynamic_fsq_v3_range.yaml'
    ckpt_path = '/mnt/scus/models/users/lijiaqi/exp_codec/v3.9/librilight_dualcodecsensevoice_sim_1_0_12hz/180000.pth'
    # model_config_path = '/data1/lijiaqi/codebase/CSDs/application/ts3codec/TS3Codec/configs/codec/amlt/librilight_dualcodecsensevoice_sim_1_0_8hz_sim0_85_init_query_repa_larger_transformer_insert_before_dynamic_fsq_v2_scale_query.yaml'
    # ckpt_path = '/mnt/scus/models/users/lijiaqi/exp_codec/v3.9/librilight_dualcodecsensevoice_sim_1_0_8hz_sim0_85_init_query_repa_larger_transformer_insert_before_dynamic_fsq_v2_scale_query/1200000.pth'
    # model_config_path = '/data1/lijiaqi/codebase/CSDs/application/ts3codec/TS3Codec/configs/codec/amlt/librilight_dualcodec8hzv2.yaml'
    # ckpt_path = '/mnt/scus/models/users/lijiaqi/exp_codec/v3.9/librilight_dualcodec8hzv2/740000.pth'

    if IS_CLUSTER:
        model_config_path = model_config_path.replace('/data1/lijiaqi/codebase/CSDs/application/ts3codec/TS3Codec', '/modelblob/users/lijiaqi/codebase/TS3Codec')
        print(f'model_config_path: {model_config_path}')
        ckpt_path = ckpt_path.replace('/mnt/wus2/models/users/lijiaqi/exp_codec/v3.9', '/modelblob/users/lijiaqi/exp_codec/v3.9')
        ckpt_path = ckpt_path.replace('/mnt/scus/models/users/lijiaqi/exp_codec/v3.9', '/modelblob/users/lijiaqi/exp_codec/v3.9')
        print(f'ckpt_path: {ckpt_path}')


    with open(model_config_path, 'r') as f:
        model_config = yaml.safe_load(f)

    if 'whisper' in model_config_path:
        # Use whisper config
        if not IS_CLUSTER:
            model_config['model']['semantic_model_path'] = '/data1/lijiaqi/whisper-medium'
        model_config['model']['semantic_model_type'] = 'whisper'
        from audio_codec.utils import build_codec_model
        model = build_codec_model(model_config['model'])
        model.load_state_dict(torch.load(ckpt_path)['soundstream'], strict=False)
        model.eval()
        model.to('cuda')
        feature_extractor_path = '/data1/lijiaqi/whisper-medium'
        feature_extractor = WhisperGen(16000, feature_extractor_path)
        return {"model": model, "feature_extractor": feature_extractor, "type": "whisper"}
    elif 'sensevoice' in model_config_path and 'w2v' not in model_config_path:
        # Use SenseVoice config
        if not IS_CLUSTER:
            model_config['model']['semantic_model_path'] = '/data1/lijiaqi/codebase/TASTE-SpokenLM/STAGE1_TRAIN/storage/pretrained_models/SenseVoiceSmall'
        model_config['model']['semantic_model_type'] = 'sensevoice'
        from audio_codec.utils import build_codec_model
        model = build_codec_model(model_config['model'])
        model.load_state_dict(torch.load(ckpt_path, map_location='cpu')['soundstream'])
        model.eval()
        model.to('cuda')
        feature_extractor = FBankGen(sr=16000)
        return {"model": model, "feature_extractor": feature_extractor, "type": "sensevoice"}
    else:
        # Use w2v-bert-2.0 config
        if not IS_CLUSTER:
            model_config['model']['semantic_model_path'] = '/mnt/wus2/models/projects/lijiaqi_csd/w2v-bert-2.0'
        from audio_codec.utils import build_codec_model
        model = build_codec_model(model_config['model'])
        model.load_state_dict(torch.load(ckpt_path)['soundstream'])
        model.eval()
        model.to('cuda')
        feature_extractor_path = '/mnt/wus2/models/projects/lijiaqi_csd/w2v-bert-2.0' if not IS_CLUSTER else '/modelblob/projects/lijiaqi_csd/w2v-bert-2.0'
        feature_extractor = SeamlessM4TFeatureExtractor.from_pretrained(feature_extractor_path)
        return {"model": model, "feature_extractor": feature_extractor, "type": "w2vbert"}

@torch.no_grad()
def infer(audio: torch.Tensor, model: dict, sample_rate: int=16000, num_quantizers: int = 8):
    """
    Performs end-to-end inference with the DualCodec model.
    Uses the correct feature extraction pipeline based on model['type'].
    """
    audio = audio.reshape(1,-1)
    codec_model = model['model']
    feature_extractor = model['feature_extractor']
    device = next(codec_model.parameters()).device
    
    # Ensure audio is mono and on the correct device
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    audio = audio.to(device)

    # 1. Resample audio: 16kHz for semantic features
    resampler_16k = torchaudio.transforms.Resample(sample_rate, 16000).to(device)
    audio_16k = resampler_16k(audio)
    duration = audio_16k.shape[-1] / 16000
    sim = None

    if model.get('type') == 'sensevoice':
        # Use SenseVoice's FBankGen (on CPU)
        features, _ = feature_extractor.extract_fbank(audio_16k.cpu())
        audio_features = features.to(device)
    elif model.get('type') == 'whisper':
        # Use Whisper's feature extractor (on CPU)
        features = feature_extractor.extract_features(audio_16k.cpu())
        audio_features = features.to(device).unsqueeze(0)
    else:
        # Use SeamlessM4TFeatureExtractor
        features = feature_extractor(audio_16k.cpu(), return_tensors="pt", sampling_rate=16000)
        audio_features = features.input_features.to(device)

    # 3. Time the encoding and decoding steps separately
    start_time = time.time()
    dl_output = {
        "audio": audio_16k,
        "x": audio_features,
        "num_quantizers": num_quantizers,
    }
    # Encode the audio to get semantic and acoustic codes
    encoded_output = codec_model(
        dl_output,
        encode_only=True,
    )
    if "audio" in encoded_output:
        assert False, "Not implemented"
        reconstructed_audio = encoded_output["audio"]
        token_ratio = 1.0
        semantic_features = encoded_output.get("semantic_features", None)
        semantic_codes = encoded_output.get("semantic_codes", None)
    else:
        # Extract the codes and token lengths
        semantic_codes = encoded_output['semantic_codes']
        acoustic_codes = encoded_output['acoustic_codes']
        token_lengths = encoded_output['token_lengths']
        alignmnet_matrix = encoded_output['alignment_matrix']
        sim = encoded_output.get('sim', None)
        
        # Decode from codes to reconstruct the audio
        reconstructed_audio = codec_model.decode_from_codes(
            semantic_codes=semantic_codes,
            acoustic_codes=acoustic_codes,
            token_lengths=token_lengths,
        )
        
        end_time = time.time()
        
        token_ratio = (alignmnet_matrix.shape[1] / alignmnet_matrix.shape[2])
        semantic_features = encoded_output.get("semantic_features", None)
    return {
        "out": reconstructed_audio.cpu().to(torch.float32),
        "compressed": semantic_codes,
        "encode_rtf": token_ratio,
        "decode_rtf": token_ratio,
        "semantic_features": semantic_features,
        "token_lengths": token_lengths,
        "alignment_matrix": alignmnet_matrix,
        "sim": sim, # shape: [1, T]
    }


# Example usage
if __name__ == '__main__':
    # Set config and checkpoint path here
    # Example: model_config_path = '/data1/lijiaqi/codebase/CSDs/application/ts3codec/TS3Codec/configs/codec/amlt/librilight_dualcodecsensevoice_sim_1_0.yaml'
    #          ckpt_path = '/mnt/wus2/models/users/lijiaqi/exp_codec/v3.8/librilight_dualcodecsensevoice_sim_1_0/40000.pth'
    # model_config_path = '/data1/lijiaqi/codebase/CSDs/application/ts3codec/TS3Codec/configs/codec/amlt/librilight_dualcodecsensevoice_sim_1_0.yaml'
    # ckpt_path = '/mnt/wus2/models/users/lijiaqi/exp_codec/v3.8/librilight_dualcodecsensevoice_sim_1_0/40000.pth'

    model_dict = prepare_model()
    
    # Load a real audio file
    audio_path = '/data1/lijiaqi/data/LibriSpeech/librispeech-cropped/clean/61/70968/61-70968-0004.wav'
    audio, sample_rate = torchaudio.load(audio_path)
    
    # Process the audio through the model
    output_dict = infer(
        audio=audio, 
        sample_rate=sample_rate, 
        model=model_dict, 
        num_quantizers=8
    )
    y = output_dict["out"]
    rtf = output_dict["encode_rtf"]
    
    # Save the reconstructed audio (output is at 16kHz)
    output_path = 'decoded_audio.wav'
    torchaudio.save(output_path, y.cpu().reshape(1, -1), 16000)
    
    print(f"Inference RTF: {rtf:.4f}")
    print(f"Saved decoded audio to {output_path}")