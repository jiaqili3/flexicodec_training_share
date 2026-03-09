import json
import os
import shutil
import torch
import time
from pathlib import Path
import torch
import torchaudio
from tqdm import tqdm
import torch.nn as nn
from cosyvoice.utils.amphion.base_trainer import BaseTrainer
import safetensors
import numpy as np
from .discriminator import Discriminator
import torch.nn.functional as F
from einops import rearrange
from easydict import EasyDict as edict
USE_HINGE_LOSS = False
from audiotools import AudioSignal

from realtime_communication.speechtokenizer_official.trainer import extract_hubert_codes

def d_axis_distill_loss(feature, target_feature):
    # input: (b, t, c)
    assert feature.size(-1) == target_feature.size(-1)
    n = min(feature.size(1), target_feature.size(1))
    assert abs(target_feature.size(1) - feature.size(1)) <= 3
    distill_loss = - torch.log(torch.sigmoid(torch.nn.functional.cosine_similarity(feature[:, :n], target_feature[:, :n], axis=1))).mean()
    return distill_loss
class Trainer(BaseTrainer):
    """Trainer"""

    def __init__(self, args=None, cfg=None, **kwargs):
        """
            Initializes the model with the given arguments and configuration.

        Args:
            args (argparse.Namespace, optional): Arguments to be passed on to the model. Defaults to None.
            cfg (dict, optional): Configuration dictionary containing parameters for the model. Defaults to None.
        """
        super().__init__(args, cfg)
        torch.backends.cudnn.benchmark = True

        from .loss import GANLoss, MelSpectrogramLoss, MultibandMelSpectrogramLoss
        self.gan_loss = GANLoss(self.cfg.discriminator_model)
        self.spec_loss = MelSpectrogramLoss(
            pow=2, 
            mag_weight=1,
            log_weight=1,
            n_mels = [40, 80, 160, 320],
            window_lengths = [256, 512, 1024, 2048],
        )
        self.semantic_spec_loss = MultibandMelSpectrogramLoss(
            # bands=[(0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)],
            # band_weights=[16,8,4,2,1],
            bands=[(0.0, 0.1)],
            band_weights=[1.0],
            loss_fn=nn.MSELoss(),
            pow=2, 
            mag_weight=1,
            log_weight=1,
            n_mels = [80, 160, 320],
            window_lengths = [512, 1024, 2048],
        )
        if (
            hasattr(self.cfg, "skip_semantic_normalize")
            and self.cfg.skip_semantic_normalize
        ):
            print('skipping semantic normalize')
        if hasattr(self.cfg, 'use_hubert') and self.cfg.use_hubert:
            self.cfg.use_hubert = True
        else:
            self.cfg.use_hubert = False
        if hasattr(self.cfg, 'use_w2vbert2') and self.cfg.use_w2vbert2:
            self.cfg.use_w2vbert2 = True
        else:
            self.cfg.use_w2vbert2 = False

        if hasattr(self.cfg, 'semantic_model'):
            for key in self.cfg.semantic_model:
                if isinstance(self.cfg.semantic_model[key], torch.nn.Module) or isinstance(
                    self.cfg.semantic_model[key], torch.Tensor
                ):
                    self.cfg.semantic_model[key] = self.cfg.semantic_model[key].to(
                        self.accelerator.device
                    )
        self.distill = False
        if hasattr(self.cfg, 'distill') and self.cfg.distill:
            self.distill = True
        if hasattr(self.cfg, 'lambda_commitment_loss'):
            self.lambda_commitment_loss = self.cfg.lambda_commitment_loss
        else:
            self.lambda_commitment_loss = 0.25

        if hasattr(self.model, 'module'):
            self.model_module = self.model.module
        else:
            self.model_module = self.model
        
        if hasattr(self.model, 'semantic_downsample_factor'):
            print(f'self.semantic_downsample_factor = {self.model.semantic_downsample_factor}')
            self.semantic_downsample_factor = self.model.semantic_downsample_factor
        elif hasattr(self.model, 'module') and hasattr(self.model.module, 'semantic_downsample_factor'):
            print(f'self.semantic_downsample_factor = {self.model.module.semantic_downsample_factor}')
            self.semantic_downsample_factor = self.model.module.semantic_downsample_factor
        elif hasattr(self.cfg, 'semantic_downsample_factor') and hasattr(self.cfg, 'semantic_downsample_factor'):
            print(f'self.semantic_downsample_factor = {self.cfg.semantic_downsample_factor}')
            self.semantic_downsample_factor = self.cfg.semantic_downsample_factor
        else:
            print(f'self.semantic_downsample_factor = 2 because not overriden')
            self.semantic_downsample_factor = 2

        if hasattr(self.model, 'override_dac_encoder'):
            print(f'self.override_dac_encoder = {self.model.override_dac_encoder}')
            self.override_dac_encoder = self.model.override_dac_encoder
        elif hasattr(self.model, 'module') and hasattr(self.model.module, 'override_dac_encoder'):
            print(f'self.override_dac_encoder = {self.model.module.override_dac_encoder}')
            self.override_dac_encoder = self.model.module.override_dac_encoder
        else:
            self.override_dac_encoder = False

        self.feature_extractor = None
    
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


    def _build_model(self):
        """
        Returns: None
        """
        return edict({
            'generator': self.cfg.model,
            'discriminator': self.cfg.discriminator_model,
        })

    def _build_optimizer(self):
        r"""Build optimizer for model."""
        return edict({
            'optimizer_g': self.cfg.train.optimizer(params=self.model.generator.parameters()),
            'optimizer_d': self.cfg.train.optimizer(params=self.model.discriminator.parameters()),
        })

    def _accelerator_prepare(self):
        """
        Returns: None
        """
        (
            self.model,
            self.discriminator,
            self.optimizer,
            self.optimizer_d,
        ) = self.accelerator.prepare(
            self.model.generator,
            self.model.discriminator,
            self.optimizer.optimizer_g,
            self.optimizer.optimizer_d,
        )

    def _build_scheduler(self):
        """
        Returns: None
        """
        return None


    def _train_step(self, batch):
        optim_g, optim_d = self.optimizer, self.optimizer_d

        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.accelerator.device)

        
        x_wav, audio_lengths = batch["speech"], batch["speech_lens"]

        x_wav = x_wav.float()[:, None, :]

        if self.override_dac_encoder:
            input_features = batch["input_features"]
            attention_mask = batch["attention_mask"]
            feat = self._extract_semantic_code(
                input_features, attention_mask
            ).transpose(1,2)
            feat = torch.nn.functional.avg_pool1d(feat, self.semantic_downsample_factor, self.semantic_downsample_factor)
            
            # pad_amount = 
            # padded_input_features = torch.nn.functional.pad(input_features.transpose(1,2), (1,0),)
            out_dict, semantic_edict = self.model(input_features, semantic_repr=feat,
                                                bypass_quantize_rate=0.125,
                                                possibly_no_quantizer=False, # internal dropout
            )
        elif self.cfg.semantic_vq:
            if self.cfg.use_hubert or self.cfg.use_w2vbert2:
                feat = None
            else:
                try:
                    input_features = batch["input_features"]
                    attention_mask = batch["attention_mask"]
                except: # features are not given in dataset loader
                    if self.feature_extractor is None:
                        import transformers
                        self.feature_extractor = transformers.SeamlessM4TFeatureExtractor.from_pretrained(
                            pretrained_model_name_or_path=self.cfg.train.w2v_path,
                        )
                    input_features = []
                    attention_masks = []
                    for batch_idx in range(batch['speech'].shape[0]):
                        # resample
                        if batch['sample_rate'] != 16000:
                            batch_speech_resampled = torchaudio.functional.resample(
                                batch['speech'][batch_idx][None, ...].cpu(),
                                orig_freq=batch['sample_rate'],
                                new_freq=16000,
                            )
                        input_values = self.feature_extractor(
                            batch_speech_resampled, sampling_rate=16000, return_tensors="pt"
                        )
                        input_features.append(input_values.input_features)
                        attention_masks.append(input_values.attention_mask)
                    input_features = torch.cat(input_features, dim=0).to(self.accelerator.device)
                    attention_mask = torch.cat(attention_masks, dim=0).to(self.accelerator.device)
                feat = self._extract_semantic_code(
                    input_features, attention_mask
                ).transpose(1,2)
                feat = torch.nn.functional.avg_pool1d(feat, self.semantic_downsample_factor, self.semantic_downsample_factor)
            out_dict, semantic_edict = self.model(x_wav, semantic_repr=feat,
                                                bypass_quantize_rate=0.125,
                                                possibly_no_quantizer=False, # internal dropout
            )
        else:
            out_dict = self.model(
                x_wav, 
            )
            semantic_edict = None

        generator_out = out_dict.x
        if hasattr(out_dict, 'penalty'):
            commitment_loss = out_dict.penalty
        elif hasattr(out_dict, 'kl'):
            commitment_loss = out_dict.kl
        else:
            commitment_loss = 0.0
        metrics = out_dict.metrics
        if hasattr(out_dict, 'vq/codebook_loss'):
            codebook_loss = out_dict['vq/codebook_loss']
        else:
            codebook_loss = 0.0
        if hasattr(out_dict, 'first_layer_quantized'):
            first_layer_quantized = out_dict['first_layer_quantized']
        else:
            first_layer_quantized = None

        matched_len = min(generator_out.shape[-1], x_wav.shape[-1])
        generator_out = generator_out[..., :matched_len]
        x_wav = x_wav[..., :matched_len]

        # --------- Discriminator training ------------
        if USE_HINGE_LOSS:
            disc_loss = self.gan_loss.discriminator_hinge_loss(generator_out, x_wav)
        else:
            disc_loss = self.gan_loss.discriminator_loss(generator_out, x_wav)
        self.optimizer_d.zero_grad()
        self.accelerator.backward(disc_loss)
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 1.0)
        self.optimizer_d.step()
        self.optimizer_d.zero_grad()

        if USE_HINGE_LOSS:
            adv_g_loss, feat_loss = self.gan_loss.generator_hinge_loss(generator_out, x_wav)
        else:
            adv_g_loss, feat_loss = self.gan_loss.generator_loss(generator_out, x_wav)
        spec_loss = self.spec_loss(AudioSignal(x_wav, 24000), AudioSignal(generator_out, 24000))
        # spec_loss = reconstruction_loss(x_wav, generator_out, args)

        total_loss = self.lambda_commitment_loss * commitment_loss \
            + 1.0 * adv_g_loss + 2.0 * feat_loss \
            + 15.0 * spec_loss \
            + 1.0 * codebook_loss
        # ---------- Generator training ----------------
        if semantic_edict:
            distill_loss = F.mse_loss(semantic_edict['semantic_repr'], semantic_edict['x'])
            total_loss += distill_loss * self.cfg.lambda_distill_loss \
                + self.cfg.lambda_semantic_commitment_loss * semantic_edict['penalty'] \
                + self.cfg.lambda_semantic_codebook_loss + semantic_edict['vq/codebook_loss']
            metrics.update({
                'semantic/semantic_commitment_loss': semantic_edict['penalty'],
                'semantic/semantic_codebook_loss': semantic_edict['vq/codebook_loss'], 
                'semantic/semantic_distill_loss': distill_loss,
            })
            if semantic_edict['bypassed_quantize']:
                metrics.update({
                    'semantic/spec_loss': spec_loss,
                })
            if self.cfg.add_semantic_spec_loss and semantic_edict['bypassed_quantize']:
                semantic_spec_loss = 15.0 * self.semantic_spec_loss(AudioSignal(x_wav, 24000), AudioSignal(generator_out, 24000))
                total_loss += semantic_spec_loss
                metrics.update({
                    'semantic/semantic_spec_loss': semantic_spec_loss,
                })
 

        if self.distill:
            if self.cfg.use_hubert:
                input_features = batch["input_features"]
                feat = extract_hubert_codes(self.cfg.semantic_model['model'], input_features, target_layer='avg')
                # (b, t, c)
                if self.model_module.downsample_rate == 480:
                    feat = torch.nn.functional.avg_pool1d(feat.transpose(1,2), 1, 1)
                    # (b,c,t)
                elif self.model_module.downsample_rate == 960:
                    feat = torch.nn.functional.avg_pool1d(feat.transpose(1,2), 2, 2)
                    # (b,c,t)
                elif self.model_module.downsample_rate == 1920:
                    feat = torch.nn.functional.avg_pool1d(feat.transpose(1,2), 4, 4)
                    # (b,c,t)
                else:
                    raise NotImplementedError
            elif self.cfg.use_w2vbert2:
                input_features = batch["input_features"]
                attention_mask = batch["attention_mask"]
                feat = self._extract_semantic_code(
                    input_features, attention_mask
                ).transpose(1,2)
                if self.model_module.downsample_rate == 480:
                    feat = torch.nn.functional.avg_pool1d(feat, 1, 1)
                    # (b,c,t)
                elif self.model_module.downsample_rate == 960:
                    feat = torch.nn.functional.avg_pool1d(feat, 2, 2)
                    # (b,c,t)
                elif self.model_module.downsample_rate == 1920:
                    feat = torch.nn.functional.avg_pool1d(feat, 4, 4)
                    # (b,c,t)
                else:
                    raise NotImplementedError
            else:
                input_features = batch["input_features"]
                feat = self._extract_semantic_code(
                    input_features, attention_mask
                ).transpose(1,2)
                feat = torch.nn.functional.avg_pool1d(feat, self.semantic_downsample_factor, self.semantic_downsample_factor)
            assert feat.shape[1] == first_layer_quantized.shape[1] # (b,c,t)
            distill_loss = d_axis_distill_loss(feat.transpose(1,2), first_layer_quantized.transpose(1,2))
            # distill_loss = F.mse_loss(feat, first_layer_quantized[..., :feat.shape[-1]])
            total_loss += distill_loss * self.cfg.lambda_distill_loss
        else:
            distill_loss = 0.0

        self.optimizer.zero_grad()
        self.accelerator.backward(total_loss)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        # print(commitment_loss, spec_loss)
        # print(x_wav.shape[0])
        # breakpoint()
        # print(out_dict.codes)
        # if self.step >= 100:
        #     breakpoint()
        metrics.update({
            'codebook_loss': codebook_loss,
            'spec_loss': spec_loss, 
            'feat_loss': feat_loss,
            'adv_g_loss': adv_g_loss,
            'total_loss': total_loss,
            "Train/Batch Size": x_wav.shape[0],
            'disc_loss': disc_loss.item(),
            'distill_loss': distill_loss,
            "self.lambda_commitment_loss": self.lambda_commitment_loss,
        })
        if hasattr(out_dict, 'penalty'):
            metrics.update({
                'commitment_loss': commitment_loss,
            })
        elif hasattr(out_dict, 'kl'):
            metrics.update({
                'kl_loss': out_dict.kl,
            })
        # print(metrics)

        return None, metrics

        # Encode and transform mels
        if hasattr(self.model, 'module'):
            module = self.model.module
        else:
            module = self.model
        with torch.no_grad():
            encoded_mels = module.encode_mel_transform(audios)
            gt_mels = module.gt_mel_transform(audios)
            quality = ((gt_mels.mean(-1) > -8).sum(-1) - 90) / 10
            quality = quality.unsqueeze(-1)
        mel_lengths = audio_lengths // module.gt_mel_transform.hop_length
        mel_masks = sequence_mask(mel_lengths, gt_mels.shape[2])
        mel_masks_float_conv = mel_masks[:, None, :].float()
        gt_mels = gt_mels * mel_masks_float_conv
        encoded_mels = encoded_mels * mel_masks_float_conv
        downsample_factor = np.prod(module.downsample_factor)
        assert encoded_mels.shape[-1] % downsample_factor == 0

        # Encode
        with self.accelerator.accumulate(self.model):
            encoded_mels = rearrange(encoded_mels, 'b c (t g) -> b (g c) t', g=downsample_factor)

            encoded_features = module.encoder(encoded_mels)

            # Quantize
            # vq_result = module.quantizer(encoded_features)
            # loss_vq = getattr(vq_result, "loss", torch.tensor(0.0))
            
            (encoded_features, indices, loss_vq), loss_breakdown = module.quantizer(encoded_features, return_loss_breakdown=True)


            vq_recon_features = encoded_features

            vq_recon_features = (
                vq_recon_features + module.quality_projection(quality)[:, :, None]
            )

            # VQ Decode
            gen_mel = (
                module.decoder(
                    torch.randn_like(vq_recon_features),
                    condition=vq_recon_features,
                )
            )

            gen_mel = rearrange(gen_mel, 'b (g c) t -> b c (t g)', g=downsample_factor)
            # Discriminator
            real_logits = self.discriminator(gt_mels)
            fake_logits = self.discriminator(gen_mel.detach())
            d_mask = F.interpolate(
                mel_masks_float_conv, size=(real_logits.shape[2],), mode="nearest"
            )

            loss_real = avg_with_mask((real_logits - 1) ** 2, d_mask)
            loss_fake = avg_with_mask(fake_logits**2, d_mask)
            loss_d = loss_real + loss_fake

            self.accelerator.backward(loss_d)
            optim_d.step()
            optim_d.zero_grad()

            # Mel Loss, applying l1, using a weighted sum
            mel_distance = (gen_mel * mel_masks_float_conv - gt_mels).abs()
            loss_mel_low_freq = avg_with_mask(mel_distance[:, :40, :], mel_masks_float_conv)
            loss_mel_mid_freq = avg_with_mask(mel_distance[:, 40:70, :], mel_masks_float_conv)
            loss_mel_high_freq = avg_with_mask(mel_distance[:, 70:, :], mel_masks_float_conv)
            loss_mel = avg_with_mask(mel_distance[:, :, :], mel_masks_float_conv)
            # loss_mel = (
            #     loss_mel_low_freq * 0.6 + loss_mel_mid_freq * 0.3 + loss_mel_high_freq * 0.1
            # )

            # Adversarial Loss
            fake_logits = self.discriminator(gen_mel)
            loss_adv = avg_with_mask((fake_logits - 1) ** 2, d_mask)

            # Total loss
            loss = (
                module.weight_vq * loss_vq
                + module.weight_mel * loss_mel
                + module.weight_adv * loss_adv
            )

            # Generator backward
            self.accelerator.backward(loss)
            optim_g.step()
            optim_g.zero_grad()

            # print({
            #     "loss": loss.item(),
            #     "loss_d": loss_d.item(),
            #     "loss_vq": loss_vq.item(),
            #     "loss_mel": loss_mel.item(),
            #     "loss_adv": loss_adv.item(),
            # })
            self.accelerator.log({
                "loss": loss.item(),
                "loss_d": loss_d.item(),
                "loss_vq": loss_vq.item(),
                "loss_mel": loss_mel.item(),
                "loss_adv": loss_adv.item(),
            }, step=self.step)
            # print(loss.item())
            self.accelerator.log(loss_breakdown._asdict(), step=self.step)
            return None

    def _test_step(self, batch):
        raise NotImplementedError

    @torch.inference_mode()
    def _valid_epoch(self):
        r"""Testing epoch. Should return average loss of a batch (sample) over
        one epoch. See ``train_loop`` for usage.
        """
        epoch_sum_loss = 0.0
        return epoch_sum_loss

    def _inference(self):
        """
            执行推理过程，不需要实现该方法。
        可以在子类中重写此方法来实现自定义的推理逻辑。

        Returns:
            None, 无返回值。
        """
        pass

    def test_loop(self):
        """
            测试循环，遍历训练数据集，对每个批次进行一次测试。
        返回值为None，不需要返回任何结果。

        Args:
            无参数，该方法不接受任何参数。

        Returns:
            无返回值，该方法不需要返回任何结果。
        """
        return
        self.model.eval()
        for batch in self.train_dataloader:
            self._test_step(batch)
