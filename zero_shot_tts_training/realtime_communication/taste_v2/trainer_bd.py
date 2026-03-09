import json
import os
import shutil
import torch
import time
import sys
from pathlib import Path
sys.path.append(f'{str(Path(__file__).parent.parent.parent.parent)}')
from pathlib import Path
import torch
import torchaudio
from tqdm import tqdm
import torch.nn as nn
from zero_shot_tts_training.cosyvoice.utils.amphion.base_trainer import BaseTrainer
import safetensors
import numpy as np
from zero_shot_tts_training.realtime_communication.codec_model.discriminator import Discriminator
import torch.nn.functional as F
from einops import rearrange
from easydict import EasyDict as edict
from audiotools import AudioSignal
USE_HINGE_LOSS = False

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

        from zero_shot_tts_training.realtime_communication.codec_model.loss import GANLoss, MelSpectrogramLoss, MultibandMelSpectrogramLoss
        self.gan_loss = GANLoss(self.cfg.discriminator_model)
        self.spec_loss = MelSpectrogramLoss(
            pow=1.0, 
            mag_weight=0,
            log_weight=2,
            n_mels = [5, 10, 20, 40, 80, 160, 320],
            window_lengths = [32, 64, 128, 256, 512, 1024, 2048],
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
        if hasattr(self.model, 'module'):
            self.model_module = self.model.module
        else:
            self.model_module = self.model

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
        if self.cfg.batch_anneal:
            if self.step < self.cfg.batch_anneal_steps:
                B, _, _ = batch['speech'].shape
                batch['speech'] = batch['speech'][B//3:]
                batch['speech_lens'] = batch['speech_lens'][B//3:]
        x_wav = batch['speech'].clone().detach()
        dl_output = {
            'audio': batch['speech'],
            'audio_lens': batch['speech_lens'],
        }

        out_dict = self.model(
            dl_output, 
        )

        generator_out = out_dict.x

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

        total_loss = 0.25 * out_dict['vq/commitment_loss'] \
            + 1.0 * adv_g_loss + 2.0 * feat_loss \
            + 15.0 * spec_loss \
            + 1.0 * out_dict['vq/codebook_loss'] \
            + 1.0 * out_dict['distill_loss'] \
            + 1.0 * out_dict['repa_loss']
        metrics = {
            'discriminator_loss': disc_loss,
            'generator_loss': adv_g_loss,
            'feature_loss': feat_loss,
            'spec_loss': spec_loss,
            'commitment_loss': out_dict['vq/commitment_loss'],
            'codebook_loss': out_dict['vq/codebook_loss'],
            'distill_loss': out_dict['distill_loss'],
            'total_loss': total_loss,
            'repa_loss': out_dict['repa_loss'],
        }

        if self.model_module.config.add_semantic_spec_loss and out_dict['bypassed_quantize']:
            semantic_spec_loss = 15.0 * self.semantic_spec_loss(AudioSignal(x_wav, 24000), AudioSignal(generator_out, 24000))
            total_loss += semantic_spec_loss
            metrics.update({
                'semantic/semantic_spec_loss': semantic_spec_loss,
            })

        self.optimizer.zero_grad()
        self.accelerator.backward(total_loss)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return None, metrics

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
        pass

    def test_loop(self):
        return