"""Thin adapter to the pinned MeanAudio implementation distributed by FdAudio."""
import importlib
import subprocess
import sys
from pathlib import Path

import torch
from torch import nn


class PANNsEncoder(nn.Module):
    def __init__(self, checkpoint):
        super().__init__()
        from .vendor.panns import Cnn14
        self.model = Cnn14(features_list=['2048', 'logits'], sample_rate=16000,
                           window_size=512, hop_size=160, mel_bins=64, fmin=50,
                           fmax=8000, classes_num=527)
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        self.model.load_state_dict(state['model'], strict=True)
        self.eval().requires_grad_(False)

    def forward(self, wave):
        return self.model(wave.float())['2048'].clone()


def load_models(config, device):
    paths = config['paths']
    upstream = paths.get('upstream_root')
    if upstream:
        if not (Path(upstream) / 'meanaudio/model/networks.py').is_file():
            raise FileNotFoundError('upstream_root must point to the pinned FdAudio checkout')
        revision = subprocess.check_output(['git', '-C', upstream, 'rev-parse', 'HEAD'], text=True).strip()
        if revision != '6a0c9c79fd42c643b6474182b1105d596fc5e69d':
            raise ValueError('upstream_root must use the documented FdAudio commit')
        if subprocess.check_output(['git', '-C', upstream, 'diff', 'HEAD', '--', 'meanaudio'], text=True):
            raise ValueError('upstream MeanAudio code has local modifications')
        sys.path.insert(0, upstream)
    else:
        raise ValueError('upstream_root is required for a verifiable generator implementation')
    networks = importlib.import_module('meanaudio.model.networks')
    features = importlib.import_module('meanaudio.model.utils.features_utils')
    model = networks.get_mean_audio('meanaudio_s', use_rope=True, text_c_dim=512).to(device)
    state = torch.load(paths['initialization'], map_location='cpu', weights_only=True, mmap=True)
    if 'weights' in state:
        state = state['weights']
    model.load_weights(state)
    model.train()
    decoder = features.FeaturesUtils(tod_vae_ckpt=Path(paths['vae']),
                           bigvgan_vocoder_ckpt=Path(paths['vocoder']),
                                     encoder_name='t5_clap', enable_conditions=False,
                                     mode='16k', need_vae_encoder=False).to(device).eval().requires_grad_(False)
    encoder = PANNsEncoder(paths['panns']).to(device)
    return model, decoder, encoder


def generate_audio(model, decoder, batch, noise):
    with torch.autocast('cuda', dtype=torch.bfloat16):
        one = torch.ones(len(noise), device=noise.device)
        zero = torch.zeros_like(one)
        z = noise - model.predict_flow(noise, one, zero, model.preprocess_conditions(batch['text'], batch['text_c']))
        wave = decoder.vocode_with_grad(decoder.decode_with_grad(z * model.latent_std + model.latent_mean))
    if wave.ndim == 3 and wave.shape[1] == 1:
        wave = wave[:, 0]
    if wave.ndim != 2:
        raise ValueError(f'unexpected waveform shape {tuple(wave.shape)}')
    return wave.float()
