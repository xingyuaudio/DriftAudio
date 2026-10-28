"""One drifting update: sample, construct a detached field, replay, update."""
import math
import time

import torch

from .field import residual_field, waveform_gradient
from .optim import learning_rate


def drift_step(generate, encoder, model_bank, real_features, optimizer,
               parameters, step, device, query_count=64, encoder_batch_size=8):
    """Update generator parameters while holding references and encoder fixed.

    ``generate(i)`` must replay the same conditioned noise realization for sample
    i. Keeping only detached waveforms between passes bounds autograd memory;
    this is an implementation of the same detached-field surrogate, not a
    second objective or a teacher model.
    """
    started = time.perf_counter()
    for group in optimizer.param_groups:
        group['lr'] = learning_rate(step)

    with torch.no_grad():
        waves = torch.cat([generate(i).detach().cpu() for i in range(query_count)])
        if not torch.isfinite(waves).all():
            raise FloatingPointError('non-finite generated audio')
        features = torch.cat([
            encoder(waves[i:i + encoder_batch_size].to(device))
            for i in range(0, query_count, encoder_batch_size)
        ])
        force, scale = residual_field(features, model_bank.values, real_features)

    gradient = waveform_gradient(
        encoder, waves, force, scale, device, microbatch=encoder_batch_size,
    )
    optimizer.zero_grad(set_to_none=True)
    replay_error = 0.0
    for i in range(query_count):
        wave = generate(i)
        error = float((wave.detach().cpu() - waves[i:i + 1]).abs().max())
        replay_error = max(replay_error, error)
        if not math.isfinite(error) or error != 0:
            raise RuntimeError(f'endpoint replay mismatch at step {step}: {error}')
        surrogate = (wave * gradient[i:i + 1].to(device)).sum() / query_count
        surrogate.backward()

    grad_norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True))
    optimizer.step()
    # Bank entries describe the PRE-update generator used to construct the field.
    model_bank.push(features)
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)
    return dict(
        step=step, lr=learning_rate(step), grad_norm_preclip=grad_norm,
        clip_factor=min(1.0, 1 / max(grad_norm, 1e-30)),
        endpoint_replay_max_abs=replay_error,
        force_rms=float(force.double().square().mean().sqrt()), scale=scale,
        bank_ptr=model_bank.ptr, bank_total_inserted=model_bank.total_inserted,
        step_seconds=time.perf_counter() - started,
    )
