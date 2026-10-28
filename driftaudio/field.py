"""The single-space residual drifting field; no FD objective or teacher."""
import math

import torch


@torch.no_grad()
def residual_field(query, model_reference, real_reference, radius=0.2):
    """Return detached float32 force and detached feature scale.

    Columns are model references followed by real references. Affinity is the
    geometric mean of row- and column-softmax probabilities, evaluated in log
    space. This is not a Sinkhorn iteration and not a conditional-text field.
    """
    if radius <= 0 or query.ndim != 2 or query.shape[0] == 0:
        raise ValueError("positive radius and nonempty [Q,D] queries required")
    if any(x.ndim != 2 or x.shape[0] == 0 or x.shape[1] != query.shape[1]
           for x in (model_reference, real_reference)):
        raise ValueError("reference matrices must be nonempty and share feature dimension")
    if not all(torch.isfinite(x).all() for x in (query, model_reference, real_reference)):
        raise FloatingPointError("non-finite query/reference")
    q = query.double()
    targets = torch.cat((model_reference.double(), real_reference.double()), dim=0)
    distance = (q.square().sum(-1, keepdim=True) + targets.square().sum(-1)[None]
                - 2 * q @ targets.T).clamp_min(1e-8).sqrt()
    scale = distance.mean().clamp_min(1e-12)
    scale_inputs = (scale / math.sqrt(q.shape[-1])).clamp_min(1e-12)
    logits = -(distance / scale) / radius
    affinity = (0.5 * (torch.log_softmax(logits, -1) + torch.log_softmax(logits, -2))).exp()
    model_count = model_reference.shape[0]
    am, ar = affinity[:, :model_count], affinity[:, model_count:]
    coefficients = torch.cat((-am * ar.sum(-1, keepdim=True), ar * am.sum(-1, keepdim=True)), -1)
    force = torch.einsum("iy,yx->ix", coefficients, targets / scale_inputs)
    force -= coefficients.sum(-1, keepdim=True) * (q / scale_inputs)
    if not torch.isfinite(force).all():
        raise FloatingPointError("non-finite drifting force")
    return force.float(), float(scale_inputs)


def waveform_gradient(encoder, waves, force, scale, device, microbatch=8):
    """VJP through a frozen encoder. Q normalization happens during replay."""
    pieces = []
    for start in range(0, len(waves), microbatch):
        wave = waves[start:start + microbatch].to(device).detach().requires_grad_(True)
        features = encoder(wave).float()
        cotangent = -2 * force[start:start + microbatch].to(device) / (features.shape[-1] * scale)
        gradient, = torch.autograd.grad(features, wave, grad_outputs=cotangent)
        pieces.append(gradient.detach().float().cpu())
    return torch.cat(pieces)
