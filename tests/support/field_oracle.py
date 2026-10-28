import math
import torch
from types import SimpleNamespace
base = SimpleNamespace(MODEL_COUNT=1024, RADIUS=0.2)
@torch.no_grad()
def field_parts(query, model_reference, real_reference):
    query = query.double()
    targets = torch.cat([model_reference.double(), real_reference.double()], dim=0)
    distance = (query.square().sum(-1, keepdim=True) + targets.square().sum(-1)[None] - 2 * query @ targets.T).clamp_min(1e-08).sqrt()
    scale = distance.mean().clamp_min(1e-12)
    scale_inputs = (scale / math.sqrt(query.shape[-1])).clamp_min(1e-12)
    logits = -(distance / scale) / base.RADIUS
    affinity = (0.5 * (torch.log_softmax(logits, -1) + torch.log_softmax(logits, -2))).exp()
    (am, ar) = (affinity[:, :base.MODEL_COUNT], affinity[:, base.MODEL_COUNT:])
    (cm, cr) = (-am * ar.sum(-1, keepdim=True), ar * am.sum(-1, keepdim=True))
    (qs, ts) = (query / scale_inputs, targets / scale_inputs)
    attraction = cr @ ts[base.MODEL_COUNT:] - cr.sum(-1, keepdim=True) * qs
    repulsion = cm @ ts[:base.MODEL_COUNT] - cm.sum(-1, keepdim=True) * qs
    coefficients = torch.cat([cm, cr], dim=-1)
    full = torch.einsum('iy,yx->ix', coefficients, ts) - coefficients.sum(-1, keepdim=True) * qs
    return (full.float(), attraction.float(), repulsion.float(), float(scale_inputs))

def combine(values, balanced):
    rms = {k: float(v.double().square().mean().sqrt()) for (k, v) in values.items()}
    raw = sum(values.values(), torch.zeros_like(next(iter(values.values()))))
    raw_rms = float(raw.double().square().mean().sqrt())
    if balanced:
        equalized = sum((v / max(rms[k], 1e-12) for (k, v) in values.items()), torch.zeros_like(raw)) / len(values)
        rescale = raw_rms / max(float(equalized.double().square().mean().sqrt()), 1e-12)
        total = equalized * rescale
        coefficients = {k: rescale / (len(values) * max(rms[k], 1e-12)) for k in values}
    else:
        total = raw
        coefficients = {k: 1.0 for k in values}
    return (total, rms, coefficients, raw_rms)
