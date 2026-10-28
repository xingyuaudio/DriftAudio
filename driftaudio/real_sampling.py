"""Independent uniform without-replacement real subset per global optimizer step."""
import torch
SEED=2026092001

def real_indices(step, population=45173, count=2048):
    if step<1 or not 0<count<=population: raise ValueError('Invalid real sampling request')
    return torch.randperm(population,generator=torch.Generator(device='cpu').manual_seed(SEED+step))[:count]
