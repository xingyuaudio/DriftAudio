"""Build train-only banks and a deterministic query plan from cached data."""
import argparse
import json
import random
from pathlib import Path

import torch

from .backend import load_models, generate_audio
from .data import CachedDataset, seed_all
from .io import atomic_json, atomic_save, load_config, sha256


@torch.no_grad()
def prepare(config, real_features, output, steps=1500):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    dataset = CachedDataset(config['paths']['train_manifest'], config['paths']['npz_root'])
    real = torch.load(real_features, map_location='cpu', weights_only=True)
    ids = [r['id'] for r in dataset.rows]
    if real['ids'] != ids or real['features'].shape != (len(ids), 2048):
        raise ValueError('real PANNs features must align exactly with training TSV IDs')
    if not torch.isfinite(real['features']).all():
        raise ValueError('non-finite real features')
    # Preserve the historical selection rule: sample4096+8192 once, then retain
    # first1024 model and first2048 real references from those disjoint blocks.
    chosen = random.Random(20260915100004).sample(range(len(ids)), 4096 + 8192)
    model_indices, real_indices = chosen[:1024], chosen[4096:4096+2048]
    excluded = set(model_indices + real_indices)
    eligible = [i for i in range(len(ids)) if i not in excluded]
    rng = random.Random(2026091203)
    plan = dict(noise_seed_base=2026091203, query_count_per_step=64, steps=steps,
                entries=[dict(query_indices=rng.sample(eligible,64), noise_seed_slots=list(range(64)),
                              noise_seed_base=2026091203) for _ in range(steps)])
    device = torch.device('cuda:0')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    model, decoder, encoder = load_models(config, device)
    model.eval()
    chunks = []
    for start in range(0,1024,8):
        batches, noises = [], []
        for position in range(start,start+8):
            batch={k:v.unsqueeze(0).to(device) for k,v in dataset[model_indices[position]].items()}
            seed_all(2091500000+position)
            noises.append(torch.randn_like(batch['mean'])); batches.append(batch)
        batch={k:torch.cat([b[k] for b in batches]) for k in batches[0]}
        waves=generate_audio(model,decoder,batch,torch.cat(noises))
        chunks.append(encoder(waves).cpu())
        if (start+8)%128==0:
            print(f'model references {start+8}/1024',flush=True)
    atomic_save(dict(model=torch.cat(chunks), real=real['features'][real_indices].float(),
                     initialization_sha256=sha256(config['paths']['initialization']),
                     train_manifest_sha256=sha256(config['paths']['train_manifest']),
                     model_reference_ids=[ids[i] for i in model_indices],
                     real_reference_ids=[ids[i] for i in real_indices]), output/'bank.pt')
    atomic_json(plan, output/'query_plan.json')
    atomic_json(dict(train_only=True, query_eligible_count=len(eligible),
                     bank_sha256=sha256(output/'bank.pt'), plan_sha256=sha256(output/'query_plan.json'),
                     note='New deterministic query plan; exact published replay requires the archived query plan.'), output/'audit.json')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True); p.add_argument('--real-features',required=True)
    p.add_argument('--output',required=True); p.add_argument('--steps',type=int,default=1500)
    a=p.parse_args(); prepare(load_config(a.config),a.real_features,a.output,a.steps)


if __name__=='__main__':main()
