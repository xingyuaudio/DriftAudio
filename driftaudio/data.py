"""Explicit NPZ index and query plan: row ordering must be preserved."""
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch


def seed_all(value):
    random.seed(value)
    np.random.seed(value % (2**32))
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


class CachedDataset:
    def __init__(self, manifest, npz_root):
        with Path(manifest).open(newline='', encoding='utf-8') as handle:
            self.rows = list(csv.DictReader(handle, delimiter='\t'))
        ids = [r['id'] for r in self.rows]
        if len(ids) != len(set(ids)) or not ids:
            raise ValueError("training TSV requires unique nonempty IDs")
        self.npz_root = Path(npz_root)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        name = self.rows[index].get('npz') or f'{index}.npz'
        with np.load(self.npz_root / name, allow_pickle=False) as sample:
            return {key: torch.from_numpy(sample[source]).float()
                    for key, source in [('mean','mean'),('text','text_features'),('text_c','text_features_c')]}


def batch_and_noise(dataset, indices, slots, seed_base, step, device):
    loaded, noises = [], []
    for index, slot in zip(indices, slots, strict=True):
        batch = {k: v.unsqueeze(0).to(device) for k, v in dataset[int(index)].items()}
        seed_all(int(seed_base) + int(step) * 10007 + int(slot) * 1000003)
        noises.append(torch.randn_like(batch['mean']))
        loaded.append(batch)
    return ({k: torch.cat([b[k] for b in loaded]) for k in loaded[0]}, torch.cat(noises))


def validate_plan(path, dataset, steps, query_count=64):
    plan = json.loads(Path(path).read_text())
    if plan['query_count_per_step'] != query_count or len(plan['entries']) < steps:
        raise ValueError("query plan does not cover requested steps/Q")
    for step, entry in enumerate(plan['entries'][:steps], 1):
        if 'step' in entry and entry['step'] != step:
            raise ValueError(f'query plan step ordering mismatch at {step}')
        if len(entry['query_indices']) != query_count or len(entry['noise_seed_slots']) != query_count:
            raise ValueError(f"invalid query count at step {step}")
        if any(not 0 <= i < len(dataset) for i in entry['query_indices']):
            raise ValueError(f"query index out of range at step {step}")
        if 'query_audio_ids' in entry and entry['query_audio_ids'] != [dataset.rows[i]['id'] for i in entry['query_indices']]:
            raise ValueError(f'query plan row/ID mismatch at step {step}')
    return plan
