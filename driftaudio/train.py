"""Single-GPU Q64 training with detached waveform VJP and exact endpoint replay."""
from .async_control import stop_requested, after_checkpoint
import argparse
import json
from pathlib import Path

import torch

from .backend import load_models, generate_audio
from .bank import ModelBank
from .data import CachedDataset, batch_and_noise, seed_all, validate_plan
from .optim import make_optimizer
from .step import drift_step
from .io import atomic_json, atomic_save, load_config, restore_rng, rng_state, sha256
from .real_sampling import real_indices, SEED


def save_checkpoint(out, step, model, optimizer, bank, signature):
    online = out / f'online_step{step:06d}.pth'
    resume = out / f'resume_step{step:06d}.pt'
    if online.exists() or resume.exists():
        raise FileExistsError(f'refusing to overwrite checkpoint step {step}')
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    atomic_save(state, online)
    atomic_save(dict(format='driftaudio-resume-v1', step=step, model=state,
                     optimizer=optimizer.state_dict(), bank=bank.state_dict(),
                     rng=rng_state(), signature=signature), resume)
    atomic_json(dict(step=step, online=online.name, online_sha256=sha256(online),
                     resume=resume.name, resume_sha256=sha256(resume)), out / f'checkpoint_step{step:06d}.json')


def train(config, output, resume=None, max_steps=None):
    steps = config['max_steps'] if max_steps is None else max_steps
    if steps < 1 or steps > config['max_steps']:
        raise ValueError('max-steps must be within the declared training horizon')
    device = torch.device('cuda:0')
    if not torch.cuda.is_available():
        raise RuntimeError('production backend requires CUDA; CPU unit tests do not')
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    q = 64
    paths = config['paths']
    dataset = CachedDataset(paths['train_manifest'], paths['npz_root'])
    plan = validate_plan(paths['query_plan'], dataset, steps, q)
    seed_all(int(plan['noise_seed_base']))
    signature = {k: sha256(paths[k]) for k in ['train_manifest', 'query_plan', 'bank', 'initialization', 'panns', 'vae', 'vocoder']}
    signature['recipe'] = 'panns-marginal-v2'
    signature['variant'] = 'full-train-real2048-resampled-per-step-v1'
    signature['full_real_pool'] = sha256(paths['full_real_pool'])
    signature['real_sampling_seed'] = SEED
    package = Path(__file__).resolve().parent
    signature['source_sha256'] = {str(p.relative_to(package)): sha256(p) for p in package.rglob('*.py')}
    # Bank bundles contain tensor data and JSON-like metadata only.
    targets = torch.load(paths['bank'], map_location='cpu', weights_only=True)
    if targets['initialization_sha256'] != signature['initialization']:
        raise ValueError('model reference bank was built from another initialization')
    if targets.get('train_manifest_sha256') != signature['train_manifest']:
        raise ValueError('bank and training manifest do not match')
    if targets['model'].shape != (1024, 2048) or targets['real'].shape != (2048, 2048):
        raise ValueError('recipe requires model1024, legacy real2048 and PANNs2048')
    model, decoder, encoder = load_models(config, device)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = make_optimizer(parameters, device)
    bank = ModelBank(targets['model'].to(device))
    pool = torch.load(paths['full_real_pool'], map_location='cpu', weights_only=True)
    assert pool['ids'] == [r['id'] for r in dataset.rows]
    assert pool['features'].shape == (len(dataset), 2048)
    assert pool['source_tsv_sha256'] == signature['train_manifest']
    assert torch.isfinite(pool['features']).all()
    real_pool = pool['features'].float().to(device)
    del pool
    start_step = 0
    output = Path(output).resolve()
    if resume:
        # Resume files are trusted local artifacts, including numpy/Python RNG.
        state = torch.load(resume, map_location='cpu', weights_only=False)
        if state['signature'] != signature and 'resume_source_migration' in config:
            previous = dict(state['signature'])
            assert previous['source_sha256'] == config['resume_source_migration'], 'Unexpected source migration'
            previous['source_sha256'] = signature['source_sha256']
            assert previous == signature, 'Non-source resume signature changed'
            state['signature'] = signature
        if state['format'] != 'driftaudio-resume-v1' or state['signature'] != signature:
            raise ValueError('resume format, assets or recipe mismatch')
        start_step = int(state['step'])
        if start_step >= steps:
            raise ValueError('requested horizon is already reached')
        model.load_state_dict(state['model'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        bank.load_state_dict(state['bank'])
        restore_rng(state['rng'])
        del state
        output.mkdir(parents=True, exist_ok=True)
    else:
        output.mkdir(parents=True, exist_ok=False)
    atomic_json(dict(config=config, signature=signature, start_step=start_step, target_step=steps,
                     torch_version=torch.__version__), output / f'run_config_from{start_step:06d}.json')
    log_path = output / f'train_from{start_step:06d}.jsonl'
    checkpoints = set(config['checkpoint_steps']) | {steps}
    with log_path.open('x') as log:
        completed_step = start_step
        for step in range(start_step + 1, steps + 1):
            if stop_requested(config): break
            entry = plan['entries'][step - 1]

            def generate(i):
                batch, noise = batch_and_noise(dataset, entry['query_indices'][i:i+1],
                                               entry['noise_seed_slots'][i:i+1], entry['noise_seed_base'], step, device)
                return generate_audio(model, decoder, batch, noise)

            indices = real_indices(step, len(real_pool))
            real = real_pool.index_select(0, indices.to(device))
            row = drift_step(generate, encoder, bank, real, optimizer, parameters, step, device, q)
            row.update(real_pool_count=len(real_pool), real_reference_count=len(indices),
                       real_sample_seed=SEED+step)
            completed_step = step
            log.write(json.dumps(row) + '\n')
            log.flush()
            if step == start_step + 1 or step % 5 == 0:
                print(json.dumps(row), flush=True)
            if step in checkpoints:
                save_checkpoint(output, step, model, optimizer, bank, signature)
                after_checkpoint(output, step, config)
    atomic_json(dict(status='complete', completed_step=completed_step), output/'TRAINING_SUMMARY.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--resume')
    parser.add_argument('--max-steps', type=int)
    args = parser.parse_args()
    train(load_config(args.config), args.output, args.resume, args.max_steps)


if __name__ == '__main__':
    main()
