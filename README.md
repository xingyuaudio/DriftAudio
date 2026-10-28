# DriftAudio: PANNs-only marginal drifting

Code accompanying [arXiv:2609.27598v1](https://arxiv.org/abs/2609.27598).
This repository contains training, generation, evaluation, and CPU tests.
The two paper-selected generator checkpoints will be uploaded to separate Hugging Face model repositories; links will be added here after publication.
See [PAPER_ALIGNMENT.md](PAPER_ALIGNMENT.md) for the paper-to-code map and
[EVALUATION.md](EVALUATION.md) for the AudioCaps-877 evaluation protocol.

Research code for post-training a one-step text-to-audio generator. The **same drifting objective and optimizer recipe** are used from MeanAudio-S-Full and from the official FdAudio checkpoint. The paper checkpoints are online step 4400 and online step 1800, respectively; see [RESULTS.md](RESULTS.md) for exact test877 results and hashes.

This directory was assembled from the two selected training-run source snapshots, not from the older fixed-real-reference draft. The shared `field.py`, `step.py`, `bank.py`, `real_sampling.py`, optimizer and model adapter have identical SHA256 values in both runs. The only training-source difference was S-Full's asynchronous validation/stop bookkeeping; it did not alter the field or update. The release retains that optional control path. [REPRODUCIBILITY.md](REPRODUCIBILITY.md) states the source and verification limits.

## Method and frozen recipe

At each optimizer step, generate Q=64 audio samples, embed them with frozen PANNs Cnn14 (2048-dimensional embedding), and calculate the marginal attraction/repulsion field. The real side is a fresh uniform, without-replacement sample of 2048 features from the entire cleaned training pool (45,173 in the paper runs); all real features are encoded once and cached. The generated-reference side is a detached FIFO bank of 1024 features, refreshed with the 64 **pre-update** query features after every step. The field, references, affinity and feature scale are detached. Gradients flow through the current audio, frozen PANNs and decoder to all trainable generator parameters. Endpoint replay and microbatching bound memory; they do not define another loss.

The same recipe applies to both initializations: AdamW (betas 0.9/0.95, eps 1e-6, weight decay 1e-6), global gradient clip 1, radius R=0.2, generator microbatch 1, encoder microbatch 8, BF16 generator/decoder, FP32 encoder and FP64 field. The LR rises to 1.05e-6 at step 35, cosine-decays to 2.1e-7 at step 75, then stays constant. One step is one complete Q64 optimizer update. There is no teacher, MeanFlow anchor, FD auxiliary loss, multi-encoder balance, semantic affinity, recency weighting, independent lambda or EMA inference checkpoint.

See [METHOD.md](METHOD.md) for the surrogate and [docs/data.md](docs/data.md) for the data contract. The `real` tensor retained in the portable bank file is a legacy schema field; **the update uses the resampled `full_real_pool`, not that fixed subset**.

## Inputs and installation

The repository contains no audio, model weights, embeddings or copyrighted upstream code. Supply the MeanAudio-S-Full or official FdAudio initialization, VAE, vocoder and a converted tensor-only PANNs checkpoint. The paper runs used the same ordered cleaned 45,173-row AudioCaps training manifest, NPZ T5+CLAP caches and a precomputed full-pool PANNs feature file. Do not include validation/test audio in the training pool.
The pinned FdAudio `scripts/extract_audio_latents.sh` creates the expected per-row NPZ caches and ordered TSV. Keep that row order unchanged; see [docs/data.md](docs/data.md) for the schema.
Before `extract_real`, add an `audio` column to that TSV with paths relative to the TSV directory; the upstream cache-extraction TSV supplies `id` and `caption` but not raw-audio paths.

The inspected environment was Python 3.10, PyTorch 2.6.0+cu124. For a *new* environment only, review and use the pinned CUDA-12.4 requirements; do not install them into a shared existing environment without authorization. The model backend depends on the pinned FdAudio/MeanAudio upstream code:

```bash
git clone https://github.com/nobel861017/FdAudio.git third_party/FdAudio
git -C third_party/FdAudio checkout 6a0c9c79fd42c643b6474182b1105d596fc5e69d
python -m pip install -r requirements-cuda124.txt
python -m pip install --no-deps -e .
```

Check third-party licenses and checkpoint terms before redistribution. The repository itself does not auto-download weights. For a trusted legacy PANNs checkpoint, `python -m tools.convert_panns_checkpoint --help` documents the verified conversion to a tensor-only file.

## Prepare and train

Copy and edit the path-only templates in `configs/`. Their horizons and checkpoint intervals correspond to the selected paper steps, but do **not** imply that those endpoints were predetermined; checkpoints were selected after inspecting validation trajectories. A new run needs its own selection protocol. The templates are not exact replay recipes because the archived query plan, banks and feature caches are not redistributed.

```bash
CUDA_VISIBLE_DEVICES=0 python -m driftaudio.extract_real \
  --manifest data/train.tsv --panns assets/panns_tensor_only.pt \
  --output data/real_features.pt
CUDA_VISIBLE_DEVICES=0 python -m driftaudio.prepare \
  --config configs/from_fdaudio.json --real-features data/real_features.pt \
  --output data/prepared_fd --steps 1800
```

Point `full_real_pool` to `data/real_features.pt`, `bank` and `query_plan` to the prepared artifacts. Prepare the generated bank separately for S-Full: generated references must come from the same initialization as the model being trained. The two paper runs used the same query plan and real pool but separate generated banks.

```bash
CUDA_VISIBLE_DEVICES=0 python -m driftaudio.train \
  --config configs/from_fdaudio.json --output outputs/fd
CUDA_VISIBLE_DEVICES=0 python -m driftaudio.train \
  --config configs/from_meanaudio_s_full.json --output outputs/sfull
```

The trainer uses one GPU per run. The two initializations may run independently in parallel. It writes online and resumable checkpoints atomically and refuses to overwrite an existing checkpoint. For unstable SSH/VPN connections, use `tools/launch_detached.py --help` and inspect its exit-code file. Resume only from a trusted checkpoint created by this release.

## Evaluation and verification

The final comparison used the same AudioCaps-877 official-caption manifest and MeanAudio evaluator for all models: FAD=FD-VGG, FD=FD-PANN, KL=KL-PANNS-softmax, IS=ISC-PANNS-mean and CLAP=LAION-CLAP-Score. See `driftaudio.generate --help` and `driftaudio.evaluate --help`. The 877 set was used during development; it is not an untouched held-out test for checkpoint selection.

```bash
OMP_NUM_THREADS=2 python -m unittest discover -s tests -v
python -m compileall -q driftaudio tools
```

The included tests check the field, detached surrogate, replay, FIFO, sampling and checkpoint path on CPU. They do not constitute a complete GPU rerun of the paper horizons. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) before attributing the archived metrics to a fresh run of this cleaned release.
