# DriftAudio

Official PyTorch implementation of [*DriftAudio: Marginal Drifting for Distributional Post-Training of One-Step Text-to-Audio Generators*](https://arxiv.org/abs/2609.27598).

DriftAudio improves a pretrained one-step text-to-audio generator by matching its marginal audio distribution to real audio in a frozen PANNs feature space. The generator remains text-conditioned, and inference still takes one model step.

## Checkpoints

The [Hugging Face release](https://huggingface.co/xingyuchen-8/DriftAudio) contains the two paper-selected generator checkpoints:

| Initialization | Checkpoint | Training step |
|---|---|---:|
| MeanAudio-S-Full | [`driftaudio_from_meanaudio_s_full_step4400.pth`](https://huggingface.co/xingyuchen-8/DriftAudio/blob/main/driftaudio_from_meanaudio_s_full_step4400.pth) | 4,400 |
| FdAudio | [`driftaudio_from_fdaudio_step1800.pth`](https://huggingface.co/xingyuchen-8/DriftAudio/blob/main/driftaudio_from_fdaudio_step1800.pth) | 1,800 |

These files contain generator weights only. The upstream text encoders, VAE, and vocoder are required for inference. Place a downloaded checkpoint in `assets/` or pass its path with `--checkpoint`.

## Setup

Tested with Python 3.10 and PyTorch 2.6.0 (CUDA 12.4). Work in a new environment. The generator backend uses this pinned [FdAudio](https://github.com/nobel861017/FdAudio) revision:

```bash
git clone https://github.com/nobel861017/FdAudio.git third_party/FdAudio
git -C third_party/FdAudio checkout 6a0c9c79fd42c643b6474182b1105d596fc5e69d
python -m pip install -r requirements-cuda124.txt
python -m pip install --no-deps -e .
```

Install the remaining FdAudio runtime dependencies from its upstream instructions without replacing the pinned PyTorch packages. Download its [inference components](https://huggingface.co/kph68/FdAudio) into `third_party/FdAudio/weights/`; the generation script does not download them when a local DriftAudio checkpoint is supplied. This repository does not redistribute AudioCaps audio, upstream checkpoints, or text-embedding caches.

## Generate audio

Provide a tab-separated manifest with `id` and `caption` columns. For example, with a one-row `data/prompts.tsv` and a downloaded checkpoint:

```bash
python -m driftaudio.generate \
  --upstream third_party/FdAudio \
  --manifest data/prompts.tsv \
  --expected-count 1 \
  --checkpoint assets/driftaudio_from_meanaudio_s_full_step4400.pth \
  --output outputs/example
```

The wrapper uses the paper's one-step generation settings and writes 16-kHz FLAC files. Use a different output directory for each run. See `python -m driftaudio.generate --help` for the arguments.

## Post-train a model

Training uses 64 current samples, 2,048 real features resampled each step from the cached training pool, and a 1,024-feature generated FIFO bank. The same objective and optimizer settings apply to both initializations. Prepare your own training manifest and T5+CLAP caches as described in [the data format guide](docs/data.md), then set the asset paths in one of the `configs/` templates. A tensor-only PANNs Cnn14 checkpoint is required; `python -m tools.convert_panns_checkpoint --help` describes conversion from a trusted upstream checkpoint.

```bash
python -m driftaudio.extract_real \
  --manifest data/train.tsv \
  --panns assets/panns_tensor_only.pt \
  --output data/real_features.pt

python -m driftaudio.prepare \
  --config configs/from_fdaudio.json \
  --real-features data/real_features.pt \
  --output data/prepared_fd \
  --steps 1800
```

Set `paths.bank` and `paths.query_plan` in the config to the files created by `prepare`, then run:

```bash
python -m driftaudio.train \
  --config configs/from_fdaudio.json \
  --output outputs/train_fd
```

For MeanAudio-S-Full, use `configs/from_meanaudio_s_full.json` and prepare a separate bank from that initialization. Checkpoints are written to the selected output directory. The [method note](METHOD.md) describes the detached surrogate; the paper contains the full derivation.

## Evaluation

The paper evaluates 877 available AudioCaps test clips using one fixed caption per clip. The [evaluation protocol](EVALUATION.md) gives the manifest format and the pinned evaluator commands. The reported metrics are FAD (VGGish), FD/KL/IS (PANNs), and LAION-CLAP score.

| Model | FAD ↓ | FD ↓ | KL ↓ | IS ↑ | CLAP ↑ |
|---|---:|---:|---:|---:|---:|
| From MeanAudio-S-Full | 1.111 | 12.715 | 1.207 | 9.834 | 0.325 |
| From FdAudio | 0.987 | 12.609 | 1.244 | 10.169 | 0.335 |

These are the archived paper-run measurements, not results from a fresh training run of this release. The 877-clip set was inspected during development and should not be described as an untouched external test.

## Tests and citation

```bash
python -m unittest discover -s tests -v
```

```bibtex
@article{chen2026driftaudio,
  title={DriftAudio: Marginal Drifting for Distributional Post-Training of One-Step Text-to-Audio Generators},
  author={Chen, Xingyu and Ma, Fei and Zhao, Sipei},
  journal={arXiv preprint arXiv:2609.27598},
  year={2026}
}
```

The repository code is MIT-licensed. Model weights and upstream components have separate terms; in particular, the official FdAudio model card labels its weights [CC BY-NC-SA 4.0](https://github.com/nobel861017/FdAudio/blob/main/MODEL_CARD.md). See [third-party notices](THIRD_PARTY_NOTICES.md).
