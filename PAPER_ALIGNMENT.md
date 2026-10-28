# Alignment with arXiv:2609.27598v1

This package targets the **main method**, not the later 1024-fresh/no-FIFO experiment. The paper version is [DriftAudio: Marginal Drifting for Distributional Post-Training of One-Step Text-to-Audio Generators](https://arxiv.org/abs/2609.27598), submitted 23 September 2026.

| Paper component | This package |
|---|---|
| Conditional one-step generator | Pinned FdAudio/MeanAudio backend; only generator weights are updated |
| Frozen audio representation | PANNs Cnn14, 2048-dimensional features |
| Real references | 2048 features sampled without replacement each optimizer step from a cached full training pool |
| Generated references | 1024 detached features in a FIFO bank, initialized from the matching starting generator and refreshed with 64 pre-update query features per step |
| Current query batch | 64 generated audios per optimizer update |
| Drifting field | `driftaudio/field.py`; attraction and repulsion both retained |
| Update and memory strategy | `driftaudio/step.py`; detached field, exact waveform replay, frozen encoder/decoder, generator gradient only |
| Training driver | `driftaudio/train.py`; the two JSON templates differ in initialization and horizon, not in the objective |
| Generation and metrics | `driftaudio/generate.py`, `driftaudio/evaluate.py`, and `EVALUATION.md` |

The paper reports AdamW learning rate approximately `1e-6`. The exact archived recipe rises to `1.05e-6` at step 35, cosine-decays to `2.1e-7` at step 75, then stays at `2.1e-7`; weight decay is `1e-6` and global gradient clipping is 1.0. The same exact schedule applies to both branches. No MeanFlow anchor, teacher, FD auxiliary loss, second audio encoder, or 1024-fresh replacement is used in the paper checkpoints.

| Initialization | Selected online step | Checkpoint SHA256 | Test877 FAD / FD / KL / IS / CLAP |
|---|---:|---|---|
| MeanAudio-S-Full | 4400 | `e7479d643ecac72e5b95e13c576b993725a15d2bc99d9ae975e4766fc33f066c` | 1.110951 / 12.714919 / 1.207199 / 9.833758 / 0.325344 |
| Official FdAudio | 1800 | `715d006ee3441b57e3c6f0998ead1b32cb9ac441710382a00b488fc3d71edb80` | 0.987234 / 12.609108 / 1.244305 / 10.168675 / 0.335118 |

These are archived run/evaluation receipts, **not** newly reproduced metrics from this cleaned code package. The release has passed CPU unit tests, but no full 4400/1800-step GPU replay or clean-environment install. The ordered cleaned training manifest has 45,173 clips; val445 and test877 audio are not bundled. The paper states validation FAD checkpoint selection. Archived release notes also record that test877 was inspected during method development, so it should not be described as a wholly untouched external test. See `REPRODUCIBILITY.md` and `RESULTS.md`.
