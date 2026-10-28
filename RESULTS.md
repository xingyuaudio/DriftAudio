# Selected archived paper results

These values were measured for the two paper-selected checkpoints using the same 877-item official-caption evaluation protocol. They are **not** a new end-to-end result produced by this cleaned release. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for verification limits.

| Method | Online step | FAD ↓ | FD ↓ | KL ↓ | IS ↑ | CLAP ↑ |
|---|---:|---:|---:|---:|---:|---:|
| DriftAudio from MeanAudio-S-Full | 4400 | 1.1109511152 | 12.7149187836 | 1.2071985006 | 9.8337578159 | 0.3253435493 |
| DriftAudio from official FdAudio | 1800 | 0.9872339114 | 12.6091079833 | 1.2443051338 | 10.1686745153 | 0.3351176083 |

## Checkpoint integrity

| Branch | Weight file | SHA256 |
|---|---|---|
| MeanAudio-S-Full | `driftaudio_from_meanaudio_s_full_step4400.pth` | `e7479d643ecac72e5b95e13c576b993725a15d2bc99d9ae975e4766fc33f066c` |
| FdAudio | `driftaudio_from_fdaudio_step1800.pth` | `715d006ee3441b57e3c6f0998ead1b32cb9ac441710382a00b488fc3d71edb80` |

The checkpoint hashes were rechecked against the staged weight files. The metrics were transcribed from archived evaluation results. No checkpoint or generated audio is bundled in this source repository.

The 877-item AudioCaps set was used during method development and checkpoint selection. Do not describe it as an untouched held-out generalization test. PANNs is both the training feature space and the evaluator space for FD/KL/IS, so these metrics are not independent representation validation. No multi-seed significance claim is made.
