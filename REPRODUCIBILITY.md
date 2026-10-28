# Source provenance and reproduction limits

This release was assembled from the code snapshots attached to the two selected paper runs:

- MeanAudio-S-Full → online step 4400;
- Official FdAudio → online step 1800.

The archived source fingerprints confirm that both selected runs used identical `field.py`, `step.py`, `bank.py`, `real_sampling.py`, `optim.py`, and `backend.py` implementations. Their `train.py` files differ only in S-Full asynchronous validation and stop coordination plus migration bookkeeping. The original research-run snapshots are retained by the authors but are not part of this public package.

The selected paper runs shared the same ordered train manifest SHA256 `e97f12150601a04dfce35e7e78908fa2383c33f2ccee7f8840e585b88520e6ff`, query-plan SHA256 `6a6780ba95b22a6d9b652e278c6998ac37cdb40d9e4245716615a0123f6df27d`, and full real-feature-pool SHA256 `4cbcdb7c60d0bd9deea1593103f48284c88bc9cb444f96077c7103ac786fd748`. They used separate generated-reference banks matched to their different initialization checkpoints. Full-pool real references were sampled at each step using CPU `randperm` with seed `2026092001 + step`; 2048 of 45,173 features were selected without replacement. The old fixed 2048 real-bank tensor remains in the bank artifact for compatibility but was **not** used as the positive reference set by either selected run.

Initialization SHA256: MeanAudio-S-Full `f90bfe1c1dd2785d584ecf64903e0ed765aadce86fd0441918d4cc06ff8f55ff`; official FdAudio `6b6cc80339671ce59312f41edf541f752904a13826519baab100190767090ee1`. The PANNs tensor-only weight SHA256 is `bfc5e91709a4ca4ec45dda3bbfc872b11c29f12b80ec8caaed05dbf90debee9d`.

## What the cleaned release changes

The release starts from the S-Full snapshot, retaining its optional asynchronous control path. The only changes inside `driftaudio/` are portability/safety: the full-real-pool shape check uses `len(dataset)` instead of the paper-specific 45,173; the extractor writes the required training-TSV SHA256 into its output; and source-migration handling is conditional on an explicit migration key. For the paper's 45,173-row inputs, these do not change the drifting field, sampling, gradients or optimizer update. Documentation and example configurations now describe the 4400/1800 selected checkpoints and per-step real resampling. The original run configs and source snapshots are retained by the authors but are not included in this package.

This release has **not** performed a full 4400/1800-step GPU rerun. The staged source passed CPU tests and command-line smoke checks, but fresh trajectories may differ if upstream assets, query plans, prepared banks, software stacks, or CUDA kernels differ. A new run must be evaluated independently before claiming numerical reproduction of the archived results.

The archived test877 results were used in development and are not untouched held-out-test evidence. For an independent generalization claim use a separately held-out dataset and checkpoint-selection protocol. Review upstream code, data and checkpoint licenses before public distribution; this directory excludes those assets.
