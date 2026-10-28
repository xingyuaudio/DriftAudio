# Generation and AudioCaps-877 evaluation

This protocol corresponds to the paper's 877 available AudioCaps test clips with one fixed official caption per clip. Audio, manifests, and caption-derived embeddings are not redistributed here. The caller must supply legally obtained audio and the exact `id,caption` TSV. Each generated filename must equal its manifest `id`.

## Dependencies

Paper evaluations used four local audio-only/tokenizer compatibility edits to the
pinned av-benchmark checkout. They lazily avoid video-model initialization when
video metrics are skipped and add a tokenizer compatibility alias; they do not
change the five reported metric formulas. Apply the included exact diff after
checking out the pinned commit:

```bash
git clone https://github.com/hkchengrex/av-benchmark.git third_party/av-benchmark
git -C third_party/av-benchmark checkout f351b9a6fc6abde746d5f8e1d4c47c883319cb41
git -C third_party/av-benchmark apply ../../patches/av-benchmark-audio-only.patch
```

Use this patch when reproducing the reported metrics; the unpatched evaluator
does not implement the same audio-only execution path.

Use the pinned [FdAudio](https://github.com/nobel861017/FdAudio) revision `6a0c9c79fd42c643b6474182b1105d596fc5e69d` for the one-step MeanAudio-compatible model and the pinned [av-benchmark](https://github.com/hkchengrex/av-benchmark) revision `f351b9a6fc6abde746d5f8e1d4c47c883319cb41` for metrics. The wrapper refuses a different revision. Install dependencies in a new, isolated environment after reviewing `requirements-cuda124.txt`; do not modify a shared environment merely to run this example.

The [released checkpoints](https://huggingface.co/xingyuchen-8/DriftAudio) contain generator state only. Supply the upstream VAE, vocoder, text encoders, and any other assets required by FdAudio.

## One-step generation

From the GitHub repository root:

```bash
python -m driftaudio.generate \
  --upstream third_party/FdAudio \
  --manifest data/test877.tsv \
  --expected-count 877 \
  --checkpoint assets/driftaudio_from_meanaudio_s_full_step4400.pth \
  --output outputs/from_meanaudio_test877
```

For the FdAudio-initialized branch, substitute `driftaudio_from_fdaudio_step1800.pth` and a new output directory. The wrapper fixes NFE=1, CFG=0.9, seed=42, duration=10 s, and validates exactly 877 mono 16-kHz FLAC files with 160,256 samples each. It refuses to overwrite an output directory.

## Metric evaluation

First create the exact text-candidate CSV from the same TSV:

```bash
python tools/make_text_csv.py --manifest data/test877.tsv --output data/test877_text.csv
python -m driftaudio.evaluate \
  --benchmark third_party/av-benchmark \
  --gt-audio data/test877_reference_audio \
  --gt-cache data/test877_reference_cache \
  --pred-audio outputs/from_meanaudio_test877 \
  --pred-cache outputs/from_meanaudio_metrics \
  --manifest data/test877.tsv \
  --text-csv data/test877_text.csv
```

Use a distinct `--pred-cache` for each model. Reuse `--gt-cache` only when its caption-binding receipt matches the manifest and text CSV; the wrapper checks this. It also checks audio IDs, cache IDs, finite metrics, and evaluator source hashes. Only use trusted local feature caches because the pinned upstream evaluator uses PyTorch pickle files.

Reported columns are FAD=`FD-VGG`, FD=`FD-PANN`, KL=`KL-PANNS-softmax`, IS=`ISC-PANNS-mean`, and CLAP=`LAION-CLAP-Score`. The wrapper fixes matmul TF32 off, cuDNN TF32 on, and cuDNN benchmark off. Its `driftaudio_metrics.json` and `evaluation_protocol.json` are the numerical result and audit receipt. Do not substitute the official 957-item AudioCaps TSV or a different caption selection when comparing with paper Table 1.

