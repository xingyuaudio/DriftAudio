# Data and artifact contracts

Training TSV requires a unique `id` per row. Its optional `npz` column names a file
relative to `npz_root`; otherwise row `i` uses `<i>.npz` (zero-based). The pinned
FdAudio `scripts/extract_audio_latents.sh` produces the per-row NPZ files and
ordered TSV. Each NPZ must contain `mean`, `text_features`, and
`text_features_c` from T5+CLAP preprocessing. Preserve row order; `caption` is
optional provenance and is not read by training.
For `extract_real`, add an `audio` column containing each audio file path relative to
the TSV directory. The upstream cache TSV does not supply this column. Preserve
exactly the same row order and IDs when adding it.

Evaluation TSV: unique `id`, `caption`. Supply one fixed caption per clip.
The generation wrapper rejects duplicate IDs, unsafe filenames, wrong counts,
missing audio, and incorrect sample rate/channel/frame count.

Full real-feature pool: `torch.save({'ids': [...], 'features': tensor[N,2048],
'source_tsv_sha256': SHA256}, path)`. The ordered IDs and SHA256 must equal the
training TSV order/file. At each optimizer step 2048 rows are sampled uniformly
without replacement using the fixed step-indexed seed. The optional raw-audio
extractor consumes an `audio` column, converts to mono16kHz and trims/pads to
160256 samples. This is an explicit new-data convention; archived preprocessing
must use archived features for exact replay.

Portable bank: `model` float32[1024,2048], `real` float32[2048,2048],
`initialization_sha256`, `train_manifest_sha256`, optionally ordered reference IDs.
The `real` tensor is retained for the historical bank schema but is not used as
the real side of the selected method: training reads `full_real_pool` instead.
Shared-initialization comparisons must use aligned query plans and real-pool
sampling rules. Generated banks must match each initialization.

Query plan: `noise_seed_base`, `query_count_per_step=64`, `steps`, `entries`.
Each entry contains64 `query_indices`,64 `noise_seed_slots`, and `noise_seed_base`.
At step s, slot j uses `noise_seed_base + s*10007 + j*1000003`, matching the
archived per-sample seeding convention. One row corresponds to a cached audio/text
pair. This trainer does not expand a clip to five captions implicitly.

The archived data protocol used 45,173 cleaned training rows and 42,101 eligible
queries after excluding the 1024 generated-bank and 2048 legacy real-bank rows
from the *query plan*. Those excluded rows remain eligible for per-step real
reference sampling from the full pool. Keep
validation/test YouTube IDs out of the training dataset before extracting caches.
The public package contains no AudioCaps audio, text-embedding cache or model weights.
