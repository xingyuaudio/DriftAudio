# Method: marginal drifting in one frozen audio space

The trainable object is a conditional generator. Its audio distribution is refined
using a frozen PANNs representation. Text conditions generation, but does not
partition or weight the drifting field: distribution matching is marginal.

## One update

1. Generate Q audio samples and embed them with frozen PANNs.
2. Compare their embeddings with a detached 1024-entry generated FIFO and 2048 real features freshly sampled without replacement from the cached full training pool.
3. Compute a detached attraction/repulsion field in that feature space.
4. Differentiate the current audio features in the field direction, updating only
   the generator. Do not differentiate the field, references, or scale.
5. Insert the detached, pre-update query features into the generated-reference FIFO.

For normalized query feature `u = z / s` and detached field `V`, the surrogate is
`L = -2/(Q*d) * sum <u, stop_gradient(V)>`. Consequently, the feature cotangent
before dividing by Q is `-2*V/(d*s)`. The waveform VJP and generator replay simply
factor this chain rule to reduce memory; they do not introduce another objective.

## What is essential, and what is engineering?

The marginal field, frozen representation, detached references and generator
update define the method. Per-step real resampling approximates the full data
distribution without re-encoding audio; the FIFO approximates the generated
distribution with finite memory.
Microbatching, deterministic waveform replay, checkpointing and the supervisor
are implementation details. The fixed optimizer schedule is a training recipe,
not an additional loss.

There is one representation space, so there are no encoder mixture weights or
cross-space balancing rules. There is no teacher, MeanFlow anchor, FD loss,
semantic weighting, recency weighting, EMA inference model, or initialization-
specific objective. The two initialization configurations use the same recipe.

The field arithmetic lives in `driftaudio/field.py`; the update lives in
`driftaudio/step.py`.
