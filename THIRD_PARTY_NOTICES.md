# Third-party components

- **FdAudio / MeanAudio implementation**: https://github.com/nobel861017/FdAudio
  at `6a0c9c79fd42c643b6474182b1105d596fc5e69d`. Used as an external dependency.
  Its MIT notice is retained under `licenses/FdAudio.txt`.
- **av-benchmark / PANNs**: https://github.com/hkchengrex/av-benchmark
  at `f351b9a6fc6abde746d5f8e1d4c47c883319cb41`. The minimal Cnn14/ConvBlock and
  helper source is vendored under `driftaudio/vendor`. Its MIT license is retained
  there and under `licenses/av-benchmark.txt`. `SOURCE_PROVENANCE.json` records the
  original file SHA256 and local changes: remove constructor downloads/weight I/O;
  replace in-place ReLU calls with non-inplace equivalents. Architecture and forward
  arithmetic are otherwise retained. PANNs upstream: https://github.com/qiuqiangkong/audioset_tagging_cnn
- **PyTorch, torchaudio, torchlibrosa and model dependencies** are external packages
  and retain their respective licenses.

No third-party model weights or audio data are redistributed. No claim is made
that code licensing grants redistribution rights for those assets. The new
DriftAudio glue/core code is provided under the repository MIT license.
