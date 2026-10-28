"""Pin evaluator numerical defaults and retain official metric definitions."""
import argparse
import csv
import json
import math
import os
import runpy
import subprocess
import sys
from pathlib import Path

import torch

from .io import atomic_json, sha256

METRICS={'FAD':'FD-VGG','FD':'FD-PANN','KL':'KL-PANNS-softmax','IS':'ISC-PANNS-mean','CLAP':'LAION-CLAP-Score'}


def validate_inputs(manifest, text_csv, gt_audio, pred_audio):
    with Path(manifest).open(newline='') as f:
        rows = list(csv.DictReader(f, delimiter='\t'))
    ids = [r['id'] for r in rows]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('manifest IDs must be nonempty and unique')
    with Path(text_csv).open(newline='') as f:
        texts = list(csv.DictReader(f))
    if len(texts) != len(ids) or {r['name']: r['caption'] for r in texts} != {r['id']: r['caption'] for r in rows}:
        raise ValueError('text CSV must exactly match manifest IDs and captions')
    for folder in [gt_audio, pred_audio]:
        files = [p for p in Path(folder).iterdir() if p.suffix.lower() in ('.wav', '.flac')]
        if len(files) != len(ids) or {p.stemfor p in files} != set(ids):
            raise ValueError(f'audio IDs differ from manifest: {folder}')
    return set(ids)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--benchmark',required=True); p.add_argument('--gt-audio',required=True)
    p.add_argument('--gt-cache',required=True); p.add_argument('--pred-audio',required=True)
    p.add_argument('--pred-cache',required=True)
    p.add_argument('--manifest',required=True)
    p.add_argument('--text-csv',required=True)
    a=p.parse_args()
    benchmark=Path(a.benchmark).resolve()
    commit=subprocess.check_output(['git','-C',str(benchmark),'rev-parse','HEAD'],text=True).strip()
    if commit!='f351b9a6fc6abde746d5f8e1d4c47c883319cb41':raise ValueError('benchmark revision mismatch')
    expected = validate_inputs(a.manifest, a.text_csv, a.gt_audio, a.pred_audio)
    # A commit alone does not identify a locally patched evaluator. Record both.
    sources = {str(p.relative_to(benchmark)): sha256(p) for p in benchmark.rglob('*.py') if '.git' not in p.parts}
    binding = dict(manifest_sha256=sha256(a.manifest), text_csv_sha256=sha256(a.text_csv),
                   extractor_sha256=sha256(benchmark/'extract_text.py'))
    gt_cache = Path(a.gt_cache).resolve()
    gt_cache.mkdir(parents=True, exist_ok=True)
    receipt = gt_cache/'caption_binding.json'
    if receipt.exists():
        saved = json.loads(receipt.read_text())
        if saved['inputs'] != binding or saved['text_sha256'] != sha256(gt_cache/'clap_laion_text.pth'):
            raise ValueError('text cache provenance mismatch')
    else:
        if (gt_cache/'clap_laion_text.pth').exists():
            raise ValueError('unbound text cache: use a fresh cache, do not guess caption provenance')
        subprocess.run([sys.executable, str(benchmark/'extract_text.py'), '--text_csv',
                        str(Path(a.text_csv).resolve()), '--output_cache_path', str(gt_cache)],
                       cwd=benchmark, check=True)
        atomic_json(dict(inputs=binding, text_sha256=sha256(gt_cache/'clap_laion_text.pth')), receipt)
    if set(torch.load(gt_cache/'clap_laion_text.pth', map_location='cpu', weights_only=False)) != expected:
        raise ValueError('CLAP text cache IDs mismatch')
    if os.environ.get('NVIDIA_TF32_OVERRIDE')=='0' or os.environ.get('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE') in ('1','true','True'):
        raise ValueError('TF32 environment override conflicts with evaluation protocol')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=True
    torch.backends.cudnn.benchmark=False
    resolved={k:str(Path(getattr(a,k)).resolve()) for k in ['gt_audio','gt_cache','pred_audio','pred_cache']}
    if Path(resolved['pred_cache']).exists():raise FileExistsError('use a fresh prediction cache')
    sys.path.insert(0,str(benchmark)); os.chdir(benchmark)
    sys.argv=[str(benchmark/'evaluate.py')]
    for key,value in resolved.items():sys.argv += ['--'+key,value]
    sys.argv += ['--audio_length','10','--recompute_pred_cache','--skip_video_related']
    runpy.run_path(str(benchmark/'evaluate.py'),run_name='__main__')
    raw=json.loads((Path(resolved['pred_cache'])/'output_metrics.json').read_text())
    result={label:raw[key] for label,key in METRICS.items()}
    if not all(math.isfinite(float(v)) for v in result.values()):
        raise FloatingPointError('nonfinite evaluation metric')
    for folder, names in [(gt_cache, ['pann_features.pth', 'vggish_features.pth']),
                          (Path(resolved['pred_cache']), ['pann_features.pth', 'vggish_features.pth', 'clap_laion_audio.pth'])]:
        for name in names:
            if set(torch.load(folder/name, map_location='cpu', weights_only=False)) != expected:
                raise ValueError(f'feature cache IDs mismatch: {folder/name}')
    atomic_json(dict(commit=commit, source_sha256=sources, caption_binding=binding,
                     count=len(expected), metrics=METRICS), Path(resolved['pred_cache'])/'evaluation_protocol.json')
    atomic_json(result,Path(resolved['pred_cache'])/'driftaudio_metrics.json')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
