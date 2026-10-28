"""Run pinned official generation with an explicit manifest, without editing upstream."""
import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

from .io import atomic_json, sha256

UPSTREAM_COMMIT = '6a0c9c79fd42c643b6474182b1105d596fc5e69d'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--upstream',required=True); p.add_argument('--manifest',required=True)
    p.add_argument('--checkpoint',required=True); p.add_argument('--output',required=True)
    p.add_argument('--expected-count',type=int,default=877)
    a=p.parse_args()
    upstream=Path(a.upstream).resolve(); manifest=Path(a.manifest).resolve()
    checkpoint=Path(a.checkpoint).resolve(); output=Path(a.output).resolve()
    if not checkpoint.is_file():raise FileNotFoundError(checkpoint)
    commit=subprocess.check_output(['git','-C',str(upstream),'rev-parse','HEAD'],text=True).strip()
    if commit != UPSTREAM_COMMIT:raise ValueError('generation requires the pinned FdAudio revision')
    if subprocess.check_output(['git','-C',str(upstream),'diff','HEAD','--','eval.py','meanaudio'],text=True):
        raise ValueError('upstream generation code has local modifications')
    rows=list(csv.DictReader(manifest.open(),delimiter='\t'))
    ids=[row['id'] for row in rows]
    if len(rows)!=a.expected_count or len(set(ids))!=len(ids):raise ValueError('manifest count/IDs mismatch')
    if any(not x or '/' in x or '\\' in x or x in ('.','..') for x in ids):raise ValueError('unsafe audio ID')
    source=(upstream/'eval.py').read_text()
    old="eval_file =  './sets/test-audiocaps.tsv'"
    if source.count(old)!=1:raise ValueError('unexpected upstream manifest assignment')
    source=source.replace(old, f'eval_file = {str(manifest)!r}')
    output.mkdir(parents=True,exist_ok=False)
    atomic_json(dict(checkpoint_sha256=sha256(checkpoint),manifest_sha256=sha256(manifest),
                     commit=commit,seed=42,cfg_strength=.9,nfe=1,duration=10,full_precision=True),output/'generation_protocol.json')
    sys.path.insert(0,str(upstream)); os.chdir(upstream)
    sys.argv=[str(upstream/'eval.py'),'--model_path',str(checkpoint),'--output',str(output),
              '--cfg_strength','0.9','--encoder_name','t5_clap','--duration','10','--use_rope',
              '--text_c_dim','512','--num_steps','1','--use_meanflow','--full_precision','--seed','42']
    exec(compile(source,str(upstream/'eval.py'),'exec'),{'__name__':'__main__','__file__':str(upstream/'eval.py')})
    import torchaudio
    audio=list(output.glob('*.flac'))
    if {x.stem for x in audio}!=set(ids):raise RuntimeError('generated IDs differ from manifest')
    for f in audio:
        info=torchaudio.info(f)
        if (info.sample_rate,info.num_channels,info.num_frames)!=(16000,1,160256):
            raise RuntimeError(f'audio shape mismatch: {f.name}')
    atomic_json(dict(status='complete',count=len(audio)),output/'generation_complete.json')


if __name__=='__main__':main()
