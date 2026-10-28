"""Extract aligned PANNs train references; TSV needs id and audio columns."""
import argparse
import csv
from pathlib import Path

import torch
import torchaudio

from .backend import PANNsEncoder
from .io import atomic_save, sha256


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True); p.add_argument('--panns',required=True)
    p.add_argument('--output',required=True); a=p.parse_args()
    path=Path(a.manifest).resolve(); output=Path(a.output)
    if output.exists():raise FileExistsError(output)
    rows=list(csv.DictReader(path.open(),delimiter='\t'))
    encoder=PANNsEncoder(a.panns).cuda().eval()
    ids=[]; features=[]
    for row in rows:
        wav,sr=torchaudio.load(path.parent/row['audio'])
        wav=wav.mean(0)
        if sr!=16000:wav=torchaudio.functional.resample(wav,sr,16000)
        # Match the documented training representation convention explicitly.
        wav=wav[:160256]
        wav=torch.nn.functional.pad(wav,(0,160256-len(wav)))
        features.append(encoder(wav[None].cuda()).cpu()); ids.append(row['id'])
    output.parent.mkdir(parents=True,exist_ok=True)
    atomic_save(dict(ids=ids,features=torch.cat(features),sample_rate=16000,
                     num_samples=160256,source_tsv_sha256=sha256(path)),output)


if __name__=='__main__':main()
