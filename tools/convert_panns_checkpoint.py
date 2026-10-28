"""Convert a SHA-verified trusted legacy PANNs checkpoint to tensor-only form.

The explicit hash is mandatory because legacy pickle loading may execute code.
Never use this on a checkpoint whose source is not trusted.
"""
import argparse
from pathlib import Path
import torch
from driftaudio.io import sha256, atomic_save, atomic_json


def convert(source, expected_sha256, output):
    source=Path(source);output=Path(output)
    if output.exists():raise FileExistsError(output)
    actual=sha256(source)
    if len(expected_sha256)!=64 or actual!=expected_sha256:
        raise ValueError('trusted checkpoint SHA256 mismatch; refusing pickle load')
    old=torch.load(source,map_location='cpu',weights_only=False)
    state=old['model']
    if not state or not all(isinstance(k,str) and isinstance(v,torch.Tensor) for k,v in state.items()):
        raise ValueError('PANNs model state is not tensor-only')
    if not all(torch.isfinite(v).all() for v in state.values() if v.is_floating_point()):
        raise ValueError('non-finite pretrained weights')
    output.parent.mkdir(parents=True,exist_ok=True)
    atomic_save({'model':{k:v.detach().cpu().contiguous() for k,v in state.items()}},output)
    reloaded=torch.load(output,map_location='cpu',weights_only=True)['model']
    assert state.keys()==reloaded.keys()
    for k,v in state.items():
        assert v.dtype==reloaded[k].dtype and v.shape==reloaded[k].shape
        assert torch.equal(v,reloaded[k]),k
    record=dict(source=str(source.resolve()),source_sha256=actual,output=str(output.resolve()),
output_sha256=sha256(output),tensor_count=len(state),all_tensors_bitwise_equal=True,
                change='Discard non-model legacy metadata only; no numerical weight changes')
    atomic_json(record,output.with_suffix('.provenance.json'))
    return record


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trusted-input',required=True);p.add_argument('--expected-sha256',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();print(convert(a.trusted_input,a.expected_sha256,a.output))

if __name__=='__main__':main()
