"""Persistent-training control only: no changes to sampler, loss or gradients."""
import json,time
from pathlib import Path
import torch
import torch.distributed as dist
def stop_requested(config):
    root=config.get('async_control')
    if not root:return False
    distributed=dist.is_initialized()
    rank=dist.get_rank() if distributed else 0
    stop=Path(root,'STOP.json').exists() if rank==0 else False
    if distributed:
        flag=torch.tensor([int(stop)],device='cuda:0',dtype=torch.int32)
        dist.broadcast(flag,src=0);stop=bool(flag.item())
    return stop
def after_checkpoint(output,step,config):
    root=config.get('async_control')
    if not root:return
    distributed=dist.is_initialized();rank=dist.get_rank() if distributed else 0
    if rank==0:
        from .io import atomic_json
        root=Path(root)
        receipt=json.loads(Path(output,f'checkpoint_step{step:06d}.json').read_text())
        atomic_json(dict(step=step,checkpoint=str(Path(output)/receipt['online']),receipt=str(Path(output,f'checkpoint_step{step:06d}.json'))),root/f'ready_{step:06d}.json')
        previous=step-config['async_interval']
        # At most one validation interval ahead; stay resident if evaluator lags.
        while previous>=config['async_first_eval'] and not (root/f'ack_{previous:06d}.json').exists() and not (root/'STOP.json').exists():
            time.sleep(1)
    if distributed:dist.barrier()
