import hashlib
import json
import os
from pathlib import Path

import torch


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(value, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(temporary, path)


def atomic_save(value, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.partial')
    torch.save(value, temporary)
    os.replace(temporary, path)


def load_config(path):
    path = Path(path).resolve()
    config = json.loads(path.read_text())
    for key, value in config['paths'].items():
        if value is not None:
            config['paths'][key] = str((path.parent / os.path.expandvars(value)).resolve())
    return config


def rng_state():
    import random
    import numpy as np
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    import random
    import numpy as np
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda']:
        torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda']])
