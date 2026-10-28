import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from driftaudio.io import sha256
from tools.convert_panns_checkpoint import convert


class PannsConversionTests(unittest.TestCase):
    def test_legacy_numpy_metadata_preserves_all_model_tensors(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'legacy.pt';out=Path(d)/'clean.pt'
            model={'weight':torch.randn(3,5),'count':torch.tensor(7)}
            torch.save({'model':model,'legacy_numpy_metadata':np.arange(4)},p)
            with self.assertRaises(ValueError):convert(p,'0'*64,out)
            self.assertFalse(out.exists())
            result=convert(p,sha256(p),out)
            self.assertTrue(result['all_tensors_bitwise_equal'])
            loaded=torch.load(out,weights_only=True)['model']
            for k,v in model.items():self.assertTrue(torch.equal(v,loaded[k]))
            with self.assertRaises(FileExistsError):convert(p,sha256(p),out)
