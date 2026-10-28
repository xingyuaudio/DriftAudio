"""Exercise the production update and checkpoint path, not a duplicate loop."""
import copy
import tempfile
import unittest
from pathlib import Path

import torch

from driftaudio.bank import ModelBank
from driftaudio.optim import make_optimizer
from driftaudio.step import drift_step
from driftaudio.train import save_checkpoint


class TrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(41)

    def test_update_checkpoint_and_resume(self):
        model = torch.nn.Linear(4, 12)
        encoder = torch.nn.Linear(12, 6).eval().requires_grad_(False)
        parameters = list(model.parameters())
        optimizer = make_optimizer(parameters, 'cpu')
        bank = ModelBank(torch.randn(1024, 6))
        real = torch.randn(2048, 6)
        inputs = torch.randn(64, 4)
        generate = lambda i: model(inputs[i:i + 1])
        initial = copy.deepcopy(model.state_dict())
        frozen = copy.deepcopy(encoder.state_dict())
        expected_entries = torch.cat([encoder(generate(i)).detach() for i in range(64)])
        row = drift_step(generate, encoder, bank, real, optimizer, parameters, 1, 'cpu')
        self.assertEqual(row['endpoint_replay_max_abs'], 0)
        self.assertTrue(any(not torch.equal(initial[k], v) for k, v in model.state_dict().items()))
        torch.testing.assert_close(bank.values[:64], expected_entries)
        for key, value in encoder.state_dict().items():
            torch.testing.assert_close(value, frozen[key], rtol=0, atol=0)
        with tempfile.TemporaryDirectory() as folder:
            save_checkpoint(Path(folder), 1, model, optimizer, bank, {'recipe': 'test'})
            saved = torch.load(Path(folder) / 'resume_step000001.pt', weights_only=False)
            drift_step(generate, encoder, bank, real, optimizer, parameters, 2, 'cpu')
            expected = copy.deepcopy(model.state_dict())
            expected_bank = bank.values.clone()
            model.load_state_dict(saved['model'])
            optimizer.load_state_dict(saved['optimizer'])
            bank.load_state_dict(saved['bank'])
            drift_step(generate, encoder, bank, real, optimizer, parameters, 2, 'cpu')
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
            torch.testing.assert_close(bank.values, expected_bank, rtol=0, atol=0)
            with self.assertRaises(FileExistsError):
                save_checkpoint(Path(folder), 1, model, optimizer, bank, {})

    def test_nondeterministic_replay_is_rejected(self):
        model = torch.nn.Linear(3, 8)
        encoder = torch.nn.Linear(8, 4).requires_grad_(False)
        parameters = list(model.parameters())
        with self.assertRaisesRegex(RuntimeError, 'replay mismatch'):
            drift_step(lambda i: model(torch.randn(1, 3)), encoder,
                       ModelBank(torch.randn(16, 4)), torch.randn(32, 4),
                       make_optimizer(parameters, 'cpu'), parameters, 1, 'cpu', 2, 1)

    def test_update_through_full_panns_architecture(self):
        from driftaudio.vendor.panns import Cnn14
        from driftaudio.backend import PANNsEncoder
        network = Cnn14(features_list=['2048'], sample_rate=16000, window_size=512,
                        hop_size=160, mel_bins=64, fmin=50, fmax=8000,
                        classes_num=527).eval().requires_grad_(False)
        # Exercise the production tensor adapter without downloading a checkpoint.
        class SyntheticPANNs(PANNsEncoder):
 def __init__(self):
                torch.nn.Module.__init__(self)
      self.model = network
        encoder = SyntheticPANNs().eval().requires_grad_(False)
        model = torch.nn.Linear(4, 16000)
        parameters = list(model.parameters())
        inputs = torch.randn(2, 4)
        before = model.weight.detach().clone()
     result = drift_step(lambda i: model(inputs[i:i + 1]), encoder,
                            ModelBank(torch.randn(16, 2048)), torch.randn(32, 2048),
                            make_optimizer(parameters, 'cpu'), parameters, 1, 'cpu', 2, 1)
        self.assertGreater(result['grad_norm_preclip'], 0)
        self.assertFalse(torch.equal(before, model.weight))
        self.assertTrue(all(p.grad is None for p in encoder.parameters()))


if __name__ == '__main__':
    unittest.main()
