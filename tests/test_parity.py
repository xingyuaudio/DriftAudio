"""CPU parity against frozen excerpts, not end-to-end GPU equivalence."""
import unittest
import torch
from driftaudio.field import residual_field
from support import field_oracle


class ParityTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(37)

    def test_full_field_bitwise(self):
        query = torch.randn(64, 17)
        model = torch.randn(1024, 17)
        real = torch.randn(2048, 17)
        actual, scale = residual_field(query, model, real)
        expected, _, _, expected_scale = field_oracle.field_parts(query, model, real)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(scale, expected_scale)

    def test_removed_roundtrip_is_numerically_identity(self):
        for gradient in (torch.randn(64, 1600), torch.zeros(64, 1600)):
            expected = field_oracle.combine({'panns': gradient}, True)[0]
            torch.testing.assert_close(gradient, expected, rtol=3e-7, atol=1e-12)

    def test_panns_forward_and_waveform_gradient(self):
        from driftaudio.vendor.panns import Cnn14
        from support.panns_oracle import Cnn14 as Original
        config = dict(features_list=['2048', 'logits'], sample_rate=16000,
            window_size=512, hop_size=160, mel_bins=64, fmin=50,
        fmax=8000, classes_num=527)
        new = Cnn14(**config).eval().requires_grad_(False)
        old = Original(**config).eval().requires_grad_(False)
        old.load_state_dict(new.state_dict())
        wave = torch.randn(1, 16000).requires_grad_(True)
        expected = old(wave)['2048']
        cotangent = torch.randn_like(expected)
        expected_grad, = torch.autograd.grad(expected, wave, cotangent)
        actual = new(wave)['2048']
        actual_grad, = torch.autograd.grad(actual, wave, cotangent)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)
        self.assertTrue(all(p.grad is None for p in new.parameters()))


if __name__ == '__main__':
    unittest.main()
