import copy
import math
import tempfile
import unittest
from pathlib import Path

import torch

from driftaudio.bank import ModelBank
from driftaudio.field import residual_field, waveform_gradient
from driftaudio.optim import learning_rate
from driftaudio.io import atomic_save, restore_rng, rng_state


class CoreTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(11)

    def test_schedule(self):
        self.assertAlmostEqual(learning_rate(1),3e-8)
        self.assertAlmostEqual(learning_rate(35),1.05e-6)
        self.assertAlmostEqual(learning_rate(75),2.1e-7)
        self.assertEqual(learning_rate(1200),learning_rate(200))
        self.assertTrue(all(learning_rate(i+1)<learning_rate(i) for i in range(35,75)))
        with self.assertRaises(ValueError):learning_rate(0)

    def test_permutation_and_detachment(self):
        q=torch.randn(8,5,requires_grad=True); m=torch.randn(16,5,requires_grad=True); r=torch.randn(32,5,requires_grad=True)
        f,s=residual_field(q,m,r)
        shuffled,t=residual_field(q,m[torch.randperm(16)],r[torch.randperm(32)])
        torch.testing.assert_close(f,shuffled,rtol=1e-6,atol=1e-6)
        self.assertAlmostEqual(s,t)
        self.assertFalse(f.requires_grad)
        f.sum().item()
        self.assertIsNone(m.grad)
        with self.assertRaises(FloatingPointError):residual_field(q*float('nan'),m,r)

    def test_fifo_wrap_and_restore(self):
        b=ModelBank(torch.zeros(5,3)); x=torch.arange(9.).reshape(3,3).requires_grad_(True)
        b.push(x); b.push(x+10)
        self.assertEqual(b.ptr,1); self.assertEqual(b.total_inserted,6)
        torch.testing.assert_close(b.values[0],x[2]+10)
        self.assertFalse(b.values.requires_grad)
        c=ModelBank(torch.zeros(5,3)); c.load_state_dict(b.state_dict())
        b.push(x); c.push(x)
        torch.testing.assert_close(b.values,c.values,rtol=0,atol=0)

    def test_replay_vjp_matches_direct_surrogate(self):
        model=torch.nn.Linear(4,12); encoder=torch.nn.Sequential(torch.nn.Linear(12,6),torch.nn.Tanh())
        encoder.requires_grad_(False)
        inputs=torch.randn(8,4); real=torch.randn(16,6); historical=torch.randn(10,6)
        cached=torch.cat([model(x[None]).detach() for x in inputs])
        force,scale=residual_field(encoder(cached),historical,real)
        gradient=waveform_gradient(encoder,cached,force,scale,'cpu',microbatch=2)
        model.zero_grad()
        for i in range(8):
            (model(inputs[i:i+1])*gradient[i:i+1]).sum().div(8).backward()
        replay=[p.grad.clone() for p in model.parameters()]
        model.zero_grad()
        waves=torch.cat([model(x[None]) for x in inputs])
        (-2*(encoder(waves)*force).sum()/(8*6*scale)).backward()
        for actual,param in zip(replay,model.parameters()):
            torch.testing.assert_close(actual,param.grad,rtol=2e-5,atol=2e-6)
        self.assertTrue(all(p.grad is None for p in encoder.parameters()))

    def test_resume_optimizer_bank_and_rng(self):
        model=torch.nn.Linear(3,2); optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4)
        bank=ModelBank(torch.randn(8,2))
        def step():
            x=torch.randn(2,3); y=model(x); optimizer.zero_grad(); y.square().mean().backward()
            optimizer.step(); bank.push(y.detach())
        step()
        snapshot=copy.deepcopy(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),bank=bank.state_dict(),rng=rng_state()))
        step(); expected=copy.deepcopy(model.state_dict()); expected_bank=bank.values.clone()
        with tempfile.TemporaryDirectory() as folder:
            f=Path(folder)/'resume.pt'; atomic_save(snapshot,f)
            saved=torch.load(f,weights_only=False)
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        bank.load_state_dict(saved['bank']); restore_rng(saved['rng']); step()
        for k,v in model.state_dict().items():torch.testing.assert_close(v,expected[k],rtol=0,atol=0)
        torch.testing.assert_close(bank.values,expected_bank,rtol=0,atol=0)


if __name__=='__main__':unittest.main()
