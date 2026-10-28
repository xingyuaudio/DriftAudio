"""Detached historical model features. Push pre-update features after each step."""
import torch


class ModelBank:
    def __init__(self, initial):
        if initial.ndim != 2 or len(initial) == 0 or not torch.isfinite(initial).all():
            raise ValueError("finite nonempty [M,D] bank required")
        self.values = initial.detach().float().clone()
        self.ptr = 0
        self.total_inserted = 0

    @torch.no_grad()
    def push(self, features):
        if features.ndim != 2 or features.shape[1] != self.values.shape[1] or not 0 < len(features) <= len(self.values):
            raise ValueError("invalid FIFO update")
        n = len(features)
        end = self.ptr + n
        first = min(n, len(self.values) - self.ptr)
        self.values[self.ptr:self.ptr + first].copy_(features[:first].detach())
        if first < n:
            self.values[:n - first].copy_(features[first:].detach())
        self.ptr = end % len(self.values)
        self.total_inserted += n

    def state_dict(self):
        return dict(values=self.values.detach().cpu(), ptr=self.ptr,
                    total_inserted=self.total_inserted)

    def load_state_dict(self, state):
        if state['values'].shape != self.values.shape or not 0 <= state['ptr'] < len(self.values):
            raise ValueError("incompatible FIFO state")
        self.values.copy_(state['values'])
        self.ptr = int(state['ptr'])
        self.total_inserted = int(state['total_inserted'])
