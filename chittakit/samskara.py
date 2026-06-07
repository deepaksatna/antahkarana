"""
chittakit.samskara — the saṃskāra memory: continual learning with growth + decay.

This is the core novel module of the Antaḥkaraṇa-Net. It extends Elastic Weight
Consolidation (EWC, Kirkpatrick et al. 2017) with the one thing EWC lacks: the
*decay* of importance. In the Yogic model a saṃskāra (latent impression) is NOT
eternal — unused grooves fade (YS 2.10–2.11). That decay is the missing half that
lets a lifelong learner stay *plastic* instead of ossifying after many tasks.

    Ω  ← (1 - λ) · Ω  +  γ · Fisher           # grow with use, decay with disuse
    L_consolidation = β · Σ Ω_i (θ_i − θ*_i)²  # protect what is proven important

Plain EWC is the special case λ = 0 (importance only ever accumulates).
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional

import torch
import torch.nn as nn


class SamskaraMemory:
    """Importance-weighted continual-memory wrapper around any nn.Module.

    Args:
        model:  the network whose parameters carry saṃskāras.
        grow:   γ — how strongly importance accrues with use (abhyāsa).
        decay:  λ — how fast unused importance fades (the non-eternal store).
                Set decay=0.0 to recover classic (online) EWC.
        normalize_fisher: rescale each task's Fisher to unit mean so that
                grow/decay are interpretable across tasks of different scale.
    """

    def __init__(
        self,
        model: nn.Module,
        grow: float = 1.0,
        decay: float = 0.1,
        normalize_fisher: bool = True,
        omega_cap: float = 10.0,
    ) -> None:
        self.model = model
        self.grow = float(grow)
        self.decay = float(decay)
        self.normalize_fisher = normalize_fisher
        # Cap per-parameter importance: Fisher is spiky, and an unbounded Ω makes
        # the quadratic penalty numerically stiff (β·Ω can blow up an SGD step).
        self.omega_cap = float(omega_cap)

        # Ω (importance) and θ* (reference snapshot), one tensor per trainable param.
        self.omega: Dict[str, torch.Tensor] = {}
        self.theta_star: Dict[str, torch.Tensor] = {}
        for name, p in self._named_params():
            self.omega[name] = torch.zeros_like(p)
            self.theta_star[name] = p.detach().clone()

    # ------------------------------------------------------------------ utils
    def _named_params(self) -> Iterable:
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                yield name, p

    # --------------------------------------------------------------- penalty
    def penalty(self, beta: float = 1.0) -> torch.Tensor:
        """The consolidation loss: β · Σ Ω_i (θ_i − θ*_i)².

        Add this to the task loss during training. β is supplied by the
        GunaController (sattva ↑ → consolidate harder).
        """
        device = next(self.model.parameters()).device
        loss = torch.zeros((), device=device)
        for name, p in self._named_params():
            om = self.omega[name].to(device)
            ts = self.theta_star[name].to(device)
            loss = loss + (om * (p - ts).pow(2)).sum()
        return beta * loss

    # ----------------------------------------------------------- consolidate
    @torch.enable_grad()
    def consolidate(
        self,
        data: Iterable,
        loss_fn: nn.Module,
        device: Optional[torch.device] = None,
        max_batches: Optional[int] = None,
        forward_fn=None,
    ) -> None:
        """Suṣupti (deep sleep): estimate Fisher importance on the just-learned
        task, then update Ω with growth + decay and snapshot θ*.

        `data` yields (inputs, targets); `loss_fn` is e.g. CrossEntropyLoss.
        `forward_fn(x) -> logits` lets multi-head models route to the right head
        (defaults to `self.model(x)`).
        """
        if device is None:
            device = next(self.model.parameters()).device
        if forward_fn is None:
            forward_fn = self.model
        self.model.eval()

        fisher: Dict[str, torch.Tensor] = {
            name: torch.zeros_like(p) for name, p in self._named_params()
        }
        n = 0
        for i, (x, y) in enumerate(data):
            if max_batches is not None and i >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            self.model.zero_grad(set_to_none=True)
            out = forward_fn(x)
            loss = loss_fn(out, y)
            loss.backward()
            for name, p in self._named_params():
                if p.grad is not None:
                    fisher[name] += p.grad.detach().pow(2) * x.size(0)
            n += x.size(0)

        if n == 0:
            return
        for name in fisher:
            fisher[name] /= n

        if self.normalize_fisher:
            total = sum(f.sum() for f in fisher.values())
            count = sum(f.numel() for f in fisher.values())
            mean = (total / count).clamp_min(1e-12)
            for name in fisher:
                fisher[name] = fisher[name] / mean

        # Ω ← (1 − λ)·Ω + γ·Fisher  (then cap) ; θ* ← current weights
        for name, p in self._named_params():
            om = (1.0 - self.decay) * self.omega[name] + self.grow * fisher[name]
            self.omega[name] = om.clamp_(max=self.omega_cap)
            self.theta_star[name] = p.detach().clone()

        self.model.zero_grad(set_to_none=True)

    @torch.enable_grad()
    def consolidate_losses(self, loss_iter) -> None:
        """Generic consolidation for non-classification settings (e.g. RL TD loss).
        `loss_iter` yields scalar losses (each still attached to the graph). We use
        the squared gradients as a Fisher proxy, then grow+decay Ω and snapshot θ*.
        """
        fisher: Dict[str, torch.Tensor] = {
            name: torch.zeros_like(p) for name, p in self._named_params()
        }
        count = 0
        for loss in loss_iter:
            self.model.zero_grad(set_to_none=True)
            loss.backward()
            for name, p in self._named_params():
                if p.grad is not None:
                    fisher[name] += p.grad.detach().pow(2)
            count += 1
        if count == 0:
            return
        for name in fisher:
            fisher[name] /= count
        if self.normalize_fisher:
            total = sum(f.sum() for f in fisher.values())
            num = sum(f.numel() for f in fisher.values())
            mean = (total / num).clamp_min(1e-12)
            for name in fisher:
                fisher[name] = fisher[name] / mean
        for name, p in self._named_params():
            om = (1.0 - self.decay) * self.omega[name] + self.grow * fisher[name]
            self.omega[name] = om.clamp_(max=self.omega_cap)
            self.theta_star[name] = p.detach().clone()
        self.model.zero_grad(set_to_none=True)

    # --------------------------------------------------------------- metrics
    def importance_mass(self) -> float:
        """Total Ω across the network — a scalar 'how much is being protected'.
        Under classic EWC this only grows; under saṃskāra it can plateau/fall."""
        return float(sum(o.sum() for o in self.omega.values()))

    def plasticity_headroom(self) -> float:
        """Fraction of parameters with near-zero importance — i.e. still freely
        learnable. A proxy for how 'young'/plastic the network remains."""
        free, total = 0, 0
        for o in self.omega.values():
            free += int((o < 1e-3).sum())
            total += o.numel()
        return free / max(total, 1)
