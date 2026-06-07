"""
chittakit.witness — the turīya monitor: a task-reward-invariant overseer.

We do NOT build "pure awareness" (that is metaphysics, see the design doc §12).
We build its functional shadow: a slowly/never-updated observer that checks the
system stays *the same agent* — consistent and calibrated — across operating
states and across time. It is never optimized by task reward, so it cannot be
gamed by it. Here it tracks representational drift on a fixed probe set.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class WitnessMonitor:
    def __init__(self, probe_inputs: torch.Tensor) -> None:
        # A fixed probe batch; the witness remembers the model's response to it.
        self.probe = probe_inputs
        self.reference: Optional[torch.Tensor] = None

    @torch.no_grad()
    def imprint(self, model: nn.Module) -> None:
        """Record the current 'self' — the model's representation of the probe."""
        model.eval()
        self.reference = self._features(model, self.probe).clone()

    @torch.no_grad()
    def drift(self, model: nn.Module) -> float:
        """How far has the agent moved from its imprinted self? (1 − mean cosine).
        0 = perfectly stable identity; larger = more drift."""
        if self.reference is None:
            return 0.0
        cur = self._features(model, self.probe)
        cos = F.cosine_similarity(cur, self.reference, dim=-1).mean().item()
        return 1.0 - cos

    @staticmethod
    def _features(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
        # Use backbone features if exposed, else the raw output.
        if hasattr(model, "features"):
            return model.features(x)
        return model(x)
