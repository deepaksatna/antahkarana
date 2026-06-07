"""
chittakit.klesha — the affliction monitor: turns the five kleśas into measurable
quantities to watch and (optionally) regularize. Here we expose the most useful
one for a classifier: avidyā ≈ epistemic uncertainty (predictive entropy), which
drives active learning and flags over-confidence.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def avidya_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Predictive entropy H[p(y|x)] — high = the model 'does not know' (avidyā).
    Returned per-sample; mean it for a batch-level ignorance signal in [0, log K]."""
    p = F.softmax(logits, dim=-1)
    logp = F.log_softmax(logits, dim=-1)
    return -(p * logp).sum(dim=-1)


def normalized_avidya(logits: torch.Tensor) -> float:
    """Mean predictive entropy normalized to [0, 1] (divide by log num_classes).
    A drop-in 'novelty/uncertainty' signal for the GunaController."""
    k = logits.size(-1)
    h = avidya_entropy(logits).mean().item()
    return h / max(torch.log(torch.tensor(float(k))).item(), 1e-8)
