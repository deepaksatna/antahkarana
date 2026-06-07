"""
chittakit.pramana — the Pramāṇa Gate: validity for extended perception.

Divya-dṛṣṭi ("divine sight") in Indian thought is perception beyond the normal
senses — seeing the distant, the hidden, the future. But Nyāya epistemology only
admits it as *yogaja-pratyakṣa*: a VALID means of knowledge (pramāṇa), not fancy.
That distinction is the antidote to the AI failure mode of the same shape —
HALLUCINATION (confident perception of what isn't there).

The Pramāṇa Gate makes a model's extended-perception output a valid pramāṇa only
when its CALIBRATED confidence clears a threshold; otherwise it abstains
("na paśyāmi" — I do not see clearly). Calibration is temperature scaling
(Guo et al. 2017) so that confidence means what it says.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


class PramanaGate:
    def __init__(self, threshold: float = 0.85) -> None:
        self.threshold = float(threshold)
        self.T = 1.0  # temperature; 1.0 until calibrated

    def calibrate(self, logits: torch.Tensor, labels: torch.Tensor,
                  iters: int = 300, lr: float = 0.05) -> float:
        """Fit a single temperature on held-out (logits, labels) so that softmax
        confidence is calibrated. Returns the fitted temperature."""
        logits = logits.detach()
        T = torch.ones(1, requires_grad=True)
        opt = torch.optim.Adam([T], lr=lr)
        for _ in range(iters):
            opt.zero_grad()
            loss = F.cross_entropy(logits / T.clamp_min(1e-2), labels)
            loss.backward()
            opt.step()
        self.T = float(T.detach().clamp_min(1e-2))
        return self.T

    def judge(self, logits: torch.Tensor):
        """Return (accept_mask, confidence, prediction).

        accept = True where the calibrated confidence ≥ threshold (a valid
        pramāṇa). Elsewhere the system should abstain rather than hallucinate.
        """
        p = F.softmax(logits / self.T, dim=-1)
        conf, pred = p.max(dim=-1)
        accept = conf >= self.threshold
        return accept, conf, pred

    @staticmethod
    def expected_calibration_error(logits: torch.Tensor, labels: torch.Tensor,
                                   T: float = 1.0, bins: int = 10) -> float:
        """ECE — gap between confidence and accuracy. Lower = better calibrated
        = the 'sight' can be trusted as a pramāṇa."""
        p = F.softmax(logits / max(T, 1e-2), dim=-1)
        conf, pred = p.max(dim=-1)
        correct = (pred == labels).float()
        ece = 0.0
        for b in range(bins):
            lo, hi = b / bins, (b + 1) / bins
            m = (conf > lo) & (conf <= hi)
            if m.any():
                ece += m.float().mean().item() * abs(conf[m].mean().item() - correct[m].mean().item())
        return ece
