"""
chittakit.guna — the Guṇa Controller: one interpretable (sattva, rajas, tamas)
signal that jointly modulates every learning hyperparameter.

This is the second novel module. Instead of tuning learning-rate, exploration,
consolidation and pruning separately, a single 3-vector on the 2-simplex
(s + r + t = 1) governs all of them, driven by the system's felt state:

    rajas  (r) ↑  when error & novelty & energy are high  →  explore, be plastic
    sattva (s) ↑  when learning is clean (low error/novelty) →  consolidate
    tamas  (t) ↑  when energy is low / fatigue high          →  conserve, prune

Outputs:
    alpha (plasticity)        = α_min + (α_max−α_min)·[ r·(1−t) ]
    tau   (exploration temp)  = τ0 · r
    beta  (consolidation)     = β0 · s
    prune (forget-rate)       = ρ0 · t
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import math


@dataclass
class GunaState:
    sattva: float
    rajas: float
    tamas: float
    alpha: float
    tau: float
    beta: float
    prune: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "sattva": self.sattva, "rajas": self.rajas, "tamas": self.tamas,
            "alpha": self.alpha, "tau": self.tau, "beta": self.beta, "prune": self.prune,
        }


def _softmax3(a: float, b: float, c: float):
    m = max(a, b, c)
    ea, eb, ec = math.exp(a - m), math.exp(b - m), math.exp(c - m)
    z = ea + eb + ec
    return ea / z, eb / z, ec / z


class GunaController:
    """Rule-based regime controller (meta-learnable later via meta-gradient/PBT).

    Signals are expected in [0, 1]:
        error   : 1 − recent accuracy (how badly we are doing)
        novelty : distribution shift since last task (e.g. normalized KL)
        reward  : recent task reward / success
        energy  : available compute/battery (1 = plenty, 0 = depleted)
    """

    def __init__(
        self,
        alpha_min: float = 1e-4,
        alpha_max: float = 1e-2,
        tau0: float = 1.0,
        beta0: float = 1.0,
        rho0: float = 1.0,
        temp: float = 1.5,
    ) -> None:
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.tau0 = tau0
        self.beta0 = beta0
        self.rho0 = rho0
        self.temp = temp

    def step(
        self,
        error: float = 0.5,
        novelty: float = 0.5,
        reward: float = 0.5,
        energy: float = 1.0,
    ) -> GunaState:
        e = _clip(error); n = _clip(novelty); r_ = _clip(reward); en = _clip(energy)

        # logits for the three guṇas (interpretable, hand-set; meta-learn later)
        rajas_logit  = (0.5 * e + 0.7 * n) * en            # struggle on novelty + have energy → explore
        sattva_logit = (0.6 * (1 - e) + 0.4 * (1 - n)) * en  # doing well, familiar → consolidate
        tamas_logit  = 0.8 * (1 - en) + 0.2 * (1 - r_)     # low energy / low reward → conserve

        s, r, t = _softmax3(
            sattva_logit / self.temp, rajas_logit / self.temp, tamas_logit / self.temp
        )

        alpha = self.alpha_min + (self.alpha_max - self.alpha_min) * (r * (1.0 - t))
        tau = self.tau0 * r
        beta = self.beta0 * s
        prune = self.rho0 * t
        return GunaState(s, r, t, alpha, tau, beta, prune)


def _clip(x: float) -> float:
    return max(0.0, min(1.0, float(x)))
