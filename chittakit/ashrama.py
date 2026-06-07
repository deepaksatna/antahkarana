"""
chittakit.ashrama — the developmental (lifelong) plasticity schedule.

The four āśramas (life-stages) become a curriculum on plasticity itself, giving
the "always enhanceable, childhood → old age" property:

    α(a) = α_min + (α_peak − α_min)·exp(−a / τ_dev)         # high young, decays
           + κ · 1[ shift > δ ]                             # critical period RE-OPENS

Three guarantees of always-enhanceability:
  (1) α_min > 0          — plasticity never reaches zero (the mind is never frozen)
  (2) re-open on novelty — distribution shift restores high plasticity locally
  (3) meta-learning      — (handled elsewhere) the learner improves the learner

Stages by experience-age a (NOT clock time):
    brahmacarya (student)   → gṛhastha (householder)
    → vānaprastha (mentor)  → saṃnyāsa (sage)
"""
from __future__ import annotations

from dataclasses import dataclass
import math


STAGES = ("brahmacarya", "gṛhastha", "vānaprastha", "saṃnyāsa")


@dataclass
class AshramaState:
    age: float
    stage: str
    envelope: float   # max plasticity available at this age (the developmental ceiling)
    reopened: bool    # did a critical period just re-open?


class AshramaSchedule:
    def __init__(
        self,
        alpha_min: float = 1e-4,
        alpha_peak: float = 1e-2,
        tau_dev: float = 3.0,        # how fast childhood plasticity decays (in age units)
        reopen_boost: float = 0.8,   # κ: fraction of α_peak restored on a shift
        shift_threshold: float = 0.3,
        stage_bounds=(2.0, 5.0, 8.0),  # age cutoffs between the four stages
    ) -> None:
        self.alpha_min = alpha_min
        self.alpha_peak = alpha_peak
        self.tau_dev = tau_dev
        self.reopen_boost = reopen_boost
        self.shift_threshold = shift_threshold
        self.stage_bounds = stage_bounds

    def stage_for(self, age: float) -> str:
        b0, b1, b2 = self.stage_bounds
        if age < b0:
            return STAGES[0]
        if age < b1:
            return STAGES[1]
        if age < b2:
            return STAGES[2]
        return STAGES[3]

    def envelope(self, age: float, shift: float = 0.0) -> AshramaState:
        """Developmental plasticity ceiling at this age, with novelty re-opening."""
        base = self.alpha_min + (self.alpha_peak - self.alpha_min) * math.exp(-age / self.tau_dev)
        reopened = shift > self.shift_threshold
        if reopened:
            base = base + self.reopen_boost * (self.alpha_peak - self.alpha_min)
        env = min(base, self.alpha_peak)            # never exceed the peak
        env = max(env, self.alpha_min)              # never below the floor  (guarantee 1)
        return AshramaState(age=age, stage=self.stage_for(age), envelope=env, reopened=reopened)


def effective_alpha(envelope: float, guna_alpha: float, alpha_min: float = 1e-4) -> float:
    """Combine the developmental ceiling (āśrama) with the moment-to-moment
    guṇa modulation. The āśrama says how plastic we *may* be; the guṇa says how
    plastic we *choose* to be now. The floor keeps a learning channel always open.
    """
    use = min(guna_alpha, envelope)
    return max(use, alpha_min)
