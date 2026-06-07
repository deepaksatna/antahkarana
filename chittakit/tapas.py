"""
chittakit.tapas — Tapas (तपस्): concentrated, deliberate effort allocated where
transformation is hardest.

Tapas in the tradition literally means "heat" — the disciplined energy expended to
transform (Rig Veda: by tapas the cosmos is generated; Yoga Sutra 2.1: tapas is the
first limb of kriyā-yoga). It is more than repetition (abhyāsa): it is *intense,
chosen effort directed at what most needs it*.

In ML terms: given a FIXED effort/compute budget and a per-item "need" signal
(e.g. current error on each past task), allocate the budget toward need rather
than uniformly — "apply the heat where the ore is hardest to melt." This is the
engine that drives focused practice (samyama) and, with enough of it, the emergence
of new capability (siddhi).
"""
from __future__ import annotations

from typing import List
import math


class TapasController:
    def __init__(self, temperature: float = 0.5, floor: float = 0.05) -> None:
        # temperature: low → concentrate hard on the neediest; high → near-uniform.
        # floor: every item keeps at least this fraction of an equal share (so a
        #        task is never wholly abandoned — discipline, not neglect).
        self.temperature = float(temperature)
        self.floor = float(floor)

    def allocate(self, budget: int, needs: List[float]) -> List[int]:
        """Split an integer `budget` across items in proportion to softmax(need).

        needs: per-item difficulty in [0, 1] (e.g. 1 − accuracy). Higher → more tapas.
        Returns a list of non-negative ints summing to `budget`.
        """
        k = len(needs)
        if k == 0 or budget <= 0:
            return [0] * k

        # softmax over needs, blended with a uniform floor
        m = max(needs)
        ex = [math.exp((n - m) / max(self.temperature, 1e-6)) for n in needs]
        z = sum(ex)
        prop = [e / z for e in ex]
        unif = 1.0 / k
        prop = [(1 - self.floor * k) * p + self.floor * unif * k * unif for p in prop]
        # renormalize (guard)
        s = sum(prop) or 1.0
        prop = [p / s for p in prop]

        # to integers that sum exactly to budget (largest-remainder method)
        raw = [budget * p for p in prop]
        base = [int(math.floor(r)) for r in raw]
        rem = budget - sum(base)
        order = sorted(range(k), key=lambda i: raw[i] - base[i], reverse=True)
        for i in range(rem):
            base[order[i % k]] += 1
        return base
