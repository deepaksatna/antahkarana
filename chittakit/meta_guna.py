"""
chittakit.meta_guna — a META-LEARNED guṇa controller.

The rule-based GunaController uses hand-set logic. Here the mapping
signals → (sattva, rajas, tamas) → (learning-rate, consolidation-strength) is a
small parameter matrix W that is LEARNED by evolution strategies (see
experiments/meta_train_guna.py), optimizing long-horizon continual performance.

Crucially it takes a measured *forgetting* signal as input, so it can learn the
adaptive policy the rule-based version lacked:  forget a lot → raise sattva
(protect hard);  forget little → raise rajas (learn fast, don't over-protect).
This is the fix for the Split-MNIST over-regularization.

Signals (each ~[0,1]): error, novelty, forgetting, energy.
"""
from __future__ import annotations

import torch


class MetaGunaController:
    N_SIG = 4  # error, novelty, forgetting, energy

    def __init__(self, base_lr: float, alpha_min: float = None, alpha_max: float = None,
                 beta_max: float = 1.0, temp: float = 1.0, params: torch.Tensor = None):
        self.base_lr = base_lr
        self.alpha_min = alpha_min if alpha_min is not None else base_lr * 0.1
        self.alpha_max = alpha_max if alpha_max is not None else base_lr * 2.0
        self.beta_max = beta_max
        self.temp = temp
        self.W = params.clone() if params is not None else self._init()

    @property
    def n_params(self) -> int:
        return 3 * (self.N_SIG + 1)  # 3 guṇas × (signals + bias) = 15

    def _init(self) -> torch.Tensor:
        # sensible prior (rows: sattva, rajas, tamas; cols: error,novelty,forgetting,energy,bias)
        W = torch.zeros(3, self.N_SIG + 1)
        W[0, 2] = 1.0                 # forgetting → sattva (protect)
        W[1, 1] = 1.0; W[1, 0] = 0.5  # novelty, error → rajas (explore)
        W[2, 3] = -1.0                # low energy → tamas (conserve)
        return W.flatten()

    def get_params(self) -> torch.Tensor:
        return self.W.clone()

    def set_params(self, p: torch.Tensor) -> None:
        self.W = p.clone()

    def step(self, error: float = 0.5, novelty: float = 0.5,
             forgetting: float = 0.0, energy: float = 1.0) -> dict:
        x = torch.tensor([error, novelty, forgetting, energy, 1.0])
        logits = (self.W.view(3, self.N_SIG + 1) @ x) / self.temp
        s, r, t = torch.softmax(logits, 0).tolist()
        alpha = self.alpha_min + (self.alpha_max - self.alpha_min) * (r * (1.0 - t))
        beta = self.beta_max * s
        return dict(sattva=s, rajas=r, tamas=t, alpha=alpha, beta=beta, prune=t)
