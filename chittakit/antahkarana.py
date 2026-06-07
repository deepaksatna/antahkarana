"""
chittakit.antahkarana — the INTEGRATED agent.

Until now each faculty was tested alone. This wires them all into one agent that
lives a continual stream through the four-state cycle, with every Vedic module
active together:

    GUṆA + ĀŚRAMA   set the learning rate (regime + developmental plasticity)
    TAPAS           allocates the replay budget where the agent is weakest
    WAKE (jāgrat)   train current task + replay, under the SAṂSKĀRA penalty
    SLEEP (suṣupti) consolidate (Fisher grow+decay), optional homeostatic downscale,
                    deposit exemplars (the saṃskāra store)
    TURĪYA          the witness tracks identity drift across the lifetime
    PRAMĀṆA         at eval, the agent reports only what it can validly perceive
    KLEŚA (avidyā)  predictive entropy supplies the novelty signal to the guṇa

The backbone is the cognitive substrate; ChittaKit is the memory/effort/control/
safety layer wrapped around it. Swap the backbone (MLP→CNN→transformer→policy) and
the same agent scales — which is what makes Tracks B (embodiment) and C
(neuromorphic) extensions of THIS, not separate builds.
"""
from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .samskara import SamskaraMemory
from .guna import GunaController
from .meta_guna import MetaGunaController
from .ashrama import AshramaSchedule, effective_alpha
from .tapas import TapasController
from .pramana import PramanaGate
from .witness import WitnessMonitor
from .klesha import normalized_avidya


def _exemplars(loader, k: int):
    xs, ys = [], []
    for x, y in loader:
        xs.append(x); ys.append(y)
        if sum(len(t) for t in ys) >= k:
            break
    return torch.cat(xs)[:k], torch.cat(ys)[:k]


class Antahkarana:
    def __init__(self, model: nn.Module, n_tasks: int, *,
                 forward: Optional[Callable] = None, controller=None,
                 base_lr: float = 1e-2, ewc_lambda: float = 40.0, decay: float = 0.5,
                 replay_budget: int = 64, mem_size: int = 64, homeostasis: float = 1.0,
                 gate_threshold: float = 0.90) -> None:
        self.model = model
        # forward(x, task) routes to the right head; default assumes model(x, task)
        self.forward = forward or (lambda x, t: model(x, t))
        self.n_tasks = n_tasks
        self.base_lr = base_lr
        self.ewc_lambda = ewc_lambda
        self.replay_budget = replay_budget
        self.mem_size = mem_size
        self.homeostasis = homeostasis
        self.peak = {}                                                       # per-task peak acc

        # the faculties — default to the forgetting-aware controller (it strictly
        # beat the rule-based one: it fixes the easy-task over-regularization).
        self.chitta = SamskaraMemory(model, grow=1.0, decay=decay)            # memory
        self.guna = controller or MetaGunaController(base_lr, alpha_min=base_lr * 0.1,
                                                     alpha_max=base_lr * 2.0)
        self.ashrama = AshramaSchedule(alpha_min=base_lr * 0.1, alpha_peak=base_lr * 1.5)
        self.tapas = TapasController(temperature=0.4, floor=0.1)              # effort
        self.pramana = PramanaGate(threshold=gate_threshold)                 # validity
        self.witness: Optional[WitnessMonitor] = None                        # turīya
        self.memory = {}                                                     # replay store
        self.age = 0
        self.log: List[dict] = []
        self.device = next(model.parameters()).device                        # CPU or CUDA

    # ---- small helpers ----
    def _novelty(self, loader, task) -> float:
        x, _ = next(iter(loader))
        with torch.no_grad():
            return normalized_avidya(self.forward(x.to(self.device), task))

    def _acc(self, X, Y, task) -> float:
        with torch.no_grad():
            X, Y = X.to(self.device), Y.to(self.device)
            return float((self.forward(X, task).argmax(1) == Y).float().mean())

    def _ctrl(self, novelty, forgetting):
        """Call the controller (rule-based or forgetting-aware MetaGuna) uniformly.
        Returns (alpha, beta_frac, sattva, rajas, tamas)."""
        c = self.guna
        try:
            g = c.step(error=0.6, novelty=novelty, forgetting=forgetting, energy=1.0)
        except TypeError:
            g = c.step(error=0.6, novelty=novelty, reward=0.5, energy=1.0)
        if isinstance(g, dict):
            return g["alpha"], g.get("beta", 1.0), g["sattva"], g["rajas"], g["tamas"]
        return g.alpha, g.beta, g.sattva, g.rajas, g.tamas

    # ---- the lifelong loop ----
    def live(self, task_loaders, test_sets, epochs: int = 8,
             loss_fn: Optional[nn.Module] = None, verbose: bool = True):
        loss_fn = loss_fn or nn.CrossEntropyLoss()
        self.device = next(self.model.parameters()).device
        if self.witness is None:
            self.witness = WitnessMonitor(test_sets[0][0][:64].to(self.device))
            self.witness.imprint(self.model)                                 # imprint initial self
        acc_matrix = [[0.0] * self.n_tasks for _ in range(self.n_tasks)]

        for t in range(self.n_tasks):
            train_loader, _ = task_loaders[t]
            past = list(range(t))

            # measured FORGETTING signal: mean drop of past tasks from their peak
            forgetting = 0.0
            if past:
                drops = [max(0.0, self.peak.get(k, 0.0) - self._acc(*test_sets[k], k)) for k in past]
                forgetting = sum(drops) / len(drops)

            # GUṆA + ĀŚRAMA → learning rate AND adaptive consolidation strength
            novelty = self._novelty(train_loader, t)
            g_alpha, g_beta, g_s, g_r, g_t = self._ctrl(novelty, forgetting)
            env = self.ashrama.envelope(self.age, shift=novelty)
            lr = effective_alpha(env.envelope, g_alpha, alpha_min=self.base_lr * 0.1)
            beta = self.ewc_lambda * g_beta            # protect in proportion to forgetting
            opt = torch.optim.Adam(self.model.parameters(), lr=lr)

            # TAPAS → replay allocation across past tasks by current error
            if past:
                needs = [1.0 - self._acc(*test_sets[k], k) for k in past]
                b = self.tapas.allocate(self.replay_budget, needs)
                alloc = {past[i]: b[i] for i in range(len(past))}
            else:
                alloc = {}

            # WAKE (jāgrat): train current + replay, under the saṃskāra penalty
            self.model.train()
            for _ in range(epochs):
                for x, y in train_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    opt.zero_grad()
                    loss = loss_fn(self.forward(x, t), y) + self.chitta.penalty(beta)
                    for k, n in alloc.items():
                        if n > 0 and k in self.memory:
                            X, Y = self.memory[k]
                            idx = torch.randint(0, X.size(0), (n,))
                            loss = loss + loss_fn(self.forward(X[idx].to(self.device), k),
                                                  Y[idx].to(self.device)) * (n / self.replay_budget)
                    loss.backward(); opt.step()

            # SLEEP (suṣupti): consolidate, optional homeostatic downscale, deposit
            self.chitta.consolidate(train_loader, loss_fn, max_batches=15,
                                    forward_fn=lambda xb, _t=t: self.forward(xb, _t))
            if self.homeostasis < 1.0:
                with torch.no_grad():
                    for p in self.model.parameters():
                        p.mul_(self.homeostasis)
            self.memory[t] = _exemplars(train_loader, self.mem_size)

            # TURĪYA: identity drift since the lifetime began
            drift = self.witness.drift(self.model)
            self.age += 1
            for k in range(self.n_tasks):
                acc_matrix[t][k] = self._acc(*test_sets[k], k)
            for k in range(t + 1):                                # update per-task peak
                self.peak[k] = max(self.peak.get(k, 0.0), acc_matrix[t][k])

            entry = dict(task=t, stage=env.stage, sattva=round(g_s, 2), rajas=round(g_r, 2),
                         tamas=round(g_t, 2), lr=round(lr, 4), beta_frac=round(g_beta, 2),
                         forgetting=round(forgetting, 3), novelty=round(novelty, 2),
                         reopened=env.reopened,
                         plasticity=round(self.chitta.plasticity_headroom(), 2),
                         drift=round(drift, 3))
            self.log.append(entry)
            if verbose:
                print(f"  task {t}: stage={entry['stage']:<11} guṇa(s,r,t)="
                      f"({entry['sattva']},{entry['rajas']},{entry['tamas']}) lr={entry['lr']} "
                      f"β={entry['beta_frac']} forget={entry['forgetting']} "
                      f"novelty={entry['novelty']} plast={entry['plasticity']} drift={entry['drift']}")

        return self._summary(acc_matrix, test_sets)

    # ---- metrics, incl. the pramāṇa "knows what it knows" report ----
    def _summary(self, acc_matrix, test_sets):
        final = acc_matrix[-1]
        avg_acc = sum(final) / self.n_tasks
        forgets = [max(acc_matrix[tt][k] for tt in range(k, self.n_tasks)) - final[k]
                   for k in range(self.n_tasks - 1)]
        forgetting = sum(forgets) / max(len(forgets), 1)

        # PRAMĀṆA: gate the agent's perceptions on the whole test set
        all_logits, all_y = [], []
        for k, (X, Y) in enumerate(test_sets):
            with torch.no_grad():
                all_logits.append(self.forward(X.to(self.device), k).cpu()); all_y.append(Y)
        logits = torch.cat(all_logits); y = torch.cat(all_y)
        self.pramana.calibrate(logits, y)
        accept, conf, pred = self.pramana.judge(logits)
        coverage = float(accept.float().mean())
        sel_acc = float((pred[accept] == y[accept]).float().mean()) if accept.any() else float("nan")
        return dict(avg_acc=avg_acc, forgetting=forgetting, per_task=final,
                    gated_coverage=coverage, gated_accuracy=sel_acc, log=self.log)
