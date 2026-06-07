"""
Synthetic continual-learning benchmark — fully offline (no downloads).

Conflicting-teacher stream (the setting where forgetting is strong and real):

  - Inputs x ~ N(0, I) in R^dim, shared across all tasks.
  - Each task t has its OWN random teacher hyperplane w_t: label y = 1[ w_t·x > 0 ].
  - A SINGLE shared head is used for every task.

Because each task demands a DIFFERENT decision boundary, learning task t+1
genuinely overwrites the weights that solved task t → catastrophic forgetting.
Consolidation (EWC / saṃskāra) protects the weights that mattered for earlier
teachers, trading some new-task fit for retained old-task accuracy.

Deliberately small — the whole study runs on CPU in well under a minute.
"""
from __future__ import annotations

from typing import List, Tuple

import torch
from torch.utils.data import TensorDataset, DataLoader


def make_tasks(
    n_tasks: int = 6,
    dim: int = 32,
    n_train: int = 1000,
    n_test: int = 400,
    margin: float = 0.15,   # drop points too close to the boundary (cleaner labels)
    seed: int = 0,
) -> List[Tuple[TensorDataset, TensorDataset]]:
    g = torch.Generator().manual_seed(seed)
    tasks = []
    for _ in range(n_tasks):
        w = torch.randn(dim, generator=g)
        w = w / w.norm()
        train = _teacher(w, n_train, margin, dim, g)
        test = _teacher(w, n_test, margin, dim, g)
        tasks.append((train, test))
    return tasks


def _teacher(w, n, margin, dim, g) -> TensorDataset:
    xs, ys = [], []
    while sum(len(y) for y in ys) < n:
        x = torch.randn(n, dim, generator=g)
        s = x @ w
        keep = s.abs() > margin                # enforce a margin → clean labels
        xs.append(x[keep]); ys.append((s[keep] > 0).long())
    x = torch.cat(xs)[:n]; y = torch.cat(ys)[:n]
    return TensorDataset(x, y)


def loaders(tasks, batch_size: int = 64):
    out = []
    for train, test in tasks:
        out.append((
            DataLoader(train, batch_size=batch_size, shuffle=True),
            DataLoader(test, batch_size=256, shuffle=False),
        ))
    return out


# ---------------------------------------------------------------------------
# Capacity-headroom stream: tasks share a common nonlinear feature map φ (one
# fixed random teacher MLP), and differ only in a per-task linear readout. A
# backbone that learns φ can solve ALL tasks jointly (so avg accuracy can be
# high — there is real headroom), but training a new task can overwrite the
# parts of φ the old tasks relied on (forgetting). This is the setting where
# consolidation/replay/decay/tapas can actually differentiate.
# Use with a MULTI-HEAD model: evaluate task k with head k.
# ---------------------------------------------------------------------------
def make_shared_feature_tasks(
    n_tasks: int = 8,
    dim: int = 24,
    teacher_hidden: int = 32,
    n_train: int = 800,
    n_test: int = 400,
    margin: float = 0.2,
    seed: int = 0,
):
    g = torch.Generator().manual_seed(seed)
    # one shared, fixed nonlinear teacher  φ: R^dim -> R^teacher_hidden
    W1 = torch.randn(dim, teacher_hidden, generator=g) / (dim ** 0.5)
    b1 = torch.randn(teacher_hidden, generator=g) * 0.1

    def phi(x):
        return torch.tanh(x @ W1 + b1)

    tasks = []
    for _ in range(n_tasks):
        u = torch.randn(teacher_hidden, generator=g)          # per-task readout
        u = u / u.norm()
        train = _shared(phi, u, n_train, dim, margin, g)
        test = _shared(phi, u, n_test, dim, margin, g)
        tasks.append((train, test))
    return tasks, dim


def make_difficulty_tasks(difficulty: float, n_tasks: int = 5, dim: int = 24,
                          n_train: int = 400, n_test: int = 200, margin: float = 0.1,
                          seed: int = 0):
    """Single-head continual stream with a tunable FORGETTING knob.

    Each task's teacher is w_t = normalize((1-d)·w0 + d·r_t):
      d=0  → every task identical (no forgetting; protection only hurts)
      d=1  → every task a fresh random boundary (severe forgetting; protection helps)
    The controller must read the resulting forgetting and adapt. Returns task list
    + dim; use with a SINGLE-head model."""
    g = torch.Generator().manual_seed(seed)
    w0 = torch.randn(dim, generator=g); w0 = w0 / w0.norm()
    tasks = []
    for _ in range(n_tasks):
        r = torch.randn(dim, generator=g)
        w = (1 - difficulty) * w0 + difficulty * r
        w = w / w.norm()
        tr = _hyper(w, n_train, dim, margin, g)
        te = _hyper(w, n_test, dim, margin, g)
        tasks.append((tr, te))
    return tasks, dim


def _hyper(w, n, dim, margin, g) -> TensorDataset:
    xs, ys = [], []
    while sum(len(y) for y in ys) < n:
        x = torch.randn(n, dim, generator=g)
        s = x @ w
        keep = s.abs() > margin
        xs.append(x[keep]); ys.append((s[keep] > 0).long())
    return TensorDataset(torch.cat(xs)[:n], torch.cat(ys)[:n])


def _shared(phi, u, n, dim, margin, g) -> TensorDataset:
    xs, ys = [], []
    while sum(len(y) for y in ys) < n:
        x = torch.randn(n, dim, generator=g)
        s = phi(x) @ u
        keep = s.abs() > margin
        xs.append(x[keep]); ys.append((s[keep] > 0).long())
    x = torch.cat(xs)[:n]; y = torch.cat(ys)[:n]
    return TensorDataset(x, y)
