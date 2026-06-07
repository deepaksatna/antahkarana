"""
Capacity-headroom benchmark — the setting where the novel mechanisms can show.

Tasks share a common nonlinear feature map φ (one fixed teacher) and differ only
in a per-task linear readout (see data.make_shared_feature_tasks). A MULTI-HEAD
model (shared backbone + one head per task) CAN solve all tasks jointly if the
backbone learns φ — so average accuracy has real headroom — but naive sequential
training overwrites the backbone (forgetting). This is where consolidation, the
saṃskāra DECAY, and TAPAS-guided replay can actually differentiate.

Regimes:
    naive            — no protection, no replay
    ewc              — SamskaraMemory, decay=0 (classic; protects but ossifies)
    samskara         — SamskaraMemory, decay>0 (protects AND stays plastic)
    replay-uniform   — fixed replay budget split equally across past tasks
    replay-tapas     — SAME budget, concentrated where current error is worst

Run:  python3 capacity_benchmark.py            # offline, CPU, ~1-2 min, 5 seeds
"""
from __future__ import annotations

import argparse, os, sys
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from chittakit import SamskaraMemory, TapasController
import data as data_mod


class MultiHeadMLP(nn.Module):
    def __init__(self, dim, hidden, n_tasks):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(),
                                      nn.Linear(hidden, hidden), nn.ReLU())
        self.heads = nn.ModuleList([nn.Linear(hidden, 2) for _ in range(n_tasks)])

    def forward(self, x, task):
        return self.heads[task](self.backbone(x))


@torch.no_grad()
def accuracy(model, X, Y, task):
    model.eval()
    return float((model(X, task).argmax(1) == Y).float().mean())


def _exemplars(loader, k):
    xs, ys = [], []
    for x, y in loader:
        xs.append(x); ys.append(y)
        if sum(len(t) for t in ys) >= k:
            break
    return torch.cat(xs)[:k], torch.cat(ys)[:k]


def _split_even(budget, k):
    base = [budget // k] * k
    for i in range(budget - sum(base)):
        base[i] += 1
    return base


def run(regime, tasks, dim, hidden, epochs, device, ewc_lambda=40.0,
        mem_size=64, replay_budget=64, seed=0):
    torch.manual_seed(seed)
    n_tasks = len(tasks)
    model = MultiHeadMLP(dim, hidden, n_tasks).to(device)
    loss_fn = nn.CrossEntropyLoss()
    tapas = TapasController(temperature=0.4, floor=0.1)

    protect = regime in ("ewc", "samskara")
    decay = 0.0 if regime == "ewc" else 0.5
    mem = SamskaraMemory(model, grow=1.0, decay=decay) if protect else None
    do_replay = regime in ("replay-uniform", "replay-tapas")

    loaders = data_mod.loaders(tasks)
    test_sets = [(_exemplars(te, 400)) for _, te in loaders]
    test_sets = [(X.to(device), Y.to(device)) for X, Y in test_sets]
    memory = {}
    acc_matrix = [[0.0] * n_tasks for _ in range(n_tasks)]

    for t in range(n_tasks):
        train_loader, _ = loaders[t]
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        past = list(range(t))

        for _ in range(epochs):
            alloc = {}
            if do_replay and past:
                if regime == "replay-tapas":
                    needs = [1.0 - accuracy(model, *test_sets[k], k) for k in past]
                    b = tapas.allocate(replay_budget, needs)
                else:
                    b = _split_even(replay_budget, len(past))
                alloc = {past[i]: b[i] for i in range(len(past))}

            model.train()
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                loss = loss_fn(model(x, t), y)
                if protect:
                    loss = loss + mem.penalty(beta=ewc_lambda)
                if alloc:
                    for k, n in alloc.items():
                        if n <= 0 or k not in memory:
                            continue
                        X, Y = memory[k]
                        idx = torch.randint(0, X.size(0), (n,))
                        loss = loss + loss_fn(model(X[idx], k), Y[idx]) * (n / replay_budget)
                loss.backward()
                opt.step()

        if protect:
            # consolidate using this task's head (forward_fn routes to head t)
            mem.consolidate(train_loader, loss_fn, device=device, max_batches=15,
                            forward_fn=lambda xb, _t=t: model(xb, _t))
        if do_replay:
            Xm, Ym = _exemplars(train_loader, mem_size)
            memory[t] = (Xm.to(device), Ym.to(device))

        for k in range(n_tasks):
            acc_matrix[t][k] = accuracy(model, *test_sets[k], k)

    final = acc_matrix[-1]
    avg_acc = sum(final) / n_tasks
    worst = min(final)
    forgets = [max(acc_matrix[tt][k] for tt in range(k, n_tasks)) - final[k]
               for k in range(n_tasks - 1)]
    forgetting = sum(forgets) / max(len(forgets), 1)
    return {"regime": regime, "avg_acc": avg_acc, "forgetting": forgetting, "worst_task": worst}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-tasks", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--ewc-lambda", type=float, default=40.0)
    ap.add_argument("--replay-budget", type=int, default=64)
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} | shared-feature tasks={args.n_tasks} | seeds={args.seeds} "
          f"| ewc_lambda={args.ewc_lambda} | replay_budget={args.replay_budget}\n")

    regimes = ("naive", "ewc", "samskara", "replay-uniform", "replay-tapas")
    agg = {r: {"avg_acc": [], "forgetting": [], "worst_task": []} for r in regimes}
    for seed in range(args.seeds):
        tasks, dim = data_mod.make_shared_feature_tasks(n_tasks=args.n_tasks, seed=seed)
        for r in regimes:
            res = run(r, tasks, dim, args.hidden, args.epochs, device,
                      ewc_lambda=args.ewc_lambda, replay_budget=args.replay_budget, seed=seed)
            for k in agg[r]:
                agg[r][k].append(res[k])
        print(f"  seed {seed} done")

    def ms(xs):
        m = sum(xs) / len(xs)
        return m, (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5

    print(f"\n====  CAPACITY-HEADROOM benchmark  over {args.seeds} seeds  (mean ± std)  ====")
    print(f"{'regime':<16}{'avg_acc ↑':>16}{'forgetting ↓':>16}{'worst_task ↑':>16}")
    for r in regimes:
        a, f, w = ms(agg[r]['avg_acc']), ms(agg[r]['forgetting']), ms(agg[r]['worst_task'])
        print(f"{r:<16}{a[0]:>8.3f}±{a[1]:<6.3f}{f[0]:>8.3f}±{f[1]:<6.3f}{w[0]:>8.3f}±{w[1]:<6.3f}")


if __name__ == "__main__":
    main()
