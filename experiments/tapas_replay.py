"""
Tapas-guided replay — concentrate rehearsal effort where forgetting is worst.

This experiment builds the dream/sleep REPLAY mechanism (suṣupti: rehearse stored
exemplars while learning new tasks) and asks the tapas question:

    Given a FIXED replay budget per step, is it better to rehearse all past tasks
    EQUALLY (uniform), or to pour the effort into the tasks you are currently
    WORST at (tapas)?

Tapasya says: apply the heat where the ore is hardest to melt. So we allocate the
replay budget across past tasks in proportion to their current error.

Regimes (all single shared head; conflicting-teacher stream):
    no-replay        — train each task on its own data only (baseline; forgets)
    replay-uniform   — fixed replay budget split EQUALLY across past tasks
    replay-tapas     — SAME budget, allocated by current per-task error (TapasController)

Run:  python3 tapas_replay.py            # offline, CPU, ~1 min, 5 seeds
"""
from __future__ import annotations

import argparse, os, sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from chittakit import TapasController
import data as data_mod


class SingleHeadMLP(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(),
                                      nn.Linear(hidden, hidden), nn.ReLU())
        self.head = nn.Linear(hidden, 2)

    def forward(self, x):
        return self.head(self.backbone(x))


@torch.no_grad()
def accuracy(model, X, Y):
    model.eval()
    return float((model(X).argmax(1) == Y).float().mean())


def stored_exemplars(loader, k):
    xs, ys = [], []
    for x, y in loader:
        xs.append(x); ys.append(y)
        if sum(len(t) for t in ys) >= k:
            break
    X = torch.cat(xs)[:k]; Y = torch.cat(ys)[:k]
    return X, Y


def run(regime, tasks, dim, hidden, epochs, mem_size, replay_budget, device, seed=0):
    torch.manual_seed(seed)
    model = SingleHeadMLP(dim, hidden).to(device)
    loss_fn = nn.CrossEntropyLoss()
    tapas = TapasController(temperature=0.4, floor=0.1)

    task_loaders = data_mod.loaders(tasks)
    memory = {}                       # task -> (X, Y) stored exemplars
    test_sets = []
    for _, test_loader in task_loaders:
        Xt, Yt = stored_exemplars(test_loader, 400)
        test_sets.append((Xt.to(device), Yt.to(device)))

    n_tasks = len(tasks)
    acc_matrix = [[0.0] * n_tasks for _ in range(n_tasks)]

    for t in range(n_tasks):
        train_loader, _ = task_loaders[t]
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        past = list(range(t))

        for _ in range(epochs):
            # --- tapas allocation of the replay budget across past tasks ---
            alloc = {}
            if regime != "no-replay" and past:
                if regime == "replay-uniform":
                    needs = [1.0] * len(past)
                else:  # replay-tapas: need = current error on each past task
                    needs = [1.0 - accuracy(model, *test_sets[k]) for k in past]
                budget = tapas.allocate(replay_budget, needs) if regime == "replay-tapas" \
                    else _split_even(replay_budget, len(past))
                alloc = {past[i]: budget[i] for i in range(len(past))}

            model.train()
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                loss = loss_fn(model(x), y)
                # --- replay: rehearse stored exemplars per the allocation ---
                if alloc:
                    rx, ry = _draw_replay(memory, alloc, device)
                    if rx is not None:
                        loss = loss + loss_fn(model(rx), ry)
                loss.backward()
                opt.step()

        # store exemplars for this task (the saṃskāra deposit)
        Xm, Ym = stored_exemplars(train_loader, mem_size)
        memory[t] = (Xm.to(device), Ym.to(device))

        for k in range(n_tasks):
            acc_matrix[t][k] = accuracy(model, *test_sets[k])

    final = acc_matrix[n_tasks - 1]
    avg_acc = sum(final) / n_tasks
    worst = min(final)                          # the floor tapas is meant to lift
    forgets = []
    for k in range(n_tasks - 1):
        peak = max(acc_matrix[tt][k] for tt in range(k, n_tasks))
        forgets.append(peak - final[k])
    forgetting = sum(forgets) / max(len(forgets), 1)
    return {"regime": regime, "avg_acc": avg_acc, "forgetting": forgetting, "worst_task": worst}


def _split_even(budget, k):
    base = [budget // k] * k
    for i in range(budget - sum(base)):
        base[i] += 1
    return base


def _draw_replay(memory, alloc, device):
    xs, ys = [], []
    for task, n in alloc.items():
        if n <= 0 or task not in memory:
            continue
        X, Y = memory[task]
        idx = torch.randint(0, X.size(0), (n,))
        xs.append(X[idx]); ys.append(Y[idx])
    if not xs:
        return None, None
    return torch.cat(xs).to(device), torch.cat(ys).to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-tasks", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--hidden", type=int, default=100)
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--mem-size", type=int, default=64)
    ap.add_argument("--replay-budget", type=int, default=64)
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} | tasks={args.n_tasks} | replay_budget={args.replay_budget} "
          f"(EQUAL for uniform & tapas) | seeds={args.seeds}\n")

    regimes = ("no-replay", "replay-uniform", "replay-tapas")
    agg = {r: {"avg_acc": [], "forgetting": [], "worst_task": []} for r in regimes}
    for seed in range(args.seeds):
        tasks = data_mod.make_tasks(n_tasks=args.n_tasks, dim=args.dim, seed=seed)
        for r in regimes:
            res = run(r, tasks, args.dim, args.hidden, args.epochs,
                      args.mem_size, args.replay_budget, device, seed=seed)
            for k in agg[r]:
                agg[r][k].append(res[k])
        print(f"  seed {seed} done")

    def ms(xs):
        m = sum(xs) / len(xs)
        return m, (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5

    print(f"\n====  TAPAS-REPLAY  over {args.seeds} seeds  (mean ± std)  ====")
    print(f"{'regime':<16}{'avg_acc ↑':>16}{'forgetting ↓':>16}{'worst_task ↑':>16}")
    for r in regimes:
        a, f, w = ms(agg[r]['avg_acc']), ms(agg[r]['forgetting']), ms(agg[r]['worst_task'])
        print(f"{r:<16}{a[0]:>8.3f}±{a[1]:<6.3f}{f[0]:>8.3f}±{f[1]:<6.3f}{w[0]:>8.3f}±{w[1]:<6.3f}")
    print("\nReading: replay >> no-replay (rehearsal beats forgetting). tapas vs uniform")
    print("at EQUAL budget tests whether concentrating effort where you are weakest")
    print("(tapasya) beats spreading it evenly — watch avg_acc and especially worst_task.")


if __name__ == "__main__":
    main()
