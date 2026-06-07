"""
Continual-learning benchmark — the first falsifiable test of ChittaKit.

Compares three regimes on a sequential task stream (task-incremental):
    naive     — no protection (baseline; forgets)
    ewc       — SamskaraMemory with decay=0  (classic online EWC: protects, but
                importance only accumulates → rigidity over many tasks)
    samskara  — SamskaraMemory with decay>0  (grow + decay → protects AND stays
                plastic) + GunaController + AshramaSchedule modulating plasticity

Metrics after the full stream:
    avg_acc    — mean test accuracy over ALL tasks (higher = better)
    forgetting — mean drop from each task's peak accuracy to its final accuracy
                 (lower = better; the catastrophic-forgetting measure)
    learned    — accuracy on the LAST task right after learning it (a plasticity
                 proxy: did protection make us too rigid to learn new things?)

Run:  python3 continual_benchmark.py            # synthetic, offline, ~seconds
      python3 continual_benchmark.py --dataset mnist   # needs torchvision
"""
from __future__ import annotations

import argparse
import sys
import os

import torch
import torch.nn as nn

# allow running from the experiments/ dir without installing the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from chittakit import SamskaraMemory, GunaController, AshramaSchedule, effective_alpha
from chittakit.klesha import normalized_avidya
import data as data_mod


# --------------------------------------------------------------------- model
class SingleHeadMLP(nn.Module):
    """Shared backbone + one shared 2-way head (domain-incremental / permuted).

    All tasks share the head, so forgetting shows as backbone+head drift when the
    input distribution (the permutation) changes between tasks."""

    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.head = nn.Linear(hidden, 2)

    def features(self, x):
        return self.backbone(x)

    def forward(self, x):
        return self.head(self.backbone(x))


# ----------------------------------------------------------------- evaluation
@torch.no_grad()
def accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        correct += int((pred == y).sum()); total += y.numel()
    return correct / max(total, 1)


# ------------------------------------------------------------------- training
def run(regime: str, tasks, dim, hidden, epochs, device, ewc_lambda=100.0, base_lr=1e-2,
        seed=0, verbose=True):
    torch.manual_seed(seed)
    n_tasks = len(tasks)
    model = SingleHeadMLP(dim, hidden).to(device)
    loss_fn = nn.CrossEntropyLoss()

    use_protection = regime in ("ewc", "samskara", "samskara+guna")
    guna_on = regime == "samskara+guna"
    # The isolated variable across ewc vs samskara is importance DECAY at the SAME
    # learning rate: ewc keeps importance forever (decay=0) and ossifies; saṃskāra
    # lets it fade (decay>0) so the net stays plastic. The 4th regime adds the
    # guṇa/āśrama adaptive learning rate on top.
    decay = 0.0 if regime == "ewc" else (0.5 if use_protection else 0.0)
    mem = SamskaraMemory(model, grow=1.0, decay=decay) if use_protection else None

    guna = GunaController(alpha_min=base_lr * 0.1, alpha_max=base_lr * 1.5) if guna_on else None
    ashrama = AshramaSchedule(alpha_min=base_lr * 0.1, alpha_peak=base_lr * 1.5) if guna_on else None

    acc_matrix = [[0.0] * n_tasks for _ in range(n_tasks)]  # acc_matrix[after_task][task]
    task_loaders = data_mod.loaders(tasks)  # [(train, test), ...] built once

    for t in range(n_tasks):
        train_loader, _ = task_loaders[t]

        # --- guṇa + āśrama set the learning rate (samskara+guna regime only) ---
        lr = base_lr
        if guna_on:
            probe_logits = _peek_logits(model, train_loader, device)
            novelty = normalized_avidya(probe_logits)  # high when the net is unsure on new data
            g = guna.step(error=0.7, novelty=novelty, reward=0.5, energy=1.0)
            env = ashrama.envelope(age=float(t), shift=novelty)
            lr = effective_alpha(env.envelope, g.alpha, alpha_min=base_lr * 0.05)
            if verbose:
                print(f"  [task {t}] stage={env.stage:<11} novelty={novelty:.2f} "
                      f"guṇa(s,r,t)=({g.sattva:.2f},{g.rajas:.2f},{g.tamas:.2f}) "
                      f"lr={lr:.4f} reopened={env.reopened}")

        # Adam: adaptive step keeps the stiff quadratic consolidation penalty stable.
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        beta = ewc_lambda if use_protection else 0.0  # SAME for ewc & samskara (fair)

        model.train()
        for _ in range(epochs):
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                out = model(x)
                loss = loss_fn(out, y)
                if use_protection:
                    loss = loss + mem.penalty(beta=beta)
                loss.backward()
                opt.step()

        # --- suṣupti: consolidate (estimate Fisher, grow+decay Ω, snapshot θ*) ---
        if use_protection:
            mem.consolidate(train_loader, loss_fn, device=device, max_batches=20)

        # evaluate all tasks seen so far
        for k in range(n_tasks):
            _, test_loader = task_loaders[k]
            acc_matrix[t][k] = accuracy(model, test_loader, device)

    # ---- metrics ----
    final = acc_matrix[n_tasks - 1]
    avg_acc = sum(final) / n_tasks
    # forgetting: peak over time minus final, averaged over tasks learned before the last
    forgets = []
    for k in range(n_tasks - 1):
        peak = max(acc_matrix[t][k] for t in range(k, n_tasks))
        forgets.append(peak - acc_matrix[n_tasks - 1][k])
    forgetting = sum(forgets) / max(len(forgets), 1)
    learned_last = acc_matrix[n_tasks - 1][n_tasks - 1]
    plast = mem.plasticity_headroom() if mem else 1.0
    imp = mem.importance_mass() if mem else 0.0
    return {
        "regime": regime, "avg_acc": avg_acc, "forgetting": forgetting,
        "learned_last": learned_last, "plasticity_headroom": plast, "importance_mass": imp,
    }


@torch.no_grad()
def _peek_logits(model, loader, device):
    model.eval()
    x, _ = next(iter(loader))
    return model(x.to(device))


# ----------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["synthetic", "mnist"], default="synthetic")
    ap.add_argument("--n-tasks", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=100)
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--ewc-lambda", type=float, default=50.0)
    ap.add_argument("--base-lr", type=float, default=1e-2)
    ap.add_argument("--seeds", type=int, default=5, help="number of seeds to average")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device} | dataset = {args.dataset} | tasks = {args.n_tasks} "
          f"| ewc_lambda = {args.ewc_lambda} | seeds = {args.seeds}\n")

    regimes = ("naive", "ewc", "samskara", "samskara+guna")
    agg = {r: {"avg_acc": [], "forgetting": [], "learned_last": [], "plasticity_headroom": []} for r in regimes}

    for seed in range(args.seeds):
        if args.dataset == "synthetic":
            dim = args.dim
            tasks = data_mod.make_tasks(n_tasks=args.n_tasks, dim=dim, seed=seed)
        else:
            from mnist_tasks import make_split_mnist
            tasks, dim = make_split_mnist(n_tasks=args.n_tasks)
        for regime in regimes:
            res = run(regime, tasks, dim, args.hidden, args.epochs, device,
                      ewc_lambda=args.ewc_lambda, base_lr=args.base_lr,
                      seed=seed, verbose=False)
            for k in agg[regime]:
                agg[regime][k].append(res[k])
        print(f"  seed {seed} done")

    def ms(xs):
        m = sum(xs) / len(xs)
        sd = (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5
        return m, sd

    print(f"\n========  SUMMARY over {args.seeds} seeds  (mean ± std)  ========")
    print(f"{'regime':<14}{'avg_acc':>14}{'forgetting':>16}{'learned_last':>16}{'plasticity':>14}")
    for r in regimes:
        a = ms(agg[r]["avg_acc"]); f = ms(agg[r]["forgetting"])
        l = ms(agg[r]["learned_last"]); p = ms(agg[r]["plasticity_headroom"])
        print(f"{r:<14}{a[0]:>7.3f}±{a[1]:<5.3f}{f[0]:>8.3f}±{f[1]:<6.3f}"
              f"{l[0]:>8.3f}±{l[1]:<6.3f}{p[0]:>7.2f}±{p[1]:<5.2f}")
    print("\nReading:")
    print("  naive          — forgets catastrophically (high forgetting).")
    print("  ewc            — protects (low forgetting) but ossifies: importance only")
    print("                   accumulates, so learned_last & plasticity_headroom fall.")
    print("  samskara       — same protection, but importance DECAYS → keeps higher")
    print("                   plasticity_headroom / learned_last (the enhanceable core).")
    print("  samskara+guna  — adds the guṇa/āśrama adaptive learning rate on top.")


if __name__ == "__main__":
    main()
