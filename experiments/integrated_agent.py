"""
The integrated Antaḥkaraṇa-Net agent — all faculties in ONE lifelong loop.

This is the keystone: where chitta, guṇa, āśrama, tapas, pramāṇa, and the turīya
witness run TOGETHER on a continual stream, not in separate scripts. It prints the
agent's "mind state" each task (life-stage, guṇa mix, learning rate, novelty,
plasticity, witness drift) and the final continual-learning + self-knowledge
metrics. Tracks B (embodiment) and C (neuromorphic) extend THIS agent.

Run:  python3 integrated_agent.py            # offline, CPU, ~1-2 min, 3 seeds
"""
from __future__ import annotations

import os, sys
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from chittakit import Antahkarana
import data as data_mod


class MultiHeadMLP(nn.Module):
    def __init__(self, dim, hidden, n_tasks):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(),
                                      nn.Linear(hidden, hidden), nn.ReLU())
        self.heads = nn.ModuleList([nn.Linear(hidden, 2) for _ in range(n_tasks)])

    def features(self, x):                 # used by the turīya witness
        return self.backbone(x)

    def forward(self, x, task):
        return self.heads[task](self.backbone(x))


def exemplars(loader, k):
    xs, ys = [], []
    for x, y in loader:
        xs.append(x); ys.append(y)
        if sum(len(t) for t in ys) >= k:
            break
    return torch.cat(xs)[:k], torch.cat(ys)[:k]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-tasks", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} | integrated agent | tasks={args.n_tasks} | seeds={args.seeds}\n")

    agg = {k: [] for k in ("avg_acc", "forgetting", "gated_coverage", "gated_accuracy")}
    for seed in range(args.seeds):
        torch.manual_seed(seed)
        tasks, dim = data_mod.make_shared_feature_tasks(n_tasks=args.n_tasks, seed=seed)
        loaders = data_mod.loaders(tasks)
        test_sets = [exemplars(te, 400) for _, te in loaders]

        model = MultiHeadMLP(dim, args.hidden, args.n_tasks)
        agent = Antahkarana(model, args.n_tasks, base_lr=1e-2, ewc_lambda=40.0, decay=0.5)
        print(f"--- lifetime, seed {seed} (mind-state trace) ---")
        res = agent.live(loaders, test_sets, epochs=args.epochs, verbose=(seed == 0))
        for k in agg:
            agg[k].append(res[k])
        print(f"  → avg_acc={res['avg_acc']:.3f}  forgetting={res['forgetting']:.3f}  "
              f"gated: keeps {res['gated_coverage']:.0%} at acc {res['gated_accuracy']:.3f}\n")

    def ms(xs):
        m = sum(xs) / len(xs)
        return m, (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5

    print(f"====  INTEGRATED AGENT  over {args.seeds} seeds  ====")
    for k in ("avg_acc", "forgetting", "gated_coverage", "gated_accuracy"):
        m, s = ms(agg[k])
        print(f"  {k:<16}: {m:.3f} ± {s:.3f}")
    print("\nOne agent, one wake/dream/sleep loop, every faculty active: it learns the")
    print("stream continually (low forgetting), allocates effort by tapas, adapts its")
    print("learning rate by guṇa/āśrama, watches its own drift (turīya), and reports")
    print("only what it can validly perceive (pramāṇa). This is the model the B/C tracks extend.")


if __name__ == "__main__":
    main()
