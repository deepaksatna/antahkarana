"""
Phase II — the integrated agent on a REAL backbone (CNN) and REAL images.

Proves the Antaḥkaraṇa control layer is backbone-agnostic: swap the toy MLP for a
convolutional network and run the SAME agent on Split-MNIST / Split-CIFAR10
(task-incremental: 5 tasks of 2 classes each, one head per task). Designed for GPU.

Compares:
    naive  — CNN trained sequentially, no faculties (catastrophic forgetting)
    agent  — the integrated Antaḥkaraṇa (chitta+guṇa+āśrama+tapas+pramāṇa+turīya)

Run:  CUDA_VISIBLE_DEVICES=3 python3 phase2_vision.py --dataset mnist
      CUDA_VISIBLE_DEVICES=3 python3 phase2_vision.py --dataset cifar10
"""
from __future__ import annotations

import argparse, os, sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from chittakit import Antahkarana, GunaController, MetaGunaController


# ----------------------------------------------------------------- data
def split_tasks(name, root, n_tasks=5):
    """Task-incremental: 5 tasks, each a 2-class problem (labels remapped to 0/1)."""
    if name == "mnist":
        tf = transforms.ToTensor()
        tr = datasets.MNIST(root, train=True, download=True, transform=tf)
        te = datasets.MNIST(root, train=False, download=True, transform=tf)
        ch = 1
    else:
        tf = transforms.ToTensor()
        tr = datasets.CIFAR10(root, train=True, download=True, transform=tf)
        te = datasets.CIFAR10(root, train=False, download=True, transform=tf)
        ch = 3

    def tensorize(ds):
        X = torch.stack([ds[i][0] for i in range(len(ds))])
        Y = torch.tensor([ds[i][1] for i in range(len(ds))])
        return X, Y
    Xtr, Ytr = tensorize(tr); Xte, Yte = tensorize(te)

    tasks, tests = [], []
    for t in range(n_tasks):
        a, b = 2 * t, 2 * t + 1
        def subset(X, Y):
            m = (Y == a) | (Y == b)
            return X[m], (Y[m] == b).long()
        xtr, ytr = subset(Xtr, Ytr); xte, yte = subset(Xte, Yte)
        tasks.append((DataLoader(TensorDataset(xtr, ytr), batch_size=128, shuffle=True),
                      DataLoader(TensorDataset(xte, yte), batch_size=256)))
        tests.append((xte[:1000], yte[:1000]))
    return tasks, tests, ch


# ----------------------------------------------------------------- model
class CNN(nn.Module):
    def __init__(self, ch, n_tasks, size):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        feat = 64 * (size // 4) * (size // 4)
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(feat, 128), nn.ReLU())
        self.heads = nn.ModuleList([nn.Linear(128, 2) for _ in range(n_tasks)])

    def features(self, x):
        return self.fc(self.conv(x))

    def forward(self, x, task):
        return self.heads[task](self.features(x))


# ----------------------------------------------------------------- baselines
def acc(model, X, Y, task, device):
    model.eval()
    with torch.no_grad():
        return float((model(X.to(device), task).argmax(1) == Y.to(device)).float().mean())


def run_naive(tasks, tests, ch, size, n_tasks, epochs, device):
    model = CNN(ch, n_tasks, size).to(device)
    lossf = nn.CrossEntropyLoss()
    accm = [[0.0] * n_tasks for _ in range(n_tasks)]
    for t in range(n_tasks):
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        model.train()
        for _ in range(epochs):
            for x, y in tasks[t][0]:
                x, y = x.to(device), y.to(device)
                opt.zero_grad(); lossf(model(x, t), y).backward(); opt.step()
        for k in range(n_tasks):
            accm[t][k] = acc(model, *tests[k], k, device)
    final = accm[-1]
    forget = sum(max(accm[tt][k] for tt in range(k, n_tasks)) - final[k]
                 for k in range(n_tasks - 1)) / (n_tasks - 1)
    return sum(final) / n_tasks, forget


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["mnist", "cifar10"], default="mnist")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--root", default=os.path.expanduser("~/akn-data"))
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_tasks = 5
    print(f"Phase II | dataset={args.dataset} | device={device} "
          f"| {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}\n")

    tasks, tests, ch = split_tasks(args.dataset, args.root, n_tasks)
    size = 28 if args.dataset == "mnist" else 32

    print("=== naive CNN (sequential, no faculties) ===")
    n_avg, n_forget = run_naive(tasks, tests, ch, size, n_tasks, args.epochs, device)
    print(f"  avg_acc={n_avg:.3f}  forgetting={n_forget:.3f}\n")

    results = {}
    for kind in ("rule", "meta"):
        print(f"=== integrated agent — {kind}-based guṇa controller ===")
        torch.manual_seed(0)
        model = CNN(ch, n_tasks, size).to(device)
        # base_lr=2e-3 so effective LR lands near naive 1e-3
        if kind == "meta":
            ctrl = MetaGunaController(2e-3, alpha_min=2e-4, alpha_max=4e-3)   # forgetting-aware
        else:
            ctrl = GunaController(alpha_min=2e-4, alpha_max=4e-3)
        agent = Antahkarana(model, n_tasks, controller=ctrl, base_lr=2e-3,
                            ewc_lambda=40.0, decay=0.5, replay_budget=128, mem_size=128)
        res = agent.live(tasks, tests, epochs=args.epochs, verbose=(kind == "meta"))
        results[kind] = res
        print(f"  → avg_acc={res['avg_acc']:.3f}  forgetting={res['forgetting']:.3f}\n")

    print(f"SUMMARY ({args.dataset}, CNN, GPU):")
    print(f"  {'naive':<22} avg_acc={n_avg:.3f}  forgetting={n_forget:.3f}")
    print(f"  {'agent (rule guṇa)':<22} avg_acc={results['rule']['avg_acc']:.3f}  "
          f"forgetting={results['rule']['forgetting']:.3f}")
    print(f"  {'agent (forgetting-aware)':<22} avg_acc={results['meta']['avg_acc']:.3f}  "
          f"forgetting={results['meta']['forgetting']:.3f}")
    print("\nThe forgetting-aware controller scales protection to MEASURED forgetting —")
    print("the test is whether it recovers accuracy on easy MNIST (low forgetting → less")
    print("protection) while keeping the win on hard CIFAR (high forgetting → protect).")


if __name__ == "__main__":
    main()
