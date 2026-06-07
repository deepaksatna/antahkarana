"""
Meta-learn the guṇa controller with Evolution Strategies (OpenAI-ES).

The controller's signals→guṇa matrix W is optimized so the agent does well across
a RANGE of difficulties (forgetting severity d ~ U(0,1)). Because W sees a measured
forgetting signal, ES should discover the adaptive policy:
    forget a lot  → high sattva → strong consolidation (protect)
    forget little → high rajas  → fast learning, light protection (don't over-regularize)

Then we compare the META-LEARNED controller against the RULE-BASED GunaController on
held-out easy (d=0.2) and hard (d=0.9) streams.

Run:  CUDA_VISIBLE_DEVICES=3 python3 meta_train_guna.py --gens 20 --pop 16
"""
from __future__ import annotations

import argparse, os, sys
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from chittakit import SamskaraMemory, MetaGunaController, GunaController, effective_alpha
from chittakit.klesha import normalized_avidya
import data as data_mod

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EWC_LAMBDA = 60.0
BASE_LR = 1e-2


class MLP(nn.Module):
    def __init__(self, dim, hidden=64):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(),
                                      nn.Linear(hidden, hidden), nn.ReLU())
        self.head = nn.Linear(hidden, 2)

    def forward(self, x):
        return self.head(self.backbone(x))


def _acc(model, X, Y):
    model.eval()
    with torch.no_grad():
        return float((model(X).argmax(1) == Y).float().mean())


def trial(controller, difficulty, seed, epochs=4):
    """One continual lifetime at a given difficulty. Controller may be MetaGuna or
    rule-based GunaController. Returns final average accuracy."""
    torch.manual_seed(seed)
    tasks, dim = data_mod.make_difficulty_tasks(difficulty, n_tasks=5, seed=seed)
    loaders = data_mod.loaders(tasks)
    tests = []
    for _, te in loaders:
        xs = torch.cat([b[0] for b in te]); ys = torch.cat([b[1] for b in te])
        tests.append((xs.to(DEVICE), ys.to(DEVICE)))

    model = MLP(dim).to(DEVICE)
    chitta = SamskaraMemory(model, grow=1.0, decay=0.5)
    loss_fn = nn.CrossEntropyLoss()
    peak = {}
    n = len(tasks)
    accm = [[0.0] * n for _ in range(n)]

    for t in range(n):
        train_loader, _ = loaders[t]
        past = list(range(t))
        # signals
        xb, _ = next(iter(train_loader))
        with torch.no_grad():
            novelty = normalized_avidya(model(xb.to(DEVICE)))
        forgetting = 0.0
        if past:
            drops = [max(0.0, peak[k] - _acc(model, *tests[k])) for k in past]
            forgetting = sum(drops) / len(drops)
        g = controller.step(error=0.6, novelty=novelty, forgetting=forgetting, energy=1.0)
        lr = g["alpha"]; beta = EWC_LAMBDA * g["beta"]

        opt = torch.optim.Adam(model.parameters(), lr=lr)
        model.train()
        for _ in range(epochs):
            for x, y in train_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                opt.zero_grad()
                loss = loss_fn(model(x), y) + chitta.penalty(beta)
                loss.backward(); opt.step()
        chitta.consolidate(((a, b) for a, b in train_loader), loss_fn, device=DEVICE, max_batches=8)
        for k in range(n):
            accm[t][k] = _acc(model, *tests[k])
        for k in range(t + 1):
            peak[k] = max(peak.get(k, 0.0), accm[t][k])

    return sum(accm[-1]) / n


def fitness(W, seeds, difficulties):
    ctrl = MetaGunaController(BASE_LR, params=W)
    return sum(trial(ctrl, d, s) for d, s in zip(difficulties, seeds)) / len(seeds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", type=int, default=20)
    ap.add_argument("--pop", type=int, default=16)
    ap.add_argument("--sigma", type=float, default=0.3)
    ap.add_argument("--lr-meta", type=float, default=0.15)
    args = ap.parse_args()
    print(f"meta-training the guṇa controller via ES | device={DEVICE} | "
          f"gens={args.gens} pop={args.pop}\n")

    base = MetaGunaController(BASE_LR)
    W = base.get_params()
    dim = W.numel()
    rng = torch.Generator().manual_seed(0)

    # FIXED evaluation panel spanning easy → hard, same every generation, so the
    # ES gradient is consistent and progress is measurable.
    PANEL_D = [0.2, 0.5, 0.9]
    PANEL_S = [11, 22, 33]
    print(f"  gen -1: fitness(prior)={fitness(W, PANEL_S, PANEL_D):.3f}")

    for gen in range(args.gens):
        eps = torch.randn(args.pop // 2, dim, generator=rng)
        eps = torch.cat([eps, -eps], 0)                      # antithetic
        fits = torch.tensor([fitness(W + args.sigma * e, PANEL_S, PANEL_D) for e in eps])
        ranks = fits.argsort().argsort().float()             # rank shaping
        shaped = ranks / (len(ranks) - 1) - 0.5
        grad = (shaped.unsqueeze(1) * eps).sum(0) / (args.pop * args.sigma)
        W = W + args.lr_meta * grad
        if gen % 2 == 0 or gen == args.gens - 1:
            print(f"  gen {gen:2d}: fitness(W)={fitness(W, PANEL_S, PANEL_D):.3f}  "
                  f"(pop mean {fits.mean():.3f})")

    torch.save(W, "meta_guna_W.pt")
    print("\nsaved learned controller → meta_guna_W.pt")

    # ---- validation: meta-learned vs rule-based, on held-out easy & hard ----
    print("\n====  VALIDATION (held-out seeds; avg_acc)  ====")
    meta = MetaGunaController(BASE_LR, params=W)
    rule = GunaController(alpha_min=BASE_LR * 0.1, alpha_max=BASE_LR * 2.0)

    class RuleAdapter:
        """Wrap rule-based GunaController to the same .step() signature."""
        def step(self, error, novelty, forgetting, energy):
            gs = rule.step(error=error, novelty=novelty, reward=0.5, energy=energy)
            return dict(alpha=gs.alpha, beta=gs.beta, sattva=gs.sattva, rajas=gs.rajas, tamas=gs.tamas)

    val_seeds = [1234, 5678, 4321]
    print(f"{'controller':<14}{'easy (d=0.2)':>16}{'hard (d=0.9)':>16}{'mean':>10}")
    for name, ctrl in [("rule-based", RuleAdapter()), ("meta-learned", meta)]:
        easy = sum(trial(ctrl, 0.2, s) for s in val_seeds) / len(val_seeds)
        hard = sum(trial(ctrl, 0.9, s) for s in val_seeds) / len(val_seeds)
        print(f"{name:<14}{easy:>16.3f}{hard:>16.3f}{(easy+hard)/2:>10.3f}")

    # ---- show the learned policy: does protection rise with forgetting? ----
    print("\n====  learned policy: consolidation β vs the forgetting signal  ====")
    for f in (0.0, 0.1, 0.3, 0.5):
        g = meta.step(error=0.6, novelty=0.5, forgetting=f, energy=1.0)
        print(f"  forgetting={f:.1f} → sattva={g['sattva']:.2f} rajas={g['rajas']:.2f} "
              f"lr={g['alpha']:.4f} β_frac={g['beta']:.2f}")
    print("\nIf β_frac rises with the forgetting signal, the controller LEARNED to protect")
    print("hard tasks and back off on easy ones — the adaptive policy the rule-based lacked.")


if __name__ == "__main__":
    main()
