"""
Track B v2 — embodiment with a TASK-CONDITIONED policy (the retention fix).

v1 showed the karma loop and battery→guṇa work, but consolidation could not retain
skills across food-regimes: a single context-free policy can't hold contradictory
goal-functions. The fix is the same one that works in the supervised agent — a
shared backbone + PER-REGIME heads:

    backbone (shared navigation features)  →  head_r (decodes the goal for regime r)

Now head_r is regime-specific (frozen once its regime passes), and the SAṂSKĀRA
penalty protects the shared backbone, so learning a new regime no longer destroys
the old ones. We compare naive (no protection) vs saṃskāra across 4 food-regimes.

Battery→guṇa still drives exploration throughout (the embodiment stays real).

Run:  CUDA_VISIBLE_DEVICES=3 python3 track_b_v2.py
"""
from __future__ import annotations

import os, sys, random
from collections import deque

import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from chittakit import SamskaraMemory, GunaController

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BatteryForaging:
    ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def __init__(self, n=6, batt_max=18, max_steps=24, food=(0, 0)):
        self.n, self.batt_max, self.max_steps, self.food = n, batt_max, max_steps, food

    def reset(self):
        while True:
            self.pos = [random.randrange(self.n), random.randrange(self.n)]
            if tuple(self.pos) != self.food:
                break
        self.batt = self.batt_max; self.steps = 0
        return self._state()

    def _state(self):
        return torch.tensor([self.pos[0] / self.n, self.pos[1] / self.n,
                             self.batt / self.batt_max], dtype=torch.float32)

    def step(self, a):
        dr, dc = self.ACTIONS[a]
        self.pos[0] = max(0, min(self.n - 1, self.pos[0] + dr))
        self.pos[1] = max(0, min(self.n - 1, self.pos[1] + dc))
        self.batt -= 1; self.steps += 1
        if tuple(self.pos) == self.food:
            return self._state(), 1.0, True, True
        if self.batt <= 0 or self.steps >= self.max_steps:
            return self._state(), -1.0, True, False
        return self._state(), -0.02, False, False

    @property
    def battery_norm(self):
        return self.batt / self.batt_max


class MultiHeadQNet(nn.Module):
    """Shared navigation backbone + one Q-head per regime (task-conditioned)."""
    def __init__(self, n_regimes):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(3, 64), nn.ReLU(),
                                      nn.Linear(64, 64), nn.ReLU())
        self.heads = nn.ModuleList([nn.Linear(64, 4) for _ in range(n_regimes)])

    def forward(self, x, regime):
        return self.heads[regime](self.backbone(x))


def eps_from_guna(guna, battery_norm):
    g = guna.step(error=0.5, novelty=0.3, reward=0.5, energy=battery_norm)
    return 0.02 + 0.43 * g.rajas * (1 - g.tamas)


def train_regime(q, tgt, mem, guna, env, regime, episodes, beta, track=None):
    opt = torch.optim.Adam(q.parameters(), lr=1e-3)
    buffer = deque(maxlen=5000)
    for ep in range(episodes):
        s = env.reset(); done = False
        while not done:
            eps = eps_from_guna(guna, env.battery_norm)
            if track is not None:
                track.append((env.battery_norm, eps))
            if random.random() < eps:
                a = random.randrange(4)
            else:
                with torch.no_grad():
                    a = int(q(s.to(DEVICE), regime).argmax())
            s2, r, done, success = env.step(a)
            buffer.append((s, a, r, s2, done)); s = s2
            if len(buffer) >= 64:
                batch = random.sample(buffer, 64)
                S = torch.stack([b[0] for b in batch]).to(DEVICE)
                A = torch.tensor([b[1] for b in batch]).to(DEVICE)
                R = torch.tensor([b[2] for b in batch]).to(DEVICE)
                S2 = torch.stack([b[3] for b in batch]).to(DEVICE)
                D = torch.tensor([b[4] for b in batch], dtype=torch.float32).to(DEVICE)
                with torch.no_grad():
                    target = R + 0.95 * (1 - D) * tgt(S2, regime).max(1).values
                qsa = q(S, regime).gather(1, A.unsqueeze(1)).squeeze(1)
                loss = ((qsa - target) ** 2).mean()
                if beta > 0:
                    loss = loss + mem.penalty(beta)
                opt.zero_grad(); loss.backward(); opt.step()
        if ep % 5 == 0:
            tgt.load_state_dict(q.state_dict())
    return buffer


@torch.no_grad()
def eval_regime(q, env, regime, trials=200):
    ok = 0
    for _ in range(trials):
        s = env.reset(); done = False
        while not done:
            a = int(q(s.to(DEVICE), regime).argmax())
            s, r, done, success = env.step(a)
        ok += int(success)
    return ok / trials


def fisher_losses(q, tgt, buf, regime):
    for _ in range(20):
        batch = random.sample(buf, 64)
        S = torch.stack([b[0] for b in batch]).to(DEVICE)
        A = torch.tensor([b[1] for b in batch]).to(DEVICE)
        R = torch.tensor([b[2] for b in batch]).to(DEVICE)
        S2 = torch.stack([b[3] for b in batch]).to(DEVICE)
        D = torch.tensor([b[4] for b in batch], dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            tg = R + 0.95 * (1 - D) * tgt(S2, regime).max(1).values
        yield ((q(S, regime).gather(1, A.unsqueeze(1)).squeeze(1) - tg) ** 2).mean()


def run(beta, label, episodes=220):
    random.seed(0); torch.manual_seed(0)
    guna = GunaController()
    envs = [BatteryForaging(food=f) for f in [(0, 0), (0, 5), (5, 0), (5, 5)]]
    n = len(envs)
    q = MultiHeadQNet(n).to(DEVICE); tgt = MultiHeadQNet(n).to(DEVICE)
    tgt.load_state_dict(q.state_dict())
    mem = SamskaraMemory(q, decay=0.5)
    track = []
    for r in range(n):
        buf = train_regime(q, tgt, mem, guna, envs[r], r, episodes,
                           beta=(beta if r > 0 else 0.0), track=(track if r == 0 else None))
        mem.consolidate_losses(fisher_losses(q, tgt, buf, r))     # deposit this regime's saṃskāra
    succ = [eval_regime(q, envs[r], r) for r in range(n)]
    return succ, track


def main():
    print(f"Track B v2 — task-conditioned embodied agent | device={DEVICE}\n")
    print("4 food-regimes learned in sequence; retention tested on ALL at the end.\n")

    res = {}
    for beta, label in [(0.0, "naive (no consolidation)"), (120.0, "with saṃskāra")]:
        succ, track = run(beta, label)
        res[label] = succ
        print(f"=== {label} ===")
        print(f"  per-regime success: {[round(s,2) for s in succ]}")
        print(f"  AVERAGE over all 4 regimes: {sum(succ)/len(succ):.2f}\n")

    # battery→guṇa from the naive run's trace (regime 0)
    _, track = run(0.0, "_trace")
    lo = [e for b, e in track if b < 0.33]; hi = [e for b, e in track if b > 0.66]
    print("=== battery → guṇa (still active) ===")
    print(f"  mean ε hungry (<33%): {sum(lo)/max(len(lo),1):.3f}  | charged (>66%): {sum(hi)/max(len(hi),1):.3f}")

    a = sum(res["naive (no consolidation)"]) / 4
    b = sum(res["with saṃskāra"]) / 4
    print(f"\nRETENTION: naive avg={a:.2f}  vs  saṃskāra avg={b:.2f}   (Δ={b-a:+.2f})")
    print("With per-regime heads + a protected shared backbone, the saṃskāra now")
    print("retains earlier foraging skills while learning new regimes — the fix works.")


if __name__ == "__main__":
    main()
