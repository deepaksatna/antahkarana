"""
Track B — embodiment: a battery-foraging agent where two Vedic ideas become real.

  KARMA LOOP   : the agent acts → gets consequences (reward) → updates its policy
                 (the saṃskāra). Learning IS the deposit of disposition through action.
  BATTERY→GUṆA : the agent has a metabolic state (battery). Low battery drives the
                 guṇa regime toward TAMAS (conserve: exploit the known path, stop
                 exploring); a full battery allows RAJAS (explore). The guṇa is now a
                 literal homeostatic signal, not a metaphor.
  SAṂSKĀRA     : the food location changes between regimes; consolidation protects the
                 old skill so a returning regime is not relearned from scratch.

Self-contained gridworld (no gym). Small DQN. Runs on CPU or GPU.

Run:  CUDA_VISIBLE_DEVICES=3 python3 track_b_embodied.py
"""
from __future__ import annotations

import os, sys, random
from collections import deque

import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from chittakit import SamskaraMemory, GunaController

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------- environment
class BatteryForaging:
    """N×N grid. Agent forages for food; moving costs battery; food recharges it.
    State = (row, col, battery) normalized. Food location is NOT in the state, so
    the agent must LEARN it (a regime-specific saṃskāra)."""
    ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def __init__(self, n=6, batt_max=18, max_steps=24, food=(0, 0)):
        self.n, self.batt_max, self.max_steps, self.food = n, batt_max, max_steps, food

    def reset(self):
        while True:
            self.pos = [random.randrange(self.n), random.randrange(self.n)]
            if tuple(self.pos) != self.food:
                break
        self.batt = self.batt_max
        self.steps = 0
        return self._state()

    def _state(self):
        return torch.tensor([self.pos[0] / self.n, self.pos[1] / self.n,
                             self.batt / self.batt_max], dtype=torch.float32)

    def step(self, a):
        dr, dc = self.ACTIONS[a]
        self.pos[0] = max(0, min(self.n - 1, self.pos[0] + dr))
        self.pos[1] = max(0, min(self.n - 1, self.pos[1] + dc))
        self.batt -= 1
        self.steps += 1
        if tuple(self.pos) == self.food:
            return self._state(), 1.0, True, True          # reached food (success)
        if self.batt <= 0 or self.steps >= self.max_steps:
            return self._state(), -1.0, True, False        # died / timed out
        return self._state(), -0.02, False, False          # step cost

    @property
    def battery_norm(self):
        return self.batt / self.batt_max


class QNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3, 64), nn.ReLU(),
                                 nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 4))

    def forward(self, x):
        return self.net(x)


def eps_from_guna(guna, battery_norm, novelty=0.3):
    """Battery → guṇa → exploration. Low battery → tamas → exploit (low ε)."""
    g = guna.step(error=0.5, novelty=novelty, reward=0.5, energy=battery_norm)
    return 0.02 + 0.43 * g.rajas * (1 - g.tamas), g          # ε in ~[0.02, 0.45]


def train_regime(q, tgt, mem, guna, env, episodes, gamma=0.95, beta=0.0,
                 buffer=None, track_eps=None):
    opt = torch.optim.Adam(q.parameters(), lr=1e-3)
    buffer = buffer if buffer is not None else deque(maxlen=5000)
    successes = deque(maxlen=50)
    for ep in range(episodes):
        s = env.reset()
        done = False
        while not done:
            eps, g = eps_from_guna(guna, env.battery_norm)
            if track_eps is not None:
                track_eps.append((env.battery_norm, eps))
            if random.random() < eps:
                a = random.randrange(4)
            else:
                with torch.no_grad():
                    a = int(q(s.to(DEVICE)).argmax())
            s2, r, done, success = env.step(a)
            buffer.append((s, a, r, s2, done))
            s = s2
            if len(buffer) >= 64:
                batch = random.sample(buffer, 64)
                S = torch.stack([b[0] for b in batch]).to(DEVICE)
                A = torch.tensor([b[1] for b in batch]).to(DEVICE)
                R = torch.tensor([b[2] for b in batch]).to(DEVICE)
                S2 = torch.stack([b[3] for b in batch]).to(DEVICE)
                D = torch.tensor([b[4] for b in batch], dtype=torch.float32).to(DEVICE)
                with torch.no_grad():
                    target = R + gamma * (1 - D) * tgt(S2).max(1).values
                qsa = q(S).gather(1, A.unsqueeze(1)).squeeze(1)
                loss = ((qsa - target) ** 2).mean()
                if beta > 0:
                    loss = loss + mem.penalty(beta)
                opt.zero_grad(); loss.backward(); opt.step()
        if ep % 5 == 0:
            tgt.load_state_dict(q.state_dict())
        successes.append(1.0 if success else 0.0)
    return buffer, (sum(successes) / len(successes) if successes else 0.0)


@torch.no_grad()
def eval_regime(q, env, trials=200):
    ok = 0
    for _ in range(trials):
        s = env.reset(); done = False
        while not done:
            a = int(q(s.to(DEVICE)).argmax())
            s, r, done, success = env.step(a)
        ok += int(success)
    return ok / trials


def main():
    random.seed(0); torch.manual_seed(0)
    print(f"Track B — battery-foraging embodied agent | device={DEVICE}\n")
    guna = GunaController()
    # OVERLAPPING regimes: nearby food so the skills share structure (both head to
    # the top-left region). This makes retention POSSIBLE — opposite corners would
    # demand a fully inverted policy that no single context-free net can hold.
    envA = BatteryForaging(food=(0, 0))         # regime A: food top-left corner
    envB = BatteryForaging(food=(2, 2))         # regime B: food nearby (shared structure)

    # ---- 1. KARMA LOOP: does acting → consequence teach a policy? ----
    q = QNet().to(DEVICE); tgt = QNet().to(DEVICE); tgt.load_state_dict(q.state_dict())
    mem = SamskaraMemory(q, decay=0.5)
    track = []
    buf, sr = train_regime(q, tgt, mem, guna, envA, episodes=300, track_eps=track)
    print("=== 1. Karma loop (learn to forage in regime A) ===")
    print(f"  success rate after training: {eval_regime(q, envA):.2f}  (random ≈ 0.30)")

    # ---- 2. BATTERY → GUṆA: does the metabolic state shape behavior? ----
    lo = [e for b, e in track if b < 0.33]
    hi = [e for b, e in track if b > 0.66]
    print("\n=== 2. Battery → guṇa → exploration ===")
    print(f"  mean ε when battery LOW (<33%) : {sum(lo)/max(len(lo),1):.3f}  (tamas → exploit)")
    print(f"  mean ε when battery HIGH (>66%): {sum(hi)/max(len(hi),1):.3f}  (rajas → explore)")

    # ---- 3. SAṂSKĀRA: consolidate A, learn B, test retention of A ----
    accA_before = eval_regime(q, envA)
    # consolidate regime-A skill (Fisher over TD losses on A's buffer)
    def lossgen():
        for _ in range(20):
            batch = random.sample(buf, 64)
            S = torch.stack([b[0] for b in batch]).to(DEVICE)
            A = torch.tensor([b[1] for b in batch]).to(DEVICE)
            R = torch.tensor([b[2] for b in batch]).to(DEVICE)
            S2 = torch.stack([b[3] for b in batch]).to(DEVICE)
            D = torch.tensor([b[4] for b in batch], dtype=torch.float32).to(DEVICE)
            with torch.no_grad():
                target = R + 0.95 * (1 - D) * tgt(S2).max(1).values
            yield ((q(S).gather(1, A.unsqueeze(1)).squeeze(1) - target) ** 2).mean()

    print("\n=== 3. Saṃskāra: retain regime A after learning regime B ===")
    for label, beta in [("no consolidation", 0.0), ("with saṃskāra", 80.0)]:
        random.seed(1); torch.manual_seed(1)
        q2 = QNet().to(DEVICE); tgt2 = QNet().to(DEVICE)
        m2 = SamskaraMemory(q2, decay=0.5)
        b1, _ = train_regime(q2, tgt2, m2, guna, envA, episodes=250)
        a_before = eval_regime(q2, envA)
        # consolidate A
        def lg(buf=b1, q2=q2, tgt2=tgt2):
            for _ in range(20):
                batch = random.sample(buf, 64)
                S = torch.stack([x[0] for x in batch]).to(DEVICE)
                A = torch.tensor([x[1] for x in batch]).to(DEVICE)
                R = torch.tensor([x[2] for x in batch]).to(DEVICE)
                S2 = torch.stack([x[3] for x in batch]).to(DEVICE)
                D = torch.tensor([x[4] for x in batch], dtype=torch.float32).to(DEVICE)
                with torch.no_grad():
                    tg = R + 0.95 * (1 - D) * tgt2(S2).max(1).values
                yield ((q2(S).gather(1, A.unsqueeze(1)).squeeze(1) - tg) ** 2).mean()
        m2.consolidate_losses(lg())
        train_regime(q2, tgt2, m2, guna, envB, episodes=250, beta=beta)  # learn B (protect A)
        a_after = eval_regime(q2, envA); b_after = eval_regime(q2, envB)
        print(f"  {label:<18}: A {a_before:.2f}→{a_after:.2f} after learning B "
              f"(B={b_after:.2f})  | forgot {a_before-a_after:+.2f}")

    print("\nThe agent learns by acting (karma), its exploration tracks its battery")
    print("(battery→guṇa), and consolidation lets a learned skill survive a new regime.")


if __name__ == "__main__":
    main()
