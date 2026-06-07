"""
Divya-dṛṣṭi demo — extended perception, gated by validity (Pramāṇa).

Story: an agent must "see" a hidden truth y from indirect features. When the
signal is present it can see (divya-dṛṣṭi works); when the signal is occluded it
is BLIND (like the blind king Dhṛtarāṣṭra) and should NOT pretend to see. The
Pramāṇa Gate calibrates the agent's confidence and lets it report only what it
genuinely perceives — abstaining on the blind cases instead of hallucinating.

We show:
  1. The faculty sees: high accuracy on clear inputs, ~chance on occluded ones.
  2. Naive use HALLUCINATES on a mixed stream (confidently wrong on the blind half).
  3. The Pramāṇa Gate fixes it: accepted predictions are far more accurate, and the
     gate abstains preferentially on exactly the occluded (blind) inputs.

Run:  python3 divya_drsti.py            # offline, CPU, seconds
"""
from __future__ import annotations

import os, sys
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from chittakit import PramanaGate


def make_data(n, dim=20, n_signal=4, mu=1.3, noise=1.0, occlude_frac=0.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    y = (torch.rand(n, generator=g) > 0.5).long()
    x = torch.randn(n, dim, generator=g) * noise
    sign = (2 * y - 1).float().unsqueeze(1)
    x[:, :n_signal] += sign * mu                       # the perceptible signal
    occluded = torch.zeros(n, dtype=torch.bool)
    if occlude_frac > 0:
        k = int(n * occlude_frac)
        occluded[:k] = True
        x[:k, :n_signal] = torch.randn(k, n_signal, generator=g) * noise  # blind: signal gone
        perm = torch.randperm(n, generator=g)
        x, y, occluded = x[perm], y[perm], occluded[perm]
    return x, y, occluded


class Seer(nn.Module):
    def __init__(self, dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 2))

    def forward(self, x):
        return self.net(x)


def acc(logits, y):
    return float((logits.argmax(1) == y).float().mean())


def main():
    torch.manual_seed(0)
    dim = 20
    Xtr, Ytr, _ = make_data(4000, dim=dim, seed=1)
    Xval, Yval, _ = make_data(1500, dim=dim, seed=2)               # clean, for calibration
    Xte, Yte, occl = make_data(3000, dim=dim, occlude_frac=0.4, seed=3)  # 40% blind

    model = Seer(dim, hidden=32)
    # weight decay keeps the seer HONEST: it won't manufacture confidence from the
    # noise dimensions, so when the signal is occluded it is genuinely uncertain.
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=2e-3)
    lossf = nn.CrossEntropyLoss()
    for _ in range(25):
        for i in range(0, len(Xtr), 128):
            xb, yb = Xtr[i:i+128], Ytr[i:i+128]
            opt.zero_grad(); loss = lossf(model(xb), yb); loss.backward(); opt.step()

    model.eval()
    with torch.no_grad():
        val_logits = model(Xval)
        te_logits = model(Xte)

    # 1) The faculty sees where there is signal, is blind where occluded
    clear_mask = ~occl
    print("=== 1. Does the sight work? ===")
    print(f"  accuracy on CLEAR inputs   : {acc(te_logits[clear_mask], Yte[clear_mask]):.3f}  (sees the hidden y)")
    print(f"  accuracy on OCCLUDED inputs: {acc(te_logits[occl], Yte[occl]):.3f}  (~chance: it is blind here)")

    # 2) Naive use on the mixed stream — hallucination on the blind half
    print("\n=== 2. Naive (no gate) on the mixed stream ===")
    print(f"  blanket accuracy (all 3000): {acc(te_logits, Yte):.3f}  ← dragged down by confident-but-wrong blind cases")

    # 3) Pramāṇa Gate: calibrate, then accept only valid sight
    gate = PramanaGate(threshold=0.85)
    ece_before = PramanaGate.expected_calibration_error(val_logits, Yval, T=1.0)
    T = gate.calibrate(val_logits, Yval)
    ece_after = PramanaGate.expected_calibration_error(val_logits, Yval, T=T)
    print("\n=== 3. Pramāṇa Gate (validity) ===")
    print(f"  calibration: T={T:.2f}  ECE {ece_before:.3f} → {ece_after:.3f} (lower = confidence is trustworthy)")

    # risk–coverage sweep: as the validity bar rises, accepted accuracy rises and
    # the blind (occluded) cases are filtered out.
    import torch.nn.functional as F
    p = F.softmax(te_logits / T, dim=-1)
    conf, pred = p.max(dim=-1)
    print(f"  {'threshold':>10}{'coverage':>10}{'accepted_acc':>14}{'occl%accepted':>15}{'occl_abstained':>16}")
    for thr in (0.50, 0.70, 0.80, 0.90, 0.95, 0.99):
        acc_mask = conf >= thr
        cov = float(acc_mask.float().mean())
        if acc_mask.any():
            sa = float((pred[acc_mask] == Yte[acc_mask]).float().mean())
            oa = float(occl[acc_mask].float().mean())
        else:
            sa = oa = float("nan")
        oab = float((~acc_mask[occl]).float().mean()) if occl.any() else float("nan")
        print(f"  {thr:>10.2f}{cov:>10.3f}{sa:>14.3f}{oa:>15.3f}{oab:>16.3f}")
    print(f"  (blanket accuracy, no gate = {acc(te_logits, Yte):.3f}; occluded share in stream = 0.40)")

    print("\nReading: the faculty genuinely perceives the hidden y when signal is present")
    print("(divya-dṛṣṭi), but on blind/occluded inputs it must not hallucinate. The Pramāṇa")
    print("Gate — calibrated confidence ≥ threshold — accepts the valid sight and abstains")
    print("on the blind cases, turning raw perception into a VALID means of knowledge.")


if __name__ == "__main__":
    main()
