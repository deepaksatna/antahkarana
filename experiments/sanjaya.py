"""
Sañjaya — divya-dṛṣṭi acquired by privileged-information distillation, then gated.

Sañjaya is granted divine sight by Vyāsa so he can see the whole battlefield and
narrate it to the blind king. The AI analogue is PRIVILEGED-INFORMATION
DISTILLATION: a TEACHER that sees the full world state (god-view) trains a STUDENT
that sees only a partial observation, so the student learns to INFER the hidden
state it cannot directly perceive. (Established technique: LUPI / Vapnik;
"Learning by Cheating", Chen et al. 2019; asymmetric actor-critic.)

Setup (linear-Gaussian latent world):
    z ~ N(0, I)          latent factors of the world
    o = z·Wo + noise     the OBSERVED part   (what the student sees — the palace)
    h = z·Wh + noise     the HIDDEN part     (the distant battlefield)
    y = 1[u·h > 0]       the outcome depends on the HIDDEN state

Models, all evaluated on `o` ALONE at test time:
    teacher       — sees the full state [o, h]; an upper bound (god-view)
    student-direct— sees o, supervised only by the binary label y (1 bit)
    student-sañjaya — sees o, ALSO trained to infer h from o using the privileged
                      true-h target (a dense teaching signal) → learns to "see"
Then the Pramāṇa gate wraps the Sañjaya student so it reports only valid sight.

Run:  python3 sanjaya.py            # offline, CPU, ~30s, 5 seeds
"""
from __future__ import annotations

import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from chittakit import PramanaGate


def make_world(n, m=8, d_o=12, d_h=6, o_noise=0.4, h_noise=0.05, gen=None, Wo=None, Wh=None, u=None):
    # The student's observation o is NOISY (a murky view); the hidden h is clean and
    # decides y. So a god-view teacher is strong, an o-only student is limited, and
    # there is a real gap for privileged knowledge to transfer.
    z = torch.randn(n, m, generator=gen)
    o = z @ Wo + o_noise * torch.randn(n, d_o, generator=gen)
    h = z @ Wh + h_noise * torch.randn(n, d_h, generator=gen)
    y = (h @ u > 0).long()
    return o, h, y


class MLP(nn.Module):
    def __init__(self, d_in, d_out, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, d_out))

    def forward(self, x):
        return self.net(x)


class Sanjaya(nn.Module):
    """Shared trunk on o; one head infers h (privileged target), one predicts y."""
    def __init__(self, d_o, d_h, hidden=64):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(d_o, hidden), nn.ReLU(),
                                   nn.Linear(hidden, hidden), nn.ReLU())
        self.h_head = nn.Linear(hidden, d_h)     # "see" the hidden state
        self.y_head = nn.Linear(hidden, 2)

    def forward(self, o):
        f = self.trunk(o)
        return self.y_head(f), self.h_head(f)


def train(model, X, Y, epochs, lr=1e-3, wd=1e-3, aux=None, aux_w=1.0):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    for _ in range(epochs):
        for i in range(0, len(X), 64):
            xb = X[i:i+64]; yb = Y[i:i+64]
            opt.zero_grad()
            if aux is None:
                loss = F.cross_entropy(model(xb), yb)
            else:
                ylog, hhat = model(xb)
                loss = F.cross_entropy(ylog, yb) + aux_w * F.mse_loss(hhat, aux[i:i+64])
            loss.backward(); opt.step()


def acc(logits, y):
    return float((logits.argmax(1) == y).float().mean())


def run(seed, n_train=400, epochs=60):
    g = torch.Generator().manual_seed(seed)
    m, d_o, d_h = 8, 12, 6
    Wo = torch.randn(m, d_o, generator=g)
    Wh = torch.randn(m, d_h, generator=g)
    u = torch.randn(d_h, generator=g); u = u / u.norm()

    otr, htr, ytr = make_world(n_train, m, d_o, d_h, gen=g, Wo=Wo, Wh=Wh, u=u)
    ote, hte, yte = make_world(3000, m, d_o, d_h, gen=g, Wo=Wo, Wh=Wh, u=u)
    oval, hval, yval = make_world(1000, m, d_o, d_h, gen=g, Wo=Wo, Wh=Wh, u=u)

    torch.manual_seed(seed)
    # teacher: god-view on [o, h]
    teacher = MLP(d_o + d_h, 2)
    train(teacher, torch.cat([otr, htr], 1), ytr, epochs)
    teach_acc = acc(teacher(torch.cat([ote, hte], 1)), yte)
    with torch.no_grad():
        teacher_soft = teacher(torch.cat([otr, htr], 1)).detach()  # privileged soft labels

    # student-direct: o -> y only (hard labels)
    direct = MLP(d_o, 2)
    train(direct, otr, ytr, epochs)
    direct_acc = acc(direct(ote), yte)

    # student-sañjaya: o -> y, distilling the god-view teacher's soft labels ("dark
    # knowledge" carrying hidden-state info) + a light privileged h-inference aux.
    sanj = Sanjaya(d_o, d_h)
    opt = torch.optim.Adam(sanj.parameters(), lr=1e-3, weight_decay=1e-3)
    Tkd, alpha, aux_w = 3.0, 0.7, 0.2
    for _ in range(epochs):
        for i in range(0, len(otr), 64):
            ob, yb, hb = otr[i:i+64], ytr[i:i+64], htr[i:i+64]
            ts = teacher_soft[i:i+64]
            opt.zero_grad()
            ylog, hhat = sanj(ob)
            hard = F.cross_entropy(ylog, yb)
            soft = F.kl_div(F.log_softmax(ylog / Tkd, 1), F.softmax(ts / Tkd, 1),
                            reduction="batchmean") * (Tkd ** 2)
            loss = (1 - alpha) * hard + alpha * soft + aux_w * F.mse_loss(hhat, hb)
            loss.backward(); opt.step()
    with torch.no_grad():
        sanj_logits_te, _ = sanj(ote)
    sanj_acc = acc(sanj_logits_te, yte)

    # Pramāṇa gate on the Sañjaya student: calibrate on val, gate a mixed test where
    # half the cases are 'fog of war' (heavy obs noise → hidden state not recoverable).
    with torch.no_grad():
        val_logits, _ = sanj(oval)
    gate = PramanaGate(threshold=0.9)
    T = gate.calibrate(val_logits, yval)

    # fog of war: the view collapses (near-zero input) — no signal reaches the seer,
    # so it is honestly uncertain and the Pramāṇa gate should abstain.
    ofog = 0.1 * torch.randn(3000, d_o, generator=g)
    yfog = (torch.rand(3000, generator=g) > 0.5).long()
    o_mix = torch.cat([ote, ofog]); y_mix = torch.cat([yte, yfog])
    fog = torch.cat([torch.zeros(len(yte), dtype=torch.bool), torch.ones(len(yfog), dtype=torch.bool)])
    with torch.no_grad():
        mix_logits, _ = sanj(o_mix)
    p = F.softmax(mix_logits / T, 1); conf, pred = p.max(1)
    blanket = acc(mix_logits, y_mix)
    accept = conf >= gate.threshold
    cov = float(accept.float().mean())
    gated_acc = float((pred[accept] == y_mix[accept]).float().mean()) if accept.any() else float("nan")
    fog_in_accept = float(fog[accept].float().mean()) if accept.any() else float("nan")

    return dict(teacher=teach_acc, direct=direct_acc, sanjaya=sanj_acc,
                gate_cov=cov, gate_acc=gated_acc, blanket_mix=blanket, fog_accept=fog_in_accept)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--n-train", type=int, default=150)
    args = ap.parse_args()
    print(f"privileged-information distillation | n_train={args.n_train} | seeds={args.seeds}\n")

    keys = ["teacher", "direct", "sanjaya", "gate_cov", "gate_acc", "blanket_mix", "fog_accept"]
    agg = {k: [] for k in keys}
    for s in range(args.seeds):
        r = run(s, n_train=args.n_train)
        for k in keys:
            agg[k].append(r[k])
    def ms(xs):
        mn = sum(xs) / len(xs)
        return mn, (sum((x - mn) ** 2 for x in xs) / len(xs)) ** 0.5

    print("=== 1. Does privileged-info distillation grant sight? (test on o ALONE) ===")
    for k, label in [("direct", "student-direct  (o→y, 1-bit label)"),
                     ("sanjaya", "student-sañjaya (o→y + infer hidden h)"),
                     ("teacher", "teacher         (god-view [o,h] — upper bound)")]:
        mn, sd = ms(agg[k])
        print(f"  {label:<42}: {mn:.3f} ± {sd:.3f}")
    gain = (ms(agg['sanjaya'])[0] - ms(agg['direct'])[0])
    closed = gain / max(ms(agg['teacher'])[0] - ms(agg['direct'])[0], 1e-6)
    print(f"  → Sañjaya recovers {100*closed:.0f}% of the teacher-minus-direct gap "
          f"(+{100*gain:.1f} pts over direct) by learning to see the hidden state.")

    print("\n=== 2. Pramāṇa gate on a mixed clear+fog-of-war stream (50% blinding fog) ===")
    print(f"  blanket accuracy (no gate)   : {ms(agg['blanket_mix'])[0]:.3f}")
    print(f"  coverage (accepted fraction) : {ms(agg['gate_cov'])[0]:.3f}")
    print(f"  accuracy ON ACCEPTED         : {ms(agg['gate_acc'])[0]:.3f}")
    print(f"  fog share among ACCEPTED     : {ms(agg['fog_accept'])[0]:.3f}   (vs 0.50 in stream)")
    print("\nReading: the Sañjaya student, trained with a god-view teacher's privileged")
    print("hidden-state target, learns to SEE more of the battlefield than the o-only")
    print("student — and the Pramāṇa gate makes it abstain in the fog instead of guessing.")


if __name__ == "__main__":
    main()
