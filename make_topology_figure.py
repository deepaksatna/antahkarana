"""Render the Antaḥkaraṇa-Net architecture & topology as a polished PNG.

Trunk (L1->L2->L3) produces logits; a control ring (Guṇa->Āśrama->Tapas) sets
the learning dynamics; faculty layers (L4-L7) read features/logits/weights and
write back penalties, gates, and drift alarms.

    python3 make_topology_figure.py   ->  assets/topology.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

# ----------------------------------------------------------------------------- palette
BG       = "#0d1b2a"   # deep indigo background
RING     = "#1b263b"   # control-ring panel
TRUNK    = "#274060"   # trunk panel
C_CTRL   = "#e0a458"   # guṇa/āśrama/tapas  (amber – the mind)
C_TRUNK  = "#4cc9f0"   # manas/backbone/buddhi (cyan – the body)
C_MEM    = "#b5179e"   # chitta memory (magenta)
C_SELF   = "#90be6d"   # ahaṃkāra self (green)
C_GATE   = "#f9844a"   # pramāṇa gate (orange)
C_WIT    = "#c77dff"   # turīya witness (violet)
INK      = "#e9eef5"   # text
MUTE     = "#9fb3c8"   # secondary text

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10.5,
})

fig, ax = plt.subplots(figsize=(14.5, 11), dpi=170)
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")


def panel(x, y, w, h, color, alpha=0.55, lw=1.4, ec=None, r=0.025):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle=f"round,pad=0,rounding_size={r*100}",
                 fc=color, ec=ec or color, lw=lw, alpha=alpha, zorder=1))


def box(x, y, w, h, title, sub, fc, tcol=INK, r=2.2, fs_t=12, fs_s=8.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle=f"round,pad=0,rounding_size={r}",
                 fc=fc, ec="white", lw=1.6, alpha=0.96, zorder=3))
    ax.text(x + w / 2, y + h * 0.63, title, ha="center", va="center",
            color="#0d1b2a", fontsize=fs_t, fontweight="bold", zorder=4)
    ax.text(x + w / 2, y + h * 0.26, sub, ha="center", va="center",
            color="#0d1b2a", fontsize=fs_s, zorder=4, linespacing=1.15)


def arrow(p1, p2, color=INK, lw=2.0, style="-|>", mut=16, rad=0.0, ls="-", z=5):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=mut,
                 color=color, lw=lw, linestyle=ls,
                 connectionstyle=f"arc3,rad={rad}", zorder=z))


def label(x, y, s, color=MUTE, fs=8.6, style="italic", ha="center", weight="normal"):
    ax.text(x, y, s, ha=ha, va="center", color=color, fontsize=fs,
            style=style, fontweight=weight, zorder=6)


# ============================================================ TITLE
ax.text(50, 97.3, "Antaḥkaraṇa-Net", ha="center", color=INK,
        fontsize=23, fontweight="bold")
ax.text(50, 93.6, "Neural Architecture & Topology  ·  a trunk body governed by a Vedic control ring",
        ha="center", color=MUTE, fontsize=11.5, style="italic")

# ============================================================ CONTROL RING (mind)
panel(13, 78.5, 74, 12.2, RING, alpha=0.9, ec=C_CTRL, lw=1.8)
ax.text(15.4, 88.6, "CONTROL  RING   ·  the mind  (runs once per task)",
        color=C_CTRL, fontsize=11, fontweight="bold", va="center")

box(24, 80.3, 16, 6.2, "GUṆA", "(sattva, rajas, tamas)", C_CTRL, fs_t=12.5, fs_s=8.2)
box(46, 80.3, 16, 6.2, "ĀŚRAMA", "plasticity envelope", C_CTRL, fs_t=12.5, fs_s=8.2)
box(68, 80.3, 16, 6.2, "TAPAS", "replay budget", C_CTRL, fs_t=12.5, fs_s=8.2)

arrow((40, 83.4), (46, 83.4), color="#0d1b2a", lw=2.2)
arrow((62, 83.4), (68, 83.4), color="#0d1b2a", lw=2.2)

# input signals into the ring
label(8.5, 83.4, "novelty\nforgetting\nenergy", color=C_CTRL, fs=8.4, ha="center")
arrow((13.5, 83.4), (24, 83.4), color=C_CTRL, lw=2.0)

# ring -> trunk (hyperparameters)
for sx in (32, 54, 76):
    arrow((sx, 80.3), (sx, 70.2), color=C_CTRL, lw=1.8, rad=0.0, ls=(0, (5, 3)))
label(50, 75.2, "α  learning-rate     ·     β  protect strength     ·     replay allocation",
      color=C_CTRL, fs=9.2, style="italic", weight="bold")

# ============================================================ TRUNK (body) L1-L3
panel(13, 56.5, 74, 11.6, TRUNK, alpha=0.85, ec=C_TRUNK, lw=1.8)
ax.text(15.4, 65.9, "TRUNK  ·  the body  (forward path  x → logits)",
        color=C_TRUNK, fontsize=11, fontweight="bold", va="center")

box(17, 57.8, 19, 6.0, "L1 · MANAS", "attention /\nprecision gate", C_TRUNK, fs_t=11.5, fs_s=8.0)
box(40.5, 57.8, 19, 6.0, "L2 · BACKBONE φ", "shared feature\nencoder", C_TRUNK, fs_t=11.5, fs_s=8.0)
box(64, 57.8, 19, 6.0, "L3 · BUDDHI", "per-task heads\n+ decision", C_TRUNK, fs_t=11.5, fs_s=8.0)

# x input + internal trunk arrows + logits output
label(7.5, 60.8, "x", color=INK, fs=15, style="normal", weight="bold")
arrow((9.2, 60.8), (17, 60.8), color=INK, lw=2.4)
arrow((36, 60.8), (40.5, 60.8), color="#0d1b2a", lw=2.2)
arrow((59.5, 60.8), (64, 60.8), color="#0d1b2a", lw=2.2)
arrow((83, 60.8), (92.5, 60.8), color=INK, lw=2.4)
label(94.6, 60.8, "logits", color=INK, fs=10.5, style="normal", weight="bold")

# features tap-off downward
arrow((50, 57.8), (50, 51.5), color=C_TRUNK, lw=2.0)
label(57.5, 54.6, "features  f(x)", color=C_TRUNK, fs=9, style="italic")

# ============================================================ FACULTY LAYERS L4-L7
fy = 38.0
box(13.5, fy, 19, 9.2, "L4 · AHAṂKĀRA", "z_self latent\n(identity / self-model)", C_SELF, fs_t=11, fs_s=8.2)
box(40.5, fy, 19, 9.2, "L5 · CHITTA", "SamskaraMemory\nΩ · θ* · replay", C_MEM, tcol=INK, fs_t=11, fs_s=8.2)
box(67.5, fy, 19, 9.2, "L6 · PRAMĀṆA", "calibrated\nconfidence gate", C_GATE, fs_t=11, fs_s=8.2)

# feature distribution arrows to L4,L5,L6
arrow((50, 51.5), (23, 47.2), color=MUTE, lw=1.6, rad=-0.15)
arrow((50, 51.5), (50, 47.2), color=MUTE, lw=1.6)
arrow((50, 51.5), (77, 47.2), color=MUTE, lw=1.6, rad=0.15)

# L5 writes penalty back up to backbone (protect)
arrow((46, 47.2), (44, 57.8), color=C_MEM, lw=1.9, rad=0.25, ls=(0, (4, 2)))
label(34.5, 52.5, "β·ΣΩ(θ−θ*)²\npenalty", color=C_MEM, fs=8.0)

# L6 accept/abstain out
arrow((86.5, fy + 4.6), (95.5, fy + 4.6), color=C_GATE, lw=2.0)
label(91, fy + 7.6, "accept /\nabstain", color=C_GATE, fs=8.4)

# L7 witness
box(13.5, 24.5, 28, 8.4, "L7 · TURĪYA", "WitnessMonitor", C_WIT, fs_t=11, fs_s=8.6)
ax.text(27.5, 26.6, "drift = 1 − cos(probe_now, probe_imprint)",
        ha="center", va="center", color="#0d1b2a", fontsize=8.0, zorder=4)
arrow((23, fy), (23, 32.9), color=C_WIT, lw=1.8)
# witness -> safety alarm back to memory
arrow((41.5, 28.7), (50, 38), color=C_WIT, lw=1.7, rad=-0.25, ls=(0, (4, 2)))
label(50.5, 31.8, "drift alarm → raise β / halt", color=C_WIT, fs=8.0)

# ============================================================ FOUR OPERATING STATES strip
panel(52, 24.5, 35, 8.4, RING, alpha=0.9, ec=MUTE, lw=1.3)
ax.text(53.6, 31.4, "OPERATING STATES (per task cycle)", color=INK, fontsize=8.6,
        fontweight="bold", va="center")
states = [("jāgrat", "wake: train + replay"),
          ("svapna", "dream: exemplar rehearsal"),
          ("suṣupti", "sleep: consolidate Ω, θ*"),
          ("turīya", "witness: drift check")]
for i, (s, d) in enumerate(states):
    yy = 29.4 - i * 1.55
    ax.text(54.0, yy, f"•  {s}", color=C_CTRL, fontsize=8.2, va="center", fontweight="bold")
    ax.text(63.0, yy, d, color=MUTE, fontsize=7.9, va="center", style="italic")

# ============================================================ MEASURED RESULTS strip
panel(13, 6.5, 74, 13.5, RING, alpha=0.9, ec=C_TRUNK, lw=1.4)
ax.text(15.4, 17.6, "WHAT THE TOPOLOGY BUYS  ·  measured", color=C_TRUNK,
        fontsize=10.5, fontweight="bold", va="center")
metrics = [
    ("forgetting", "0.242 → 0.003", "~60–80× via Chitta L5"),
    ("gated accuracy", "0.978 @ 0.49 cov", "Pramāṇa L6 abstains"),
    ("identity drift", "≤ 0.008 / 8 tasks", "Turīya L7 holds self"),
    ("energy (spiking φ)", "0.52× ANN", "1.9× cheaper, no acc loss"),
]
colw = 18.2
for i, (k, v, note) in enumerate(metrics):
    x0 = 15.5 + i * colw
    ax.text(x0, 13.6, v, color=INK, fontsize=12.5, fontweight="bold", va="center")
    ax.text(x0, 11.0, k, color=C_TRUNK, fontsize=8.8, va="center", fontweight="bold")
    ax.text(x0, 8.9, note, color=MUTE, fontsize=7.8, va="center", style="italic")
    if i < 3:
        ax.add_line(Line2D([x0 + colw - 2.2, x0 + colw - 2.2], [8.2, 15.2],
                    color=MUTE, lw=0.6, alpha=0.4))

# ============================================================ LEGEND
leg = [("mind / control ring", C_CTRL), ("trunk body (L1–L3)", C_TRUNK),
       ("memory (L5)", C_MEM), ("self (L4)", C_SELF),
       ("gate (L6)", C_GATE), ("witness (L7)", C_WIT)]
lx = 13
for name, col in leg:
    ax.add_patch(plt.Rectangle((lx, 2.3), 1.6, 1.6, fc=col, ec="white", lw=0.5))
    ax.text(lx + 2.1, 3.1, name, color=MUTE, fontsize=7.8, va="center")
    lx += len(name) * 0.52 + 5.0

plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
fig.savefig("assets/topology.png", facecolor=BG, bbox_inches="tight", pad_inches=0.25)
print("wrote assets/topology.png")
