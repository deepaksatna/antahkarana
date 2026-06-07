#!/usr/bin/env python3
"""Generate the README figures: banner, architecture diagram, and results panel.
All numbers are from the actual experiment runs (see RESULTS.md). Outputs → assets/."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

os.makedirs("assets", exist_ok=True)
C = {"navy": "#0e1630", "navy2": "#1a2547", "cyan": "#22d3ee", "saffron": "#f59e0b",
     "green": "#84cc16", "red": "#ef4444", "white": "#f8fafc", "grey": "#94a3b8",
     "violet": "#a78bfa"}


# ───────────────────────────────────────────────────────── banner
def banner():
    fig = plt.figure(figsize=(16, 5.2), dpi=120)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    for i in range(120):
        t = i / 120
        ax.add_patch(plt.Rectangle((0, 1 - (i + 1) / 120), 1, 1 / 120,
                     color=(0.055 + 0.05 * t, 0.086 + 0.06 * t, 0.19 + 0.06 * t), lw=0))
    ax.text(0.5, 0.80, "Antaḥkaraṇa-Net", color=C["white"],
            fontsize=46, fontweight="bold", ha="center")
    ax.text(0.5, 0.635, "A working AI built on the Vedic model of mind",
            color=C["cyan"], fontsize=22, ha="center", style="italic")
    ax.text(0.5, 0.545, "continual learning · without forgetting · effort-aware · self-monitoring · embodied · spiking",
            color=C["grey"], fontsize=13.5, ha="center")
    chips = [("~6–80×", "less forgetting", C["green"]),
             ("0.38 → 1.00", "embodied retention", C["saffron"]),
             ("0.943", "spiking accuracy", C["cyan"]),
             ("deep-research", "POC", C["violet"])]
    cw, gap = 0.205, 0.02
    x0 = 0.5 - (4 * cw + 3 * gap) / 2
    for i, (big, small, col) in enumerate(chips):
        x = x0 + i * (cw + gap)
        ax.add_patch(FancyBboxPatch((x, 0.16), cw, 0.21, boxstyle="round,pad=0.01,rounding_size=0.03",
                     facecolor=C["navy2"], edgecolor=col, lw=2.2))
        ax.text(x + cw / 2, 0.30, big, color=col, fontsize=21, fontweight="bold", ha="center")
        ax.text(x + cw / 2, 0.215, small, color=C["grey"], fontsize=12, ha="center")
    fig.savefig("assets/banner.png", facecolor=C["navy"]); plt.close(fig)


# ───────────────────────────────────────────────────────── architecture
def box(ax, x, y, w, h, title, sub, col):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.02",
                 facecolor=C["navy2"], edgecolor=col, lw=2.2))
    ax.text(x + w / 2, y + h - 0.045, title, color=col, fontsize=13, fontweight="bold", ha="center")
    ax.text(x + w / 2, y + 0.035, sub, color=C["grey"], fontsize=9.0, ha="center")


def arrow(ax, p, q, col=C["grey"]):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=14,
                 color=col, lw=1.8, shrinkA=2, shrinkB=2))


def architecture():
    fig = plt.figure(figsize=(15, 8.6), dpi=120)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, color=C["navy"]))
    ax.text(0.5, 0.955, "Antaḥkaraṇa-Net  —  the inner instrument as a cognitive architecture",
            color=C["white"], fontsize=18, fontweight="bold", ha="center")
    ax.text(0.5, 0.915, "Sanskrit faculty  →  ML module.  The backbone thinks; the Vedic layer remembers, regulates, perceives & monitors.",
            color=C["grey"], fontsize=11, ha="center")

    # main perception→decision pipeline
    box(ax, 0.03, 0.62, 0.165, 0.16, "MANAS", "attention gate\n(precision-weighted)", C["cyan"])
    box(ax, 0.255, 0.62, 0.175, 0.16, "CHITTA", "saṃskāra memory\n(EWC + decay)", C["green"])
    box(ax, 0.49, 0.62, 0.175, 0.16, "BUDDHI", "executive / decide\n(accumulate→bound)", C["saffron"])
    box(ax, 0.725, 0.62, 0.175, 0.16, "AHAṂKĀRA", "self-model\n(identity latent)", C["violet"])
    ax.text(0.012, 0.70, "world", color=C["white"], fontsize=11, ha="center", rotation=90)
    arrow(ax, (0.028, 0.70), (0.03, 0.70), C["white"])
    arrow(ax, (0.195, 0.70), (0.255, 0.70)); arrow(ax, (0.43, 0.70), (0.49, 0.70))
    arrow(ax, (0.665, 0.70), (0.725, 0.70))
    arrow(ax, (0.81, 0.62), (0.58, 0.62), C["violet"])   # self feeds back to buddhi
    ax.text(0.90, 0.70, "action", color=C["white"], fontsize=11, ha="center", rotation=90)
    arrow(ax, (0.90, 0.70), (0.93, 0.70), C["white"])

    # guṇa controller band
    box(ax, 0.03, 0.40, 0.87, 0.13, "GUṆA CONTROLLER   (sattva · rajas · tamas)",
        "one regime signal modulates plasticity, exploration, consolidation & pruning  ·  forgetting-aware  ·  battery→guṇa when embodied", C["red"])
    for x in (0.115, 0.34, 0.58, 0.81):
        arrow(ax, (x, 0.53), (x, 0.62), C["red"])

    # four-state cycle
    box(ax, 0.03, 0.18, 0.40, 0.14, "FOUR STATES (operating cycle)",
        "JĀGRAT wake → SVAPNA dream/replay → SUṢUPTI sleep/consolidate ↻", C["cyan"])
    # safety overlays
    box(ax, 0.47, 0.18, 0.20, 0.14, "PRAMĀṆA gate", "divya-dṛṣṭi validity\n(abstain, not hallucinate)", C["green"])
    box(ax, 0.70, 0.18, 0.20, 0.14, "TURĪYA witness", "reward-invariant\nidentity monitor", C["violet"])

    # āśrama lifelong band
    ax.add_patch(FancyBboxPatch((0.03, 0.05), 0.87, 0.08, boxstyle="round,pad=0.006,rounding_size=0.02",
                 facecolor="#11182e", edgecolor=C["saffron"], lw=1.6))
    ax.text(0.5, 0.09, "ĀŚRAMA developmental schedule  —  brahmacarya → gṛhastha → vānaprastha → saṃnyāsa   "
            "(lifelong plasticity, re-openable critical periods: 'always enhanceable, childhood → old age')",
            color=C["saffron"], fontsize=10.5, ha="center")
    fig.savefig("assets/architecture.png", facecolor=C["navy"]); plt.close(fig)


# ───────────────────────────────────────────────────────── results
def results():
    plt.rcParams.update({"axes.facecolor": "#11182e", "figure.facecolor": C["navy"],
                         "text.color": C["white"], "axes.labelcolor": C["grey"],
                         "xtick.color": C["grey"], "ytick.color": C["grey"],
                         "axes.edgecolor": C["grey"], "axes.titlecolor": C["white"]})
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), dpi=120)
    fig.suptitle("Antaḥkaraṇa-Net — key results (all from real runs; see RESULTS.md)",
                 fontsize=16, fontweight="bold")

    # 1. CIFAR forgetting
    ax = axes[0, 0]
    ax.bar(["naive", "agent"], [0.217, 0.008], color=[C["red"], C["green"]])
    ax.set_title("Phase II — Split-CIFAR forgetting ↓"); ax.set_ylabel("forgetting")
    ax.text(0, 0.22, "0.217", ha="center", color=C["white"]); ax.text(1, 0.02, "0.008 (~27×)", ha="center", color=C["white"])

    # 2. Track B retention
    ax = axes[0, 1]
    r = range(4)
    ax.bar([i - 0.18 for i in r], [0.10, 0.23, 0.20, 1.00], 0.36, label="naive", color=C["red"])
    ax.bar([i + 0.18 for i in r], [1, 1, 1, 1], 0.36, label="saṃskāra", color=C["green"])
    ax.set_title("Track B — embodied retention (4 regimes)"); ax.set_ylabel("success rate")
    ax.set_xticks(list(r)); ax.set_xticklabels(["A", "B", "C", "D"]); ax.legend(fontsize=8)

    # 3. divya-dṛṣṭi risk-coverage
    ax = axes[0, 2]
    cov = [1.0, 0.947, 0.917, 0.864, 0.825, 0.699]
    acc = [0.795, 0.808, 0.818, 0.839, 0.851, 0.905]
    ax.plot(cov, acc, "o-", color=C["cyan"], lw=2)
    ax.set_title("Divya-dṛṣṭi — Pramāṇa gate"); ax.set_xlabel("coverage (kept)"); ax.set_ylabel("accuracy on accepted")
    ax.invert_xaxis()

    # 4. spiking
    ax = axes[1, 0]
    ax2 = ax.twinx()
    ax.bar([0, 1], [0.929, 0.943], 0.5, color=[C["grey"], C["cyan"]])
    ax2.plot([0, 1], [935, 484], "s--", color=C["saffron"], lw=2)
    ax.set_title("Track C — spiking (accuracy = bars)"); ax.set_ylabel("accuracy"); ax.set_ylim(0.85, 0.97)
    ax2.set_ylabel("energy nJ (line)", color=C["saffron"]); ax.set_xticks([0, 1]); ax.set_xticklabels(["ANN", "SNN"])

    # 5. capacity benchmark forgetting
    ax = axes[1, 1]
    ax.bar(["naive", "EWC", "saṃskāra"], [0.242, 0.004, 0.003], color=[C["red"], C["saffron"], C["green"]])
    ax.set_title("Continual benchmark — forgetting ↓"); ax.set_ylabel("forgetting")

    # 6. MNIST over-reg fix
    ax = axes[1, 2]
    ax.bar(["naive", "rule\nguṇa", "forgetting\naware"], [0.995, 0.974, 0.986],
           color=[C["grey"], C["red"], C["green"]])
    ax.set_title("Forgetting-aware controller fix (MNIST)"); ax.set_ylabel("accuracy"); ax.set_ylim(0.95, 1.0)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig("assets/results.png"); plt.close(fig)


if __name__ == "__main__":
    banner(); architecture(); results()
    print("wrote assets/banner.png, assets/architecture.png, assets/results.png")
