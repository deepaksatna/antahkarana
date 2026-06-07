# Results — honest writeup (Track A prototype)

Same spirit as the rest of this folder: **report what reproduces, flag what's marginal, name what's
next.** No overclaiming. All numbers are 5-seed means ± std on CPU.

---

## ★★ Phase II — the agent on a REAL CNN + REAL images, on GPU (4× A10)

`phase2_vision.py` runs the **same integrated agent** with a **convolutional backbone** on real
torchvision datasets (task-incremental, 5 tasks × 2 classes, one head per task), on an NVIDIA A10.
Three arms: naive CNN, agent with the **rule-based** guṇa controller, and agent with the
**forgetting-aware** controller (`MetaGunaController`, which scales protection to *measured* forgetting).

| | Split-MNIST (easy) acc ↑ | Split-CIFAR-10 (hard) acc ↑ | CIFAR forgetting ↓ |
|---|---|---|---|
| naive CNN | 0.995 | 0.726 | 0.217 |
| agent (rule guṇa) | 0.974 | 0.827 | 0.033 |
| **agent (forgetting-aware)** | **0.986** | 0.816 | **0.008** |

**Three findings, all honest:**
1. **Backbone-agnostic ✅** — the agent runs *unchanged* on a real CNN over real images on GPU; only
   the backbone swapped (MLP → CNN). The control layer scales as the architecture claimed.
2. **The agent helps in proportion to the forgetting.** On **hard CIFAR** (naive forgets 0.217) both
   agents crush forgetting (→ 0.008–0.033) and add **+9–10 pts** accuracy. On **easy MNIST** (naive
   barely forgets) the *rule-based* agent over-regularizes and trails naive (0.974 vs 0.995).
3. **The forgetting-aware controller fixes the over-regularization.** Seeing forgetting ≈ 0 on MNIST it
   drops protection (β → 0.20) and **recovers accuracy (0.974 → 0.986)**; seeing high forgetting on
   CIFAR it protects hardest, giving the **lowest forgetting of all (0.008)**. It trades a hair of
   CIFAR accuracy (0.816 vs 0.827) for being the only arm that behaves well across *both* regimes —
   the adaptive policy the rule-based controller structurally couldn't express.

> **On the ES meta-training itself (honest):** the evolution-strategy run (`meta_train_guna.py`) *tied*
> the hand-set prior on the synthetic panel (fitness flat ~0.61) — that toy is capacity-bound and
> doesn't reproduce the over-regularization failure, so there was little for ES to gain. The real win
> came from the **architectural change** the meta-controller embodies — *making protection
> forgetting-aware* — validated above on real MNIST/CIFAR, not from the weight search.

## ★ Track B — embodiment: the battery-foraging agent (`track_b_embodied.py`)

A self-contained gridworld (no gym) where two Vedic ideas become literal. On GPU:

| deliverable | result | status |
|---|---|---|
| **Karma loop** (act → consequence → policy) | forage success **1.00** vs random ≈0.30 | ✅ clean |
| **Battery → guṇa** (metabolic state → behavior) | ε = **0.087** when hungry (tamas→exploit) vs **0.122** when charged (rajas→explore) | ✅ clean |
| **Saṃskāra retention** (v1: context-free policy) | A 1.00 → 0.02 (consolidation couldn't help) | ❌ → fixed in v2 |
| **Saṃskāra retention** (v2: task-conditioned policy) | **naive 0.38 vs saṃskāra 1.00 over 4 regimes (Δ+0.62)** | ✅ fixed |

**All three embodiment goals now succeed.** The agent learns purely by acting (karma loop), its
exploration is driven by its battery (*the guṇa is a real homeostatic signal, not a metaphor*), and —
with the v2 fix — it retains its skills across regimes.

**The v1 retention miss, and the v2 fix (`track_b_v2.py`).** Context-free continual RL is genuinely
harder than supervised continual learning: with no task-context, distinct food-regimes are
*contradictory functions of the same state*, so a single policy net cannot hold them — EWC can only
freeze one (blocking the next). The principled fix is the same one that works on CIFAR: a **shared
navigation backbone + per-regime heads**. With it, `track_b_v2.py` learns 4 food-regimes in sequence
and retention jumps from **naive 0.38 → saṃskāra 1.00** (all four regimes kept perfectly). So the
saṃskāra mechanism *does* carry over to RL — it just needed the task-conditioned policy, not a
context-free one.

## ★ Track C — neuromorphic: the spiking perception net (`track_c_spiking.py`)

The manas-style perception front-end converted to a **spiking neural network** (LIF neurons,
rate-coded input, surrogate-gradient training, snnTorch) on MNIST, vs a same-size dense ANN. On GPU:

| | result |
|---|---|
| **Accuracy** — does the spiking path work? | spiking **0.943** vs dense ANN **0.929** ✅ (no accuracy cost) |
| **Sparsity** | only **10.7%** of hidden neurons fire per timestep |
| **Energy estimate** (45 nm CMOS, Horowitz 2014) | SNN ≈ **51.7%** of ANN energy (**~1.9× cheaper**) 🟡 |

**Honest reading.** The spiking path **works** — comparable accuracy at a small fraction of the
activation density. The energy number is **modest and conservative on purpose**: the SNN does *more*
synaptic ops (537k ACs vs 203k MACs — the cost of T=20 timesteps of rate-coded spikes), but each
accumulate is ~5× cheaper than a MAC, netting ~1.9×. The headline "100–1000×" neuromorphic figures come
from effects this software estimate deliberately **does not** count (event-skipping of zero activity,
in-memory compute, temporal coding, no off-chip memory movement) and require a real chip (Loihi/Akida).
So this is the *floor*, measured, not the ceiling, marketed. And crucially: **the chitta/guṇa/tapas
faculties wrap the SNN's nn-params unchanged** — the continual-learning machinery carries to the spiking
substrate. Software-SNN is the honest reachable milestone; hardware deployment is the remaining hop.

## ★ The integrated agent — all faculties in one loop (the keystone)

`integrated_agent.py` runs **one `Antahkarana` agent** through a continual stream with *every* faculty
active together — chitta + guṇa + āśrama + tapas + pramāṇa + turīya in a single wake/dream/sleep loop
(3 seeds, 8 tasks):

| metric | value |
|---|---|
| avg_acc | 0.855 ± 0.012 |
| forgetting | **0.008 ± 0.005** |
| pramāṇa gated coverage | 0.49 |
| accuracy on accepted (gated) | **0.978 ± 0.003** |

It learns the stream continually (near-zero forgetting), and its **mind-state is legible**: the trace
shows it move *brahmacarya → gṛhastha → vānaprastha*, the guṇa (s,r,t) mix and learning rate adapt to
novelty, plasticity headroom shrinks over the lifetime (0.26 → 0.09, consolidating), and the turīya
witness tracks identity drift each task. When uncertain it **abstains** (pramāṇa) and its accepted
perceptions are 0.978-accurate. This is the assembled model that Tracks B (embodiment) and C
(neuromorphic) extend — see `ROADMAP.md`.

## The benchmark journey (and why it matters)

The first benchmark (`continual_benchmark.py`, single shared head, fully-conflicting tasks) reliably
showed **catastrophic forgetting** and that **consolidation cures it ~3.4–7×** — but it was
*capacity-saturated*: with one head and contradictory tasks, average accuracy is pinned near chance,
so the *subtler* mechanisms (importance **decay**, **tapas** allocation) had no room to show. That is
a real property of the benchmark, not the method — diagnosed, not hand-waved.

So we built the benchmark `RESULTS.md` originally promised: **`capacity_benchmark.py`** — a
**shared-feature, multi-head** stream where all tasks share one nonlinear teacher φ, so a backbone that
learns φ can solve *all* tasks (real headroom), while naive sequential training overwrites it
(forgetting). *This* is where the mechanisms differentiate.

---

## Headline results — `capacity_benchmark.py` (8 tasks, 5 seeds) ✅

| regime | avg_acc ↑ | forgetting ↓ | worst_task ↑ |
|---|---|---|---|
| naive | 0.764 | 0.242 | 0.579 |
| ewc | 0.860 | 0.004 | 0.748 |
| **samskara** (decay) | **0.867** | **0.003** | **0.762** |
| replay-uniform | 0.882 | 0.103 | 0.791 |
| replay-tapas | 0.883 | 0.101 | 0.783 |

**What this shows, cleanly:**
1. **Consolidation eliminates forgetting** — 0.242 → 0.003–0.004, a **~60–80× reduction**, while
   lifting average accuracy 0.764 → 0.86+. Large, robust, low-variance.
2. **saṃskāra (importance decay) beats plain EWC on all three metrics** — avg_acc 0.867 > 0.860,
   worst_task 0.762 > 0.748, forgetting 0.003 < 0.004. Small but **consistent and in the predicted
   direction**: letting old importance fade keeps the backbone plastic enough to keep improving. This
   is the decay claim that the saturated benchmark *couldn't* show — now visible in the right setting.
3. **Replay reaches the highest average accuracy** (0.88) — rehearsing real exemplars refines the
   shared backbone the most (at the cost of slightly higher peak-relative forgetting).

## Tapas — concentrate effort where you're weakest ✅ (small, consistent)

Tapas-replay vs. uniform-replay at **equal** rehearsal budget:

| setting | regime | avg_acc ↑ | forgetting ↓ | worst_task ↑ |
|---|---|---|---|---|
| budget 64, 8 tasks | uniform | 0.882 | 0.103 | 0.791 |
| | tapas | 0.883 | 0.101 | 0.783 |
| **tight** budget 20, 10 tasks | uniform | 0.862 | 0.117 | 0.793 |
| | **tapas** | **0.866** | **0.112** | **0.798** |

With a generous budget, tapas ≈ uniform (when you can rehearse everything, *how* you split it barely
matters). With a **tight budget** — the regime where concentration should matter — **tapas edges out
uniform on all three metrics**. Small, but consistent and in the predicted direction: *apply the heat
where the ore is hardest to melt*. (It should widen further with **heterogeneous task difficulty** —
the next test.)

## Divya-dṛṣṭi + Pramāṇa gate — extended perception that doesn't hallucinate ✅

`divya_drsti.py`: a "seer" infers a hidden truth `y` from indirect features. On clear inputs it sees
(acc **0.992**); on occluded/blind inputs it is genuinely uncertain (acc **0.501**, chance). Naive use
on a mixed stream (40% blind) hallucinates → blanket acc **0.795**. The **Pramāṇa Gate** (calibrated
confidence ≥ threshold; ECE 0.002) turns raw perception into a *valid pramāṇa* — risk–coverage sweep:

| threshold | coverage | accepted_acc | occluded-% in accepted | occluded abstained |
|---|---|---|---|---|
| 0.50 | 1.000 | 0.795 | 0.400 | 0.000 |
| 0.90 | 0.864 | 0.839 | 0.316 | 0.318 |
| 0.99 | 0.699 | **0.905** | **0.193** | **0.663** |

As the validity bar rises, accepted accuracy climbs **0.795 → 0.905** and the gate **filters out the
blind cases** (occluded share among accepted 0.40 → 0.19; abstains on 66% of blind inputs). The faculty
*sees what it can* and *abstains rather than confabulate* — the epistemic discipline (yogaja-pratyakṣa
as a valid pramāṇa) made concrete, and a clean anti-hallucination primitive.

## Sañjaya — privileged-information distillation grants sight, then gated ✅ (modest + clean)

`sanjaya.py` (8 seeds, n_train=150): a **god-view teacher** (sees full state `[o, h]`) distills to a
**student that sees only the murky observation `o`**, training it to infer the hidden state (Sañjaya
seeing the far battlefield). Tested on `o` alone:

| model | accuracy |
|---|---|
| student-direct (o→y, hard labels) | 0.895 |
| **student-sañjaya** (o→y + KD from teacher + infer h) | **0.906** |
| teacher (god-view [o,h], upper bound) | 0.936 |

The privileged channel transfers **+1.1 pts** (recovers ~27% of the teacher gap), and the gain
**grows as data shrinks** (the classic LUPI / dark-knowledge regularization regime) — modest but
consistent. Then the **Pramāṇa gate** on a 50%-fog stream: blanket acc 0.701 → **accepted acc 0.983**
at coverage 0.356, with **0% of accepted cases being fog** — in the fog it abstains completely instead
of guessing. Extended perception that is *both earned and validated*.

## The controllers run and behave as designed ✅

Verbose runs of `continual_benchmark.py --seeds 1` show the guṇa/āśrama machinery working: the system
moves `brahmacarya → gṛhastha → vānaprastha`, the avidyā-entropy novelty signal drives the (s,r,t)
mix, **critical periods re-open** (`reopened=True`) on distribution shift, and the learning rate
tracks all of it interpretably.

---

## Honest caveats ⚠️

- **The novel-mechanism wins (decay, tapas) are small** (~0.5–1 accuracy point) though consistent
  across seeds and metrics. They are **directional evidence**, not yet a decisive result — the *large*
  wins here belong to consolidation and replay, which are established methods.
- **Synthetic data.** φ-teacher tasks are a clean testbed, not real-world. Split-MNIST/CIFAR
  (`--dataset mnist` scaffold in place) is the next rung.
- **Tapas needs heterogeneous difficulty to shine** — the current φ-tasks are roughly equal
  difficulty, which caps the achievable tapas advantage.

## Next to make the novel claims decisive
1. **Heterogeneous-difficulty tasks** (vary margin / data-per-task) → tapas should win clearly.
2. **Longer streams (30–50 tasks)** → EWC's ossification compounds; plot plasticity vs. task index.
3. **Split-MNIST/CIFAR** multi-head (real images).
4. **Forward-transfer / learning-speed** metrics (the decay & tapas benefits partly hide inside
   *how fast* new tasks are learned, which final-accuracy averages obscure).
5. **Meta-learn the guṇa & tapas controllers** (PBT / meta-gradient).

## Component scorecard — what carries weight vs. what's scaffolding
*Keeping the architecture honest: not every faculty is a proven win. We keep the full set (the Vedic
model is the point), but the defaults use what works, and we don't oversell the rest.*

| Faculty | Status | Evidence |
|---|---|---|
| **chitta** (SamskaraMemory) | 🟢 **strong, core** | kills forgetting ~6–80× on synthetic + real CIFAR |
| **replay** (dream/sleep) | 🟢 **strong** | highest accuracy; large forgetting reduction |
| **pramāṇa** (divya-dṛṣṭi gate) | 🟢 **strong, clean** | calibrated abstention; fog filtered to 0% |
| **forgetting-aware guṇa** (MetaGuna) | 🟢 **good — now the default** | fixed MNIST over-regularization, kept CIFAR win |
| **saṃskāra decay** vs plain EWC | 🟡 moderate | consistent but small (~0.5–1 pt) |
| **tapas** | 🟡 moderate | small, config-dependent (wins under tight budget) |
| **sañjaya** (privileged distillation) | 🟡 modest | +1.1 pts, grows at low data |
| **rule-based GunaController** | ⚪ superseded | kept only as a baseline; MetaGuna replaces it |
| **āśrama** developmental schedule | ⚪ conceptual | runs, but its isolated benefit is **not yet demonstrated** (the novelty threshold over-triggers re-opening) — flagged, not claimed |
| **ES meta-training loop** | ⚪ neutral | tied the prior on the (capacity-bound) synthetic panel |
| **turīya** (WitnessMonitor) | ⚪ monitor only | tracks drift; doesn't affect learning (a safety log) |

**Action taken:** the agent now **defaults to the forgetting-aware controller**; the rule-based one is
demoted to a baseline. The 🟡/⚪ items are kept (architectural completeness) but explicitly **not
oversold** — āśrama and the ES loop need either a fix or an honest "conceptual" label, which they now
have.

## Bottom line
The prototype **works on real images on GPU**, the integrated agent runs all faculties in one loop, and
the strong components (consolidation, replay, pramāṇa, forgetting-aware control) carry it; the weaker
ones (decay, tapas, sañjaya) help modestly, and the conceptual ones (āśrama, ES, witness) are labeled
honestly rather than dressed up. An honest, falsifiable foundation — now extended toward Track B.
