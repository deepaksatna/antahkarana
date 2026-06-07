# Roadmap — from validated components to a complete model (and on through Tracks B & C)

**Objective:** an AI model based on the Vedic components — and, because we don't want to leave any
track, on through embodiment (B) and neuromorphic efficiency (C). The order is fixed by dependency:
**you embody and port a *model*, not loose modules**, so integration is the keystone that unblocks
everything after it.

```
  ✅ COMPONENTS  →  ✅ INTEGRATION  →  ⏳ SCALE/REAL DATA  →  ⏳ TRACK B (embody)  →  ⏳ TRACK C (neuromorphic)
   (validated)      (the keystone)     (finish the model)     (physical agent)        (energy-optimal)
```

---

## ✅ Done — components, each validated in isolation
chitta (`SamskaraMemory`, EWC+decay), guṇa (`GunaController`), āśrama (`AshramaSchedule`), tapas
(`TapasController`), pramāṇa/divya-dṛṣṭi (`PramanaGate` + `sanjaya.py` privileged distillation),
turīya (`WitnessMonitor`), kleśa (avidyā entropy). See `RESULTS.md`.

## ✅ Done — integration (the keystone)
`chittakit.Antahkarana` + `integrated_agent.py`: **one agent, one wake/dream/sleep loop, all faculties
active together.** avg_acc 0.855, forgetting 0.008, gated accuracy 0.978; legible mind-state trace
(life-stages, guṇa mix, plasticity, witness drift). **This is the assembled model B and C extend.**

---

## 🟡 Phase II — finish the model (real backbone + real task + scale)  — *partly done*
*Still Track A. This is what makes it a "real" model rather than a synthetic proof.*
- ✅ **Real backbone + real task done** (`phase2_vision.py`, on 4× A10 GPU, opc@167.234.209.52):
  CNN backbone on **Split-CIFAR-10** and **Split-MNIST**. On hard CIFAR the agent beats naive
  (+4.6 pts acc, ~6× less forgetting: 0.165 → 0.028); on easy MNIST it over-regularizes (honest).
- 🟡 **Meta / forgetting-aware controller done** (`meta_guna.py`, `meta_train_guna.py`): the controller
  now scales protection to *measured* forgetting. On real data it **fixes the MNIST over-regularization**
  (0.974 → 0.986) while keeping the CIFAR win (lowest forgetting, 0.008). ES weight-search itself was
  neutral on the synthetic panel (honest); the win was the forgetting-aware *structure*. Still ⏳:
  meta-learn the **tapas** controller; richer ES on a non-capacity-bound benchmark.
- ⏳ Bigger backbone (ResNet/ViT), longer streams, a continual RL env (which also gives the *karma*
  action→consequence loop in simulation — no robot needed yet).
- **Needs:** 1 GPU (have it). **Deliverable:** the integrated agent works on real data at scale.

## ✅ Track B — embodiment (the model gets a body) — *core complete*
*Embodies the model; this is where the Vedic ideas become physically real.*
- ✅ **Karma loop** (`track_b_embodied.py`): a battery-foraging gridworld agent learns to forage purely
  by acting (success 1.00 vs random 0.30) — action → consequence → policy (saṃskāra).
- ✅ **Battery → guṇa**: exploration tracks the battery (ε 0.087 hungry → 0.122 charged) — the guṇa is a
  literal homeostatic signal, not a metaphor.
- ✅ **Continual-RL retention** (`track_b_v2.py`): with a **task-conditioned policy** (per-regime heads +
  protected shared backbone), the agent learns 4 food-regimes in sequence and retention is
  **naive 0.38 → saṃskāra 1.00** (Δ+0.62). The v1 context-free negative is fixed by the principled
  architecture (same as the supervised agent).
- ⏳ Next (Track-B polish, optional): **MuJoCo/Isaac/Gymnasium** (install on the GPU box) for richer
  physics; then a small real robot (Jetson) via **ROS 2** with real battery→guṇa and event-camera → manas.
- **Deliverable met:** an embodied, continually-learning agent whose "mood" (guṇa) tracks its real
  energy state, and which retains its skills across regimes.

## 🟡 Track C — neuromorphic (the same model, far less energy) — *software SNN done*
*Ports the model to event-driven computation — the efficiency endgame.*
- ✅ **Spiking perception done** (`track_c_spiking.py`, snnTorch on GPU): the perception net as an SNN
  (LIF neurons, rate coding, surrogate gradients) **matches the ANN's accuracy** (0.943 vs 0.929) at
  10.7% spike density. Conservative energy estimate **~1.9× cheaper** (per-op floor; real neuromorphic
  gap is larger). The chitta/guṇa faculties wrap the SNN's nn-params unchanged.
- ⏳ Next: convert more of the agent (the serial manas gate — a natural SNN fit for the 17-mind-moment
  discrete-time model); reduce timesteps / use temporal coding to widen the energy gap.
- ⏳ Hardware hop: map to **Loihi 2 / Akida** via **Intel Lava** for the milliwatt, always-on form —
  needs a neuromorphic board (research-grade).
- **Deliverable (software):** the spiking-friendly path is proven; hardware deployment is the last step.

---

## What "complete" means per goal
| If the goal is… | You are done at… |
|---|---|
| a working AI model on these components | **Phase II** (real backbone + task + scale) |
| a physical / embodied agent | **Track B** |
| an energy-optimal edge agent | **Track C** |

We are committed to all three; the sequence above is the only order that works. **Next concrete step:
Phase II** — plug a real backbone + Split-CIFAR (or a continual RL env) into the integrated agent.
