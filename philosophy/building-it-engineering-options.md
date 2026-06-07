# Building the Antaḥkaraṇa-Net — Engineering Options, Libraries & the Energy Case
### What exists off-the-shelf, what we build ourselves (ChittaKit), how it touches the physical world, and why it saves energy vs. today's LLMs

> **Companion to** [`vedic-cognitive-ai-architecture.md`](vedic-cognitive-ai-architecture.md) (the
> design) and the four study files in this folder. This file is the **engineering plan**: concrete
> frameworks, the build-vs-buy split, embodiment, hardware, and a phased path with a recommendation.

> **Stance.** Nothing here requires inventing physics. ~80% of the Antaḥkaraṇa-Net (AKN) is
> **assembling mature libraries**; ~20% is a **thin new library (“ChittaKit”)** for the genuinely
> novel parts (the guṇa controller, the saṃskāra growth+decay rule, the āśrama schedule, the turīya
> monitor, the four-state orchestrator). We start in software to validate the novel claims, then
> embody it, then push to neuromorphic hardware for the energy endgame.

---

## 1. The limitations of today's AI that this design targets

Design choices below are *answers* to specific, well-documented failures of current large models —
this is the "reference from existing AI model limitations" the brief asks for:

| Limitation of current LLMs / deep nets | Consequence | AKN's structural answer |
|---|---|---|
| **No continual learning** — frozen after training | Can't learn on the job; knowledge goes stale | **chitta** store: incremental updates, never re-train from zero |
| **Catastrophic forgetting** — fine-tuning erases old skills | Must retrain on everything, every time | **saṃskāra Ω** protects important weights; grows + decays |
| **Full-retrain energy cost** — GPT-scale training ≈ GWh | Enormous, repeated energy + $ | update incrementally + **sleep-time batch consolidation** |
| **Dense compute** — every parameter fires for every token | Power ∝ model size, always | **conditional compute**: MoE buddhi + top-k manas + SNN sparsity |
| **Always-on, fixed effort** — no rest/low-power mode | Wastes energy on easy inputs | **guṇa-tamas** low-power duty-cycling; effort ∝ difficulty |
| **No self-monitoring of values across modes** | Drift, jailbreaks, misalignment | **turīya** reward-invariant oversight monitor |
| **Disembodied** — no action→consequence loop | Brittle, ungrounded | **karma loop** via embodiment (§5) |
| **Poor uncertainty / overconfidence** | Hallucination, miscalibration | **kleśa monitor**: avidyā = epistemic uncertainty head |

> The headline: today's models are **giant, frozen, dense, always-full-power**. AKN is **adaptive,
> lifelong, sparse, effort-scaled** — which is *also* the energy story (§6).

---

## 2. The build-vs-buy map — every module to a concrete library

| AKN module | Use off-the-shelf | Library / framework | We build (ChittaKit) |
|---|---|---|---|
| **Base framework** | yes | **PyTorch** (ecosystem, embodiment) + **JAX/Flax** for meta-gradient parts | — |
| **manas** (attention + precision) | mostly | `torch.nn` attention; sparse-attention kernels; **Predify** (predictive-coding) | precision-gating + top-k *serial* select |
| **buddhi** (decision + MoE) | yes | **Tutel** / **DeepSpeed-MoE** (conditional compute); SB3/**CleanRL** for policy | drift-diffusion bound set by guṇa |
| **ahaṃkāra** (self/world model) | yes | **DreamerV3**, world-models; or a recurrent state-space model | asmitā anti-overfit penalty |
| **chitta** (continual memory) | partly | **Avalanche** (ContinualAI: EWC, replay); **FAISS** for episodic recall | **Ω growth+decay rule** (the missing half of EWC) |
| **guṇa controller** (meta-dynamics) | partly | **Ray Tune PBT**, **learn2learn**, **higher** (meta-grad) | **the (s,r,t) simplex regime controller** |
| **kleśa monitor** (uncertainty/bias) | yes | **Laplace**, **Uncertainty-Toolbox**, **Fairlearn**, temp-scaling | wiring into the loss as regularizers |
| **turīya** (oversight) | partly | **TRL** (RLHF), a frozen reference model | **state-invariance consistency monitor** |
| **four-state cycle** | no | — | **wake/dream/sleep orchestrator + scheduler** |
| **āśrama schedule** | partly | curriculum-learning utils; LR schedulers | **re-openable critical-period plasticity law** |
| **generative "dream" replay** | yes | small **VAE/diffusion** world model; Dreamer imagination | dream→buffer augmentation policy |

**Verdict:** the five rows that need ChittaKit are exactly the **novel contributions** from the design
file §10 — so building them *is* the research, and they sit as thin layers on top of mature deps.

---

## 3. ChittaKit — the new library we design (thin, composable, PyTorch-first)

A small library exposing the novel pieces as drop-in modules. Sketch of the API:

```python
import torch, chittakit as ck

# 1) Chitta: continual memory with growth+decay (extends EWC, adds release)
chitta = ck.SamskaraMemory(model,
            grow=0.1,            # γ: importance accrues with use (abhyāsa)
            decay=0.01,          # λ: unused saṃskāras fade (non-eternal store)
            protect=ck.fisher)   # Ω weighting

# 2) Guṇa controller: one (s,r,t) signal modulates all learning dynamics
guna = ck.GunaController(inputs=["error","novelty","reward","energy"])
#   -> returns alpha (plasticity), tau (exploration), beta (consolidation), prune_rate

# 3) Āśrama scheduler: lifelong plasticity with re-openable critical periods
life = ck.AshramaSchedule(alpha_min=1e-4, alpha_peak=1e-2,
                          reopen_on=ck.distribution_shift(kl_thresh=0.3))

# 4) Four-state orchestrator: wake / dream / sleep duty cycle
loop = ck.StateCycle(wake=infer_and_collect,
                     dream=generative_replay,        # precision_sense=0
                     sleep=consolidate_and_prune)    # + homeostatic downscale

# 5) Turīya monitor: reward-invariant consistency / alignment overseer
witness = ck.WitnessMonitor(reference=frozen_values, invariance="across_states")

agent = ck.Antahkarana(manas, chitta, buddhi, ahankara,
                       guna, life, witness, klesha=ck.KleshaRegularizers())
agent.live(env)     # the lifelong loop
```

Design principles: **(a)** every ChittaKit object is a standard `nn.Module` or optimizer-hook so it
composes with any PyTorch model; **(b)** back-ends are swappable (dense GPU now, **SNN/neuromorphic
later** — §6); **(c)** the guṇa signal and āśrama age are **observable/loggable** so the system is
*interpretable* (you can watch its "state of mind" and life-stage).

---

## 4. The two framework tracks — PyTorch vs JAX (and the call)

- **PyTorch** — best ecosystem for **embodiment** (ROS, Isaac, ExecuTorch edge), MoE (Tutel),
  continual learning (Avalanche), SNNs (snnTorch). **→ primary.**
- **JAX/Flax** — best for the **meta-gradient guṇa controller** and fast vectorized
  meta-learning/PBT (`vmap`, `grad` of `grad`), and TPU energy efficiency. **→ use for the
  meta-controller research module**, bridge results back to PyTorch.

**Call:** PyTorch-first for the full system; prototype the guṇa meta-controller in JAX where
meta-gradients are cleaner, then port. Don't split the whole codebase.

---

## 5. Grounding in the physical world — embodiment & the karma loop

The Vedic model is **action-centric** (karma): disposition forms through *acting in a world and
absorbing consequences*. A disembodied AKN can't form *vāsanā* properly — so embodiment isn't
optional, it's the data source.

- **Simulation first:** **NVIDIA Isaac Sim / Isaac Lab**, **MuJoCo**, **Gymnasium**, **PyBullet** —
  cheap, safe, infinite "lives" for the āśrama schedule to run through.
- **Real robot / edge:** **ROS 2** as the nervous system; **NVIDIA Jetson** (Orin) or a neuromorphic
  board as the brain; **ExecuTorch / ONNX Runtime / TensorRT** for on-device inference.
- **Sensorimotor mapping to the model:**
  - sensors (camera/IMU/lidar, ideally **event cameras / DVS**) → **manas** (precision-gated, sparse).
  - actuators + outcomes → **buddhi** action → **karma** reward → **chitta** saṃskāra deposit.
  - **battery / thermal / compute-load → the guṇa controller’s "energy" input.** This is the elegant
    part: a real body has a **metabolic state**, so **low battery literally drives the system toward
    *tamas*** (conserve, sparse, low-plasticity), high-opportunity novelty drives **rajas** (explore).
    The guṇas become a **real homeostatic power-management signal**, not a metaphor.
- **The four states map to real power modes:** *jāgrat* = full active inference; *svapna/suṣupti* =
  the robot "rests" and **consolidates offline** (on-board low-power, or offloaded to a server/cloud
  overnight on cheap/renewable power). Sleep is when an embodied AKN learns most — exactly as in
  animals.

---

## 6. The energy case — why AKN is structurally cheaper, and the neuromorphic endgame

Three independent levers, stackable:

**(a) Conditional / sparse compute (software, available now).**
- **MoE (buddhi)** activates a few experts per input, not the whole model (Tutel/DeepSpeed-MoE).
- **Top-k attention (manas)** processes a serial, bounded selection, not all tokens densely.
- **Effort scales with difficulty** via the guṇa state — easy inputs run in a cheap *tamas* mode.
- → typical **2–10×** compute reduction vs. dense models of equal capacity (MoE literature).

**(b) Lifelong learning instead of retraining (the biggest long-run saving).**
- Current practice: knowledge drift → **retrain from scratch** (GPT-class training ≈ hundreds of
  MWh–GWh). AKN **never retrains**; it updates incrementally and **consolidates in batched sleep**
  windows (schedulable on off-peak/renewable power). Over a deployed lifetime this is the dominant
  saving — you amortize one training, not dozens.

**(c) Event-driven Spiking Neural Networks on neuromorphic hardware (the endgame).**
- **SNNs** compute only when events occur (spikes) — naturally sparse, and a beautiful fit for the
  **manas serial gate** and the **17-mind-moment discrete-time** model (Sanskrit-formula file §8).
- Software: **snnTorch**, **Norse**, **SpikingJelly**, **Intel Lava** (for Loihi 2), **Nengo**.
- Hardware: **Intel Loihi 2**, **BrainChip Akida**, **SpiNNaker 2**, **IBM NorthPole**, **SynSense
  Speck/Xylo** — these run inference at **milliwatts** (often **100–1000× lower energy** than GPU for
  event-sparse workloads), with always-on, duty-cycled operation that matches AKN's wake/sleep cycle.
- → an embodied, always-on AKN agent on neuromorphic silicon is the **energy-optimal target form**.

> **Honest maturity note:** (a) is production-ready; (b) is research-ready (continual-learning libs
> exist, our growth+decay rule is the new bit); (c) is genuine research — neuromorphic toolchains are
> improving fast but training high-accuracy SNNs and mapping complex models to chips like Loihi 2 is
> still hard. Treat neuromorphic as the **direction**, not the day-one substrate.

---

## 7. The options — three tracks, and the recommendation

| Track | What it is | Stack | Pros | Cons |
|---|---|---|---|---|
| **A. Software research prototype** | Validate the novel claims on continual-learning + RL benchmarks | PyTorch + Avalanche + ChittaKit (+JAX guṇa) | fastest to results; publishable; cheap | not embodied/energy-optimal yet |
| **B. Embodied edge agent** | AKN in sim→robot, with the karma loop + guṇa-from-battery | ROS 2 + Isaac/MuJoCo + Jetson + ExecuTorch | grounds the design physically; the karma loop works | more moving parts; hardware cost |
| **C. Neuromorphic energy-optimal** | Event-driven AKN on Loihi/Akida | snnTorch/Lava + neuromorphic board | the energy endgame (mW-scale) | research-grade toolchain; accuracy/mapping hard |

**Recommendation — phased A → B → C:**
1. **Start Track A.** Build ChittaKit's five novel modules; prove on **split-CIFAR / Continual-World**
   that *Ω-growth+decay beats plain EWC* and *re-openable critical periods recover faster under
   distribution shift*. This de-risks the whole idea cheaply and gives publishable wins.
2. **Then Track B.** Drop the validated agent into MuJoCo/Isaac, wire **battery→guṇa**, run the
   āśrama life-cycle in sim, then a small real robot (Jetson). This is where "relate to the physical
   world" becomes real.
3. **Then Track C.** Port the sparse/event-friendly parts (manas gate, SNN encoders) to snnTorch →
   Loihi 2 for the energy-optimal embodied agent.

Each phase is independently useful and independently fundable/publishable.

---

## 8. First concrete steps (if we start Track A)

1. **Scaffold the repo:** `chittakit/` (library) + `experiments/` (benchmarks) + `configs/`.
2. **Implement `SamskaraMemory`** (EWC + growth + decay) and benchmark vs. Avalanche's EWC on a 10-task
   stream — the cleanest first falsifiable result.
3. **Implement `GunaController`** (start rule-based: error/novelty/reward → α,τ,β,prune; later
   meta-learn it) and show it auto-balances plasticity vs. stability.
4. **Implement `AshramaSchedule`** with the novelty-triggered re-opening; test mid-stream domain shift.
5. **Wrap in `StateCycle`** (wake-collect / dream-replay / sleep-consolidate) and measure
   sample-efficiency + retention.
6. **Log the guṇa state and āśrama age** every step — the interpretable "mind-state" dashboard.

---

## 9. Caveats

- **ChittaKit is a thin research layer, not a moonshot framework** — it leans on PyTorch/Avalanche/
  snnTorch; we own only the ~5 novel modules. Keep it that way; don't reinvent autograd.
- **Energy numbers above are order-of-magnitude** from the MoE/neuromorphic literature, not our
  measurements — Track A must *measure* J/inference and forgetting to substantiate them.
- **Neuromorphic is a direction, not a guarantee** — plan the software so back-ends are swappable, so
  a hardware bet failing doesn't sink the project.
- **No consciousness claims** (see design file §12) — turīya here is strictly an oversight monitor.
- This is a **plan to build and measure**, with the philosophy supplying the hypotheses and the
  control structure — the experiments decide what's true.
