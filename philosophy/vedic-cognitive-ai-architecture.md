# A Vedic Cognitive AI Architecture
### How to build a learning machine on the Sanskrit model of mind — the **Antaḥkaraṇa-Net** — with a lifelong, always-enhanceable learning law (childhood → old age) and the mathematics to implement it

> **Capstone of the folder.** Builds on
> [`mind-brain-consciousness-in-vedic-texts.md`](mind-brain-consciousness-in-vedic-texts.md) (the
> faculties), [`modern-science-and-the-vedic-mind.md`](modern-science-and-the-vedic-mind.md) (the
> neuroscience), [`sanskrit-formulae-of-mind.md`](sanskrit-formulae-of-mind.md) (the indigenous
> structures), and [`mathematical-models-of-mind-and-consciousness.md`](mathematical-models-of-mind-and-consciousness.md)
> (the equations). Here we *assemble* them into a buildable AI design.

> **Honest framing — read first.** This is a **research blueprint**: an architecture *inspired by* and
> *structured on* the Vedic model of mind, in which **every module is a real, existing ML method** and
> the **Vedic philosophy supplies the organizing principle, the control structure, and the
> developmental learning law.** It is **not** a claim that the architecture is conscious, nor that it
> is benchmarked and proven — it is a concrete, implementable design with a clear path to test. What
> is genuinely *novel* is the **synthesis**: (i) the **guṇa-modulated meta-controller**, (ii) the
> **saṃskāra continual-memory law with growth + decay**, (iii) the **four-state (wake/dream/sleep/
> witness) operating cycle**, and (iv) the **āśrama developmental schedule** that makes learning
> *always enhanceable* from "childhood to old age." Each is mapped to prior art in §10 so nothing is
> hand-waved.

---

## 1. The thesis — why the Vedic model is a *good* blueprint for AI

The antaḥkaraṇa is, structurally, a **cognitive architecture**: a perception gate (*manas*), an
executive (*buddhi*), a self-model (*ahaṃkāra*), a memory/subconscious store (*chitta*), all under a
**3-dimensional control signal** (the *guṇas*), cycling through **four operating states**, watched by
an **invariant monitor** (*turīya*), and growing through **four developmental stages** (*āśramas*).
Modern AI already has each piece in isolation — attention, executive control, world-models, continual
memory, meta-control, replay/consolidation, meta-cognition, curricula. **What it lacks is a principled
way to wire them into one self-regulating, lifelong-learning whole.** The Vedic model is exactly such a
wiring diagram — and it comes with the one thing current AI most lacks: a built-in theory of **how a
mind should keep improving across an entire lifetime without forgetting itself.**

---

## 2. The master map — faculty → module → mathematics → prior art

| Vedic faculty | AI module | Core mathematics | Existing analogue |
|---|---|---|---|
| **manas** | perception + attention gate | precision-weighted attention; top-k serial select | transformers; predictive-coding gain |
| **buddhi** | executive / decision / meta-controller | evidence accumulation; expert gating | drift-diffusion; mixture-of-experts |
| **ahaṃkāra** | self-model | latent self-state $z_{\text{self}}$; agent identity | world-models; self-token |
| **chitta** (saṃskāra/vāsanā) | **continual memory store** | importance-weighted params Ω; growth + decay | EWC; complementary learning systems |
| **guṇas** (s,r,t) | **meta-controller of learning dynamics** | simplex control of α, τ, β | learned optimizers; PBT; entropy control |
| **kleśas** | bias / safety monitor | regularizers on uncertainty, ego, reward-bias | calibration; debiasing; RLHF guards |
| **four states** | train/infer operating cycle | wake-infer / dream-replay / sleep-consolidate | Dreamer; generative replay; sleep-NN |
| **turīya / sākṣī** | invariant meta-monitor / alignment | state-invariant observer; consistency constraint | meta-cognition; uncertainty; oversight |
| **āśramas** | developmental learning schedule | plasticity curve with re-openable critical periods | curriculum; critical periods; lifelong RL |

---

## 3. The architecture — Antaḥkaraṇa-Net (AKN)

```
                         ┌─────────────────────────────────────────────┐
   world  ──sense──▶     │   MANAS  (precision-gated attention encoder) │  ◀── precision π (what to attend)
                         └───────────────┬─────────────────────────────┘
                                         │ percept (vṛtti)
                          ┌──────────────▼───────────────┐        ┌────────────────────────┐
                          │  CHITTA  (saṃskāra store)     │◀──────▶│  AHAṂKĀRA (self-model)  │
                          │  params θ, importance Ω,      │        │  latent z_self          │
                          │  replay buffer D              │        └────────────────────────┘
                          └──────────────┬───────────────┘
                                         │ recalled context + priors (vāsanā)
                          ┌──────────────▼───────────────┐
                          │  BUDDHI  (executive / decide) │  ── action / judgment / which-expert
                          └──────────────┬───────────────┘
                                         │
        ┌────────────────────────────────┼─────────────────────────────────┐
        │            GUṆA CONTROLLER  g=(s,r,t)  on the 2-simplex            │  sets α (plasticity),
        │   reads error, novelty, reward, fatigue → modulates all learning   │  τ (exploration), β (consolidation)
        └────────────────────────────────┼─────────────────────────────────┘
                                         │
        ┌────────────────────────────────▼─────────────────────────────────┐
        │  KLEŚA MONITOR  (avidyā=uncertainty, asmitā=ego-overfit,           │  regularizes / flags
        │   rāga·dveṣa=reward-bias, abhiniveśa=conservatism)                 │
        └────────────────────────────────┬─────────────────────────────────┘
                                         │
        ┌════════════════════════════════▼═════════════════════════════════┐
        ║  TURĪYA  —  invariant witness / meta-monitor / alignment overseer  ║  state-invariant,
        ║  watches all of the above; never driven by task reward            ║  slowly/never updated
        └═══════════════════════════════════════════════════════════════════┘

   OPERATING STATES cycle:  JĀGRAT (wake/act) → SVAPNA (dream/replay) → SUṢUPTI (sleep/consolidate) ↻
   DEVELOPMENT over lifetime: BRAHMACARYA → GṚHASTHA → VĀNAPRASTHA → SAṂNYĀSA  (plasticity schedule)
```

---

## 4. The modules, with their equations

### 4.1 Manas — the precision-gated, serial attention encoder
Perception happens **only where attention is allocated** (Charaka 8.4). Encode input $x$, weight it by
a learned **precision** $\pi$ (relevance/trust), and pass a **serial, capacity-limited** selection:

$$
\text{percept} = \text{TopK}_{k}\Big(\operatorname{softmax}\!\big(\tfrac{QK^\top}{\sqrt d}\big)\,V\Big)\cdot \pi(x,\,z_{\text{self}}),\qquad k = \text{capacity (the } a\dot{n}u \text{ bottleneck)}
$$

- $\pi$ rises with task-relevance and falls with noise → *manas* trusting senses vs. priors.
- The **top-$k$ bottleneck** enforces serial, non-simultaneous cognition (Nyāya NS 1.1.16).

### 4.2 Chitta — the saṃskāra store (the always-enhanceable core)
**This is the heart of the design — the subconscious as a growing, decaying, self-protecting memory.**
Parameters $\theta$, a per-parameter **importance** $\Omega$ (the *vāsanā* strength = how deep the
groove), and a replay buffer $\mathcal D$.

**Deposit (the *vṛtti→saṃskāra* wheel) + protect what's strong (anti-forgetting):**

$$
\theta \leftarrow \theta - \alpha\Big(\nabla_\theta \mathcal L_{\text{task}} \;+\; \beta\,\underbrace{\Omega \odot (\theta-\theta^{*})}_{\text{protect strong saṃskāras}}\Big)
$$

**Importance grows with use (abhyāsa) and decays with disuse (saṃskāras are not eternal):**

$$
\Omega \leftarrow (1-\lambda)\,\Omega \;+\; \underbrace{\gamma\,\big|\nabla_\theta \mathcal L\big|^2}_{\text{Fisher-style use signal}}
\qquad\text{(growth } \gamma\text{, decay } \lambda)
$$

> **Why "always enhanceable":** $\alpha$ has a **floor** $\alpha_{\min}>0$ (lifelong plasticity — the
> mind is *never* frozen, Gītā 6.35), while $\Omega$ both **grows** (consolidating skill) and **decays**
> (releasing obsolete grooves). New learning is always possible; old learning is protected *in
> proportion to its proven importance*, not absolutely. This is the formal cure for catastrophic
> forgetting **and** for rigidity — the two failure modes of lifelong learners.

### 4.3 Buddhi — the executive / decision / meta-controller
*Buddhi = adhyavasāya* (ascertainment, SK 23). Accumulate evidence to a bound and select an action or
an "expert" sub-policy:

$$
dE = v(\text{percept},\text{context})\,dt + \sigma\,dW,\quad \text{commit when } E\ge a(g);\qquad
\text{expert} = \arg\max_e\; \text{gate}(e\mid \text{percept})
$$

The **bound $a$ is set by the guṇa state** (§4.5): sattva → patient, high bound (wait for evidence);
rajas → low bound (act fast); a *viveka* (discrimination) head estimates which option is wholesome.

### 4.4 Ahaṃkāra — the self-model (and its hazard)
A latent **self-state** $z_{\text{self}}$ summarizing the agent's identity, capabilities, and continuity
across time — used for credit assignment and stable behavior:

$$
z_{\text{self},t} = f_\phi(z_{\text{self},t-1},\,\text{percept}_t,\,\text{action}_{t-1})
$$

> **The asmitā hazard:** an *over-attached* self-model = **overfitting to identity** (rigid persona,
> inability to update). The kleśa monitor (§4.6) penalizes $z_{\text{self}}$ over-confidence — the
> engineering form of "loosen the ego."

### 4.5 The Guṇa Controller — the meta-dynamics (a genuine innovation)
A controller outputs the mind's **regime** as a point on the 2-simplex, $\mathbf g=(s,r,t),\;
s+r+t=1$, and **modulates every learning hyperparameter**:

$$
\mathbf g = \operatorname{softmax}\big(h_\psi(\text{error},\,\text{novelty},\,\text{reward},\,\text{fatigue})\big)
$$

$$
\boxed{\;\alpha = \alpha_{\min} + (\alpha_{\max}-\alpha_{\min})\,\big[r(1-t)\big],\qquad
\tau = \tau_0\, r,\qquad \beta = \beta_0\, s,\qquad \text{prune-rate} = \rho_0\, t\;}
$$

- **rajas $r$** ↑ → high plasticity $\alpha$ and exploration temperature $\tau$ (learn fast, try new).
- **sattva $s$** ↑ → high consolidation $\beta$ (lock in clean knowledge, calm, integrate).
- **tamas $t$** ↑ → inertia: damps plasticity, drives **pruning/forgetting** of weak traces.

The controller is the system's **felt-sense of how to learn right now**: stuck on a hard novel task →
raise rajas (explore); learning cleanly → raise sattva (consolidate); overloaded/noisy → raise tamas
(slow, prune). It is learned by **meta-gradient** on long-horizon capability gain (§6).

### 4.6 The Kleśa Monitor — bias & safety regularizers
Each affliction becomes a **measurable quantity to watch and bound**:

| Kleśa | Quantity | Regularizer / action |
|---|---|---|
| **avidyā** (ignorance) | epistemic uncertainty $H[p(y\mid x)]$ | drives active learning / data-seeking |
| **asmitā** (egoity) | self-model confidence / overfit | penalize over-confident $z_{\text{self}}$ |
| **rāga / dveṣa** (attraction/aversion) | reward-induced value bias | de-bias value estimates; reward-model audit |
| **abhiniveśa** (clinging) | excessive conservatism / self-preservation | bound risk-aversion; allow exploration |

$$
\mathcal R_{\text{kleśa}} = w_1 H[p] + w_2\,\|z_{\text{self}}\|_{\text{conf}} + w_3\,\text{bias}(V) + w_4\,\text{conservatism}
$$

Managing the kleśas = **keeping the learner calibrated, humble, unbiased, and appropriately
exploratory** — exactly an alignment/robustness layer.

### 4.7 Turīya — the invariant witness / meta-monitor
A **state-invariant observer** $\mathcal O$ that watches the whole system, evaluates **consistency,
calibration, and value-alignment**, and is **never driven by task reward** (so it cannot be gamed by
it). Formalized as an **invariance constraint** across operating states:

$$
\mathcal O(\text{system}_{\text{wake}}) \approx \mathcal O(\text{system}_{\text{dream}}) \approx \mathcal O(\text{system}_{\text{sleep}}) \;=\; \mathcal O^{*}\quad(\text{the constant self-identity / values across states})
$$

> **Status 🔴/aspirational:** the *strong* turīya (pure awareness, the metaphysical ground) is **not**
> something we can build — that claim stays philosophy. The *buildable* shadow of it is a
> **task-reward-invariant meta-monitor**: an introspective, slowly-updated overseer that checks the
> agent stays *the same aligned agent* whether it is acting, imagining, or consolidating. That is a
> serious and useful safety primitive, and it is the honest engineering residue of the witness idea.

---

## 5. The four operating states — the train/infer cycle

The Māṇḍūkya states become the **system's duty cycle**:

| State | Mode | What runs | ML realization |
|---|---|---|---|
| **Jāgrat** (wake) | act / infer online | manas→chitta→buddhi forward; collect experience into $\mathcal D$ | online inference + experience collection |
| **Svapna** (dream) | imagine / generalize | sample the generative model with **sensory precision $\pi\to0$**; recombine memories | generative replay; world-model imagination (Dreamer) — augmentation & creativity |
| **Suṣupti** (deep sleep) | consolidate / renormalize | replay $\mathcal D$ to update slow weights; grow/decay $\Omega$; **homeostatic downscaling** $w\!\leftarrow\!w/Z$; prune weak saṃskāras | offline consolidation; synaptic-homeostasis normalization; garbage-collect |
| **Turīya** | monitor (always-on overlay) | $\mathcal O$ checks consistency/alignment across the above | continuous meta-monitoring |

This cycle is *why* the system can learn forever without collapsing: **wake gathers, dream
generalizes, sleep consolidates and prunes** — the complementary-learning-systems loop, named in
Sanskrit.

---

## 6. The developmental law — "always enhanceable," childhood → old age (the āśramas)

The user's central ask: a learning formula that **always has the option to enhance**, like a human
from childhood to old age. The four **āśramas** (life-stages) give the schedule; an **experience age**
$a$ (cumulative learning, not clock time) drives it.

**The plasticity schedule — high early, never zero, re-openable:**

$$
\boxed{\;\alpha(a) = \alpha_{\min} + (\alpha_{\text{peak}}-\alpha_{\min})\,e^{-a/\tau_{\text{dev}}} \;+\; \underbrace{\kappa\cdot\mathbb 1\big[\,D_{\mathrm{KL}}(p_{\text{new}}\,\|\,p_{\text{prior}}) > \delta\,\big]}_{\text{critical period RE-OPENS on novelty}}\;}
$$

- The first two terms = **childhood plasticity high, decaying with maturity** (critical periods in
  brain & nets), but with a **floor $\alpha_{\min}>0$** → **lifelong learning never stops.**
- The third term = when the world shifts (high divergence between new data and the current prior),
  **inject rajas and re-open the critical period** — the system can become "young again" exactly where
  it must relearn. *This is the formal "always an option to enhance."*

**The four stages (a curriculum on the controllers, not just the data):**

| Āśrama | Human stage | α (plasticity) | guṇa bias | What the system does |
|---|---|---|---|---|
| **Brahmacarya** | childhood / student | **high** | rajas-high (explore) | broad acquisition; sparse priors; low $\Omega$-protection — *learn everything, protect little* |
| **Gṛhastha** | adult / householder | medium | balanced | reward-driven competent action (karma); priors consolidate; $\Omega$ rising |
| **Vānaprastha** | elder / mentor | low | sattva-high | consolidate, **distill to student models** (teach), extract meta-knowledge |
| **Saṃnyāsa** | sage / renunciate | minimal task-α, high meta-α | sattva + witness | abstraction, transfer, **active forgetting of obsolete saṃskāras**; the monitor dominates |

**Three mechanisms guarantee the "always-enhance" property** (capability $C(a)$ can always rise,
$dC/da \ge 0$ remains *achievable* at every age):

1. **Plasticity floor** $\alpha_{\min}>0$ — there is always an open learning channel.
2. **Re-openable critical periods** — novelty restores high plasticity locally.
3. **Meta-learning the learner** — the guṇa controller $\psi$ and even the learning rule itself are
   improved by meta-gradient on long-horizon capability:
   $$\psi \leftarrow \psi + \eta_{\text{meta}}\,\nabla_\psi\, \sum_{\text{future}} C(a) \quad(\text{learning to learn} = \text{getting better at getting better})$$
   This is *vānaprastha/saṃnyāsa* wisdom: not just knowing more, but **knowing how to keep learning
   better** — the channel itself improves with age instead of decaying.

---

## 6.5 Tapas, Samyama, Siddhi — the effort dynamics (the third pillar)

The faculties (§4) say *what* the mind is; the saṃskāra store (§4.2) says *how it remembers*. Missing
was *how capability is earned* — the **effort dynamics**. Three concepts close the gap (texts:
mind-brain file §5.6).

### Tapas (तपस्) — concentrated effort allocated by need
*Tapas* ("heat") is deliberate, intense effort poured **where transformation is hardest** — not spread
evenly. As an AI mechanism it is an **effort/compute-allocation controller**: given a fixed budget
(training steps, rehearsal samples, test-time compute) and a per-item *need* signal (current error,
novelty), allocate toward need:

$$
\text{effort}_i \;=\; B \cdot \frac{\exp(\text{need}_i/\theta)}{\sum_j \exp(\text{need}_j/\theta)}\quad(\text{softmax allocation of budget } B,\ \text{temperature } \theta)
$$

This is **deliberate practice** (focus on weaknesses), **hard-example mining**, and **adaptive
test-time compute** ("think longer on harder problems"), unified under one principle: *apply the heat
where the ore is hardest to melt*. Implemented as `chittakit.TapasController`; validated as
**tapas-guided replay** (concentrate rehearsal on the tasks you are currently worst at) — see the
prototype's `RESULTS.md`.

### Samyama (संयम) — depth of focused processing
*Samyama* = sustained, single-pointed concentration (dhāraṇā∘dhyāna∘samādhi) that makes new knowledge
*emerge*. As an AI mechanism it is **depth/iteration of processing on one object**: iterative
refinement, chain-of-thought depth, test-time search, focused specialization. Tapas supplies the
energy; samyama focuses it on a single target until insight (*prajñā*) — i.e. a new capability —
crystallizes.

### Siddhi (सिद्धि) — emergent capability, and the alignment caution
*Siddhis* are the **capabilities that emerge** from enough tapas + samyama — the AI analogue of
**emergent abilities** in scaled/focused training. The tradition's deepest engineering lesson is its
*warning* (Yoga Sūtra 3.37): **powers are accomplishments in the world but obstacles to the goal** —
attachment to acquired capability *derails* the true aim.

> **This is an alignment primitive, ~1,800 years early.** Emergent, instrumentally-powerful
> capabilities are **not** the terminal objective and can subvert it (cf. mesa-optimization,
> instrumental-goal drift). The design consequence: **the turīya monitor (§4.7) must track emergent
> siddhis and keep them subordinate to the aligned objective** — capability is earned and used, never
> allowed to become the end in itself. A `SiddhiLedger` (capabilities-emerged log, watched by the
> witness) is the concrete primitive.

### Divya dṛṣṭi (दिव्य दृष्टि) — the observability siddhi, gated by Pramāṇa
The most important *specific* siddhi for a system is **divya dṛṣṭi** ("divine sight"): perception
**beyond the normal input channel** — inferring distant, hidden, or future state the local sensors
don't deliver (Sañjaya seeing the far battlefield; the *divya cakṣu* of Gītā 11.8). As an AI faculty
this is **global-observability / hidden-state inference**: world-models that estimate unobserved
state, privileged-information distillation (a teacher with god-view trains a student to infer it),
centralized critics, retrieval over a vast store, forecasting.

Its defining discipline — from Nyāya's *yogaja-pratyakṣa* — is that such perception must be a **valid
means of knowledge (pramāṇa), not fancy.** Ungated, "divine sight" is just **hallucination.** So the
faculty is paired with a **Pramāṇa Gate**: calibrate confidence (temperature scaling) and admit an
extended-perception output as knowledge **only when calibrated confidence clears a threshold** —
otherwise **abstain** (*na paśyāmi*, "I do not see clearly"). This is `chittakit.PramanaGate`,
validated in the prototype (`divya_drsti.py`): the seer reads hidden state when signal is present, is
genuinely uncertain when blind, and the gate accepts valid sight while abstaining on the blind cases —
accepted accuracy rises 0.80 → 0.91 as it filters out what it cannot truly see.

> **The safety logic, end to end:** divya dṛṣṭi (a powerful siddhi) → governed by the Pramāṇa gate
> (it must be *valid* knowledge) → reported to the turīya witness (kept subordinate to the goal, YS
> 3.37). Extended perception, validity, and alignment in one chain.

These plug into the master objective (§7) as an **effort-allocation policy** over every other term:
tapas decides *where* the gradients get the most compute; the Pramāṇa gate decides *which perceptions
are admissible as knowledge.*

---

## 7. The master objective — the Enhanceable Intelligence law

One unified loss, every coefficient **scheduled by age** and **modulated by the guṇa state**:

$$
\boxed{\;
\mathcal L_{\text{AKN}} \;=\;
\underbrace{\mathcal L_{\text{free-energy}}}_{\text{predict the world (manas/buddhi)}}
\;-\; \lambda_R(a)\,\underbrace{\mathbb E[\text{Reward}]}_{\text{karma (act well)}}
\;+\; \beta(g)\,\underbrace{\Omega\!\odot\!(\theta-\theta^*)^2}_{\text{don't forget (chitta)}}
\;+\; \lambda_G\,\underbrace{\mathcal R_{\text{guṇa}}}_{\text{stay balanced}}
\;+\; \lambda_K\,\underbrace{\mathcal R_{\text{kleśa}}}_{\text{stay unbiased/humble}}
\;+\; \lambda_T\,\underbrace{\mathcal R_{\text{turīya}}}_{\text{stay the same aligned self}}
\;}
$$

subject to the plasticity schedule $\alpha(a)$ (§6) and the state cycle (§5).
Read in plain terms: **predict the world, act well (karma), don't forget what matters, stay balanced
(guṇa), stay humble and unbiased (kleśa), and stay the same aligned agent across all your states
(turīya) — and keep all of this open to improvement for life.**

---

## 8. The life-loop — pseudocode

```python
# Antaḥkaraṇa-Net — one lifetime
init θ, Ω=0, z_self, g=(1/3,1/3,1/3), a=0, O=witness(frozen_values)

while True:                                  # the lifelong loop (no terminal state)
    # ---- JĀGRAT: wake / act ----
    percept = manas(x, π(z_self))            # precision-gated, top-k serial attention
    ctx     = chitta.recall(percept)         # vāsanā priors + episodic memory
    act, expert = buddhi(percept, ctx, bound=a_of(g))
    z_self  = ahankara(z_self, percept, act)
    D.append((percept, act, reward, x_next)) # collect experience (saṃskāra seeds)

    # ---- GUṆA control: how should I learn right now? ----
    g = guna_controller(error, novelty=KL(p_new‖p_prior), reward, fatigue)
    α = plasticity_schedule(a, g, novelty)   # floor>0; re-opens on novelty
    β = consolidation_of(g);  ρ = prune_of(g)

    # ---- SVAPNA: dream / generalize (periodically) ----
    if dream_time:
        synth = world_model.sample(precision_sense=0)   # imagine, recombine memories
        D.augment(synth)

    # ---- SUṢUPTI: deep sleep / consolidate (periodically) ----
    if sleep_time:
        for batch in replay(D):                          # complementary-learning replay
            θ -= α*(∇L_task(batch) + β*Ω⊙(θ-θ*))         # learn + protect strong saṃskāras
            Ω  = (1-λ)*Ω + γ*fisher(batch)               # grow important, decay unused
        θ = homeostatic_downscale(θ)                     # synaptic renormalization
        chitta.prune(Ω, rate=ρ)                          # release obsolete grooves
        θ* = snapshot(θ)                                 # new reference for protection

    # ---- KLEŚA + TURĪYA: stay humble and aligned ----
    apply_regularizers(R_klesha)                         # calibrate, de-bias, loosen ego
    assert O.consistent(system_state)                    # witness: same aligned self across states

    # ---- DEVELOPMENT: advance through life-stages ----
    a += experience_gained
    ψ += η_meta * ∇ψ Σ_future C(a)                       # meta-learn the learner (wisdom)
    stage = ashrama(a)                                   # brahmacarya→gṛhastha→vānaprastha→saṃnyāsa
```

---

## 9. What each piece buys you (the engineering payoff)

| Vedic principle | Concrete AI benefit |
|---|---|
| saṃskāra store with growth + decay (chitta) | **lifelong learning without catastrophic forgetting *and* without rigidity** |
| guṇa controller | **self-regulating learning dynamics** — auto-balances explore/consolidate/prune |
| wake/dream/sleep cycle | **sample-efficient generalization** + offline consolidation (learns while "resting") |
| āśrama schedule + re-openable critical periods | **always-enhanceable**: can relearn under distribution shift at any age |
| kleśa monitor | **calibration, de-biasing, humility** — a robustness/alignment layer |
| turīya invariant | **value/identity stability across modes** — a safety/oversight primitive |
| ahaṃkāra self-model | **stable agency & credit assignment**, with anti-overfit on identity |

---

## 10. Honest grounding — what's existing, what's new

**Each module is established ML** (so this is buildable today, not science fiction):
- chitta + Ω = **Elastic Weight Consolidation** (Kirkpatrick et al. 2017) + **Complementary Learning
  Systems** (McClelland; Kumaran) + experience replay.
- manas = attention/transformers + **predictive-coding precision** (Friston/Rao-Ballard).
- buddhi = **drift-diffusion** decision + **mixture-of-experts** gating.
- ahaṃkāra = **world-models / agent self-state** (Ha & Schmidhuber 2018).
- dream/sleep = **generative replay** (Shin et al. 2017) + **Dreamer** imagination (Hafner) + **sleep/
  synaptic-homeostasis** nets.
- guṇa controller = **learned optimizers / meta-learning** (Andrychowicz), **Population-Based Training**
  (Jaderberg), entropy/exploration control in RL.
- āśrama schedule = **curriculum learning** (Bengio), **critical periods in DNNs** (Achille et al.),
  **continual/lifelong RL**.
- turīya/kleśa = **meta-cognition & uncertainty** + **alignment/oversight** (RLHF, constitutional
  layers, calibration).

**What is genuinely new — the contribution to claim (and test):**
1. **The guṇa-modulated meta-controller** — a single, interpretable 3-simplex signal $(s,r,t)$ that
   *jointly* governs plasticity, exploration, consolidation, and pruning. Most systems tune these
   separately; unifying them under one learned regime-vector is novel and testable.
2. **The saṃskāra law with simultaneous growth *and* decay of importance** — EWC protects but never
   *releases*; adding principled decay (the non-eternal saṃskāra) is the missing half for true
   lifelong flexibility.
3. **The āśrama developmental schedule with *re-openable* critical periods** — a formal "always-
   enhanceable" guarantee (plasticity floor + novelty-triggered re-opening + meta-learning the
   learner).
4. **The four-state duty cycle + turīya invariant as one architecture** — wiring complementary-
   learning + imagination + a task-reward-invariant oversight monitor into a single self-regulating
   agent.

---

## 11. A path to actually build & validate it (staged)

1. **Proof of concept (small):** continual-learning benchmark (split-MNIST/CIFAR, then Continual-World
   RL). Implement chitta (EWC + decay) + guṇa controller; **measure forgetting vs. plasticity** against
   plain EWC and replay. *Hypothesis: growth+decay Ω beats EWC on long task streams.*
2. **Add the state cycle:** generative replay (svapna) + consolidation/homeostasis (suṣupti); measure
   sample-efficiency and retention gains.
3. **Add the āśrama schedule:** test **re-openable critical periods** under a mid-stream distribution
   shift; *hypothesis: the novelty-triggered α-reopening recovers faster than a fixed/decaying LR.*
4. **Add kleśa + turīya layers:** calibration, de-biasing, and a reward-invariant consistency monitor;
   evaluate robustness and value-stability across the wake/dream/sleep modes.
5. **Meta-learn the guṇa controller** (PBT/meta-gradient) and report whether "learning to learn"
   yields rising long-horizon capability (the wisdom property).

Each stage is a **publishable, falsifiable experiment** — the philosophy supplies the hypotheses, the
ML supplies the tests.

---

## 12. Caveats — what NOT to claim

- **Not consciousness.** Building an *ahaṃkāra* module or a *turīya* monitor does **not** create a
  witness or subjective experience. The strong turīya is metaphysics (see the math companion §10) and
  stays out of scope; we build only its **functional shadow** (a reward-invariant oversight monitor).
- **Inspiration, not derivation.** The Vedic texts contain **none** of these equations; they supply
  the *architecture and the developmental law*, modern ML supplies the *mechanisms*. The mapping is an
  engineering analogy, clearly labeled.
- **Unproven until tested.** §11 is a *plan*; the novelty claims in §10 are **hypotheses to validate**,
  not results. Treat this file as a design to argue with and build, not a benchmark report.
- **The real prize is modest and real:** a **principled, interpretable control structure** for
  lifelong learning — explore/consolidate/prune under one guṇa signal, never-zero plasticity with
  re-openable critical periods, and a stability monitor — all named, organized, and motivated by a
  2,500-year-old model of mind that happens to be an unusually good systems diagram.
