# Mathematical Models of Mind & Consciousness
### The modern equations — and how each one formalizes a Vedic faculty: vāsanā as *prior*, saṃskāra as *attractor*, buddhi as *accumulator*, manas as *serial bottleneck*, chit/turīya as *invariant*, consciousness as *Φ*

> **Companion to** [`mind-brain-consciousness-in-vedic-texts.md`](mind-brain-consciousness-in-vedic-texts.md),
> [`modern-science-and-the-vedic-mind.md`](modern-science-and-the-vedic-mind.md), and the indigenous-
> formula file [`sanskrit-formulae-of-mind.md`](sanskrit-formulae-of-mind.md).

> **What this file is — and is not.** These are **modern mathematical models of the *phenomena* the
> Vedic texts described** — *not* translations of the texts (which contain no such equations, see the
> Sanskrit-formula file). Each section gives a real, citable model, then states the **mapping** to the
> Vedic faculty as an **interpretive bridge** (clearly flagged). The mappings are *analogies of
> role*, defensible at the functional level, never claims that the ṛṣis wrote these equations. The
> **subconscious** (*saṃskāra/vāsanā*) gets the most weight, per the brief — it is, of all the
> faculties, the one with the cleanest mathematics today (§§2, 6, 7, 12).

---

## 1. Orientation — five faculties, five families of equation

| Vedic faculty | Function | Modern mathematical model | § |
|---|---|---|---|
| **manas** | attention-gate, sensory binding, serial sampler | precision-weighting; serial bottleneck; information bottleneck | 5, 8 |
| **buddhi** | decision, *adhyavasāya* (ascertainment) | drift-diffusion / evidence accumulation to a bound | 4 |
| **ahaṃkāra** | the "I-maker," self-model | self-model as a hierarchical generative model; DMN dynamics | 9 |
| **chitta / saṃskāra / vāsanā** | memory-store & **the subconscious** | **Bayesian prior · Hopfield attractor · Hebbian deposit · RL value** | 2, 6, 7, 12 |
| **chit / puruṣa / turīya** | witnessing awareness, the constant ground | invariant / fixed-point of the state dynamics; (and IIT-Φ as a rival) | 9, 10 |

---

## 2. Perception as inference — *vāsanā = the prior* (the Bayesian brain)

The **single most important bridge.** Modern perception is modeled as **Bayesian inference**: the
brain infers the hidden cause $h$ of sensory data $d$ by combining likelihood and prior:

$$
\underbrace{P(h \mid d)}_{\text{percept (posterior)}} \;=\; \frac{\overbrace{P(d \mid h)}^{\text{sensory likelihood}}\;\overbrace{P(h)}^{\textbf{prior}}}{P(d)}
\qquad\Longrightarrow\qquad \hat h_{\text{MAP}} = \arg\max_{h}\; P(d\mid h)\,P(h)
$$

> **Mapping [interpretive]:** the **prior $P(h)$ is *vāsanā*** — the accumulated disposition that
> pre-shapes what you perceive *before awareness*. *Saṃskāra* is what *updates* the prior; *vṛtti* is
> the posterior that reaches the conscious "stream." Perception being "controlled hallucination" (the
> percept is mostly prior) is the computational form of the Yogic claim that **we see the world
> through our impressions, not as it is** (the field of *avidyā*, YS 2.4–5).

**Precision-weighting** (how much the brain trusts senses vs. prior) is the gain $\pi$ on each term;
**manas as attention** = allocating precision. **Dreaming** falls straight out: set the **sensory
precision to zero** ($\pi_{\text{sense}}\to 0$) and the posterior is driven entirely by the prior /
generative model — the brain *samples its own model* with no sensory anchor. That is exactly **svapna
/ Taijasa**: a world built from impressions (Bṛhadāraṇyaka 4.3.10; Praśna 4.5). 🟢

---

## 3. The Free-Energy Principle — *citta-vṛtti-nirodha* as surprise-minimization (Friston)

Predictive processing makes the above a single optimization. The brain minimizes **variational free
energy** $F$, an upper bound on sensory surprise $-\ln P(o)$:

$$
F \;=\; \underbrace{D_{\mathrm{KL}}\!\big[q(s)\,\|\,P(s\mid o)\big]}_{\text{approximation gap}} \;-\; \ln P(o)\;\;\ge\;\; -\ln P(o)
\qquad\Longleftrightarrow\qquad F = \underbrace{\mathbb{E}_{q}[-\ln P(o\mid s)]}_{\text{prediction error}} + \underbrace{D_{\mathrm{KL}}[q(s)\,\|\,P(s)]}_{\text{deviation from prior}}
$$

Perception lowers $F$ by updating beliefs $q(s)$; **action** lowers $F$ by changing the world to match
predictions (*active inference*). The driving signal is **prediction error** $\varepsilon = o-\hat
g(\mu)$, minimized by gradient descent $\dot\mu \propto -\partial F/\partial\mu$.

> **Mapping [interpretive]:** the mind is a machine that **continuously reduces the mismatch between
> model and world** — a restless error-minimizer. *Citta-vṛtti-nirodha* (YS 1.2, "stilling the
> modifications") is, in this frame, **driving the vṛttis (prediction-error-driven updates) toward
> zero** — a quiescent generative model that no longer churns. The *kleśas* (avidyā/rāga/dveṣa) are
> **maladaptive priors** that keep generating error (craving = a prior the world keeps violating).
> Note 🟡: this is a *resonance*, not an identity — and "minimize surprise" is not the same as the
> soteriological goal of liberation.

---

## 4. Decision — *buddhi = adhyavasāya* as evidence accumulation to a bound

Sāṃkhya Kārikā 23 defines **buddhi as *adhyavasāya*** — "determination, ascertainment." The standard
model of a decision is the **drift-diffusion model** (Ratcliff): noisy evidence accumulates until it
crosses a threshold:

$$
dx = v\,dt + \sigma\,dW,\qquad \text{decide when } x \ge a \ (\text{or } x \le 0)
$$

with **drift $v$** = evidence quality, **bound $a$** = caution/criterion, $dW$ = Wiener noise. This
predicts reaction-time *and* accuracy distributions and is among the best-validated models in
cognitive science (neural correlate: ramping firing in LIP/parietal).

> **Mapping [interpretive]:** *buddhi*'s act of **ascertainment (niścaya)** = the accumulator
> **crossing the bound** — the moment doubt (*saṃśaya*, manas's *vikalpa*) collapses into a settled
> judgment. *Manas* supplies the candidate (the noisy evidence, *saṃkalpa-vikalpa*); *buddhi*
> integrates to a verdict. Raising the bound $a$ = the deliberate, dispassionate intellect (*sāttvika
> buddhi*, Gītā 18.30) that waits for evidence; a low bound = the impulsive *rājasika* mind. 🟢

---

## 5. Manas as a serial, capacity-limited gate

Nyāya (NS 1.1.16) and Charaka (manas is *aṇu/eka*) argue cognitions are **serial** because the mind
is atomic. Modern formalizations:

- **Single-channel / bottleneck** (Broadbent; psychological refractory period): the gate passes ~one
  stream at a time; concurrent tasks queue. A **serial sampler** at rate $1/\tau$ — one "selection"
  per dwell — is the continuous analogue of the Abhidhamma's 17-mind-moment series.
- **The Information Bottleneck** (Tishby): attention as compression — keep the bits of input $X$ that
  predict what matters $Y$, discard the rest:

$$
\min_{p(t\mid x)}\; I(X;T) - \beta\, I(T;Y)
$$

$T$ is the attended/"bound" representation; $\beta$ trades compression against relevance.

> **Mapping [interpretive]:** *manas* = the **lossy, serial gate** that selects one structured percept
> from the sensory flood — Charaka's "perception arises only when manas attends." Inattentional
> blindness is the empirical face of *manas* being engaged elsewhere (Charaka Sūtra 8.4). 🟢

---

## 6. The subconscious as an attractor — *saṃskāra = a basin, vāsanā = its width* (Hopfield)

**The first of three equations of the subconscious.** Associative memory is modeled as an energy
landscape (Hopfield network); stored patterns are **minima (attractors)**; recall is **descent** into
the nearest basin:

$$
E(\mathbf{s}) = -\tfrac12 \sum_{i\ne j} w_{ij}\,s_i s_j,\qquad
w_{ij} = \frac1N\sum_{\mu} \xi_i^{\mu}\,\xi_j^{\mu}\ \ (\text{Hebbian storage of patterns } \xi^\mu)
$$

A cue relaxes downhill to the stored pattern $\xi^\mu$ — *content-addressable* recall (a fragment
retrieves the whole).

> **Mapping [interpretive]:** each **stored $\xi^\mu$ is a *saṃskāra*** — a deposited trace; the
> **basin of attraction is its *vāsanā*** — a deep, wide basin = a strong habit that captures nearby
> states (the mind "falls into" the groove). The **engram** (Tonegawa) is the neural $\xi^\mu$. The
> "wheel" *vṛtti-saṃskāra-cakra* = a cue → relax to attractor → that attractor is re-deposited,
> deepening the basin. **Counter-conditioning (*nirodha-saṃskāra*)** = carving a competing, deeper
> minimum until it dominates the landscape. Pathological loops (rumination, addiction) = spurious
> over-deep attractors — the *kliṣṭa* (afflicted) vāsanās. 🟢

---

## 7. The subconscious as learning — *vāsanā/karma = value & policy* (reinforcement learning)

**The second equation of the subconscious**, and the closest formal cousin of **karma**. An agent
caches the value of states/actions from accumulated reward (temporal-difference learning):

$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)\quad(\text{prediction error}),\qquad V(s_t) \leftarrow V(s_t) + \alpha\,\delta_t
$$

and acts by a learned **policy** $\pi(a\mid s)$. Habits are a **cached policy** run without
deliberation (basal-ganglia actor-critic; dopamine encodes $\delta_t$).

> **Mapping [interpretive]:** **action → consequence → updated disposition** is precisely *karma*:
> deeds deposit value that biases future action **automatically**, below awareness. $V$/$\pi$ = the
> *vāsanā* (learned tendency); the **eligibility trace** $e_t$ (which past states get credit) is the
> "freshness" of an impression; $\gamma$ (discount) and forgetting give the **decay** the tradition
> insists on (saṃskāras are not eternal, YS 2.10–11). *Kuśala/akuśala* (wholesome/unwholesome)
> karma ≈ reward sign. The whole *abhyāsa* program = deliberately generating experiences that
> **re-shape $V$ and $\pi$** toward sattva. 🟢

**The deposit rule beneath both §6 and §7 — Hebbian plasticity** (the literal *vṛtti → saṃskāra*
event):

$$
\Delta w_{ij} = \eta\, x_i\, x_j \quad(\text{"fire together, wire together"; STDP adds timing})
$$

Every co-active experience **physically writes the trace.** This is the mechanism of the wheel.

---

## 8. Sleep as offline optimization — where the subconscious is *rewritten*

Tying the subconscious (§§6–7) to the four states (Māṇḍūkya):

- **Replay = experience replay.** Hippocampal reactivation in sleep (Wilson–McNaughton) is
  algorithmically the **experience replay** that stabilizes RL/deep learning — re-running stored
  $(s,a,r,s')$ to consolidate $V,\pi$. *Svapna* (dream) as replaying impressions (Praśna 4.5) ↔
  offline replay. 🟢
- **Consolidation = batch transfer** of traces from fast (hippocampal) to slow (neocortical) stores —
  the *saṃskāra* migrating from labile to stable form.
- **Synaptic homeostasis** (Tononi–Cirelli): net **downscaling/normalization** of weights in deep
  sleep, $w \leftarrow w - c$ (or $w \leftarrow w/Z$), preserving signal-to-noise. A precise reason
  *suṣupti* is **restorative** and "contentless" (Māṇḍūkya 5): the store is being *renormalized*, not
  read out. 🟡→🟢

---

## 9. The self & consciousness measures — *ahaṃkāra* and the rivals to *puruṣa*

- **The self-model (ahaṃkāra) as a hierarchical generative model** (Metzinger's self-model theory;
  Friston's "self as the model's deepest prior"): the "I" is the brain's **model of the modeler** —
  a high-level latent that attributes experiences to one owner. Its network correlate is the **Default
  Mode Network**; its quieting under meditation is measurable (Brewer 2011). *Ahaṃkāra = the
  self-prior*; "ego-dissolution" = down-weighting that prior. 🟢
- **Integrated Information Theory — a number for consciousness:**

$$
\Phi \;=\; \min_{\text{partition}} \; \big[\text{integrated information across the minimum-information partition}\big]
$$

$\Phi$ quantifies how much a system is **more than the sum of its parts**. **Mapping [interpretive,
and a *rival* to the texts]:** $\Phi$ formalizes the **unity** of awareness the texts insist on
(consciousness is *one*, undivided — *prajñāna-ghana*, Māṇḍūkya 5) — **but** IIT says consciousness
*is* this integration (a property of organized matter), directly **contradicting** the Sāṃkhya claim
that awareness is non-material *puruṣa*. The math captures the **unity**, not the **witness**. 🟡/🔴
- **Global ignition** (Dehaene): a content becomes conscious when recurrent activity crosses a
  threshold and **ignites** the global workspace — a **bifurcation** in the network dynamics.
  *Manas/buddhi making a content globally available* ≈ ignition. (Access, not the felt light.) 🟢

---

## 10. *Turīya / sākṣī* as an invariant — the one honestly-metaphysical "equation"

The texts' hardest claim: the witness (*sākṣī*, turīya) is **constant across all states**. The
cleanest formal rendering [reconstruction, explicitly metaphysical]:

Let the brain/mind state evolve under dynamics $F$ across regimes, $\;S_{t+1} = F(S_t)\;$ cycling
through $\{\text{wake},\text{REM},\text{NREM}\}$. Turīya is posited as an **invariant functional**
$\mathcal{O}$ — the "witnessing" — that is **unchanged by the dynamics**:

$$
\mathcal{O}(S_t) = \mathcal{O}^{*}\quad\text{for all } t \quad(\text{a fixed point / conserved quantity of the state-flow})
$$

i.e. whatever changes (the three states are the *orbit*), turīya is the **constant** the orbit moves
*within*. The Self is the **invariant**, the states are the **variables** (Māṇḍūkya 7: "the cessation
of the manifold … the Fourth").

> **Honest status 🔴:** this is **metaphysics, not physics.** Science can ask the *empirical* version —
> "is there awareness with no content, e.g. in deep sleep?" (the *minimal phenomenal experience* /
> witnessing-sleep research, modern-science file §2) — but the *strong* claim, that this invariant is
> the non-material ground of reality, is **not testable** and is neither confirmed nor refuted. The
> Buddhist rival denies any such invariant (the self is the orbit, no constant) — and **that
> disagreement is still live** in consciousness science (substrate-independent "ground" vs. emergent
> "process").

---

## 11. The guṇas as a dynamical state-vector — a cautious modern reading

The guṇa-simplex (Sanskrit-formula file §2) $\mathbf g=(s,r,t),\ s+r+t=1$ invites a **dynamical-
systems** reading [speculative 🟡]:

- treat $(s,r,t)$ as the **regime** of brain/mind dynamics: **sattva** ↔ ordered, integrated, low-
  entropy clarity; **rajas** ↔ high-energy, high-arousal activity; **tamas** ↔ low-arousal, inert,
  high-inertia dullness.
- This rhymes with the **"entropic brain"** hypothesis (Carhart-Harris): richness/criticality of
  conscious state tracks neural **entropy** — too-ordered (rigid) and too-disordered (chaotic) are
  both dysfunctional, with a **critical** sweet spot. One can read **sattva as the critical, balanced
  regime**, rajas as over-driven, tamas as collapsed.
- A toy flow on the simplex, $\dot{\mathbf g} = G(\mathbf g) + u(t)$, with practice/diet as control
  $u$ pushing toward the sattva vertex, formalizes Gītā 6.35's *abhyāsa* as **control toward a target
  regime**.

> Flagged **🟡 speculative**: the (s,r,t)↔(order/energy/inertia) mapping is a *suggestive analogy*,
> not established neuroscience. Included because it is the natural mathematical home of the guṇa idea
> — and clearly labeled so it is not mistaken for a finding.

---

## 12. The subconscious — one consolidated picture (the brief's emphasis)

Pulling §§2, 6, 7, 8 into a single account of *saṃskāra/vāsanā*, the part the user asked to foreground:

**The store has four mathematical faces, all the same underlying object:**

| Face | Equation | Vedic term |
|---|---|---|
| **as prior** (how it shapes perception) | $P(h\mid d)\propto P(d\mid h)\,P(h)$ — vāsanā = $P(h)$ | *vāsanā* biasing the next *vṛtti* |
| **as attractor** (how it stores & recalls) | $E=-\tfrac12\sum w_{ij}s_is_j$, basin width = strength | *saṃskāra* = stored pattern, *vāsanā* = basin |
| **as value/policy** (how it drives action) | $V\!\leftarrow\!V+\alpha\delta$, cached $\pi(a\mid s)$ | *karma/vāsanā* = learned tendency |
| **as deposit rule** (how it's written) | $\Delta w_{ij}=\eta\,x_ix_j$ (Hebb/STDP) | *vṛtti → saṃskāra* (the wheel) |

**The loop** (*vṛtti-saṃskāra-cakra*) as one recurrent update:

$$
\text{prior}_{t+1} \;=\; (1-\lambda)\,\text{prior}_{t} \;+\; \eta\,\underbrace{\Delta(\text{experience}_t)}_{\text{new saṃskāra}}
$$

— accumulation ($\eta$), **decay/forgetting** ($\lambda$, the tradition's non-eternal store), and the
updated prior feeding the next perception. **Liberation-as-engineering:** *nirodha-saṃskāra* /
*abhyāsa* deposits a strong competing groove until it dominates the landscape (§6) and re-shapes
$V,\pi$ (§7); **sleep** consolidates and renormalizes the store offline (§8). This is a complete,
mechanistic restatement of the Yogic subconscious — every step has both a Sanskrit name and an
equation.

> 🟢 on the engineering (priors, attractors, RL, Hebbian deposit, replay are all well-established);
> 🟡 on the precise *mapping* of each to its Sanskrit term (interpretive but defensible at the
> functional level).

---

## 13. Master mapping table

| Vedic term | Modern model | Core equation | Bucket |
|---|---|---|---|
| **vāsanā** (disposition) | Bayesian prior | $P(h\mid d)\propto P(d\mid h)P(h)$ | 🟢 |
| **citta-vṛtti-nirodha** | free-energy minimization | $F = \mathbb E_q[-\ln P(o\mid s)] + D_{\mathrm{KL}}[q\|P(s)]$ | 🟡 |
| **buddhi** (ascertainment) | drift-diffusion decision | $dx=v\,dt+\sigma\,dW,\ x\!\ge\!a$ | 🟢 |
| **manas** (serial gate) | bottleneck / info-bottleneck | $\min I(X;T)-\beta I(T;Y)$ | 🟢 |
| **saṃskāra** (trace) | Hopfield attractor / engram | $E=-\tfrac12\sum w_{ij}s_is_j$ | 🟢 |
| **karma / vāsanā** (driving action) | reinforcement learning | $V\!\leftarrow\!V+\alpha\delta_t$ | 🟢 |
| **vṛtti→saṃskāra** (deposit) | Hebbian/STDP plasticity | $\Delta w_{ij}=\eta x_i x_j$ | 🟢 |
| **svapna** (dream) | generative sampling, $\pi_{\text{sense}}\to0$ | replay of priors | 🟢 |
| **suṣupti** (deep sleep) | consolidation + homeostatic downscaling | $w\!\leftarrow\!w/Z$ | 🟢 |
| **ahaṃkāra** (I-maker) | self-model / DMN | self = deepest prior | 🟢 |
| **awareness-unity** (prajñāna-ghana) | IIT integrated information | $\Phi$ | 🟡/🔴 |
| **turīya / sākṣī** | invariant / fixed point of state-flow | $\mathcal O(S_t)=\mathcal O^{*}$ | 🔴 (metaphysical) |
| **guṇas** (s,r,t) | dynamical regime / entropic brain | $\dot{\mathbf g}=G(\mathbf g)+u$ | 🟡 |

---

## 14. Caveats (read before quoting any of this)

1. **These equations are modern.** The Vedic texts contain none of them; the mapping is an
   **interpretive bridge at the functional level**, not a claim that the ṛṣis derived calculus or
   probability. (The *indigenous* formal structures — counts, simplex, nestings — are in the
   Sanskrit-formula file.)
2. **Bucket honestly.** 🟢 = the model is well-established *and* the mapping is defensible; 🟡 = model
   established but mapping/relevance is suggestive or the science is young; 🔴 = the ancient claim is
   metaphysical (turīya as ground) and untestable, or a modern model (IIT) actively *contradicts* the
   ancient metaphysics.
3. **The subconscious is the strongest case** — priors, attractors, RL, Hebbian deposit, and sleep-
   replay give *saṃskāra/vāsanā* a genuinely precise, multi-angle mathematical home. The *witness*
   (turīya) is the weakest — it is exactly the part that escapes equations, in both the ancient and
   the modern account.
4. **Do not over-quantify the guṇas or chakras.** The (s,r,t) dynamics (§11) is a labeled *speculation*;
   chakra/nāḍī numbers are symbolic (Sanskrit-formula file), not parameters of any real equation.
