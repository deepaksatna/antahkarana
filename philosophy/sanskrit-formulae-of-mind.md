# Sanskrit Formulae of Mind
### The enumerative, proportional, combinatorial & structural "equations" of the Indian psychology — tattvas · the guṇa-simplex · pañca-kośa · AUM-mātrā · chakra-arithmetic · the saṃskāra accumulator

> **Companion to** [`mind-brain-consciousness-in-vedic-texts.md`](mind-brain-consciousness-in-vedic-texts.md)
> and [`modern-science-and-the-vedic-mind.md`](modern-science-and-the-vedic-mind.md). For the *modern*
> equations (Bayesian priors, Hopfield energy, drift-diffusion, IIT-Φ, RL) see
> [`mathematical-models-of-mind-and-consciousness.md`](mathematical-models-of-mind-and-consciousness.md).

> **An honest preface — what "formula" means here.** The classical mind-texts contain **no algebra,
> calculus, or dynamical equations**. What they *do* contain — and what this file collects — are
> genuine **formal structures**: exact **enumerations** (the 25 tattvas), **proportional/mixture
> models** (the three guṇas), **nested hierarchies** (the five sheaths), **combinatorial
> correspondences** (chakra-petals ↔ the 50 phonemes), **discrete-time process counts** (the 17
> mind-moments), and **accumulation rules** (the karma/saṃskāra store). These are "formulae" in the
> structural and combinatorial sense — and India's mathematical tradition (Piṅgala's binary
> combinatorics, §9) shows the culture *could* formalize rigorously. Each entry below is tagged
> **[canonical]** (the structure is explicit in the texts) or **[reconstruction]** (a faithful
> formal restatement *I* give of a textual idea, made explicit for clarity). Nothing here is
> numerology; where a correspondence is symbolic, it is flagged.

---

## 1. The tattva-enumeration — Sāṃkhya's "25 = the complete count of reality" [canonical]

Sāṃkhya is, at its core, an **enumeration** (the word *sāṃkhya* itself means "relating to number /
counting"). Reality is exhaustively counted as **25 tattvas** ("thatnesses," principles), of which the
mind-faculties are members:

$$
\underbrace{\text{Puruṣa}}_{1}\;+\;\underbrace{\text{Prakṛti}}_{1}\;+\;\underbrace{\text{buddhi/mahat}}_{1}\;+\;\underbrace{\text{ahaṃkāra}}_{1}\;+\;\underbrace{\text{manas}}_{1}\;+\;\underbrace{\text{5 jñānendriya}}_{5}\;+\;\underbrace{\text{5 karmendriya}}_{5}\;+\;\underbrace{\text{5 tanmātra}}_{5}\;+\;\underbrace{\text{5 mahābhūta}}_{5}\;=\;25
$$

Grouped (Sāṃkhya Kārikā 3) into four classes by causal role:

| Class | Members | Count |
|---|---|---|
| *prakṛti* only (uncaused cause) | Prakṛti | 1 |
| *prakṛti-vikṛti* (both cause & effect) | buddhi, ahaṃkāra, 5 tanmātra | 7 |
| *vikṛti* only (effect, not cause) | manas, 5 jñānendriya, 5 karmendriya, 5 mahābhūta | 16 |
| *neither* (pure witness) | Puruṣa | 1 |
| | **total** | **25** |

> **The mind in the count:** *buddhi, ahaṃkāra, manas* are tattvas **3–5** — the *antaḥkaraṇa* — and
> the senses are tattvas 6–15. The entire cognitive apparatus is **13 of the 25** (buddhi + ahaṃkāra
> + manas + 10 senses = the *thirteenfold instrument*, *trayodaśa-karaṇa*, SK 32).

**Tantric extension — 36 tattvas** [canonical, Śaiva]: Kashmir Śaivism keeps the 25 and adds **11
"pure/impure" tattvas** above them — *māyā* + the **5 kañcukas** (the five "sheaths/limiters":
*kalā, vidyā, rāga, kāla, niyati*) + *śuddha-vidyā, īśvara, sadāśiva, śakti, śiva* — giving
**25 + 11 = 36**. The 5 kañcukas are a precise **"limitation operator"**: they are *what reduces*
infinite consciousness (śiva) to a finite, located, time-bound knower — a formal model of
**individuation** (universal awareness → bounded ego) by five successive constraints.

---

## 2. The guṇa-simplex — mind as a *proportion of three qualities* [canonical idea / reconstruction of the formalism]

Sāṃkhya's deepest quasi-mathematical claim: every material/mental phenomenon is a **mixture of three
qualities (*guṇa*)** — *sattva* (clarity/order), *rajas* (activity/energy), *tamas* (inertia/mass) —
in varying proportion (SK 12–13). A mind-state is therefore not a type but a **point in a
3-component mixture**:

$$
\mathbf{g} = (s,\,r,\,t),\qquad s+r+t = 1,\qquad s,r,t \ge 0
$$

i.e. a point on the **2-simplex** (a triangle whose vertices are pure sattva, pure rajas, pure
tamas). [reconstruction — the *normalization* is my explicit statement; the *mixture* is canonical.]

- The **vertices** are limiting ideals; real minds sit in the interior.
- **State change = movement on the simplex.** *Abhyāsa/vairāgya* (Gītā 6.35) and a sāttvika diet/
  conduct (Gītā 17) are operations that **push g toward the sattva vertex**; indulgence pushes toward
  rajas/tamas. The three *citta* states map onto regions: agitation (*kṣipta/vikṣipta*) ≈ high rajas;
  dullness (*mūḍha*) ≈ high tamas; one-pointedness (*ekāgra*) ≈ high sattva.
- **The Gītā's typology is a classifier on this simplex:** ch. 14/17/18 grade buddhi, dhṛti, action,
  knowledge, faith, food, even charity into *sāttvika / rājasika / tāmasika* by which coordinate
  dominates. **Charaka's 16 personality types** (7 sāttvika + 6 rājasika + 3 tāmasika, *Śārīra 4*)
  is the same classifier applied to temperament.

> **Why this is the single most "formula-like" idea in the corpus:** it replaces *categorical* mind-
> types with a **continuous, normalized, 3-dimensional state vector** — structurally a probability
> distribution / barycentric coordinate. The modern echo (a dynamical-systems / entropy reading of
> (s,r,t)) is developed in the math companion, §11.

---

## 3. The pañca-kośa — a nested-composition (layered-encoder) formula [canonical]

Taittirīya Up. 2.1–2.5 wraps the *ātman* in **five sheaths**, each *contained within* the previous —
an explicit **nested structure**:

$$
\text{Person} = K_{\text{anna}}\!\big(K_{\text{prāṇa}}\!\big(K_{\text{manas}}\!\big(K_{\text{vijñāna}}\!\big(K_{\text{ānanda}}(\,\text{ātman}\,)\big)\big)\big)\big)
$$

[reconstruction of the canonical nesting as function composition.] Each $K$ is a "sheath/wrapper"
around a subtler core; **the ātman is the fixed innermost argument, not any wrapper.** Read outward it
is a **gross→subtle decomposition**; read inward it is **peeling layers toward an invariant core** —
the same shape as a hierarchical encoder collapsing toward a latent, or as the faculty hierarchy of
Kaṭha 1.3.10. The **layer order matches the state model**: the *ānandamaya* (bliss) sheath ↔ deep
sleep's bliss (*suṣupti/Prājña*).

---

## 4. The AUM-mātrā formula — counting the syllable of consciousness [canonical]

Māṇḍūkya 8–12 makes **OM a measured object**: it has *mātrās* ("measures/morae"), and the states map
to them. Later texts (e.g. *Nāda-bindu / Dhyānabindu Up.*) total the measures:

$$
\text{AUM} = \underbrace{A}_{1\,\text{mātrā}} + \underbrace{U}_{1\,\text{mātrā}} + \underbrace{M}_{1\,\text{mātrā}} + \underbrace{\text{ardha-mātrā (nāda/bindu)}}_{\tfrac12\,\text{mātrā}} \;\;=\;\; 3\tfrac12\ \text{mātrās}
$$

with the **fourth, turīya, as *amātra* — "measure-less"** (Māṇḍūkya 12, *amātraś caturthaḥ*):

$$
\text{turīya} = \lim \;(\text{measure} \to 0)\;:\quad \text{the silent } a\text{-mātra into which AUM resolves}
$$

[the 3½-mātrā total is canonical in the later Upaniṣads; the "limit" notation is a reconstruction.]
**Reading:** the three positive states are *countable* (finite measure); the ground is the
**measureless** that bounds them — a structure exactly mirrored by the modern "states are measurable,
the witness is the un-measurable invariant" puzzle (math companion, §10).

---

## 5. Chakra arithmetic — the petal-count ↔ phoneme correspondence [canonical, symbolic]

The six lower chakras carry lotus-**petals**, each inscribed with a Sanskrit phoneme (*varṇa/
mātṛkā*). The counts are fixed and they **sum exactly to the 50 letters of the alphabet**:

$$
\underbrace{4}_{\text{mūlādhāra}}+\underbrace{6}_{\text{svādhiṣṭhāna}}+\underbrace{10}_{\text{maṇipūra}}+\underbrace{12}_{\text{anāhata}}+\underbrace{16}_{\text{viśuddha}}+\underbrace{2}_{\text{ājñā}} \;=\; 50 \;=\; \underbrace{16\ \text{vowels} + 34\ \text{consonants}}_{\text{the 50 mātṛkā varṇas}}
$$

and the crown:

$$
\text{sahasrāra} = 1000\ \text{petals} = 50\ \text{varṇas} \times 20\ \text{layers}
$$

> **Status [symbolic correspondence].** This is a real, intended numerical scheme (*Ṣaṭ-cakra-
> nirūpaṇa*): the body's centres are made to *contain the whole sound-alphabet*, so that the ascent of
> *kuṇḍalinī* through 50 petals = traversing the entire matrix of language/creation, culminating in
> the 1000-petalled crown where all 50 recur twentyfold. It is **numerologically deliberate, not
> anatomical** — a *formula of symbolic completeness*, flagged as such. (The total nāḍī count
> **72,000** is similarly a "completeness" number, not a dissection count.)

---

## 6. The saṃskāra / karma accumulator — the subconscious as an integrator [canonical idea / reconstruction]

**This is the formula of the subconscious — the user's emphasis — stated in the tradition's own
terms.** Yoga gives the store, the deposit rule, and the ripening rule:

**(a) The deposit (the wheel, *vṛtti-saṃskāra-cakra*).** Every mental modification leaves a trace; the
store accumulates:

$$
\text{chitta-store}_{t+1} = \text{chitta-store}_{t} \;\oplus\; \text{saṃskāra}(\text{vṛtti}_t)
$$

and like-traces **reinforce into a tendency** (*vāsanā*) — repetition deepens the groove
(*abhyāsa*, YS 1.14: practice becomes firm-grounded when **long, uninterrupted, and earnest**). [the
accumulation + reinforcement is canonical; the recurrence notation is a reconstruction.]

**(b) The feedback.** A *vāsanā* re-emerges as the next *vṛtti*, biasing perception **before
awareness** — the store is a **recurrent prior on experience** (YS 4.8: from each kind of karma, only
the matching *vāsanā* manifests).

**(c) The ripening (*karma-vipāka*).** YS 2.12–2.13 [canonical]: while the *kleśa*-root persists, the
*karma-āśaya* (karmic deposit) **fruits in three coordinates**:

$$
\text{karma-āśaya} \;\xrightarrow{\text{ripens as}}\; (\;\underbrace{jāti}_{\text{birth/type}},\;\underbrace{\bar{a}yus}_{\text{duration}},\;\underbrace{bhoga}_{\text{experience}}\;)
$$

**(d) The decay / overwrite.** The store is **not** permanent: YS 2.10–2.11 (subtle kleśas dissolved
by tracing them to their source; gross ones by meditation) and YS 3.9–10 (the *nirodha-saṃskāra* of
stillness progressively dominates the outgoing traces). [canonical] → a **counter-conditioning**
rule: deposit a stronger opposite groove until it wins. The kleśas themselves carry an **activation
level**: *prasupta* (dormant) → *tanu* (attenuated) → *vicchinna* (interrupted) → *udāra* (active)
(YS 2.4) — a four-step "arousal state" of a latent disposition.

> **The shape:** the subconscious is an **accumulating, decaying, self-biasing store** — an
> integrator with reinforcement and counter-conditioning. The modern equations that realize this rule
> (Hebbian deposit, attractor basin, RL value, prior update) are in the math companion, §6–7 & §12.

---

## 7. The aṣṭāṅga pipeline & the samādhi taxonomy — staged process formulae [canonical]

**Eight limbs (YS 2.29) as an 8-stage pipeline** toward stilling the mind:

$$
\text{yama} \to \text{niyama} \to \text{āsana} \to \text{prāṇāyāma} \to \text{pratyāhāra} \to \text{dhāraṇā} \to \text{dhyāna} \to \text{samādhi}
$$

The last three **compose into one operation**, *saṃyama* (YS 3.4):

$$
\text{saṃyama} = \text{dhāraṇā} \;\circ\; \text{dhyāna} \;\circ\; \text{samādhi}\quad(\text{concentration} \to \text{flow} \to \text{absorption, on one object})
$$

**Samādhi is itself graded** (YS 1.17) — *samprajñāta* ("with-seed") has four nested supports that
**drop off one by one**:

$$
\text{vitarka (gross object)} \supset \text{vicāra (subtle object)} \supset \text{ānanda (bliss)} \supset \text{asmitā (bare I-am)} \;\longrightarrow\; \text{asamprajñāta (seedless, } nirbīja)
$$

a **monotone reduction**: each stage removes one layer of content until only objectless awareness
remains — the contemplative analogue of the pañca-kośa peeling (§3) and the AUM resolving to silence
(§4). The same "remove a layer → approach the invariant" structure recurs three times in the corpus.

---

## 8. The 17 mind-moments — a discrete-time cognition formula [canonical, Abhidharma]

The Buddhist **Abhidhamma** quantizes cognition in **time**. The smallest unit is the *citta-kkhaṇa*
(mind-moment); a full sense-cognition (*citta-vīthi*, "cognitive series") for a vivid object unfolds
over **exactly 17 mind-moments**:

$$
\underbrace{\text{bhavaṅga}\times3}_{\text{rest + vibration + arrest}} \to \text{5-door advert.} \to \text{sense-consciousness} \to \text{receiving} \to \text{investigating} \to \text{determining} \to \underbrace{\text{javana}\times7}_{\text{impulsion (where karma forms)}} \to \underbrace{\text{registration}\times2}_{} \;=\; 17
$$

This is a genuine **clocked, serial, fixed-length pipeline** for a single perception — and the **seven
*javana* moments are exactly where a *saṃskāra* is laid down** (karma is generated in impulsion). It
is the ancient world's most explicit **"cycles per perception"** model, and it operationalizes the
Nyāya claim (NS 1.1.16) that the atomic *manas* makes cognitions **serial, never simultaneous**:
roughly one cognitive series at a time, mind-moment by mind-moment.

---

## 9. The proof that the culture *could* formalize — Piṅgala's combinatorics [canonical, but prosody not psychology]

To pre-empt the objection "these aren't *real* math": the same civilization produced rigorous
combinatorial mathematics — in **prosody (*chandaḥśāstra*)**, not psychology — centuries early:

- **Piṅgala (c. 300–200 BCE)** enumerated poetic meters by **binary** (light *laghu* / heavy *guru*
  syllables): a meter of $n$ syllables has $2^n$ patterns — **binary numerals** and place-value
  combinatorics.
- **Meru-prastāra** ("the staircase of Mount Meru") — Piṅgala's rule for counting meters by syllable-
  weight, elaborated by **Halāyudha (10th c.)** into what is now **Pascal's triangle** ($\binom{n}{k}$).
- **Mātrā-meru** — counting meters by *total duration* (mātrā) yields the **Fibonacci sequence**
  ($F_{n}=F_{n-1}+F_{n-2}$), described by Piṅgala/Virahāṅka before Fibonacci.

> **The point:** the absence of equations *for the mind* is a choice of register (the mind-texts are
> contemplative/analytic, not quantitative), **not** an inability to formalize. The structures in
> §§1–8 are formal; the §9 tradition shows the formal capacity was fully present.

---

## 10. Master formula-sheet

| # | "Formula" | Canonical expression | Models | Tag |
|---|---|---|---|---|
| 1 | Tattva count | $1{+}1{+}1{+}1{+}1{+}5{+}5{+}5{+}5 = 25$ (→36 Tantra) | complete inventory of reality + mind | canonical |
| 2 | Guṇa-simplex | $s+r+t=1,\ s,r,t\ge0$ | mind-state as continuous 3-vector | canonical/recon |
| 3 | Pañca-kośa | $K_{anna}\!\circ\!K_{prāṇa}\!\circ\!K_{manas}\!\circ\!K_{vijñāna}\!\circ\!K_{ānanda}(\text{ātman})$ | nested gross→subtle layers | canonical/recon |
| 4 | AUM-mātrā | $A{+}U{+}M{+}\tfrac12 = 3\tfrac12$ ; turīya $=$ amātra | states as measures; ground as measureless | canonical |
| 5 | Chakra-petals | $4{+}6{+}10{+}12{+}16{+}2 = 50$ varṇas; crown $=1000$ | symbolic completeness of the sound-matrix | symbolic |
| 6 | Saṃskāra store | accumulate ⊕ reinforce → ripen $(jāti,āyus,bhoga)$ → decay | **the subconscious as integrator** | canonical/recon |
| 7 | Saṃyama | $\text{dhāraṇā}\circ\text{dhyāna}\circ\text{samādhi}$ | composed concentration operation | canonical |
| 7b | Samādhi reduction | vitarka ⊃ vicāra ⊃ ānanda ⊃ asmitā → nirbīja | monotone layer-stripping | canonical |
| 8 | Citta-vīthi | $3{+}1{+}1{+}1{+}1{+}1{+}7{+}2 = 17$ mind-moments | clocked serial perception pipeline | canonical |
| 9 | Piṅgala | $2^n$ meters; $\binom{n}{k}$; $F_n=F_{n-1}+F_{n-2}$ | the culture's formal capacity | canonical |

---

## 11. Caveats

- **[canonical]** entries are explicit structures in the texts (counts, mixtures, nestings, stage-
  lists). **[reconstruction]** entries restate a textual idea in modern formal notation for clarity —
  the *idea* is canonical, the *notation* is mine and is labeled. **[symbolic]** entries (chakra/nāḍī
  numbers) are deliberate symbolic schemes, **not** anatomical or empirical counts.
- **No numerology.** Where a number is symbolic (50, 1000, 72,000), it is named as such; no hidden-
  meaning claims are made.
- The genuine *dynamical mathematics* of these phenomena (rates, equations of change, optimization)
  comes from **modern science** and lives in the companion file — these Sanskrit formulae are
  **structural and combinatorial**, and that is exactly their character.
