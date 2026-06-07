# Vedas — Mind, Brain & Consciousness in the Indian Traditions

A four-file study of the **philosophy of mind, brain, consciousness, the subconscious, and dream** in
the Vedic / Sanskrit traditions, paired with what **modern science** says, and the **formulae** —
both the tradition's own and the modern equations — behind it.

| File | What it covers |
|---|---|
| [`mind-brain-consciousness-in-vedic-texts.md`](mind-brain-consciousness-in-vedic-texts.md) | **The texts & mantras.** Antaḥkaraṇa (manas/buddhi/ahaṃkāra/chitta), the four states (waking/dream/deep-sleep/turīya), Sāṃkhya, the subconscious (saṃskāra/vāsanā/kleśa), pañca-kośa, **tapas/samyama/siddhi** (effort & emergent capability), Āyurvedic anatomy, Yoga–Tantra subtle body, the other schools — with Devanāgarī + IAST + translation + citation. |
| [`modern-science-and-the-vedic-mind.md`](modern-science-and-the-vedic-mind.md) | **What modern science says.** Neuroscience, sleep science, the cognitive unconscious, contemplative neuroscience — each ancient claim bucketed 🟢 convergence / 🟡 open research / 🔴 superseded-or-untestable. |
| [`sanskrit-formulae-of-mind.md`](sanskrit-formulae-of-mind.md) | **The tradition's own formal structures.** The 25/36 tattvas, the guṇa-simplex, pañca-kośa nesting, AUM-mātrā, chakra-arithmetic (50 petals = 50 phonemes), the 17 mind-moments, and the saṃskāra/karma accumulator. |
| [`mathematical-models-of-mind-and-consciousness.md`](mathematical-models-of-mind-and-consciousness.md) | **The modern equations.** vāsanā = Bayesian prior, saṃskāra = Hopfield attractor, karma = reinforcement learning, buddhi = drift-diffusion, manas = serial bottleneck, consciousness = IIT-Φ, turīya = invariant of the state-flow. |
| [`vedic-cognitive-ai-architecture.md`](vedic-cognitive-ai-architecture.md) | **The capstone — building an AI on this model.** The *Antaḥkaraṇa-Net*: manas/buddhi/ahaṃkāra/chitta as modules, a guṇa meta-controller, a wake/dream/sleep cycle, a turīya oversight monitor, and an **āśrama lifelong-learning law** (childhood→old age) that is *always enhanceable*. Real ML methods + the Vedic structure as organizing principle. |
| [`building-it-engineering-options.md`](building-it-engineering-options.md) | **The engineering plan.** Build-vs-buy per module (PyTorch/JAX, Avalanche, MoE, snnTorch…), the new *ChittaKit* library, physical-world embodiment (ROS 2/Isaac/Jetson, battery→guṇa), the **energy case** vs. today's LLMs, neuromorphic hardware (Loihi/Akida), and a phased A→B→C build path. |
| [`antahkarana-net/`](antahkarana-net/) | **Runnable Track-A prototype (code).** ChittaKit in PyTorch — `SamskaraMemory` (EWC + decay), `GunaController`, `AshramaSchedule`, `TapasController`, `WitnessMonitor` — plus three benchmarks. Runs on CPU. In the capacity-headroom benchmark, consolidation **eliminates forgetting ~60–80×** (0.242→0.003), saṃskāra-decay beats EWC, and tapas-replay beats uniform. See its `README.md` + honest `RESULTS.md`. |

**Reading order:** texts → modern science → Sanskrit formulae → mathematical models → **AI architecture**.
The **subconscious** (*saṃskāra/vāsanā*) is treated in depth in every file (texts §5; science §4;
Sanskrit-formulae §6; math §12; architecture §4.2 — the always-enhanceable memory core).

**Stance throughout:** the tradition's strength is a **functional model of mind** that often carved
mind at joints modern science also finds; its **physical brain anatomy** is thin (heart-centric,
no functional map). Every ancient↔modern pairing is flagged as **interpretation**, and metaphysical
claims (turīya as the ground of reality) are marked **untestable** — never "the ancients already knew
neuroscience."
