# Memory for Embodied AI

### A critical survey of robot memory architectures (2024–2026), the evidence they rest on, and a low-latency architecture derived from it

**Compiled:** 16 October 2026
**Method:** five parallel research agents, each assigned a non-overlapping slice of the literature, each instructed to separate *demonstrated / simulated-only / proposed* and to refuse unverified claims. Findings below are synthesised and cross-checked; disagreements between slices are preserved rather than reconciled.
**Scope:** physical robots, embodied AI, VLA policies. Language-agent memory is used only as a contrast case.

---

## Table of contents

- [0. TL;DR](#0-tldr)
- [1. Evidence conventions](#1-evidence-conventions)
- [2. The central finding](#2-the-central-finding)
- [3. The evidence ledger](#3-the-evidence-ledger)
- [4. Taxonomy: four memory lineages](#4-taxonomy-four-memory-lineages)
- [5. A recurrent state is not a belief state](#5-a-recurrent-state-is-not-a-belief-state)
- [6. Memory inside current VLAs](#6-memory-inside-current-vlas)
- [7. Episodic, retrieval and scene memory](#7-episodic-retrieval-and-scene-memory)
- [8. Video as a memory substrate](#8-video-as-a-memory-substrate)
- [9. Video-based answering](#9-video-based-answering)
- [10. The robot advantage: verification by acting](#10-the-robot-advantage-verification-by-acting)
- [11. What industry actually ships](#11-what-industry-actually-ships)
- [12. Evaluation: the ladder](#12-evaluation-the-ladder)
- [13. Benchmarks](#13-benchmarks)
- [14. Recommended architecture](#14-recommended-architecture)
- [15. Build path](#15-build-path)
- [16. Open problems](#16-open-problems)
- [17. Citation hygiene](#17-citation-hygiene)
- [18. Caveats and known gaps](#18-caveats-and-known-gaps)

---

## 0. TL;DR

**Memory is not capacity. Memory is selection plus verification.**

Five independent literatures converge on this, and the convergence is the strongest signal in this document. Every mechanism that works is *selective and checked against evidence*. Every mechanism that fails is *voluminous and unverified* — and the failures are not subtle.

The most important individual results:

| Claim | Source |
|---|---|
| Frozen VLAs' stored history contains **≈ zero information** not already present in the current frame | `arXiv:2607.03372` |
| **Storage** is solved (94%); **update** and **compression** are not (4.4%) | MEMOBench `arXiv:2609.07047` |
| Naive frame stacking *degrades* performance — LIBERO 95.6 → 86.8 | HAMLET `arXiv:2510.00695`, ICLR 2026 |
| Stale memory is **2.7× deadlier** than having no memory | `arXiv:2608.04574` |
| An **interrupted** memory is worse than no memory (73% vs 96%) | DRAM `arXiv:2609.32453` |
| Keeping **100% of frames is the worst retention ratio tested** | FrameSkip `arXiv:2605.13757` |
| Retrieval quality does **not** predict task success; **random retrieval is non-trivial** | `arXiv:2610.05492` |
| Success does not certify the *decision* — 5/9 wrong-target on hardware | CMA `arXiv:2609.27247` |
| **Two frames already** triples emergent-skill rate (14.5% → 44.4%) | Open X-Embodiment `arXiv:2310.08864`, ICRA 2024 |

The survey also produces a deployable architecture (§14) targeting **30–50 Hz action-loop memory** with a **~66 KB fixed-size state**, at roughly 40 M trainable parameters.

**What a practitioner should take away, in order:**

1. Do not put a VLM on the fast path. It costs 9× latency for memory frames that are nearly free.
2. Sample frames uniformly from a window. Never use the last *N* contiguous frames.
3. Gate memory writes on *verified* progress. Naive accumulation breaks the train/test distribution.
4. Always keep a stateless fallback. A corrupted state is worse than no state.
5. Prove your memory works with rung-3 and rung-4 ablations (§12), not with an aggregate success number.

---

## 1. Evidence conventions

Used throughout, and applied strictly:

| Tag | Meaning |
|---|---|
| **[P]** | Peer-reviewed / published at a named venue |
| **[TR]** | Company tech report or arXiv preprint, unreviewed |
| **[B]** | Blog, press, or third-party claim |
| **[?]** | Could not verify |

Two framing criteria recur and are worth stating up front:

**What makes a task require memory.** A task is memory-requiring at state *s* iff two histories `h¹ ≠ h²` exist with the **same** current observation `o` but **different optimal actions** — equivalently, `I(H_t ; A*_t | O_t) > 0`. POBAX (`arXiv:2508.00046`) formalises this as **memory improvability**: a benchmark is valid only if there is a measurable gap between agents with and without the state information, all else equal.

**Full state is unobservable, so memory exists.** A policy sees `o_t`, never `s_t`. When `P(a_t | o_t)` is genuinely ambiguous, no architecture fixes it — the information is not in the observation. Every system called "memory" is an approximation to the belief state

$$b_t(s) = P(s_t = s \mid o_{1:t},\, a_{1:t-1})$$

This is the single line the whole field is engineering against.

---

## 2. The central finding

Across five slices, the same conclusion reached from four incompatible directions.

**Recurrent/SSM policies** — DRAM's reset sweep shows an interrupted state falls *below* a stateless baseline. The state carries something real, and carrying it badly is worse than not carrying it.

**VLA policies** — naive multi-frame conditioning measurably degrades LIBERO, RoboCasa and SimplerEnv. History is not a free lunch; it is a distribution-shift hazard.

**Retrieval-augmented policies** — retrieval quality metrics are disconnected from task success, and random retrieval is non-trivial. Adding a retrieval module may not be where the value is.

**Video-conditioned policies** — 100% frame retention is the worst option; 0.5 fps matches 32 frames; 4 frames matches 13 published streaming-memory baselines.

The mechanism behind much of this is now known. **GAP** (ICLR 2026) found that during motion-transition sub-phases, policies *"naturally gravitate toward concise proprioceptive signals that offer faster loss reduction, thereby dominating the optimization and **suppressing the learning of the visual modality**."* The optimiser prefers the cheap signal. This is why adding frames often fails: you are not adding information, you are adding a nuisance the optimiser routes around.

> **The design principle.** Memory earns its cost by being *selective* (writing and reading chosen information, not everything) and *verified* (checked against physical evidence, not trusted because it was written). Every mechanism that does both is in §14 or its relatives. Every mechanism that does neither is in §3.

---

## 3. The evidence ledger

The negative results. These are the substance of the field, and they are better replicated than the positive ones.

### 3.1 History that isn't there

**`Present but Not Remembered`** (Liao & Cao, UNSW, `arXiv:2607.03372`) audits how frozen VLAs encode, deploy and steer visual history, using a linear decode probe and layer-resolved interchange interventions on frozen, open-loop policies.

Two probes:

- `A₁(ℓ) = R²(φ_ℓ(x), s_{t−1})` — *is the past present?* Answer: **past-frame content is linearly decodable at every depth.**
- `A₄(ℓ) = R²(φ_ℓ(x), s_{t−1} − ŝ_{t−1|t})` — *is anything unique to the past?* Answer: **A₄ ≈ 0**, a ceiling established over an 81-configuration probe sweep across three models and two architecture families.

> *"The stored history is a redundant copy of the present."*

Their closing prescription is the most actionable sentence in the survey: **"future memory augmentation should inject information unique to the past rather than simply more history."**

### 3.2 Storage solved, everything else unsolved

**MEMOBench** (`arXiv:2609.07047`) evaluates memory operations through *executable simulator predicates* rather than learned judges, with sequence-aware prefix-completion scoring so end-state credit cannot mask a process failure.

It separates three operations — **Storage**, **Update**, **Compression** — and the dissociation is stark:

```
π0.5, temporal tasks, 5-step replan:
  Memory Storage Rate      94.0%
  Success Rate              4.4%
```

Every memory architecture in the literature optimises what goes **in**. Almost none handles revising a stored belief when contradicted (**Update**) or abstracting correctly (**Compression**). MEM's language memory, MemER's keyframes and RoboMME's subgoal variants are all fundamentally **append-memory** designs.

The shuffled-subgoal control is decisive: 90% at the first storage checkpoint → 24% after a state update → **2% final**.

### 3.3 Memory that is worse than none

**Stale memory.** `arXiv:2608.04574` — 1,800 detection runs, 12,000 navigation episodes, 6 VLMs:

| GPT-4o, FrozenLake | Death rate |
|---|---|
| Trust raw memory | **74.4%** |
| No memory at all | **28.0%** |

2.7× deadlier. And the sobering follow-on: **oracle** stale labels gave *no significant gain* over learned text filtering. Per-seed detection F1 and success are **uncorrelated** (r ∈ [0.005, 0.060], p > 0.67). Every remaining death occurred in a run with **F1 above 0.9** — the entry *was* detected and removed and the agent stepped on the hole anyway.

> The bottleneck has moved from *detecting* stale memory to **acting safely** on memory you have already filtered.

**Interrupted memory.** DRAM (`arXiv:2609.32453`) reset the state at varying intervals:

| Memory state | Result |
|---|---|
| Full memory | 99.0% |
| No DRAM at all | **96.0%** |
| Reset every 256 frames | 91.3% |
| Reset every 128 / 32 / 8 frames | ~76% / ~75% / **~73%** |

At short reset intervals the policy falls **below** a stateless baseline. A half-alive memory is a liability. This mandates a hard fallback path (§14, feature ④).

**Interference.** RoboMME-Interference (`arXiv:2606.22338`) inserts `k ∈ {0,1,3,7}` unrelated sessions between memory and query. Perceptual memory decays steadily as distractors accumulate; symbolic/subgoal variants have a lower peak but hold up better. Adding a retrieval step *restores* the no-distractor success rate at every interference level — which makes retrieval a **mitigation for interference**, not an enhancement in itself.

MemTransfer (`arXiv:2609.32313`) adds the sobering coda: *"robustness to one kind of mismatch does not imply robustness to another."*

### 3.4 Memory that changes actions without choosing correctly

**CMA** (Counterfactual Memory Audit, `arXiv:2609.27247`) is the most methodologically important paper in the survey. Protocol: construct two task-consistent histories, update their native memories, reconverge to a **verified-identical** current input, query a frozen policy once per memory under common randomness, and evaluate each action in both worlds.

- **Later Swap:** all 24 pairs change their action, yet the selected branch **never changes** (p_correct = 0.5, F = 0)
- **Real dual-arm hardware:** memory demonstrably changed every saved action, and **5 of 9 completed Put-Back manipulations reached the wrong target**
- **Positive control:** swapping the history bank redirects behaviour 48/48 toward the donor target (official success 0/48) — memory *does* exert closed-loop influence
- **Recovery:** restoring one protected 4,096-byte component recovers **38.9 points** of lost success

> *"A robot can remember and react without reliably using memory to choose the behavior its past warrants."*

**Consequence for evaluation:** episode success is the wrong instrument. You need decision-level audits. CMA also yields the design principle that memory should be a **declared, protectable component** with source tags, expiry and provenance.

### 3.5 Mechanisms that do not pay

| Mechanism | Evidence | Number |
|---|---|---|
| Naive frame stacking into VLA context | HAMLET `arXiv:2510.00695` | LIBERO **95.6 → 86.8**; RoboCasa −3.3; SimplerEnv −4.2 |
| History-conditioned IL (causal confusion) | de Haan, Jayaraman, Levine, NeurIPS 2019 `arXiv:1905.11979` | better val loss, **much worse** driving (144.9 vs 269.0) |
| Long context without an auxiliary objective | PTP `arXiv:2505.09561`, CoRL 2025 | DP-Transformer and VQ-BeT **collapse**; worse than no history |
| Appending history to a strong VLA | RoboTTT `arXiv:2607.15275` | GR00T N1.7-Hist **39.5%** vs 57.0% no-history |
| Richer memory tokens without a selector | HAMLET Table 5c | "Moment Concat" **62.7** vs 62.6 baseline — worth nothing |
| Recording raw instruction history | MEM `arXiv:2603.03596` | naive concatenation **significantly worse** than a learned compressor |
| Keeping all frames | FrameSkip `arXiv:2605.13757` | 100% retention **47.80** vs 59.75 at 50% |
| More frames | RTV-Bench `arXiv:2505.02064` | "increasing frame density does not consistently improve results" |
| Retrieval as a quality lever | `arXiv:2610.05492` | no method dominates; **random is non-trivial**; recall@k uninformative |
| Pixel rollout in the control loop | V-JEPA 2 / Cosmos, same GPU | **240 s vs 16 s** per action |
| Partial-credit scoring for long-horizon | — | inflates capability; a memoryless policy earns credit by accident |
| LIBERO / SimplerEnv / CALVIN deltas as memory evidence | survey analysis | saturated — **>70%** of sim-evaluated papers cluster on LIBERO |

### 3.6 Mechanisms that do pay

| Mechanism | Evidence | Number |
|---|---|---|
| Action chunking, H = 16–50 | SmolVLA `arXiv:2506.01844` | LIBERO 50.0 → **84.0** (chunk 1→10); OpenVLA-OFT 76.5 → **97.1** with **26×** speedup |
| Slow-language / fast-motor split | SayCan → π₀.₅ → Helix → Gemini | the only long-horizon mechanism in production |
| Learned compression with success-gated writes | MEM `arXiv:2603.03596` | required for the 54 s horizon |
| Compression inside the vision encoder | MEM (zero new params, bit-identical at K=1); HAMLET | precondition for >4 frames |
| Attention-based **selection** over compressed memory | HAMLET | 65.4 vs 62.7 concat vs 62.6 none |
| Past-Token Prediction | PTP `arXiv:2505.09561` | 3× improvement, **>10×** faster training |
| Prefix KV-cache for flow-matching actions | π₀ `arXiv:2410.24164` | the enabler of 50 Hz |
| Retrieval as an interference mitigation | RoboMME-Interference | restores no-distractor rate at every *k* |
| Memory as **soft bias** on a feedback-driven policy | IOM `arXiv:2607.23702` | wrong memory costs ≈ 0 on ~88% of instances |
| Uniform frame sampling from a wide window | IntentVLA `arXiv:2605.14712` | **28.1** vs 10.4 same frame count |
| Frame retention at 20–50% in the dataloader | FrameSkip | 66.5 → **76.2** macro SR, zero architecture change |

---

## 4. Taxonomy: four memory lineages

The literature split into four families that **largely do not interoperate**. Conflating them is the single largest source of confusion in this area.

| Lineage | Substrate | Failure it fixes | Anchors |
|---|---|---|---|
| **A. In-policy recurrent / latent** | Fixed state vector, fast weights, KV cache | Latency and context growth of naive history | DRAM, ReCAT, μVLA, MemoryVLA, Fast-VLA |
| **B. Episodic / retrieval** | Vector store of trajectories, keyframes, text lines | Non-Markovianity *within* an episode | VIN, MCNN, RICL, STRAP, MemER, MemBodied |
| **C. Persistent scene record** | Scene graphs, ledgers, 3D Gaussians, keyframe banks | Object identity across occlusion, exit from FOV, sessions | ConceptGraphs, GaussMemory, Ledger, MessyMem, DAAAM |
| **D. Compressed language** | Text subtask / caption stream | Minutes-to-hours horizon at O(1) cost | MEM, GAMMA, CapMem, REMM |

**Ledger** (`arXiv:2609.34554`) states the cleanest version of the split:

> *"short-term perceptual memory (repetition, timing, retracing) belongs inside the policy, while long-term object memory (persistent spatial state, containment, event history) belongs outside as an explicit, readable record."*

They implement both over a single fine-tuned π₀.₅ and report 64.3% four-suite average on RoboMME vs 45.9% for the strongest prior under identical evaluation. **[TR]**, simulation-only.

### 4.1 The 3D substrate question is genuinely open

Three incompatible positions, no controlled three-way comparison exists:

| Position | Claim | Source |
|---|---|---|
| Explicit geometry wins | 3D Gaussian memory **94.1** vs 2D tokens **90.2** with *identical* attention machinery | GaussMemory `arXiv:2608.14986` |
| No reconstruction at all | Store posed keyframes only; **>100×** faster scene indexing | Memory over Maps `arXiv:2603.20530` |
| A tracker and a captioner suffice | SAM3 tracker + VLM captioner + LLM planner on π₀.₅ | Ledger `arXiv:2609.34554` |

GaussMemory's own caveat: it is **entirely simulation**, needs multi-view input (LIBERO's single default camera had to be supplemented with two *rendered virtual* viewpoints at ±30°), and cost 120 K steps on 4×A100. Its honest ablation shows the passive 3D memory already buys +8.0 of the +11.7.

**Real-robot counter-evidence for static maps** — MIF (`arXiv:2605.21935`, RSS 2026) on a Unitree G1 in a dynamic office: static scene-graph memory gives **12%** relocation success; discrepancy-triggered local updates give **94%**, with 91.4% footprint reduction. **Staleness, not representation capacity, is the dominant failure mode for humanoids.**

---

## 5. A recurrent state is not a belief state

This is the most tempting design and the evidence is against it.

### 5.1 The empirical calibration

**μVLA** (`arXiv:2606.12497`) is the one clean controlled study. Memory tokens carried across environment steps inside backbone self-attention, TBPTT, no auxiliary losses, frozen backbone:

| Split (MIKASA-Robo-VLA) | OpenVLA-OFT | +episodic loader | **μVLA** | 1st-obs oracle |
|---|---|---|---|---|
| 5 training tasks | 0.42 | 0.48 | **0.84** | 0.85 |
| 11 held-out, *matched* memory semantics | 0.01 | 0.07 | **0.23** | 0.24 |
| 7 held-out, **novel** memory semantics | 0.00 | 0.07 | **0.16** | 0.19 |

Recurrence buys ~2× on semantics it has seen. On **novel** semantics it barely beats memoryless. The oracle column matters as much: first-frame observation is not the answer either.

The state also fails to **scale with the state space**. On RememberColor5-PhaseNN (N colours per phase, trained on N ∈ {1..5}), recall collapses ~75% by N = 20 — exactly what a compressed sufficient statistic does and exactly what a true posterior would not.

### 5.2 The capacity argument

**DIACRITIC** (`arXiv:2609.25757`) characterises the minimum code rate required:

> *"Across manipulation tasks, learned code rates remain near zero- and two-bit requirements as hidden modes grow to 512."*

A 2-bit code cannot be `P(s_t | o_{1:t})` for a high-dimensional state. It is a **behavioral quotient** — sufficient for *acting like* the expert, strictly weaker than knowing where you are. The paper also finds **larger memory load reduces reliability** (negative slopes across 16-seed sweeps), while full-history attention reaches 0.88–1.00 closed-loop success — so the gap is **representational, not fundamental**.

### 5.3 The update rule determines the *kind* of memory

**ReCAT** (`arXiv:2609.35200`) holds robot, task and model fixed and swaps only the recurrence algebra:

| Rule | Update | Semantics | counting | spatial recall |
|---|---|---|---|---|
| Mamba-2 | `diag(a_t)h_{t−1} + v_tk_tᵀ` | **accumulate** | **85%** | 20% |
| Mamba-3 | `diag(a_t e^{iθ_t})h_{t−1} + v_tk_tᵀ` | accumulate + **rotate** | **90%** | 100% |
| Gated DeltaNet-2 | `α_t(I − β_tk_tk_tᵀ)h_{t−1} + β_tv_tk_tᵀ` | **replace** under matching key | 20% | **40%** |

The ordering **inverts by task**. "Use Mamba because it's state of the art" is the wrong instinct — counting wants accumulation, recall wants key-addressed replacement.

### 5.4 What does validate the belief-state framing

For balance, the positive evidence:

- **Direct probing in toy POMDPs** recovers Bayesian belief from RNN hidden states (`arXiv:2208.03520`): mutual information with control-*relevant* variables **increases** during learning, with control-*irrelevant* variables **decreases**. This is the classical Kalman intuition, confirmed.
- **Reset ablation** confirms the state is causally necessary, not decorative: SmoLSTM loses 77.5% → 7.0% when reset every step; the air-hockey study (`arXiv:2609.39151`) shows the same.
- **μVLA's interventions** distinguish two qualitatively different contents in one state: `freeze_first` preserves an early cue (0.36) but collapses interception (0.07), while `noise` collapses both (0.09, 0.21).
- **Readout routing behaves like a belief-adequacy test** — see §5.5.

### 5.5 Readout strength is not a tunable you can set conservatively

ReCAT varies how `m_t` enters the flow-matching decoder:

| Integration | Real-robot success |
|---|---|
| cross-attention in **every** block | **66.7%** |
| late fusion (last 2 blocks) | 38.3% |
| AdaLN modulation | 31.7% |
| learned scale | 28.3% |
| memory dropout (training-time) | 20.0% |
| content gate | **8.3%** |

A weak gate is **worse than useless**. Memory must be read hard, from every layer. Also note the frame encoder matters as much as the memory: a pure-Transformer encoder drops the counting task 85% → 0%.

### 5.6 Chunk length × recurrence is unresolved

μVLA abandoned chunked deployment (H=8 → receding horizon, execute 1) to align memory cadence, at a cost of **20.9 Hz → 9.2 Hz** closed-loop throughput. ReCAT kept chunking but advances memory at *every control step including during chunk execution*. ChainVLA carries the unexecuted suffix. **No head-to-head comparison exists.**

### 5.7 Is chunking even memory?

A live disagreement, preserved rather than resolved:

**For long open-loop:** OpenVLA-OFT, ACT, π₀, SmolVLA — chunk 1→10 is +34 points; 26–43× speedup.

**Against:** **Zeng, Agarwal, Bati, Lee, Ancha, Tedrake** (`arXiv:2608.15938`, MIT/UCB):

> *"long open-loop execution primarily helps short-context policies imitate 'non-Markovian demonstrations'... when policies are provided with a sufficiently long context, open-loop execution is no longer beneficial and the most reactive, closed-loop policies perform best."*

They show optimal execution horizon `T*_exec = 1` for policies trained on **Markovian** expert data, and argue the standard Tool-Hang result is likely an artifact of small data plus a ~70%-success filtered expert. Convergent evidence from Lazzati et al. (`arXiv:2608.02547`): chunking's benefit is non-Markovian expressivity + variance reduction, fully capturable by delayed policies.

**Likely both true**, measuring different regimes (small-data IL vs large-data settings). But note PTP's independent finding, which *contradicts* Zeng: diffusion policies **underuse** past actions (predictability ratios below 1) despite long histories — the opposite failure from "copycat." The honest position is that **chunking is credit assignment, not memory, and its value depends on the data regime.**

---

## 6. Memory inside current VLAs

### 6.1 The baseline is memoryless

**OpenVLA, π₀, π₀.₅, GR00T N1/N1.5, SmolVLA, CogACT and Helix all consume a single frame.** Three independent sources state it:

> *"While many of today's state-of-the-art VLAs are trained without memory and act purely based on the current observation of the environment…"* — MEM, `arXiv:2603.03596`

> *"They typically adopt a single-frame assumption, predicting each action solely from the current observation."* — HAMLET, ICLR 2026

> *"most VLAs encode only the current image and ignore prior timesteps, which is a major limitation for many tasks."* — Reuss, *State of VLA Research at ICLR 2026* **[B]**

**Two corrections to widely-repeated facts:**

- **RT-1 used 6 frames, not 15.** The RT-1 paper tokenises 6 images × 8 tokens = 48 tokens. The 15-frame figure belongs to **RT-1-X**.
- **π₀.₅ has no visual memory at all.** Its `ℓ̂` (predicted subtask) is *re-derived from the current frame every step* — a compressed near-Markovian state. Its long-horizon asset is the action chunk (H=50 ≈ 1.0 s at 50 Hz), which is credit assignment, not memory.

### 6.2 Concrete configurations

| System | Frames | Vision tokens | Chunk H | Params | Venue |
|---|---|---|---|---|---|
| RT-1 | **6** | 48 | 1 step | 35M | RSS 2023 **[P]** |
| RT-1-X | 15 | ~1,215 | 1 | 35M | ICRA 2024 **[P]** |
| OpenVLA | **1** | 256 | **none** | 7B | CoRL 2024 **[P]** |
| Octo | 2 | ~320 | 4 | 93M | RSS 2024 **[P]** |
| π₀ | **1** | 256 × 2–3 views | **50** | 3.3B | RSS 2025 **[P]** |
| π₀.₅ | **1** | 4 cams (HL) | 50 | 2B + 300M | **[TR]** |
| GR00T N1 | **1** | 64 × n_views | 16 | 2.2B | **[TR]** |
| SmolVLA | **1** | **64** (PixelShuffle from 512²) | 50 | 450M | **[TR]** |
| Helix | **1** (S2→latent→S1) | 1 latent vector | continuous | 7B + 80M | **[B]** |
| π₀.₆-MEM | **18 @ 1 s (54 s)** | ≈ single-frame count | 50 | +860M expert | **[TR]** |
| π₀.₇ | ≤6 @ 1 s + ≤3 subgoal imgs | ≤4 cams | 50 (exec 15/25) | 5B | **[TR]** |

**Headline:** the state of the art for visual memory in a deployed VLA is **54 seconds**. Everything beyond is handled in compressed natural language.

### 6.3 The wall is the vision encoder, not the LLM

**No published work has meaningfully extended a VLA's language-model context to hundreds of frames.** Nearly every "long context for VLAs" claim is actually describing a *video encoder*.

MEM's design is the elegant one: causal temporal attention every 4th ViT layer, **O(Kn² + nK²)** instead of `O(n²K²)`, past-frame tokens dropped in upper layers, **zero new learnable parameters**, and a fixed sinusoidal temporal encoding that is exactly 0 at t=0 — so a pretrained ViT initialises **bit-identically** to single-image behaviour.

HAMLET's cost measurement on GR00T N1.5 (A100, per-step, excluding base params):

| Method | History | Latency | Peak mem |
|---|---|---|---|
| baseline | 1 | 80.5 ms (1.00×) | 289 MB (1.00×) |
| + naive multi-frame | 8 | 193.0 ms (**2.40×**) | 2023 MB (**7.00×**) |
| + HAMLET | 8 | 85.8 ms (**1.07×**) | 578 MB (2.00×) |

Compression buys back a 7× memory blowup to 2×, and a 2.4× latency hit to 1.07×. Without it, >4 frames is not affordable.

### 6.4 Compression without consolidation buys nothing

HAMLET Table 5c, RoboCasa-100:

| Memory treatment | Success |
|---|---|
| no memory | 62.6 |
| moment tokens only | 63.1 |
| + time-contrastive init | 63.4 |
| + memory module (RNN / LSTM / GRU) | 64.3–65.0 |
| **"Moment Concat"** (compress then just append) | **62.7** |
| **attention-based Transformer consolidation** | **65.4** |

Naive concatenation is worth **+0.1 points**. *"Make the memory richer"* is not an architecture; you need a **learned selection mechanism over** compressed memory.

### 6.5 MEM's write-gating result

π₀.₆-MEM's language memory is a summary the high-level policy **rewrites each step**, trained on LLM-generated summaries. The LLM is explicitly told to compress:

> *"instead of remembering the precise attributes of all objects that were manipulated ('I put a light green bowl, a dark blue bowl and a bright yellow bowl into the top right cabinet'), it is often sufficient to just remember where the bowls were placed."*

The ablation is the important part — a **naive** memory that just concatenates all prior subtasks is *"significantly worse"*, because:

> *"During training, most episodes utter any given subtask instruction only once... at inference time policies may repeatedly fail on a given subtask ('pick up bowl' → 'pick up bowl' → 'pick up bowl'), leading to a **distribution shift** that can degrade overall policy performance. In contrast, MEM's language memory would simply not update until the bowl was successfully picked up."*

> **This is the clearest published statement of what a memory write policy must do:** writes gated on *verified* semantic progress, and **idempotent under failure**.

MEM also reports that pretraining the video encoder matters enormously — introducing it only at post-training is *"noticeably worse at leveraging the information from past timesteps."* Expanding the horizon 5 s → 54 s only works when the encoder saw memory during pretraining.

### 6.6 Does history actually help? The honest split

**Where it clearly helps:**

| Result | Number |
|---|---|
| Open X-Embodiment Table II, RT-2-X 5B: history 2 vs none | emergent skills **14.5% → 44.4%** (3.06×); generalisation 30% → 52% |
| HAMLET real-world non-Markovian tasks (24 trials/task) | π₀ 31.9 → naive multi-frame 45.8 → **HAMLET 76.4** |
| MEM: π₀.₆ without memory on memory tasks | **chance** — 25% on "which drawer holds the object" |

**Where it is neutral or harmful:**

| Result | Number |
|---|---|
| HAMLET, naive multi-frame on standard benchmarks | LIBERO 95.6 → **86.8**; RoboCasa 62.6 → 59.3 |
| CronusVLA, 6 past frames | +1.4 points while **halving** the control rate |
| Diffusion Policy Policy `arXiv:2410.15959` | drops at history = 3; 2 frames best |
| RoboTTT on Pup Go Car | GR00T-Hist **39.5%** vs **57.0%** no-history |

**Verdict.** History is the dominant variable when the task is genuinely non-Markovian, and **two frames already capture most of the available gain**. But the gain concentrates almost entirely on tasks someone deliberately constructed to be non-Markovian; on general manipulation it is neutral-to-negative with naive implementations. Every successful mechanism is **selective** — an attention-based consolidation module, a learned compressor, an auxiliary prediction target. Every "more tokens in the context" approach fails.

### 6.7 The most robust long-horizon mechanism is not a memory module

It is the **slow-language / fast-motor split**, which the field keeps mislabelling as memory. SayCan (up to 50 stages), π₀.₅'s HL ablation (removing high-level labels drops Items-in-Drawer and Dishes-in-Sink substantially), Helix's S2→S1 (7B @ 7–9 Hz → 80M @ 200 Hz), GR00T's S2 (which uses **middle-layer** LLM embeddings, layer 12 — reported to beat final-layer on both speed and success), Gemini Robotics 1.5's thinking traces appended to the context window.

Four independent reasons it survives: a semantic event is a few bits versus thousands for a pixel; it reuses the VLM's native token space; text memory is `O(L)` tokens not `O(L × tokens/frame)`; and **it is inspectable** — π₀.₅, π₀.₇ and Gemini all print their subtask. The systems that show you their memory are the ones that work.

**Honest correction to the S1/S2 narrative:** GR00T N1's paper states S1 and S2 are *"tightly coupled and jointly trained end-to-end"* in one forward pass — that is not a dual system. Helix's are genuinely separate networks on separate GPUs at different rates. On current evidence the "System 1 / System 2" framing is **largely a description of latency budgeting**.

---

## 7. Episodic, retrieval and scene memory

### 7.1 The deflationary result

`arXiv:2610.05492` compared four retrieval methods inside RICL — image-only, image+state, backbone features, and **random**:

1. **No method consistently dominates** on task success
2. **Random retrieval achieves a non-trivial success rate**
3. **Recall@k does not reliably reflect downstream VLA performance**
4. Demonstrations from *different but related* tasks provide useful transferable information

This is the most important single paper for anyone planning a RAG-style robot memory. The retrieval module may not be where the value is, and the standard offline metric will not tell you. **Open problem: the right retrieval key is probably the highest-leverage unsolved question in the field.**

### 7.2 What retrieval does achieve

**RICL** (`arXiv:2508.02062`, CoRL 2025) — the canonical schema. π₀-FAST-DROID base; 400 priming demos; only the LLM finetuned, image encoder frozen. Per timestep, store (3 images, prompt, state, action chunk); embed **only the top image** with frozen DINOv2; retrieve 4 neighbours by ℓ2. The fusion is the interesting part — a **differentiable soft-RAG over action tokens**:

$$\pi_\theta(\text{retrieved}, \text{query}) = e^{-\lambda d}\cdot\mathrm{onehot}(a') + (1-e^{-\lambda d})\cdot\mathrm{softmax}(\pi_\theta(\text{retrieved},\text{query}))$$

Hard retrieval is `d→0`; pure parametric is `d→∞`.

Real robot, 8 tasks × 10 rollouts: base **2.5%** → RICL **31.25%** (no gradient updates) → RICL-finetuned **61.67%**. Needs ≥10 demos. **Author-stated limits:** does not enable tasks far beyond the base VLA; struggles with novel motions; still needs some teleoperated demos.

The rest of the lineage — **VIN** (`arXiv:2112.01511`, RSS 2022, non-parametric locally weighted regression, no parametric action head), **MCNN** (`arXiv:2310.06171`, ICLR 2024, codebook + geometric constraint giving a *guaranteed* suboptimality bound), **STRAP** (ICLR 2025, DTW over kinematics, retrieval *into training* not at test time), **RECAP** (`arXiv:2606.15631`, frozen policy absorbs new tasks by appending pool-side demos), **MemER** (`arXiv:2510.20328`, high-level policy emits both a subtask *and* a candidate keyframe; a filter commits it) — share one theme: **retrieval is a filter, not a compressor.**

### 7.3 The robot-specific advantage

This is where robotics genuinely diverges from LLM RAG, and it is not rhetorical.

| | LLM RAG | Robot memory |
|---|---|---|
| Ground-truth check on a retrieved chunk | Not available | **Available by acting, re-looking, re-deriving** |
| Cost of a bad retrieval | A wrong sentence | A wrong grasp, a broken object, a collision |
| Recovery from bad memory | Needs an external signal | **Built in** — closed-loop feedback |
| Negative evidence | Rarely modelled | First-class (OmniNav, AECNav) |
| Uncertainty calibration of retrieved items | Open problem | Beginning to be solved (UQ-DAAAM) |

The strongest demonstration is **IOM** (`arXiv:2607.23702`, "Try Once, Then Optimal"):

> *"Because the procedure is a soft bias on a feedback-driven policy, an incorrect memory is recovered from rather than obeyed: success holds even when a retrieved procedure is wrong, as for ~12% of door instances."*

A wrong memory costs nothing on ~88% of instances **because the policy is still closed-loop**. You cannot buy that property in an LLM agent. Architecturally: **memory biases a feedback-driven policy, it never replaces it.**

### 7.4 Verification and truth maintenance

Six distinct strategies, all 2026:

- **Verify-on-read** — GAMMA (`arXiv:2610.12793`) admits/deferes/corrects/rejects every agent claim against mechanical visual evidence, with a writer and reasoner separated so *"hallucination corrupts the memory itself"* is recoverable. UQ-DAAAM (`arXiv:2606.08277`, Gorlo/Carlone) scores object-level **semantic uncertainty** by cross-view caption scatter and actively re-observes under a fixed budget. MemoGuard (`arXiv:2607.15589`) attaches **execution contracts** — −76.6% battery violations, −21.4% fallback calls.
- **Verify-and-revise** — REMM (`arXiv:2610.12710`) separates append-only evidence from a revisable projection with **explicit support links**, withdraws dependent conclusions on new evidence, and schedules bounded repair at the next verified stationary skill boundary. Numbers: intervention recovery 57% → 91%, sequence recall 43% → 70%. **[TR]**, 51 physical trials including operator-assisted runs.
- **Negative evidence** — OmniNav (`arXiv:2609.08159`), AECNav (`arXiv:2608.10817`).
- **Write-time geometry checks** — GlassGuard (`arXiv:2610.02110`) verifies plane orientation before glass-plane hypotheses enter a map, with 5–17× fewer false voxels.
- **Surprise-gated formation** — *Worth Remembering* (`arXiv:2606.03787`, Gorlo/Carlone) uses Bayesian surprise in a V-JEPA 2 latent space, unsupervised and causal, ≥12% over prior robot memory methods.
- **Defer** — RefGuard (`arXiv:2609.06221`) executes / clarifies / reobserves / **aborts**.

**No shared formalism, no shared benchmark.** CMA is the closest thing to a scoring protocol. Verification cost is almost never reported alongside accuracy — MemoGuard is the exception (76.6% / 21.4% / **3.67 s / 36.97 J** per trial) and should be the template.

### 7.5 Persistent object identity is the unsolved core

Three independent 2026 systems converge here — the strongest signal in this slice:

- **RefGuard** baselines suffer identity switches on **33–46%** of ambiguity-stress trials on a real UF850 arm; RefGuard records **zero** while deferring *less* often (19.5% vs 43.4%)
- **CMA** — 5/9 wrong-target on hardware
- **Dual-Stream Memory** (`arXiv:2610.13262`) — *"failures under cup swaps expose persistent identity tracking as an unresolved requirement"*

And perception will not save you. **MG-VQA** (`arXiv:2608.17129`): 8 VLMs score 36.2% with images only and 37.6% with pointing, segmentation **and scene graphs**, against 32.7% blind chance. Manipulation raises this to 56.8%. **Scene graphs as a perception modality are near-worthless; only physically moving things produces the evidence.**

---

## 8. Video as a memory substrate

### 8.1 Five architecturally distinct uses

| # | Role | Latency | On the control loop? |
|---|---|---|---|
| **A** | Forward rollout buffer (world models) | 0.16–240 s/action | **No** |
| **B** | Frozen video backbone → action head | 30–70 ms | Yes |
| **C** | Video-as-history for the policy | 1 frame → 8K steps | Partially |
| **D** | Episodic record you *query* | 0.1–2 s | Off-loop |
| **E** | Video-LLM reasoning over hours | seconds | Off-loop |

The tension: **everything on the loop must destroy most of the video; everything that keeps video is off the loop.** Nobody has demonstrated a robot holding minutes of visual memory *and* closing a 50 Hz loop on it.

### 8.2 The cost wall, measured on identical hardware

Same paper, same RTX 4090:

| | Per action |
|---|---|
| **V-JEPA 2-AC** (latent prediction) | **16 s** |
| **Cosmos** (pixel prediction) | **240 s** |

**15×** — the difference between "plausible for control" and "not." Latent prediction is non-negotiable for anything on the loop.

Full measured control-loop budget:

| System | Hardware | Latency | Rate |
|---|---|---|---|
| **FlashVLA** | RTX 4090 / 5090 | **26.7 ms** action decode | **≥30 Hz**, 50 Hz @ 5090 |
| **GlanceWAM** | A100 | **48 ms** / 16-action chunk, no VAE decode | 24× lower than sync WAMs |
| Cosmos Policy | H100 | 0.61 s @ 5 steps / **0.16 s @ 1 step** | chunk = 16 steps |
| CronusVLA | — | — | **8.73 Hz** at 70.9% |
| InternVLA-A1 | RTX 4090 | — | ~13 Hz |
| VideoVLA | H100 | ~1.1 s | ~3 Hz |
| V-JEPA 2-AC (CEM) | RTX 4090 | 16 s | 0.06 Hz |
| Cosmos MPC | RTX 4090 | 240 s | 0.004 Hz |

**Rule: anything above ~200 ms cannot run a 50 Hz loop.**

### 8.3 Latent world models have a hard, characterised limit

`arXiv:2609.39235` — required reading:

- With 5-step rollouts (the training length), the model ranks actions reliably only **5–10 control steps** ahead. Task goals are **16–53 steps** away.
- **An 81× larger predictor does not extend the range** — it improves prediction loss, not plannable range.
- **With the real simulator substituted in**, success still collapses **92% → 41%** as the target moves beyond range. Not a capacity problem.
- Inside its range it does help: choosing among 8 VLA-proposed actions raises success **65% → 77%** across 16 tasks.

And the follow-up nobody expected (`arXiv:2608.12959`): swapping **only the distance objective** moved success **26.0% → 98.0%**, because the planner's squared-latent-distance cost correlates with true distance at only **r = 0.426** and *decreases* past ~120 units — **moving away from the goal can lower the cost**. A meaningful fraction of published latent-WM results may be objective artifacts. **Evaluate how far a world model can rank actions, not how well it predicts.**

### 8.4 Temporal abstraction: the numbers that decide everything

**Token exchange rates** (tokens/frame):

| Encoder | Tokens |
|---|---|
| SmolVLA PixelShuffle @512² | **64** |
| Gemini default / PaliGemma-ish | 256 |
| CLIP ViT-L/14 @224 | 256 |
| Cosmos CI8×8 VAE @256² | 1,024 |
| SigLIP-so400m-patch14 @384 | 729 |
| Claude 720p, 28-px patches | 1,196 |

**Your arithmetic** — 10 Hz, 60 s, 729 tok/frame = 437,400 tokens ≈ **53 GiB** of KV cache for an 8B-class model at 128 KiB/token. That single number explains why every deployed VLA uses 1–3 frames.

**More frames ≠ more memory** (fourth independent literature to say so):

| Finding | Number |
|---|---|
| FrameSkip retention sweep | **100% is the worst** (47.80) vs 59.75 at 50% |
| FrameSkip at 20% | 66.50 → **76.15** macro SR, zero architecture change |
| CronusVLA frame sweep | 7B best at **7 total frames**; 0.5B at 4 |
| StreamingTOM | **0.5 fps matches 32 frames** (56.3 vs 56.4) |
| SimpleStream | **4 frames match 13 published streaming baselines** |
| RTV-Bench | *"increasing frame density does not consistently improve results"* |

**And the sampler matters more than the count** — IntentVLA / AliasBench:

| Sampler (same budget) | Success |
|---|---|
| no history | 9.0 |
| last 4 **contiguous** | 10.4 |
| last 8 / 16 contiguous | **OOM** |
| **4 uniformly sampled from last 16** | **28.1** |

### 8.5 Why adding frames often fails — the mechanism

**GAP** (ICLR 2026) is the mechanistic explanation. During motion-transition sub-phases the vision modality *"plays a limited role"* and the policy *"naturally gravitates toward concise proprioceptive signals that offer faster loss reduction, thereby dominating the optimization and **suppressing the learning of the visual modality**."*

You are not adding information. You are adding a nuisance the optimiser routes around.

### 8.6 The one clean win

**RoboTTT** (`arXiv:2607.15275`, NVIDIA GEAR / Stanford / UT-Austin) — test-time-training fast weights inside GR00T N1.7. 8 K timesteps ≈ **5 minutes** at 30 Hz, *"without growing inference latency."* 79% task completion vs 42% single-step and 56% best baseline. Scaling 128 → 8 K gives 43.9% → **71.5%**, no saturation.

Its ablations are the point:

- **Gated DeltaNet** — the linear-complexity recurrent memory — **does not benefit from longer context at all** in the same setup. The mechanism matters, not the context.
- TTT-**linear** is **27% worse** than TTT-MLP
- Action tokens in the TTT layer: **+23%** relative
- Register tokens: **+18%** — and **do not help without TTT**

**Skepticism required:** an independent write-up notes the 79% is a *rubric partial-completion score*; strict full success was **24/50**, with 2/10 on the hardest task. 538 M → 690 M params, 30 K steps on 16 GB200, and no code/weights released at review.

### 8.7 Video pretraining as a representation prior

Strong but bounded. Video supplies **observations, not actions**: *"video shows what happens… but it does not record the motor commands that produced those motions."* Every successful system inserts an explicit embodiment bridge — IK retargeting (EgoVLA), hand-pose regression (EgoDex, **829 h** / 338 K demos), latent motion tokens (Moto), latent-frame co-denoising (Cosmos Policy). Sample efficiency is real: mimic-video reports **10×** over a π₀.₅-style decoder.

**X-Sim** (`arXiv:2505.07096`) is the interesting counterfactual for deployment: image-based policies beat **privileged-state** pose-based baselines sim-to-real (Letter Arrange 83.3% vs 43.3% task progress). A genuine case for vision over ground-truth state.

### 8.8 Efficient video memory mechanisms (with real configs)

- **LongVU** (ICML 2025) — DINOv2 similarity over 8-frame windows drops **45.9%** of frames; spatial token compression with sliding window K=8, cosine θ=0.8 removes **40.4%** of tokens; net ~**2 tokens/frame average**, fitting an hour into 8 K context.
- **Video-XL** (CVPR 2025) — Visual Summarization Tokens every αᵢ tokens in chunks of w=1440; **1024 frames on one 80 GB GPU** at ~100% needle-in-haystack, **8× compression**. Quantifies the field: LongVA tops out ~400 frames/80 GB.
- **StreamingVLM** (ICLR 2026) — sliding window + KV reuse; **66.18% win rate vs GPT-4o-mini** on videos averaging >2 hours; stable real-time to 8 FPS on one H100.
- **StreamingTOM** (CVPR 2026) — streaming attention inside SigLIP; **KV bounded at ~1.2 GB per hour of video**; sub-0.4 s TTFT.
- **STC / ReKV** (CVPR 2026) — caches reference-frame tokens, computes only *dynamic* tokens; **24.5% ViT latency and 45.3% prefill reduction**. Diagnostic worth remembering: *"the repeated ViT inference dominates overall latency before tokens even reach the LLM."*
- **CapMem** (EMNLP 2026) — see §9. The cheapest memory in the survey.

---

## 9. Video-based answering

Three findings, cleanly separated.

### 9.1 It works for bounded predicates — and is deployed

**SuccessVQA** (CoRL 2023) reformulated success detection as VQA so a pretrained Flamingo could do it zero/few-shot with better OOD robustness than task-specific detectors.

**AutoEval** (`arXiv:2503.24278`, CoRL 2025, Berkeley) is the production exemplar: a fine-tuned VLM success classifier trained on **~1,000 images collected in <10 minutes** of teleoperation, binary *"is the drawer open?"*, with a **>95% accuracy deployment threshold**, plus an auto-reset policy and programmatic safety detection. Result: **60,000 evaluation steps/day per cell (~850 episodes)**, **>99% reduction in human evaluator time**, stable at 2-month retest (96%/96%). The paper's honest caveat: it does *not* perfectly match human evaluation.

**Reward models from video** — Robo-Dopamine (CVPR 2026) reaches **VOC 0.94–0.96** across seven datasets from 35 M samples / 3,400 h, and adapts a new task to 95% from a single expert trajectory plus ~1 hour of real rollouts.

### 9.2 It fails for self-reference — measured

**MyEgo** (CVPR 2026) is the sharpest available result: 541 long videos, 5 K personalized questions about *"my things / my activities / my past."*

> **GPT-5 ≈ 46%, Qwen3-VL ≈ 36%** — ~40–50 points below humans. Gains **decay over time**. Neither explicit reasoning nor scale improves this consistently.

This is a direct measurement that current video LLMs **lack self-referential episodic memory over time** — precisely the capability a robot needs to answer *"where am I in this task?"*

**SimpleStream** (`arXiv:2604.02317`) names the structural reason: a **perception–memory trade-off** — *"adding more historical context can improve recall, but often weakens real-time perception."* And: *"the value of longer context is backbone-dependent rather than uniformly increasing with model scale."*

**MyEgo's complement**, Grounded QA in Long Egocentric Videos (CVPR 2024), opens with the right framing: *"allowing individuals **or robots** to inquire about their own past visual experiences."* Its EgoTimeQA ships 5,389 videos / 303 K samples with *"where did I put lettuce?"* → (B) refrigerator.

### 9.3 Text memory beats re-encoded video

**CapMem** (`arXiv:2609.17688`, EMNLP 2026), 75 videos / 33.7 h: on videos **>20 min**, caption-window QA with 30 s and 60 s windows **beats direct video QA for 10/12 and 8/12 models**. A matched-frame control across six Qwen models retains +3.22/+2.55; a retrieve-and-verify harness adds up to +5.3.

**LangRepo** (ACL Findings 2025) does the same with zero training: fully-textual multi-scale chunk descriptions, readable by humans.

This corroborates §6.5 from an entirely separate literature. **Compressed language is the cheapest memory in the survey, and it wins on long horizons in both.**

### 9.4 The recipe

```
TIER 3 — EPISODIC RECORD (off-loop, 0.1–2 s/query, hours)
  two stores, queried not pre-filled:
    · CAPTION MEMORY — 30 s / 60 s text windows + timestamps
    · KEYFRAME INDEX — uniform + change-point keyframes, ~8/min
  query path: text retrieval → 8–96 frames → short VLM pass
  answers: "what am I doing / did that succeed / why did it fail"
```

**Hard rule:** if `frames × tokens_per_frame × 128 KiB > 8 GiB`, compress, subsample, or switch to fast weights. **Do not let it silently truncate** — LongVU's Fig. 1 shows uniform dense sampling overshooting context *truncating exactly the targeted frames*.

---

## 10. The robot advantage: verification by acting

This is the conceptual centre of the survey, and it is what makes robot memory a different problem from agent memory.

**Negative evidence is first-class.** OmniNav and AECNav treat unsuccessful search as evidence. `Worth Remembering` uses Bayesian surprise to decide what to store — because future tasks are not known in advance, so selection must be generic.

**Memory as verifier rather than conditioner is almost unexplored.** PTP's test-time self-verification is essentially the only published example of scoring candidate actions against remembered past. This is the most under-explored direction in the field.

**Memory as a safety surface.** CMA shows a protected 4 KB component restores 38.9 points of lost success. `Present but Not Remembered` explicitly flags the dual use: *"the same diagnosis that guides safer memory design could inform an adversary constructing occlusion conditions under which a deployed policy silently falls back on stale history."* AURA (`arXiv:2606.02775`) adds a hardware-economic dimension — flash **program/erase endurance** makes write rate an architectural parameter, and memory state can be held **constant at 4,224 bytes** with zero variance over 80 K steps.

**No demonstrated physical memory-poisoning attack exists.** The channels are obvious; the empirical work is on LLM agents. Given that robot memory writes are increasingly LLM-produced, this is a near-term threat with no defences in the field beyond provenance and CMA's protectable-component idea.

---

## 11. What industry actually ships

**Labelling is critical here.** Most "company architecture" detail circulating online is aggregator-site or AI-generated content. Documented, inferred and rumoured are kept strictly separate below.

### 11.1 Latency is decided by what is *not* on the fast path

The best-documented measurement, **MemER** (`arXiv:2510.20328`, ICLR 2026):

| Configuration | Inference time | VRAM |
|---|---|---|
| π₀.₅ (no memory) | **0.088 s** | 6.25 GB |
| + 1 recent frame | 0.532 s | 15.55 GB |
| + 8 recent frames | 0.591 s | — |
| **+ 8 recent + 8 keyframes (MemER)** | **0.787 s** | 15.93 GB |

Read carefully: **π₀.₅ alone is 88 ms and 6.25 GB. The moment a 7B VLM enters the loop it goes to ~530 ms and ~15.5 GB.** Most of the cost is the VLM, not the memory frames. Going 1 → 8 → 8+8 frames costs 0.532 → 0.591 → 0.787 s. **Memory tokens are cheap relative to the model that reads them.** MemER needs two 4090s.

> The deployable architecture on current hardware is **one VLM call per ~0.5–0.8 s over a compact keyframe set**, not per-frame recurrence. That yields **~1–2 Hz semantic memory updates over a 200 Hz fast controller** — a real two-timescale system arrived at by engineering rather than neuroscience.

### 11.2 Deployability table

| Memory tier | Deployable on-robot? | Evidence |
|---|---|---|
| Control reflex / balance net | ✅ 2–3 ms | Tesla **[D]** |
| Fast action expert 100–200 Hz | ✅ | Helix S1 200 Hz; GR00T 120 Hz |
| Short horizon (~8 frames) | ✅ | MemER 0.591 s @ ~2 Hz |
| Keyframe episodic (≤ tens of frames) | ✅ ~1 Hz | MemER 15.93 GB; Figure vision memory |
| Long-horizon semantic (minutes) | ⚠️ **language-summarised only** | MEM; RoboMME subgoal variants |
| Spatial map / 4D scene graph | ⚠️ real-time only with aggressive batching | DAAAM ~10× batch speedup |
| **Online consolidation at deployment** | ❌ **nowhere** | Sleeping Robots is a paper; 1X lists this as *future work* |
| **Cross-session memory over days** | ❌ **nowhere** | RoboMME-Interference: even single-session interference is unsolved |

### 11.3 Systems

| System | What is documented |
|---|---|
| **π₀ / π₀.₅ / π₀.₆-MEM / π₀.₇** | Documented: 54 s visual memory (MEM), compressed language memory, prefix KV-cache, RTC training with 0–12 timestep simulated delay. **[P]** for π₀/π₀.₅, **[TR]** beyond |
| **GR00T N1 / N1.5 / N1.6 / N1.7** | Documented: single-frame, 64 tokens/view, chunk 16→40, S1 @120 Hz, middle-layer (L12) embeddings into the DiT head. N1.7 is Apache-2.0. **[TR]** |
| **Helix** | Documented: fully onboard on dual embedded GPUs, S2 and S1 on separate GPUs, 7B @ 7–9 Hz → 80M @ 200 Hz, gradients flow back into S2. **Latent vector dimension, S1 architecture, and all latency figures are UNDISCLOSED.** Helix 02 adds palm cameras and tactile. **[B]** — no arXiv, no ablations |
| **Gemini Robotics On-Device** | Model card states on-device Gemma. **No parameter count, no latency, no memory-window details.** Acknowledged weak on high-DoF robots. **[U]** |
| **Gemini Robotics 1.5** | Thinking traces appended to the VLA context; GR-ER orchestrator with explicit success detection. **Frame history, chunk horizon and model size all undisclosed.** Motion Transfer acknowledged as necessary for embodiment gaps and **unreleased**. **[TR]** |
| **TinyVLA** | **[P]** RA-L 2025. Pythia-LM + LLaVA VLMs, **70M–1.4B**, diffusion head, LoRA ≈5% trainable. **≈20× lower inference latency than OpenVLA** while matching or exceeding it. The counterfactual: *tiny and memoryless* beats *large with keyframe memory* on latency |
| **SONIC** | 42M params, ONNX + C++ stack **including Jetson**. Sub-100M control networks onboard are routine |
| **Dyna Robotics** | *"In continuous deployment settings, robot data does not naturally come with episodic boundaries"* **[D]**. **This is the deepest structural problem and it is acknowledged by a company building the product** |
| **GR00T N2** | ⚠️ **Circulates only on `techbytes.app` and `embodiedglobal.com`. No primary NVIDIA source. Do not present as fact.** |
| Helix 2.5 / "System 0 @ 1 kHz" / "13 GB bf16" | ⚠️ **Aggregator sites and AI-generated summaries only.** Figure has published no latency figures |

### 11.4 Small models as the real counterfactual

The efficiency survey (`arXiv:2510.24795`) names three pillars and, notably, lists **pruning image tokens** — *discarding* memory — as a mainstream efficiency lever. **BitVLA** (ternary 1-bit), **Evo-1** (77M params), **SQAP-VLA** (~2× speedup, half memory), **Spec-VLA** (`arXiv:2507.22424`, speculative decoding with relaxed acceptance), **AURA** (constant 4,224-byte memory state).

The uncomfortable implication: much of what is attributed to "memory architecture" may be **representational or architectural rather than retrieval**.

---

## 12. Evaluation: the ladder

How to *prove* a system uses memory. Ordered by rigor. Rungs 0–3 are common; **6–8 are rare and are where the science lives.**

| Rung | Intervention | Published example |
|---|---|---|
| **0** | Benchmark with **reactive controls** (memory cue kept available) as negative controls | MIKASA-Robo-VLA's 10 reactive controls among 90 tasks — the gold standard |
| **1** | **Zero-memory control.** Report lift over **your own floor**, never another system's number | RoboMME-Interference **+27.1 pp** (18.2 → 45.3) |
| **2** | Component removal (anchor / sliding / keyframe buffers) | RMBench Mem-0; RoboMemArena PrediMem scaling curves |
| **3** | **Write suppression** — reads intact, writes disabled | T²Mem `arXiv:2609.36720`: **−82.67 pp** on VideoUnmask, −64.67 SwingXtimes, −41.33 MoveCube |
| **4** | **Donor content replacement** — swap in a conflicting memory | T²Mem; MEMOBench shuffled-subgoal: **90% → 24% → 2%** |
| **5** | **Distractor injection** `k ∈ {0,1,3,7}` | RoboMME-Interference — the most rigorous interference design published |
| **6** | Anchor manipulation | *Memory Anchors* `arXiv:2608.26545`: excluding **10%** of anchors → **>4.5×** forgetting; enriching → **−63%** high-conflict forgetting |
| **7** | **Interchange intervention** (layer-resolved causal) | `Present but Not Remembered`; VLA-Trace `arXiv:2605.30117` attention knockout |
| **8** | **Uniqueness test** — is the memory *new* information? | **A₄ ≈ 0** across 81 configs × 3 models × 2 families |
| **9** | De-confounding | Executable simulator predicates; prefix-completion scoring; **format/source confound controls**; **Wilson 95% CIs** |

**Rung 9 deserves emphasis.** TrustNLP 2026 (`arXiv:2606.02907`): an identical linear probe predicting *dataset source* achieves **100%**; after ridge-regressing out format features, reasoning-mode probe accuracy **drops to chance (33.3%)**. Any robot-memory probe needs a source/format confound control — which is exactly why `Present but Not Remembered` residualises against the current frame (rung 8).

> **The obvious next experiment, and it is cheap: no published work has run interchange interventions (rung 7) on a memory-augmented VLA** to establish *which memory token* causes *which action* at *which layer*. The strongest audit paper does rungs 7–8 on frozen models with *no memory module*; the strongest memory-module papers do rungs 1–5.

---

## 13. Benchmarks

### 13.1 Memory-specific manipulation

| Benchmark | Size | Design | Key number |
|---|---|---|---|
| **MIKASA-Robo** `arXiv:2502.10550` ICLR 2026 | 32 tasks, 12 categories, 4 memory types | One memory skill per task; RGB+joints | PPO-LSTM ≫ PPO-MLP on RememberColor |
| **MIKASA-Robo-VLA** `arXiv:2610.00604` | **90 tasks, 10 memory types, 22,500 oracle trajectories** | **80 memory-dependent + 10 reactive controls**; Short 38 / Medium 30 / Long 22; **28 tasks have gaps exceeding the widest fixed-context VLA's 16-frame window** | π₀.₅ no-history: **0.211 ± 0.044** |
| **RoboMME** `arXiv:2603.04639` ICML 2026 Oral | 16 tasks, 1,600 demos, 770 K timesteps, **14 memory-augmented π₀.₅ variants** | Counting / Permanence / Reference / Imitation; **human baseline 90.5%** | FrameSamp-Modul **18.2 → 45.3%**, **+27.1 pp** |
| **RoboMME-Interference** `arXiv:2606.22338` | 12 systems × 9 families × 50 episodes × 5 session lengths | Cross-session `k ∈ {0,1,3,7}` | Perceptual memory decays; symbolic holds |
| **RoboMemArena** `arXiv:2605.10921` | 26 tasks, avg 1,076 steps, 15,100 keyframe-aligned segments | Native keyframe annotations; paired real-robot eval | π₀.₅ 21.5% → PrediMem 38.5%; **real robot 52% vs 20%** |
| **MEMOBench** `arXiv:2609.07047` | 30 tasks, 1,500 demos, 84 checkpoint templates → 4,200 instances | **Process-level** Storage / Update / Compression via simulator predicates | **MSR 94.0% vs SR 4.4%** |
| **RMBench** `arXiv:2603.01229` | 9 tasks | **Task Memory Complexity** `M(1)` vs `M(n)` — characterises the *task*, so comparable across architectures | ⚠️ widely-quoted ablation deltas come from an **AI-generated summary**, not paper text. Verify Table 2 |

### 13.2 Theory-side and navigation

- **POBAX** `arXiv:2508.00046` — **memory improvability** as the benchmark-validity criterion. Deliberately excludes needle-in-a-haystack as "diagnostic… as opposed to partial observability of interest."
- **Memory Gym** `arXiv:2309.17207`, JMLR 2025 — endless tasks probe memory *effectiveness* rather than sample efficiency. **GRU beats TrXL significantly on all endless tasks**; TrXL wins on finite tasks *only with* an auxiliary reconstruction loss.
- **Synthetic POMDPs** `arXiv:2508.04282` — Memory Demand Structure; constructs POMDPs from MDPs via state aggregation and reward delay *while preserving the optimal policy and the belief MDP*.
- **NaVQA / OC-NaVQA** — up to 35.8 min, 1.64 km. DAAAM **+53.6%** accuracy.
- **DIACRITIC** `arXiv:2609.25757` — behavioral quotient, minimal rate `R_E(D) = min I(C; H_t | O_t)`.

### 13.3 Two structural threats to all of the above

**Saturation.** LIBERO/SimplerEnv/CALVIN account for **>70%** of sim-evaluated papers and **>98%** clustering on LIBERO. These are Markovian *by construction* — which is precisely why the baseline is single-frame. A memory module moving LIBERO 95.6 → 97.7 is inside the noise band of methods that use no history at all.

**Context saturation.** `arXiv:2610.17350` claims a **frozen** frontier VLM given RoboMME's full episode in context, with **no RoboMME-specific training**, reaches **92.1%** — above the 90.5% human baseline and above the best leaderboard entry (60.8%). Matches or beats humans on 13/16 tasks.

If this replicates, **RoboMME measures long-context capacity, not memory architecture.** Treat as a single unreplicated run — the authors say so — but the direction of travel is clear. **The next benchmarks must be adversarial with respect to context length.**

---

## 14. Recommended architecture

Derived from the ledger above. Optimised for **low latency** with the evidence's failure modes designed out.

### 14.0 The latency budget everything follows from

50 Hz control = **20 ms**. No architecture question exists above that number.

| System | Latency | Rate | Hardware |
|---|---|---|---|
| **FlashVLA** | **26.7 ms** | ≥30 Hz, 50 Hz @5090 | RTX 4090 |
| **ReCAT** (recurrent memory, 374M) | **59 ms** | 16.9 Hz | 1.44 GB VRAM |
| **GlanceWAM** | 48 ms / 16-chunk | — | A100 |
| MemER (hierarchical + keyframes) | **787 ms** | ~1 Hz | 2× 4090 |
| Cosmos Policy | 610 ms/chunk | 1.6 Hz | H100 |
| V-JEPA 2 latent planning | **16 s/action** | 0.06 Hz | 4090 |
| Cosmos pixel planning | **240 s/action** | 0.004 Hz | 4090 |

**Rules that fall out:**

1. **No VLM on the fast path.** MemER is 9× slower than π₀.₅ alone purely because a 7B VLM entered the loop. π₀.₅ is 88 ms / 6.25 GB with *no memory*; memory frames were cheap, the model reading them was not.
2. **Latent, never pixel.** 15× on identical hardware.
3. **Fixed-size state, not KV cache.** KV grows linearly with history; SSM/associative state does not.
4. **Amortise with chunking + async execution.** This is where the latency win actually lives.

### 14.1 Three loops, three clocks

```
╔═ LOOP A ═══ REFLEX ═══════════════════════════════════════════════╗
║  200–500 Hz  ·  budget 2 ms  ·  CPU only  ·  no learned net      ║
║  proprio + IMU → gravity-compensated PD → joint torques          ║
╚═════════════════════════════════════════════════════════════════╝
                              ↓ q_t, τ_t
╔═ LOOP B ═══ ACTION ═════════════════════════════════════════════╗
║  30–50 Hz  ·  budget 20–30 ms  ·  GPU or 16-core CPU             ║
║  proprio + 4 frames → RECURRENT POLICY → action chunk H=32       ║
║  memory updated EVERY CONTROL STEP (not every chunk)             ║
║  execute k=8, replan  ·  RTC-style async                         ║
╚═════════════════════════════════════════════════════════════════╝
                              ↑ keyframe proposals (write-gated)
╔═ LOOP C ═══ SEMANTIC ═══════════════════════════════════════════╗
║  0.5–2 Hz  ·  budget 200–800 ms  ·  GPU  ·  ONLY place VLM lives║
║  keyframe consolidation · compressed language memory             ║
║  writes gated on VERIFIED progress · never returns actions       ║
╚═════════════════════════════════════════════════════════════════╝
                                                                        ↑ query only
╔═ LOOP D ═══ RECORD ═════════════════════════════════════════════╗
║  off-loop  ·  human-scale time  ·  storage                       ║
║  scene graph / ledger / keyframe index — queried, never pre-filled║
╚═════════════════════════════════════════════════════════════════╝
```

**Why separate clocks** — D²-VLA (`arXiv:2609.34792`) found the mechanism: *"the VLM relies on broader history for **semantic reasoning**, whereas the action expert attends mostly to token subsets from **recent observations that contain motion information**."* Two consumers, two memory rates. This is architecture, not folklore.

**The cadence rule** — μVLA had to abandon chunked deployment entirely because memory was updated per step but consumed per chunk. Their fix cost **20.9 → 9.2 Hz**. Ours avoids it: **memory advances every control step, including during chunk execution.** Chunk replanning is a separate, slower clock.

### 14.2 Loop B in detail

**Input — sampling, not stacking.** IntentVLA measured that uniform sampling from a window is worth ~18 points at identical frame cost, and that contiguous history at 8–16 frames **OOMs**. Adjacent frames are near-duplicates.

```
proprio  q_t ∈ R^11                    (7 joints + 3 EE + gripper)
frames  4 × uniform(last 16) @ 224²  → frozen encoder → pool 64 tok/frame
                                      → 256 visual tokens
```

**64 tokens/frame** is SmolVLA's number (512² → PixelShuffle → 64 instead of 1024) and is the bandwidth cap that makes 4 frames affordable — a deliberate *discard* of the VLM's own tiling.

**Memory — a dual-rule fixed-size state.**

ReCAT measured that the update rule determines what the memory can represent, and **the ordering inverts by task**: Mamba-2 (accumulate) 85% counting / 20% recall; Gated DeltaNet (replace) 20% counting / 40% recall. Manipulation needs both. So **two channels**:

```
Δ-CHANNEL  (recall)    Gated delta-rule, key-addressed replace
   S_t = α_t(I − β_t K_t K_tᵀ) S_{t−1} + β_t V_t K_tᵀ
   S ∈ R^{H × d_k × d_v},  H=4, d_k=d_v=64   →  ~65 KB

Σ-CHANNEL  (accumulate)  gated additive, scalar-decayed
   C_t = a_t ⊙ C_{t−1} + γ_t ⊙ z_t
   C ∈ R^{256}                                  →  ~1 KB
```

Both closed-form, differentiable, ~200 lines (DRAM's shape). Read-before-write, so the current frame never leaks into its own readout.

> ⚠️ **The dual-channel split is a synthesis, not a published result.** The evidence shows each rule wins on a different task family; running both follows from that, but I have not seen it published. It is the piece most worth validating, and rungs 3–4 test it directly. **Total state: ~66 KB, constant in episode length.**

**Frame-synchronous write.** DRAM's best-engineered contribution: write a whole frame as **one joint least-squares solve**, not token by token.

$$S_t=\arg\min_S \tfrac12\|S-\bar S\|_F^2+\tfrac12\|W_t^{1/2}(K_tS-V_t)\|_F^2,\quad \bar S=\gamma_t S_{t-1}$$

Three properties that matter: reduces exactly to gated DeltaNet for a single token; **permutation-invariant within a frame** (spatial layout and camera order are semantically meaningless); evidence accumulation **saturates** at `nw/(1+nw)`, which is what makes counting work. Cost: one small Cholesky solve per frame, sub-millisecond.

**Readout — every block, cross-attention.** ReCAT's ablation: cross-attention in every block **66.7%**, late fusion 38.3%, AdaLN 31.7%, learned scale 28.3%, content gate **8.3%**. A weak gate is *worse than useless*.

**Action head — flow matching with prefix caching.** π₀'s trick, and the only reason flow-matching VLAs run at 50 Hz: cache KV for the observation prefix once, recompute only the 50 action tokens × denoise steps. Cosmos Policy showed 5 → 1 step costs only 0.7 pp (67.1 → 66.4), so **start at 1–2 steps**. Chunk H=32, execute k=8.

### 14.3 The six features that make it work

Each prevents a specific published failure.

**① Write gate on verified progress.** MEM's result (§6.5): naive accumulation breaks the train/test distribution because inference repeats failed subtasks. **Writes must be idempotent under failure.** Gate on detected success, not elapsed time.

**② Memory as soft bias, never as the action.** IOM: a wrong retrieved procedure costs ≈ 0 on ~88% of instances *"because the procedure is a soft bias on a feedback-driven policy."* Memory conditions; the closed loop decides. **Unbuyable in LLM RAG.**

**③ Past-Token Prediction loss.** PTP: without an auxiliary objective, long context makes diffusion policies **worse than no history**; with it, 3× improvement and >10× faster training. This is what makes memory *readable* rather than decorative. Non-optional.

**④ Graceful degradation — a hard code path.** DRAM's reset sweep (§3.3): reset every 8 frames → **73%**, *below* the 96% memoryless baseline. **A half-alive memory is worse than none.** Watch `‖S_t‖` and the Δ-channel write rate; on divergence or detected reset, **hard-fall-back to stateless** rather than limp on with a corrupted state.

**⑤ Cadence decoupling, explicit.** Memory clock ≠ chunk clock. Advance state every control step during chunk execution. Two lines of code; getting it wrong costs 2.3× throughput.

**⑥ Ablation switches in the code, not a notebook.** Rungs 3 and 4 as runtime flags:

- `MEM_WRITE=0` — reads intact, writes disabled (T²Mem measured **−82.67 pp** this way)
- `MEM_SOURCE=donor` — swap in a conflicting memory, to test *reading* vs *presence*

If you cannot run these two flags you will never know whether the memory works or is decoration.

### 14.4 Training recipe

| Stage | What | Why |
|---|---|---|
| 1 | BC on proprio + 4 sampled frames, **memoryless** | establish the floor first; everything is measured against this |
| 2 | Add Δ/Σ channels, **frozen visual trunk**, **episodic (not shuffled) dataloader** | shuffling destroys temporal structure — +6 pp alone (μVLA) |
| 3 | Add **PTP auxiliary loss** | without it long context *hurts* |
| 4 | TBPTT, **sweep K ∈ {1,2,4,8}** | U-shaped: **K=2 ≫ K=1 and K=8** (0.84 vs 0.57) |
| 5 | Teacher-force observations; let the state be model-generated | otherwise it never learns to *write* |
| 6 | **Attention-mask the degenerate path** | without it the memory just copies the predicted action |
| 7 | FrameSkip-style retention at **20–50%**, never 100% | 66.5 → 76.2 macro SR, pure dataloader change |
| 8 | Train Loop C's language memory **jointly** | keeps it calibrated to what Loop B needs |

### 14.5 Where the budget goes

| Stage | Budget | Measured anchor |
|---|---|---|
| Frozen visual encode, 4 frames @ 64 tok | 5–8 ms | SmolVLA's 64-tok cap |
| Memory update (2 Cholesky + SSM) | **<1 ms** | DRAM frame-synchronous solve |
| Action head + 1–2 denoise steps | 8–14 ms | FlashVLA 26.7 ms decode @4090 |
| **Total Loop B** | **~20 ms → 45 Hz** | ReCAT: 59 ms @ 16.9 Hz, 374 M |

**Sizing (4 GB GPU, or CPU with 16 cores):**

| | Params | Fits |
|---|---|---|
| Visual trunk, frozen, pooled | ~90 M | ✅ |
| Δ + Σ memory | ~0.2 M (66 KB state) | ✅ |
| Action head, width 512, 8 blocks | ~120 M | ✅ |
| **Trainable** | **~40 M** | ✅ comfortably |

Batch-1 CPU forward for ~200 M params is roughly 8 ms on 16 cores, so **~45 Hz on GPU, ~20–25 Hz on CPU** is realistic.

**Trim list if you need faster**, in order of cost:

1. denoise steps 2 → 1 (~0.7 pp)
2. 4 frames → 2 uniformly sampled
3. 64 → 32 tokens/frame
4. cross-attn every block → every *other* block — **this one hurts a lot** (66.7 → 38.3)
5. drop the Σ-channel if the task is pure recall

### 14.6 Traceability

| Choice | Evidence | Measured |
|---|---|---|
| no VLM in fast loop | MemER | 88 ms → 787 ms |
| latent over pixel | V-JEPA 2 | 16 s vs 240 s, same GPU |
| uniform frame sampling | IntentVLA | 28.1 vs 10.4 |
| memory every control step | μVLA | 20.9 → 9.2 Hz if you don't |
| hard cross-attn readout | ReCAT | 66.7% vs 8.3% gated |
| prefix KV-cache | π₀ | enabler of 50 Hz |
| PTP loss | PTP | 3×; without it context *hurts* |
| gated writes | MEM | naive accumulation is worse |
| soft bias, not override | IOM | wrong memory costs ≈ 0 |
| stateless fallback | DRAM | interrupted 73% < 96% |
| TBPTT K=2 | μVLA | 0.84 vs 0.57 |
| 20–50% frame retention | FrameSkip | 76.2 vs 66.5; 100% worst |

---

## 15. Build path

Sequenced so the hard, failure-prone parts are verified before any gradient touches them.

**Step 0 — Loop A is already done.** Gravity-compensated PD, verified to **0.001 mm** static accuracy in this repo. Reflex layer complete.

**Step 1 — the memory module, with no learning at all.**
Δ + Σ channels, frame-synchronous write, read-before-write. ~200 lines. Unit-testable immediately:
- write 5 identical cues → Σ saturates, Δ does not
- swap read/write order → results differ (proving read-before-write)
- permute a frame's tokens → state unchanged (permutation invariance)
- force a reset → confirm the fallback path triggers

**Step 2 — a task that provably requires memory.**
MuJoCo, masked observation, POBAX-style **memory-improvability** criterion. Include a **reactive control** where the cue stays visible — if the reactive version is not solved, the task is broken. Prefer MIKASA-style shells/cups, or a counting task (3DPS-style), which exercises both channels differently.

**Step 3 — the memoryless floor.** Confirm it is at chance. Everything after is measured against this number, and it must be reported.

**Step 4 — Loop B with the ablation flags from day one.** `MEM_WRITE`, `MEM_SOURCE`, `MEM_SIZE`, `MEM_CLIP`, `MEM_SAMPLER`. Training recipe per §14.4.

**Step 5 — the ablation ladder.** Rungs 1, 3, 4. The headline deliverable is rung 3: **the number you get when you disable writes while keeping reads.** T²Mem got −82.67 pp. If yours drops meaningfully, the write path is real.

**Step 6 — build the negative result deliberately.** Add naive contiguous frame stacking, show it *hurts*, then show uniform sampling helps (IntentVLA's 9.0 / 10.4 / 28.1). Cite both directions. Then add 100% frame retention and show it is the worst setting (FrameSkip).

---

## 16. Open problems

Ranked by tractability × importance.

1. **The write/update path is the gap.** Storage is 94%; update and compression are 4.4%. Every architecture optimises what goes in. Almost none handles **revising a stored belief when contradicted**, and almost none handles **correct abstraction**.
2. **Nobody has run rung-7 interchange interventions on a memory-augmented VLA.** Cheap, tractable, and would convert the whole design literature from correlation to causation. **The single highest-value missing experiment.**
3. **Persistent object identity is unsolved.** 33–46% identity-switch rates on real arms; 5/9 wrong-target. Should be the field's #1 benchmark target. Only *acting* produces the evidence — MG-VQA shows perception saturates near chance.
4. **Memory drift at scale.** Every hierarchical system updates at 1–2 Hz. A 15-minute task is ~1,000 updates. **No published system handles silent corruption of language memory over hundreds of updates.** This is the most likely failure mode of the MEM architecture at scale.
5. **Episodic boundaries do not exist in production** (Dyna's own statement). Every architecture and benchmark assumes discrete episodes. Cross-session, week-scale memory is unaddressed by both literature and product.
6. **Consolidation exists in no deployed robot.** Sleeping Robots is simulated; 1X lists it as future work. The most important transfer from neuroscience to robotics is entirely unrealised.
7. **The right retrieval key is unknown.** `arXiv:2610.05492` shows no method dominates and random is non-trivial. **Robustness to one kind of mismatch does not imply robustness to another** (MemTransfer). Highest-leverage unsolved retrieval question.
8. **The 3D-vs-graph-vs-keyframes question has no controlled three-way comparison.** GaussMemory says geometry (+3.9); Memory over Maps says no reconstruction (>100× faster indexing); Ledger says a tracker and captioner suffice.
9. **Benchmarks are saturable by brute-force context.** The frozen-frontier-VLM claim (92.1% vs 90.5% human) would mean current suites measure context length, not architecture. Needs adversarial benchmarks where the relevant memory is a vanishing fraction of a long stream.
10. **No standard metrics.** Four 2026 benchmarks with incompatible metric sets, no cross-benchmark leaderboard, no "ImageNet moment."
11. **Nobody measures write quality.** MSR measures whether a *labelled* fact was stored, not whether the system chose the *right* things to store.
12. **Memory is a security surface with no benchmark.** No demonstrated physical poisoning attack, but writes are increasingly LLM-produced.
13. **Peripersonal space / body-schema memory is a total void** in the VLA era, despite manipulation's hardest problems being body-centric.
14. **Event cameras as a memory channel are unclaimed.** Sub-ms persistence is exactly the short-term tier that everyone pays 5–8 frames of tokens for. Nobody has run the experiment.
15. **Mobile manipulation × episodic recall is unsolved and acknowledged as such.** Navigation memory and manipulation memory are two literatures never joined in one system.
16. **Interference has no general solution.** RoboMME-Interference shows retrieval restores it *at the tested scale*; nobody has characterised the frontier as a function of (bank size, task count, session length).

---

## 17. Citation hygiene

### 17.1 A correction to this document's own research process

**"Recurrent Action Chunking (RAC, 2025)" does not exist.** It was supplied as a research lead and the agent tasked with verifying it searched the arXiv API exhaustively:

- `all:"recurrent action chunking"` → **0 results**
- `abs:"recurrent" AND abs:"action chunk"` → 3 results, all 2026
- `ti:"recurrent action"` → 4 results, none matching

Nearest real names: **RaC** (`arXiv:2509.07953`, human-in-the-loop recovery/correction data collection — **not** recurrent), **RACE** (`arXiv:2610.05719`), **RACCT** (`arXiv:2508.01808`, IROS 2025, nasotracheal intubation). **Do not cite it.**

### 17.2 Claims that must not be presented as fact

| Claim | Status |
|---|---|
| **GR00T N2** ("world model," 2× novel-task success, Mar 2026) | Only `techbytes.app`, `embodiedglobal.com`. **No primary NVIDIA source** |
| "48 networks / 70,000 GPU hours / 8.2B miles / 67K H100-equivalent / AI5 40×" | **Exclusively** `optimusk.blog`, an unsourced aggregator |
| Helix 2.5 / "System 0 at 1 kHz" / "13 GB bf16" / "9%→56%" | Aggregator sites and AI summaries. **Figure has published no latency figures** |
| Skild "10 Hz high-level / kHz low-level" | From a GitHub mirror of an Intel deck, **not** Skild's blog |
| RMBench ablation deltas (67→15, 68→5, 90→78) | From an **AI-generated** alphasXiv summary, **not paper text** |
| "RLSim" | **No such system verified.** Likely conflation with MimicGen / robomimic / RLBench / RealLab |
| Jetson Thor GR00T N1.5 figures (41.5 / 15.2 / 2.74) | Appeared **without units**. Do not quote without the datasheet |
| VPP CALVIN improvement | **+28.1%** (arXiv v2 abstract) vs **+41.5%** (project page). Unresolved |
| Cosmos-Predict1-7B "20M hours of video" | Second-hand, quoted from the V-JEPA 2 paper, not the Cosmos report |
| π₀.₅ "action horizon of 50, i.e. H=49" | Internal inconsistency in the paper appendix; treat H=50 as operative |
| GR00T N1 S1 rate | Paper says **120 Hz**; some secondary sources say 30 Hz. **Trust the paper** |
| MemoryBench, LIBERO-Mem, HAMI | Cited-only, not opened |
| AX-Task / ACE-Bench | **Could not be found.** Do not cite |

### 17.3 Vendor figures (authoritative for what ships, not peer-reviewed)

Token rates for Gemini / Claude / OpenAI come from vendor API documentation. NVIDIA VSS RTVI-VLM throughput comes from NVIDIA product documentation. Genie 3 specifications come from the DeepMind blog and video only — and **Genie 3 has no robot control demonstrated**; Fruchter concedes on the Hannah Fry podcast that you cannot drop it into the real world today.

### 17.4 Standing caveat

Because this survey was compiled in October 2026, the 2026 arXiv identifiers (2601–2610) dominate recent results. Several — notably `arXiv:2610.17350` (the frozen-frontier-VLM claim) and `arXiv:2606.02907` — may be pre-replication or pre-review. Their claims are reported, not established. **Before this document is relied on, verify the ~15 load-bearing citations directly against arXiv.**

---

## 18. Caveats and known gaps

**A research agent's output was truncated.** The agent covering brain-inspired cognition and production systems returned a partial report beginning mid-sentence. Its evaluation and benchmark sections (§11.1, §12, §13) came through and are among the strongest in the survey. **Its Parts A and B are missing** — specifically:

- Complementary Learning Systems (McClelland et al. 1995; Teyler & DiScenna 1986; Go-CLS, *Nat. Neurosci.* 2023) and algorithmic consequences for robot memory design
- Tiered memory (Atkinson–Shiffrin-derived implementations; MemoAct `arXiv:2603.18494`, VPWEM `arXiv:2603.04910`)
- Predictive processing / active inference as a sleep-time consolidator
- Event cameras and SNN memory as a temporal medium
- Vector Symbolic Architectures and hyperdimensional computing — theoretically the most principled distributed memory representation (Clarkson et al., JAIR; real capacity theory; real neuromorphic implementations) and **absent from every 2026 robot memory benchmark**
- Detailed per-company architecture comparison for PI / Figure / 1X / Tesla / Google DeepMind / Skild / Covariant / Dyna
- The single bona fide sleep-consolidation paper for robot policies (`arXiv:2606.17493`, simulated), and `arXiv:2309.17207` Memory Gym

**These gaps are material.** The single most important transfer from neuroscience to robotics is currently unrealised (§16.6), and the absent VSA/HDC representation is the survey's largest unfilled niche. **Recommend a follow-up research pass specifically on Parts A and B.**

**Other gaps:**

- **Sim-to-real for memory is essentially unmeasured.** Exceptions are 3DGS-specific (YUBI-Real2Sim: novel-view quality scores fail to predict policy ranking; Neverwhere: don't train exclusively on 3DGS data). **No real-vs-sim ablation on episodic memory exists.**
- **Trial counts are tiny.** REMM: 51 physical trials including operator-assisted runs. ReMemArena real robot: 10 rollouts/task. ReCAT: 20 rollouts/task, single seed, Wilson CI 54.1–77.3% for a 66.7% figure. The cumulative real-robot memory evidence base is on the order of hundreds of trials.
- **Memory-vs-more-tokens is under-controlled.** Fast-VLA (1.22 GiB, 17× faster than continually-conditioned baselines) and TacitVLA (matches explicit reasoning at 3.4× lower latency) suggest a substantial share of "memory" gains are representational or architectural.
- **Latency accounting is inconsistent across papers.** MEM's H100 4-camera, HAMLET's A100 single-view, GR00T's L40 are not comparable. π₀.₇ trains for 240 ms of simulated delay; nobody reports behaviour at 500 ms.
- **Peer review is thin in exactly the interesting areas.** The peer-reviewed spine is: RT-1 (RSS'23), RT-2 (CoRL'23), OXE (ICRA'24), OpenVLA (CoRL'24), MaIL (CoRL'24), Octo (RSS'24), π₀ (RSS'25), FAST (RSS'25), MCNN (ICLR'24), STRAP (ICLR'25), RICL (CoRL'25), PTP (CoRL'25), RTC (NeurIPS'25), HAMLET (ICLR'26), RoboMME (ICML'26 Oral), MIF (RSS'26). DRAM, μVLA, ReCAT, Chronos, RoboTTT, MemBodied, Ledger, MessyMem, GaussMemory are **arXiv-only**.

---

*End of document.*