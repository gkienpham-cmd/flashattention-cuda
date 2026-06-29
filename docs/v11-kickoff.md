# v11 kickoff — MLA (Multi-head Latent Attention) latent-KV decode on B300/sm_103 (the shape-change closer)

> Paste-in starter for a fresh session. Follows the per-step loop in `ROADMAP.md`
> (roofline-first → explain → write → correctness → bench → results/decisions → quiz) and mirrors
> `docs/v10-kickoff.md`. Born from the **v10 deep-research close-out (2026-06-30, 18-agent
> verify+adversarial+research pass)**. Read FIRST: `docs/decisions.md` Step 10 + the v10 close-out,
> `docs/results.md` Step 10 (the B300/sm_103 record), `docs/decode-replan.md` §5 (v11 row),
> `docs/v10-kickoff.md` (the format + the per-CTA constraint this inherits).

## 0. The one honesty constraint that governs everything (read first)

v10 measured, **confound-free across T4 / B200 / B300**, to **2M tokens past an 8× L2 overflow** (1 GB working
set ≫ the measured 132.6 MB L2 — kills the L2-residency confound by construction, so the verdict is earned
*without* ncu): **decode on this kernel is PER-CTA / low-MLP latency-bound (1 active warp at N_q=1), NOT
bandwidth-bound.** Cutting KV bytes buys *nothing* for decode latency — NVFP4 is even **12–30% latency-NEGATIVE**
past L2 (the dequant ALU sits on the per-CTA critical path). The single most defensible evidence: the
**architecture-independent ~40 GB/s B=1 FP16 ceiling** — identical achieved throughput on a T4 (320 GB/s bus)
and a B300 (8 TB/s bus), a 25× span that extracts no more at B=1.

**Therefore v11 cannot be sold as "FP4 compute / fewer bytes / a faster kernel buys decode latency."** v10
already disproved that. **v11's entire reason to exist is to RAISE M above 1 — change the attention SHAPE —
so more than one warp is active and arithmetic intensity rises off the floor.** The per-CTA wall is the
contribution to *overturn (or confirm immovable)*, not bury.

**Inherit and pay down v10's honesty debts** (all logged in the v10 close-out):
- **Lock clocks.** Every v10 record run was **unlocked** (`lock_clocks` failed, `locked=False`); cross-precision
  µs/tok is clock-confounded. Get root/bare-metal and lock, or run all backends in one process at one clock.
- **Get ncu on a privileged Blackwell box.** The counter-free %HBM proxy is ncu-validated only **once, on a T4**
  (v9 Task 1: proxy 13.8% vs ncu DRAM 12.85%). On Blackwell it's pigeonhole-from-capacity, not measured. ncu
  was **never even installed** on the v10 B300/B200 boxes — Pass-2 needs *both* an `ncu` install *and* privilege.
- **Regenerate the nsys schedule in-notebook.** The 99.4%/99.3% partial-kernel tables are **hand-saved `.txt`**;
  the committed regime notebook's nsys cell ran 2025.1.3 → *empty* sm_103 trace. Standardize **nsys 2025.3.2+**
  and capture in a re-runnable cell.

## 1. The thesis — contributions ranked by defensibility against the per-CTA verdict

**T1 (PRIMARY, the shape change): MLA latent-KV decode raises decode AI ~16–30× → the first decode shape in the
arc that is plausibly compute-bound.** MLA shares **one** low-rank latent KV across **all** `h_q` query heads,
so decode AI ≈ `2·h_q` FLOP/byte (vs ~1 MHA, `2G/b` ≤ 8–16 GQA). At `h_q=128` that is **AI ≈ 256**, sitting just
under the B300 FP16-TC ridge (~312 FLOP/B) — the first time in the v1→v10 arc that decode plausibly crosses from
per-CTA/HBM-bound toward compute/tensor-core-bound. **The deliverable is the open, roofline-documented,
prediction-vs-measured determination of whether MLA decode is compute- / memory- / per-CTA-bound on sm_103** —
the empty cell (FlashMLA ships the kernel; nobody has published the *kernel-level roofline* on sm_103).

**T2 (the tensor-core M-gate, solved for free): `h_q=128` packing meets the tcgen05 NVFP4 `M≥128` gate BY
CONSTRUCTION at N_q=1.** No speculative draft model needed. This is the cleanest path in the whole arc to
engaging Blackwell's 5th-gen FP4 cores — MLA packs all 128 query heads into the M dimension against the single
shared latent, so M=128 falls out of the math, not out of a draft-acceptance gamble (contrast the speculative
path, which needs `q_len ≥ 8–16` just to clear M=64).

**T3 (conditional — the prize, either sign publishable): does the limiter FLIP, or just MOVE?** Raising AI to
~256 and packing 128 heads *should* relieve the 1-warp per-CTA wall — but it may simply **trade per-CTA-bound
for smem-capacity / register-pressure-bound**, because MLA's absorbed up-projection weight (`W^UK`/`W^UV`) and the
512-wide latent must be staged **on-chip** or the recompute benefit evaporates. *Does the limiter flip to
compute, or rename itself to capacity?* That is the project's signature question, and either answer is a result.
Carry **v8.8's 4-blocks/SM occupancy residency** for the large-batch serving regime (a cheap, already-measured
~1.4× at B≥32 — see the v10 close-out).

**T4 (the sm_103 hook, now possibly live): with M=128 putting many independent exps in flight, does the 2×-exp
SFU lever finally convert?** On v10's M=1 decode the exp lever was a measured dud (EX2 0.50×, mufu share <3% and
*N_k-invariant*). MLA's M=128 changes the exp parallelism — re-ablate hardware-exp2 on/off and see if the
marquee sm_103 softmax lever matters when there are 128× more exps to overlap.

> **The paper, stated honestly:** *An open, roofline-documented, prediction-vs-measured kernel-level
> compute-vs-memory-vs-per-CTA characterization of MLA (latent-KV) decode on B300/sm_103 — testing whether
> decode can EVER leave per-CTA-bound on Blackwell — with the FP8/NVFP4 latent-storage recipe carried from
> v9/v10, benchmarked against FlashMLA / FlashInfer.* The wedge is **openness + the per-CTA-honest methodology +
> the compute-vs-memory crossover determination**, NOT a headline speedup and NOT "first MLA decode" (FlashMLA
> ships it; see §7).

## 2. Roofline FIRST (record before coding)

**MLA decode AI ≈ 2·h_q FLOP/byte** (the single shared latent is read once and reused by all `h_q` heads).
Record the table before touching the kernel:

| attention | KV bytes/token (approx) | decode AI (FLOP/byte) | regime on B300 (FP16 ridge ≈ 312) |
|---|---|---|---|
| MHA | `2·H·d` | ~1 | deep HBM/per-CTA-bound |
| GQA-8 (v8–v10) | `2·H_kv·d` | `2G/b` ≤ 8–16 | per-CTA-bound (MEASURED) |
| **MLA (v11)** | **~576 dims/token (512 latent + 64 RoPE), ~656 B fp8** | **~2·h_q ≈ 256** | **toward compute-bound — the crossover** |

- **The recompute(absorb) vs reuse(materialize) fork — predict BOTH OIs before coding.** *Recompute* (MLA_rc):
  absorb `W^UK`/`W^UV` into Q/O, apply the up-projection on-chip per step (~100M FLOPs) → OI ≈ independent of KV
  size, **compute-leaning**. *Materialize*: expand the latent back to full K/V → OI degrades with KV size,
  **memory-bound**. The absorbed weight **must stay on-chip** (smem/register) or the recompute win evaporates —
  this is the likely new limiter.
- **B300 ridges:** FP16-TC `2.5e15/8e12 = 312.5` FLOP/B; NVFP4-TC `15e15/8e12 = 1875` FLOP/B. AI≈256 crosses
  toward **FP16-compute-bound** but stays ~7× below the **FP4** ridge — so FP4 *compute* only pays if M=128
  packing genuinely engages tcgen05 (T2) *and* the kernel is otherwise compute-bound. Predict, then measure.
- **Two-layer prediction (the honesty constraint).** *Pure roofline:* compute-leaning, AI ~256 near the FP16
  ridge. *Per-CTA-corrected (the real one):* the model is blind to on-chip capacity — packing 128 heads + staging
  the absorbed weight + the 512-latent may move the limiter to **smem-capacity/register-pressure**, not vanish it.
  *Counter-prediction (the prize):* M=128 packing flips decode to **compute/TC-bound on sm_103** → FP4 cores +
  the 2×-exp lever + the v10 byte cut finally convert. Overturns the per-CTA expectation; first-class either way.

One code change to the model first: `roofline/model.py` — add an MLA branch (`AI = 2·h_q`, latent byte count
`b_latent`) and an `--attn mla` (or `--mla h_q kv_lora_rank`) knob to `roofline/predict.py`. Record the
`predict` output in `results.md` Step 11 **before** writing the kernel.

## 3. The single variable = the attention SHAPE (GQA-over-128-heads → MQA-over-512-latent)

Fork `kernels/v10_nvfp4/` → `kernels/v11_mla/` (`mla_attention.cu` + `binding.cpp`). Carry **byte-identical**
from v8.7/v9/v10 where possible: the score-stationary inner-loop *philosophy*, split-KV partial + LSE merge,
`choose_splits`, host launch, and the FP8/NVFP4 *latent* storage (the latent is the new "KV pool"). Change ONLY:

- **The attention math → latent-absorbed MQA.** One shared latent per token: `kv_lora_rank = 512` content +
  `64` decoupled-RoPE dims (DeepSeek-V2/V3 shape). Absorb `W^UK` into Q and `W^UV` into O so the kernel never
  materializes full K/V — the score is `q_absorbed · c_latent` (MQA over a 512-wide latent), the output a
  weighted sum of latents re-projected through `W^UV`. The **RoPE part is decoupled** (carried separately and
  concatenated) — watch that it does not fragment the M tile.
- **Default v11 = CUDA-core / dequant-to-FP16 MLA** (clean single-variable A/B vs v10's GQA shape: same storage,
  same split-KV, *only* the shape changes). This isolates the **shape/AI** win from the **FP4-compute** win.
- **ONLY-IF the data warrants → native FP4 tcgen05 compute** (the `M=128` GEMM). Gate this on the dev-rung
  *first* proving (a) M=128 packs as **one** GEMM (not fragmented by the RoPE/absorbed-matrix layout) and (b) the
  limiter actually flips. tcgen05 specifics from the research (record them as constraints): the FP4 MMA is the
  **CTA-scoped, single-thread `tcgen05.mma.kind::mxf4nvf4`** with **TMEM** accumulators (256 KB/SM, replaces
  registers); CUTLASS exposes NVFP4 **only at bM=128** (tiles 128×128×256, 128×192×256, 128×256×256 for 1-SM);
  there is **no `mma.sync` FP4 fallback** on datacenter Blackwell, and it compiles **only to `sm_103a`**
  (architecture-specific, not forward-compatible). Mixing tcgen05 with CUDA-core softmax is flagged **expensive**
  (async-proxy sync) → if going native-FP4 you must commit to a warp-specialized TMA/cp.async producer/consumer
  structure (the FlashMLA SM100 pattern).
- **`fa_kernels/`:** new `mla_attention()` API + an MLA latent pool builder (FP8/NVFP4 latent + scales) +
  `sdpa_reference_mla` oracle (explicit-materialization MLA: expand the latent, run reference attention — the
  apples-to-apples truth). Register `v11_mla` in `bindings/load.py` `_SOURCES` and `dispatch.py` `_MIN_CAPABILITY`
  (a CUDA-core T4-buildable path for correctness; the native-FP4 arm gates at `sm_103a`).

## 4. Correctness (Gate 1)

- **Oracle:** explicit-materialization MLA (expand the latent through `W^UK`/`W^UV`, run SDPA) vs the
  latent-absorbed kernel — they must match to tolerance. On **T4 (sm_75, no model weights)** use **synthetic
  low-rank `W^UK`/`W^UV`** (random low-rank projections) so the absorption identity is exercised without a real
  checkpoint. **Open question to resolve early (§9): is a synthetic oracle a *meaningful* correctness check, or
  does a real DeepSeek/MLA checkpoint shift this from a kernel study toward a model-dependent one?**
- Sweep `h_q ∈ {16, 32, 64, 128}` (the M-packing range; 128 is the tcgen05 gate), non-multiple `N_k`, causal both
  ways, **seed-varied** (close v9/v10's single-seed gap — report a small distribution). Idle-head + multi-tile
  edge cases as in v8/v10.

## 5. Benchmark (Gate 2 inputs) + deliverables

1. **Capacity:** MLA latent KV (~576 dims/token) vs MHA/GQA — the published ~**93% KV reduction**; measure max
   context at fixed B300 HBM (288 GB).
2. **Latency, three regimes (mirror v10):** (a) L2-resident micro-bench; (b) **same-session clock-matched
   `vs naive`** (the ONLY trustworthy number — v10 proved cross-clock/cross-process is confounded); (c) **B300
   long-context past-L2** (does the higher AI finally reach a bandwidth- *or* compute-bound regime?).
3. **THE T1 DELIVERABLE — the compute-vs-memory crossover plot:** achieved %HBM **and** achieved TFLOPS vs
   `N_k`/batch, with the per-CTA cap and the FP16/FP4 ridges marked. Does MLA leave the ~0.5%-HBM / ~40 GB/s
   per-CTA floor that GQA decode never left?
4. **The FP4-compute A/B (only if the native arm runs):** FP16-TC MLA vs FP4-TC MLA **at matched M=128** — isolates
   the FP4-*compute* win from the shape/M-lift win (single-variable discipline preserved).
5. **sm_103 deltas:** re-run the 2×-exp ablation now that M=128 puts many exps in flight (T4).
6. **Comparators, CLOCK-LOCKED this time, matched precision:** **FlashMLA** (SM100 backend, tcgen05 — *the* bar),
   **FlashInfer trtllm-gen**, **vLLM**. Add an **FP8-Q path** so the FlashInfer comparison is apples-to-apples
   (v10's ~3× was FP16-Q vs its FP8-Q + an NHD→HND transpose penalty — fix both before quoting a number).

## 6. Hardware

- **Record on B300/GB300 sm_103a** (the FP4/exp exclusives need the `a` target). **ROOT/bare-metal mandatory** for
  clock-lock + ncu (both owed since v9/v10).
- **Dev rung: B200 sm_100f** (portable; tcgen05/TMEM ISA ports sm_100→sm_103). Iterate correctness + the MLA
  absorption + the FP4 path cheaply; record on B300.
- **Use the MEASURED arch constants** (not the aggregators): **148 SMs**, **132.6 MB L2**, **228 KB smem/SM**,
  **2032 MHz**, **288 GB**, FP16-TC 2.5 PF / FP8 5 PF / **NVFP4 15 PF** dense, exp 10.7 TExp/s (achievable ~5.3).
  (Reject the spec/marketing 160 SMs / 192 MB L2 / 2600 MHz — v10 refuted all three.)
- **Toolchain:** CUDA 12.9+ / PTX 8.6+ for sm_103a + tcgen05; nsys **2025.3.2+** (2025.1.3 records an empty
  Blackwell trace). TMEM accumulator management is mandatory for the native-FP4 arm.

## 7. Novelty — defensible but NARROW (adversarially checked)

Dead claims (do not make them): "first MLA decode" (**FlashMLA ships it, Blackwell-native, tcgen05+TMEM**),
"beat FlashMLA/FlashInfer." **What survives** (empty cell as of June 2026): *an **open, roofline-documented,
prediction-vs-measured KERNEL-LEVEL compute-vs-memory-vs-per-CTA characterization of MLA decode on sm_103** —
the determination of whether decode can ever leave per-CTA-bound on Blackwell.* GB300 MLA serving exists but is
reported **only at the system level** (~2,200→11,200 tok/s/GPU), never as a kernel roofline. Frame **"complementing,
not beating," timestamped.** The field is moving to **sparse** latent attention (DeepSeek V3.2 DSA, V4 CSA/HCA)
and to **GLA** (Grouped Latent Attention, Tri Dao — beat FlashMLA ~20% @ q_len=1 via a single `h_c` knob) — flag
both as **v12** (dense MLA is the correct roofline-clean pedagogical rung first; see §9 open question).

## 8. Traps (carried from v8/v9/v10 + new for MLA)

- **Carried:** causal mask uses `i_q` not `m_row`; oracle uses `repeat_interleave` not `repeat`; **fused dequant,
  NEVER a prepass** (QServe); **same-session clock-matched `vs naive` only**; count **4.5 b/elem including the
  per-16 E4M3 micro-scale**; the **score path must be ≥FP16** (v10 measured FP4-score collapses softmax, RMSE
  4.6–6.6×) — keep Q·latent reconstruction at ≥FP16 even on the native-FP4 arm.
- **NEW — the tcgen05 NVFP4 gate is `M≥128`** (CUTLASS exposes only bM=128; FP8 allows M=64) — *stricter* than the
  v8/old-v10 "M≥16/64" assumption. MLA's `h_q=128` meets it; nothing smaller does.
- **NEW — the absorbed weight (`W^UK`/`W^UV`) MUST stay on-chip** or the recompute benefit evaporates (the likely
  new limiter — measure smem/register pressure explicitly).
- **NEW — the decoupled-RoPE part may fragment the M tile** (FlashMLA pads `num_heads` to 128, but the *kernel
  internals* decide whether tcgen05 actually engages as one GEMM) — verify M=128 is one GEMM before banking T2.
- **NEW — tcgen05 + CUDA-core softmax is expensive** (async-proxy sync) → commit to warp-specialized TMA/cp.async
  if going native-FP4. **TMEM accumulator management** is required (no register accumulators).
- **NEW — lock clocks + regenerate nsys in-notebook** (v10's provenance/clock debts — don't repeat them).

## 9. Open questions to resolve BEFORE coding (roofline-predict, then dev-rung-measure)

1. **Does MLA matrix-absorption pack to M≥128 as ONE GEMM at N_q=1, or does the RoPE/absorbed-matrix layout
   fragment it into small-M tiles?** This gates the entire native-FP4-compute thesis (T2). Predict, then T4/B200-measure.
2. **Does the higher AI RELIEVE the per-CTA limiter, or just RENAME it to smem-capacity/register-pressure?** The
   signature question (T3). Needs the T4 occupancy/smem analysis up front.
3. **Is MLA decode compute- or memory-bound at the project's REACHABLE batch/context on B300?** Sources *conflict*
   (AI~256 says compute-bound near the ridge; FlashMLA's padding note + a hardware-centric paper say "remains
   memory-bound in decode"). **This crossover IS the deliverable** — resolve by measurement, not assumption.
4. **Can correctness be validated on a T4 with synthetic low-rank `W^UK`/`W^UV`, or does a *meaningful* v11 need a
   real DeepSeek checkpoint** (and thus become a model-dependent study)?
5. **Is dense MLA the right v11, or is GLA (single `h_c` knob, published FlashMLA-beating) the better
   single-variable rung?** Dense MLA establishes the latent-shape baseline first; GLA/sparse → v12.
6. **Carry the v10 debts in the first paid session:** privileged sm_103 box for ncu, lock clocks, nsys 2025.3.2
   in-notebook, and re-run the v10 accuracy ablations on a **known-sink GQA model** (Llama/Mistral/Qwen — gpt2-small's
   weak V sinks left the per-token-V lever untestable) so the asymmetric recipe is paper-citable.

## 10. North star

The paper: the first **open**, roofline-documented, prediction-vs-measured FA **decode** study on **sm_103**.
v10 delivered the GQA+NVFP4 **per-CTA core** (decode is per-CTA-bound on Blackwell, bytes aren't the wall).
**v11 (MLA) is the shape-change that tests whether decode can EVER leave per-CTA-bound on Blackwell** — the
natural climax of the arc. The contribution is the **open roofline + the per-CTA-honest methodology + the
compute-vs-memory crossover determination**, NOT "first to run." If MLA flips the limiter to compute → the byte
and exp levers finally convert and the arc closes on a win. If it only renames the limiter to on-chip capacity →
that is *itself* a publishable per-CTA-class result, and the speculative-tree shape-lever (q_len=k, EAGLE-3
τ≈6.2) becomes the fallback. Either sign closes the story.
