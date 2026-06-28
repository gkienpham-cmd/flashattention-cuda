# v10 kickoff — NVFP4 + asymmetric-precision KV on B300/sm_103 (the headline + the paper)

> Paste-in starter for a fresh session. Follows the per-step loop in `ROADMAP.md`
> (roofline-first → explain → write → correctness → bench → results/decisions → quiz) and mirrors
> `docs/v9-kickoff.md`. Born from the **v9 deep-research close-out (2026-06-28, 7-agent verify+research
> pass)**. Read FIRST: `docs/decisions.md` Step 9 (Task 1 + Task 2 + the close-out), `docs/results.md`
> Step 9 close-out, `docs/decode-replan.md` §5 v10, `docs/b300-decode-research.md` (with its v10 banner).

## 0. The one honesty constraint that governs everything (read first)

v9 Task 1 measured, **confound-free** (root T4, clocks locked, L2 flushed to 537 MB ≫ 4 MB L2, ncu live):
decode on this kernel is **NOT bandwidth-bound** — past L2 the ncu L2-hit-rate is **1.1%** (data genuinely
from HBM) yet DRAM throughput is only **12.85%**. Achieved HBM BW caps at **~28% of peak** even at full
occupancy and a 1 GB working set. The residual limiter is **per-CTA / low-MLP latency** (1 active warp at
N_q=1), occupancy-lifted to that ~28% ceiling — *not* bytes.

**Therefore v10 cannot be sold as "NVFP4 buys decode latency by cutting bandwidth."** Our own data refutes
that at the occupancy/context a decode micro-bench reaches. v9 Task 2 then *sharpened* it: FP8 bought a
real but **regime-specific** ~1.2–1.3× latency win **only when the KV was L2-resident and occupancy was
high**, and that win **flips negative under L2-flush** (the software-dequant ALU tax dominates once bytes
stop being the shared bottleneck). So the byte-cut latency effect is fragile, not a general decode win.

This plan is built to survive exactly that adversarial reading. The per-CTA-bound finding is not a
weakness to bury — **it is the paper's intellectual contribution.** Most FP4-KV work *assumes* the
bandwidth win; we *measured* that it doesn't hold at realistic decode occupancy and built the
clock-lock/L2-flush/counter-free/ncu methodology to prove which regime you're in.

## 1. The thesis — four contributions, ranked by how defensible they are against the per-CTA verdict

**T1 (PRIMARY, holds regardless of limiter): NVFP4 KV is a CAPACITY + ACCURACY recipe.**
- Capacity: NVFP4 ≈ **0.5625 B/elem** (E2M1 4 b + one E4M3 block-scale per 16 = 4.5 b/elem) vs FP16's 2 B
  → **~3.55× more context (or batch) resident at fixed HBM.** Arch-independent arithmetic; the durable
  headline. On B300's **288 GB** this is the lever that even *holds* a cache long enough to attempt a
  bandwidth-bound regime (T4's job done).
- Accuracy: the **asymmetric-precision recipe** is the genuine research content. v9 measured per-tensor
  E4M3 RMSE ~6–7e-4; v10 must produce the analogous FP4 number, reported as **logit / perplexity-proxy
  error vs an FP16-KV reference (NOT MSE)**, with the asymmetry ablation (see §3).

**T2 (the sm_103 hook — softmax/exp, B300-specific): center the roofline novelty on the exp/SFU term.**
B300 doubles transcendental throughput (**5 → 10.7 TeraExp/s**, ~2×). Decode's softmax is one exp per
score. External research (Hao AI Lab Attn-QAT) explicitly says *"B300 doubles the exp throughput … which
should make a quantized P·V GEMM faster"* — i.e. the standard "keep P·V in BF16" decision may **flip on our
exact target GPU.** This is the cleanest sm_103-vs-sm_100 delta and it is *measurable*; model it in the
roofline (add the MUFU term) and ablate hardware-exp2 on/off. Likely small for a per-CTA-bound M=1 decode,
but it is a first-class prediction-vs-measured number and the honest novelty wedge.

**T3 (conditional — the ONE bandwidth claim we may make, only if earned): does a bandwidth-bound decode
regime exist on B300, and does NVFP4 win there?** v9 Task 1 said decode is per-CTA-bound *at all context
reachable on T4's tiny 4 MB L2*. B300 has 288 GB and a **~126 MB L2** (see §6) — the spill point is pushed
to N_k in the **hundreds-of-thousands to millions**. v10 carries the Task-1 methodology to a **root B300**
and hunts the bandwidth knee there. Find it → NVFP4 wins there (a real result). Don't find it (per-CTA-bound
to 1M tokens) → **the stronger, more surprising result**: decode stays per-CTA-bound on sm_103 at all
reachable context; FP4 is capacity+accuracy, full stop. Either sign is publishable.

**T4 (DEFERRED to v11, not v10 — the gate moved): native FP4 tensor-core COMPUTE.** ⚠️ **Correction from
v9 close-out research:** the Blackwell `tcgen05.mma` gate is **M ≥ 64 (50% datapath), M = 128 for 100%** —
NOT the M≥16 the v8/old-v10 plans assumed (that was the legacy `mma.sync` path). A single GQA-8 group packs
to **M = 8 — below *both* gates.** To feed FP4 tensor cores in decode you must also pack the **query-token
axis** (speculative / multi-token decode) to reach M≥64. So native-FP4-compute is a **v11** lever
(speculative decode), not v10. This *generalizes* v8's measured "tensor cores are the wrong tool for
decode" finding onto native-FP4 silicon: at N_q=1, M=G<64, the 5th-gen cores stay dark. **v10 decode is
CUDA-core / dequant-to-FP16, byte-identical to v9 except storage format.**

> **The paper, stated honestly:** *An open, roofline-documented, prediction-vs-measured NVFP4
> asymmetric-precision FlashAttention DECODE kernel on B300/GB300 (sm_103), reporting (1) the FP4 KV
> capacity + accuracy recipe; (2) the sm_103 2×-exp softmax delta; (3) a confound-free determination of
> whether decode is ever bandwidth-bound on B300 — carrying the v9 Task-1 clock-lock/L2-flush/ncu method to
> B300's large L2 — benchmarked against FlashInfer/FlashMLA.* The wedge is **openness + prediction-vs-measured
> + per-CTA-honest methodology**, NOT a headline speedup, and NOT "first to run" (see §7).

## 2. Roofline FIRST (record before coding)

One code change: add `"nvfp4": 0.5625` to `roofline/model.py` `_BYTES`, and `"nvfp4"` to
`roofline/predict.py` `--precision` choices. Then decode AI = 2G/b:

| precision | b (B/elem) | decode AI (G=1) | GQA-8 AI | GQA-16 AI |
|---|---|---|---|---|
| FP16 | 2 | 1.0 | 8.0 | 16.0 |
| FP8 (v9) | 1 | 2.0 | 16.0 | 32.0 |
| **NVFP4 (v10)** | **0.5625** | **3.56** | **28.4** | **56.9** |

- **B300 FP4 ridge = fp4_tc_flops / hbm = 15e15 / 8e12 = 1875 FLOP/B.** Even GQA-16 NVFP4 (AI 57) is
  **~33× below the ridge** → roofline says **HBM-bound, FP4 tensor cores idle on intensity grounds** (and
  they're dark on the M≥64 shape grounds too, §1 T4). NVFP4 cuts the HBM floor ~3.55× vs FP16.
- **THE CAVEAT (the recorded prediction):** `model.py` has **no schedule term** — it is blind to the
  per-CTA/MLP wall, the dequant tax, and the L2. **Two-layer prediction:**
  - *Pure roofline:* HBM-bound, floor ~3.55× below FP16, ~33–57× below the FP4 ridge.
  - *Per-CTA-corrected (the real prediction):* on a B300 decode micro-bench at the same occupancy regime,
    NVFP4 will **NOT** convert 3.55× fewer bytes into 3.55× µs/tok — the kernel is per-CTA-bound, so the
    byte cut relieves a term it is far from. Expect: **capacity win (certain)**, **accuracy delta (the
    headline number)**, **latency win only** (i) as a v9-style fragile L2-resident load effect that
    *shrinks as G grows* and likely flips negative under flush, and (ii) in a genuinely bandwidth-bound
    long-context regime **iff one exists on B300** (T3, must be measured).
  - *Counter-prediction (the prize):* if at long context on B300 %HBM **climbs** toward the ceiling as the
    working set spills the ~126 MB L2 (the knee that was *absent* on T4), decode finally becomes
    bandwidth-bound on sm_103 → NVFP4's byte cut converts. Overturns the per-CTA expectation; first-class.

## 3. The single variable = KV storage format (FP8 1 B → NVFP4 0.5625 B)

Fork `kernels/v9_fp8/` → `kernels/v10_nvfp4/` (`nvfp4_attention.cu` + `binding.cpp`). Carry **byte-identical**
from v9/v8.7: the score-stationary inner loop, M-packing grid, split-KV partial, LSE merge,
`[B,H_q,N_q,S,*]` workspace, `choose_splits`, host launch. Change ONLY:
- Pool stores **packed E2M1 nibbles + per-16 E4M3 micro-scales** instead of E4M3 bytes.
- `dequant_e4m3` → `dequant_nvfp4` (unpack nibble → E2M1 value → × micro-scale × per-tensor scale → FP16)
  at the **same cooperative smem gather** v9 used, into the **same FP16 sK/sV smem**. **Fused per-tile,
  NEVER a full-cache prepass** (a prepass re-reads the cache and eats the byte savings — QServe). Smem
  stays FP16 → occupancy identical → clean byte-only A/B vs v9 *and* v8.7.
- `fa_kernels/paged.py`: `build_paged_kv_fp8` → `build_paged_kv_nvfp4` (packed pool + block-scale tensor)
  + `quantize_nvfp4`/`dequantize_nvfp4` (the dequant is the apples-to-apples oracle).
- New `nvfp4_attention()` API + `sdpa_reference_gqa_nvfp4` oracle (dequant the SAME bytes; `repeat_interleave(G)`).
- Register `v10_nvfp4` in `bindings/load.py` `_SOURCES` and `dispatch.py` `_MIN_CAPABILITY` (sm_103;
  keep a T4-emulated `(7,0)` fallback path: store 4-bit, unpack in-kernel — for no-rental correctness +
  capacity + accuracy, NOT a valid latency number, see §6).

### The asymmetric-precision recipe — REFRAMED by the v9 close-out research (the deliverable)
The old project thesis was "scores fragile → high precision; P·V safe (convex combo) → FP4." External
research (KIVI/KVQuant/QServe/KVTuner/Attn-QAT) **half-refutes** it. The corrected recipe:

| Tensor | Decision (v10 = FP4 *storage* + dequant; no FP4 matmul) | Why |
|---|---|---|
| **V (cache)** | **NVFP4**, **per-token** scale | Post-softmax weights are a bounded convex combination → FP4 storage error averages out; per-token is the research-favoured V granularity. Biggest single byte-saver. |
| **K (cache)** | **NVFP4 storage but per-CHANNEL scale + watch the score** | Score fragility is driven by **K** (KVTuner: K 8→4→2 bit blows up score error 13.9×). K's NVFP4 storage is OK *iff* the Q·Kᵀ dot is reconstructed at ≥FP16 from the dequant; per-channel scaling is the K lever, not per-token. |
| **Q** | FP16 | Tiny (O(d)); never degrade the query that gates softmax. |
| **scores / softmax / O / merge** | FP32 (hardware exp2 on B300) | v9/v8.7 already FP32-accum; dodges the FA-3 FP8-accum cliff. |

⚠️ **The compute asymmetry is OPPOSITE to what the project assumed — but it's a v11 concern, not v10.**
Research (Attn-QAT, on B200) keeps **P·V in BF16** and only does **Q·Kᵀ in FP4**, because quantizing P
(post-softmax) piles `cvt`/scale instructions onto the softmax bottleneck and *slows* the kernel. For v10
decode this is **moot** — there is no FP4 *matmul* (M<64, CUDA-core, dequant-to-FP16). The P-quantization
cost + the B300-2×-exp-might-flip-it question only bite when native FP4 tensor cores engage = **v11**
(multi-token, M≥64). Record it now so v11 inherits the framing. The accuracy ladder
(per-tensor → native 1×16 micro-scale → per-token V / per-channel K) is a paper figure; the
**FP4-everything ablation (show the softmax collapse)** justifies the asymmetry.

## 4. Correctness (Gate 1)
- Oracle: SDPA on the **same dequantized NVFP4 bytes** (`dequantize_nvfp4`), tol loosened (expect ~5e-2 or
  wider — report the distribution, don't just pass/fail). Separate deliverable: **logit/perplexity-proxy
  error vs the FP16-KV reference** (the real quant cost). Sweep **G∈{1,2,4,8,16}** × non-multiple N_k × d ×
  causal both ways; idle-warp G=3, multi-tile G=16. **Vary the seed** (v9 was single-seed=9 — a known gap;
  report a small distribution, not one point).

## 5. Benchmark (Gate 2 inputs) + deliverables
1. **Capacity:** NVFP4 vs FP8 vs FP16 footprint; max context/batch at fixed B300 HBM (~3.55× vs FP16,
   ~1.78× vs FP8). **State it by construction AND back it with `max_memory_allocated`** (v9 asserted 2×
   but never measured it — close that gap here).
2. **Accuracy:** logit error vs FP16 KV; the granularity ladder; **vs FP8** (the real question: *is FP4
   actually enough, or is FP8 the accuracy-safe floor?* — if FP4 degrades perplexity unacceptably, "FP8 is
   the floor; FP4 buys capacity at measured accuracy cost X" is the honest result).
3. **Latency, three regimes** (mirroring v9): (a) L2-resident micro-bench (expect capacity-only + a
   v9-style fragile load win that shrinks with G); (b) **same-session clock-matched `vs naive`** (NVFP4 ÷
   FP16, same packing — the ONLY trustworthy latency number; v9 learned cross-clock and re-quantized-oracle
   comparisons are confounded); (c) **B300 long-context past-~126 MB-L2** (the T3 knee hunt).
4. **Roofline read per regime:** achieved %HBM + distance-to-floor; state the model's blindness.
5. **sm_103 deltas:** the 2×-exp ablation (T2). (Native-FP4-MMA arm is **v11**, not here.)

### The Task-1 carry to B300 (the methodological centerpiece for T3)
Carry `bench/regime.py` to a **ROOT B300** (clock-lock + ncu; containerized rentals give
`ERR_NVGPUCTRPERM` — confirmed across hosts). Add a `v10_nvfp4` branch + the 0.5625 byte-count. B300's
~126 MB L2 needs ~**30×** larger WS to spill than T4's 4 MB → sweep N_k to **256K–1M** (the 288 GB +
NVFP4's 3.55× capacity is what *lets* you hold a past-L2 cache). Decisive plot: %HBM & L2-hit-rate vs N_k,
L2 crossing marked. Knee → bandwidth-bound regime found. No knee, flat per-CTA cap to 1M tokens →
per-CTA-bound on sm_103 at all reachable context. (Remember the counter-free `L2!` test is **one-sided** —
it confirms L2-streaming but a slow per-CTA-bound kernel never exceeds HBM peak even when L2-resident, so
**ncu's L2-hit-rate is what settles L2-vs-HBM** — root box mandatory.)

### Comparators (the real bar — "complement", not "beat", see §7)
- **FlashInfer `trtllm-gen`** — the default sm_103 decode backend; **ships NVFP4 KV decode today**
  (`out_dtype="nvfp4"`). The M-packing/production SOTA.
- **FlashMLA** — latent-KV decode (DeepSeek), ran GB300 day-0; the MLA bar (and the v11 bridge).
- **vLLM PagedAttention v2** — the no-M-packing CUDA-core floor.
Same B300, same shapes, clock-locked. They will likely win raw µs/tok (production-tuned); v10's value is
the **open roofline + FP4 recipe + per-CTA-honest methodology**, and the comparison's worth is showing
*where* the open kernel sits and *why* (the roofline read explains the gap).

## 6. Hardware
- **Record on B300/GB300 sm_103** (~$5.44/hr, spot ~$2.45) — **root/bare-metal for the Task-1 carry + ncu.**
- **B200 sm_100** (~$3.44/hr) = optional **dev rung** (tcgen05/TMEM/FP4-MMA ISA ports sm_100→sm_103) — iterate
  correctness + the FP4 path cheaply; NOT the destination.
- **T4-emulated NVFP4** (store 4-bit, unpack in-kernel) = no-rental correctness + capacity + accuracy
  fallback. ⚠️ **NOT a valid latency number** — software FP4 unpack is *more* ALU than v9's E4M3 (which
  already power-capped the T4 clock 1590→1350); the emulated-dequant tax confounds µs/tok. Trust only
  same-session clock-matched B300 ratios.
- **L2 size is UNCONFIRMED:** "192 MB" appears only in third-party aggregators, not NVIDIA docs. B200 is
  **~126 MB total / ~63 MB per-die-partition** (NVIDIA Tuning Guide); B300 is the same die → likely
  ~126 MB. A single decode CTA effectively sees ~63 MB. **Plan with ~126 MB; measure on first B300 rent.**
- **Toolchain:** CUDA 12.9+ / PTX 8.8 for sm_103 + NVFP4; `sm_100f` portable, `sm_103a` for the exclusives.

## 7. Novelty — defensible but NARROW (adversarially checked)
Dead claims (do not make them): "first to run GB300 decode," "first FP4/FP8 KV decode," "first
Blackwell-attention roofline." **FlashInfer's `trtllm-gen` is the default sm_103 decode backend and ships
NVFP4 KV decode now; vLLM published reproducible GB300 NVFP4 decode (Feb 2026); FA4 already roofs B200
prefill.** What **survives** (empty cell as of June 2026): *no **open, roofline-documented,
prediction-vs-measured DECODE study specifically on sm_103/B300 with an asymmetric FP4 KV recipe and a
confound-free per-CTA-vs-bandwidth methodology.*** FA4 is confirmed BF16-prefill / B200-focused with decode
pieces only in in-progress repo PRs (CLAUDE.md framing is accurate). **Frame: "complementing, not beating"
FlashInfer/FlashMLA, timestamped "as of June 2026."** The per-CTA-bound finding is the contribution.

## 8. Before v10 — the cheap experiment the v9 close-out demands (do this first)
The adversarial pass found the highest-value follow-up is **NOT v10** — it's a one-notebook re-test that
settles three open threads at once (is the residual limiter latency or occupancy? is "decode-schedule
CLOSED" premature? does FP8 flip negative because of flush or occupancy?):
> **Re-run v8.5 (double-buffer) and v8.6 (occupancy/ILP) through `bench/regime.py` PAST L2** (N_k≥32K,
> clocks locked, L2-flushed). They were measured NULL only at **L2-resident sizes** — the exact confound
> Task 1 exists to kill. Task 1 shows real 29%→70% headroom past L2 where latency-hiding could finally
> bite. If double-buffer lifts %HBM past L2, "decode-schedule CLOSED" reopens and the limiter is *latency*
> (→ deeper pipelining / persistent kernel), not pure occupancy. The kernels already exist; it's one
> notebook (`notebooks/v8_5_v8_6_pastL2_regime.ipynb`).
This is a T4 job (no rental beyond the root T4 already used for Task 1). Run it, record the result, *then*
start v10 — or fold it into v10's opening as the limiter-confirmation cell.

## 9. Traps (carried from v8/v9 + new)
- Causal mask uses **`i_q`** not `m_row`. GQA oracle uses **`repeat_interleave(G)`** not `repeat`.
  Idle-warp barrier participation. Fully-future split → (m=−inf, ℓ=0, O=0). Warp-uniform reductions on
  masked lanes via s=−inf/p=0.
- **Fused dequant, NEVER a prepass** (QServe). **Don't compare wall-times across clocks OR across the
  dequant power-cap** — same-session clock-matched `vs naive` only.
- **NEW — count 4.5 b/elem INCLUDING the per-16 E4M3 scale** in `kv_bytes` (regime.py) and `_BYTES`
  (roofline) — not 4 b, or the capacity/AI numbers are wrong.
- **NEW — NVFP4 block-scale layout** must be read coalesced alongside the nibbles, or a second uncoalesced
  stream pollutes the byte accounting.
- **NEW — the FP4 score path must NOT be a raw FP4 dot** (Q·Kᵀ feeds exp → error amplifies → softmax
  collapses). Reconstruct the score at ≥FP16 from the K dequant.
- **NEW — Blackwell tensor-core gate is M≥64 (M=128 for 100%), not M≥16.** N_q=1 decode (M=G<64) keeps the
  5th-gen cores dark → native FP4 compute is v11 (multi-token), not v10.

## 10. North star
The paper: the first **open**, roofline-documented, prediction-vs-measured FA **decode** study on
**sm_103**, with the asymmetric-precision FP4 KV recipe, vs FlashInfer/FlashMLA. v10 IS the paper's core.
Honest scope: production libs already run GB300 NVFP4 decode — the contribution is the **open roofline +
FP4 recipe + the confound-free per-CTA methodology**, not "first to run." The per-CTA-bound verdict (v9
Task 1) is the headline *insight*, and v10 turns it into the paper's spine.
