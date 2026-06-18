# Decision log

One entry per step: the bottleneck identified, options considered, what we chose, and what
would change on a different architecture. This is the "defend it in an interview" record.

---

## Step 0 — Foundation & hardware

**Hardware:** Colab **T4, Turing sm_75** (40 SMs, 320 GB/s HBM, 64 KB smem/SM, 2nd-gen tensor
cores ~65 FP16 TFLOPS / ~130 INT8 TOPS, MUFU for `exp2`). Build loop: Colab notebook +
`cpp_extension.load()` JIT. A100/H100 rented per-step.

**Why a versioned `kernels/vN_*` library instead of scripts:** the end goal is a mini-vLLM
inference engine that does `from fa_kernels import attention`. A stable public API
(`fa_kernels/__init__.py`) over swappable versioned kernels lets the engine pin a backend while
the journey keeps adding faster ones. The single `AttnConfig` keeps tile shapes / dtypes /
scale from drifting between kernel, test, bench, and roofline tool.

**What changes on another arch:** the build's `-gencode` and the roofline arch constants
(`roofline/archs.py`) are the only places hardware leaks in; everything above the dispatch
layer is arch-agnostic.

**Build/profile environment:** dev runs on **free Colab T4** for compile + correctness + timing
(`notebooks/colab_bootstrap.ipynb`: clone → clean stale JIT cache → `pip install ninja` → build →
pytest → bench). ncu **did run** on the current Colab runtime (it connected and wrote a
`.ncu-rep`), so Colab profiling is not universally blocked — but it's runtime-dependent and not
guaranteed, and `profiling/capture.sh` currently profiles the wrong kernels (it caught the
`torch.randn` RNG kernel, not qk/softmax/pv — needs a `--kernel-name` filter / launch skip). So
for a *reliable* ncu reading a **dedicated rented T4 (vast.ai ~$0.15/hr)** is still the safer bet;
free Colab covers build/test/bench. This is also why the plan rents A100/H100 per-step (Phase
2/3) rather than buying Colab Pro. Repo: `github.com/gkienpham-cmd/flashattention-cuda` (public,
so Colab `git clone` needs no token).

---

## Step 1 — Naive attention (the bandwidth wall)

**Bottleneck identified (predicted):** HBM bandwidth, at *every* head dim. Two sources of HBM
traffic: (1) the full `S = QK^T` matrix round-trips global memory ~4× (write in QK, read+write in
softmax, read in PV) — O(N²) bytes; and (2) — the dominant one — the matmuls **re-read their
operands from HBM with no reuse**: `qk_kernel` re-reads a full Q row and K row for every one of
the N² score entries (`naive_attention.cu:50-54`), `pv_kernel` likewise (`:116`). That is
**O(N²·d)** traffic, a factor of *d* larger than the S round-trip. Counting it,
`python -m roofline.predict --shape 1x8x2048x64 --precision fp32 --materialize-s --tile 1x1`
gives arithmetic intensity **~0.2 FLOP/byte** vs the FP32 CUDA-core ridge of **25.3** → deeply
memory-bound, predicted limiter **HBM** (~100% util, MMA ~1%, MUFU ~0%; lower-bound ~109 ms at
d=64, ~216 ms at d=128). Intentionally awful — this is the "before."

**Correction to an earlier number (kept honest, not hidden):** an earlier cut of the roofline
model assumed operands were read *once* from HBM, which gave "~15 FLOP/byte" at d=64 and a
spurious "d=128 flips to MMA-bound at ~30." Both were artifacts of ignoring the redundant operand
reads — the read-once assumption is actually closer to a *tiled* kernel than to the naive one.
With the operand-reuse term added to `roofline/model.py`, true-naive v1 is HBM-bound at AI ~0.2
for **all** d. (See [Step 2](#step-2--shared-memory-tiling-the-operand-reuse-win) for the model
fix and the tiled prediction.)

**v1's two real weaknesses, correctly placed:** (a) HBM traffic — redundant operand reads (big)
+ the S round-trip (smaller); HBM-bound now, attacked by Steps 2–4. (b) no tensor cores — this
bites *later*: only once a fused kernel reaches AI ~512 does the limiter become MMA, and then on
the weak 8 TFLOPS FP32 **CUDA cores** (fused d=64 ≈ 1.06 ms, d=128 ≈ 2.12 ms, both MMA-bound).
Step 6 (tensor cores) raises that ceiling. So (a) is a v1 problem; (b) is a post-fusion problem.

**Options considered:**
- **A1 three-pass (chosen)** vs A2 single fused-naive kernel — three separate launches make the
  S round-trip unmissable in `ncu` (each pass profiles independently), maximizing the teaching
  value and the drama of the later fused speedup.
- **B1 FP32 everywhere (chosen)** vs B2 FP16-in/FP32-accum — FP32 is a clean, exact-ish
  correctness anchor (atol/rtol 1e-4 vs SDPA) before precision enters as its own variable at v2.

**What we chose:** A1 + B1. One thread per output element, no tiling, no reuse — deliberately
naive so every later optimization has a measured "before."

**What changes on another arch:** an A100 (1.5 TB/s, 2 TB/s) raises the HBM roof ~5–6× but the
*shape* of the wall is identical — naive attention is memory-bound on every GPU; that
architecture-independence is the whole reason FlashAttention exists.

**MEASURED (Colab T4, 2026-06-18) — and the L2 plot twist:** 9/9 correctness tests pass vs SDPA
(atol/rtol 1e-4). Bench is ~0.03–0.06× SDPA (≈20–30× slower) across the sweep — the intended
"before." But the honesty check surfaced something: measured p50 lands *below* the cache-free
roofline lower bound at mid-N (2048×64: 85.6 ms measured vs 109 ms predicted = 0.78×; 2048×128:
0.77×), then *converges* to it at N=8192 (64: 1.01×; 128: 0.90×). You can't beat a true HBM
floor — so the floor's assumption is wrong: the model counts every redundant operand read as HBM
traffic, but the T4's **4 MB L2 absorbs most of them** while the working set fits. At N=8192 a
single head's K (2 MB) + head-interleaving overflows L2, the hit rate collapses, and measured
meets the cache-free floor. Takeaway: the AI≈0.25 cache-free roofline is the **worst case**,
realized exactly when L2 can't help (large N). This sharpens Step 2's thesis — explicit
shared-memory tiling makes reuse *guaranteed and N-independent*, so **its biggest win is predicted
at N=8192**, where L2 currently fails. (Model refinement worth considering later: an L2 hit-rate
term so the predicted floor tracks the measured mid-N speedup; logged, not yet built.)

---

## Step 2 — Shared-memory tiling (the operand-reuse win)

**Roofline model fix (done before predicting):** Step 1's prediction exposed that
`roofline/model.py` assumed each operand was read from HBM once — so it could not see the
redundant O(N²·d) operand reads that *are* the naive wall, and could not distinguish naive from
tiled. We added tile-aware operand traffic: for a tiled GEMM with output tile `tile_m × tile_n`,
each operand is read `M*N*K / tile` times (naive = 1×1 = re-read per output element). The flag
`--tile MxN` drives it; `--materialize-s` off remains the fused read-once ideal.

**Bottleneck identified (predicted):** still **HBM**, but far less badly. Tiling raises on-chip
reuse, cutting operand traffic ~`tile×`. Predictions at the tiles the v2 kernel will use
(FP32, N=2048):

| regime                         | d=64            | d=128           |
|--------------------------------|-----------------|-----------------|
| naive (`--tile 1x1`)           | AI 0.2, HBM     | AI 0.2, HBM     |
| **tiled (Step 2)**             | **AI 8.0, HBM** (`64x64`) | **AI 6.4, HBM** (`32x32`) |
| fused ideal (`no -materialize-s`) | AI 512, MMA  | AI 512, MMA     |

**The key prediction:** tiling is a ~30× traffic cut (AI 0.2 → ~6–8) but **does NOT cross the
25.3 ridge** — v2 stays HBM-bound at *both* head dims, because the S round-trip survives (only
online softmax, Step 3, removes it). So we predict a large measured speedup over v1 with the
limiter *unchanged*. That "still HBM-bound" is the motivation handed to Step 3.

**Nuance — bigger d means less reuse:** d=128 must drop to a 32×32 tile (a 64×64 FP32 Q+K tile
is 64 KB, leaving no room), so its reuse factor is smaller and its tiled AI (6.4) is *below*
d=64's (8.0). Head dim eats the shared-memory budget that buys reuse.

**Options / what we chose:** keep the three-pass structure (S still materialized) so Step 2
isolates the operand-reuse lesson from the S-elimination lesson (Step 3). Tile sizes adapt to d
to fit the 64 KB smem. Online softmax and fusion are deliberately deferred.

**Gate (cleared 2026-06-19):** Step 1 went green + benched + quiz-passed, so v2 shipped:
`kernels/v2_tiled/tiled_attention.cu`, FP32, three-pass, S still materialized. Tiles are picked
per head dim (64x64 @ d=64, 32x32 @ d=128) to fit the T4's 48 KB static smem; correctness is
**26/26 vs SDPA** at atol/rtol 1e-4, incl. non-tile-multiple shapes for the boundary guard.

**MEASURED (Colab T4, 2026-06-19) — prediction held on the limiter, missed on the location:**
v2 is **1.3–3.2× faster than v1** (clock-matched at SM ~300 MHz), and the **predicted limiter
(HBM) held at every shape** — tiling did not cross the ridge, exactly as predicted. It's still
0.04–0.11× SDPA (fused FlashAttention), because the S round-trip survives; that's v3's job.

But the Step 1 corollary — *"biggest win at N=8192, where L2 fails"* — was **wrong**. The measured
v2/v1 win *peaks at mid-N* (2048×128 = 3.20×) and *shrinks* at N=8192 (2.02–2.84×). The honest
post-mortem, separating two effects:
- **Magnitude (why ~3×, not the predicted ~30×):** the roofline says traffic drops ~30×, but
  *neither kernel sits on its roofline*. v1 runs faster than its cache-free floor (L2 help), and
  v2 runs 6–21× *above* its own floor (scalar loads, low occupancy from the 32 KB tile, PV-pass
  `__syncthreads` overhead). Those compress 30× → ~3×. The S round-trip caps the *roofline* win
  (AI 8, ~32×), not the realized one — so S is not why the win is small.
- **Shape (why it peaks at mid-N, not 8192):** operand and S traffic both scale as N², so the
  composition is N-independent and cannot drive a trend. The trend is purely off-roofline — v2's
  distance-from-floor is worst at the extremes (N=512 = 21× above, overhead-bound; N=8192 = 15×,
  load/sync-bound) and best at mid-N. The Step 1 corollary tracked only v1's L2 cliff and missed
  that **v2's own efficiency curve dominates where the win lands.**
- **The one predictable part:** d=128 beats d=64 at every N because the operand term tiling cuts
  is a larger traffic share there (operand:S ≈ 4:1 vs 1:1 at d=64).

**Lesson:** the roofline bounds *traffic*, not a kernel's distance from that bound, and realized
speedup is the ratio of two such distances — so a traffic-only model can call the limiter right
(it did: HBM) yet miss the magnitude and the location of the win. Full table + the pending ncu
`dram__bytes_read` read in [results.md](results.md#step-2--shared-memory-tiling-fp32-three-pass-s-still-materialized).

**Next (Step 3 — online softmax):** the surviving S round-trip is now the named target. Replace
the three-pass materialize-S structure with running max/sum so S never touches HBM — the model
predicts AI jumps to ~512 and the limiter finally crosses to MMA. Step 2's remaining gates before
it ships: read the v1-vs-v2 `dram__bytes_read` in Nsight, and pass the Step 2 quiz.

**What changes on another arch:** more shared memory (A100 164 KB, H100 228 KB vs T4 64 KB)
allows larger tiles → higher reuse → higher tiled AI, and async copy (Phase 2) hides the load
latency. The *ridge* also moves, so where "tiled" lands relative to it is arch-specific — but on
every arch, tiling-without-fusion stays bandwidth-bound because S still round-trips.
