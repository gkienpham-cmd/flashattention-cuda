# Interview prep — the conceptual chains

The "defend it out loud" companion to [decisions.md](decisions.md). `decisions.md` logs *what we
chose and why*; this doc holds the **reasoning chains** — the first-principles explanations I
should be able to recite cold in an interview. Each entry states a question, gives the
interview-ready answer, then unpacks the mechanism. Grows one concept at a time as the journey does.

Conventions for this doc:
- **Plain-text math only** (no LaTeX) — formulas in code blocks.
- Every chain ends with a one-line **say-this** summary: the compressed version to actually speak.

---

## C0 — The GPU memory hierarchy (the foundation everything rests on)

**Q: Sketch the memory hierarchy of the GPU you're optimizing for.**

A GPU is ~40 independent compute units called **SMs** (Streaming Multiprocessors) on the T4.
From fastest/nearest to slowest/farthest from an SM:

```
   registers     per-thread, on the SM         ~tens of TB/s    tiny (a few KB/thread)
   shared mem    per-SM scratchpad, on chip    ~8 TB/s          64 KB per SM on T4
   L2 cache      shared by all SMs, on chip     fast            4 MB
   HBM           off-chip main memory           320 GB/s        16 GB
   -------------------------------------------------------------------------
   the cliff:  on-chip  >>  off-chip   (~25x bandwidth drop at the HBM boundary)
```

- **HBM** = High Bandwidth Memory: the GPU's main DRAM (the T4's "16 GB"). Physically *separate
  silicon* from the compute chip, on the same package, joined by a wide bus. Big but far and
  bandwidth-limited (320 GB/s). All your `q,k,v` tensors live here.
- **Shared memory** = a small (64 KB/SM), fast, on-chip scratchpad. Two things make it special:
  (1) ~25x the bandwidth of HBM because it never crosses the cliff; (2) it is **programmer-managed
  and block-shared** — unlike a CPU cache that fills automatically, *you* write the code that
  copies specific bytes in, and every thread in the block can then read them. That cooperation is
  the entire lever this project pulls.

**Say-this:** "Registers in-hand, shared memory is the per-SM scratchpad, HBM is the far main
memory. There's a ~25x bandwidth cliff at HBM, so the game is to move data across that cliff as
few times as possible and reuse it on-chip."

---

## C1 — Arithmetic intensity and the roofline

**Q: What is arithmetic intensity, and why does it decide whether you're memory- or compute-bound?**

```
arithmetic intensity (AI) = FLOPs performed / bytes moved across the HBM<->chip boundary   [FLOP/byte]
```

It's a tug-of-war: computation in the numerator, HBM traffic in the denominator. The **roofline
model** compares AI to the hardware's **ridge point**:

```
ridge point = peak_compute (FLOP/s) / peak_bandwidth (bytes/s)   [FLOP/byte]

   AI <  ridge  ->  memory-bound  (starved waiting on HBM; compute units idle)
   AI >  ridge  ->  compute-bound (HBM keeps up; math throughput is the limit)
```

T4 ridge points: FP32 CUDA-core **25.3**, FP16 tensor-core **203**, INT8 tensor-core **406**.

For naive attention, the N-by-N terms cancel top and bottom, so **AI scales with d** (head dim),
not sequence length. Counting the *redundant* operand re-reads (each Q/K row re-fetched per output
element — see C3), v1 lands at **~0.25 FLOP/byte** at every d — far under the 25.3 ridge, so v1 is
pinned to the bandwidth roof. (An earlier read-once cut of the model said "~14-16"; that was the
bug C3 dissects — it described a *tiled* kernel, not the naive one.) Every Phase-1 optimization is
a campaign to raise AI (shrink the denominator) until we hit the compute roof, then a campaign to
raise the roof itself (tensor cores, then low precision).

**Say-this:** "Arithmetic intensity is FLOPs per byte moved across HBM. Compare it to the ridge
point — peak compute over peak bandwidth. Below the ridge you're memory-bound; the whole project
is raising AI until the bottleneck moves."

---

## C2 — The bandwidth wall and why tiling beats it

**Q: Why is naive attention memory-bound, and how does shared-memory tiling fix it?**

### The disease (naive v1)

The score matrix S = QK^T is N-by-N; entry `S[i,j]` is the dot product of Q row i with K row j.
Naive kernel: one thread computes one entry, reading its two operand rows straight from HBM.
Freeze on Q row i — which entries need it?

```
S[i,0], S[i,1], ..., S[i,N-1]   -> N entries, N different threads,
                                   each independently re-fetching Q row i from HBM
```

So **every Q row crosses the cliff N times** (same for every K row). The threads don't share what
they fetched, so the redundancy is pure waste. On top of that, the three-pass kernel writes the
whole S matrix to HBM and reads it back (~4x round-trip), which is the second source of traffic.

### The fix (tiling, Step 2)

**Tiling = chop Q/K/V into small blocks ("tiles"), process one block-pair at a time, and stage
each block in shared memory so a whole block of threads reuses it on-chip.** Assign one *block of
threads* to compute a tile of S, e.g. `S[0:64, 0:64]`:

```
1. Cooperatively copy Q rows 0..63 from HBM -> shared memory   (64*d values, ONE trip)
2. Cooperatively copy K rows 0..63 from HBM -> shared memory   (64*d values, ONE trip)
3. Every thread computes its S[i,j] reading Q,K rows from SHARED MEMORY (no HBM):
   64*64 = 4096 dot products served from the on-chip scratchpad.
```

Now Q row i is fetched from HBM **once** (step 1), then **reused** by all 64 threads in the block
that need it (step 3). N-trips-per-row collapses to one-trip-per-tile. Bytes-across-HBM drop,
FLOPs unchanged, so AI rises and we climb off the bandwidth roof. (Step 3, online softmax, then
lets us discard S after each tile and never write it to HBM at all — killing the round-trip term too.)

### Why not one gigantic tile?

Shared memory is a **fixed 64 KB per SM** — a hard ceiling, not a speed knob. Loading all of Q at
N=8192, d=64, FP32 would need `8192*64*4 = 2 MB`, ~32x too big to physically exist in the box. So
tiling is *forced*: the tile must fit Q-tile + K-tile + V-tile + softmax scratch inside 64 KB
(a 64x64 FP32 tile is 16 KB). Secondary cost: a bigger tile also lowers **occupancy** (fewer
blocks resident per SM to hide latency), so even a tile that fit could starve parallelism.

### The full chain (interview-ready)

> Naive: each of the N threads computing row i of S independently fetches Q row i from HBM -> N
> redundant cliff-crossings, plus the whole S matrix round-trips HBM. Tiled: the block fetches Q
> row i from HBM **once** into shared memory, then every thread that needs it **reuses** it on-chip
> -> one cliff-crossing. The single HBM read is amortized across many on-chip reuses. Bytes-across-
> HBM drop, FLOPs unchanged, arithmetic intensity rises, and we climb off the bandwidth roof.

The load-bearing word is **reuse**: reading once into shared memory only helps *because* many
threads then reuse it on-chip. Without reuse you've just relocated the redundancy.

**Say-this:** "Naive attention re-reads every Q/K row from HBM N times and round-trips the whole
score matrix — pure bandwidth waste. Tiling stages a tile in shared memory once and reuses it
across the block, so HBM bytes fall, AI rises, and the limiter moves off the bandwidth roof."

---

## C3 — Quantifying the operand-reuse win (and a model that was lying)

**Q: Put numbers on it. How much does tiling cut HBM traffic, and does it cross the ridge?**

This is C2 made quantitative — and a lesson in not trusting a model you didn't pressure-test.

### The traffic formula

Model HBM traffic as a tiled GEMM. For `C[M,N] = A[M,K] · B[K,N]` with output tile `tile_m x
tile_n`, each operand is re-read from HBM a number of times set by reuse:

```
A reads = M*N*K / tile_n     (each A row reused across tile_n output columns)
B reads = M*N*K / tile_m     (each B col reused across tile_m output rows)
```

Attention has two matmuls — QK^T (contraction K=d) and PV (contraction K=N_k). Summing both,
with a square tile T = tile_m = tile_n, total operand traffic is:

```
operand bytes ~= 4 * bh * N^2 * d / T  * nbytes      (the "/T" is the whole point)
```

- **Naive, T=1:** `4*bh*N^2*d*nbytes`. With FLOPs = `4*bh*N^2*d`, arithmetic intensity is
  `AI ~= 1/nbytes` -> **0.25 FLOP/byte** (FP32). Deeply memory-bound: every Q/K row re-read once
  per output element.
- **Tiled, T=64:** traffic drops ~64x. AI climbs to **~8** (d=64) — a ~30x intensity gain.

### The result, and why it still loses

T4 numbers (FP32, N=2048), predicted by `roofline/model.py`:

```
              d=64            d=128         limiter
naive (1x1)   AI 0.2          AI 0.2        HBM   (~109 / ~216 ms — intentionally awful)
tiled (Step2) AI 8.0 (64x64)  AI 6.4 (32x32) HBM  (still below the 25.3 ridge!)
fused ideal   AI 512          AI 512        MMA   (S never touches HBM)
```

Two things to say in the room:
1. **Tiling wins big but does not cross the ridge.** AI 0.2 -> ~6-8 is a huge bandwidth cut, yet
   both stay under 25.3, so v2 is *still HBM-bound*. Reason: tiling kills the redundant *operand*
   reads, but the score matrix S still round-trips HBM. Removing that needs online softmax (Step
   3). Tiling is necessary, not sufficient.
2. **Bigger head dim buys less reuse.** d=128 can't afford a 64x64 tile (a 64x64 FP32 Q+K tile is
   64 KB — the entire shared-memory budget), so it drops to 32x32, less reuse, *lower* tiled AI
   (6.4 < 8.0). Head dim competes with reuse for the 64 KB scratchpad.

### The model was lying (the meta-lesson)

The first cut of the roofline model assumed each operand was read **once** — so it reported "AI
~15" for naive v1 and even "d=128 is MMA-bound." Both false: that read-once assumption describes
a *tiled* kernel, not the naive one, and it was blind to the O(N^2 * d) redundant reads that are
the actual wall. We fixed the model (added the `/tile` operand term) before trusting its Step 2
prediction. **A roofline model that assumes ideal reuse cannot see the cost of bad reuse** —
which is exactly the cost you're trying to optimize. Pressure-test the denominator.

**Say-this:** "Tiling cuts operand traffic by the tile factor — naive AI ~0.25 FP32, tiled ~6-8 —
a ~30x bandwidth win. But it stays HBM-bound: tiling removes redundant operand reads, not the S
round-trip, so you don't cross the ridge until online softmax kills S. And watch the model itself
— the naive '0.25' only shows up once you stop assuming operands are read once."

---

## C4 — Prediction vs measured: why the tiling win was 3x, not 30x, and peaked in the wrong place

**Q: You predicted tiling cuts traffic ~30x. You measured ~2-3x, peaking at mid-N. Reconcile that.**

This is the Step 2 honesty payoff: the roofline called the **limiter** right and the **magnitude
and location** wrong — and the gap is itself the lesson.

### What was predicted vs what happened (T4, FP32, clock-matched ~300 MHz)

```
              predicted (roofline)      measured v2/v1
d=64          ~30x traffic cut, HBM     1.3x (N=512) .. 2.8x (N=2048) .. 2.0x (N=8192)
d=128         ~30x traffic cut, HBM     2.9x (N=512) .. 3.2x (N=2048) .. 2.8x (N=8192)
```

Limiter: predicted HBM, measured HBM at every shape. ✅ That part held.

### Why ~3x and not ~30x: neither kernel sits on its roofline

The roofline is a *lower bound on time* = a bound on traffic. Realized speedup is the ratio of two
*actual* runtimes, and both kernels miss their bound in opposite directions:

```
v1 runs FASTER than its cache-free floor  -> the L2 catches redundant reads (the Step 1 finding)
v2 runs 6-21x ABOVE its own floor         -> scalar loads, low occupancy (32 KB tile), PV __syncthreads
```

A model that bounds traffic cannot see either gap — L2 isn't in it, and kernel inefficiency isn't
in it. So 30x (the ratio of *floors*) compresses to ~3x (the ratio of *realized* times). **The S
round-trip is a red herring for the magnitude**: it caps the *roofline* win (AI 8 ≈ 32x), not the
realized one.

### Why mid-N, not N=8192 (the failed corollary)

Operand traffic and the S round-trip *both* scale as N^2, so the traffic *composition* is
N-independent — S cannot produce an N-trend. The trend is entirely off-roofline: v2's
distance-from-its-floor is a U in N (worst at the extremes, best in the middle):

```
N=512 :  21x above floor   work too small to hide launch/sync overhead
N=2048:  6-9x above floor   the sweet spot -> biggest v2/v1
N=8192:  15x above floor    un-vectorized loads + 128-chunk PV syncs scale badly
```

Step 1 predicted "biggest win at 8192" by tracking only *v1*'s L2 cliff. It missed that the win is
a *ratio*, and *v2*'s own efficiency curve dominates where that ratio peaks.

### The one part that was predictable: d=128 > d=64

At every N, d=128 wins more. The operand term tiling actually removes is a bigger share of traffic
at d=128 (operand:S ≈ 16:4 = 4:1) than at d=64 (≈ 4:4 = 1:1), so tiling has more to cut. This
*is* visible in a traffic model — and it shows up cleanly in the data.

**Say-this:** "The roofline nailed the limiter — still HBM — but predicted ~30x and I measured ~3x,
peaking at mid-N not the longest sequence. Because realized speedup is the ratio of two runtimes,
and neither is on its roofline: v1 is faster than its floor (L2 absorbs the redundant reads) and
v2 is well above its floor (scalar loads, low occupancy, sync overhead). A traffic-bound model is
blind to both, so it gets the limiter right and the magnitude and location wrong. The only N-trend
it *could* call — d=128 beating d=64, because the operand term it cuts is a larger traffic share —
it did call."

> **Update after the ncu read (C5).** The claim above that "the roofline nailed the limiter — still
> HBM" was itself wrong — the bench only showed v2 was *below the ridge*, not that it was
> *bandwidth-bound*. The DRAM counters (C5) show it never was. Keep C4's speedup reasoning; read C5
> for what the hardware actually did to the traffic.

---

## C5 — The ncu read: tiling didn't cut DRAM at all, and the score matrix is 99% of it

**Q: You finally read `dram__bytes_read` for v1 vs v2. Did the predicted ~30x traffic cut show up?**

No — and the *way* it failed is the best lesson in the project so far. Measured total DRAM reads:

```
            qk        softmax(S)   pv        total     v1/v2
N=512  v1   3.68 MB   49.83 MB     18.36 MB  71.9 MB
N=512  v2   3.11 MB   49.91 MB     13.49 MB  66.5 MB   1.08x
N=8192 v1   0.074 GB  26.33 GB     4.39 GB   30.80 GB
N=8192 v2   1.433 GB  25.53 GB     3.26 GB   30.22 GB  1.02x
```

Tiling cut **~0%** of DRAM traffic, at both shapes. Three things the traffic-only roofline could
not see:

### 1. The L2 owns the operands — at every N

The cache-free model charges every redundant Q/K re-read to HBM: ~558 GB of operand traffic for v1
at N=8192. Measured, v1's qk pass reads **74 MB**. The T4's 4 MB L2 catches the structured reuse
(a Q row is shared by all 256 threads in a block; K rows stay warm while streamed past many
blocks), so the operand mountain is a *cache illusion*. Tiling's whole job — make operands read
~once — L2 had already done. That's why v2 is no better, and on qk **19x worse** (1.43 vs 0.07 GB):
its tiled access pattern plus halved occupancy gets *less* L2 reuse than v1's plain streaming.

### 2. The score matrix S is ~99% of DRAM, not a floor under the operands

```
N=8192, v1:  softmax re-reads S ~12x = 26.3 GB  +  pv reads S again ~4 GB  =  ~30 of 30.8 GB
             Q/K/V operands  <1%
```

S is byte-identical in v1 and v2 (26.33 vs 25.53 GB) because tiling the *matmul operands* never
touches the *score matrix*. The model named the right villain — only online softmax removes S — but
mis-sized it ~100x: S isn't the floor, **S is the entire mountain.**

### 3. Nothing was HBM-bandwidth-bound

DRAM throughput tops out at ~35% (softmax), ≤15% at N=512. pv is compute-bound (~80% SM), qk is
latency-bound (~6% SM). "Below the ridge" (not compute-bound) was true; "bandwidth-bound" was the
model's guess and the hardware never honored it. **v2's speedup is therefore a compute/scheduling
win, not a bandwidth win** — DRAM traffic is flat.

### The meta-lesson (one level deeper than C3/C4)

C3 fixed the model to *count* redundant reads; C4 noted neither kernel sits on its roofline. C5 is
the punchline: **a traffic-bound roofline is blind to the cache that serves the traffic.** On a real
GPU the L2 silently converts "O(N²·d) redundant reads" into "read once," so an operand-traffic
optimization (tiling) buys nothing in bytes — the win, where there is one, is in *instructions and
scheduling*. Always confirm the predicted limiter with the DRAM-throughput counter before you
believe it.

**Say-this:** "I read the DRAM counters and tiling cut zero traffic — 1.02x at N=8192. The L2
already served the redundant operand reads the roofline charged to HBM, so v1's qk pass reads 74 MB,
not hundreds of GB; tiling had nothing to cut. And 99% of DRAM is the materialized S matrix the
softmax pass re-reads a dozen times — byte-identical in both kernels. Nothing saturated HBM
(≤35% throughput). The lesson: a traffic model is blind to the cache that serves the traffic, so
verify the limiter with the throughput counter — and the data makes online softmax, which deletes
that S, the only thing that matters next."

---

## C6 — Online softmax: I deleted 99% of DRAM traffic and the kernel got *slower*

Step 3 fused the softmax so the N×N score matrix S never touches HBM (running max/sum, two passes:
pass 1 → final `(m, l)`; pass 2 recomputes scores and forms O with no rescale). It is correct, it
provably removes S — and it is **3–7× slower than v2.** That contradiction is the whole lesson.

### 1. The S-elimination is real — measured without a profiler

Counters were blocked on every cloud rental (`ERR_NVGPUCTRPERM` — containers don't get
`CAP_SYS_ADMIN`), so I proved it with `torch.cuda.max_memory_allocated()` instead: at 8192×64, v2
allocates **+2164 MB** (the 2147 MB S matrix), v3 allocates **+17 MB** (just O + the `m,l` stats).
**125× less memory — that *is* S, gone.** A memory measurement substitutes for a traffic counter
when the algorithm's footprint is the thing in question.

### 2. Deleting it bought nothing — because S was never the *bottleneck*, only the *traffic*

Step 2 (C5) already showed the T4 is never HBM-bandwidth-bound (≤35% DRAM throughput; L2 owns the
operands). S was 99% of *DRAM traffic* — but DRAM traffic wasn't the constraint. **Removing the
biggest line item from a budget you weren't over doesn't make you faster.** This is the trap of
optimizing what you can *measure* (bytes) instead of what actually *binds* (here, the schedule).

### 3. The real limiter: occupancy/latency — and the roofline missed it a *third* time

Measured 2556 ms at 8192×64 vs a **17 ms MMA roofline floor → 151× above it.** Not compute-bound,
not bandwidth-bound (memory proof), not MUFU. The CUPTI trace (nsys-free, no counters) localizes it:
**pass2 = 88.6%** of runtime. The kernel is one-thread-per-row, so 192 of 256 threads sit idle, and
pass 2 re-reads K and V unstaged from global per key. That's an **occupancy + memory-latency** wall.
The roofline keeps mispredicting (HBM in C5, MMA here) for the same root cause: **it assumes an
efficient kernel and is blind to the schedule**, which on a real GPU is the dominant term.

### The meta-lesson

A correct algorithmic win (fuse softmax, kill S) can be a wall-clock *loss* if it's paid for with a
worse schedule on a machine that wasn't bound by the thing you fixed. v3's value is that it
*isolates and proves the mechanism* FlashAttention is built on; the speedup waits for v4 (single-pass
+ O-rescale + one-warp-per-row + staged operands), which keeps S off HBM *and* schedules like a GEMM.

**Say-this:** "Online softmax deleted 99% of DRAM traffic — I proved S is gone with peak-memory:
125× less than the tiled kernel, no profiler needed because counters are blocked on cloud rentals.
But it ran 3–7× *slower* than v2, because Step 2 had already shown nothing was bandwidth-bound — I
optimized a resource that wasn't the constraint. The CUPTI trace put 88% of the time in pass 2, and
measured was 151× above the MMA floor, so the real limiter is occupancy and memory latency from a
one-thread-per-row schedule — which the roofline can't see. The fix isn't more bandwidth saving;
it's v4's fused single-pass with proper warp-level parallelism."

## C7 — Fusing the schedule: I beat the tiled kernel, and learned what "compute-bound" really costs

Step 4 fused v3's S-elimination with v2's parallelism: one warp per query row, staged K/V,
register-resident O, single pass with the O-rescale. It beats v2 ~2× and v3 ~8× — the first time
S-off-HBM is *also* a wall-clock win. And it's still ~6× slower than SDPA. Both halves are the lesson.

### 1. The fix that worked: thread → warp

v3 mapped one *thread* to a query row — 31/32 lanes idle, uncoalesced per-row K/V reads, 151× off the
MMA floor, 88.6% of time stuck in pass 2. v4 maps one *warp* to a row: 32 lanes cooperate on the dot
product, the block stages each K/V tile into smem once, O lives in registers. CUPTI shows a single
kernel at 100% of CUDA time — the pass2/occupancy wall is structurally gone (no passes left).
Distance to floor: **151× → ~18×.** That ~8.5× is pure scheduling.

### 2. The O-rescale, finally paid

Single-pass means O accumulates while the running max is still moving, so every max climb rescales the
partial O by `exp(m_old−m_new)` *before* adding `p·V`. v3 dodged this by going two-pass (final `m,l`
→ no rescale). The N=16384 stability test (d=64 *and* d=128) is what proves the rescale is right
across the whole key axis — apply it to `l` but not `O`, or after the add instead of before, and only
long-N drifts.

### 3. Why it's *still* 18× off the floor: GEMV, not GEMM

The 17 ms floor assumes the 8.1 TFLOPS FP32 FMA peak. v4 scores one key at a time per warp: a 5-step
`__shfl` butterfly reduction + 2 FMAs. ~70% of the instruction stream is reduction, not math — it
can't saturate the FMA pipes. The win came from fixing *occupancy*; the remaining gap is *FMA
utilization*, a different axis. Roofline missed the magnitude a 4th time (17 vs 292 ms) for the same
reason it always does: flops/bytes can't see reduction overhead.

### The meta-lesson

"Compute-bound" on the roofline doesn't mean "near peak FLOPs." A kernel can be compute-bound *and*
18× slow if the compute is shaped wrong (GEMV reductions instead of GEMM FMAs). Fixing the schedule
(v3→v4) and fixing the math-shape (v4→v5 tensor cores) are two separate optimizations; you need both.

**Say-this:** "v4 fused single-pass with one warp per row, staged K/V, register-O and the O-rescale.
It beats v2 ~2× and v3 ~8× — S off HBM finally wins in wall-clock — and CUPTI shows one kernel, so
v3's occupancy wall is gone: 151× → 18× off the floor. But it's still ~6× slower than SDPA, because
scoring one key per warp via shuffle reductions is GEMV-shaped — ~70% reduction overhead, never near
FP32 FMA peak. Step 5's tensor cores fix the shape *and* raise the ceiling."

---

## C7.5 — Tensor cores: the GEMV→GEMM fix (v5 WMMA)

*(numbered C7.5 to sit between Step 4's C7 and Step 6's C8 without renumbering C8–C10. **Status: the
kernel is correct and wired; the prefill speedup was never measured** — `step5_run_of_record.ipynb`
is unexecuted. So this chain is the *prediction and the structural argument*, with the headline number
flagged as outstanding. Don't claim a v5 speedup in an interview — claim the *fix* and the *predicted*
8× headroom, and that the measurement is the next thing to capture.)*

Step 4 diagnosed v4's wall: compute-bound but ~18× off the floor because scoring is GEMV-shaped
(shuffle reductions, not FMAs). v5 keeps v4's entire fused schedule and changes exactly one thing —
both matmuls become tensor-core WMMA (FP16-in/FP32-accum).

### 1. Why tensor cores attack *both* axes at once
v4's gap had two parts: a low ceiling (8.1 TFLOPS FP32) and bad shape (reductions, not FMAs). Tensor
cores fix both in one move — the FP16 peak is ~65 TFLOPS (**8× ceiling**) *and* one `mma_sync` is a
16×16×16 GEMM tile, so the reduction over `d` happens *inside* the core with no `__shfl`. The roofline
floor drops exactly 8× (16.97 → 2.114 ms at 8192×64) and the ridge moves 25.3 → 203.1 FLOP/byte —
still compute-bound, just against a far lower floor.

### 2. The opaque-fragment tax (the price, and the new risk)
A WMMA accumulator's 256 results are scattered across the warp's lanes in an **un-indexable** layout.
v4 kept O in named registers and softmaxed there; v5 *can't* — so it forces S through smem: `QK → store
S to smem → row-softmax in smem (each lane owns a whole row, no shuffle since the GEMM already reduced)
→ write P back as half → reload as fragments for PV`. The running FP32 O (`oRun`) also lives in smem so
the O-rescale folds into the PV accumulator (`C += A·B`). **That smem round-trip is the new structural
risk** — it (or the small 16×16 tiles under-filling the warp at d=64) could become the next limiter
even as the FMA wall falls. Which one wins is exactly what the un-run bench would have told us.

### 3. What's proven vs what's pending
Proven: **correctness** at the first loosened tolerance (2e-2, FP16-in) — 17 cases including the
N=16384 O-rescale/FP16-drift stability test at d=64 and d=128. Pending: the speedup. The honest state
is "the fix is in and correct; the measurement is owed."

### The meta-lesson
This is the cleanest example of *predict-then-don't-measure being a visible gap*, not a silent one.
The roadmap's whole point is prediction-vs-measured; v5 has the prediction (8× lower floor, GEMM
shape) but the measured half is missing, so the step is honestly **partial** — correct and predicted,
not yet confirmed. The roofline has mispredicted magnitude four straight times (Steps 1–4); v5 is the
first where we *don't yet know* if it's wrong, and saying so is the deliverable.

**Say-this:** "v5 keeps v4's fused schedule and swaps both matmuls to Turing WMMA tensor cores,
FP16-in/FP32-accum. That attacks v4's gap on both axes — 8× higher ceiling and GEMV→GEMM so the
reduction lives inside the MMA, no shuffle. The cost is the opaque-fragment tax: WMMA accumulators are
un-indexable, so softmax can't stay in registers — S goes through smem, which is the new candidate
limiter. It's correct at 2e-2 FP16 tolerance, but — being honest — I never captured the prefill bench:
the run-of-record notebook is unexecuted, so I have the *predicted* 8× floor drop but not the measured
speedup. That measurement is the next thing to run."

---

## C8 — Decode ≠ prefill: split-KV (the v6 keystone)

*(measured 2026-06-27, T4 sm_75; design + B300 arc in `docs/b300-decode-research.md`)*

### Why prefill kernels die at decode
v1–v5 parallelize over query rows: grid `(ceil_div(N_q, rows), B·H)`. At decode `N_q = 1` that is
`(1, B·H)` — a handful of blocks, the SMs idle. Both matmuls are `M = 1` (GEMV), so **tensor cores idle
too** — v5's whole advantage evaporates. The problem isn't bandwidth or FLOPs; it's that there is **no
parallel work to spread** across the SMs.

### The decode roofline
One step, `N` keys: work `= 4Nd` FLOPs, traffic `= 2Nd·b` bytes (read K and V; Q, O are `O(d)`,
negligible). **`AI = 2/b`, independent of `N`** — pure memory-bound, hundreds× below the ridge. Two ways
to climb: lower `b` (FP8 → AI 2, NVFP4 → AI ~3.5) or **share KV across heads** (GQA/MLA → `AI = 2G/b`).

### Split-KV (Flash-Decoding) — the fix
Partition the `N` keys across blocks; each block runs online softmax over its chunk and emits an
**unnormalized** partial `(O, m, ℓ)`. A merge kernel recombines across splits with the *same*
log-sum-exp algebra online softmax already uses across keys:
`m = max_s m_s ; ℓ = Σ_s e^{m_s−m} ℓ_s ; O = Σ_s e^{m_s−m} O_s ; O /= ℓ`. Fills the SMs at `N_q = 1`
without changing the result. (On B300 the merge fuses on-chip via a 2-CTA cluster + DSMEM; on T4 it's a
second kernel.)

### What the T4 measured (the honest ceiling)
v6 **beat the naive `N_q=1` loop 5.7–8.2×** and **torch SDPA 1.5–3.3×** (non-causal) — the split-KV win
is real. But it reached **only ~9–15% of HBM bandwidth (≈7–9× above the roofline floor)**. So even after
split-KV the limiter is *still* occupancy/launch, not bandwidth: at BH=8 the grid is only 64–80 blocks on
40 SMs (~2/SM), plus a two-kernel launch and an under-occupied merge. The lesson repeats v3/v4: the
roofline named the *location* (HBM) but not the magnitude, because it can't see the schedule. **Practical
consequence:** the next lever is *more occupancy* (bigger batch / GQA M-packing → `AI = 2G/b`), not fewer
bytes — FP8/NVFP4 KV only pays once you're actually bandwidth-bound, and we're 8× short of that wall.

*(Gotcha worth stating: causal decode with the query at row 0 degenerates to 1 key — SDPA short-circuits
it, a split-KV kernel doesn't. Realistic causal decode puts the query at `N_k−1`, which equals the
non-causal full-cache scan. Don't quote causal-decode-at-row-0 speedups.)*

### Forward pointer (v10): asymmetric precision
The two matmuls have **opposite FP4 tolerance**: `P·V` is a convex combination (`P ∈ [0,1]`, sums to 1)
→ FP4 error *averages out*, **safe**; `Q·Kᵀ` feeds `exp` → error amplifies (`δ → e^δ`) and logits carry
outliers → keep **MXFP8 + outlier residual**. That asymmetry is the v9 headline.

**Say-this:** "Decode is `N_q = 1`, so prefill kernels under-occupy the SMs and the matmuls go GEMV —
even tensor cores idle. v6 splits the KV axis so each block owns a chunk and emits a partial `(O, m, ℓ)`;
a merge recombines them with the same LSE algebra as online softmax. On the T4 it beat a naive `N_q=1`
loop ~6–8× and SDPA ~1.5–3×, but only hit ~12% of HBM — so it's still occupancy-bound, not bandwidth-
bound. That's the tell: decode AI is `2/b`, memory-bound and N-independent, so the next win is occupancy
(GQA `2G/b`) then fewer KV bytes — which is why the B300 headline (v10) is FP4 KV, not a better GEMM."

---

## C9 — The reorder: occupancy before bytes (and the batch caveat)

*(deep-research close-out 2026-06-27; full synthesis + math + 5 diagrams in `docs/decode-replan.md`)*

### The decision
v6 measured 12% of HBM → **occupancy-bound, not bandwidth-bound.** FP4 KV is a *bytes* lever, and cutting
bytes multiplies the HBM term — which isn't binding when you're 8× off the wall. So the plan **reorders**:
the occupancy lever **GQA M-packing (v8)** moves *ahead of* the byte levers **FP8 (v9) / NVFP4 (v10).**
Not a pivot — the decode/B300/FP4 thesis is intact; only the *order* changed.

### Why GQA M-packing leads (one change, three wins)
Pack the `G` query heads of a GQA group into the CTA's `M` dim: (1) **occupancy** — `+G×` useful work per
block fills the SMs; (2) **intensity** — `AI = 2/b → 2G/b`; (3) **reuse** — one KV read serves `G` heads
instead of `G` reads. And `M = G > 1` turns the decode GEMV back into a small GEMM, so **tensor cores
re-engage** (the v5 path, recovered in the decode regime). It wins in *both* batch regimes, so it leads
regardless of deployment.

### The batch caveat (the honest qualifier)
"Occupancy before bytes" is **batch-conditional.** `choose_splits` self-disables (`num_splits→1`) once
`base_blocks = BH ≥ 2·num_sm`, so at production batch (T4 `BH≥80`, B300 `BH≥320`) batch *alone* fills the
SMs and decode becomes genuinely bandwidth-bound — there *bytes-first* is right. v6 only benched `B=1`
(the worst corner), so the large-batch end-state is **predicted, not measured** → v7 adds a `--batch`
sweep. On a 160-SM B300 the `M=1` starvation is *worse* (needs `B≥40` to fill).

### Claim discipline (don't let the headline age badly)
"Beat FA4 in decode" is true of the **published** FA4 (a BF16 prefill/training kernel, <0.8% MMA util at
`M=1`) but FA4 is *acquiring* a decode path (Modal upstreamed split-KV + GQA-packing). FlashInfer (vLLM
GB300) and FlashMLA (SGLang GB300, DeepSeek-V4 day-0) **are** B300-proven. So the defensible contribution
is **"an open, roofline-documented, asymmetric-precision FP4 split-KV decode kernel, measured vs the real
bar (FlashInfer/FlashMLA),"** plus the prediction-vs-measured methodology — *not* "we beat FA4."

### The B300 fact that frames everything
B300 HBM bandwidth is **flat 8 TB/s** vs B200 (verified — only capacity grew 192→288 GB, because 8-Hi→12-Hi
stacks add layers, not pins). So a memory-bound decode kernel's *only* levers on B300 are **fewer bytes
(FP4), more occupancy (GQA), faster exp (2× MUFU.EX2)** — exactly the v8→v11 set.

**Say-this:** "v6 came out occupancy-bound at 12% of HBM, so I reordered: GQA M-packing before low-precision
KV. GQA packs the group's query heads into M — it fixes occupancy, raises AI to `2G/b`, reads KV once, and
re-engages tensor cores, all in one change, and it wins at any batch. FP4 KV is the headline but it only
pays once you're near the bandwidth wall, and on B300 bandwidth is *flat* vs B200 — fewer bytes is the only
bandwidth lever left. The honest caveat is that 12% is a single-stream number; at serving batch the grid
fills itself, so I'm adding a batch sweep to measure the crossover. And I frame the result as an open,
roofline-documented decode kernel vs FlashInfer/FlashMLA — not 'we beat FA4,' which is a prefill kernel."

---

## C10 — Paged KV + the measurement that *broke* the reorder's premise (v7)

*(v7 — paged KV gather + decode-harness fixes. The step that tested C9's central claim against data —
and the data said no. Measured 2026-06-27, vast.ai T4: 51/51 correct; the `--batch` sweep refuted the
predicted occupancy→bandwidth crossover.)*

### Why a whole step for "plumbing"
A real KV cache isn't contiguous — it's a pool of fixed-size pages plus a per-sequence **block table**
mapping logical token positions to physical pages (vLLM's layout). To read logical key `j`:
`lb = j/page_size; pb = block_table[b][lb]; off = j%page_size; pool[pb*page_size+off, h]`. v7 swaps v6's
one contiguous-offset line for that lookup and **changes nothing else** — split-KV partial + LSE merge
are byte-identical. It's the import surface my from-scratch mini-vLLM consumes, and the foundation v8–v11
ride on. I gave it its own API (`paged_attention`, per-sequence table, vLLM-faithful) instead of
overloading the dense `attention(q,k,v)`.

### The roofline didn't move — on purpose
`AI = 2/b = 1.0`, unchanged. The gather adds `O(N_k/page_size)` int32 index reads — ~0.1% of the KV
bytes — so v7 is **byte-neutral and occupancy-neutral by construction.** That's the point: I deliberately
did NOT attack the limiter this step. Isolating one variable means any wall-clock change is the gather's,
and it sets up v8 (GQA, the actual occupancy lever) cleanly.

### The deliverable was a measurement — and it refuted the prediction
C9's load-bearing claim — "occupancy before bytes, but batch-conditional" — rested on a *code-trace*:
`choose_splits` self-disables (`num_splits→1`) once `BH ≥ 2·SM` (=80 on T4), so batch alone *should*
fill the SMs and drive `%HBM` from 12% toward saturation. v7's **`--batch` sweep** (B=1→64, BH=8→512 at
fixed N_k=8192) measured it. **`%HBM` stayed flat at 9.4–12.4% the whole way** — even at BH=512
(`num_splits→1`, 512 blocks = 12.8/SM, far past full occupancy), per-token cost is constant. **The
crossover does not exist.** Batch over-fills the grid and bandwidth utilization does not move.

### Why batch can't help — the real limiter is per-CTA (code-verified)
Two structural caps that batch *replicates* instead of curing: (1) **smem caps residency at 2 blocks/SM**
— `sK+sV = 32 KB`/block, T4 has 64 KB/SM, so more blocks just means more *waves* at the same per-SM
occupancy → linear time → flat %HBM; (2) **at `N_q=1` only 1 of 8 warps computes** (warp `w` owns query
row `w`), so ~2 active compute-warps/SM can't hide the per-key shuffle-reduction latency — the GEMV wall
again. So decode here was never grid-starved (split-KV already gives 80 blocks at B=1); it's bound by
what happens *inside* each block.

### What it does to the thesis: sharpens the reorder, kills the hedge
The reorder (GQA M-packing before bytes) **survives and gets a better reason.** GQA leads not because it
"fills the SMs" — batch does that and it doesn't help — but because `M = G > 1` activates **G
compute-warps/block** and turns the GEMV into a tensor-core GEMM with KV read once: it hits the actual
limiter. And the **batch-conditional hedge is refuted**: bytes-first (FP8/FP4) is premature at *all*
batch sizes, because v7 never gets within ~8× of the bandwidth wall at any BH — not just at B=1. The
honest update to C9: "occupancy before bytes" → "**GEMV→GEMM (per-CTA efficiency) before bytes**, at all
batch sizes."

### The causal-decode bug I inherited and fixed
v6's causal decode was degenerate: query at row 0 with mask `j > i` attends *only key 0* (SDPA
short-circuits it, so the bench rows were meaningless). v7 adds a **query-offset** — place the decode
query at logical `N_k−1`, so causal == the full-cache scan. Now the causal rows do real work and time
≈ the non-causal rows. Small change, but it turns six garbage bench rows into honest ones.

### How I prove the gather is correct without a profiler
Scatter a dense KV into **shuffled** physical pages + the matching block table, run v7, compare to SDPA
on the *original* dense KV. The shuffle is the test: a kernel that ignored the block table (read
contiguously) would grab the wrong tokens and fail. Cases include a `page_size` that doesn't divide
`N_k` (partial last page) and the square-shape `num_splits→1` regression.

**Say-this:** "v7 is paged KV — the block-table indirection a real cache needs — on top of v6 with one
changed line, so it's byte-neutral by design. I built it to *measure*, not to go faster. The batch sweep
was supposed to confirm the occupancy→bandwidth crossover; instead it refuted it — %HBM stayed flat at
~10–12% from BH=8 all the way to BH=512, even with 12.8 blocks per SM. The reason is per-CTA, not the
grid: 32 KB of smem caps me at 2 blocks/SM, and at N_q=1 only one of eight warps actually computes, so
adding batch just runs more copies of an inefficient block. That's a *better* argument for GQA
M-packing — M=G turns the GEMV into a GEMM and lights up G warps — and it kills the 'bytes-first at large
batch' hedge: I'm 8× from the bandwidth wall at every batch size, so cutting bytes is premature
regardless. Correctness is a shuffled-pool gather vs dense SDPA — ignore the block table and you read the
wrong tokens and fail."

### Close-out addenda (deep-research, 2026-06-27)
- **The %HBM number is honest (fp16).** A skeptic asks: the bench prints `precision=fp32` and allocates fp32
  K/V — isn't your 10–12% really 25%? No. The kernel *casts to half* before reading
  (`paged_attention.cu:274–276`); every HBM KV load is `__half` = 2 bytes, and the denominator uses 2 bytes/
  elem, so it matches what the kernel actually streams. The header is a stale input-contract label, not
  bytes-on-the-wire. (The "2× undercount" claim died 3/3 in the adversarial pass.)
- **SDPA beats me at batch — and I know exactly why.** I win only at B=1 (2.65×). By B=8 torch SDPA overtakes
  (~0.5×). It's not that I got slower — my per-token cost is *flat* (I'm SM-saturated at B=1 via 10 splits =
  80 blocks = the 2-blocks/SM cap), while SDPA's per-token cost collapses ~4.5× as batch fills its grid. At
  B=1 I win *because* SDPA under-fills the grid even worse than my split-KV does. That serving regime (B≥8) is
  what v8 has to reclaim.
- **The tensor-core gate for v8.** Packing G heads into M only reaches tensor cores at **M≥16** (FlashInfer:
  "tensor core instruction m minimum rows is 16"). Llama-3-70B has G=8 < 16, so a single group needs padding
  to 16 (half row-util), or pack two groups, or keep CUDA-core scoring for QK. That trade is the v8 ablation,
  and it's why I target **sm_80 (A100)** — the m16n8k16 MMA atom + cp.async, not Turing WMMA.
