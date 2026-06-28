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

## C11 — GQA M-packing: turning the GEMV into a GEMM (v8)

*(v8 — the reorder's payoff lever. Phase 0 roofline + Cut 1 CUDA-core code complete; GPU gate pending.
Built STAGED: Cut 1 = CUDA-core M-pack on T4, Cut 2 = sm_80 tensor cores + the M≥16 ablation.)*

### The one-line idea
A GQA model has `H_q` query heads but only `H_kv = H_q/G` KV heads — the `G` query heads of a group all
attend the **same** KV head. v7 ran one (batch, query-head) per block, so the G heads of a group each
re-read that KV head and each used only warp 0 (the 1-of-8 wall). v8 **packs the G query heads into the
score GEMM's M dimension**: one block, one KV head staged **once**, G query rows computed against it. So
`G` warps light up instead of 1, KV bytes drop by `G`, and `AI = 2/b → 2G/b`. It's the smallest change
that hits the limiter v7 *measured*.

### Why this is the right lever (not bytes, not the merge)
v7 proved decode is per-CTA-bound at every batch size — flat ~10–12% HBM, BH=8→512. So the binding
constraint is *inside* the block: 1-of-8 warps + 32 KB smem capping residency at 2 blocks/SM. Cutting
bytes (FP8/FP4) multiplies a term I'm already 8× away from reaching. Fusing the merge doesn't touch the
GEMV. M-packing is the only move that converts the wasted 7 warps into work and reads KV once.

### What stays identical — and what the one change is
I forked v7 and changed *only the index math*. The cooperative paged gather, the online-softmax core with
the O-rescale, the LSE merge, and `choose_splits` are byte-for-byte. The change is the warp→work mapping:
the block's z-dimension now iterates **KV heads** (`B·H_kv`, G× fewer), and packed row `m_row =
blockIdx.x·8 + warp` decodes to `(g_local = m_row/N_q, i_q = m_row%N_q)` → global query head `h_q =
h_kv·G + g_local`. At decode (N_q=1, M=G) warp `w` simply owns query head `h_kv·G + w`. The KV tile is the
shared head, read once.

### Two traps I had to get exactly right
1. **The causal mask uses the query *position* `i_q`, not the packed row `m_row`.** Packed rows interleave
   G heads at the *same* position; masking on `m_row` would give each head a different (wrong) cutoff.
2. **Workspace + merge stay query-head-shaped `[B,H_q,N_q,S,*]`** while the pool gather is KV-head-shaped.
   The G active warps write distinct `h_q` slices (no collisions); mixing the two head counts drops G−1 of
   every G outputs. And the correctness oracle must expand KV with **`repeat_interleave(G)`** (head `h_kv`
   → query heads `[h_kv·G, h_kv·G+G)`), not `repeat`/tile — that mapping has to match the kernel exactly.

### The roofline prediction (and what it deliberately does NOT say)
`AI = 2G/b` rises exactly `G×` and the HBM floor drops `G×` (0.132→0.016 ms at G=8 on A100). But the
limiter **stays HBM** — A100's fp16 ridge is 153, so even G=8 (AI=8) and G=32 (AI≈32) are far below. So I
do *not* claim "v8 becomes compute-bound." The model has no schedule term (it's been magnitude-wrong 5
steps straight); the real win it can't see is per-CTA efficiency — G warps + KV-once should move me much
closer to that now-8×-lower floor. The deliverable is the measured µs/tok drop and **reclaiming SDPA at
B≥8** (where v7 lost 0.5×), not the floor itself.

### Why CUDA-core first (the staging)
The kickoff said target sm_80 with `mma.m16n8k16` + `cp.async`. But a T4 can't run those, and my whole
cheap correctness loop is on T4. So Cut 1 keeps both matmuls on CUDA cores — it still gets G-warps +
KV-once + `AI=2G/b`, just no tensor-core GEMM — which isolates M-packing as one variable and validates it
for ~$0.15/hr. Cut 2 adds the tensor-core path on a rented A100, where the M<16→M≥16 gate forces the
three-way ablation (pad-to-16 / pack-two-groups / CUDA-core-QK). Tensor-core uplift is then measured
*against the CUDA-core M-packed baseline* — a clean two-variable decomposition.

**Say-this:** "GQA gives me G query heads sharing one KV head. v7 wasted that — each head re-read the KV
and used one of eight warps. v8 packs the G heads into the GEMM's M dimension: stage the KV head once, run
G rows against it, so G warps work and KV bytes drop by G — AI goes 2/b to 2G/b. That's not a guess about
the bottleneck; it's the exact thing v7 *measured* — per-CTA-bound, 1-of-8 warps. I forked v7 and changed
only the warp-to-head index math; the gather, softmax, and merge are byte-identical. The roofline says AI
rises G× but I stay HBM-bound — A100's ridge is 153 — so the headline isn't 'compute-bound now,' it's
moving from 10% of the floor toward it and reclaiming the batch regime SDPA took. I built it CUDA-core
first so I could prove M-packing on a cheap T4 before renting an A100 for the tensor-core GEMM."

### Cut 1 measured (2026-06-28, Colab T4) — the thesis landed *without* tensor cores
The G-sweep is the proof: `vs no-pack` (v8 ÷ v7 on the identical workload) is **8.59×/8.71× at G=8** and
tracks ~`G×` all the way up — i.e. packing G heads buys G× wall-clock, exactly what `AI=2G/b` implies, on
plain CUDA cores. And the serving headline: at G=8, v8 beats torch SDPA **6–10× at every batch from 1 to
64**, where v7 *lost* (0.3–0.5× past B=8). So the reorder ("GEMV→GEMM before bytes") is now measured, not
just argued. **The honest asterisk:** `%HBM` never exceeds ~11% even at the best G — I'm still
per-CTA-bound, not bandwidth-bound. M-packing closed most of the per-CTA gap but left ~9× of headroom,
which is exactly what Cut 2's tensor-core `M=G` GEMM (and only later, the FP8/FP4 byte cuts) goes after.
This is the **first step in six where the roofline's number was directionally right** — `AI=2G/b` predicted
the speedup *magnitude* (~G×), even though its absolute HBM floor stays unreached because the model still
has no schedule term. Say it crisply: "I predicted G×, I measured ~G×, and I can tell you precisely why
I'm *still* not at the bandwidth wall — that gap is the next kernel, not hand-waving."

### Cut 2a (Turing WMMA) — the tensor-core fix BACKFIRED on decode, and why that's the real result
I then tried the obvious "next kernel": move the M=G score matmul onto tensor cores, the same GEMV→GEMM
fix that won for v5 *prefill*. I predicted it would beat Cut 1. **It lost — 1.4–1.6× slower (clock-
normalized), at every G, even at G=16/32 where the 16×16×16 WMMA tile is completely full.** So it wasn't
the pad-to-16 waste; it's structural. The reason is the thing v5 already named — the **opaque-fragment
tax**: WMMA accumulators are un-indexable, so softmax has to round-trip through smem (QK→smem→row-softmax→
write P as half→reload→PV) with extra barriers. For *prefill* that overhead is amortized over a big M-tile;
for *decode* M=G≤16 is tiny, so the GEMM is too small and the fixed fragment/smem overhead dominates. The
lean, register-resident CUDA-core GEMV (Cut 1) just wins. The deep point: **Cut 1's 8.6× came from packing
G query rows into G active warps that read KV once — a scheduling/occupancy win — NOT from needing tensor
cores at all.** Tensor cores were the prefill tool I wrongly assumed transferred. Two things keep me honest:
the A/B's two runs read different throttle clocks (so ~1.5× of the raw gap is clock — but %HBM, a clock-
robust ratio, is 2–3× lower, and the sign is the same in 12/12 rows), and my Cut 2a is a 1-warp/block
schedule whose KV load is under-fed, so a tuned WMMA kernel would narrow but (the small-M argument says) not
flip it. **Say-this:** "I predicted tensor cores would help and measured that they hurt — decode's matmul is
too small to pay for the WMMA fragment overhead, so the CUDA-core GEMV is the right decode primitive. That's
the sixth time the roofline-blind-to-schedule story bit me, and the cleanest evidence that v8's win was
occupancy, not compute. Whether an A100 mma.m16n8k16 + cp.async kernel can overturn it is an open question I'd
spend rented hours on only because the answer is publishable either way — not because the decode thesis needs it."

### The A100 probe settled it — WMMA loses HARDER on Ampere (the smoking gun)
Instead of blind-writing the hard mma+cp.async kernel on a weak hypothesis, I spent ~10 min of A100 rental
on a *probe*: run the kernels I already have on Ampere and ask whether the verdict flips. It didn't — WMMA
went from 1.2–3.3× slower (T4) to **1.8–4.6× slower (A100)**. The decisive number isn't the ratio, it's the
*scaling*: WMMA barely moved from T4 to A100 (G8/d128, 42→39 µs/tok) even though the A100 has ~5× the
tensor-core throughput and ~6× the bandwidth — while the CUDA-core kernel nearly halved (16.9→9.7). A kernel
that doesn't speed up on a 5×-bigger tensor core **isn't tensor-core-bound** — it's pinned by per-CTA
scheduling overhead (the opaque-fragment smem round-trip + my 1-warp load) that doesn't scale with the GPU.
That's the cleanest possible proof that decode's bottleneck is the *schedule*, not the math. Reclaim-at-batch
agreed: on A100 the CUDA-core kernel beats SDPA 2.5–8.6× at every batch, while the tensor-core one *loses* to
SDPA past B=8. **Say-this:** "I didn't just measure that tensor cores are slower — I measured that they don't
*scale*, which tells me *why*: decode is bound by per-CTA scheduling, not compute, so throwing a bigger tensor
core at it can't help. The probe was the cheap experiment that let me kill the expensive kernel with evidence
instead of a hunch. v8's deliverable is the CUDA-core M-packing; the tensor-core path is a documented,
two-architecture dead end — and knowing *why* it's dead is the result."

## C11.5 — Hiding the reduction latency: occupancy vs ILP (v8.5 → v8.6)

**The chain that got me here.** Three measurements pinned the decode floor: v8 Cut 1 wins (M-packing, 8.6×)
but sits at ~10% HBM; Cut 2 (tensor cores) lost and *didn't scale* T4→A100; v8.5 (double-buffer) was a clean
null — prefetching the KV load moved nothing. Triangulated, the kernel is **compute-latency-bound at ~10%
HBM**, and the latency is the **per-key warp-shuffle reduction + serial online-softmax recurrence**: one warp
runs a 5-deep `__shfl_xor` butterfly per key, chained into a dependent softmax update, with nothing to
overlap it. v8.5 ruled OUT memory; v8.6 asks whether the compute latency is **hideable**.

**Why an ablation, and what each arm isolates.** The user-scoped levers are all about *hiding* (not removing)
the latency, so I split them so any gain is attributable:
- **Arm 1 — occupancy (`v8_gqa_occ`):** stage KV as FP16 smem (16 KB single buffer) → **4 blocks/SM** (Cut 1
  is 32 KB FP32 → 2). The subtlety that makes this *not* a re-run of v8.5: v8.5 also halved smem to FP16, but
  *spent* the savings on a 2nd buffer and stayed at 2 blocks/SM. Same half tile, opposite use — v8.5 bought
  overlap (null), Arm 1 buys **occupancy**. Doubling resident warps hides the whole serial chain with TLP.
- **Arm 2 — key-ILP (`v8_gqa_ilp`):** unroll the key loop KU=4 — compute 4 independent partials, fire their
  4 reductions back-to-back so the independent shfl chains pipeline, then 4 sequential softmax updates. FP32
  smem kept (2 blocks/SM) so ILP is the lone variable. Note: I *rejected* float2/half2 vectorization here
  because the lane-strided layout (`lane + 32·e`) isn't contiguous — and v8.5 already showed loads aren't the
  wall, so a wider load is a near-null control, not a lever.

**The prediction I'm committing to before the GPU run.** The byte-roofline is **blind** — same AI=2G/b, same
floor, same limiter for both arms (I verified it on the T4 arch: G=8 → AI=8.0, HBM-bound, 0.105 ms). So
v8.6's claim is purely about the *schedule*, which the model can't represent. Mechanistically I predict
**occupancy > ILP**: more warps hide the *entire* chain (reduction + exp + softmax update) via thread-level
parallelism, while ILP only overlaps the *reduction* sub-part and leaves the serial softmax recurrence
exposed. **The counter-prediction is the real prize:** if BOTH are null, the floor is the serial
online-softmax recurrence itself — which neither occupancy nor ILP can touch — and the only remaining lever
is a **score-stationary redesign** (one key per lane → no per-key cross-lane reduction) that *removes* the
wall instead of hiding it (a future v8.7). Either way, v9 FP8 stays premature until a kernel is actually
bandwidth-bound. **Say-this:** "By v8.6 I'd localized the decode floor to a single serial inner loop and ran
a two-arm ablation — occupancy vs ILP — to test whether that latency is hideable. I committed to occupancy
being the stronger lever and to a falsifiable counter-prediction: if neither moves %HBM, the bottleneck is
the online-softmax recurrence, which means the fix is an algorithmic relayout, not a knob. That's the
discipline — isolate one variable, predict the mechanism, and let the null result point at the next kernel."

### v8.6 measured — both arms null, the counter-prediction landed
The run confirmed the counter-prediction. Correctness 190/190; on the clock-robust `%HBM` (the SM clocks
swung 360–1590 MHz across runs, so µs/tok was unusable for cross-backend comparison) **both arms were flat at
~10%.** Two clinching details: occupancy *never engaged* at the micro-bench batch — split-KV already emits
~80 blocks = exactly 2/SM on the T4, so Arm 1's 4-block ceiling had no third block to schedule (it only
showed a ~1% nudge in the one large-batch corner where spare blocks existed); and ILP tracked Cut 1's %HBM
to the decimal. **Say-this:** "Both levers were null, exactly as the counter-prediction said they might be —
which is the *informative* outcome: it proves the floor is the per-row serial recurrence, not a knob I forgot
to turn. And I caught a subtle confound — occupancy can't help when the grid doesn't even have enough blocks
to fill the higher ceiling. Four straight negatives (tensor cores, double-buffer, occupancy, ILP) all
triangulate the same serial inner loop. That earns the next kernel: stop hiding, relayout."

## C11.6 — Removing the wall: the score-stationary relayout (v8.7)
Four negatives said the per-key warp-shuffle reduction + serial online-softmax recurrence is the decode
floor and it's unhideable. So v8.7 *removes* it instead of hiding it, by flipping the inner-loop layout to
the textbook FlashDecoding form. Cut 1 is **output-stationary**: one warp owns a query row, the 32 lanes
split the head dim, and each key costs a 5-deep `__shfl_xor` butterfly (to assemble the scalar score across
lanes) plus a serial `(m,l,O)` update. v8.7 is **score-stationary**: assign **lane = key** — lane `l`
computes the *whole* dot product q·k_c itself, so the score lives in its own register with **no cross-lane
reduction in QK at all**. The softmax then runs **once per 32-key group** (one `warp_reduce_max`, one
`warp_reduce_sum`), so the serial recurrence — the actual wall — gets **32× shorter**. The cost that moves is
PV: now `O[d]=Σ_c p_c V[c][d]` needs `p_c` (held in lane `c`) spread across the output dims, done with a
single-hop `__shfl` broadcast per key — but those broadcasts are *independent*, so they pipeline, unlike the
recurrence they replace. The layout literally inverts where the cross-lane traffic lives (Cut 1: reduction in
QK, free PV; v8.7: free QK, broadcasts in PV).

Two engineering subtleties worth saying out loud: (1) **bank conflicts** — with lane=key, 32 lanes read
`sK[key][d]` at stride D (a multiple of 32) → 32-way conflict, so I stage K **transposed `[d][key]` + a
1-element pad** to make the read (and the transposed write) conflict-free. (2) **holding occupancy fixed** —
the relayout needs extra smem (the query in smem + that pad), which in FP32 would drop the T4 from 2→1
block/SM and *confound* a null result with lost occupancy. Since v8.6 already proved FP16 smem is
perf-neutral, I stage everything FP16 (~18 KB → ~3 blocks/SM) so the inner-loop layout is the *only* variable
— and as a bonus `v8_gqa_ss` differs from `v8_gqa_occ` (also FP16-smem) in nothing but the layout, a clean
isolated A/B. **The honest prediction:** wins at d=64; d=128 is at risk because each key now reads the full D
from smem (more smem-read traffic than Cut 1's D/32) and could flip to smem-BW-bound. And the counter that
keeps me honest: if µs/tok finally drops but %HBM *stays* ~10%, the floor was per-CTA **load** latency, and
that's the result that finally licenses v9 FP8 (capacity). **Say-this:** "v8.7 is the first decode kernel
where I changed the *algorithm's* data layout, not a scheduling knob — lane-per-key removes the per-key
reduction and shortens the softmax recurrence 32×. I predicted it wins at d=64 and flagged d=128 as a
smem-bandwidth risk *before* running, and I deliberately held occupancy fixed with FP16 smem so a null
couldn't be blamed on residency. Whatever the number, it's decisive: either removing the reduction was the
fix, or the floor was load latency all along and bytes are finally the right lever."

### v8.7 measured — the win, and the nuance that made it *more* interesting
It won — and the result is richer than a clean yes. Correctness 228/228. The clocks cooperated this time:
`v8_gqa_ss` happened to run at 375/465 MHz, right next to Cut 1's 360/480, so the headline comparison was
clock-fair (and `occ` ran hot at 1590, which is why I trust ss-vs-Cut-1 over ss-vs-occ on raw µs/tok). **ss
beat Cut 1 1.1–1.6× at matched clock across every G and every batch, beat SDPA 8–16×, and — for the first
time since Cut 1 — moved the clock-robust `%HBM` up (~8% → ~10–12%).** The d=64/d=128 split came out exactly
as predicted: d=64 won more (1.18–1.35×) than d=128 (1.16–1.18×), the fingerprint of the d=128 smem-read-BW
drag I'd flagged — but it didn't null d=128, it just taxed it. So **two things are true, and saying both is
the point:** removing the per-key reduction *was* a real win (so the reduction genuinely was part of the
floor — v8.6 was right that you can't *hide* it, and v8.7 shows you *can* remove it), **but** %HBM only
climbed to ~10–12%, not to the bandwidth ceiling — a residual per-CTA limit (load latency / small-CTA launch)
is still there. That's the honest close: v8.7 is the inner-loop decode win, M-packing is the per-CTA decode
win, and together they're the two real levers; the leftover ~10% ceiling is something bytes (v9 FP8) won't
fix, which is exactly why FP8's pitch is capacity + accuracy, not micro-bench latency. **Say-this:** "v8.7
won — 1.1–1.6× over my own M-packing kernel at a fair clock, and the first time I moved %HBM since the
original. But the better answer is the nuance: it proved the reduction was a real wall *and* that removing it
doesn't make decode bandwidth-bound — there's a residual per-CTA ceiling underneath. So I can say precisely
what the two decode levers are (pack the heads, restructure the inner loop), what the dead ends were (tensor
cores, double-buffer, occupancy, ILP), and why low-precision is a capacity play, not a latency one. That's
the whole decode-schedule story, measured end to end."

---

## C12 — I caught my own six-step conclusion was confounded (the L2-residency trap)

**The chain:** Before building v9 I ran an adversarial close-out on the whole v8 family, and it caught
something I'd been repeating since **v6**: "decode is per-CTA-bound, not bandwidth-bound — ~10% of HBM,
flat across batch." That verdict drove three steps of design (v8.5 double-buffer, v8.6 occupancy + ILP,
all nulls). **It's confounded.** At my bench sizes the KV cache *fits in the T4's 4 MB L2* — reclaim
G=8/B=1 is H_kv=1, so KV = `2·1·8192·128·2 B ≈ 4.2 MB ≈ L2 exactly`. An L2-resident working set streams
from L2 at ~1.3 TB/s (≈4× HBM), so the **DRAM counter reads ~10% even if the kernel is fully memory-bound**
— it's just bound by the *wrong memory*. I'd been reading "10% of HBM" as "not bandwidth-bound" when it
could equally mean "never left L2." On top of that, free Colab gives no root, so I never locked clocks
(they swung 360–1590 MHz). So the metric I leaned on was confounded two ways and I never measured L2
traffic or pushed N_k past ~16K.

**What saves it from being just an error:** I questioned the critique too. The G-sweep's low-G corner
(G=2 → H_kv=16 → ~67 MB KV, way past L2) still showed ~11% HBM — real evidence the per-CTA verdict
*survives* past L2. So the honest state isn't "I was wrong," it's "probably per-CTA-bound, but unproven —
and I can say exactly which experiment settles it."

**The fix (v9 Task 1):** rent a *root* T4 (so clocks lock and ncu counters work), pin clocks, flush L2
between timed iterations, and sweep N_k 1K→128K × batch × H_kv while measuring HBM% *and* L2 hit-rate *and*
the counter-free L2 test (effective BW = bytes/time; if it exceeds HBM peak, the data came from L2). Where
HBM% plateaus as L2-hit-rate falls is the genuine bandwidth-bound regime. If it *still* reads ~10% past L2
at large batch, "per-CTA/launch-bound" is finally earned.

**Why it matters for the roadmap:** this is also why "FP8 is capacity-only, not latency" was too strong —
it's true for an L2-resident micro-bench, but FP8 KV is a real latency win past ~4–7k tokens / under load
(vLLM measured per-token cost → 54% of BF16). So FP8 (v9) does double duty: halving KV bytes is exactly
what creates the memory-bound regime to test the limiter in.

**Say-this:** "The strongest thing I did in this project was catch my own recurring conclusion. I'd said
'decode isn't bandwidth-bound' for six steps — then realized my benchmark's KV cache fit inside the GPU's
L2, so the HBM counter *couldn't* show bandwidth, confound, not result. I also never locked clocks. The
lesson is that '% of peak HBM' is meaningless for an L2-resident kernel; you have to size the working set
past L2, flush caches, pin clocks, and measure L2 traffic before you're allowed to name the limiter. v9 is
built to earn that verdict, not assume it."

**Resolution (v9 Task 1, measured on a ROOT T4 — and the counters finally worked).** I rented a
bare-metal T4 so I could lock clocks (pinned 1590 MHz, no throttle) and — for the first time in the whole
project — run ncu (every prior step had `ERR_NVGPUCTRPERM` on containerized rentals). Then I flushed L2
between iterations and swept N_k to 128K so the working set hit 537 MB, ~130× past the 4 MB L2. **The
verdict came back confound-free: per-CTA-bound, not bandwidth-bound.** Achieved HBM tops out at ~28–29%
(and only that high *with* enough occupancy — at low occupancy it's ~11%), never near the ~70% ceiling,
even with the data definitely coming from HBM. ncu nailed it: past L2 the **L2 hit-rate is 1.1%** (so the
bytes really are streaming from DRAM) while **DRAM throughput is only 12.85%** — HBM-served and ~13% busy
is the definition of per-CTA-bound. Two things I'm proud of here: (1) I *refined* my own story rather than
just confirming it — "~10% HBM" turned out to be an occupancy artifact, the true cap is ~28%, set by
having one active warp per CTA at N_q=1; and (2) **my counter-free proxy was validated** — `eff_bw =
bytes/time` read 13.8% where ncu's hardware counter read 12.85%, within a point, so the whole project's
profiler-free methodology holds up against the real counters. **Say-this:** "I didn't just suspect the
confound — I went and removed it. Root box, locked clocks, flushed L2, swept 130× past L2, and ran ncu for
the first time. The kernel is genuinely per-CTA-bound — ~28% of HBM peak with the data provably coming
from DRAM — and my counter-free bandwidth proxy matched ncu to within a point, which means the dozens of
profiler-free measurements I'd taken when ncu was blocked were trustworthy all along."

## C13 — FP8 KV cache: I predicted "capacity-only" and the data proved me wrong (a load-bandwidth win)

**The setup:** v9 stores the KV cache as FP8 E4M3 (1 byte) instead of FP16 (2). The roofline doubles the
decode arithmetic intensity (AI = 2G/b: G=8 fp16 → 8.0, fp8 → 16.0) and halves the HBM floor (13.1 → 6.6
µs). After C12 (the L2 confound), I was *sure* this would be capacity-only — the kernel sits ~10× above
the HBM floor and the bench KV fits in L2, so halving HBM bytes should move nothing. **I recorded that
prediction, and the measurement refuted it.**

**What actually happened:** FP8 bought a real, same-session, **clock-matched ~1.3× decode-latency win**
(`vs naive` = FP8 kernel ÷ FP16 v8.7 kernel, 0.96–1.52× across G, flat across batch B=1→64). The win is
**airtight as a load-bytes effect**: the *only* kernel change is the global/L2 load — 1 byte vs 2 per KV
element — while smem stays FP16 (identical occupancy and inner loop), and FP8 *adds* ALU (software
E4M3→half on sm_75). It's faster *despite doing more compute*, so the bytes moved from L2 into the SM were
a genuine bottleneck.

**The smoking gun that nails the mechanism:** the win *shrinks as the GQA group G grows* (d=64: 1.37 at
G2 → 0.96 at G32 — slightly *slower*). M-packing reads each KV tile once and runs it against G query
heads, so as G rises the per-CTA cost shifts from *loading KV* to *computing G dot products*; FP8's byte
saving matters less, and at G32 the dequant ALU finally exceeds it. d=128 wins more (double the bytes per
key). That `win ∝ bytes-loaded ∝ 1/amortization` double-dependence is the fingerprint of an **L2→SM
load-bandwidth** limit — not HBM bandwidth (`%HBM` *dropped* and my "L2!" flag never fired), not pure
compute.

**The corrected lesson:** "L2-resident ≠ memory is free." An L2-resident working set still has to be
*loaded* L2→SM every tile, and that path is a real bandwidth wall. My six-step "per-CTA-bound, bytes won't
help" conclusion was too strong: the residual post-v8.7 ceiling was *partly* L2-load bandwidth, which
fewer bytes relieve (~1.3×), and *partly* per-CTA latency/compute, which they don't. FP8 didn't flip the
limiter (still ~10% HBM, ~14× above the halved floor) — it shaved one of two stacked components.

**The engineering traps I had to get right.** (1) **Fused per-tile dequant, never a prepass** — a
full-cache dequant pass re-reads the whole KV cache and *eats the byte savings* (QServe). I dequant at the
smem gather, so the score-stationary inner loop is byte-identical → clean one-variable ablation. (2)
**FP32 accumulation** — only *storage* is 8-bit; the accumulator stays FP32 to dodge FA-3's FP8-accum
cliff. (3) **Two references, not one** — correctness vs SDPA on the *same* dequantized E4M3 bytes
(isolates kernel math, oracle RMSE ~6e-6); accuracy vs the *original* fp16 KV (the real E4M3 quant cost,
~7e-4). (4) **A measurement bug I caught:** my FP8 SDPA oracle re-quantized K,V *inside* the timed
baseline, inflating `vs sdpa` ~4× — so I trust the same-session `vs naive` and moved the dequant out of
the timed loop. Cheap to write a confounded benchmark; the discipline is sanity-checking the baseline.

**Say-this:** "I predicted FP8 KV would be capacity-only on my micro-bench — the kernel was 10× off the
HBM floor and the cache fit in L2, so cutting HBM bytes shouldn't matter. The data refuted me: FP8 was a
clock-matched ~1.3× faster, and the win shrank as GQA amortized the KV load — which proves it's an L2→SM
*load*-bandwidth win, not HBM bandwidth and not capacity. The lesson I now repeat is 'L2-resident doesn't
mean memory is free' — loading L2 into the SM is itself a wall. And the kernel-engineering subtleties:
dequant fused per-tile so a prepass doesn't give the bytes back, accumulation stays FP32, and I caught my
own oracle re-quantizing inside the timed baseline."

## C14 — Refining my own FP8 win under adversarial review (the close-out that sharpened C13)

After the v9 gates, I red-teamed my *own* conclusions before they hit the permanent record. It didn't
overturn anything — but it caught three places where my prose ran ahead of my data, and I'd rather recite
the precise version. This is the "I check my own claims" story.

**What I corrected in C13.** (1) I'd called the FP8 win an "L2-**bandwidth**" win. ncu says L2 throughput
was **<3%** at those shapes — so it was never bandwidth *saturation*; it's a bytes-sensitive load-**latency
/ issue-rate** effect. (2) The "shrinks as G grows ⇒ compute-amortization" smoking gun is **confounded**:
my G-sweep fixed `--heads 32`, so raising G also *lowers* H_kv (occupancy) and shrinks the working set —
amortization and occupancy can't be separated from that one sweep (the clean version is a *fixed-H_kv*
G-sweep). I lead with d=128 now because d=64 is non-monotone (G1=1.08 < G2=1.37). (3) The win is
**regime-specific and flips negative under L2-flush**: in Task 1, with L2 flushed and clock-corrected, FP8
was ~1.2× *slower* — once bytes stop being the shared bottleneck, the software E4M3 dequant ALU tax wins.
So the honest headline is "FP8 buys a real but **fragile** latency win when L2-resident; the durable wins
are **2× capacity and ~7e-4 accuracy**."

**The bigger reframe — naming the limiter.** I'd been saying "per-CTA-bound." ncu proves *not
bandwidth-bound* (L2-hit 1.1% + DRAM 12.85% past L2), but it doesn't separate "occupancy-starved" from
"latency / low-MLP" — and my batch sweep is the tell: %HBM rises to 29% at B=8 then **declines** to 25% at
B=128, a *latency/MLP* fingerprint, not occupancy starvation. So I now say **"per-CTA / low-MLP
latency-bound, occupancy-lifted to a hard ~28% cap."** That distinction changes the next lever (deeper
pipelining vs persistent kernel) and means "decode-schedule is CLOSED" is only safe **for the L2-resident
regime** — the cheap experiment to settle it is re-running my v8.5 double-buffer / v8.6 ILP kernels *past*
L2, where Task 1 shows real 29%→70% headroom (they were only ever measured L2-resident).

**Why it matters for the B300 paper (v10).** If decode is per-CTA-bound, **FP4's value is capacity +
accuracy, not bandwidth-latency** — most FP4-KV work *assumes* the bandwidth win; I *measured* that it
doesn't hold at realistic decode occupancy. Two research corrections fell out: the Blackwell `tcgen05`
tensor-core gate is **M≥64** (M=128 for 100%), not M≥16 — so N_q=1 decode keeps the FP4 cores dark and
native-FP4 *compute* is a speculative-decode (v11) lever, not v10; and the "P·V-is-cheap-to-FP4" intuition
is refuted for *compute* (quantizing post-softmax P piles cvt onto the softmax bottleneck). On novelty:
FlashInfer already ships NVFP4 KV decode on sm_103, so I frame the contribution as **"open,
roofline-documented, prediction-vs-measured sm_103 decode + asymmetric FP4 recipe, complementing — not
beating — FlashInfer/FlashMLA."**

**Say-this:** "Before I wrote up v9, I red-teamed my own result. I'd called the FP8 win an 'L2-bandwidth'
win — but ncu showed L2 was under 3% busy, so it's really a bytes-sensitive load-*latency* effect, and it
even flips negative when I flush L2 because the dequant ALU tax takes over. The durable wins are 2× capacity
and 7e-4 accuracy; the latency win is fragile. And I retitled my limiter from 'per-CTA' to 'low-MLP
latency-bound, capped ~28%' once I noticed my batch sweep *declines* past B=8 — that's a latency fingerprint,
not occupancy. The payoff is for the B300 paper: if decode is per-CTA-bound, FP4 is a capacity-plus-accuracy
recipe, not a bandwidth play — the opposite of what most FP4-KV work assumes, and that honest finding is the
contribution."

**Closing the loop (2026-06-29 — the one experiment C14 flagged).** I'd flagged "re-test v8.5/v8.6 past L2"
as the cheap way to settle whether the residual is latency or occupancy. I ran it (clock-robust
speedup-vs-Cut-1, L2-flushed; the runtime even locked clocks + ran ncu — though it still power-capped under
load, which is *why* I trust the back-to-back ratio not the absolute %HBM). **Double-buffer and ILP are dead
even at N_k=131072** — so the B=1 floor really is the serial recurrence, only the relayout removes it, and the
"nulls were an L2 artifact" worry is refuted. **But the surprise that makes it a good story:** the occupancy
arm was dead at B=1 yet **revives to ~1.4× at B≥32 past L2** — because its 4-blocks/SM residency only pays once
the grid fills it, and every prior measurement (v8.6, v9, Task 1) lived at B=1. So I'd been calling occupancy a
"dead end" on the strength of a regime that structurally couldn't show it. **Say-this:** "I re-tested my own
dead ends past L2. Double-buffer and ILP stayed dead — good, my 'CLOSED' was right at batch=1. But occupancy
came back to ~1.4× at batch ≥32, because its extra residency needs a full grid to matter and I'd only ever
measured batch=1. The lesson I keep relearning: a null is only as broad as the regime you measured it in."

## C15 — NVFP4 KV decode: building the prediction so it survives my own per-CTA verdict (v10 kickoff)

**The setup.** v10 is the paper's core: an NVFP4 (4-bit) KV-cache decode kernel on B300/sm_103. The
naive pitch writes itself — "4-bit KV → 3.55× fewer bytes → 3.55× faster decode." **My own data kills
that pitch before I make it.** v9 Task 1 measured, confound-free on a root T4 (clocks locked, L2
flushed to 537 MB, ncu live), that decode on this kernel is **per-CTA / low-MLP latency-bound, ~28%
HBM cap — NOT bandwidth-bound** at any context T4 could reach. So cutting bytes relieves a term the
kernel is nowhere near. **Say-this:** "I built v10's prediction to lose the argument I'd most want to
win. NVFP4 is a *capacity + accuracy* recipe first — the latency-from-bandwidth story is conditional
and I have to *measure* whether it even exists on B300, not assume it."

**The one variable, kept honest.** v10 forks v9 changing ONLY the KV storage format — packed E2M1
nibbles + a per-16 E4M3 micro-scale (0.5625 B/elem, and I count the micro-scale, or my AI is a lie).
The score-stationary loop, M-packing, split-KV, merge are byte-identical, so it's a clean byte-only
A/B. The dequant is **fused per-tile** (unpack nibble → ×micro ×scale → FP16 smem), never a prepass —
a prepass re-reads the cache and eats the saving (QServe). And the Q·Kᵀ score is reconstructed at FP16,
never a raw FP4 dot, because that error feeds exp and the softmax collapses.

**Why no tensor cores (the v8 lesson generalizes).** Blackwell's 5th-gen FP4 cores gate at **M≥64**
(M=128 for 100%). N_q=1 decode packs M=G<64 — below the gate. So v10 decode is CUDA-core /
dequant-to-FP16, and native FP4 *compute* slips to v11 (multi-token, where you pack the query axis to
reach M≥64). This is exactly v8's measured "tensor cores are the wrong tool for decode," now on
native-FP4 silicon. **Say-this:** "The 4-bit *storage* and the 4-bit *math* are different levers with
different shape gates. v10 takes the storage win; the compute win needs a different problem shape."

**The two-layer prediction (recorded before coding).** Pure roofline: HBM-bound, floor 3.55× below
FP16. Per-CTA-corrected (the real one): capacity ~3.55× is *certain*; the FP4 accuracy delta is the
*headline number*; a latency win only as a fragile v9-style L2-resident load effect that shrinks as G
grows and likely flips negative under flush (the dequant ALU tax — same thing that explains v9's
"flips-negative" result). **The counter-prediction is the prize:** B300 has a ~126 MB L2 (vs T4's
4 MB), so if I sweep N_k to ~1M and the achieved %HBM finally *climbs* toward the ceiling as the working
set spills L2, decode becomes bandwidth-bound on sm_103 for the first time and the byte cut converts.
Find the knee → a real bandwidth result. No knee → the more surprising result: per-CTA-bound to a
million tokens. **Either sign is publishable, and I wrote the methodology (clock-lock / L2-flush /
counter-free / ncu) precisely so I can tell which regime I'm in.**

**Scope honesty.** FlashInfer already ships NVFP4 KV decode; vLLM published GB300 NVFP4 decode. I'm not
"first to run." The contribution is the **open, roofline-documented, prediction-vs-measured sm_103
decode study with an asymmetric FP4 recipe and a confound-free per-CTA methodology — complementing, not
beating, FlashInfer/FlashMLA.** The per-CTA-bound verdict is the spine, not a caveat I bury.
