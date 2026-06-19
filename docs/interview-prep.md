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
