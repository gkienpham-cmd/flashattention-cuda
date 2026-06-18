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
not sequence length. At d=64 that lands around 14-16 FLOP/byte — under the 25.3 ridge — so v1 is
pinned to the bandwidth roof. Every Phase-1 optimization is a campaign to raise AI (shrink the
denominator) until we hit the compute roof, then a campaign to raise the roof itself (tensor
cores, then low precision).

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
