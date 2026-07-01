# Narrative & voice — Kien Pham

The site LEADS with the **mechanical-engineer → ML crossover**. The through-line is a single
sentence Kien learned on a race track and now applies to silicon: **find the limiting factor, and
trim time against a measurement.** Everything else is proof.

Voice: honest, measured, deeply technical, quietly confident. Never hypey. The differentiator is
that the *misses are on display* — that's the strongest evidence the wins are real.

---

## The one-liner (hero)

**Find the limiting factor. Trim time against a measurement.**
A mechanical engineer's discipline, applied to GPU kernels — rebuilding FlashAttention from scratch,
one measured speedup at a time.

## The crossover (About section)

Kien races go-karts competitively. On a track you don't guess — you find the corner that's costing
you, measure it, and trim time against the clock. That's the whole mindset behind this project: a
**roofline model** names the limiting factor *before* any code is written, then the kernel is built,
measured, and the prediction is checked against reality — honestly, including every time the model
was wrong.

He came to GPU kernels from **mechanical engineering** and an **ML-research** background: at UC Santa
Cruz he worked on **Sconce**, a model-compression system (structured pruning + post-training
quantization across 6 CNNs — ~70% memory and ~90% parameter reduction, 2.8× inference speedup, ~93%
accuracy retained). That quantization fluency is his edge for the low-precision attention work
(FP8, NVFP4). He'd never written an attention kernel at the metal before this project. Now there are
seventeen.

He also shipped **TGC Speedway**, a companion app for kart racing (React Native / Expo / Supabase,
live timing, lap times as integer milliseconds).

The long game: this kernel library is the foundation for a **from-scratch mini-vLLM** inference
engine (paged attention + continuous batching) that will one day serve a Sconce-compressed model.

## The method (Methodology section)

Every one of the 17 kernels followed the same loop, in order: **predict the limiter with a roofline
model before coding → explain the memory-hierarchy reasoning → write and hand-verify → test against
PyTorch SDPA → benchmark → record prediction-vs-measured, including the miss.** A step wasn't done
until the tests were green and the reasoning held.

The recurring lesson — and the reason this reads as engineering, not marketing — is that the roofline
kept naming the *limiter location* right but missing the wall-clock *magnitude*, five steps straight,
because a pure FLOPs/bytes model is blind to the schedule (L2 caching, occupancy, launch overhead).
By v11 the model had become two-layered: pure roofline + a per-CTA correction. Catching that
recurring blind spot — and removing the confounds to prove it — is the spine of the work.

---

## Vivid quotes (pull-quotes / kinetic-type moments)

Use these verbatim as large pull-quotes in the Methodology / Journey sections.

> "Registers in-hand, shared memory is the per-SM scratchpad, HBM is the far main memory. There's a
> ~25× bandwidth cliff at HBM, so the game is to move data across that cliff as few times as possible
> and reuse it on-chip." — the memory-hierarchy foundation

> "I read the DRAM counters and tiling cut zero traffic. The L2 already served the redundant reads
> the roofline charged to HBM. A traffic model is blind to the cache that serves the traffic — so
> verify the limiter with the throughput counter." — the L2 lesson (v2)

> "Deleting 99% of DRAM traffic made the kernel slower. Nothing was bandwidth-bound, so removing
> bandwidth-bound work while regressing the schedule was a net loss." — the S-elimination paradox (v3)

> "v8.7 won — and the better answer is the nuance: it proved the reduction was a real wall *and* that
> removing it doesn't make decode bandwidth-bound. I can name the two decode levers, the four dead
> ends, and why low-precision is a capacity play, not a latency one." — closing the decode arc (v8.7)

> "'Per-CTA-bound' was real — but it was a statement about the engine, not the problem. The right
> engine converts work to throughput." — the v12 correction

---

## Facts for copy (do not embellish beyond these)

- Solo project. 17 CUDA kernels. 4 GPU architectures (T4 Turing sm_75, A100 Ampere sm_80,
  B200 Blackwell sm_100, B300 Blackwell Ultra sm_103).
- Research target: a PMBS@SC-tier characterization paper — the first open, kernel-level,
  prediction-vs-measured roofline characterization of FlashAttention decode on sm_103. Complements
  (does **not** beat) production kernels like FlashInfer / FlashMLA.
- Honesty guardrails to preserve on the site: v5 prefill bench is unmeasured; the 4× cuBLAS gap is a
  structural inference (not a torch-side profile); Blackwell verdicts are proxy-grade (ncu was
  blocked; the proxy is validated once, on T4); accuracy RMSE numbers are single-seed.
