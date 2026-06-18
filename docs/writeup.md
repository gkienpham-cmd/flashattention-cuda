# From the bandwidth wall to the exponential wall: rebuilding FlashAttention across four hardware eras

> Final essay. Written incrementally as the journey progresses; each phase adds a section with
> the bottleneck it faced, the co-design that answered it, and the measured curve. Drafted by
> the mentor, rewritten in my own words (per the engineering-discipline rule). Stub for now.

## Thesis
Every hardware generation moves the bottleneck. FlashAttention's lineage (FA-2/3/4) is a story
of *finding the current limiting factor and co-designing the algorithm around it* — the same
instinct I bring from kart racing, where you trim time against whatever the measurement says is
slow. This is that story, rebuilt from scratch, one measured speedup at a time, on the hardware
I actually have.

## Outline (fills in per phase)
1. **The bandwidth wall** (Phase 1, T4) — naive attention is memory-bound; tiling + online
   softmax + fusion raise arithmetic intensity until tensor cores become the point.
2. **The latency wall** (Phase 2, Ampere/Hopper) — hiding data movement with async copy and
   warp-specialization.
3. **The precision frontier** (Phase 3, the Sconce twist) — INT8/FP8/INT4 raise the roof itself;
   block quantization + incoherent processing; RMSE-vs-FP64 accuracy as a first-class result.
4. **The exponential wall** (Phase 4, FA-4 era) — when matmul is cheap, softmax's `exp` and
   shared-memory traffic dominate; conditional rescaling and software `exp2` emulation.
5. **Serving it** (Phase 5) — causal/var-len masking, KV-cache, paged attention: the bridge to a
   from-scratch mini-vLLM serving a compressed model.

## Show HN
> Draft when the curve is real.
