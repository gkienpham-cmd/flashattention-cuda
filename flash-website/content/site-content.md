# Site content — section by section

All copy for the site. Structured data (kernels, metrics) lives in `kernels.json` / `metrics.json`.
Narrative + pull-quotes in `narrative.md`. Every section is **100vh**.

---

## 1 — Hero (cinematic KART → GPU morph)

**Scene:** a scroll-scrubbed `<canvas>` sequence. Frame 1 = a go-kart on a racing line; as you
scroll the track lines morph into circuit traces → a GPU die → the B300 board. The roofline overlay
draws itself on top; the title sits *behind* the subject through a gradient mask.

- **Eyebrow:** Mechanical engineer · GPU kernels · ML systems
- **Title (kinetic):** FlashAttention, from scratch
- **Sub (types in as the morph resolves):** Find the limiting factor. Trim time against a measurement.
- **Beat text at the die (appears behind the silicon):** the same discipline — on silicon now
- **Count-up chips (from metrics.json → headline):** 17 kernels · 4 architectures · 8–16× vs SDPA · 1785 TFLOP/s
- **Scroll cue:** "scroll" ↓

---

## 2 — About (the crossover)

**Heading:** From the racing line to the roofline

On a track you don't guess — you find the corner that's costing you, measure it, and trim time
against the clock. I brought that to GPUs: a **roofline model names the limiting factor before I write
any code**, then I build the kernel, measure it, and check the prediction against reality — honestly,
including every time the model was wrong.

I came from **mechanical engineering** and ML research. At UC Santa Cruz I worked on **Sconce**, a
model-compression system — structured pruning + post-training quantization across 6 CNNs, ~70% memory
and ~90% parameter reduction, 2.8× inference speedup, ~93% accuracy retained. That quantization
fluency is my edge for the low-precision attention work. I'd never written an attention kernel at the
metal before this. Now there are seventeen.

**Micro-facts (small stat row):** Sconce · ~70% mem ↓ · 2.8× faster · UC Santa Cruz · TGC Speedway app

---

## 3 — The Journey (v1 → v12)

**Heading:** Seventeen kernels, one measured speedup at a time

**Intro:** Three eras, each with a different limiting factor. The roofline predicted the limiter
before every step; the honest misses are on the record. (Scroll-driven timeline from `kernels.json`,
era bands colored teal / amber / clay.)

**The dead-ends beat (highlight within Era 2):** After M-packing worked, four attempts to close the
gap all failed — tensor cores, double-buffering, occupancy, ILP. Four straight negatives proved the
floor was a *serial dependency chain* you can only remove, not hide. Then v8.7 removed it.

---

## 4 — Benchmark terminal

**Heading:** The receipts

A live-typing terminal streaming a benchmark run; the headline numbers count up as the lines print.

```
$ python -m bench.harness --backend v8_gqa_ss --gqa-group 8 --precision fp16
  building kernel (JIT)....................... ok
  correctness vs SDPA: 228/228 pass
  v8_gqa_ss vs SDPA:      8–16×   ▮▮▮▮▮▮▮▮▮▮
  %HBM (counter-free):    ~11%    per-CTA-bound
$ python -m bench.regime --backend v12_mla_tc --arch sm_103 --scale
  B=64 · K=524288:        1785 TFLOP/s   = 36% FP8 peak · 46% HBM
```

**Caption:** Every number on this site traces to a measured run in `docs/results.md`.

---

## 5 — Results

**Heading:** What the curve says

- **Speedup progression** (animated): v1 baseline → v4 fused (first S-off-HBM win) → v8 M-packing (8.6×)
  → v8.7 score-stationary (8–16× vs SDPA) → v12 tcgen05 (1785 TFLOP/s).
- **Memory & capacity wins:** 125× (S elimination) · 3.56× (NVFP4 capacity) · 202× (MLA vs MHA).
- **The roofline scorecard:** limiter location right most steps; magnitude missed 5 straight — the
  reason the model became two-layered.

Use `assets/diagrams/v1-v12-arc-summary.svg`, `v12-throughput-regime.svg`, `decode-roofline.svg`.

---

## 6 — Independent Validation

**Heading:** It reproduces

An independent review re-ran the roofline tool and reproduced every published arithmetic-intensity
number exactly (MLA FP16 234.8 · FP8 469.7 · NVFP4 835.0 · GQA-8 3.6), and cross-checked the external
claims against current literature — B300 constants, the decode-is-work-starved characterization, the
MLA capacity math, and the "complement, not beat" positioning all held. The counter-free %HBM proxy
matched Nsight Compute within one point (13.8% vs 12.85%).

> "The honesty about misses is not a weakness — it is the strongest evidence that the wins are real."

**Honesty note (keep visible):** Blackwell verdicts are proxy-grade (ncu was blocked on the rented
box; the proxy is validated once, on T4). The 4× cuBLAS gap is a structural inference. Accuracy RMSE
numbers are single-seed. Use `assets/diagrams/portfolio-validation-scorecard.svg`.

---

## 7 — Methodology / Honest Misses (the differentiator)

**Heading:** The misses are the method

**Body:** A pure FLOPs/bytes roofline is blind to the schedule. It named the limiter *location* right
most steps but missed the wall-clock *magnitude* five straight — because L2 caching, occupancy, and
launch overhead live in the schedule, not the arithmetic. Recording that honestly, and removing the
confounds (locked clocks, L2 flush, past-L2 sweeps, one ncu ground-truth run) to earn the diagnosis,
is the whole contribution. (Kinetic pull-quotes from `narrative.md`.)

---

## 8 — Diagram reel

**Heading:** The work, in figures

A horizontal scroll/scrub montage of the hand-authored diagrams
(`assets/diagrams/*.svg`) — the roofline, the per-CTA anatomy, GQA M-packing, the NVFP4 dataflow, the
v12 work-starvation correction. Each scrubs into frame as you move horizontally.

---

## 9 — Contact

**Heading:** Let's build fast things

**Body:** Open to ML-systems / GPU-kernel / research internships. The kernels are the foundation for a
from-scratch mini-vLLM.

- **GitHub:** github.com/gkienpham-cmd/flashattention-cuda
- **Email:** pgkien11@gmail.com
- **LinkedIn:** {{LINKEDIN_URL}}  ← placeholder, fill in

**Primary bullet (resume.md, Version C):** Applied bottleneck-hunting from mechanical systems to GPU
kernels — 17 CUDA kernels, 4 architectures, 8–16× over PyTorch, characterized NVIDIA's B300.
