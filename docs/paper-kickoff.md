# Paper-writing kickoff — PMBS@SC 2026 (paste into a fresh session)

> Born from the **Arm 2 Stage A close-out (2026-07-02)**: Stage A measured NEGATIVE (native FP4
> compute does not help MLA decode), which *strengthens* the thesis and closes the last open kernel
> arm. The measured arc **v1→v12 is complete**; per `docs/research-strategy.md` the prescribed move
> is **Path A = write the characterization paper**. This is a **direction change** from the kernel
> per-step loop to **paper assembly**. Nothing blocks the paper — all data + 8 figures exist.
>
> **How to use:** paste the fenced block below into a new Claude Code session, or just tell that
> session *"read `docs/paper-kickoff.md` and execute it."* Format decided: **LaTeX, IEEE/SC
> two-column.** Deadlines: **arXiv ~Jul 28, PMBS full-paper Aug 5 (AoE)**, late-breaking Aug 26.

```
You are starting the PAPER-WRITING stage of the flashattention-cuda project (repo root
/Users/kienpham/Documents/flashattention-cuda). The measured kernel arc v1→v12 AND the Arm 2 Stage A
boundary test are all DONE — no new kernels, no GPU runs. Your mission: produce a complete first
LaTeX draft of the PMBS@SC 2026 paper. Target arXiv ~Jul 28, PMBS full-paper deadline Aug 5 (AoE).

## Read first (in order), then restate the north star + honesty constraints back to me
1. CLAUDE.md — north star + Status (v1→v12 + the "Arm 2 Stage A" entry).
2. docs/paper-outline.md — THE blueprint: title, abstract, §1–§7 structure, word budgets, figure map.
3. docs/results.md + docs/decisions.md — the measured record (Steps 1–12 + "Arm 2 Stage A"). Every
   number in the paper comes from here.
4. docs/stage-a-results.md — the §5.3 boundary result + caveats.
5. docs/interview-prep.md — C1–C19, the plain-English reasoning per finding (raw material for prose).
6. docs/research-strategy.md — venue/deadline context. roofline/archs.py — measured B300 constants.

## §0 — Honesty constraints that govern the whole paper (non-negotiable)
- Every load-bearing number MUST reconcile to results.md/decisions.md/the CSVs. If you can't trace
  it, don't print it — list it as unreconciled.
- "Complement, not beat." FlashInfer/FlashMLA/CUTLASS ex77 already run B300 decode and are faster by
  construction. The contribution is the open prediction-vs-measured characterization + the two-layer
  model + the boundary results — NOT a SOTA speed claim. No "we beat X" anywhere.
- Prediction-vs-measured MISSES are first-class results (v2 L2 blind spot, v3 ~150× off floor, the
  "per-CTA-bound forever" → work-starvation correction at v12). Report them plainly.
- Disclose ALL threats (§5.4): counter-free proxy ncu-validated on T4 but NOT on B300; v5 prefill
  bench was NEVER executed (prediction only — NEVER call it measured); single-seed accuracy
  (gpt2-small); nsys provenance (v10 data from an off-notebook 2025.3.2 run, committed cell was empty).
- VERIFY the two-layer scorecard before asserting it. The outline claims "Layer 2 correct 7/7, Layer
  1 wrong 5/7." Recount from the v6–v12 record and use the DEFENSIBLE tally; if it isn't exactly
  7/7 vs 5/7, state the real numbers.
- NAMING TRAP: the paper's "Stage A" (§5.3) = the Arm 2 FP4-vs-FP8 GEMM COMPUTE test. Do NOT conflate
  it with the v10 "Stage A / A′" asymmetric-precision ACCURACY ablation (results.md Step 10) — keep
  them distinct in prose.

## Mission / deliverable — a compiling IEEE two-column draft
- FIRST: web-search "PMBS 2026 call for papers" + the SC26 workshop proceedings template to confirm
  the exact class + page limit. Default to IEEEtran (conference option), two-column, ≤8 pages, ~6000
  words if unconfirmed. State which you used.
- Create: paper/main.tex, paper/sections/{intro,background,method,results,discussion,related,
  conclusion}.tex, paper/refs.bib.
- paper/figures/: convert the 8 main SVGs (docs/diagrams/v12-{related-work-landscape,
  two-layer-prediction-model,throughput-regime,work-starvation-correction,engine-ridge-comparison,
  paper-positioning,arm2-research-plan}.svg + v1-v12-arc-summary.svg) to PDF via rsvg-convert or
  inkscape; if neither is installed, add a Makefile target + note the user can convert on Overleaf.
  Every figure the outline lists (Fig 1–8) must appear with a caption.
- refs.bib: build the bibliography. The outline cites GLA (arXiv 2505.21487), roofline-method
  (2605.04178), Attn-QAT (2603.00040), SnapMLA, SageAttention3, FA4, FlashInfer, FlashMLA, +others.
  VERIFY each resolves to a real paper via web search — SOME arXiv IDs in the project notes may be
  internal placeholders; fix them or mark [CITATION-VERIFY] rather than inventing.

## Source → section map (follow docs/paper-outline.md word budgets)
- §1 Intro: outline §1 + Fig 1 (empty-cell landscape) + Fig 7 (positioning); the 3 contributions.
- §2 Background: MHA/GQA/MLA; roofline model + why it fails for decode; measured B300 constants from
  roofline/archs.py (148 SM, 8 TB/s, 132.6 MB L2, 2032 MHz, FP16 2.5 / FP8 5 / NVFP4 15 PF, EX2 5.33).
- §3 Methodology: two-layer model (Fig 3) + pre-registered predictions + 12-step table (Fig 2) +
  counter-free proxy validation (v9 Task 1: 13.8% vs ncu 12.85%).
- §4 Results (the meat): §4.1 prefill v1–v5 brief (FLAG v5 predicted-not-measured); §4.2 decode
  v6–v12 (split-KV/paging, GQA M-packing + score-stationary, FP8/NVFP4, MLA, tcgen05) with Fig 4 +
  Fig 5; §4.3 cross-arch ~40 GB/s per-CTA ceiling (T4/B200/B300). Compress results.md Steps 6–12 HARD.
- §5 Discussion: engine-ridge framework (Fig 6); serving implications; §5.3 Stage A (Fig 8, from
  stage-a-results.md — negative, with the cross-harness caveat); §5.4 threats.
- §6 Related work: the 3 closest (GLA, Attn-QAT, roofline-method) + brief mentions; Table = Fig 1.
- §7 Conclusion: per-CTA wall = work-starvation; two-layer track record; empty cell filled; future
  (FP4 compute settled NEGATIVE, GLA/sparse = v13, cross-gen roofline spine).

## Gates (the paper equivalent of the two-gate rule)
1. BUILD gate: main.tex compiles to PDF (or is Overleaf-ready with a clear note on any missing local
   tooling), ≤8 pages two-column.
2. HONESTY gate: run a self-review pass (mirror the project's deep-research close-out) — reconcile
   every headline number to the record, verify citations, confirm the scorecard tally, confirm every
   §5.4 threat is disclosed, confirm zero "beat production" overclaims. Output a list of any number
   you could NOT reconcile and any citation needing verification.

## Then
- Update docs/paper-outline.md section statuses to "drafted."
- Commit to main (CLAUDE.md git rules: commit only when asked / at a clean stopping point; after
  push, git -C /Users/kienpham/Documents/flashattention-cuda pull --ff-only origin main).
- Report: word count, page count, unreconciled numbers, citations needing verification, and the
  punch-list left before arXiv.

Do NOT invent results. Do NOT claim v5 was benchmarked. Do NOT claim a SOTA win. When a number or
citation can't be verified, FLAG it — never guess.
```

## After the first draft (post-arXiv, non-gating)

Optional strengthening for reviewers, in priority order — all deferred to *after* the arXiv timestamp:
1. **Cross-generation spine:** add a clean A100 (sm_80) data point so the ~40 GB/s per-CTA ceiling is
   shown architecture-independent across T4→A100→B200→B300 (~1 week GPU). Dilutes "just a SKU bump."
2. **ncu on a privileged B300** — upgrades the counter-free %HBM proxy to profiler-validated on sm_103
   (currently validated only on T4). Belt-and-suspenders.
3. **v13 (GLA / sparse DSA/CSA)** — a new kernel arc / follow-up paper, not a prerequisite for this one.
