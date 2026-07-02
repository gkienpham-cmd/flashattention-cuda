# Fable 5 — Final Submission-Readiness Check Brief
## Paper: "Roofline-Guided Characterization of Attention Decode on Blackwell" (Kien Pham)

A prior review pass verified citations and headline numbers against external
sources and the repo's data of record. This brief tells you what is ALREADY
CLEARED (do not re-spend budget) and what remains OPEN (please close).

---

### ALREADY VERIFIED — do not re-check
- **All 17 references are real and correctly attributed** (checked vs arXiv /
  primary sources): [1] 2501.01005, [6] 2603.05451, [7] 2505.21487 (Zadouri,
  Strauss, Dao — GTA+GLA both correct), [8] 2602.10718, [9] 2505.11594,
  [10] 2603.00040, [11] 2605.04178, [17] 2512.02189, plus [2]–[5],[12]–[16].
  No fabricated citations. No mis-ID'd authors. [11] and [17] (both Jarmusch &
  Chandrasekaran) are correctly kept distinct.
- **External claims match sources**: FA4 1605 TFLOP/s / 71% on B200 [6];
  SnapMLA 3000 GB/s + 580 TFLOP/s on H800 [2]; Jarmusch 1.31% MAE vs >95%
  naive-roofline error [11]; GLA "up to 2×" [7].
- **Internal numbers reconcile to data of record**: MLA AI 235/470/835 and
  ridges 312.5/625/1875 reproduce from roofline.predict; Section V-C numbers
  (5.8%, 21.7%, and the K=576/N=131072 point 1086 vs 739) are exact against
  notebooks/stage_a_results.csv; v12 Table IV %-of-peak arithmetic checks;
  Table I arch constants match NVIDIA B300 specs.
- **Novelty claim is defensible**: surveyed works (GLA=H100, SnapMLA=Hopper,
  SageAttn3=RTX5090 prefill, FA4=B200 prefill) leave the sm_103-decode-roofline
  cell empty. Scope is honestly hedged ("complement, not beat").

### OPEN ITEMS — please close before submission

1. **[Consistency — HIGH] "seven decode steps" vs 8 table rows.** Abstract,
   Sec III-D, and Conclusion all claim the per-CTA layer "matched at all seven
   decode steps." Table II lists EIGHT decode rows: v6, v7, v8, v8.7, v9, v10,
   v11, v12. The tally reconciles only if v8.7 is folded into v8. Because
   "L2 correct at all seven" is a headline claim, add a one-line footnote to
   Table II: "v8.7 is a sub-step of v8; the seven main decode versions are
   v6–v12." Confirm the L1 wording ("wrong regime at every single-stream step")
   is consistent with the same 7-vs-8 accounting.

2. **[Consistency — MED] "twelve kernels" vs the enumerated list.** Abstract
   says "twelve attention kernels (naive, tiled, fused, split-KV, GQA-packed,
   score-stationary, FP8, NVFP4, MLA, and tensor-core tcgen05)" — that
   parenthetical lists only 10 names and omits online-softmax (v3), WMMA (v5),
   and paged (v7). A reader counts 10 under the word "twelve." Fix: either make
   the list clearly illustrative ("e.g., ...") or complete it to all twelve.

3. **[Consistency — LOW] CUTLASS version across experiments.** v12 (Sec IV-B)
   uses "CUTLASS 4.6, example 77"; the Sec V-C boundary test uses "CUTLASS
   v4.5.2, example 72a." These are two different runs/dates so both can be
   correct — please confirm the version difference is intentional and, if so,
   consider a half-sentence noting they are separate measurement campaigns so
   a reviewer doesn't read it as an error.

4. **[Reviewer-anticipation — already disclosed, verify wording holds]** The
   following are honestly flagged in Sec V-D; confirm no stronger claim leaks
   elsewhere in the text:
   - ~4× cuBLAS/MQA gap cause is INFERRED, not torch-profiled (Sec IV-B
     already says so — check the abstract/intro don't over-state it).
   - Stage A is cross-harness (CUTLASS FP4 example vs cuBLAS FP8); the kill
     correctly rests on peak-fraction/roofline, not the head-to-head ratio.
     Confirm no sentence presents the FP4/FP8 ratio as a clean comparison.
   - %HBM proxy is ncu-validated only once (root T4); every Blackwell reading
     is proxy, not counter. Confirm no Blackwell number is called "measured by
     ncu."
   - v5 WMMA prefill was never benchmarked (prediction only) — confirm it is
     never counted as a measured result in any summary sentence.

5. **[Check — LOW] Affiliation.** Byline reads "Tufts University." Confirm this
   is current/correct as intended (author-supplied; not independently checkable).

### VERDICT (pre-Fable)
No hallucinations found. No fabricated or mis-attributed citations. All
load-bearing numbers trace to the repo's data of record. Remaining items are
internal-consistency polish (items 1–3) and confirm-the-hedges (item 4), not
substantive errors. The paper is close to submission-ready pending items 1–2.

---

## Resolution (2026-07-02, applied same day)

All open items are closed; the paper edits are in `flash-paper/` (commit of record on `main`).

1. **[HIGH] 7-vs-8 decode rows — FIXED.** The Table II (scorecard) caption in
   `flash-paper/sections/method.tex` now states: "v8.7 is a sub-step of v8 (its inner-loop
   relayout); the seven main decode versions are v6--v12." The four "seven decode steps"
   sentences (abstract, intro contribution 2, Sec III-D, conclusion) are consistent with this
   accounting, and the tally is robust either way (v8.7's L2 verdict is also a hit).
2. **[MED] "twelve kernels" vs 10 names — FIXED.** The abstract parenthetical (in
   `main.tex` and `arxiv-abstract.txt`) now lists exactly the twelve version names in arc
   order: naive, tiled, online-softmax, fused, tensor-core WMMA, split-KV, paged, GQA-packed,
   FP8, NVFP4, MLA, tcgen05. (Score-stationary left the list; it is v8's sub-step per item 1.)
3. **[LOW] CUTLASS 4.6 vs v4.5.2 — CONFIRMED INTENTIONAL + NOTED.** Both are correct per the
   record: v12's characterization ran 2026-06-30 (box with CUTLASS 4.6, example 77); the
   Stage-A boundary test ran 2026-07-02 on a different rented B300 (CUTLASS v4.5.2, example
   72a). Sec V-C now says so in-line ("a separate measurement campaign on a different rented
   box than v12's, hence the different CUTLASS version").
4. **[Hedge audit] — PASSES, no edits needed.** Grep-audited: (a) the 4x cuBLAS-gap cause
   appears only in Sec IV-B with its "attributed, not torch-profiled" hedge; the abstract and
   intro never mention it. (b) The FP4/FP8 ratio appears only alongside the cross-harness
   caveat; every summary sentence rests on peak-fraction/roofline. (c) ncu is credited only
   for the root-T4 proxy validation; every Blackwell reading stays proxy-labeled. (d) v5 is
   marked "not measured (prediction only)" in the scorecard, Sec IV-A, and Sec V-D.
5. **[Affiliation] — CONFIRMED.** "Tufts University" is author-supplied and intended.

Also produced the same day: a first-person copy of the paper
(`flash-paper/first-person/`, "I/my" instead of "we/our", sole-contributor phrasing), and the
figure label "RGD (ours)" was changed to the pronoun-neutral "RGD (this work)" in
`v12-related-work-landscape.svg`/`.pdf`.
