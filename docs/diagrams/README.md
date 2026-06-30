# B300 decode-kernel research diagrams

Standalone SVGs accompanying [`../b300-decode-research.md`](../b300-decode-research.md) and the post-v6
re-plan [`../decode-replan.md`](../decode-replan.md). Each is self-contained (embedded colors, white
background) — open in a browser, macOS Preview/Quick Look, or VS Code and it renders correctly on its own.

**Post-v6 figures (the new measured result + the reorder):**

| File | What it shows |
|---|---|
| `decode-roofline-crossover.svg` | **The headline — REFRESHED to MEASURED (v7, 2026-06-27).** Achieved %HBM vs effective batch `BH`. The v7 `--batch` sweep measured it **FLAT at 9.4–12.4% from BH=8→512** (d=64 and d=128 lines); the predicted rising crossover is now **ghosted and labelled REFUTED**. The wall is per-CTA (32 KB smem → 2 blocks/SM; 1-of-8 warps at N_q=1), so batch only adds waves. The next lever is GQA M-packing (per-CTA efficiency), not fewer bytes. |
| `gqa-mpacking.svg` | The occupancy lever: packing `G` query heads into `M` turns the `M=1` GEMV (KV read `G×`) into an `M=G` GEMM (KV read once), raising `AI = 2/b → 2G/b` and re-engaging tensor cores. |
| `splitkv-lse-merge-dataflow.svg` | The two-kernel schedule the roofline can't see — split-KV partial + LSE merge, and where the launch / under-occupied-merge / warp-shuffle overhead lives. |
| `b200-b300-seams.svg` | Verified B200→B300 deltas with the **flat-bandwidth** surprise highlighted; the decode consequence of each. |
| `build-roadmap-v6-v11.svg` | **Reordered** plan: v6 split-KV → v7 paged KV → **v8 GQA M-packing (the reorder)** → v9 FP8 KV → v10 NVFP4 + asymmetric precision (headline) → v11 MLA / speculative. Old FP8/FP4-before-GQA order shown struck through. |

**v9 result figures (FP8 KV + the regime verdict — 2026-06-28, hand-authored from the run-of-record
`notebooks/v9_fp8_gate_output.ipynb` + `notebooks/v9_task1_regime_output.ipynb`):**

| File | What it shows |
|---|---|
| `v9-task1-regime.svg` | **The bandwidth verdict, EARNED (per-CTA-bound).** Achieved %HBM vs N_k (log-x, **L2-flushed, clocks LOCKED 1590 MHz, root T4**, d=128). `v8_gqa_ss` FP16 plateaus at a hard **~28% cap at high occupancy (H_kv=8)** and ~14% at low occupancy (H_kv=1) — **no bandwidth knee at the 4 MB L2 crossing**, never near the ~70% achievable ceiling; `v9_fp8` sits ~10% (fewer bytes over a per-CTA-bound time). The ncu callout (DRAM 12.85% / L2-hit 1.1% past L2, matching the counter-free 13.8%) settles it: **HBM-served yet 13% busy → per-CTA / low-MLP bound, not bandwidth-bound.** (The notebook's matplotlib `savefig` ran on the Colab host and was never committed; this is the committed hand-authored version of the same data.) |
| `v9-fp8-win-anatomy.svg` | **The FP8 win, anatomized honestly.** Left: the clock-matched `vs naive` (FP8÷FP16) speedup at d=128 is real but shrinks monotonically 1.52×(G1)→1.05×(G32), the mechanism entangled (↑G ⟺ ↓H_kv ⟺ ↓working set) and **not L2-BW-saturated** (ncu L2 thrpt <3%) → a bytes-sensitive load-*latency* effect that **flips negative under L2-flush** (the sm_75 dequant ALU tax). Right: the verdict cards — **capacity (2×, by construction) and accuracy (~7e-4 RMSE) are DURABLE; latency is FRAGILE/CONDITIONAL** because decode is per-CTA-bound (~28% cap). FP8/NVFP4 = capacity + accuracy, not a decode-latency play. |

**v8.5/v8.6 past-L2 re-test figure (2026-06-29 — hand-authored from `notebooks/v8_5_v8_6_pastL2_regime_output.ipynb`):**

| File | What it shows |
|---|---|
| `v8_5_v8_6_pastL2.svg` | **The reopener, resolved + the one surprise.** Clock-robust speedup vs Cut-1 (`time(Cut-1)/time(arm)`, L2-flushed). LEFT (B=1, H_kv=1 N_k sweep): `ss` (v8.7 relayout) holds ~1.45× across the whole sweep incl. deep past the 4 MB L2 crossing, while `db` (v8.5), `occ` (v8.6) and `ilp` (v8.6) sit **flat at ~1.0 past L2** → the v8.5/v8.6 nulls were **not** L2 artifacts; the B=1 floor is the serial recurrence. RIGHT (batch sweep, N_k=32768 past L2): `db`/`ilp` stay ~1.0 but the **occupancy arm jumps to 1.47× at B=32 / 1.38× at B=64** — occupancy revives as a live serving-regime lever the B=1-only v8.6 measurement missed (4 blocks/SM only pays once the grid fills it). |

**v7 deep-research close-out figures (2026-06-27 — accompany [`../v7-deep-research.md`](../v7-deep-research.md)):**

| File | What it shows |
|---|---|
| `v7-vs-sdpa-batch-crossover.svg` | **The v8 motivation.** v7's speedup over torch SDPA vs decode batch B (log). v7 wins **only at B=1** (2.65×/1.46×); SDPA overtakes by **B=8** (v7 drops to ~0.5×/0.32×) and stays ahead through B=64. v7 is SM-saturated at B=1 → per-token cost flat, while SDPA's collapses ~4.5× with batch. The serving regime (B≥8) is what **v8 must reclaim**. |
| `per-cta-limiter-anatomy.svg` | **Why batch can't help.** One T4 SM holds only **2 resident blocks** (32 KB smem each / 64 KB); inside each block, all 8 warps load KV but at `N_q=1` only **warp 0 computes**. Batch adds serial *waves* at the same 2-blocks/SM cap → flat %HBM. Right panel: v8 M-packing lights up G warps + GEMV→GEMM (M≥16 tensor-core gate). |

**v11 result figure (MLA latent-KV decode, B300/sm_103 — 2026-06-30, hand-authored from `notebooks/v11_mla_gate_b300output.ipynb` + `v11_regime.txt`):**

| File | What it shows |
|---|---|
| `v11-per-cta-wall-b300.svg` | **The MLA decode limiter, EARNED (per-CTA-bound to 2M tokens).** Achieved %HBM (`eff_bw = logical KV bytes / time ÷ 8 TB/s`) vs N_k (log-x) on B300/sm_103, NVFP4 latent. The measured line is **flat at ~0.01% (eff_bw ~0.9 GB/s logical, ~14 GB/s on a 16× physical-reread upper bound)** across the whole 8K→2M sweep — **no bandwidth knee** at the L2 crossing (WS passes the **measured 132.6 MB** L2 between 131K=42 MB and 524K=170 MB), and at **N_k=2M the WS is 679 MB = 5.1× the L2** yet %HBM stays ~0%. The pre-registered counter-prediction (%HBM climbs IF bandwidth-bound) is ghosted in purple. **Honesty guardrail baked in:** at this high-AI MLA shape (nvfp4 AI 835 ≫ the 312.5 ridge), flat-%HBM-near-0 only proves *not* bandwidth-bound — the per-CTA verdict rests on the **0.75 TFLOP/s = <1% of either compute peak** and the **~4× cuBLAS gap** (warp-per-head CUDA-core GEMV vs torch's M=128 batched-GEMM), not on %HBM alone. **Two-layer prediction:** pure roofline said FLIP to MMA-bound (the arc's first); the per-CTA-corrected (real) layer said it stays per-CTA on CUDA cores (M=128 packs as 16 blocks × 8 warps, not one GEMM) → **measurement confirmed the real layer.** Counter-free (no ncu; proxy ncu-validated once on T4), clocks idle-pinned (2032 MHz) not hard-locked. |
| `v11-cublas-gap.svg` | **The headline gap (the v12 motivation).** Two panels: (A) `vs_sdpa` = 0.29×/0.24×/0.23× at N_k 2048/8192/16384 (= 3.4×/4.2×/4.3× **slower** than torch dense-MQA, worsening with N_k); (B) our warp-per-head **CUDA-core GEMV** (tensor cores idle, M=128 packed as 16 blocks × 8 warps) vs torch's one **batched cuBLAS** path. The gap proves the M=128 shape is tensor-core-*friendly* and the CUDA-core default is the wrong tool on Blackwell → **v12 = put M=128 on tcgen05.** Honesty card: the "torch hits an M=128 TC GEMM" *cause* is an **inference** (the torch baseline is an FP32 batched-GEMV M=1, no torch-side profile captured), not a measured mechanism. |
| `v11-roofline-two-layer.svg` | **The two-layer prediction, drawn.** A B300 log-log roofline (HBM 8 TB/s, FP16-TC 2.5 PF, ridge AI=312.5). The AI ladder GQA-8 fp16 **8** → MLA fp16 **234.8** (0.75× ridge) → MLA nvfp4 **835** (2.67× ridge). The *pure-roofline* prediction sits ON the ceiling (★, the arc's first predicted compute-flip); the *per-CTA-corrected (real)* prediction is dragged to the floor (dot); the **MEASURED** point lands at the per-CTA floor (0.75 TFLOP/s = 0.03% of FP16-TC peak). Green "real layer landed" / orange "compute-flip did NOT realize" cards. |
| `v11-capacity-accuracy.svg` | **The durable win and its FP8-floor cost, side by side.** LEFT (log-y capacity, N_k=65536): MHA 4295 MB → GQA-8 268 MB (16×) → **MLA·NVFP4 21 MB** = 202× vs MHA / 12.6× vs GQA-8 / 99.5%, decomposed **56.9× shape × 3.56× NVFP4 bytes** (durable by construction). RIGHT (RMSE): kernel-vs-oracle ~2e-5 (exact) · FP8-latent ~7e-4 · **NVFP4-latent ~2.5e-3 = 3.5× FP8** (the accuracy floor, latent-width-independent; *our substrate-specific measurement, not a published constant*). |
| `v11-nsys-schedule.svg` | **The kernel schedule (nsys 2025.3.2, sm_103 — provenance PAID).** `mla_partial_kernel<32,576,512>` owns **99.9%** of GPU time (100 launches, 387 ms each @ N_k=1M — cross-checks the regime row `3023 µs/tok × 128 = 387 ms` to 0.03%); `mla_merge_kernel` = **0.0% (3.5 µs/call, 0.0009%** — a 44× correction to the digest's "0.04%"); torch quant + page-table setup < 0.01%. The wall is *inside* the partial kernel (per-CTA), not the launch/merge structure. |
| `v11-to-v12-tcgen05.svg` | **Where the lever lives → v12.** Left-to-right: v11 measured ("shape correct, **wrong engine** — CUDA-core GEMV at 0.75 TFLOP/s, ~4× < cuBLAS") → the M=128 GEMM is real (cuBLAS proves it) → **v12 = fork CUTLASS ex77, 2-SM, Arm 1 FP8 MMA (M≥64) → Arm 2 native NVFP4 (M≥128, K=256, TN)** → the **pre-registered prize**: even with TC, decode stays SMEM-BW / pipeline-depth-bound → likely won't reach the FP4 peak (*that shortfall is the paper-grade result*). Orange banner: data-motivated, **not yet measured.** Footer corrects the tcgen05 gates (FP8 M≥64 / NVFP4 block-scaled M≥128). |

**Original decode-arc figures (still accurate):**

| File | What it shows |
|---|---|
| `decode-roofline.svg` | Decode arithmetic intensity (≈ 2/b FLOP/byte) vs the B300 FP4 ridge — why decode is ~300–500× memory-bound, and how FP4 KV / GQA / MLA walk it toward the ridge. |
| `asymmetric-precision-dataflow.svg` | The "smart trick": FP4-safe `P·V` (convex combination) vs precision-critical `Q·Kᵀ` + softmax (exp-amplified), with the KV cache in NVFP4. |
| `split-kv-schedule.svg` | FA4's `(1, heads, batch)` grid leaving ~147 SMs idle vs split-KV across all 160 SMs + a log-sum-exp merge. |

> Note: GitHub's inline Markdown preview sanitizes some SVG; if a diagram looks off in the web UI,
> open the raw file or view it locally — it renders fully in any standalone SVG viewer.
