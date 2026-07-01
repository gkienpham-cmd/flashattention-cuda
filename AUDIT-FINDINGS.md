# AUDIT-FINDINGS — read-only audit run of 2026-07-02

Executed per `FABLE5-AUDIT-INSTRUCTIONS.md` (Claude Code, Fable 5). Mode: **read-only static
analysis** — no code, notebook, doc, or config file was modified; no git-writing command, install,
build, test, or GPU/notebook execution was run. The only write is this report. Suggested fixes appear
as unified diffs and were **not applied**.

Machine context: macOS author machine, no CUDA toolchain — kernel findings are *suspicions with a
GPU verification recipe*, not verified defects.

**Not-applicable (generic rubric, one line):** no HTTP/REST/auth/sessions/JWT/CORS/SQL/ORM/browser/
npm surface exists in this repo — injection/authn/IDOR/security-header/crypto/JWT/Promise/React/
`.env`-secret/npm-dependency passes are N/A and were skipped.

---

## Pass A — Orientation map

17 kernel directories, every one a `{kernel}.cu + binding.cpp` pair, and every one consistently
registered in `_SOURCES` (`bindings/load.py:21-39`) and `_MIN_CAPABILITY` (`fa_kernels/dispatch.py:17-60`).
Build gencode: default sm_75+sm_80 (`load.py:46-49`), GPU auto-detect with `FA_CUDA_ARCH` override
(`load.py:63-89`); the per-kernel `_ARCH` override table is currently empty (two deliberate
commented-out future entries: `v8_gqa_tc_sm80`, `v12_mla_tc` sm_103a — `load.py:52,59`).

| version | .cu | min cap | API entry (`fa_kernels`) | test tol (atol=rtol) | oracle (`reference.py`) |
|---|---|---|---|---|---|
| v1_naive | naive_attention.cu | (7,0) | `attention` | 1e-4 | `sdpa_reference` |
| v2_tiled | tiled_attention.cu | (7,0) | `attention` | 1e-4 | `sdpa_reference` |
| v3_online | online_attention.cu | (7,0) | `attention` | 1e-4 | `sdpa_reference` |
| v4_fused | fused_attention.cu | (7,0) | `attention` | 1e-4 | `sdpa_reference` |
| v5_wmma | wmma_attention.cu | (7,5) | `attention` | 2e-2 | `sdpa_reference` |
| v6_splitkv | splitkv_attention.cu | (7,0) | `attention` | 2e-2 | `sdpa_reference` |
| v7_paged | paged_attention.cu | (7,0) | `paged_attention` | borrows `tol_for("v6_splitkv")` (deliberate, test_correctness.py:235) | `sdpa_reference` |
| v8_gqa (+`_tc/_db/_occ/_ilp/_ss`) | gqa\*_attention.cu | (7,0); tc (7,5) | `gqa_attention` | 2e-2 | `sdpa_reference_gqa` |
| v9_fp8 | fp8_attention.cu | (7,0) | `fp8_attention` | 5e-2 | `sdpa_reference_gqa_fp8` |
| v10_nvfp4 | nvfp4_attention.cu | (7,0) | `nvfp4_attention` | 5e-2 | `sdpa_reference_gqa_nvfp4` |
| v11_mla | mla_attention.cu | (7,0) | `mla_attention` | 5e-2 | `sdpa_reference_mla` (+ `_materialized` absorption oracle) |
| v12_mla_tc | mla_tc_attention.cu | (10,0) | `mla_tc_attention` | 5e-2 | `sdpa_reference_mla` |

## Pass B — Wiring & registration integrity: **no half-wired version found**

Cross-checked (inventory agent + direct reads of `load.py`, `dispatch.py`, `__init__.py`,
`paged.py`, `reference.py`, `tests/test_correctness.py`):
- every `kernels/` dir ↔ `_SOURCES` entry with on-disk filename pair — **all 17 match**;
- every `_SOURCES` key ↔ `_MIN_CAPABILITY` key — **all 17 match**; gate comparison `have < need`
  (dispatch.py:75) is correct tuple semantics;
- all 9 `__all__` symbols (`fa_kernels/__init__.py:19-20`) have backing defs (`attention` in
  `__init__.py`; the rest in `paged.py`);
- every oracle referenced by tests exists in `reference.py`; the documented
  `repeat_interleave`-not-`repeat` GQA trap is correctly implemented (`reference.py:41-42`);
- every test backend name is a registered dispatch key; v7's missing `_TOL` entry is deliberate
  (tests use `tol_for("v6_splitkv")`, "same precision class" — `test_correctness.py:235,257,278`);
- gencode strings in `load.py` agree with the arch table in `roofline/archs.py` (sm_75/80/100/103),
  and `FA_CUDA_ARCH` accepts both `103` and `10.3` forms (`load.py:72-75`).

**Info (no action):** the two commented-out `_ARCH` entries (`load.py:52,59`) are documented staging
for future arch-specific builds, not dead wiring — they are unreachable by design until uncommented.

## Pass C — Pure-Python / CPU logic

Read in full: `roofline/model.py`, `roofline/predict.py`, `roofline/archs.py` (logic),
`bench/harness.py`, `bench/regime.py`, `fa_kernels/{dispatch,paged,reference,config,nvfp4_recipes}.py`.
Verified-clean highlights (checked, not just skimmed): decode `AI=2G/b` and MLA
`AI=2·h_q·(2L+R)/((L+R)·b)` formulas reproduce the documented values (CLI run: G=8 fp16 → AI 8.0;
MLA fp16 → 241.3 @ N_k=128K, 234.8 @ 8K — matches results/CLAUDE.md); NVFP4 `b=0.5625` counts the
micro-scale consistently in `model.py:31`, `harness.py:296`, `regime.py:196`; the NVFP4
quantize/dequantize pair round-trips (nibble order low=even/high=odd, two-level scaling keeps
micro-scales ≤ 448 by construction, `searchsorted` midpoints implement round-to-nearest with natural
clamp at 6.0); `build_paged_kv` permutation ↔ block-table inverse is correct (`pool[perm] = logical`,
`table=perm`); µs/tok convention agrees between `harness.py:246,293` and `regime.py:247,250`
(both `p50/(B·H_q·N_q)`); the v9 dequantize-once-outside-the-timed-lambda oracle fix is in place
(`harness.py:171-181`); v9/v10's `vs naive` correctly isolates bytes (v8.7 on an FP16 pool of the
same KV), v8's isolates packing (v7 on a G×-expanded pool), v12's isolates the engine (v11 on the
same latent pools).

### F1 — Pass C, **Medium**: `Arch.exp_per_s` is documented as the MUFU rate but never consumed; B300 MUFU bound silently uses the placeholder ratio

- **Location:** [roofline/model.py:118](roofline/model.py#L118) and [roofline/model.py:214](roofline/model.py#L214); contract at [roofline/archs.py:38-43](roofline/archs.py#L38), [roofline/archs.py:151](roofline/archs.py#L151), [roofline/archs.py:157](roofline/archs.py#L157)
- **Evidence:** `archs.py:38-40` — "exp_per_s is an absolute MUFU exp throughput where a vendor states
  it directly (B300); when None we fall back to mufu_ratio * (fp32 FMA/s)". But both MUFU sites in
  `model.py` compute `mufu_rate = (arch.fp32_cuda_flops / 2.0) * arch.mufu_ratio` unconditionally;
  `rg exp_per_s` shows no consumer anywhere.
- **Why it matters here:** on B300 the model derives the MUFU bound from two constants tagged
  *placeholder* (`mufu_ratio=0.25`) and *UNCONFIRMED* (`fp32_cuda_flops=105e12`) → 13.125 TExp/s,
  instead of the one tagged *FACT* (`exp_per_s=10.7e12`). The B300 `t_mufu` is ~1.23× too optimistic.
  No recorded limiter flips (MUFU share of decode is <3% in every measured step, and the sm_103
  2×-exp analysis compared measured-vs-vendor directly, not via this model), so this is a
  latent-model bug, not a published-number bug — hence Medium, not Critical.
- **Suggested fix (not applied):**
```diff
--- a/roofline/model.py
+++ b/roofline/model.py
@@ -115,7 +115,8 @@
         exp_count = float(B) * h_q * N_q * N_k             # one exp per (head, key) score entry
-        mufu_rate = (arch.fp32_cuda_flops / 2.0) * arch.mufu_ratio
+        mufu_rate = arch.exp_per_s if arch.exp_per_s is not None \
+            else (arch.fp32_cuda_flops / 2.0) * arch.mufu_ratio
         t_mufu = exp_count / mufu_rate
@@ -211,7 +212,8 @@
     exp_count = float(bh) * N_q * N_k
     # MUFU op/s ~= (FP32 FMA/s) * ratio. fp32_cuda_flops counts 2 FLOPs per FMA, so /2.
-    mufu_rate = (arch.fp32_cuda_flops / 2.0) * arch.mufu_ratio
+    mufu_rate = arch.exp_per_s if arch.exp_per_s is not None \
+        else (arch.fp32_cuda_flops / 2.0) * arch.mufu_ratio
     t_mufu = exp_count / mufu_rate
```
- **Verification (CPU, no GPU needed):** `python -m roofline.predict --arch sm_103 --mla --shape
  1x1x1x128 --kv-len 131072` — today `t_mufu ≈ 0.0013 ms / util 6.8%`; after the fix it should read
  ~1.23× larger (util ~8.3%), limiter unchanged (HBM). T4/A100 outputs must be byte-identical
  (`exp_per_s=None` there). Decide deliberately whether the vendor-peak or the measured-achievable
  5.33 TExp/s (archs.py:157-159) is the right numerator — that is an owner modeling choice.

### F2 — Pass C, **Low**: stale comment in `config.py` contradicts the measured v9 design ("fp8 is illegal on sm_75")

- **Location:** [fa_kernels/config.py:45](fa_kernels/config.py#L45)
- **Evidence:** `# arch gates which backends are legal (e.g. fp8 is illegal on sm_75). See dispatch.py.`
- **Why it matters here:** dispatch deliberately keeps `"v9_fp8": (7, 0)` — E4M3 is software-emulated
  on sm_75/T4 and the whole v9 Task-2 result was measured on a T4. The comment states the opposite of
  a load-bearing, measured design decision and could steer a future reader to re-gate v9 at (8,9)+.
- **Suggested fix (not applied):**
```diff
--- a/fa_kernels/config.py
+++ b/fa_kernels/config.py
@@ -44,5 +44,6 @@
     # --- hardware target ---
-    # arch gates which backends are legal (e.g. fp8 is illegal on sm_75). See dispatch.py.
+    # arch gates which backends are legal (e.g. v12_mla_tc needs sm_100+; v9_fp8 DOES run on sm_75 —
+    # E4M3 is software-emulated there, see dispatch.py "v9_fp8"). See dispatch.py.
     arch: str = "sm_75"
```
- **Verification:** none needed (comment-only); dispatch.py:37-39 is the authority.

### F3 — Pass C, **Low**: the harness's counter-free `L2!` flag fires only for `v9_fp8`, unlike `regime.py` which applies it to every backend

- **Location:** [bench/harness.py:347](bench/harness.py#L347)
- **Evidence:** `l2flag = " L2!" if (is_fp8 and hbm_pct == hbm_pct and hbm_pct > 100.0) else ""` —
  vs `regime.py:252` `l2_served = bool(eff_bw > hbm_peak)` (unconditional).
- **Why it matters here:** the L2-served test is a physical impossibility check (eff_bw > HBM peak ⇒
  data came from L2) and is equally valid for fp16/NVFP4/MLA rows. A v10/v11 harness row that streams
  from L2 would print `%HBM > 100` with no flag — a latent footgun for exactly the L2-residency
  confound the project spent v9 Task 1 removing. (No recorded run hit it: measured v10/v11 %HBM was
  ≤3%.)
- **Suggested fix (not applied):**
```diff
--- a/bench/harness.py
+++ b/bench/harness.py
@@ -344,7 +344,7 @@
-                l2flag = " L2!" if (is_fp8 and hbm_pct == hbm_pct and hbm_pct > 100.0) else ""
+                l2flag = " L2!" if (hbm_pct == hbm_pct and hbm_pct > 100.0) else ""
```
- **Verification (GPU, cheap):** on a Colab T4 run `python -m bench.harness --backend v10_nvfp4
  --decode --seq 1024 --dim 64 --heads 8 --gqa-group 8` (KV ≪ 4 MB L2, no flush in the harness) —
  an L2-resident row exceeding 100% should now carry the flag.

---

## Pass D — Kernel correctness reasoning (current versions; read-only, suspicions only)

Read in full: `kernels/v8_gqa_ss/gqa_ss_attention.cu` (the base), then the forks `v9_fp8`,
`v10_nvfp4`, `v11_mla`, and the `v12_mla_tc` scaffold + the v11 binding. v1–v5 were skimmed only via
their CLAUDE.md/decisions notes, per the timebox rule. **No correctness (High) flag survives** — the
documented traps are all handled:

- causal mask uses `i_q`, never the packed `m_row` (`gqa_ss_attention.cu:145`, and identically in all
  three forks); the monotone per-tile `c_lim` correctly replaces a per-key break and clamps at 0;
- online softmax + O-rescale are right: `α = exp(m_run − m_new)` multiplies `o_reg` **before** the PV
  add (`gqa_ss_attention.cu:172-186`); an all-masked 32-key group degenerates to `α=1, l_tile=0`
  (no-op), and a fully-future split writes `(m=−FLT_MAX, l=0, O=0)`, which the LSE merge absorbs
  (`exp(−FLT_MAX−m)=0`; `inv=0` guard at merge line 223 covers the all-empty row);
- barriers are warp-uniform (`active` is per-warp; idle warps still run the cooperative gather and
  both `__syncthreads`); masked lanes feed the reduction identities (`s=−inf`, `p=0`) and all 32 lanes
  always enter `warp_reduce_max/sum`;
- the paged-gather index math matches `build_paged_kv`'s layout; v10's nibble select (`t&1`,
  low=even), `t>>1` packed index, `t>>4` micro-scale index, `kE2M1` table, and two-level scale
  application all match `fa_kernels/paged.py`'s quantizer exactly (so the 5e-2 apples-to-apples oracle
  really sees the kernel's operands);
- v11's single transposed `sK_T[DQK][TN+1]` buffer is genuinely conflict-free for **both** reads: the
  QK read is stride-1 in the key index, and the PV read's odd stride (33 halves) maps the 32 lanes of
  an output group onto 32 distinct banks (checked arithmetically); PV stays within the first `DV`
  dims; the merge runs at width `DV`; smem totals match the documented ~46.1 KB at (576,512);
- v11's `binding.cpp` argument order matches `paged.py:339-340`'s positional call exactly; v12's
  scaffold fails loudly by design (`TORCH_CHECK(false)`) and its engine gate (0=fp8/1=nvfp4) matches
  `_MLA_TC_ENGINES`.

### F4 — Pass D, **Medium**: `choose_splits` hardcodes `num_sm = 40` (T4) in every current fork — on the 148-SM B300 the v11 headline shape ran with ~3.8× fewer splits than the kernel's own heuristic would pick there

- **Location:** [gqa_ss_attention.cu:248](kernels/v8_gqa_ss/gqa_ss_attention.cu#L248) /
  [fp8_attention.cu:258](kernels/v9_fp8/fp8_attention.cu#L258) /
  [nvfp4_attention.cu:288](kernels/v10_nvfp4/nvfp4_attention.cu#L288) /
  [mla_attention.cu:301](kernels/v11_mla/mla_attention.cu#L301), each called with the default
  (e.g. `mla_attention.cu:362`).
- **Evidence:** `int choose_splits(int64_t B, int H_kv, int G, int N_q, int N_k, int num_sm = 40)` …
  `const int target_blocks = 2 * num_sm;` — and every call site omits `num_sm`.
- **Why it matters here:** the heuristic targets `2×SM` blocks, but `num_sm` is pinned to the T4's 40
  even on the B200/B300 (148 SMs) where the paper's measured core ran. Worked examples:
  - **v11 @ (h_q=128, B=1, N_k=8192)** — the B300 headline shape: `row_tiles=16` → `by_occ =
    ceil(80/16) = 5` splits (80 blocks on 148 SMs), where a 148-SM-aware heuristic picks
    `ceil(296/16) = 19` (304 blocks). Since the kernel is latency/per-CTA-bound and splits divide the
    serial key range, up to ~3.8× of the measured v11 B300 µs/tok — and therefore part of the
    **"~4× slower than torch dense-MQA"** gap magnitude — may be launch-heuristic, not engine. The
    *direction* of the conclusion survives (v12/tcgen05 independently proved the engine story), but
    the 4× magnitude is confounded.
  - **v10/v8.7/v9 B=1, H_kv=1 sweeps are NOT affected** (`by_occ=80` → `S_CAP=32` binds identically
    at 40 or 148 SMs), so the flat-%HBM / no-knee / per-CTA conclusions stand unchanged.
  - **H_kv=8 and large-batch rows are partially affected** (e.g. B=64,H_kv=1: S=2 vs 5), so the
    absolute "occupancy lever ~4×, caps ~2.3% HBM" magnitudes on Blackwell are heuristic-dependent;
    directions unchanged.
- **Suggested fix (not applied; same one-liner in each fork's host entry):**
```diff
--- a/kernels/v11_mla/mla_attention.cu
+++ b/kernels/v11_mla/mla_attention.cu
@@ -359,8 +359,11 @@
     const int64_t BH_kv = (int64_t)B * H_kv;   // = B
-    const int S = choose_splits(B, H_kv, G, N_q, N_k);
+    int num_sm = 40;   // fallback = T4
+    cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, q.get_device());
+    const int S = choose_splits(B, H_kv, G, N_q, N_k, num_sm);
     const int chunk = ceil_div(N_k, S);
```
  (Note: this changes the launch configuration, i.e. it is a *new measurement condition* — prior T4
  numbers are unaffected (40 is correct there), but any refreshed B300 numbers should be labeled.)
- **Verification (GPU — the highest-value rerun this audit found):** on a B300 rental, print `S` and
  A/B `python -m bench.regime --backend v11_mla --dim 576 --gqa-group 128 --h-kv 1 --kv-lens 8192
  131072` with `num_sm` 40 vs 148. If µs/tok drops materially (up to ~3.8×), re-derive the v11
  "4× vs torch" sentence in `results.md` Step 11 before a reviewer does; if it doesn't, the per-CTA
  reading gets *stronger* (split parallelism wasn't binding) — either outcome is paper-grade.

### F5 — Pass D, **Low**: two latent scale ceilings (fine at every recorded shape; noted so they fail loudly if the sweep grows)

- **Location:** merge launches, e.g. [mla_attention.cu:399-400](kernels/v11_mla/mla_attention.cu#L399)
  (`mgrid(N_q, B*H_q)`); gather index, e.g. [gqa_ss_attention.cu:129](kernels/v8_gqa_ss/gqa_ss_attention.cu#L129).
- **Evidence:** `dim3 mgrid((unsigned)N_q, (unsigned)(B * H_q));` — CUDA grid.y caps at 65535; and
  `(int64_t)(pb * page_size + off)` — the product is computed in int32 *before* the widening cast.
- **Why it matters here:** merge grid.y = `B·H_q` exceeds 65535 at e.g. B=512, h_q=128 MLA (recorded
  max was B=64 → 8192, fine); the int32 flat-token product overflows only past 2^31 pooled tokens
  (≫ any real pool — informational). Neither bites any recorded run; both are one-line guards
  (`TORCH_CHECK(B*(int64_t)H_q <= 65535, …)`; cast `(int64_t)pb * page_size + off`).
- **Verification:** none needed now; the TORCH_CHECK makes the ceiling loud if a future sweep crosses it.

## Pass E — Data & document consistency (spot-checks; notebooks + `archs.py` = ground truth)

Not an exhaustive reconciliation — the repo's own per-step deep-research passes already did that.
Spot-checked and **consistent**: B300 arch constants (148 SMs / 132.6 MB L2 / 2032 MHz) agree across
`archs.py`, CLAUDE.md, and the docs; v10 B300 headline µs/tok (24798 vs 27579 @ 2M) appears
identically in `decisions.md:1106` and `notebooks/v10_b300_regime_outputb300.ipynb`; v11's
0.23–0.29× / 0.75 TFLOP/s numbers carry their close-out caveats in `results.md`; all ridge values in
docs reproduce from `archs.py` constants (T4 25.3/203.1/406.25; B300 312.5/625/1875 — CLI-confirmed).

### F6 — Pass E, **Low**: the explicitly-REFUTED "FA4 stops at B200" phrasing survives as a live comment in `roofline/archs.py`

- **Location:** [roofline/archs.py:126](roofline/archs.py#L126) (also, as *historical record only*:
  `docs/decisions.md:765`, `docs/decode-replan.md:359`)
- **Evidence:** archs.py:126 — "(first open roofline-documented FA decode study on sm_103; FA4 stops
  at B200)". But `docs/results.md:1959` records this exact claim as "**REFUTED as stated**", and
  CLAUDE.md:509 says the north-star line was fixed to "deployed but never characterized on sm_103".
- **Why it matters here:** the close-out's claim-discipline fix didn't propagate to this live code
  comment — the one place a future reader (or a paper reviewer grepping the repo) will see the killed
  framing stated as current fact. The decisions.md/decode-replan.md occurrences sit in dated
  historical sections and are arguably deliberate record-keeping; only the archs.py comment is live
  guidance.
- **Suggested fix (not applied; comment-only, constants untouched):**
```diff
--- a/roofline/archs.py
+++ b/roofline/archs.py
@@ -125,3 +125,4 @@
-# (first open roofline-documented FA decode study on sm_103; FA4 stops at B200). Device constants now
+# (first open roofline-documented FA decode study on sm_103; FA4 targets/benchmarks B200 and is
+# deployed-but-never-characterized on sm_103 — results.md close-out #9). Device constants now
```
- **Verification:** none needed (comment-only); `results.md:1959` is the authority.

### F7 — Pass E, **Low**: `roofline/model.py`'s L2-confound comment still declares the decode limiter "OPEN" — v9 Task 1 closed it

- **Location:** [roofline/model.py:198-200](roofline/model.py#L198)
- **Evidence:** "v9 Task 1 (root T4: lock clocks, flush L2, sweep N_k 1K..128K, measure L2 hit-rate)
  earns or overturns the per-CTA verdict. Until then the bytes-vs-per-CTA decode limiter is OPEN."
- **Why it matters here:** v9 Task 1 ran (2026-06-28, ncu-validated) and settled it — per-CTA-bound,
  confound-free (results.md Step 9 Task 1; reaffirmed on B200/B300 in Step 10). The project's own
  precedent is to refresh stale model.py commentary (it did exactly that for the "split-KV fills the
  SMs" comment after v7), and this paragraph is the first thing a reader of the fused-decode branch
  meets.
- **Suggested fix (not applied):** replace the final sentence with e.g. "v9 Task 1 (2026-06-28,
  root T4, ncu-validated) SETTLED it: per-CTA-bound, confound-free — see results.md Step 9 Task 1;
  the counter-free eff_bw test below remains the per-run guard."
- **Verification:** none needed (comment-only); results.md Step 9 Task 1 is the authority.

## Pass F — Robustness observations (observations with intent, not defects)

- `FA_CUDA_ARCH` env fallback (`load.py:72-75`): intentional escape hatch for CUDA-version-gated
  Blackwell targets; accepts `103`/`10.3`/`103a`. Silent typo risk (e.g. `FA_CUDA_ARCH=57`) surfaces
  only at nvcc time — acceptable for a research repo, documented in the file.
- `torch` intentionally unpinned / `dependencies = []` (`pyproject.toml`): deliberate — a pin would
  fight the Colab/rental images (per instructions, not flagged).
- `nvidia-*-cu12` include shim (`load.py:92-104`): deliberate workaround for slim CUDA images missing
  dev headers (the documented cusparse-header fix from the v10 B200 run); returns `[]` harmlessly on
  full toolkits.
- Cross-session clock/throttle caveats: already first-class in the docs (clock~CUR/MAX on every bench
  row; regime.py locks clocks where root allows); no further exposure found.
- `subprocess` usage in `bench/{harness,regime}.py` is list-args `nvidia-smi` only — no `shell=True`.
- **Secrets grep: none found** (py/sh/toml/md *and* the committed notebooks: no ghp_/hf_/AKIA/private-key
  patterns). **Dynamic-exec grep: none found** (no `eval(`/`exec(`/`os.system`/`subprocess(shell=True)`
  in any `.py`).

---

## Summary

| Severity | Count | IDs |
|---|---|---|
| Critical | 0 | — |
| High | 0 | — |
| Medium | 2 | F1 (exp_per_s never consumed), F4 (choose_splits hardcodes 40 SMs) |
| Low | 5 | F2 (config.py stale fp8 comment), F3 (L2! flag fp8-only), F5 (scale ceilings), F6 (archs.py refuted phrasing), F7 (model.py "OPEN" comment) |
| Info / N-A | 2 | web-app rubric N/A; `_ARCH` commented staging entries |

**Wiring (Pass B): fully clean** — all 17 versions consistently registered end-to-end; no half-wired
backend, no dangling `__all__` symbol, no missing oracle, no manifest/disk mismatch.

**Top 3 for the owner's scarce GPU time:**
1. **F4 — one A/B run on the next B300 rental** (print `S`; rerun v11 @ 576/512 with a
   148-SM-aware `choose_splits`). It either sharpens or partially deflates the paper's "4× vs torch"
   magnitude *before* a reviewer asks — and it's minutes on a box you'll rent anyway. The v10 B=1
   regime conclusions are unaffected either way.
2. **F1 — no GPU needed**: two-line model fix + re-print the sm_103 predictions; decide explicitly
   whether the MUFU numerator should be the vendor 10.7 TExp/s or the measured-achievable 5.33.
3. **F3 — free rider**: un-gate the `L2!` flag before the next harness run on any GPU; it costs
   nothing and removes a footgun from every future decode row.
