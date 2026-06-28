# v10 B300 rental runbook — vast.ai, ncu-critical, tiered for cost

The smooth-and-cheap operating manual for v10's **paid measured core** (the T3 knee-hunt + sm_103
2×-exp delta + FlashInfer comparator). Distilled from the B300-research pass (2026-06-29). The notebooks
this drives: `notebooks/v10_b300_regime.ipynb` (headline), `v10_b300_exp_ablation.ipynb` (T2),
`v10_b300_comparators.ipynb` (FlashInfer). Build auto-detects sm_103 (`bindings/load.py` —
`_detect_arch_flags`), so no manual gencode editing.

## 0. The one thing that decides the session: ncu must work

`ncu`'s `ERR_NVGPUCTRPERM` is a **host kernel-module gate** (`NVreg_RestrictProfilingToAdminUsers`,
default `=1`), set when the *host* loads the `nvidia` driver. **You cannot fix it from inside a tenant
container** — not as container-root, not by editing `/etc/modprobe.d`, and **`--cap-add SYS_ADMIN` is
necessary but not sufficient** if the host kept the default. This is why every prior containerized
rental (vast.ai/RunPod/Lambda) gave `ERR_NVGPUCTRPERM` and why v9 Task-1 only worked on a root T4.

**→ Method = PROBE-THEN-KEEP.** There is *no* vast.ai filter that advertises "ncu works." Rent, run the
probe **first thing**, and **destroy + re-rent another host** if it fails. Budget 2–3 host attempts.
Prefer hosts offering **privileged / bare-metal / `--cap-add`** launch.

```bash
# THE PROBE — run before any real work. If this errors, destroy the instance and try another host.
nvidia-smi -q | grep -i "driver version"
nvcc --version | grep release
ncu --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed --target-processes all \
  python -c "import torch; (torch.randn(4096,4096,device='cuda')@torch.randn(4096,4096,device='cuda')).sum().item()"
# PASS = prints a DRAM throughput %. FAIL = ERR_NVGPUCTRPERM -> wrong host, re-rent.
```
The **counter-free `%HBM` / `L2!` sweep still runs without ncu** (intra-run, clock-robust) — so a
non-privileged box still yields the headline verdict; ncu only *corroborates* it. But the paper wants
the ncu L2-hit-rate, so hunt for a privileged host.

## 1. Hardware path (decided): B200 dev-rung → B300 record

| Step | GPU | ~Cost | Purpose |
|---|---|---|---|
| **Dev-rung** | **B200** (sm_100, ~$3.4/hr floor) | cheap | De-risk: build (validate sm_100 gencode), install FlashInfer, **run the ncu probe** (confirm the privileged-host workflow), smoke test. NO record sweeps. |
| **Record** | **B300** (sm_103, ~$5.4–6/hr floor) | the paid core | Arch-measure + the tiered knee-hunt + exp ablation + comparator. |

⚠️ Pricing correction from research: the **~$2.45 "spot" figure was Spheron, not vast.ai** — budget
**~$5.4–6/hr on-demand** for B300. **Target the discrete B300, not GB300** (GB300/NVL72 not on vast.ai).

## 2. Instance + image + driver

- **Search:** `https://cloud.vast.ai/?gpu_option=B200` then `…=B300` (or CLI `vastai search offers
  'gpu_name=B300'`). Trust the `gpu_name` token, not the spec row (some mislabel the chip).
- **Image:** **`nvidia/cuda:12.9.0-devel-ubuntu24.04`** (ships `nvcc`; CUDA 12.9 is the floor for sm_103,
  and still builds the T4 sm_75 baseline — do NOT jump to CUDA 13.x, which prunes old arches).
- **Driver:** **R580 (≥580.x)** recommended; R570 is the Blackwell minimum. The host's advertised "Max
  CUDA" is the driver ceiling; your image's toolkit is what compiles — they must be compatible.
- **PyTorch:** stable cu128 wheels lagged (silent cu124 fallback). Use **nightly cu129**:
  `pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu129` — or an NGC
  `nvcr.io/nvidia/pytorch:25.xx-py3` container. Verify `torch.version.cuda` ≥ 12.9 and
  `torch.cuda.get_device_capability() == (10, 3)` before trusting a build.

## 3. Env setup (vast.ai gotchas)

```bash
. /venv/main/bin/activate                 # torch lives here; a bare `python` misses it
export PATH=/venv/main/bin:$PATH           # so `!python -m ...` notebook cells resolve
pip install -q ninja pytest numpy matplotlib   # install explicitly into the venv
```
In notebooks use `%pip` / `import sys; !{sys.executable} -m pip ...`. **Do NOT** symlink system python3
over the venv. Disk is **ephemeral** — `git clone` per-rent, **commit + push before destroy**. (The
`v10_b300_*` notebooks already encode the venv/PATH dance in their deps cell.)

## 4. Build (auto-detected gencode — no editing)

`bindings/load.py` `_detect_arch_flags()` queries `torch.cuda.get_device_capability()` → emits
`compute_103,code=sm_103` (+ PTX) on B300, `compute_100` on B200, `compute_75` on Colab. **Our v10
kernel uses no tcgen05 / FP4-MMA, so plain `sm_103` is correct and compiles on CUDA 12.9.** (Force with
`FA_CUDA_ARCH=103` if detection is ever wrong.) *v11's native-FP4 compute will need `sm_103a` — out of
scope here.* The regime notebook's §3 smoke cell is where a gencode/toolchain mismatch surfaces — **if
it errors, STOP and fix before burning sweep time.**

## 5. Clock lock (root) — and its Blackwell caveat

```bash
nvidia-smi -pm 1
nvidia-smi -q -d SUPPORTED_CLOCKS | head        # pick a supported SM clock
nvidia-smi -lgc <max>,<max>                      # -lmc may return NOT_SUPPORTED on Blackwell
nvidia-smi -q -d CLOCK,PERFORMANCE | grep -A12 "Clocks Throttle"   # confirm the lock HELD
```
A lock is a **range, not a hard pin**: Blackwell's aggressive DVFS + (GB300) shared Grace power budget
override it under heavy FP4/exp load — exactly like the T4 sagging 1590→1350 under FP8 dequant. **Always
read back the throttle reasons**; an active `SW Power Cap` means absolute µs/tok is clock-confounded for
that run (trust the same-session ratios + %HBM, not cross-run wall-time). The notebook's §2 lock cell +
§11 reset cell wrap this; `bench/regime.lock_clocks()` warns-and-continues if not root.

## 6. The tiered measurement order (cost-first — the most decisive data lands first)

Run notebooks **in this order**, committing + pushing outputs after each so analysis happens off-box:

**Dev-rung (B200, cheap):**
0. Probe ncu (§0). Build all three backends (`v10_b300_regime.ipynb` §3 smoke). Install + smoke-run
   FlashInfer (`v10_b300_comparators.ipynb` install cell). Confirm the whole pipeline before paying for B300.

**B300 — Tier 1 (SHORT ~1–2 hr): the headline.**
1. `v10_b300_regime.ipynb` §3 smoke → §4 **arch-measure** (fills L2 + clock + smem — the unknowns) →
   §5 roofline → §6 **Tier-1 knee-hunt to 256K** (all three KV formats) → §9 plot. Commit. **This alone
   gives the per-CTA-vs-bandwidth verdict for FP16/FP8** (their L2 crossings are < 256K).

**B300 — Tier 2 (MEDIUM ~3–4 hr): complete the core.**
2. Regime §7 **Tier-2 to 524K + large-batch**. Then `v10_b300_exp_ablation.ipynb` (the sm_103 2×-exp
   delta, T2). Then `v10_b300_comparators.ipynb` (FlashInfer vs v10_nvfp4, same shapes, clock-locked).
   Commit each.

**B300 — Tier 3 (OPEN, no cap): the full curve + noise.**
3. Regime §8 **Tier-3 to 1M+** (crosses NVFP4's L2 ~870K — where NVFP4 either goes bandwidth-bound or
   proves per-CTA-bound to a million tokens), repeats for noise, §10 ncu cross-check. Commit the figure.

**Cost guards:** one backend at a time; `--max-ws-gb` rises per tier (32→48→96) as an OOM guard
(288 GB HBM holds even d128@1M = 512 MB easily); decode sweeps are µs-fast so wall-time is dominated by
allocation, not compute.

## 7. Teardown (every time, before destroy)

```bash
# in-notebook: §11 reset_clocks()  — or:
nvidia-smi -rgc ; nvidia-smi -rmc ; nvidia-smi -pm 0
git add -A && git commit -m "v10 B300 <tier> measured" && git push origin HEAD:main
# THEN destroy the vast.ai instance (disk is wiped on destroy).
```
Pull `main` locally afterward; analyze + write up Step 10 B300 results in `docs/results.md`/`decisions.md`.

## 8. What the run measures (fills the speculative `roofline/archs.py` B300 fields)

Research-filled (best estimates, in `archs.py`): `l2_mb=126` (LIKELY), `smem_per_sm_kb=228` (FACT),
`fp16_tc_flops=2.5e15` (LIKELY), `fp32_cuda_flops≈105e12` (unconfirmed), `boost_clock_mhz≈2600`
(unconfirmed), `int8_tc_ops` gutted ~95%. **The arch-measure cell (§4) is the source of truth** for L2
size (the #1 unknown — 126 vs the aggregator's 192), num_sm, clock, smem/SM. The science questions —
**does %HBM climb past L2 (knee → bandwidth-bound) or stay flat (per-CTA-bound to 1M)** and the **2×-exp
softmax delta** — are what the paper reports as prediction-vs-measured.
