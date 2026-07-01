# Stage A runbook — FP4 vs FP8 GEMM crossover at M=128 (B300)

Copy-paste runbook to execute `notebooks/stage_a_gemm_profiler.ipynb` on a rented B300 and push the
result. Modeled on `docs/v10-b300-runbook.md`. **The author machine can't compile CUDA or rent GPUs
— this is a manual SSH run.**

**What Stage A answers:** does native FP4 tensor-core compute beat FP8 at the MLA-decode QK-GEMM
shape (M=128, K∈{256,512,576}, N=1K…524K)? Pre-registered prediction: **NO** (decode is
HBM/work-starved). Kill condition: FP4 ≤ FP8 everywhere → publish the negative in paper §5.3.

**Cost/time:** ~1 hr wall on one B300 (~$5–6/hr on-demand). The CUTLASS build is the long pole.

---

## 0. Rent the box (vast.ai)

- GPU: **B300** (sm_103a). Search `cloud.vast.ai/?gpu_name=B300` or `vastai search offers 'gpu_name=B300'`.
- Image: **`nvidia/cuda:12.9.0-devel-ubuntu24.04`** (needs the `-devel` image for `nvcc`).
- Driver: **≥ R580** (Blackwell requires it).
- Disk: **≥ 40 GB** (CUTLASS build tree is large).
- SSH in, then verify hardware **before** building:

```bash
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader   # -> B300, 10.3, ~288 GB
nvcc --version | tail -1                                                       # -> release 12.9+
```

If either is wrong, destroy and rent another host. (The notebook's Cell 2 also asserts all of this
and stops loudly, so you won't waste a build — but check here first to save the rental minute.)

## 1. Environment

```bash
# vast.ai torch lives in a venv; activate it and put it first on PATH so `python -m ...` resolves.
. /venv/main/bin/activate 2>/dev/null || true
export PATH=/venv/main/bin:$PATH

pip install -q ninja pytest numpy pandas matplotlib jupyter nbconvert
git clone https://github.com/gkienpham-cmd/flashattention-cuda.git ~/fa
cd ~/fa

# Torch must see the B300 as (10, 3):
python -c "import torch; print('cap', torch.cuda.get_device_capability())"     # -> (10, 3)
```

If torch is a CPU/older build (`cap` wrong or CUDA error), install the cu129 nightly:
`pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu129`.

## 2. Run the notebook (run-of-record)

The notebook clones + builds CUTLASS 4.6 (profiler + example 72a/72b) itself, then runs the sweep.
Headless execution writes an executed copy you commit:

```bash
cd ~/fa
jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.timeout=3600 \
  --output stage_a_gemm_profiler_output.ipynb \
  notebooks/stage_a_gemm_profiler.ipynb
```

(Or open `notebooks/stage_a_gemm_profiler.ipynb` in Jupyter and **Run All**.) First run spends most
of its time in the CUTLASS build cell; it is cache-guarded so re-runs skip it.

**Clock lock is NOT required.** The verdict is a TFLOP/s *ratio* (FP4 vs FP8) measured back-to-back
on the same box, so DVFS cancels. (If you want absolute numbers to be clean too, `nvidia-smi -pm 1 &&
nvidia-smi -lgc 2032,2032` before running — needs a privileged box; not gating.)

## 3. Inspect the result

The notebook prints, near the end:
- the **FP4 vs FP8 TFLOP/s** pivot table (per K,N),
- a **roofline table** — expect every shape `limiter=HBM` with low `pct_peak`,
- a **VERDICT** banner: `KILL (FP4<=FP8 everywhere)` or `PROCEED (FP4 wins, max N.NNx)`.

It also writes `notebooks/stage_a_results.csv` (all rows incl. any failures, with `source`/
`out_dtype`/`kernel_name` provenance columns).

**Sanity checks before trusting it:**
- `stage_a_results.csv` has ~30 rows (15 shapes × 2 precisions), and neither precision is all-NaN.
- FP8 and FP4 absolute TFLOP/s are both **far below** their PF peaks (5 PF / 15 PF). If either shows
  near-peak `pct_peak`, the measurement is wrong — re-inspect the harness path used.
- Check the `source` column: ideally both precisions are `profiler` (matched f32 output). If FP4 fell
  back to `example72a`, FP8 should read `out_dtype=bf16` too (matched). If they are mismatched, the
  ratio is not trustworthy — see the notebook's Cell 8/10 notes.

## 4. Persist (vast.ai disk is ephemeral — push BEFORE destroy)

```bash
cd ~/fa
git add notebooks/stage_a_gemm_profiler_output.ipynb notebooks/stage_a_results.csv
git commit -m "Stage A GEMM results: FP4 vs FP8 at M=128 on B300"
git push origin HEAD:main
```

## 5. Sync Kien's local main (per CLAUDE.md git rule)

After the push lands on `origin/main`, on the **Mac** fast-forward the main checkout so the files
appear locally:

```bash
git -C /Users/kienpham/Documents/flashattention-cuda pull --ff-only origin main
```

## 6. Destroy the instance

Destroy on the vast.ai portal once the push is confirmed (`git log origin/main` shows the commit).

---

## Fallback notes (if the build or sweep misbehaves)

- **Profiler doesn't enumerate block-scaled NVFP4** → the notebook auto-falls back to example 72a
  (NVFP4→BF16) for FP4 and profiler `e4m3→bf16` for FP8 (matched output). Nothing to do; just confirm
  the `source` columns as in §3.
- **A single shape fails** (e.g. an alignment/timeout) → that row is recorded as NaN and the sweep
  continues; the verdict uses the rows that succeeded.
- **Both profiler and 72a fail for FP4** → build/adapt `72b_blackwell_nvfp4_nvfp4_gemm.cu` directly
  (accepts `--m/--n/--k`) and pair with a BF16-output FP8 baseline (never F32-output — see notebook
  Cell 10). This is the documented escape hatch, normally unused.
- **CUTLASS build OOM / disk full** → the notebook builds only 3 targets (profiler, 72a, 72b) in
  Release; if disk is still tight, rent with 60 GB.

## Recording the outcome in the paper

- **KILL (expected):** copy the pivot + roofline tables into `docs/results.md` and `docs/decisions.md`
  Stage A, and into paper **§5.3**: "native FP4 compute does not help MLA decode — the QK GEMM at
  M=128 is HBM/work-starved, not compute-bound; the M=128 packing advantage is a red herring for
  decode." Stage B/C not needed.
- **PROCEED (surprise):** record the crossover shape(s) and move to Stage B (accuracy on real KV).
