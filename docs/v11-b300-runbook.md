# v11 MLA — B300/sm_103 runbook (unprivileged container: nsys yes, ncu no, clock-lock warns)

The B300 measured core for v11. Mirrors `docs/v10-b300-runbook.md`. **Honesty constraints on an
unprivileged vast.ai container:** `ncu` counters are blocked (host kernel-module gate) → skip, log as
owed; `nvidia-smi --lock-gpu-clocks` needs root → `--lock-clocks` prints "CLOCKS NOT LOCKED" and
continues, so **cross-run `us/tok` is clock-confounded — read the counter-free `%HBM` proxy and trends,
not absolute latency**; `nsys` works but the stock binary is 2025.1.3 (records an EMPTY sm_103 trace) →
**install + use 2025.3.2+ explicitly** (Part 3).

What this run settles: **the T1 crossover — does MLA decode leave the per-CTA floor past the 132.6 MB
B300 L2 (bandwidth-bound, the counter-prediction), or stay ~0.1–0.5 %HBM (per-CTA-bound, as on T4)?**
The CUDA-core kernel cannot engage tcgen05, so the native-FP4 compute-flip is NOT tested here (deferred,
§9 Q1/Q2). L2-crossing for MLA (one latent, 576×0.5625 B/token): WS > 132.6 MB at **N_k ≳ 410k**, so
524288 / 1M / 2M are the past-L2 points (and the 288 GB HBM means no OOM, unlike the 16 GB T4).

---

## Part 1 — Notebook (sm_103 re-verify: correctness + capacity + accuracy + the harness A/B)

Open `notebooks/v11_mla_gate.ipynb` on the B300 Jupyter, **Run All** (cells 0–18). On sm_103:
- cell 8 builds for sm_103 automatically (no sm_103a features → plain sm_103 compiles);
- cell 10 (pytest) re-confirms correctness on the real arch;
- cells 12/14 re-confirm accuracy + capacity (arch-independent);
- cell 16 (harness A/B) now runs on real B300 — **read `%HBM` + the roofline column, not `us/tok`** (clocks unlocked);
- cell 18 (regime sweep to N_k=524288) now runs without the T4 OOM.

Save the executed notebook back to GitHub as `notebooks/v11_mla_gate_b300output.ipynb` (branch `main`).

---

## Part 2 — Terminal: the %HBM crossover sweep (the T1 deliverable), extended past L2

Paste into the Jupyter terminal:

```bash
. /venv/main/bin/activate
export PATH=/venv/main/bin:$PATH
cd /flashattention-cuda
git pull origin main          # pick up the v11 source + the paged.py fix

# MLA: real DeepSeek-V3 latent (576/512), h_q=128, sweep N_k across the 132.6 MB L2 crossing (~410k).
# --lock-clocks WARNS (unprivileged) and continues; the counter-free %HBM/eff_bw proxy is clock-robust.
python -m bench.regime --backend v11_mla --dim 576 --h-kv 1 --gqa-group 128 \
  --kv-lens 8192 32768 131072 524288 1048576 2097152 --lock-clocks | tee /tmp/v11_regime.txt

# Matched comparators on the SAME box (the per-CTA story is the %HBM-vs-N_k shape, not absolute us/tok):
python -m bench.regime --backend v10_nvfp4 --dim 128 --h-kv 1 --gqa-group 8 \
  --kv-lens 8192 32768 131072 524288 1048576 2097152 --lock-clocks | tee /tmp/v10_regime.txt
```

**Read:** the `%HBM` and `L2served` / `WS(MB)` columns vs the 132.6 MB L2.
- **%HBM CLIMBS** as N_k passes ~410k (524288→2M) → MLA went **bandwidth-bound on sm_103** = the
  counter-prediction lands, the byte cut converts. **Publishable.**
- **%HBM FLAT ~0.1–0.5%** past L2 (`L2served=no`, WS ≫ 132.6 MB) → MLA **stayed per-CTA-bound** even at
  2M, confound-free (matches T4) → the shape change did not leave the floor on CUDA cores; the
  native-FP4-TC arm (or speculative q_len>1) is the remaining lever. **Also publishable.**

---

## Part 3 — Terminal: nsys schedule (partial vs merge split) with the 2025.3.2+ binary

The stock `nsys` is 2025.1.3 → empty sm_103 trace. Install + use the newer one explicitly:

```bash
. /venv/main/bin/activate
export PATH=/venv/main/bin:$PATH
cd /flashattention-cuda

# 1) Find the newest nsys the repo offers (want >= 2025.3.x — it has sm_103 CUPTI).
apt-get update -q
apt-cache search nsight-systems | grep -E 'nsight-systems-20' | sort -V

# 2) Install the newest date-versioned one apt-cache showed (substitute if a higher one exists).
apt-get install -y nsight-systems-2025.3.2

# 3) Use the NEWEST installed binary explicitly (do NOT trust `nsys` on PATH — it's still 2025.1.3).
#    RUN STEPS 1-4 IN ONE SHELL SESSION: $NSYS is set here and used in step 4. (If you paste only step 4,
#    $NSYS is empty and bash errors ": command not found".)
NSYS_DIR=$(ls -d /opt/nvidia/nsight-systems/*/ | sort -V | tail -1)
NSYS="$NSYS_DIR/bin/nsys"; [ -x "$NSYS" ] || NSYS="$NSYS_DIR/target-linux-x64/nsys"
echo "using: $NSYS"; "$NSYS" --version          # MUST print 2025.3.2+ , NOT 2025.1.3

# 4) Profile ONE MLA shape (h_q=128, real 576 latent, 1M context) + dump the kernel-time table.
#    --profile loops the kernel 100x (no timing) so nsys attaches; --gqa-group 128 -> h_q=128.
#    (Drop N_k to 131072 if the 1M loop is slow — the partial/merge split is shape-independent.)
"$NSYS" profile -o /tmp/v11_nsys3 --force-overwrite true --trace=cuda --cuda-event-trace=false \
  python -m bench.regime --profile 1,1,1048576,576 --backend v11_mla --gqa-group 128
"$NSYS" stats --report cuda_gpu_kern_sum /tmp/v11_nsys3.nsys-rep 2>/dev/null | tee /tmp/v11_kern_sum.txt | head -45
```

**Read** `/tmp/v11_kern_sum.txt`: the `mla_partial_kernel` should dominate (~99%+), `mla_merge_kernel`
tiny — confirms the time is in the attention compute, not the LSE merge (the schedule is sound; not
merge-bound). Save `/tmp/v11_kern_sum.txt` + `/tmp/v11_regime.txt` + `/tmp/v10_regime.txt` into the repo
(e.g. `notebooks/`) and push, or paste them back.

---

## What to bring back

1. `/tmp/v11_regime.txt` (+ `/tmp/v10_regime.txt`) — the %HBM-vs-N_k crossover (the headline).
2. `/tmp/v11_kern_sum.txt` — the nsys partial/merge schedule split.
3. The executed `v11_mla_gate_b300output.ipynb`.

Owed (not achievable here, log honestly): **ncu** L2-hit-rate (privileged box), **clock-locked** us/tok,
the **native-FP4 tcgen05** arm, and **FlashMLA/FlashInfer** comparators (clock-locked, matched precision).
