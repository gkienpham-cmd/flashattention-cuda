# Nsight Compute reading guide

We profile to *verify the roofline prediction*, not to collect numbers for their own sake. For
each step, capture the kernel(s), then read the four metrics below and ask: **does the measured
limiter match what `roofline/` predicted?** Disagreement is the next investigation.

## Capture

On Colab/T4 (see `profiling/capture.sh`):

```bash
ncu --set full --launch-count 3 \
    --kernel-name-base demangled \
    -o profiling/raw/v1_naive \
    python -m bench.harness --backend v1_naive --precision fp32
```

`--set full` is slow but gives the roofline + memory-workload sections we need. For a quick
read use `--section SpeedOfLight --section MemoryWorkloadAnalysis`.

## The four numbers we read every step

1. **Occupancy** (Achieved Occupancy %). How full the SMs are. Low occupancy + memory-bound
   means latency isn't being hidden; low occupancy + compute-bound may be fine. For v1 naive we
   expect decent occupancy (lots of independent threads) but it won't help — we're bandwidth-bound.

2. **Memory throughput** (DRAM Throughput % of peak, in the Speed-of-Light section). This is the
   bandwidth-wall meter. For Step 1 at d=64 we expect this **near 100%** while compute sits low —
   the direct confirmation that the S round-trip pins us to HBM.

3. **MMA / tensor-core utilization** (Compute (SM) Throughput, and the Tensor pipe in the
   instruction-mix). v1 uses **no** tensor cores, so expect ~0 on the tensor pipe and modest FP32
   pipe utilization. This number climbing is how we'll know Step 6 worked.

4. **MUFU utilization** (the XU / special-function pipe in the instruction-mix). Softmax's `exp`.
   For Step 1 it's tiny (~2% predicted). It becomes the headline in Phase 4 (the exponential wall),
   where we watch this pipe saturate and then relieve it with conditional rescaling / SW exp.

## Prediction-vs-measurement bookkeeping

Each step's `docs/results.md` row has both a predicted-limiter column and an ncu-limiter column.
Fill both. When they disagree, write *why* in `docs/decisions.md` — a wrong constant in
`roofline/archs.py`, an unmodeled cost (launch overhead, L2 hits), or a real surprise.
