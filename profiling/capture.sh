#!/usr/bin/env bash
# Nsight Compute capture for a kernel version. Run on the GPU box (Colab T4 / rented).
# Usage: profiling/capture.sh v1_naive
set -euo pipefail

BACKEND="${1:-v1_naive}"
mkdir -p profiling/raw

# --set full gives the Speed-of-Light + memory-workload + instruction-mix sections we read in
# GUIDE.md. --launch-count 3 profiles a few launches so the bench's warmup doesn't pollute it.
#
# --kernel-name filters to the attention passes ONLY. Without it, --launch-count 3 caught the
# first 3 launches of the run — torch.randn's RNG kernel that builds the bench inputs — not our
# qk/softmax/pv. The regex matches qk_kernel/softmax_kernel/pv_kernel and the v2 tiled variants
# (qk_tiled_kernel, ...), so it survives the version bump; it never matches the RNG/elementwise
# kernels. With the filter, --launch-count 3 = the three passes of the first profiled iteration.
ncu --set full --launch-count 3 \
    --kernel-name-base demangled \
    --kernel-name 'regex:(qk|softmax|pv)' \
    -o "profiling/raw/${BACKEND}" \
    python -m bench.harness --backend "${BACKEND}" --precision fp32

echo "wrote profiling/raw/${BACKEND}.ncu-rep (qk/softmax/pv passes) — open in Nsight Compute UI, read per GUIDE.md"
