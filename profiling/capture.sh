#!/usr/bin/env bash
# Nsight Compute capture for a kernel version. Run on the GPU box (Colab T4 / rented).
# Usage: profiling/capture.sh v1_naive
set -euo pipefail

BACKEND="${1:-v1_naive}"
mkdir -p profiling/raw

# --set full gives the Speed-of-Light + memory-workload + instruction-mix sections we read in
# GUIDE.md. --launch-count 3 profiles a few launches so the bench's warmup doesn't pollute it.
ncu --set full --launch-count 3 \
    --kernel-name-base demangled \
    -o "profiling/raw/${BACKEND}" \
    python -m bench.harness --backend "${BACKEND}" --precision fp32

echo "wrote profiling/raw/${BACKEND}.ncu-rep — open in Nsight Compute UI, read per GUIDE.md"
