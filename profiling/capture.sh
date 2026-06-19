#!/usr/bin/env bash
# Nsight Compute capture for a kernel version. Run on the GPU box (Colab T4 / rented).
# Usage: profiling/capture.sh v1_naive [extra bench args]
#   profiling/capture.sh v3_online --seq 8192 --dim 64   # target one shape for the deliverable
# Any args after the backend pass straight through to bench.harness (e.g. --seq/--dim/--causal).
set -euo pipefail

BACKEND="${1:-v1_naive}"
mkdir -p profiling/raw

# --set full gives the Speed-of-Light + memory-workload + instruction-mix sections we read in
# GUIDE.md. --launch-count 3 profiles a few launches so the bench's warmup doesn't pollute it.
#
# --kernel-name filters to the attention passes ONLY. Without it, --launch-count 3 caught the
# first 3 launches of the run — torch.randn's RNG kernel that builds the bench inputs — not our
# attention kernels. The regex matches v1/v2's qk_kernel/softmax_kernel/pv_kernel (+ the v2 tiled
# variants qk_tiled_kernel, ...) AND v3's online-softmax passes (pass1_stats/pass2_output), so it
# survives the version bump; it never matches the RNG/elementwise kernels. With the filter,
# --launch-count 3 = the first few passes of the first profiled iteration (warmup is steady-state).
ncu --set full --launch-count 3 \
    --kernel-name-base demangled \
    --kernel-name 'regex:(qk|softmax|pv|pass1_stats|pass2_output)' \
    -o "profiling/raw/${BACKEND}" \
    python -m bench.harness --backend "${BACKEND}" --precision fp32 "${@:2}"

echo "wrote profiling/raw/${BACKEND}.ncu-rep — open in Nsight Compute UI, or read headless on the"
echo "box via: ncu -i profiling/raw/${BACKEND}.ncu-rep --page raw --csv  (see GUIDE.md)"
