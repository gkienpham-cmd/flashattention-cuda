"""The roofline model for attention.

Given a problem shape and precision, estimate the time each of the three resources would take
*if it were the only bottleneck*, then the real lower bound is the max of the three and the
limiter is whichever resource owns that max. This is the first-principles prediction we record
BEFORE writing/optimizing a kernel, and check against ncu AFTER.

The three resources (see docs and roofline/archs.py):
  1. MMA compute   — the two matmuls QK^T and PV.
  2. HBM traffic   — bytes moved to/from global memory (this includes the naive S round-trip).
  3. MUFU exp      — the softmax exponentials on the special-function unit.

We model whole-problem attention (summed over B,H). `materialize_s=True` models the non-fused
versions (v1 naive, v2 tiled): the N_q x N_k score matrix round-trips HBM AND the matmul
operands are re-read from HBM `M*N*K / tile` times each, so `tile_m`/`tile_n` capture how much
shared-memory tiling cuts that redundant operand traffic (naive = 1x1 = each row re-read per
output element). Fused kernels set `materialize_s=False`: S stays on-chip and operands are read
~once. NOTE: the tile_m/tile_n operand-reuse term is why a naive kernel's true arithmetic
intensity is ~1/nbytes (deeply memory-bound), not the read-once ideal an earlier cut assumed.
"""

from __future__ import annotations

from dataclasses import dataclass

from .archs import Arch

# nvfp4 = 0.5625 B/elem = 4.5 b/elem: a 4-bit E2M1 nibble PLUS one E4M3 micro-scale per 16 elems
# (8 b / 16 = 0.5 b/elem). Counting the micro-scale matters — a naive "4 b -> 0.5 B" undercounts the
# byte traffic and overstates the AI (v10 trap). int4 stays the scale-free 0.5 for the old callers.
_BYTES = {"fp32": 4, "fp16": 2, "bf16": 2, "int8": 1, "fp8": 1, "nvfp4": 0.5625, "int4": 0.5}


@dataclass
class RooflineEstimate:
    limiter: str            # "mma" | "hbm" | "mufu"
    seconds: float          # predicted lower-bound runtime = max of the three
    t_mma: float
    t_hbm: float
    t_mufu: float
    arithmetic_intensity: float   # total FLOPs / total HBM bytes
    ridge: float                  # arch ridge point for this precision (FLOP/byte)

    def utilization(self) -> dict[str, float]:
        """Fraction of the bound each resource would hit; the limiter is 1.0."""
        return {
            "mma": self.t_mma / self.seconds,
            "hbm": self.t_hbm / self.seconds,
            "mufu": self.t_mufu / self.seconds,
        }


def estimate(arch: Arch, *, B: int, H: int, N_q: int, N_k: int, d: int,
             precision: str = "fp16", materialize_s: bool = False,
             use_tensor_core: bool = True, tile_m: int = 1, tile_n: int = 1,
             G: int = 1, mla: bool = False, h_q: int = 128,
             kv_lora_rank: int = 512, rope_dim: int = 64) -> RooflineEstimate:
    # --- MLA (v11) latent-KV decode: the SHAPE change ---
    # MLA (Multi-head Latent Attention) shares ONE low-rank latent KV across all h_q query heads, so
    # decode packs M = h_q (>1, unlike GQA's M=1 single-warp) and reads the latent ONCE for every head.
    # Two things change vs the GQA decode branch below: (1) the latent serves as BOTH K and V, so the
    # usual 2x K/V read collapses to 1x; (2) all h_q heads share that one read (bh/G -> B) while FLOPs
    # stay proportional to h_q. DeepSeek-V2/V3 shape: L=kv_lora_rank=512 content dims + R=rope_dim=64
    # decoupled-RoPE dims = ~576 dims/token stored. The QK dot is over (L+R); the PV weighted-sum is
    # over L (RoPE carries no value). The absorbed up-projections W^UK/W^UV fold into the OFFLINE Q/O
    # projections, so the attention kernel reconstructs the score as q_absorbed . c_latent (an MQA over
    # a 512-wide latent) and never materializes full K/V. Net decode AI:
    #     AI = 2*h_q*(2L+R) / ((L+R)*b)  =  ~3.78*h_q/b  (fp16 h_q=128 -> ~242; "2*h_q"~=256 is the
    # round shorthand). At h_q=128 this is ~30x the GQA-8 fp16 AI of 8 and lands AT/PAST the FP16-TC
    # ridge -> the first decode shape in the v1->v11 arc the PURE roofline puts compute-bound (fp8/nvfp4
    # latent push it firmly past the ridge). M=h_q>=16 genuinely engages tensor cores, so the mma bound
    # uses the TC peak (unlike the GQA decode branch, where M=1 keeps the cores dark). The model is
    # BLIND to on-chip capacity: staging q_absorbed (h_q*(L+R) fp16 ~= 147 KB at h_q=128) may cap
    # occupancy at ~1 block/SM and RENAME the limiter to smem/TMEM-capacity rather than flip it to
    # compute -- that per-CTA-corrected layer lives in results.md Step 11, not in this math.
    if mla:
        nbytes = _BYTES[precision]
        L, R = kv_lora_rank, rope_dim
        if precision == "fp32":
            peak = arch.fp32_cuda_flops
        elif precision == "int8":
            peak = arch.int8_tc_ops
        else:  # fp16/bf16 + fp8/nvfp4 latent storage, dequant-to-fp16 compute on the TC path
            peak = arch.fp16_tc_flops if use_tensor_core else arch.fp32_cuda_flops
        mma_flops = 2.0 * B * h_q * N_q * N_k * (L + R)   # QK^T over (L+R)
        mma_flops += 2.0 * B * h_q * N_q * N_k * L         # PV over L (RoPE carries no value)
        t_mma = mma_flops / peak

        kv_bytes = B * N_k * (L + R) * nbytes              # latent read ONCE for all h_q heads
        q_bytes = B * h_q * N_q * (L + R) * nbytes
        o_bytes = B * h_q * N_q * L * nbytes
        hbm_bytes = kv_bytes + q_bytes + o_bytes
        t_hbm = hbm_bytes / (arch.hbm_bw_gbps * 1e9)

        exp_count = float(B) * h_q * N_q * N_k             # one exp per (head, key) score entry
        mufu_rate = (arch.fp32_cuda_flops / 2.0) * arch.mufu_ratio
        t_mufu = exp_count / mufu_rate

        times = {"mma": t_mma, "hbm": t_hbm, "mufu": t_mufu}
        limiter = max(times, key=times.get)
        ai = mma_flops / hbm_bytes
        ridge = peak / (arch.hbm_bw_gbps * 1e9)
        return RooflineEstimate(limiter=limiter, seconds=times[limiter],
                                t_mma=t_mma, t_hbm=t_hbm, t_mufu=t_mufu,
                                arithmetic_intensity=ai, ridge=ridge)

    bh = B * H
    nbytes = _BYTES[precision]
    # GQA group factor: G query heads share one KV head (H_kv = H/G). The G query heads still each
    # do their matmul (mma_flops unchanged), but the KV they read is shared, so KV bytes drop by G
    # -> decode AI = 2/b -> 2G/b. G=1 is plain MHA (every prior version), so all old calls are
    # byte-identical. Only the fused-decode KV term below divides by G; Q-read and O-write are
    # per-query-head and stay at bh.

    # --- compute: two matmuls, each 2*M*N*K FLOPs ---
    mma_flops = 2.0 * bh * N_q * N_k * d   # QK^T
    mma_flops += 2.0 * bh * N_q * N_k * d  # PV
    # MMA peak is precision-specific: fp32 on CUDA cores (8.1 TFLOPS), int8 on its own tensor-core
    # peak (130 TOPS, 2x fp16 on Turing), fp16/bf16 on the fp16 tensor-core peak (65 TFLOPS) — or the
    # CUDA-core peak if tensor cores are disabled. (Previously every non-fp32 precision took the fp16
    # peak, which made the int8 MMA bound 2x too slow and left arch.int8_tc_ops dead.)
    if precision == "fp32":
        peak = arch.fp32_cuda_flops
    elif precision == "int8":
        peak = arch.int8_tc_ops
    else:  # fp16 / bf16 / fp8 / nvfp4
        # fp8 and nvfp4 are STORAGE formats here: v9/v10 decode dequantizes the KV to FP16 in smem
        # and runs the matmuls on the fp16 path (CUDA-core GEMV at N_q=1, M=G<64 keeps tensor cores
        # dark). So they intentionally take the fp16 peak, NOT arch.fp4_tc_flops — the FP4 tensor-core
        # ridge (B300: 15e15/8e12 = 1875 FLOP/B) is a separate observation that native-FP4 *compute*
        # (v11, M>=64 multi-token) would invoke, not v10 decode. Decode stays far below either ridge.
        peak = arch.fp16_tc_flops if use_tensor_core else arch.fp32_cuda_flops
    t_mma = mma_flops / peak

    # --- HBM traffic ---
    # The output O is written exactly once in every version.
    o_write = bh * N_q * d * nbytes

    if materialize_s:
        # Non-fused versions (v1 naive, v2 tiled): the score matrix S round-trips HBM AND the
        # matmul operands get re-read from HBM, with the re-read count set by tiling. For a tiled
        # GEMM C[M,N] = A[M,K]·B[K,N] with output tile (tile_m x tile_n), each operand is read
        # M*N*K / tile times (A reused across tile_n output cols, B across tile_m output rows).
        # Naive = tile 1x1: each Q/K row re-read once per output element -> the real O(N^2 * d)
        # cost the first model wrongly ignored (it assumed read-once). Bigger tile = more on-chip
        # reuse = fewer HBM reads. Two matmuls, four operands: QK^T (contraction K=d) and PV
        # (contraction K=N_k); the per-operand counts are symmetric in tile_m/tile_n for both.
        reuse = (1.0 / tile_m + 1.0 / tile_n)
        qk_operand_reads = bh * N_q * N_k * d * reuse    # Q and K
        pv_operand_reads = bh * N_q * d * N_k * reuse    # P and V
        operand_bytes = (qk_operand_reads + pv_operand_reads) * nbytes
        # The S round-trip: pass1 writes S, pass2 reads+writes S, pass3 reads S (~4 sweeps over
        # the N_q x N_k matrix). Tiling does NOT remove this — only online softmax (v3) does.
        s_bytes = 4.0 * bh * N_q * N_k * nbytes
        hbm_bytes = operand_bytes + s_bytes + o_write
    else:
        # Fused versions (v3+ online softmax): S never touches HBM and streaming reads each operand
        # ~once. K and V are the two N_k-sized operands; Q is only N_q rows and O is written once. At
        # DECODE (N_q=1, G=1) this is the research §4 decode roofline: work = 4*N_k*d FLOPs, traffic =
        # 2*N_k*d*b bytes -> AI = 4Nd/(2Nd*b) = 2/b FLOP/byte, INDEPENDENT of N_k -> pure HBM-bound,
        # far below the ridge (the tensor cores idle).
        #
        # NOTE on what t_hbm is and is NOT: it is a FLOOR the kernel is far from, not a bound split-KV
        # attains. v7's --batch sweep measured only ~10% of HBM on T4 decode, FLAT from BH=8 to 512 (no
        # occupancy->bandwidth crossover). The kernel LOOKS per-CTA-bound, not bandwidth-bound: at N_q=1
        # only 1 of 8 warps computes (GEMV shape) and sK+sV=32 KB caps residency at 2 blocks/SM. Filling
        # the grid with splits does NOT move %HBM. So the candidate lever is per-CTA EFFICIENCY, not bytes.
        #
        # *** L2-RESIDENCY CONFOUND (v8 deep-research close-out, 2026-06-28) ***: that "~10% HBM" is
        # NOT yet an earned bandwidth verdict. At the bench sizes the KV working set FITS in the T4's
        # 4 MB L2 (e.g. reclaim G=8/B=1 -> H_kv=1 -> KV = 2*1*8192*128*2 B ~= 4.2 MB ~= L2). An
        # L2-resident working set streams from L2 (~1.3 TB/s, ~4x HBM), so the DRAM counter reads low
        # even if the kernel IS memory-bound -- just bound by the wrong memory. The bench also never
        # locked clocks (free Colab) nor pushed N_k past ~16K. COUNTER-FREE L2 TEST: effective_bw =
        # kv_bytes / measured_time; if effective_bw > arch.hbm_bw_gbps the data came from L2, so %HBM
        # is meaningless as a boundedness metric. v9 Task 1 (root T4: lock clocks, flush L2, sweep N_k
        # 1K..128K, measure L2 hit-rate) earns or overturns the per-CTA verdict. Until then the
        # bytes-vs-per-CTA decode limiter is OPEN. See docs/v9-kickoff.md, results.md threats-section.
        #
        # v8 GQA M-packing IS that lever (promoted from the old "v10/v11, not modeled" note): G query
        # heads share one KV head, so KV is read once per group (bh/G) instead of per head (bh). KV bytes
        # drop by G while FLOPs hold -> AI = 2/b -> 2G/b, and G warps light up instead of 1. The win is
        # the G-fold per-CTA work-per-KV-byte, not occupancy. (Prefill N_q=N_k recovers the old
        # ~3*bh*N_k*d read-once estimate; there KV is read once regardless of G within a CTA.)
        kv_bytes = 2.0 * (bh / G) * N_k * d * nbytes
        hbm_bytes = kv_bytes + bh * N_q * d * nbytes + o_write
    t_hbm = hbm_bytes / (arch.hbm_bw_gbps * 1e9)

    # --- MUFU exp: one exp per score entry ---
    exp_count = float(bh) * N_q * N_k
    # MUFU op/s ~= (FP32 FMA/s) * ratio. fp32_cuda_flops counts 2 FLOPs per FMA, so /2.
    mufu_rate = (arch.fp32_cuda_flops / 2.0) * arch.mufu_ratio
    t_mufu = exp_count / mufu_rate

    times = {"mma": t_mma, "hbm": t_hbm, "mufu": t_mufu}
    limiter = max(times, key=times.get)
    ai = mma_flops / hbm_bytes
    # Ridge = the active compute peak / HBM bandwidth, so it tracks whichever `peak` was selected
    # above (T4: fp32 -> 25.3, fp16 -> 203, int8 -> 406 FLOP/byte). Tying it to `peak` keeps the
    # ridge consistent with the MMA bound for every precision (incl. int8 and --no-tensor-core).
    ridge = peak / (arch.hbm_bw_gbps * 1e9)

    return RooflineEstimate(limiter=limiter, seconds=times[limiter],
                            t_mma=t_mma, t_hbm=t_hbm, t_mufu=t_mufu,
                            arithmetic_intensity=ai, ridge=ridge)
