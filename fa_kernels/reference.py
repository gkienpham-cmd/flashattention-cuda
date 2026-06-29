"""Correctness anchors.

Two references, used at different phases:
  - `sdpa_reference`: torch's scaled_dot_product_attention. The everyday correctness oracle
    for the FP16 fundamentals (Phases 1-2). Same precision class as our kernels.
  - `fp64_reference`: an attention computed entirely in float64. The *ground truth* for the
    quantized phase (Phase 3), where we report RMSE of a low-precision kernel against it,
    the way FlashAttention-3 validates its FP8 path. Overkill for FP16 work, essential for INT8.

Both take [B, H, N, d] tensors and return [B, H, N, d], matching our layout convention.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def sdpa_reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                   *, causal: bool = False, scale: float | None = None) -> torch.Tensor:
    """Reference output via torch SDPA. This is what every Phase 1-2 test compares against."""
    # torch>=2.1 accepts an explicit `scale`; passing None lets it default to 1/sqrt(d),
    # which matches AttnConfig.effective_scale(), so the two agree by construction.
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal, scale=scale)


def sdpa_reference_gqa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                       *, causal: bool = False, scale: float | None = None) -> torch.Tensor:
    """GQA correctness oracle: broadcast-expand the KV heads to match q, then call SDPA.

    `q` has H_q query heads; `k, v` have H_kv = H_q / G heads (the G query heads of a group share one
    KV head). We expand with `repeat_interleave(G, dim=1)` — NOT `repeat`/`tile` — so KV head `h_kv`
    serves query heads `[h_kv*G, h_kv*G + G)`, exactly matching the kernel's `h_q = h_kv*G + g_local`
    mapping. (`repeat` would instead tile the heads `[0,1,..,H_kv-1, 0,1,..]`, a silent-wrong oracle.)
    When H_q == H_kv (G=1) this is identical to `sdpa_reference`.
    """
    H_q, H_kv = q.shape[1], k.shape[1]
    if H_kv != H_q:
        assert H_q % H_kv == 0, f"H_q={H_q} must be a multiple of H_kv={H_kv}"
        G = H_q // H_kv
        k = k.repeat_interleave(G, dim=1)
        v = v.repeat_interleave(G, dim=1)
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal, scale=scale)


def sdpa_reference_gqa_fp8(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                           scale_k: float, scale_v: float,
                           *, causal: bool = False, scale: float | None = None) -> torch.Tensor:
    """Apples-to-apples FP8 oracle for v9: dequantize the SAME E4M3 bytes the kernel reads, then run
    GQA SDPA. The kernel must match the math on identical quantized inputs, so this isolates the
    kernel's correctness from the quantization error (the latter is reported separately as RMSE vs the
    original fp16 `sdpa_reference_gqa`). `k, v` are the dense FP16 KV; we round-trip them through E4M3
    exactly as `build_paged_kv_fp8` did, so the oracle sees the kernel's actual operands.
    """
    from .paged import quantize_fp8_e4m3, dequantize_fp8_e4m3

    kb, sk = quantize_fp8_e4m3(k)
    vb, sv = quantize_fp8_e4m3(v)
    # The caller passes the scales build_paged_kv_fp8 produced; recompute-identical here, but honor the
    # passed-in scales for the dequant so a mismatch surfaces rather than silently re-deriving.
    k_hat = dequantize_fp8_e4m3(kb, scale_k).to(k.dtype)
    v_hat = dequantize_fp8_e4m3(vb, scale_v).to(v.dtype)
    return sdpa_reference_gqa(q, k_hat, v_hat, causal=causal, scale=scale)


def sdpa_reference_gqa_nvfp4(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                             *, causal: bool = False, scale: float | None = None) -> torch.Tensor:
    """Apples-to-apples NVFP4 oracle for v10: dequantize the SAME NVFP4 bytes the kernel reads (packed
    E2M1 nibbles + per-16 E4M3 micro-scales + per-tensor scale), then run GQA SDPA. Isolates the
    kernel's math from the quantization error (the latter is reported separately as RMSE vs the original
    fp16 `sdpa_reference_gqa`). `k, v` are the dense FP16 KV; we round-trip them through NVFP4 exactly as
    `build_paged_kv_nvfp4` did, so the oracle sees the kernel's actual operands. (No scale args — the
    per-tensor scale is recomputed inside `quantize_nvfp4`, identical to the build path.)
    """
    from .paged import quantize_nvfp4, dequantize_nvfp4

    kp, km, sk = quantize_nvfp4(k)
    vp, vm, sv = quantize_nvfp4(v)
    k_hat = dequantize_nvfp4(kp, km, sk).to(k.dtype)
    v_hat = dequantize_nvfp4(vp, vm, sv).to(v.dtype)
    return sdpa_reference_gqa(q, k_hat, v_hat, causal=causal, scale=scale)


def sdpa_reference_mla(q_absorbed: torch.Tensor, latent: torch.Tensor, kv_lora_rank: int,
                       *, causal: bool = False, scale: float | None = None) -> torch.Tensor:
    """Apples-to-apples MLA oracle for v11 (the primary correctness gate): dequantize the SAME NVFP4
    latent bytes the kernel reads, then compute MQA over the shared latent IN THE LATENT BASIS — exactly
    what the latent-absorbed kernel computes. `q_absorbed` is `[B,H_q,N_q,DQK]` (the query already in the
    latent basis); `latent` is `[B,1,N_k,DQK]` (one shared latent head). The score is a DQK-wide dot vs
    the latent (it serves as K); the output is the p-weighted sum of the first `kv_lora_rank` latent dims
    (the latent serves as V; the trailing RoPE dims carry no value). Returns `O_latent`
    `[B,H_q,N_q,kv_lora_rank]`. matmul broadcasts the single latent head across all H_q query heads.

    Pass the SAME `scale` the kernel used (default 1/sqrt(DQK)). At decode the kernel masks via
    `q_offset`, so the decode test calls this with `causal=False` (the query attends the whole cache),
    mirroring the v10 pattern.
    """
    from .paged import quantize_nvfp4, dequantize_nvfp4

    lp, lm, sl = quantize_nvfp4(latent)
    lat_hat = dequantize_nvfp4(lp, lm, sl).to(q_absorbed.dtype)        # [B,1,N_k,DQK] — kernel's operand
    DQK = q_absorbed.shape[-1]
    s = scale if scale is not None else 1.0 / (DQK ** 0.5)
    scores = torch.matmul(q_absorbed, lat_hat.transpose(-1, -2)) * s   # [B,H_q,N_q,N_k] (broadcast head)
    if causal:
        n_q, n_k = scores.shape[-2], scores.shape[-1]
        mask = torch.triu(torch.ones(n_q, n_k, dtype=torch.bool, device=scores.device), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
    scores = scores - scores.amax(dim=-1, keepdim=True)               # stabilize before exp
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, lat_hat[..., :kv_lora_rank])           # [B,H_q,N_q,kv_lora_rank]


def sdpa_reference_mla_materialized(q_nope: torch.Tensor, q_rope: torch.Tensor, latent: torch.Tensor,
                                    kv_lora_rank: int, W_UK: torch.Tensor, W_UV: torch.Tensor,
                                    *, causal: bool = False, scale: float | None = None) -> torch.Tensor:
    """MLA absorption-identity oracle (kickoff §9 Q4): EXPAND the (dequantized) latent through synthetic
    low-rank up-projections `W^UK`/`W^UV` to full per-head K/V, then run standard SDPA. This is the
    explicit-materialization truth the latent-absorbed kernel must equal — the identity that makes MLA
    correct (`q_absorbed·latent == q_full·K_full`, `O_latent·W^UV == p·V_full`). The test re-projects the
    kernel's `O_latent` through `W^UV` and compares to this. Inputs:
        q_nope : [B,H_q,N_q,Dn]            per-head query (content)
        q_rope : [B,H_q,N_q,R]             per-head query (decoupled RoPE)
        latent : [B,1,N_k,DQK]             the shared latent (content L=kv_lora_rank + RoPE R); the SAME
                                           tensor passed to the kernel — dequantized NVFP4-faithfully here
        W_UK   : [H_q,Dn,L]                content up-proj: K_nope_h = c_hat @ W_UK_h^T
        W_UV   : [H_q,Dv,L]                value   up-proj: V_h      = c_hat @ W_UV_h^T
    Returns O_full `[B,H_q,N_q,Dv]`. Pass the SAME `scale` the kernel used so the equal dot products map
    to equal scores (the absorbed dot spans DQK, the materialized dot spans Dn+R; they are numerically
    equal by the identity, so one shared scalar scale matches both).
    """
    from .paged import quantize_nvfp4, dequantize_nvfp4

    L, R = kv_lora_rank, latent.shape[-1] - kv_lora_rank
    lp, lm, sl = quantize_nvfp4(latent)
    lat_hat = dequantize_nvfp4(lp, lm, sl).to(q_nope.dtype)           # [B,1,N_k,DQK] — kernel's operand
    c_hat, rope_hat = lat_hat[..., :L], lat_hat[..., L:]             # [B,1,N_k,L], [B,1,N_k,R]
    # Materialize full per-head K and V from the shared latent (the up-projection MLA absorbs offline).
    K_nope = torch.einsum("bgnl,hdl->bhnd", c_hat, W_UK)             # [B,H_q,N_k,Dn]
    V_full = torch.einsum("bgnl,hdl->bhnd", c_hat, W_UV)             # [B,H_q,N_k,Dv]
    B, H_q, N_k, _ = K_nope.shape
    K_full = torch.cat([K_nope, rope_hat.expand(B, H_q, N_k, R)], dim=-1)   # [B,H_q,N_k,Dn+R]
    q_full = torch.cat([q_nope, q_rope], dim=-1)                            # [B,H_q,N_q,Dn+R]
    s = scale if scale is not None else 1.0 / (q_full.shape[-1] ** 0.5)
    return F.scaled_dot_product_attention(q_full, K_full, V_full, is_causal=causal, scale=s)


def fp64_reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                   *, causal: bool = False, scale: float | None = None) -> torch.Tensor:
    """Ground-truth attention in float64. Used to measure quantization RMSE in Phase 3.

    Computed the long way on purpose: upcast everything, do an explicit, numerically careful
    softmax (subtract the row max before exp), and keep float64 the whole way through.
    """
    qd, kd, vd = q.double(), k.double(), v.double()
    d = qd.shape[-1]
    s = scale if scale is not None else 1.0 / (d ** 0.5)
    scores = torch.matmul(qd, kd.transpose(-1, -2)) * s          # [B,H,N,N] in fp64
    if causal:
        n_q, n_k = scores.shape[-2], scores.shape[-1]
        # mask[i,j] = True where j > i (a query may not attend to future keys)
        mask = torch.triu(torch.ones(n_q, n_k, dtype=torch.bool, device=scores.device), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
    scores = scores - scores.amax(dim=-1, keepdim=True)          # stabilize before exp
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, vd)                               # [B,H,N,d] in fp64
