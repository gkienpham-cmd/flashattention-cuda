"""NVFP4 scale-granularity ablation — the v10 accuracy deliverable (pure PyTorch, no kernel).

The v10 Gate-1 measured STANDARD NVFP4 (per-tensor + per-16 micro-scale) at ~2.4e-3 output RMSE =
~4x the FP8 floor (~6e-4). KV-quant research (KIVI, KVQuant, KVTuner) says that gap is recoverable
with an ASYMMETRIC granularity: K has outlier CHANNELS (-> per-channel scale), V has outlier TOKENS
(-> per-token scale). This module *fake-quantizes* (quantize -> dequantize in fp32, NO byte packing,
NO kernel) the K and V caches at a chosen granularity and measures the end-to-end attention RMSE, so
the K x V granularity matrix is one figure. Capacity is unchanged to first order (a per-channel scale
is d values/tensor, a per-token scale is N_k values/tensor — negligible vs the nibbles), so this is a
pure accuracy lever at fixed ~0.56 B/elem; the roofline is blind to it.

Granularities (how the dequant scale varies):
  tensor  - one scalar scale (coarsest; what FP8 per-tensor uses)
  block16 - NVFP4's native per-16-along-d E4M3 micro-scale (the gate's MEASURED recipe; reuses the
            real kernel quantizer so this row matches the gate's 2.4e-3 number exactly)
  channel - per head-dim channel (the K lever: outlier channels)
  token   - per token / key position (the V lever: outlier tokens)

All functions import torch lazily so `import fa_kernels` still works on a CPU-only box.
"""

from __future__ import annotations

_E2M1_LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_E2M1_MIDPOINTS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
_E2M1_MAX = 6.0

GRANULARITIES = ("tensor", "block16", "channel", "token")


def _round_e2m1(y):
    """Round each element of `y` (already divided by its scale, target range ~[-6, 6]) to the nearest
    signed E2M1 level. Mirrors the kernel's `dequant_nvfp4` value set."""
    import torch

    mids = torch.tensor(_E2M1_MIDPOINTS, device=y.device, dtype=torch.float32)
    lev = torch.tensor(_E2M1_LEVELS, device=y.device, dtype=torch.float32)
    idx = torch.searchsorted(mids, y.abs().float().contiguous())
    return torch.sign(y) * lev[idx]


def fake_quant_nvfp4(x, granularity: str = "block16", *, e4m3_scale: bool = True):
    """Quantize `x` to NVFP4 at `granularity` and return the dequantized `x_hat` (same shape/dtype).

    `block16` reuses the REAL kernel quantizer (`quantize_nvfp4`/`dequantize_nvfp4`) so its accuracy
    row matches the gate's measured number exactly. The other granularities use a single E2M1 + scale
    level whose scale varies along the named axis (the standard KIVI-style per-channel / per-token
    quantization). `e4m3_scale` rounds the scale through E4M3 (NVFP4-faithful) vs keeping it fp32.
    """
    import torch

    if granularity == "block16":
        from .paged import dequantize_nvfp4, quantize_nvfp4
        p, m, s = quantize_nvfp4(x)
        return dequantize_nvfp4(p, m, s).to(x.dtype)

    xf = x.float()
    if granularity == "tensor":
        amax = xf.abs().amax()
    elif granularity == "channel":                       # scale per head-dim channel (K lever)
        amax = xf.abs().amax(dim=tuple(range(xf.dim() - 1)), keepdim=True)   # [1,..,1,d]
    elif granularity == "token":                         # scale per token / key position (V lever)
        amax = xf.abs().amax(dim=-1, keepdim=True)                            # [..,N,1]
    else:
        raise ValueError(f"unknown granularity {granularity!r}; pick one of {GRANULARITIES}")

    scale = (amax / _E2M1_MAX).clamp_min(1e-12)
    if e4m3_scale:
        scale = scale.to(torch.float8_e4m3fn).float().clamp_min(1e-12)
    return (_round_e2m1(xf / scale) * scale).to(x.dtype)


def attn_rmse(q, k, v, *, k_gran: str, v_gran: str, causal: bool = False) -> float:
    """End-to-end GQA-attention RMSE of (K fake-quantized at `k_gran`, V at `v_gran`) vs the fp16-KV
    reference. The headline metric of the ablation — the quantization's effect on the OUTPUT, not the
    KV MSE (logit/output error is what the model actually sees)."""
    from .reference import sdpa_reference_gqa

    kq = fake_quant_nvfp4(k, k_gran)
    vq = fake_quant_nvfp4(v, v_gran)
    out_q = sdpa_reference_gqa(q, kq, vq, causal=causal)
    out_ref = sdpa_reference_gqa(q, k, v, causal=causal)
    return (out_q - out_ref).pow(2).mean().sqrt().item()


def fp8_attn_rmse(q, k, v, *, causal: bool = False) -> float:
    """The FP8 floor for comparison: K, V per-tensor E4M3, end-to-end attention RMSE vs fp16 KV."""
    from .paged import dequantize_fp8_e4m3, quantize_fp8_e4m3
    from .reference import sdpa_reference_gqa

    kb, sk = quantize_fp8_e4m3(k)
    vb, sv = quantize_fp8_e4m3(v)
    kq = dequantize_fp8_e4m3(kb, sk).to(k.dtype)
    vq = dequantize_fp8_e4m3(vb, sv).to(v.dtype)
    out_q = sdpa_reference_gqa(q, kq, vq, causal=causal)
    out_ref = sdpa_reference_gqa(q, k, v, causal=causal)
    return (out_q - out_ref).pow(2).mean().sqrt().item()


def fp4_score_collapse_rmse(q, k, v, *, causal: bool = False) -> dict:
    """The FP4-EVERYTHING ablation: quantize the post-softmax weights P (the scores) to FP4 too, not
    just the KV storage, and show the softmax collapse. Returns RMSE for (a) FP4 KV only and (b) FP4
    KV + FP4 P, so the figure can show the blow-up that justifies keeping the score >= FP16. Uses
    full-precision math except the named quantization; GQA-expands KV to H_q."""
    import torch
    import torch.nn.functional as F

    H_q, H_kv = q.shape[1], k.shape[1]
    G = H_q // H_kv
    ke = k.repeat_interleave(G, dim=1).float() if H_kv != H_q else k.float()
    ve = v.repeat_interleave(G, dim=1).float() if H_kv != H_q else v.float()
    qf = q.float()
    scale = 1.0 / (q.shape[-1] ** 0.5)

    scores = torch.matmul(qf, ke.transpose(-1, -2)) * scale
    if causal:
        N_q, N_k = scores.shape[-2], scores.shape[-1]
        mask = torch.triu(torch.ones(N_q, N_k, device=q.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
    p = F.softmax(scores, dim=-1)
    out_ref = torch.matmul(p, ve)

    # (a) FP4 KV only (token-V / channel-K, the asymmetric recipe), exact P.
    kq = fake_quant_nvfp4(k, "channel").repeat_interleave(G, dim=1).float() if H_kv != H_q \
        else fake_quant_nvfp4(k, "channel").float()
    vq = fake_quant_nvfp4(v, "token").repeat_interleave(G, dim=1).float() if H_kv != H_q \
        else fake_quant_nvfp4(v, "token").float()
    sc = torch.matmul(qf, kq.transpose(-1, -2)) * scale
    if causal:
        sc = sc.masked_fill(mask, float("-inf"))
    p_kv = F.softmax(sc, dim=-1)
    out_kv = torch.matmul(p_kv, vq)

    # (b) + FP4 P (quantize the softmax weights per token, the thing research says collapses).
    p_q = fake_quant_nvfp4(p_kv, "token")
    p_q = p_q / p_q.sum(dim=-1, keepdim=True).clamp_min(1e-12)   # renormalize, generous to FP4-P
    out_pq = torch.matmul(p_q, vq)

    def rmse(a, b):
        return (a - b).pow(2).mean().sqrt().item()

    return {"fp4_kv_only": rmse(out_kv, out_ref), "fp4_kv_and_p": rmse(out_pq, out_ref)}
