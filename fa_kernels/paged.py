"""Paged-KV attention — the import surface a from-scratch mini-vLLM consumes (v7+).

`attention(q, k, v)` (in __init__.py) takes a CONTIGUOUS dense KV; a real inference engine keeps the
KV cache as a pool of fixed-size pages plus a per-sequence block table that maps logical token
positions to physical pages (vLLM's layout). That needs a different call signature, so paged decode
gets its own stable entry point here rather than overloading `attention`.

    from fa_kernels import paged_attention
    out = paged_attention(q, k_pool, v_pool, block_table, page_size, n_k, causal=..., q_offset=...)

Shapes:
    q           : [B, H, N_q, d]                      CUDA tensor (decode: N_q = 1)
    k_pool      : [num_blocks, page_size, H, d]       physical KV pool (pages, not contiguous)
    v_pool      : [num_blocks, page_size, H, d]
    block_table : [B, n_logical]  int32              block_table[b][lb] -> physical block index
    page_size   : tokens per page
    n_k         : true logical KV length attended this step (<= n_logical * page_size); passed
                  explicitly so a non-multiple length never scans the last page's padding tail.

The kernel reads logical key j of sequence b as
    pb = block_table[b][j // page_size]; off = j % page_size; pool[pb*page_size + off, h, :]
so the physical pages may be in any (shuffled) order. Output is dense [B, H, N_q, d].
"""

from __future__ import annotations

DEFAULT_PAGED_BACKEND = "v7_paged"
DEFAULT_GQA_BACKEND = "v8_gqa"
DEFAULT_FP8_BACKEND = "v9_fp8"
DEFAULT_NVFP4_BACKEND = "v10_nvfp4"
DEFAULT_MLA_BACKEND = "v11_mla"

# Max magnitude representable by FP8 E4M3 (e4m3fn: no inf, max normal 448). Per-tensor scale maps the
# tensor's amax onto this so the full dynamic range is used. (int8-symmetric fallback would use 127.)
_E4M3_MAX = 448.0

# NVFP4 (v10): a 4-bit E2M1 element (1 sign + 2 exp + 1 mantissa) with TWO-LEVEL scaling — one E4M3
# micro-scale per 16-element block along the head-dim PLUS one FP32 per-tensor scale. Storage =
# 4 b/elem + 8 b / 16 = 4.5 b/elem = 0.5625 B/elem. E2M1 represents 8 magnitudes (max 6.0); the
# midpoints are the round-to-nearest boundaries on |x| (e.g. 0.3 -> level 0.5, 0.2 -> level 0.0).
_NVFP4_BLOCK = 16
_E2M1_MAX = 6.0
_E2M1_LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_E2M1_MIDPOINTS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)


def build_paged_kv(k, v, page_size: int, *, shuffle: bool = True, seed: int | None = None):
    """Turn a dense KV (`[B,H,N_k,d]`) into a paged pool + block table for `paged_attention`.

    Returns `(k_pool, v_pool, block_table, n_k)` where the pools are `[B*n_logical, page_size, H, d]`
    (one physical block per logical block, in SHUFFLED physical order when `shuffle=True`, so a
    correct kernel must follow the block table rather than assume contiguity) and `block_table` is
    int32 `[B, n_logical]`. The padding tail of a partial last page is left zero — the kernel never
    reads it (it loops logical positions `< n_k`). This is the test/bench oracle for the gather: run
    `paged_attention` on the output and compare to dense SDPA on the original `k, v`.
    """
    import torch
    import torch.nn.functional as F

    B, H, N_k, d = k.shape
    n_logical = (N_k + page_size - 1) // page_size
    padded = n_logical * page_size
    pad = padded - N_k

    # Physical-block permutation: logical flat index i=(b*n_logical+lb) -> physical block perm[i].
    gen = None
    if seed is not None:
        gen = torch.Generator(device=k.device).manual_seed(seed)
    if shuffle:
        perm = torch.randperm(B * n_logical, device=k.device, generator=gen)
    else:
        perm = torch.arange(B * n_logical, device=k.device)

    def to_pool(x):
        # [B,H,N_k,d] -> pad seq -> [B,H,n_logical,page_size,d] -> [B,n_logical,page_size,H,d]
        xp = F.pad(x, (0, 0, 0, pad))                       # pad the N_k (2nd-to-last) dim with zeros
        logical = (xp.reshape(B, H, n_logical, page_size, d)
                     .permute(0, 2, 3, 1, 4)                # [B, n_logical, page_size, H, d]
                     .reshape(B * n_logical, page_size, H, d)
                     .contiguous())
        pool = torch.empty_like(logical)
        pool[perm] = logical                                # physical block perm[i] holds logical i
        return pool

    block_table = perm.reshape(B, n_logical).to(torch.int32).contiguous()
    return to_pool(k), to_pool(v), block_table, N_k


def quantize_fp8_e4m3(x):
    """Per-tensor FP8 E4M3 quantize: pick scale = amax/448, store the raw E4M3 bits as uint8.

    Returns `(bytes_u8, scale)` where `bytes_u8` has x's shape/dtype=uint8 and the kernel reconstructs
    `x_hat ~= dequant_e4m3(byte) * scale`. To dequant on the host the same way (the apples-to-apples
    oracle), reinterpret the bytes back: `bytes_u8.view(torch.float8_e4m3fn).float() * scale`.

    (int8-symmetric fallback, matching the kernel's int8 path: scale = amax/127,
    `x.div(scale).round().clamp(-127,127).to(torch.int8).view(torch.uint8)`.)
    """
    import torch

    amax = x.abs().amax().clamp_min(1e-12)
    scale = (amax / _E4M3_MAX).item()
    bytes_u8 = (x / scale).to(torch.float8_e4m3fn).view(torch.uint8).contiguous()
    return bytes_u8, scale


def dequantize_fp8_e4m3(bytes_u8, scale):
    """Inverse of `quantize_fp8_e4m3` — reinterpret uint8 as E4M3 bits, upcast, x scale. This is the
    EXACT value the kernel staged into smem, so an oracle built on it isolates the kernel's math from
    the quantization error (the latter is measured separately vs the original fp16 KV)."""
    import torch

    return bytes_u8.view(torch.float8_e4m3fn).float() * scale


def build_paged_kv_fp8(k, v, page_size: int, *, shuffle: bool = True, seed: int | None = None):
    """FP8 E4M3 analogue of `build_paged_kv`: per-tensor quantize the dense KV, then page it.

    Returns `(k_pool, v_pool, block_table, n_k, scale_k, scale_v)` where the pools are uint8 (FP8 E4M3
    bytes) in `[B*n_logical, page_size, H, d]` SHUFFLED physical order, and `scale_k/scale_v` are the
    per-tensor dequant scales the kernel multiplies by. Quantize BEFORE paging so each physical byte is
    the final stored value (the gather just reads + dequantizes it — fused, no prepass). Pass the pools
    and scales to `fp8_attention`; build the apples-to-apples oracle with `dequantize_fp8_e4m3`.
    """
    k_q, scale_k = quantize_fp8_e4m3(k)
    v_q, scale_v = quantize_fp8_e4m3(v)
    k_pool, v_pool, block_table, n_k = build_paged_kv(k_q, v_q, page_size, shuffle=shuffle, seed=seed)
    return k_pool, v_pool, block_table, n_k, scale_k, scale_v


def quantize_nvfp4(x):
    """Per-tensor + per-16-block NVFP4 quantize (the v10 storage format).

    Returns `(packed_u8, micro_u8, scale)`:
      * `packed_u8`  : `[..., d/2]` uint8 — two 4-bit E2M1 codes per byte (low nibble = even head-dim
                       index, high nibble = odd), matching the kernel's `t&1` nibble select.
      * `micro_u8`   : `[..., d/16]` uint8 — one E4M3 micro-scale per 16-element block.
      * `scale`      : the FP32 per-tensor scale.
    The kernel reconstructs `x_hat = e2m1(code) * dequant_e4m3(micro_u8) * scale`; `dequantize_nvfp4`
    reproduces the SAME value for the apples-to-apples oracle. Two-level scaling: each block's E4M3
    micro-scale carries that block's amax/6, itself divided by `scale` so it lands in the E4M3 range.
    """
    import torch

    d = x.shape[-1]
    assert d % _NVFP4_BLOCK == 0, f"head_dim {d} must be a multiple of {_NVFP4_BLOCK} for NVFP4"
    xf = x.float()
    global_amax = xf.abs().amax().clamp_min(1e-12)
    # Per-tensor scale: largest block-scale (global_amax/6) maps onto E4M3 max (448).
    scale = (global_amax / (_E2M1_MAX * _E4M3_MAX)).item()

    blocks = xf.reshape(*x.shape[:-1], d // _NVFP4_BLOCK, _NVFP4_BLOCK)
    block_amax = blocks.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)     # [...,d/16,1]
    micro_fp8 = (block_amax / _E2M1_MAX / scale).to(torch.float8_e4m3fn)      # E4M3 micro-scale
    eff_micro = micro_fp8.float() * scale                                     # [...,d/16,1] effective
    micro_u8 = micro_fp8.squeeze(-1).contiguous().view(torch.uint8)           # [...,d/16]

    # Round each element to the nearest E2M1 level on |x / eff_micro|, keep the sign.
    y = blocks / eff_micro
    mids = torch.tensor(_E2M1_MIDPOINTS, device=x.device, dtype=torch.float32)
    mag_idx = torch.searchsorted(mids, y.abs().contiguous()).to(torch.uint8)  # 0..7
    sign = (y < 0).to(torch.uint8)
    code = ((sign << 3) | mag_idx).reshape(*x.shape[:-1], d // 2, 2)          # [...,d/2,2]
    packed = (code[..., 0] | (code[..., 1] << 4)).contiguous()               # [...,d/2] uint8
    return packed, micro_u8, scale


def dequantize_nvfp4(packed_u8, micro_u8, scale):
    """Inverse of `quantize_nvfp4` — the EXACT value the kernel stages into smem (the oracle operand).
    Unpacks both nibbles, maps each E2M1 code to its magnitude, applies sign, and multiplies by the
    per-16 E4M3 micro-scale and the per-tensor scale."""
    import torch

    d = packed_u8.shape[-1] * 2
    lo = packed_u8 & 0xF
    hi = packed_u8 >> 4
    code = torch.stack([lo, hi], dim=-1).reshape(*packed_u8.shape[:-1], d)    # [...,d]
    levels = torch.tensor(_E2M1_LEVELS, device=packed_u8.device, dtype=torch.float32)
    mag = levels[(code & 0x7).long()]
    sign = ((code >> 3) & 1).float()
    e2m1 = mag * (1.0 - 2.0 * sign)                                           # signed E2M1 value
    micro = (micro_u8.view(torch.float8_e4m3fn).float() * scale)             # [...,d/16]
    micro = micro.repeat_interleave(_NVFP4_BLOCK, dim=-1)                     # [...,d]
    return e2m1 * micro


def build_paged_kv_nvfp4(k, v, page_size: int, *, shuffle: bool = True, seed: int | None = None):
    """NVFP4 analogue of `build_paged_kv_fp8`: quantize the dense KV to NVFP4, then page the packed
    nibbles AND the per-16 micro-scales through the SAME physical permutation.

    Returns `(k_pack, k_micro, v_pack, v_micro, block_table, n_k, scale_k, scale_v)`. The packed pools
    are `[B*n_logical, page_size, H, d/2]` and the micro pools `[..., d/16]`, both uint8 in the same
    shuffled physical order so ONE `block_table` indexes them together (build_paged_kv pages along N_k
    and is feature-width-agnostic; sharing a seed makes both permutations identical). Quantize BEFORE
    paging so each physical byte is final — the gather reads + dequantizes per tile (fused, no prepass).
    """
    import torch

    if seed is None:
        seed = int(torch.randint(0, 2**31 - 1, (1,)).item())   # pin a seed so both perms match
    k_pack, k_micro, scale_k = quantize_nvfp4(k)
    v_pack, v_micro, scale_v = quantize_nvfp4(v)
    kp_pool, vp_pool, block_table, n_k = build_paged_kv(k_pack, v_pack, page_size,
                                                        shuffle=shuffle, seed=seed)
    km_pool, vm_pool, bt2, _ = build_paged_kv(k_micro, v_micro, page_size, shuffle=shuffle, seed=seed)
    assert torch.equal(block_table, bt2), "NVFP4 packed/micro pools must share one permutation"
    return kp_pool, km_pool, vp_pool, vm_pool, block_table, n_k, scale_k, scale_v


def paged_attention(q, k_pool, v_pool, block_table, page_size, n_k, *,
                    scale: float | None = None, causal: bool = False, q_offset: int = 0,
                    backend: str = DEFAULT_PAGED_BACKEND):
    """Paged-KV scaled dot-product attention via a chosen hand-written kernel.

    causal   : apply a causal mask (key j excluded when j > query_i + q_offset).
    q_offset : logical position of the query's first row. At decode, set q_offset = n_k - N_q so the
               single query sits at the end of the cache and attends the whole thing (otherwise causal
               at row 0 attends only key 0 — the degenerate case v6 hit).
    scale    : softmax scale; None -> 1/sqrt(head_dim).
    """
    # Imported here (not at module top) so `import fa_kernels` succeeds on a CPU-only box; dispatch
    # pulls in torch + the CUDA toolchain only when a kernel actually runs. Mirrors attention().
    from .dispatch import get_backend

    if scale is None:
        scale = 1.0 / (q.shape[-1] ** 0.5)
    module = get_backend(backend)
    return module.forward(q, k_pool, v_pool, block_table, int(page_size), int(n_k),
                          scale, causal, int(q_offset))


def fp8_attention(q, k_pool, v_pool, block_table, page_size, n_k, scale_k, scale_v, *,
                  scale: float | None = None, causal: bool = False, q_offset: int = 0,
                  backend: str = DEFAULT_FP8_BACKEND):
    """FP8 E4M3 KV-cache GQA decode (v9). Same paged/GQA contract as `gqa_attention`, but the KV pools
    are FP8 E4M3 bytes (uint8) and the kernel dequantizes each tile with the per-tensor scales.

    Build the inputs with `build_paged_kv_fp8` (which returns `scale_k, scale_v` alongside the uint8
    pools). The score-stationary inner loop is byte-identical to v8.7; only the KV storage precision
    differs, so this is a clean byte-only ablation against `gqa_attention(backend="v8_gqa_ss")`.

    Shapes:
        q           : [B, H_q,  N_q, d]  (FP16/FP32; decode N_q = 1)
        k_pool      : [num_blocks, page_size, H_kv, d]  uint8 (FP8 E4M3 bytes)
        v_pool      : [num_blocks, page_size, H_kv, d]  uint8
        block_table : [B, n_logical]  int32
        scale_k/scale_v : per-tensor FP32 dequant scales from `build_paged_kv_fp8`.
    """
    from .dispatch import get_backend

    if scale is None:
        scale = 1.0 / (q.shape[-1] ** 0.5)
    module = get_backend(backend)
    return module.forward(q, k_pool, v_pool, block_table, int(page_size), int(n_k),
                          float(scale_k), float(scale_v), scale, causal, int(q_offset))


def nvfp4_attention(q, k_pack, k_micro, v_pack, v_micro, block_table, page_size, n_k, scale_k, scale_v,
                    *, scale: float | None = None, causal: bool = False, q_offset: int = 0,
                    backend: str = DEFAULT_NVFP4_BACKEND):
    """NVFP4 KV-cache GQA decode (v10). Forks v9 (fp8_attention) changing ONE variable — KV storage
    format: the paged K/V pools hold packed 4-bit E2M1 nibbles plus a per-16 E4M3 micro-scale, which
    the kernel dequantizes per tile (fused) into the same FP16 smem the score-stationary inner loop
    reads. So this is a clean byte-only ablation against `fp8_attention` (FP8) and v8.7 (FP16).

    Build the inputs with `build_paged_kv_nvfp4`. Shapes:
        q           : [B, H_q,  N_q, d]  (FP16/FP32; decode N_q = 1)
        k_pack/v_pack   : [num_blocks, page_size, H_kv, d/2]   uint8 (packed E2M1 nibbles)
        k_micro/v_micro : [num_blocks, page_size, H_kv, d/16]  uint8 (E4M3 micro-scales)
        block_table : [B, n_logical]  int32
        scale_k/scale_v : per-tensor FP32 scales from `build_paged_kv_nvfp4`.
    """
    from .dispatch import get_backend

    if scale is None:
        scale = 1.0 / (q.shape[-1] ** 0.5)
    module = get_backend(backend)
    return module.forward(q, k_pack, k_micro, v_pack, v_micro, block_table, int(page_size), int(n_k),
                          float(scale_k), float(scale_v), scale, causal, int(q_offset))


def build_paged_kv_mla(latent, page_size: int, *, shuffle: bool = True, seed: int | None = None):
    """MLA (v11) latent-KV pager: quantize ONE shared latent to NVFP4, then page the packed nibbles AND
    the per-16 micro-scales through the SAME physical permutation.

    Unlike `build_paged_kv_nvfp4` (separate K and V pools), MLA stores a SINGLE latent per token that
    serves as both K (full DQK width = kv_lora_rank + rope_dim) and V (the first kv_lora_rank dims), so
    there is no V pool — the real ~93% KV-cache reduction. `latent` is `[B, 1, N_k, DQK]` (one latent
    "head"; H=1). Returns `(l_pack, l_micro, block_table, n_k, scale_l)`: the packed pool is
    `[B*n_logical, page_size, 1, DQK/2]` and the micro pool `[..., 1, DQK/16]`, both uint8 in the same
    shuffled physical order so ONE `block_table` indexes them together. Quantize BEFORE paging so each
    physical byte is final — the kernel reads + dequantizes per tile (fused, no prepass). Pass the pools,
    `kv_lora_rank`, and `scale_l` to `mla_attention`; build the oracle with `sdpa_reference_mla`.
    """
    import torch

    assert latent.dim() == 4 and latent.shape[1] == 1, \
        f"latent must be [B, 1, N_k, DQK] (one latent head); got {tuple(latent.shape)}"
    if seed is None:
        seed = int(torch.randint(0, 2**31 - 1, (1,)).item())   # pin a seed so both perms match
    l_pack, l_micro, scale_l = quantize_nvfp4(latent)
    lp_pool, lm_pool, block_table, n_k = build_paged_kv(l_pack, l_micro, page_size,
                                                        shuffle=shuffle, seed=seed)
    return lp_pool, lm_pool, block_table, n_k, scale_l


def mla_attention(q, l_pack, l_micro, block_table, page_size, n_k, kv_lora_rank, scale_l, *,
                  scale: float | None = None, causal: bool = False, q_offset: int = 0,
                  backend: str = DEFAULT_MLA_BACKEND):
    """MLA latent-KV decode (v11). Forks v10 (nvfp4_attention) changing ONE variable — the attention
    SHAPE: GQA-over-H_kv-heads -> MQA-over-ONE-shared-latent. All `h_q` query heads share the single
    latent (M = h_q, not M = G), so >1 warp is active at N_q=1 (the per-CTA lever v10 proved is the
    decode wall) and decode AI rises 2G/b -> ~3.78*h_q/b. The latent serves as both K (full DQK) and V
    (first `kv_lora_rank` dims), so there is ONE pool (no V pool). The absorbed W^UK/W^UV fold into the
    OFFLINE Q/O projections, so `q` is `q_absorbed` (already in the latent basis) and the output is
    `O_latent` (re-projected by W^UV offline). NVFP4 latent storage is carried byte-identical from v10 so
    this is a clean SHAPE-only A/B vs `nvfp4_attention`.

    Build the inputs with `build_paged_kv_mla`. Shapes:
        q (q_absorbed) : [B, H_q, N_q, DQK]   (FP16/FP32; decode N_q = 1; DQK = kv_lora_rank + rope_dim)
        l_pack         : [num_blocks, page_size, 1, DQK/2]   uint8 (packed E2M1 nibbles)
        l_micro        : [num_blocks, page_size, 1, DQK/16]  uint8 (E4M3 micro-scales)
        block_table    : [B, n_logical]  int32
        kv_lora_rank   : DV — the PV/output width (the first DV latent dims; RoPE dims carry no value)
        scale_l        : per-tensor FP32 scale from `build_paged_kv_mla`.
    Returns O_latent [B, H_q, N_q, kv_lora_rank] (FP32). NOTE the default `scale` is 1/sqrt(DQK); a real
    model should pass the MLA scale 1/sqrt(qk_head_dim) explicitly (kernel and oracle must agree).
    """
    from .dispatch import get_backend

    if scale is None:
        scale = 1.0 / (q.shape[-1] ** 0.5)
    module = get_backend(backend)
    return module.forward(q, l_pack, l_micro, block_table, int(page_size), int(n_k),
                          int(kv_lora_rank), float(scale_l), scale, causal, int(q_offset))


def gqa_attention(q, k_pool, v_pool, block_table, page_size, n_k, *,
                  scale: float | None = None, causal: bool = False, q_offset: int = 0,
                  backend: str = DEFAULT_GQA_BACKEND):
    """GQA M-packed paged-KV decode (v8+). Same paged contract as `paged_attention`, but the KV pool
    has only `H_kv` heads while `q` has `H_q = G * H_kv` query heads — the `G` query heads of a group
    share one KV head, and the kernel packs them into the score GEMM's M dimension (KV read once per
    group, G compute-warps active). The group factor `G = H_q // H_kv` is derived from the shapes.

    Shapes (note the head-count split vs paged_attention):
        q           : [B, H_q,  N_q, d]                      (decode: N_q = 1)
        k_pool      : [num_blocks, page_size, H_kv, d]       H_kv = H_q / G
        v_pool      : [num_blocks, page_size, H_kv, d]
        block_table : [B, n_logical]  int32

    `causal`/`q_offset` behave exactly as in `paged_attention` (decode: q_offset = n_k - N_q so the
    single query attends the whole cache). Output is dense [B, H_q, N_q, d].
    """
    from .dispatch import get_backend

    if scale is None:
        scale = 1.0 / (q.shape[-1] ** 0.5)
    module = get_backend(backend)
    return module.forward(q, k_pool, v_pool, block_table, int(page_size), int(n_k),
                          scale, causal, int(q_offset))
