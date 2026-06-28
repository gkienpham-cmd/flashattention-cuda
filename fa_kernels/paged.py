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

# Max magnitude representable by FP8 E4M3 (e4m3fn: no inf, max normal 448). Per-tensor scale maps the
# tensor's amax onto this so the full dynamic range is used. (int8-symmetric fallback would use 127.)
_E4M3_MAX = 448.0


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
