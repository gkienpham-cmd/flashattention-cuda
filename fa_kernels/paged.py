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
