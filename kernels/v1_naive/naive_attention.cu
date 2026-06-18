// v1 naive attention — the baseline that exposes the HBM bandwidth wall.
//
// THE WHOLE POINT of this kernel is to be slow in an instructive way: it computes attention
// as three separate kernel launches, and the full S = QK^T score matrix (N_q x N_k per head)
// is written to global memory (HBM) by pass 1, read+written by pass 2, and read again by
// pass 3. That S round-trip is O(N^2) traffic with O(N^2 * d) compute, giving an arithmetic
// intensity (~54-63 FLOP/byte at d=64) far below the T4's FP16 tensor-core ridge of 203
// FLOP/byte. So we are pinned to the 320 GB/s memory roof. Every later version earns its
// speedup by killing this round-trip. We measure that, we don't assert it.
//
// Precision: FP32 in, FP32 accumulate, FP32 out (the "B1" baseline). Intentionally simple so
// it is an exact-ish correctness anchor. FP16+FP32-accum is its own later step; the FP16-fundamentals
// versions through v2 (tiling) stay FP32 so each step isolates one variable.
//
// Layout: q,k,v are [B, H, N, d] row-major (the project-wide convention). For causal masking
// we assume self-attention with N_q == N_k aligned at the end, masking key j > query i.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>   // at::cuda::getCurrentCUDAStream
#include <cuda.h>
#include <cuda_runtime.h>
#include <cfloat>   // FLT_MAX

namespace {

constexpr int kThreads = 256;  // a plain, occupancy-friendly block size; nothing clever here

// Round-up integer division, for sizing a 1D grid to cover `n` elements.
inline int ceil_div(int64_t n, int b) { return static_cast<int>((n + b - 1) / b); }

// ---------------------------------------------------------------------------------------
// Pass 1: S = scale * (Q @ K^T)
// One thread per output element S[bh, i, j]. Each thread does a length-d dot product.
// This is deliberately naive: the dot reads Q row i and K row j straight from HBM with no
// tiling and no reuse, so neighboring threads re-read the same K/Q rows from global memory.
// ---------------------------------------------------------------------------------------
__global__ void qk_kernel(const float* __restrict__ Q,
                          const float* __restrict__ K,
                          float* __restrict__ S,
                          int N_q, int N_k, int d, float scale,
                          int64_t total) {
    for (int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
         idx < total; idx += (int64_t)gridDim.x * blockDim.x) {
        // Decode the flat index into (bh, i, j). bh = batch*head, flattened together since
        // attention is computed independently per (batch, head).
        int j  = idx % N_k;
        int i  = (idx / N_k) % N_q;
        int bh = idx / ((int64_t)N_k * N_q);

        // Base offsets of query row i and key row j within this (batch,head)'s [N, d] block.
        const float* q_row = Q + ((int64_t)bh * N_q + i) * d;
        const float* k_row = K + ((int64_t)bh * N_k + j) * d;

        float acc = 0.f;
        for (int t = 0; t < d; ++t) acc += q_row[t] * k_row[t];  // FP32 accumulate

        S[idx] = acc * scale;  // O(N^2) writes to HBM — this is the traffic we will later avoid
    }
}

// ---------------------------------------------------------------------------------------
// Pass 2: row-wise softmax over the key axis, in place in S (so P overwrites S).
// One thread per row (bh, i). Three sweeps over the N_k row: max, sum(exp), normalize.
// The max-subtraction is the standard numerically-stable softmax (avoids exp overflow).
// Causal masking is applied here: a key j > query i contributes nothing.
// ---------------------------------------------------------------------------------------
__global__ void softmax_kernel(float* __restrict__ S,
                               int N_q, int N_k, bool causal,
                               int64_t n_rows) {
    int64_t row = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (row >= n_rows) return;

    int i = row % N_q;                 // query position, needed for the causal cutoff
    float* s = S + row * (int64_t)N_k; // this row's N_k scores

    // For causal self-attention, query i may attend to keys [0, i]. j_max is that cutoff.
    int j_max = causal ? min(i, N_k - 1) : (N_k - 1);

    // Sweep 1: row max (over the valid, unmasked keys).
    float m = -FLT_MAX;
    for (int j = 0; j <= j_max; ++j) m = fmaxf(m, s[j]);

    // Sweep 2: exponentiate (shifted by the max) and accumulate the normalizer.
    float denom = 0.f;
    for (int j = 0; j <= j_max; ++j) {
        float e = __expf(s[j] - m);    // __expf: fast-path exp on the MUFU unit (the future "exp wall")
        s[j] = e;
        denom += e;
    }

    // Sweep 3: normalize valid keys; zero out the masked tail so pass 3 can ignore it.
    float inv = 1.f / denom;
    for (int j = 0; j <= j_max; ++j) s[j] *= inv;
    for (int j = j_max + 1; j < N_k; ++j) s[j] = 0.f;
}

// ---------------------------------------------------------------------------------------
// Pass 3: O = P @ V
// One thread per output element O[bh, i, t] (t over head dim). Each thread sums over the
// N_k keys. Like pass 1, this re-reads V columns from HBM with no reuse — naive on purpose.
// ---------------------------------------------------------------------------------------
__global__ void pv_kernel(const float* __restrict__ S,
                          const float* __restrict__ V,
                          float* __restrict__ O,
                          int N_q, int N_k, int d,
                          int64_t total) {
    for (int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
         idx < total; idx += (int64_t)gridDim.x * blockDim.x) {
        int t  = idx % d;
        int i  = (idx / d) % N_q;
        int bh = idx / ((int64_t)d * N_q);

        const float* p_row = S + ((int64_t)bh * N_q + i) * N_k;  // probabilities for query i
        const float* v_col = V + ((int64_t)bh * N_k) * d + t;    // V[bh, :, t], strided by d

        float acc = 0.f;
        for (int j = 0; j < N_k; ++j) acc += p_row[j] * v_col[(int64_t)j * d];

        O[idx] = acc;
    }
}

}  // anonymous namespace

// Host entry point: orchestrates the three passes. Returns O = attention(Q, K, V).
// Exposed to Python by binding.cpp as `forward`.
torch::Tensor naive_attention_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                                      double scale, bool causal) {
    // --- contract checks: fail loudly rather than compute garbage ---
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "q,k,v must be CUDA tensors");
    TORCH_CHECK(q.dtype() == torch::kFloat32, "v1 naive is FP32-only (the clean baseline)");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "expected [B,H,N,d] tensors");
    q = q.contiguous(); k = k.contiguous(); v = v.contiguous();

    const int B = q.size(0), H = q.size(1), N_q = q.size(2), d = q.size(3);
    const int N_k = k.size(2);
    TORCH_CHECK(k.size(3) == d && v.size(3) == d, "head_dim mismatch across q,k,v");
    TORCH_CHECK(v.size(2) == N_k, "K and V must share sequence length");

    const int64_t BH = (int64_t)B * H;

    // S is the full score/probability matrix in HBM — the materialization we will later remove.
    auto S = torch::empty({B, H, N_q, N_k}, q.options());
    auto O = torch::empty_like(q);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Pass 1: QK^T. One thread per S element.
    const int64_t n_s = BH * N_q * N_k;
    qk_kernel<<<ceil_div(n_s, kThreads), kThreads, 0, stream>>>(
        q.data_ptr<float>(), k.data_ptr<float>(), S.data_ptr<float>(),
        N_q, N_k, d, (float)scale, n_s);

    // Pass 2: softmax. One thread per row.
    const int64_t n_rows = BH * N_q;
    softmax_kernel<<<ceil_div(n_rows, kThreads), kThreads, 0, stream>>>(
        S.data_ptr<float>(), N_q, N_k, causal, n_rows);

    // Pass 3: P@V. One thread per O element.
    const int64_t n_o = BH * N_q * d;
    pv_kernel<<<ceil_div(n_o, kThreads), kThreads, 0, stream>>>(
        S.data_ptr<float>(), v.data_ptr<float>(), O.data_ptr<float>(),
        N_q, N_k, d, n_o);

    return O;
}
