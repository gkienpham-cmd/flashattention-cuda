// pybind glue for the v11 MLA (Multi-head Latent Attention) latent-KV decode kernel. The heavy lifting
// (score-stationary partial over ONE shared latent with a single transposed FP16 smem tile + fused
// per-tile NVFP4 dequant + LSE merge) lives in mla_attention.cu; this file only exposes the host entry
// to Python as `<module>.forward`. Signature = v10's forward collapsed to ONE latent pool (the latent
// serves as both K and V), plus `kv_lora_rank` (the DV/output width; DQK = q.size(3)) — the SHAPE change.

#include <torch/extension.h>

// Declared here, defined in mla_attention.cu (compiled together by cpp_extension.load).
torch::Tensor mla_attention_forward(torch::Tensor q, torch::Tensor l_pool, torch::Tensor l_scale,
                                    torch::Tensor block_table, int64_t page_size, int64_t n_k,
                                    int64_t kv_lora_rank, double scale_l,
                                    double scale, bool causal, int64_t q_offset);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &mla_attention_forward,
          "v11 MLA latent-KV decode forward (MQA over one shared latent, M=h_q; score-stationary, "
          "single transposed FP16 tile, fused per-tile NVFP4 dequant) (q_absorbed, l_pool[uint8 "
          "nibbles], l_scale[uint8 E4M3], block_table, page_size, n_k, kv_lora_rank, scale_l, scale, "
          "causal, q_offset) -> O_latent",
          pybind11::arg("q"), pybind11::arg("l_pool"), pybind11::arg("l_scale"),
          pybind11::arg("block_table"), pybind11::arg("page_size"), pybind11::arg("n_k"),
          pybind11::arg("kv_lora_rank"), pybind11::arg("scale_l"),
          pybind11::arg("scale"), pybind11::arg("causal") = false,
          pybind11::arg("q_offset") = 0);
}
