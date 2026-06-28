// pybind glue for the v10 NVFP4 KV-cache decode kernel. The heavy lifting (score-stationary partial
// with FP16 transposed smem + fused per-tile NVFP4 dequant + LSE merge) lives in nvfp4_attention.cu;
// this file only exposes the host entry to Python as `<module>.forward`. Signature = v9's forward but
// the KV is two uint8 pools per tensor — packed E2M1 nibbles ([.,.,H_kv,d/2]) plus E4M3 micro-scales
// ([.,.,H_kv,d/16]) — alongside the per-tensor scales (scale_k, scale_v).

#include <torch/extension.h>

// Declared here, defined in nvfp4_attention.cu (compiled together by cpp_extension.load).
torch::Tensor nvfp4_attention_forward(torch::Tensor q, torch::Tensor k_pool, torch::Tensor k_scale,
                                      torch::Tensor v_pool, torch::Tensor v_scale,
                                      torch::Tensor block_table, int64_t page_size, int64_t n_k,
                                      double scale_k, double scale_v,
                                      double scale, bool causal, int64_t q_offset);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &nvfp4_attention_forward,
          "v10 NVFP4 KV-cache GQA M-packing paged split-KV decode forward (score-stationary, fused "
          "per-tile NVFP4 dequant) (q, k_pool[uint8 nibbles], k_scale[uint8 E4M3], v_pool, v_scale, "
          "block_table, page_size, n_k, scale_k, scale_v, scale, causal, q_offset) -> O",
          pybind11::arg("q"), pybind11::arg("k_pool"), pybind11::arg("k_scale"),
          pybind11::arg("v_pool"), pybind11::arg("v_scale"),
          pybind11::arg("block_table"), pybind11::arg("page_size"), pybind11::arg("n_k"),
          pybind11::arg("scale_k"), pybind11::arg("scale_v"),
          pybind11::arg("scale"), pybind11::arg("causal") = false,
          pybind11::arg("q_offset") = 0);
}
