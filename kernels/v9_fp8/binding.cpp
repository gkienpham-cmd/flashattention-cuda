// pybind glue for the v9 FP8 E4M3 KV-cache decode kernel. The heavy lifting (score-stationary partial
// with FP16 transposed smem + fused per-tile FP8 dequant + LSE merge) lives in fp8_attention.cu; this
// file only exposes the host entry to Python as `<module>.forward`. Signature = v8.7's forward plus two
// per-tensor FP8 dequant scales (scale_k, scale_v); the KV pools are FP8 E4M3 bytes passed as uint8.

#include <torch/extension.h>

// Declared here, defined in fp8_attention.cu (compiled together by cpp_extension.load).
torch::Tensor fp8_attention_forward(torch::Tensor q, torch::Tensor k_pool, torch::Tensor v_pool,
                                    torch::Tensor block_table, int64_t page_size, int64_t n_k,
                                    double scale_k, double scale_v,
                                    double scale, bool causal, int64_t q_offset);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &fp8_attention_forward,
          "v9 FP8 E4M3 KV-cache GQA M-packing paged split-KV decode forward (score-stationary, fused "
          "per-tile dequant) (q, k_pool[uint8,.,.,H_kv,d], v_pool[uint8], block_table, page_size, n_k, "
          "scale_k, scale_v, scale, causal, q_offset) -> O",
          pybind11::arg("q"), pybind11::arg("k_pool"), pybind11::arg("v_pool"),
          pybind11::arg("block_table"), pybind11::arg("page_size"), pybind11::arg("n_k"),
          pybind11::arg("scale_k"), pybind11::arg("scale_v"),
          pybind11::arg("scale"), pybind11::arg("causal") = false,
          pybind11::arg("q_offset") = 0);
}
