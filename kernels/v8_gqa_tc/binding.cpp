// pybind glue for the v8 GQA M-packing TENSOR-CORE decode kernel (Cut 2a, Turing WMMA). Same
// host-entry signature as Cut 1's gqa_attention_forward — G = H_q/H_kv is derived inside the kernel.

#include <torch/extension.h>

// Declared here, defined in gqa_tc_attention.cu (compiled together by cpp_extension.load).
torch::Tensor gqa_tc_attention_forward(torch::Tensor q, torch::Tensor k_pool, torch::Tensor v_pool,
                                       torch::Tensor block_table, int64_t page_size, int64_t n_k,
                                       double scale, bool causal, int64_t q_offset);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &gqa_tc_attention_forward,
          "v8 GQA M-packing WMMA (Cut 2a) paged split-KV decode forward, FP16-in/FP32-accum "
          "(q, k_pool[.,.,H_kv,d], v_pool, block_table, page_size, n_k, scale, causal, q_offset) -> O",
          pybind11::arg("q"), pybind11::arg("k_pool"), pybind11::arg("v_pool"),
          pybind11::arg("block_table"), pybind11::arg("page_size"), pybind11::arg("n_k"),
          pybind11::arg("scale"), pybind11::arg("causal") = false,
          pybind11::arg("q_offset") = 0);
}
