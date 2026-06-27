// pybind glue for the v8.6 Arm 2 (key-ILP) GQA M-packing decode kernel. The heavy lifting
// (GQA-packed partial with a KU=4-unrolled key loop + LSE merge) lives in gqa_ilp_attention.cu; this
// file only exposes the host entry point to Python as `<module>.forward`. Signature is identical to v8
// Cut 1's forward (G = H_q/H_kv derived inside the kernel, so no new argument).

#include <torch/extension.h>

// Declared here, defined in gqa_ilp_attention.cu (compiled together by cpp_extension.load).
torch::Tensor gqa_ilp_attention_forward(torch::Tensor q, torch::Tensor k_pool, torch::Tensor v_pool,
                                        torch::Tensor block_table, int64_t page_size, int64_t n_k,
                                        double scale, bool causal, int64_t q_offset);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &gqa_ilp_attention_forward,
          "v8.6 Arm 2 (key-ILP) GQA M-packing paged split-KV decode forward, KU=4 unrolled key loop "
          "(q, k_pool[.,.,H_kv,d], v_pool, block_table, page_size, n_k, scale, causal, q_offset) -> O",
          pybind11::arg("q"), pybind11::arg("k_pool"), pybind11::arg("v_pool"),
          pybind11::arg("block_table"), pybind11::arg("page_size"), pybind11::arg("n_k"),
          pybind11::arg("scale"), pybind11::arg("causal") = false,
          pybind11::arg("q_offset") = 0);
}
