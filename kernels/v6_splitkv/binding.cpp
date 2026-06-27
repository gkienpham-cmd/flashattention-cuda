// pybind glue for the v6 split-KV decode kernel. The heavy lifting (partial + merge kernels) lives in
// splitkv_attention.cu; this file only exposes the host entry point to Python as `<module>.forward`.

#include <torch/extension.h>

// Declared here, defined in splitkv_attention.cu (compiled together by cpp_extension.load).
torch::Tensor splitkv_attention_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                                        double scale, bool causal);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &splitkv_attention_forward,
          "v6 split-KV decode (Flash-Decoding) forward, FP16-in/FP32-accum (Q,K,V,scale,causal) -> O",
          pybind11::arg("q"), pybind11::arg("k"), pybind11::arg("v"),
          pybind11::arg("scale"), pybind11::arg("causal") = false);
}
