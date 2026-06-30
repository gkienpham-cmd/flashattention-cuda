// pybind glue for the v12 native tensor-core MLA decode kernel. Same forward contract as v11's binding
// (q_absorbed, ONE NVFP4 latent pool + E4M3 micro-scales, block table, …) PLUS `engine` to pick the
// tcgen05 arm: 0 = FP8-dense MMA (Arm 1, M≥64), 1 = native NVFP4 block-scaled MMA (Arm 2, M≥128). The
// heavy lifting (the CUTLASS ex77 fork) lives in mla_tc_attention.cu; this file only exposes the host
// entry to Python as `<module>.forward`. Storage stays byte-identical to v11 so the A/B isolates the
// engine — the kernel reads the SAME paged NVFP4 latent bytes and dequants to FP8/bf16 at the SMEM stage.

#include <torch/extension.h>

// Declared here, defined in mla_tc_attention.cu (compiled together by cpp_extension.load).
torch::Tensor mla_tc_attention_forward(torch::Tensor q, torch::Tensor l_pool, torch::Tensor l_scale,
                                       torch::Tensor block_table, int64_t page_size, int64_t n_k,
                                       int64_t kv_lora_rank, double scale_l,
                                       double scale, bool causal, int64_t q_offset,
                                       int64_t engine);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &mla_tc_attention_forward,
          "v12 native tcgen05 tensor-core MLA latent-KV decode forward (engine 0=FP8 MMA / 1=NVFP4 MMA; "
          "NVFP4 latent storage byte-identical to v11) (q_absorbed, l_pool[uint8 nibbles], l_scale[uint8 "
          "E4M3], block_table, page_size, n_k, kv_lora_rank, scale_l, scale, causal, q_offset, engine) "
          "-> O_latent",
          pybind11::arg("q"), pybind11::arg("l_pool"), pybind11::arg("l_scale"),
          pybind11::arg("block_table"), pybind11::arg("page_size"), pybind11::arg("n_k"),
          pybind11::arg("kv_lora_rank"), pybind11::arg("scale_l"),
          pybind11::arg("scale"), pybind11::arg("causal") = false,
          pybind11::arg("q_offset") = 0, pybind11::arg("engine") = 0);
}
