# Resume bullets

The site leads with the **MechE → ML crossover** framing (Version C is primary). Versions A and B are
available for tailored applications.

## Version C — MechE / crossover (PRIMARY — use in hero-adjacent copy and contact)

> Applied bottleneck-hunting methodology from mechanical systems to GPU kernel optimization: rebuilt
> FlashAttention from scratch (17 CUDA kernels, 4 architectures), using roofline models to predict
> performance limiters before coding — achieving 8–16× speedup over PyTorch and characterizing
> NVIDIA's latest B300 GPU; demonstrated systematic predict-measure-iterate engineering on hardware
> where the "limiting factor" shifts with every design change.

## Version A — ML Systems / GPU Kernel Engineering

> Built 17 CUDA attention kernels from scratch across 4 GPU architectures (T4 through B300), achieving
> 8–16× decode speedup over PyTorch SDPA and 1785 TFLOP/s (36% FP8 peak) on NVIDIA B300 tensor cores;
> developed a roofline-first methodology with prediction-vs-measured analysis validated by Nsight
> Compute, earning a confound-free per-CTA-bound diagnosis across Turing/Ampere/Blackwell.

## Version B — ML Research / Applied ML

> Designed and measured a 12-step roofline-driven optimization arc for FlashAttention decode on NVIDIA
> Blackwell (B300/sm_103), spanning MHA→GQA→MLA and FP16→FP8→NVFP4 with prediction-vs-measured
> analysis at every step; identified per-CTA work-starvation as the decode limiter (not bandwidth) via
> Nsight Compute across 3 architectures, and confirmed it at the GEMM level with a B300 FP4-vs-FP8
> characterization showing native FP4 compute stays HBM-bound (≤5.8% of its 15 PF peak) — so low
> precision cuts KV-cache size, not decode latency; being written up as a PMBS@SC 2026 characterization
> paper (in preparation).
