# v12 tensor-core MLA — B300/sm_103a runbook (PRIVILEGED box: implement → build → measure)

The B300 measured core for v12. Unlike v10/v11 (which shipped a finished CUDA-core kernel and only
*measured* on the box), **v12's kernel body is written here** — fork CUTLASS example 77 onto tcgen05.
So this runbook has an extra front half (Parts 1–4: resolve Q1 → implement Arm 1 → build → correctness)
before the measurement half (Parts 5–8) that mirrors `docs/v11-b300-runbook.md`.

**Why a PRIVILEGED / bare-metal box this time (not the unprivileged vast.ai container v10/v11 used):**
paying v11's #1 honesty debt — **ncu** — needs the host kernel-module gate open (unprivileged containers
return `ERR_NVGPUCTRPERM`). Rent a privileged container (`--cap-add` works) or a bare-metal B300/sm_103a.
Also needed: **CUDA 12.9+** (sm_103a) and **CUTLASS 4.x** (ex77 + the tcgen05/`mxf4nvf4` collectives).

**Pre-registered prediction (so you read the result, not hope for a number):** even on tensor cores,
single-token decode is SMEM-BW / MMA-pipeline-depth-bound → realized TFLOP/s **well below** the FP8/FP4
peak, v12 closes v11's 4× self-gap *toward* the TC ceiling but does **not** beat FlashMLA's ~410 TFLOP/s.
The measured shortfall is the result (`results.md` Step 12, `interview-prep.md` C18).

---

## Part 0 — Provision + verify the box (do these BEFORE any work)

```bash
# 0a. ncu probe — the whole reason for a privileged box. If this FAILS, destroy + re-rent.
. /venv/main/bin/activate 2>/dev/null; export PATH=/venv/main/bin:$PATH
ncu --version                                  # want a recent ncu (CUDA 12.9 toolkit)
python - <<'PY' > /tmp/_p.py
import torch; x=torch.randn(4096,4096,device='cuda'); (x@x).sum().item()
PY
ncu --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed -c 1 python /tmp/_p.py 2>&1 | tail -5
# PASS = prints a DRAM throughput %. FAIL = ERR_NVGPUCTRPERM -> wrong host, re-rent privileged/bare-metal.

# 0b. Arch + toolchain.
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv   # want sm_103 (10.3), 288 GB
nvcc --version | grep release                                       # want >= 12.9 (sm_103a)

# 0c. CUTLASS 4.x (ex77 + tcgen05). Clone once; note the path for the build include flags.
git clone --depth 1 --branch v4.0.0 https://github.com/NVIDIA/cutlass.git /opt/cutlass || \
  git clone --depth 1 https://github.com/NVIDIA/cutlass.git /opt/cutlass
export CUTLASS=/opt/cutlass
ls $CUTLASS/examples/77_blackwell_fmha    # confirm ex77 exists in this CUTLASS version
```

If you can only get an **unprivileged** box: you can still do Parts 1–5 + 7 (build, correctness, the
engine A/B, nsys 2025.3.2+) and log ncu as owed — but the headline Q2 deliverable (achieved-TFLOP/s +
SMEM-BW, ncu-validated) needs the privileged box, so prefer it.

---

## Part 1 — Resolve §9 Q1 in the ex77 source BEFORE implementing Arm 2

The native-FP4 arm is banked on M=128 packing as ONE tcgen05 GEMM. Verify in the source, not by hope:

```bash
cd $CUTLASS/examples/77_blackwell_fmha
# (a) the realized M-blocking / group cap (the reported "num_groups 32 vs 128" risk):
grep -rniE 'num_groups|kNumGroups|cta_group|TileShape|kBlockM|get<0>' . | grep -iE 'group|128|m\b' | head -40
# (b) how heads map into the M tile (head-count -> M):
grep -rniE 'head|num_heads|kNumHeads|seqlen_q|q_len' . | head -40
# (c) the weight-absorbed latent-512/rope-64 decode variant (NOT the prefill kernel):
grep -rniE 'latent|512|rope|64|absorb|deepseek|mla|decode' . | head -40
```

**Decide:** if the head-pack maps cleanly to M=128 in ONE GEMM → Arm 2 is structurally viable (the
decoupled-RoPE term `q_R[128×64]·k_R[N_k×64]ᵀ` accumulates into the SAME score → it splits the **K-dim**,
an extra K=64 GEMM, NOT the M-tile). If `num_groups` caps M at 32 → M fragments into 16/group → **do
NOT bank Arm 2**; ship Arm 1 only and log the fragmentation as the §9-Q1 finding (still a result). Record
the verdict in `results.md` Step 12 before writing Arm 2.

---

## Part 2 — Implement the kernel body (Arm 1 = FP8 first). The hard step.

The scaffold `kernels/v12_mla_tc/mla_tc_attention.cu` validates inputs then `TORCH_CHECK(false, …)`. Fill
the marked region. **Recommended strategy = HYBRID** (keeps the A/B clean): keep v11's host orchestration
byte-identical and replace ONLY the inner matmul with a CUTLASS tcgen05 MMA.

1. **Copy v11's host + merge + choose_splits verbatim** from `kernels/v11_mla/mla_attention.cu` into the
   v12 `.cu` (the split-KV launch, `mla_merge_kernel`, `choose_splits`, the `(DQK,DV)` dispatch macro,
   the `dequant_nvfp4`/`dequant_e4m3` helpers — already present in the scaffold). This guarantees
   split-KV / LSE merge / NVFP4 storage are byte-identical → the A/B isolates the engine.
2. **Replace the partial kernel's inner loop** (v11's `mla_partial_kernel` score-stationary GEMV) with a
   CUTLASS **tile-level / collective tcgen05 MMA**:
   - **QK:** `Q'[M=h_q × DQK] · C_tile[TN × DQK]ᵀ → S[M × TN]` per key tile. Stage the latent tile as
     today (paged gather + **fused NVFP4→FP8 dequant at the SMEM stage** — reuse the dequant helpers, but
     write FP8/`e4m3` into SMEM instead of FP16), then issue the **FP8 MMA (gate M≥64)** via a CUTLASS
     `cute` collective (study ex77's mainloop for the tmem/SMEM staging + `cta_group::2`).
   - **softmax:** keep ≥ FP16 (FP4 scores collapse softmax — v10). Run the online-softmax recurrence on
     the FP32 accumulator exactly as v11 (one max/sum per key group), so the merge stays identical.
   - **PV:** `P[M × TN] · V_tile[TN × DV] → O[M × DV]` accumulate; V is the first DV dims of the same
     dequantized latent tile (no separate V — the v11 one-pool trick).
   - **Scores ≥ FP16** and the FP32 `(m,l,O)` partial written exactly as v11 (`O_partial/m_partial/
     l_partial`), so `mla_merge_kernel` is unchanged.
3. **Engine gate (the one new variable):** `engine==0` → the FP8 MMA above. `engine==1` → swap the
   collective to `kind::mxf4nvf4` (native NVFP4, M≥128/K=256/TN, per-16 E4M3 microscales) — **only after
   Part 1 says M=128 is one GEMM AND Arm 1 is green.**
4. **Build flags** (edit `bindings/load.py` — uncomment the v12 `_ARCH` entry, or set env before launch):
   ```bash
   export FA_CUDA_ARCH=103a          # -> -gencode=arch=compute_103a,code=sm_103a (Arm 2 needs sm_103a)
   export CPLUS_INCLUDE_PATH=$CUTLASS/include:$CUTLASS/tools/util/include:$CPLUS_INCLUDE_PATH
   ```
   (Or add `extra_include_paths=[f"{CUTLASS}/include", f"{CUTLASS}/tools/util/include"]` to the v12
   `build_kernel` path.) Keep TN/the `(DQK,DV)` dispatch tuples (96/64 smoke, 576/512 real) from v11.

> This is the genuine research-engineering step — expect iteration (ptxas smem/tmem limits, the
> M=128↔tile mapping, the dequant-to-FP8 staging). Land **Arm 1 correct** before touching Arm 2.

---

## Part 3 — Build + Gate 1 correctness

Open `notebooks/v12_mla_tc_gate.ipynb`, **Run cells 0–4** (deps → repo → roofline recap → build →
pytest). Or terminal:

```bash
. /venv/main/bin/activate; export PATH=/venv/main/bin:$PATH
cd /flashattention-cuda && git pull origin main
export FA_CUDA_ARCH=103a CPLUS_INCLUDE_PATH=$CUTLASS/include:$CUTLASS/tools/util/include:$CPLUS_INCLUDE_PATH
python -c "from bindings.load import build_kernel; build_kernel('v12_mla_tc'); print('built')"
python -m pytest tests/test_correctness.py -k "v12_mla_tc or v11_mla" -q
```

**Gate 1 PASS** = v12 (fp8 arm; nvfp4 arm at h_q≥128) matches the v11 oracle `sdpa_reference_mla` at
5e-2 + the absorption identity passes + v11 regresses green. If only the fp8 arm passes, that's still a
shippable v12 (Arm 1) — log Arm 2 status honestly.

---

## Part 4 — Arm 2 (native NVFP4), gated on Part 1 + Arm 1

Only if Part 1 confirmed M=128 = one GEMM and Arm 1 is green. Implement the `mxf4nvf4` collective, rebuild,
re-run `pytest -k "v12_mla_tc"` (the nvfp4-engine cases run at h_q≥128). Expect Arm 2 may be *slower* than
Arm 1 even on TC (TRT-LLM #4412) — keep it; the NVFP4-KV-*compute* path is the open novelty regardless.

---

## Part 5 — The engine A/B + the regime sweep (clock-LOCKED — pay the debt)

```bash
. /venv/main/bin/activate; export PATH=/venv/main/bin:$PATH; cd /flashattention-cuda

# (a) Engine A/B: v12 (each arm) vs v11 on the SAME latent bytes. The "vs naive" column = vs v11.
#     Clock-lock works on a privileged box (unlike v11's run). Read us/tok AND %HBM.
python -m bench.harness --backend v12_mla_tc --mla-engine fp8   --decode --heads 128 \
  --dim 576 --seq 8192 32768 | tee /tmp/v12_ab_fp8.txt
python -m bench.harness --backend v12_mla_tc --mla-engine nvfp4 --decode --heads 128 \
  --dim 576 --seq 8192 32768 | tee /tmp/v12_ab_nvfp4.txt

# (b) Regime sweep past the 132.6 MB L2 (WS overflow by construction; the Q2 deliverable). --lock-clocks
#     now LOCKS (root). Both arms; v11 as the CUDA-core comparator on the SAME shape.
for eng in fp8 nvfp4; do
  python -m bench.regime --backend v12_mla_tc --engine $eng --dim 576 --h-kv 1 --gqa-group 128 \
    --kv-lens 8192 32768 131072 524288 1048576 2097152 --lock-clocks | tee /tmp/v12_regime_$eng.txt
done
python -m bench.regime --backend v11_mla --dim 576 --h-kv 1 --gqa-group 128 \
  --kv-lens 8192 32768 131072 524288 1048576 2097152 --lock-clocks | tee /tmp/v11_regime.txt
```

**Read:** does v12's achieved TFLOP/s climb **>~10× v11's 0.75** with %HBM/%SMEM-BW rising (limiter LEFT
per-CTA — the counter lands), or does %HBM stay ~0.1–0.5% flat (still per-CTA-bound, the prediction holds)?

---

## Part 6 — The four honesty debts (the reason for the privileged box)

```bash
# (1) ncu — achieved TFLOP/s + SMEM-BW + tensor-pipe util on ONE shape (the Q2 deliverable). --profile
#     loops the kernel so ncu attaches. This is the single highest-value hardening item.
ncu --set full --kernel-name 'regex:.*' -c 1 \
  python -m bench.regime --backend v12_mla_tc --engine nvfp4 --dim 576 --gqa-group 128 \
  --profile 1,1,131072,576 2>&1 | tee /tmp/v12_ncu.txt
#   Read: smsp__inst_executed_pipe_tensor (TC util), l1tex SMEM throughput, dram %, achieved FLOP/s.
#   -> does the limiter RENAME to SMEM-BW / pipeline-depth (predicted), or flip to compute?

# (2) clock-lock: already applied via --lock-clocks in Part 5 (root). Confirm clk_cur==clk_max in the rows.

# (3) torch-baseline profile at fp16/bf16 — convert v11's "cuBLAS-TC 4x cause" from inference to
#     MEASUREMENT. The gate notebook §7 has a RUNNABLE fp16 dense-MQA cell (defines `dense_mqa`, NOT
#     v11's FP32 batched-GEMV) that times the comparator; profile that `dense_mqa` under nsys/ncu and
#     read whether it dispatches a cuBLAS/cutlass tensor-core GEMM (a sm100/sm90 *gemm* kernel name).

# (4) 2x-exp re-ablation at M=128 (Q3): EX2 throughput on the M=128 score tile vs v10/v11's M=1 0.5x
#     measurement. The fast engine raises the exp share (results.md Step 12) -> re-measure here.
```

---

## Part 7 — nsys schedule (partial vs merge), 2025.3.2+ binary

Same as v11 (stock nsys 2025.1.3 records an EMPTY sm_103 trace — install + use 2025.3.2+ explicitly):

```bash
apt-get update -q && apt-get install -y nsight-systems-2025.3.2
NSYS_DIR=$(ls -d /opt/nvidia/nsight-systems/*/ | sort -V | tail -1)
NSYS="$NSYS_DIR/bin/nsys"; [ -x "$NSYS" ] || NSYS="$NSYS_DIR/target-linux-x64/nsys"
"$NSYS" --version    # MUST print 2025.3.2+
"$NSYS" profile -o /tmp/v12_nsys --force-overwrite true --trace=cuda \
  python -m bench.regime --backend v12_mla_tc --engine nvfp4 --gqa-group 128 --profile 1,1,1048576,576
"$NSYS" stats --report cuda_gpu_kern_sum /tmp/v12_nsys.nsys-rep 2>/dev/null | tee /tmp/v12_kern_sum.txt | head -45
```

---

## Part 8 — Teardown + what to bring back

```bash
nvidia-smi --reset-gpu-clocks 2>/dev/null   # or in-notebook reset_clocks(); THEN destroy the instance
```

Bring back (push to `main` under `notebooks/`, or paste):
1. The executed `notebooks/v12_mla_tc_gate_output.ipynb`.
2. `/tmp/v12_regime_{fp8,nvfp4}.txt` + `/tmp/v11_regime.txt` — the %HBM / achieved-TFLOP/s vs N_k (headline).
3. `/tmp/v12_ab_{fp8,nvfp4}.txt` — the engine A/B vs v11.
4. `/tmp/v12_ncu.txt` — the ncu-validated TC-util / SMEM-BW (the Q2 deliverable; the deepest reviewer wound).
5. `/tmp/v12_kern_sum.txt` — the nsys partial/merge split.
6. The §9-Q1 verdict (Part 1) + the FlashMLA/FlashInfer comparator numbers (clock-locked, matched precision).

Then fill the **Gate-2 verdict** in `notebooks/v12_mla_tc_gate.ipynb` §8 → record into `results.md` /
`decisions.md` Step 12 + `interview-prep.md` C18 → **the quiz** (deferred to the very end, per Kien).
