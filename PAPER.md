# Single-Kernel Fusion for Sequential Fitness Evaluation via WebGPU Compute Shaders

**Ahmet Baris Gunaydin**
Independent Researcher
abgunaydin94@gmail.com

---

## Abstract

Fusing sequential fitness evaluations into single GPU compute shader dispatches eliminates the per-step kernel launch overhead that dominates framework-based GPU computation. On a 1,500-timestep financial simulation, a WebGPU compute shader achieves 46.2 gen/s — 7.2× over JAX GPU with `lax.scan`+`vmap` (6.43 gen/s on Tesla T4) and 94× over PyTorch CUDA. On Acrobot-v1 (500 timesteps, RK4), the gap narrows to 1.29× over JAX GPU, revealing that the fusion advantage scales with episode length L. JAX GPU dominates on embarrassingly parallel Rastrigin (1,164 vs 170 gen/s), confirming the advantage is specific to sequential workloads. A native Metal baseline via wgpu quantifies Chrome's browser overhead at 1.92×. We show `torch.compile` fails at L≥1,000 and that WebGPU dominates CMA-ES across all tested dimensionality regimes. The insight — hand-fused compute shaders outperform even XLA-compiled loop fusion on long sequential fitness functions — applies beyond WebGPU, and WebGPU makes such fusion accessible with zero installation.

**Keywords:** WebGPU, compute shaders, kernel fusion, neuroevolution, GPU computing, evolutionary computation, browser-based computing

---

## 1. Introduction

GPU-accelerated evaluation of population-based algorithms requires dispatching the same computation across thousands of individuals per generation. When the fitness function is embarrassingly parallel (e.g., Rastrigin), modern ML frameworks like PyTorch and JAX efficiently batch these evaluations. However, many real-world fitness functions contain **sequential dependencies** — reinforcement learning rollouts, financial simulations, control system evaluations — where each timestep depends on the previous state.

For these sequential workloads, framework-based approaches face a fundamental bottleneck: **per-step kernel dispatch overhead**. PyTorch's eager execution model requires a CPU→GPU round-trip for each of the L timesteps within each generation, totaling ~15×L kernel launches per generation (matmul, activation, allocation, reduction per step). For L=1,500, this produces ~22,500 GPU operations per generation with CPU↔GPU synchronization between each.

We present a solution: **single-kernel fusion**, where the entire multi-step evaluation (all L timesteps for all N individuals) executes as a single GPU compute shader dispatch. The kernel maintains per-individual state in GPU memory and iterates through timesteps entirely on-device, eliminating all intermediate host synchronization.

This approach is implemented in WGSL (WebGPU Shading Language [3]), which provides two practical advantages: (1) zero installation — the compute shader runs in any WebGPU-capable browser, and (2) cross-vendor portability — the same shader executes on Apple Metal, NVIDIA Vulkan, AMD Vulkan, and Intel D3D12 backends via the browser's translation layer.

**Contributions:**

- **C1: Kernel fusion for sequential fitness functions.** We demonstrate that fusing a 1,500-timestep financial simulation into a single GPU dispatch yields 159× over PyTorch MPS and 94× over PyTorch CUDA, and show that `torch.compile` cannot fuse loops at this scale (Table 4). A second sequential workload (Acrobot-v1) confirms the pattern with 223× over CUDA.
- **C2: Zero-install GPU compute via WebGPU.** The complete engine (~350 lines of WGSL) runs on any WebGPU-capable browser, expanding GPU-accelerated evaluation beyond CUDA-equipped machines.
- **C3: Empirical characterization** of throughput scaling across population size, genome dimensionality, fitness function complexity, numerical precision (f32 vs f64), browser sandbox overhead (native Metal baseline), and NVIDIA CUDA comparison.

---

## 2. Related Work

**GPU-based evolutionary computation.** Langdon [1] surveys GPU implementations of genetic programming in CUDA. DEAP [2] provides a Python EA framework with optional GPU acceleration. EvoJAX [4] achieves hardware-accelerated neuroevolution via JAX on TPU/GPU. EvoGP [11] provides GPU-accelerated tree-based genetic programming. GSGP-CUDA [12] implements geometric semantic GP on CUDA. Luong et al. [10] demonstrated GPU-based island models. All require CUDA/TPU infrastructure and are not browser-accessible.

**WebGPU for computation.** ONNX Runtime Web and TensorFlow.js support WebGPU-accelerated inference. WGPY [13] provides a NumPy-like array library via WebGPU. Sengupta et al. [6] benchmark WebGL vs WebGPU for browser-based evolutionary computation, confirming WebGPU's performance advantage, but benchmark individual operations rather than implementing a full evolutionary loop in compute shaders.

**Browser-based evolutionary computation.** Duda & Dlubacz [9] demonstrated browser-based EC with JavaScript on CPU. Duda & Dlubacz [8] explored WebGL-based GPU acceleration, constrained by WebGL's lack of compute shaders. Merelo-Guervos and Garcia-Sanchez [7] modeled browser-based distributed EC systems. Our work differs by implementing the complete evolutionary loop in WGSL compute shaders with single-kernel fusion for sequential workloads.

**CMA-ES.** Hansen & Ostermeier [14] provide the foundational work on CMA-ES, the gold standard for continuous optimization in moderate dimensions. Our comparison shows that throughput-driven brute force can dominate CMA-ES under fixed wall-clock budgets when sufficient GPU parallelism is available.

---

## 3. System Architecture

### 3.1 GPU-Native Evolution Engine

The evolutionary loop runs within two WGSL compute shaders dispatched as WebGPU compute passes:

- `nn_core.wgsl` — Shared RNN forward pass (246 parameters, topology 8→16→6 with two output neurons fed back as inputs).
- `swarm_shader.wgsl` — Hosts `evaluate` and `evolve` entry points.

Genome data resides entirely in GPU VRAM. We employ a ping-pong double-buffer pattern to avoid host-side memory copies:

```
Generation N:   Read Buffer A → Evaluate → Evolve → Write Buffer B
Generation N+1: Read Buffer B → Evaluate → Evolve → Write Buffer A
```

The only CPU→GPU transfer per generation is a 16-byte uniform buffer containing the generation counter and RNG seed. Fitness results are read back via a single `mapAsync()` call per generation.

**Selection mechanism.** Tournament selection (k=3–5), uniform crossover, two-tier Gaussian mutation (5–10% large, 15–20% small perturbation). The global elite is preserved via deterministic elitism.

### 3.2 Sequential Workload: Financial Domain

Each individual is evaluated over L=1,500 timesteps of historical cryptocurrency price data (16 assets, hourly candles from Binance, ~62 days). The simulation models leveraged portfolio management with continuous allocation (1–3× leverage), asymmetric fee structures, liquidity impact, maintenance margin liquidation, and hysteresis-filtered rebalancing. Each timestep depends on the previous portfolio state, creating a strict sequential dependency chain.

### 3.3 Shared-Shader Architecture (SSoT)

The `nn_core.wgsl` kernel is injected into both training and inference compute passes at compilation time via string concatenation, enforcing mathematical parity between training and deployment. We observed 15–20% silent performance degradation when forward-pass implementations diverged by floating-point rounding differences — motivating this constraint.

### 3.4 Experimental Methodology

**Throughput benchmarks.** We benchmark against Python/NumPy (CPU), JAX with `jit` + `lax.scan` (CPU), and PyTorch (MPS and CUDA). WebGPU throughput was measured using automated Puppeteer benchmarks (N=10 runs, 20s warmup, 10 samples/run). Python baselines: N=10 runs, mean ± std.

**CMA-ES comparison.** CMA-ES [14] with default population size at DIM=100, 500, 2,000 under 30-second wall-clock budgets (N=30). Population-matched control (POP=4,096) at DIM=2,000.

**Hardware.** Primary: Apple M2 Pro (19-core GPU, 16 GB unified memory), Chrome 123, macOS 14. NVIDIA: Tesla T4 (16 GB, Google Colab), PyTorch 2.10.0+cu128, CUDA 12.8.

---

## 4. Results

### 4.1 Embarrassingly Parallel Workload (Rastrigin)

**Table 1: Rastrigin [5] Benchmark (POP=4,096, DIM=2,000)**

| System | Throughput (gen/s) | vs NumPy | N |
|---|---|---|---|
| Python/NumPy (CPU) | 3.9 ± 0.3 | — | 10 |
| JAX jit (CPU) | 20.6 ± 0.5 | 5.3× | 10 |
| PyTorch (MPS, M2 Pro) | 160.5 ± 3.6 | 41× | 10 |
| **WebGPU (Chrome, M2 Pro)** | **170.3 ± 8.4** | **44×** | 10 |
| wgpu-native (Metal, M2 Pro) | 326.5 ± 0.3 | 84× | 10 |
| PyTorch (CUDA, Tesla T4) | 311.1 ± 0.8 | 80× | 10 |
| JAX jit + vmap (CUDA, Tesla T4) | 1,163.9 ± 168.2 | 298× | 10 |

On the embarrassingly parallel Rastrigin benchmark, **JAX GPU dominates** at 1,164 gen/s — 6.8× faster than WebGPU. This is expected: XLA compilation on dedicated NVIDIA hardware with full CUDA optimization represents the ceiling for framework-based GPU computation on parallel workloads. WebGPU achieves near-parity with PyTorch MPS (1.06×) and PyTorch CUDA (0.55×).

**Native Metal baseline.** Running the identical WGSL shader through wgpu-native (Rust) calls the Metal API directly without Chrome's process isolation, IPC buffer mapping, or Dawn translation layer. Native Metal achieves **1.92× faster than WebGPU in Chrome**, quantifying browser overhead at 48%. Notably, PyTorch MPS (160.5 gen/s) is slower than WebGPU (170.3 gen/s) despite being native, suggesting framework dispatch overhead exceeds browser sandbox overhead on this workload.

### 4.2 Sequential Workload: Financial Simulation

**Table 2: Financial Simulation Benchmark (POP=10,000, 246 params, 1,500 timesteps)**

| System | Throughput (gen/s) | vs NumPy | vs PyTorch MPS | vs PyTorch CUDA |
|---|---|---|---|---|
| Python/NumPy (CPU) | 0.056 | — | — | — |
| JAX jit + lax.scan (CPU) | 0.157 | 2.8× | — | — |
| PyTorch (MPS, M2 Pro) | 0.29 | 5.2× | — | — |
| PyTorch (CUDA, Tesla T4) | 0.49 | 8.8× | 1.7× | — |
| JAX jit + lax.scan + vmap (CUDA, T4) | 6.43 | 115× | 22× | — |
| **WebGPU (Chrome, M2 Pro)** | **46.2 ± 1.0** | **825×** | **159×** | **94×** |

JAX with `lax.scan` + `vmap` on a T4 GPU achieves 6.43 gen/s — **13× over PyTorch CUDA**, confirming that XLA loop fusion provides substantial benefit. However, **WebGPU still achieves 7.2× over JAX GPU** on this workload. JAX's fusion compiles the 1,500-step loop into a single XLA trace, but the resulting kernel still incurs per-step overhead within the compiled function (register spills, memory traffic for large state vectors). WebGPU's hand-fused WGSL shader avoids this by maintaining all state in thread-local variables.

WebGPU executes the **entire 1,500-step simulation as a single compute shader dispatch**, eliminating ~22,500 kernel launches. The advantage hierarchy is clear: hand-fused shaders (46.2) > XLA loop fusion (6.43) > per-step dispatch (0.49) > CPU fusion (0.157) > CPU eager (0.056).

### 4.3 Sequential Workload: Acrobot-v1

**Table 3: Acrobot-v1 Benchmark (POP=4,096, 163 params, 500 timesteps, RK4)**

| System | Throughput (gen/s) | vs PyTorch MPS | vs PyTorch CUDA | N |
|---|---|---|---|---|
| PyTorch (CUDA, Tesla T4) | 0.61 ± 0.03 | — | — | 10 |
| PyTorch (MPS, M2 Pro) | 2.52 ± 0.04 | — | 4.1× | 10 |
| WebGPU unfused (M2 Pro) | 62.3 ± 0.1 | 24.7× | 102× | 10 |
| JAX lax.scan + vmap (CUDA, T4) | 105.1 ± 1.0 | 41.7× | 172× | 10 |
| **WebGPU fused (M2 Pro)** | **135.9 ± 4.0** | **54×** | **223×** | 10 |

**JAX GPU nearly closes the gap at L=500.** JAX with `lax.scan` on T4 achieves 105.1 gen/s on Acrobot — only 1.29× slower than WebGPU's fused shader. At L=500 timesteps, XLA's loop fusion is nearly as effective as hand-fused WGSL. Combined with the financial result (7.2× gap at L=1,500), this reveals that **the fusion advantage scales with episode length L**: hand-fused shaders maintain constant overhead per generation regardless of L, while framework-compiled kernels incur overhead that grows with L.

**Ablation: decomposing the 223× advantage over PyTorch CUDA.** Unfused WebGPU (500 separate dispatches) achieves 62.3 gen/s — already 102× over PyTorch CUDA. Fusion provides an additional 2.18×, for a total 223×. JAX's `lax.scan` fusion (105.1) falls between unfused (62.3) and fused (135.9) WebGPU, confirming that XLA achieves partial but not complete fusion.

### 4.4 torch.compile Scaling Failure

**Table 4: torch.compile scaling with sequential timesteps (POP=10,000)**

| Timesteps | Eager | Compiled | Compile time | Speedup |
|---|---|---|---|---|
| 10 | 0.067s | 0.002s | 1.8s | 30.7× |
| 100 | 0.024s | 0.015s | 4.7s | 1.5× |
| 500 | 0.158s | 0.081s | 25.5s | 2.0× |
| 1,000 | 0.227s | **RecursionError** | — | — |
| 5,000 | 3.4s | **OOM killed** | >30 min | — |

`torch.compile` (Inductor backend) fails at L=1,000 (RecursionError) and L=5,000 (OOM). The Inductor backend unrolls the full loop into a single computation graph, scaling super-linearly in memory.

### 4.5 CMA-ES Comparison

**Table 5: WebGPU vs CMA-ES (30s wall-clock budget, N=30)**

| System | DIM | Best Fitness (mean ± std) | Gen/s | Total Gens | N |
|---|---|---|---|---|---|
| **WebGPU** | **100** | **0.0 ± 0.0** | **16,257** | **487,750** | 30 |
| CMA-ES (default pop) | 100 | 272 ± 39 | 524 | 1,609 | 30 |
| **WebGPU** | **500** | **0.0 ± 0.0** | **16,109** | **483,298** | 30 |
| CMA-ES (default pop) | 500 | 3,804 ± 1,181 | 88 | 2,636 | 30 |
| **WebGPU** | **2,000** | **~15,039** | **11,335** | **~340,000** | 30 |
| CMA-ES (default pop) | 2,000 | 33,843 ± 198 | 12.4 | 374 | 30 |
| CMA-ES (POP=4,096) | 2,000 | 35,159 ± 122 | 0.22 | 7 | 30 |

Fitness values are Rastrigin values (lower = better, optimum = 0). WebGPU achieves the global optimum at DIM=100/500. At DIM=2,000, WebGPU achieves ~2.2× better fitness than CMA-ES.

**Population-matched control.** CMA-ES with POP=4,096 manages only 7 generations in 30s (0.22 gen/s), achieving the worst fitness (35,159). CMA-ES's O(n²×pop) covariance update makes large-population operation non-viable.

### 4.6 Numerical Precision (f32 vs f64)

WebGPU supports only f32. Maximum absolute error in NN outputs: 1.97×10⁻⁷. Over 5,000 cumulative multiplications, maximum relative error: 9.17×10⁻⁶. Fitness rankings perfectly preserved: Spearman ρ=1.000 across 10,000 genomes.

### 4.7 Scaling Characterization

**Table 6: Population Scaling (Rastrigin, DIM=2,000)**

| Population | Throughput (gen/s) | VRAM (MB) |
|---|---|---|
| 512 | 12,948 | 7.8 |
| 1,024 | 12,791 | 15.6 |
| 2,048 | 12,842 | 31.3 |
| 4,096 | 12,685 | 62.5 |
| 8,192 | 12,714 | 125.1 |
| 16,384 | 12,741 | 250.1 |
| 32,768 | 12,702 | 500.3 |

Throughput remains flat from POP=512 to 32,768 (<2.1% variation). Raw throughput (~12.8K gen/s) exceeds Table 1's 170 gen/s because Table 1 includes `mapAsync` readback overhead. The native wgpu baseline (Section 4.1) confirms 48% of this gap is browser overhead; the remainder is the synchronous readback pattern.

**Table 7: Multi-Benchmark Suite (POP=4,096, DIM=2,000)**

| Function | Throughput (gen/s) |
|---|---|
| Sphere | 11,988 |
| Rastrigin | 11,335 |
| Ackley | 11,678 |
| Schwefel | 12,221 |
| Griewank | 11,529 |

All five benchmarks within 8%, confirming compute-bound rather than fitness-function-specific performance.

**Table 8: Genome Size Scaling (POP=4,096, Rastrigin)**

| Dimensions | Throughput (gen/s) | VRAM (MB) |
|---|---|---|
| 100 | 11,804 | 3.2 |
| 500 | 11,599 | 15.7 |
| 1,000 | 11,722 | 31.3 |
| 2,000 | 11,775 | 62.5 |
| 4,000 | 11,692 | 125.0 |

Throughput invariant across 40× genome size range (<2% variation).

### 4.8 GPU Utilization and Thermal Profile

**Table 9: GPU Utilization (Rastrigin, POP=4,096)**

| Phase | GPU Utilization | Temperature | Throughput |
|---|---|---|---|
| Idle | 0.8% ± 0.9% | 30.7°C | — |
| Evolution | **79.7%** ± 1.5% | 30.7°C | 175.5 gen/s |

GPU reaches 79.7% utilization with no thermal throttling over 60s continuous operation (CV<5%).

### 4.9 Multi-Tab GPU Contention

**Table 10: Multi-Tab GPU Sharing (Rastrigin, POP=4,096 per tab)**

| Tabs | Per-tab (gen/s) | Total (gen/s) | Efficiency |
|---|---|---|---|
| 1 | 11,607 | 11,607 | 100% |
| 2 | 11,296 | 22,593 | 97% |
| 4 | 9,727 | 38,909 | 84% |
| 8 | 9,314 | 74,512 | 80% |

At 8 tabs, total throughput reaches 6.4× with 80% per-tab efficiency.

### 4.10 MountainCar-v0

**Table 11: MountainCar-v0 Benchmark (POP=4,096, 51 params, 200 timesteps)**

| System | Throughput (gen/s) | vs PyTorch MPS | N |
|---|---|---|---|
| PyTorch (MPS, M2 Pro) | 18.7 ± 0.6 | — | 10 |
| **WebGPU (Chrome, M2 Pro)** | **1,258.8 ± 11.4** | **67×** | 10 |

MountainCar-v0 (standard Gym benchmark, 200 timesteps, 2D state) adds a third standard RL environment alongside CartPole and Acrobot. The 67× speedup at L=200 is consistent with the pattern: shorter episodes yield smaller but still substantial fusion advantages.

### 4.11 CartPole-v1 Validation

CartPole-v1 solved in **76 ± 46 ms** (N=30, 29/30 = 97% solve rate). **Caveat:** This validates random search at scale — with 4,096 parallel rollouts, the system solves on generation 0 or 1. EvoJAX [7] reports solve times on the order of seconds on V100/TPU.

---

## 5. Limitations and Future Work

- **NVIDIA tested on Tesla T4 only.** The fusion advantage is even larger on T4 (94–223×) than MPS. Testing on A100/H100 would characterize whether the advantage persists at the high end.
- **JAX GPU tested on T4 only.** JAX with `lax.scan`+`vmap` on T4 achieves 6.43 gen/s (financial) and 105.1 gen/s (Acrobot). On TPU or A100, JAX's XLA compiler may further close the gap with WebGPU.
- **Browser overhead quantified but not decomposed.** The 1.92× gap (Section 4.1) includes process isolation, IPC, Dawn translation, and GPU process scheduling.
- **Two hardware platforms.** Testing on AMD and Intel Arc GPUs would strengthen generalizability.
- **Numerical precision.** While f32 suffices for evolutionary dynamics, domains requiring high-precision accumulation may be affected.

---

## 6. Reproducibility

- **Browser:** Chrome 120+ (WebGPU enabled by default)
- **OS:** macOS 14+ (Sonoma), Linux (Google Colab)
- **GPU backends:** Apple Metal (M2 Pro), NVIDIA CUDA (Tesla T4)
- **Benchmarks:** `npm run bench` (Puppeteer-based)
- **Python baselines:** `pip install -r benchmarks/requirements.txt`
- **Code:** https://github.com/abgnydn/webgpu-kernel-fusion

---

## 7. Conclusion

Hand-fused compute shaders outperform all tested alternatives on sequential fitness functions, including JAX's XLA loop fusion — the strongest theoretical competitor. On the 1,500-timestep financial simulation, WebGPU achieves 7.2× over JAX GPU with `lax.scan`+`vmap` (46.2 vs 6.43 gen/s) and 94× over PyTorch CUDA. The advantage scales with episode length: at L=500 (Acrobot), JAX GPU nearly closes the gap (1.29×), while at L=1,500 the gap is substantial (7.2×). This reveals a clear hierarchy: hand-fused shaders > XLA loop fusion > per-step framework dispatch.

On parallel workloads (Rastrigin), JAX GPU dominates at 1,164 gen/s — 6.8× faster than WebGPU — confirming that the advantage is specific to sequential dependency chains, not a general WebGPU superiority claim. A native Metal baseline quantifies Chrome's browser overhead at 1.92×. WebGPU makes kernel fusion accessible with zero installation across Apple Metal, NVIDIA Vulkan, AMD Vulkan, and Intel DirectX backends.

---

## References

[1] W.B. Langdon, "Large scale bioinformatics data mining with parallel genetic programming on graphics processing units," in *Parallel and Distributed Computational Intelligence*, SCI 269, Springer, 2010.

[2] F.A. Fortin, F.M. De Rainville, M.A. Gardner, M. Parizeau, and C. Gagné, "DEAP: Evolutionary algorithms made easy," *JMLR*, vol. 13, 2012.

[3] W3C, "WebGPU Specification," 2023. [Online]. Available: https://www.w3.org/TR/webgpu/

[4] Y. Tang, Y. Tian, and D. Ha, "EvoJAX: Hardware-accelerated neuroevolution," in *Proc. GECCO Companion*, 2022. Also *arXiv:2202.05008*.

[5] L.A. Rastrigin, *Systems of Extremal Control*. Moscow: Nauka, 1974.

[6] S. Sengupta, N. Wu, M. Varvello, K. Jana, and S. Chen, "From WebGL to WebGPU: A reality check of browser-based GPU acceleration," in *Proc. ACM Internet Measurement Conference*, 2025.

[7] J.J. Merelo-Guervos and P. Garcia-Sanchez, "Modeling browser-based distributed evolutionary computation systems," *arXiv:1503.06424*, 2015.

[8] J. Duda and W. Dlubacz, "GPU acceleration for the web browser based evolutionary computing system," in *Proc. ICSTCC*, 2013.

[9] J. Duda and W. Dlubacz, "Distributed evolutionary computing system based on web browsers with JavaScript," in *PARA 2012*, LNCS 7782, Springer, 2013.

[10] T.V. Luong, N. Melab, and E.-G. Talbi, "GPU-based island model for evolutionary algorithms," in *Proc. GECCO*, 2010.

[11] Z. Wu, L. Wang, K. Sun, Z. Li, and R. Cheng, "Enabling population-level parallelism in tree-based genetic programming for GPU acceleration," *arXiv:2501.17168*, 2025.

[12] L. Trujillo, J.M. Munoz Contreras, D.E. Hernandez, M. Castelli, and J.J. Tapia, "GSGP-CUDA: A CUDA framework for geometric semantic genetic programming," *SoftwareX*, vol. 18, 2022.

[13] M. Hidaka and T. Harada, "WgPy: GPU-accelerated NumPy-like array library for web browsers," *arXiv:2503.00279*, 2025.

[14] N. Hansen and A. Ostermeier, "Completely derandomized self-adaptation in evolution strategies," *Evolutionary Computation*, vol. 9, no. 2, pp. 159–195, 2001.
