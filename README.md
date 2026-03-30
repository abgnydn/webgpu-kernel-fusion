# Single-Kernel Fusion for Sequential Fitness Evaluation via WebGPU Compute Shaders

Fusing sequential fitness evaluations into single GPU compute shader dispatches eliminates per-step kernel launch overhead. On the **same M2 Pro GPU**, a WebGPU shader achieves **159x over PyTorch MPS** on a 1,500-timestep financial simulation. An unfused ablation isolates **2.18x from fusion alone**. On the **same Tesla T4**, JAX with `lax.scan` achieves **13x over PyTorch CUDA** via XLA fusion. A native Metal baseline quantifies Chrome's browser overhead at **1.92x (48%)**.

## Key Results

**Same-hardware comparisons (no cross-hardware confounding):**

| Comparison | Workload | Speedup | Hardware |
|---|---|---|---|
| WebGPU vs PyTorch MPS | Financial (1,500 steps) | **159x** | M2 Pro |
| WebGPU vs PyTorch MPS | Acrobot (500 steps) | **54x** | M2 Pro |
| WebGPU fused vs unfused | Acrobot (500 steps) | **2.18x** | M2 Pro |
| JAX GPU vs PyTorch CUDA | Financial (1,500 steps) | **13x** | Tesla T4 |
| wgpu-native vs WebGPU Chrome | Rastrigin (parallel) | **1.92x** | M2 Pro |

**Cross-hardware (includes unified memory advantage — interpret with caveat):**

| Workload | WebGPU (M2 Pro) | PyTorch CUDA (T4) | JAX GPU (T4) |
|---|---|---|---|
| Financial (1,500 steps) | **46.2 gen/s** | 0.49 gen/s | 6.43 gen/s |
| Acrobot (500 steps, RK4) | **135.9 gen/s** | 0.61 gen/s | 105.1 gen/s |
| Rastrigin (parallel) | 170.3 gen/s | 311.1 gen/s | **1,163.9 gen/s** |

The advantage is specific to **sequential workloads**. On parallel workloads, JAX GPU dominates (6.8x over WebGPU).

## Links

- **Run the benchmarks yourself:** [gpubench.dev](https://gpubench.dev)
- **Why this matters (plain language):** [gpubench.dev/why](https://gpubench.dev/why)
- **Paper (Markdown):** [PAPER.md](PAPER.md)
- **Paper (LaTeX):** [paper.tex](paper.tex)
- **arXiv:** coming soon

## Reproduce Every Result

### Prerequisites

- Chrome 120+ (WebGPU enabled by default)
- Node.js 18+
- Python 3.10+ (`pip install -r benchmarks/requirements.txt`)

### Run Benchmarks

| Table | Command | What it measures |
|---|---|---|
| Table 1 | `npm run bench:rastrigin` | Rastrigin throughput + native Metal baseline |
| Table 2 | `npm run bench` | Financial simulation (1,500 steps) |
| Table 3 | `npm run bench:acrobot` | Acrobot-v1 (500 steps, RK4) |
| Table 4 | `python3 benchmarks/pytorch_variance.py` | torch.compile scaling failure |
| Table 5 | `npm run bench:cmaes` | CMA-ES comparison (N=30) |
| Tables 6–8 | `npm run bench:comprehensive` | Population, benchmark suite, genome scaling |

### Python Baselines

```bash
python3 benchmarks/numpy_variance.py       # NumPy (N=10)
python3 benchmarks/pytorch_variance.py     # PyTorch MPS (N=10)
python3 benchmarks/jax_variance.py         # JAX CPU (N=10)
python3 benchmarks/cmaes_benchmark.py      # CMA-ES (N=30)
```

## Repository Structure

```
PAPER.md                    # The paper (Markdown)
paper.tex                   # The paper (LaTeX)
figures/                    # Figures from real benchmark data
src/
  nn_core.wgsl              # Shared RNN forward pass (~246 params)
  swarm_shader.wgsl         # Evaluate + evolve entry points
benchmarks/
  bench.js                  # Puppeteer benchmark runner
  comprehensive_benchmarks.js
  cmaes_benchmark.py        # CMA-ES (Python)
  pytorch_gpu_baseline.py   # PyTorch MPS/CUDA
  jax_baseline.py           # JAX CPU/GPU
  *_variance.py             # N=10 variance scripts
  requirements.txt
tests/
  paper_arithmetic.test.js  # Verifies every derived number in the paper
```

## Verify Paper Arithmetic

Every ratio, percentage, and comparison in the paper is tested against the raw table data:

```bash
node tests/paper_arithmetic.test.js
# 55 checks, 0 failures
```

This catches stale numbers, copy-paste errors, and miscalculations. If you change any table value, run this test to verify all derived claims still hold.

## Hardware

- **Primary:** Apple M2 Pro (19-core GPU, 16 GB unified), Chrome 123, macOS 14
- **NVIDIA:** Tesla T4 (16 GB, Google Colab), CUDA 12.8

## Citation

```bibtex
@article{gunaydin2026kernelfusion,
  title={Single-Kernel Fusion for Sequential Fitness Evaluation via WebGPU Compute Shaders},
  author={Gunaydin, Ahmet Baris},
  year={2026}
}
```

## License

MIT
