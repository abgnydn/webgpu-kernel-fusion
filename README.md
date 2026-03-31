# Single-Kernel Fusion for Sequential Fitness Evaluation via WebGPU Compute Shaders

Fusing sequential fitness evaluations into single GPU compute shader dispatches eliminates per-step kernel launch overhead. We prove this across **4 GPU APIs on 2 hardware platforms** — the fusion advantage is **GPU-API-agnostic**.

## Key Results

**Same hardware: Tesla T4 (Acrobot-v1, 500 steps, RK4)**

| System | gen/s | vs PyTorch |
|---|---|---|
| PyTorch CUDA per-step | 0.61 | 1x |
| Triton fused | 16.4 | **27x** |
| JAX lax.scan+vmap | 105.1 | **172x** |
| **Hand-fused CUDA kernel** | **439** | **720x** |

**Same hardware: Apple M2 Pro (Acrobot-v1, 500 steps, RK4)**

| System | gen/s | vs PyTorch |
|---|---|---|
| PyTorch MPS per-step | 2.52 | 1x |
| wgpu-native fused (Metal) | 30.5 | **12x** |
| WebGPU unfused (Chrome) | 62.3 | **25x** |
| **WebGPU fused (Chrome)** | **135.9** | **54x** |

**Same hardware: M2 Pro (Financial sim, 1,500 steps)**

| System | gen/s | vs PyTorch |
|---|---|---|
| PyTorch MPS | 0.29 | 1x |
| **WebGPU fused (Chrome)** | **46.2** | **159x** |

The advantage is specific to **sequential workloads**. On parallel workloads (Rastrigin), JAX GPU dominates (6.8x over WebGPU).

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
