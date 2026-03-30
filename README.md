# Single-Kernel Fusion for Sequential Fitness Evaluation via WebGPU Compute Shaders

Fusing sequential fitness evaluations into single GPU compute shader dispatches achieves **94–223x over PyTorch CUDA** and **7.2x over JAX GPU** on sequential workloads. A native Metal baseline quantifies Chrome's browser overhead at 48% — yet WebGPU in a browser still **outperforms PyTorch MPS running natively**.

## Key Results

| Workload | WebGPU (Chrome, M2 Pro) | PyTorch CUDA (T4) | JAX GPU (T4) | Advantage |
|---|---|---|---|---|
| Financial sim (1,500 steps) | **46.2 gen/s** | 0.49 gen/s | 6.43 gen/s | 94x / 7.2x |
| Acrobot-v1 (500 steps, RK4) | **135.9 gen/s** | 0.61 gen/s | 105.1 gen/s | 223x / 1.3x |
| Rastrigin (parallel) | 170.3 gen/s | 311.1 gen/s | 1,163.9 gen/s | JAX wins (expected) |

The advantage is specific to **sequential workloads** and grows on discrete NVIDIA hardware where per-step dispatch overhead is higher.

## Links

- **Run the benchmarks yourself:** [swarm-bench.vercel.app](https://swarm-bench.vercel.app)
- **Why this matters (plain language):** [swarm-bench.vercel.app/why](https://swarm-bench.vercel.app/why)
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
```

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
