#!/usr/bin/env python3
"""
PyTorch GPU Baseline Benchmark for WebGPU Neuroevolution Comparison.

Runs the SAME workloads as WebGPU on the SAME GPU (Apple MPS / CUDA),
providing a fair GPU-vs-GPU comparison.

Benchmarks:
  1. Rastrigin optimization (POP=4096, DIM=2000) — matches WebGPU benchmark
  2. Financial simulation (POP=10000, 246 params, 5000 timesteps) — matches python_baseline.py
"""

import torch
import torch.nn.functional as F
import time
import math
import numpy as np

# ─── Device Setup ─────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
    GPU_NAME = 'Apple MPS (Metal)'
elif torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    GPU_NAME = torch.cuda.get_device_name(0)
else:
    DEVICE = torch.device('cpu')
    GPU_NAME = 'CPU (no GPU found)'

print("=" * 64)
print("  PYTORCH GPU BASELINE BENCHMARK")
print(f"  Device: {GPU_NAME}")
print(f"  PyTorch: {torch.__version__}")
print("=" * 64)


# ═══════════════════════════════════════════════════════════════════════
#  BENCHMARK 1: RASTRIGIN (matches WebGPU p2p_demo benchmark)
# ═══════════════════════════════════════════════════════════════════════

def bench_rastrigin():
    print("\n" + "─" * 64)
    print("  RASTRIGIN BENCHMARK")
    print(f"  POP=4096  DIM=2000  (same as WebGPU)")
    print("─" * 64)

    POP = 4096
    DIM = 2000
    PI = math.pi
    NUM_GENS = 100
    WARMUP = 10
    SIGMA = 0.3
    TOURNAMENT_K = 5

    # Initialize population in [-5.12, 5.12]
    pop = (torch.rand(POP, DIM, device=DEVICE) * 10.24 - 5.12).float()

    def rastrigin_fitness(x):
        """Rastrigin: f(x) = 10n + sum(x_i^2 - 10*cos(2*pi*x_i))"""
        return -(10.0 * DIM + (x ** 2 - 10.0 * torch.cos(2.0 * PI * x)).sum(dim=1))

    def evolve(pop, fitness):
        """Tournament selection + Gaussian mutation."""
        # Tournament selection
        tournament_idx = torch.randint(0, POP, (POP, TOURNAMENT_K), device=DEVICE)
        tournament_fit = fitness[tournament_idx]
        winners = tournament_idx[torch.arange(POP, device=DEVICE), tournament_fit.argmax(dim=1)]
        new_pop = pop[winners].clone()

        # Mutation
        new_pop += torch.randn_like(new_pop) * SIGMA

        # Elitism: keep best
        best_idx = fitness.argmax()
        new_pop[0] = pop[best_idx]
        return new_pop

    # ── Try torch.compile ──
    compiled_fitness = None
    compiled_evolve = None
    try:
        compiled_fitness = torch.compile(rastrigin_fitness)
        compiled_evolve = torch.compile(evolve)
        print("  torch.compile: available ✓")
    except Exception as e:
        print(f"  torch.compile: not available ({e})")

    results = {}
    for mode_name, fit_fn, evo_fn in [
        ("eager", rastrigin_fitness, evolve),
        ("torch.compile", compiled_fitness, compiled_evolve),
    ]:
        if fit_fn is None:
            continue

        # Re-init population
        pop = (torch.rand(POP, DIM, device=DEVICE) * 10.24 - 5.12).float()

        # Warmup
        for _ in range(WARMUP):
            fit = fit_fn(pop)
            pop = evo_fn(pop, fit)

        if DEVICE.type == 'mps':
            torch.mps.synchronize()
        elif DEVICE.type == 'cuda':
            torch.cuda.synchronize()

        # Timed runs
        t0 = time.perf_counter()
        for gen in range(NUM_GENS):
            fit = fit_fn(pop)
            pop = evo_fn(pop, fit)

        if DEVICE.type == 'mps':
            torch.mps.synchronize()
        elif DEVICE.type == 'cuda':
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - t0
        gen_per_sec = NUM_GENS / elapsed
        best_fit = fit.max().item()

        print(f"\n  [{mode_name}]")
        print(f"  Generations: {NUM_GENS}")
        print(f"  Time:        {elapsed:.2f}s")
        print(f"  Gen/sec:     {gen_per_sec:.1f}")
        print(f"  Best fit:    {best_fit:.1f}")
        results[mode_name] = gen_per_sec

    return results


# ═══════════════════════════════════════════════════════════════════════
#  BENCHMARK 2: FINANCIAL NN (matches python_baseline.py)
# ═══════════════════════════════════════════════════════════════════════

def bench_financial():
    print("\n" + "─" * 64)
    print("  FINANCIAL SIMULATION BENCHMARK")
    print(f"  POP=10000  PARAMS=246  TIMESTEPS=5000  ASSETS=16")
    print("─" * 64)

    POP = 10000
    N_PARAMS = 246
    L = 5000
    N_A = 16
    METABOLIC_RATE = 0.00001
    NUM_GENS = 5
    WARMUP = 1
    ISLAND_SIZE = 1000
    N_ISLANDS = POP // ISLAND_SIZE

    # Generate market data (same structure as python_baseline.py)
    np.random.seed(42)
    signals_np = np.random.randn(L, 36).astype(np.float32) * 0.1
    for t in range(L):
        for a in range(N_A):
            signals_np[t, 4 + a] = max(0, signals_np[t, 4 + a] + 0.3)
            signals_np[t, 20 + a] = np.random.normal(0, 0.01)
    signals = torch.tensor(signals_np, device=DEVICE)

    # ── NN Forward Pass (V44: 8→16→6) ──
    def nn_forward_batch(X, genomes):
        """Batched forward pass on GPU: (POP, 8) → (POP, 6)"""
        W1 = genomes[:, :128].reshape(-1, 8, 16)
        B1 = genomes[:, 128:144]
        H = torch.bmm(X.unsqueeze(1), W1).squeeze(1) + B1
        H = torch.clamp(2.0 * H, -20.0, 20.0)
        ext = torch.exp(H)
        H = (ext - 1.0) / (ext + 1.0)  # tanh

        W2 = genomes[:, 144:240].reshape(-1, 16, 6)
        B2 = genomes[:, 240:246]
        Out = torch.bmm(H.unsqueeze(1), W2).squeeze(1) + B2
        Out = torch.clamp(-Out, -20.0, 20.0)
        Out = 1.0 / (1.0 + torch.exp(Out))  # sigmoid
        return Out

    # ── Evaluate ──
    def evaluate(genomes):
        pv = torch.ones(POP, device=DEVICE)
        pp = torch.ones(POP, device=DEVICE)
        mem0 = torch.zeros(POP, device=DEVICE)
        mem1 = torch.zeros(POP, device=DEVICE)
        pw = torch.zeros(POP, N_A, device=DEVICE)
        rets_cum = torch.ones(POP, device=DEVICE)
        mdd = torch.zeros(POP, device=DEVICE)
        pk = torch.ones(POP, device=DEVICE)

        for h in range(L):
            sig = signals[h]

            dd = torch.where(pp > 0, torch.clamp((pp - pv) / pp, 0, 1), torch.zeros_like(pp))
            fir_pnl = torch.clamp((pv / torch.clamp(pp, min=1e-8) - 1.0) * 10.0, 0.0, 1.0)

            X = torch.stack([
                sig[0].expand(POP),
                torch.clamp(dd * 5.0, 0, 1),
                torch.zeros(POP, device=DEVICE),
                fir_pnl,
                sig[1].expand(POP),
                sig[2].expand(POP),
                mem0, mem1
            ], dim=1)

            emotions = nn_forward_batch(X, genomes)
            caution = emotions[:, 0]
            lev_base = 1.0 + emotions[:, 1] * 2.0
            maxw_base = (1.0 - caution * 0.8) * (0.2 + emotions[:, 3] * 0.8)
            mem0 = emotions[:, 4]
            mem1 = emotions[:, 5]

            # Allocate
            scores = sig[4:4 + N_A].unsqueeze(0).expand(POP, -1)
            w = torch.clamp(scores, min=0)
            tw = w.sum(dim=1, keepdim=True).clamp(min=1e-8)
            w = w / tw
            w = torch.min(w, maxw_base.unsqueeze(1))
            tw2 = w.sum(dim=1, keepdim=True).clamp(min=1e-8)
            w = w / tw2

            # Hysteresis
            w = torch.where(torch.abs(w - pw) < 0.05, pw, w)

            # Returns
            asset_rets = sig[20:20 + N_A].unsqueeze(0).expand(POP, -1)
            rr = (w * asset_rets).sum(dim=1)
            dr = rr * lev_base

            # Friction
            diff = torch.abs(w - pw)
            friction = torch.where(w < pw, 0.0002, 0.0004)
            dr -= (diff * lev_base.unsqueeze(1) * friction).sum(dim=1)

            dr -= METABOLIC_RATE
            pv *= (1.0 + dr)
            pv = torch.clamp(pv, min=1e-8)
            pv = torch.where(pv <= 0.05, torch.full_like(pv, 0.0001), pv)
            pp = torch.max(pp, pv)
            pw = w.clone()

            rets_cum *= (1.0 + dr)
            pk = torch.max(pk, rets_cum)
            current_dd = (pk - rets_cum) / torch.clamp(pk, min=1e-8)
            mdd = torch.max(mdd, current_dd)

        return torch.where(mdd > 0.50, torch.full_like(rets_cum, -9999.0), rets_cum)

    # ── Evolve ──
    def evolve(genomes, fitness):
        new_genomes = torch.empty_like(genomes)
        for island in range(N_ISLANDS):
            s, e = island * ISLAND_SIZE, (island + 1) * ISLAND_SIZE
            f = fitness[s:e]
            g = genomes[s:e]

            elite_idx = f.argmax()
            new_genomes[s] = g[elite_idx]

            # Tournament selection
            t_idx = torch.randint(0, ISLAND_SIZE, (ISLAND_SIZE - 1, 5), device=DEVICE)
            t_fit = f[t_idx]
            p1 = t_idx[torch.arange(ISLAND_SIZE - 1, device=DEVICE), t_fit.argmax(dim=1)]
            t_idx2 = torch.randint(0, ISLAND_SIZE, (ISLAND_SIZE - 1, 5), device=DEVICE)
            t_fit2 = f[t_idx2]
            p2 = t_idx2[torch.arange(ISLAND_SIZE - 1, device=DEVICE), t_fit2.argmax(dim=1)]

            # Crossover
            mask = torch.rand(ISLAND_SIZE - 1, N_PARAMS, device=DEVICE) < 0.5
            children = torch.where(mask, g[p1], g[p2])

            # Mutation
            mut_r = torch.rand(ISLAND_SIZE - 1, N_PARAMS, device=DEVICE)
            children += torch.where(mut_r < 0.05, torch.randn_like(children) * 0.6,
                                    torch.zeros_like(children))
            children += torch.where((mut_r >= 0.05) & (mut_r < 0.20),
                                    torch.randn_like(children) * 0.03,
                                    torch.zeros_like(children))
            new_genomes[s + 1:e] = children

        return new_genomes

    # ── Try torch.compile ──
    compiled_evaluate = None
    try:
        compiled_evaluate = torch.compile(evaluate)
        print("  torch.compile: available ✓")
    except Exception as e:
        print(f"  torch.compile: not available ({e})")

    results = {}
    for mode_name, eval_fn in [
        ("eager", evaluate),
        ("torch.compile", compiled_evaluate),
    ]:
        if eval_fn is None:
            continue

        # Initialize
        genomes = (torch.randn(POP, N_PARAMS, device=DEVICE) * 1.5).float()

        # Warmup
        for _ in range(WARMUP):
            fitness = eval_fn(genomes)
            genomes = evolve(genomes, fitness)

        if DEVICE.type == 'mps':
            torch.mps.synchronize()
        elif DEVICE.type == 'cuda':
            torch.cuda.synchronize()

        # Timed runs
        times = []
        for gen in range(NUM_GENS):
            t0 = time.perf_counter()
            fitness = eval_fn(genomes)

            if DEVICE.type == 'mps':
                torch.mps.synchronize()
            elif DEVICE.type == 'cuda':
                torch.cuda.synchronize()

            t_eval = time.perf_counter() - t0
            t1 = time.perf_counter()
            genomes = evolve(genomes, fitness)

            if DEVICE.type == 'mps':
                torch.mps.synchronize()
            elif DEVICE.type == 'cuda':
                torch.cuda.synchronize()

            t_evolve = time.perf_counter() - t1
            total = time.perf_counter() - t0
            times.append(total)

            best = fitness[fitness > -9000].max().item()
            print(f"  [{mode_name}] Gen {gen} | Best: {best:.4f} | Eval: {t_eval:.1f}s | Evolve: {t_evolve:.1f}s | Total: {total:.1f}s")

        avg_time = sum(times) / len(times)
        gen_per_sec = 1.0 / avg_time
        print(f"\n  [{mode_name}] Avg time/gen: {avg_time:.2f}s")
        print(f"  [{mode_name}] Gen/sec:      {gen_per_sec:.4f}")
        results[mode_name] = gen_per_sec

    return results


# ═══════════════════════════════════════════════════════════════════════
#  RUN ALL
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    rastrigin_results = bench_rastrigin()
    financial_results = bench_financial()

    print("\n" + "=" * 64)
    print("  SUMMARY — PyTorch GPU Baseline")
    print("=" * 64)
    print(f"  Device: {GPU_NAME}")
    print()
    print("  ┌──────────────┬────────────┬──────────────┬────────────────┬──────────┐")
    print("  │ Benchmark    │ NumPy CPU  │ PyTorch eager│ PyTorch compile│ WebGPU   │")
    print("  ├──────────────┼────────────┼──────────────┼────────────────┼──────────┤")
    r_eager = rastrigin_results.get("eager", 0)
    r_compile = rastrigin_results.get("torch.compile", 0)
    f_eager = financial_results.get("eager", 0)
    f_compile = financial_results.get("torch.compile", 0)
    print(f"  │ Rastrigin    │ 3.68 g/s   │ {r_eager:>8.1f} g/s │ {r_compile:>10.1f} g/s │ 179.3 g/s│")
    print(f"  │ Financial    │ 0.056 g/s  │ {f_eager:>8.4f} g/s│ {f_compile:>10.4f} g/s│ ~24 g/s  │")
    print("  └──────────────┴────────────┴──────────────┴────────────────┴──────────┘")
    print("=" * 64)
