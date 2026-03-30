#!/usr/bin/env python3
"""CMA-ES vs WebGPU comparison on Rastrigin at DIM=100, 500, 2000."""

import numpy as np
import time
import math

try:
    import cma
except ImportError:
    import subprocess
    subprocess.check_call(['pip3', 'install', 'cma', '--break-system-packages', '-q'])
    import cma

N = 30
BUDGET_S = 30

def rastrigin(x):
    n = len(x)
    return 10 * n + sum(xi**2 - 10 * math.cos(2 * math.pi * xi) for xi in x)

def run_cmaes(dim, pop_size=None, label="default"):
    print(f"\n{'─'*60}")
    print(f"  CMA-ES {label} DIM={dim} — N={N} runs, {BUDGET_S}s budget each")
    print(f"{'─'*60}")

    results = []
    for i in range(N):
        x0 = np.random.uniform(-5.12, 5.12, dim)
        sigma0 = 2.0

        opts = {
            'verbose': -9,
            'seed': i + 1,
            'bounds': [-5.12, 5.12],
            'maxiter': 10**9,
            'tolfun': -1,
            'tolx': -1,
        }
        if pop_size is not None:
            opts['popsize'] = pop_size

        es = cma.CMAEvolutionStrategy(x0, sigma0, opts)

        t0 = time.time()
        gens = 0
        best_f = float('inf')

        while time.time() - t0 < BUDGET_S and not es.stop():
            solutions = es.ask()
            fitnesses = [rastrigin(x) for x in solutions]
            es.tell(solutions, fitnesses)
            gens += 1
            best_f = min(best_f, min(fitnesses))

        elapsed = time.time() - t0
        gps = gens / elapsed if elapsed > 0 else 0

        print(f"  Run {i+1}/{N}: fitness={best_f:.4f}, {gps:.2f} gen/s, {gens} gens")
        results.append({'fitness': best_f, 'gps': gps, 'gens': gens})

    fitnesses = [r['fitness'] for r in results]
    gps_arr = [r['gps'] for r in results]
    gens_arr = [r['gens'] for r in results]

    print(f"\n  RESULTS CMA-ES {label} DIM={dim}:")
    print(f"  Fitness: {np.mean(fitnesses):.4f} +/- {np.std(fitnesses, ddof=1):.4f}")
    print(f"  Gen/s:   {np.mean(gps_arr):.2f} +/- {np.std(gps_arr, ddof=1):.2f}")
    print(f"  Gens:    {np.mean(gens_arr):.0f} +/- {np.std(gens_arr, ddof=1):.0f}")
    print(f"  N:       {len(results)}")

    return {
        'dim': dim, 'label': label,
        'fitness_mean': np.mean(fitnesses), 'fitness_std': np.std(fitnesses, ddof=1),
        'gps_mean': np.mean(gps_arr), 'gps_std': np.std(gps_arr, ddof=1),
        'gens_mean': np.mean(gens_arr), 'gens_std': np.std(gens_arr, ddof=1),
    }

if __name__ == "__main__":
    print("=" * 60)
    print("  CMA-ES BENCHMARK (30s budget, N=30)")
    print("=" * 60)

    r1 = run_cmaes(100)
    r2 = run_cmaes(500)
    r3 = run_cmaes(2000)
    r4 = run_cmaes(2000, pop_size=4096, label="POP=4096")

    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    for r in [r1, r2, r3, r4]:
        print(f"  DIM={r['dim']:>5} {r['label']:>12}: "
              f"fitness={r['fitness_mean']:.4f} +/- {r['fitness_std']:.4f}, "
              f"{r['gps_mean']:.2f} gen/s, {r['gens_mean']:.0f} gens")
    print("=" * 60)
