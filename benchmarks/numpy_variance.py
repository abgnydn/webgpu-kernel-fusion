#!/usr/bin/env python3
"""NumPy CPU Rastrigin variance — N=10, POP=4096, DIM=2000."""
import numpy as np
import time

POP, DIM = 4096, 2000

print(f"NumPy {np.__version__}")
print("=" * 50)

runs = []
for i in range(10):
    pop = np.random.uniform(-5.12, 5.12, (POP, DIM)).astype(np.float32)
    next_pop = np.empty_like(pop)

    t0 = time.perf_counter()
    for _ in range(50):
        fitness = -(10.0 * DIM + np.sum(pop * pop - 10.0 * np.cos(2.0 * np.pi * pop), axis=1))
        best_idx = np.argmax(fitness)
        next_pop[0] = pop[best_idx]
        p1 = pop[np.random.randint(0, POP, POP - 1)]
        p2 = pop[np.random.randint(0, POP, POP - 1)]
        mask = np.random.rand(POP - 1, DIM) < 0.5
        child = np.where(mask, p1, p2)
        child += np.random.randn(POP - 1, DIM).astype(np.float32) * 0.3 * (np.random.rand(POP - 1, DIM) < 0.1)
        next_pop[1:] = child
        pop, next_pop = next_pop, pop
    elapsed = time.perf_counter() - t0
    gps = 50 / elapsed
    print(f"  Run {i+1}/10: {gps:.1f} gen/s")
    runs.append(gps)

print(f"\n  NumPy RESULT: {np.mean(runs):.1f} +/- {np.std(runs, ddof=1):.1f} gen/s (N=10)")
