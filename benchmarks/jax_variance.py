#!/usr/bin/env python3
"""JAX CPU Rastrigin variance — N=10, POP=4096, DIM=2000."""
import jax
import jax.numpy as jnp
from jax import jit, random
import time
import math
import numpy as np

POP, DIM, PI = 4096, 2000, math.pi

print(f"JAX {jax.__version__} on {jax.devices()[0]}")
print("=" * 50)

@jit
def rastrigin_fitness(x):
    return -(10.0 * DIM + jnp.sum(x**2 - 10.0 * jnp.cos(2.0 * PI * x), axis=1))

@jit
def evolve(pop, fitness, key):
    k1, k2 = random.split(key)
    t_idx = random.randint(k1, (POP, 5), 0, POP)
    winners = t_idx[jnp.arange(POP), fitness[t_idx].argmax(axis=1)]
    new_pop = pop[winners] + random.normal(k2, (POP, DIM)) * 0.3
    new_pop = new_pop.at[0].set(pop[fitness.argmax()])
    return new_pop

runs = []
for i in range(10):
    key = random.PRNGKey(i * 42)
    key, subkey = random.split(key)
    pop = random.uniform(subkey, (POP, DIM), minval=-5.12, maxval=5.12)

    # Warmup
    for _ in range(10):
        fit = rastrigin_fitness(pop)
        key, subkey = random.split(key)
        pop = evolve(pop, fit, subkey)
    fit.block_until_ready()

    t0 = time.perf_counter()
    for _ in range(100):
        fit = rastrigin_fitness(pop)
        key, subkey = random.split(key)
        pop = evolve(pop, fit, subkey)
    fit.block_until_ready()
    elapsed = time.perf_counter() - t0
    gps = 100 / elapsed
    print(f"  Run {i+1}/10: {gps:.1f} gen/s")
    runs.append(gps)

print(f"\n  JAX CPU RESULT: {np.mean(runs):.1f} +/- {np.std(runs, ddof=1):.1f} gen/s (N=10)")
