#!/usr/bin/env python3
"""
JAX Baseline Benchmark for WebGPU Neuroevolution Comparison.

Tests whether JAX's jit + lax.scan can fuse sequential timestep loops
into a single compiled kernel, which is the main theoretical challenge
to our claim that WebGPU's single-dispatch advantage is unique.

Note: JAX on macOS runs on CPU only (no Metal/MPS backend).
On CUDA machines, JAX would use GPU — results would differ.
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap, random
import jax.lax as lax
import time
import math

print("=" * 64)
print("  JAX BASELINE BENCHMARK")
print(f"  JAX: {jax.__version__}")
print(f"  Device: {jax.devices()[0]}")
print("=" * 64)


# ═══════════════════════════════════════════════════════════════════════
#  BENCHMARK 1: RASTRIGIN
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

    @jit
    def rastrigin_fitness(x):
        return -(10.0 * DIM + jnp.sum(x ** 2 - 10.0 * jnp.cos(2.0 * PI * x), axis=1))

    @jit
    def evolve(pop, fitness, key):
        k1, k2, k3 = random.split(key, 3)
        # Tournament selection
        t_idx = random.randint(k1, (POP, TOURNAMENT_K), 0, POP)
        t_fit = fitness[t_idx]
        winners = t_idx[jnp.arange(POP), t_fit.argmax(axis=1)]
        new_pop = pop[winners]

        # Mutation
        new_pop = new_pop + random.normal(k2, (POP, DIM)) * SIGMA

        # Elitism
        best_idx = fitness.argmax()
        new_pop = new_pop.at[0].set(pop[best_idx])
        return new_pop

    # Init
    key = random.PRNGKey(42)
    key, subkey = random.split(key)
    pop = random.uniform(subkey, (POP, DIM), minval=-5.12, maxval=5.12)

    # Warmup (JIT compilation)
    for _ in range(WARMUP):
        fit = rastrigin_fitness(pop)
        key, subkey = random.split(key)
        pop = evolve(pop, fit, subkey)
    fit.block_until_ready()

    # Timed
    t0 = time.perf_counter()
    for gen in range(NUM_GENS):
        fit = rastrigin_fitness(pop)
        key, subkey = random.split(key)
        pop = evolve(pop, fit, subkey)
    fit.block_until_ready()

    elapsed = time.perf_counter() - t0
    gen_per_sec = NUM_GENS / elapsed
    best_fit = float(fit.max())

    print(f"  Generations: {NUM_GENS}")
    print(f"  Time:        {elapsed:.2f}s")
    print(f"  Gen/sec:     {gen_per_sec:.1f}")
    print(f"  Best fit:    {best_fit:.1f}")
    return gen_per_sec


# ═══════════════════════════════════════════════════════════════════════
#  BENCHMARK 2: FINANCIAL with lax.scan (the key test)
# ═══════════════════════════════════════════════════════════════════════

def bench_financial():
    print("\n" + "─" * 64)
    print("  FINANCIAL SIMULATION BENCHMARK (lax.scan)")
    print(f"  POP=10000  PARAMS=246  TIMESTEPS=5000  ASSETS=16")
    print("─" * 64)

    POP = 10000
    N_PARAMS = 246
    L = 5000
    N_A = 16
    METABOLIC_RATE = 0.00001
    NUM_GENS = 3  # fewer gens since this is slow
    WARMUP = 1

    # Generate market data
    key = random.PRNGKey(42)
    key, k1, k2 = random.split(key, 3)
    signals = random.normal(k1, (L, 36)) * 0.1
    # Set asset scores positive
    signals = signals.at[:, 4:20].set(jnp.clip(signals[:, 4:20] + 0.3, 0, None))
    # Set returns small
    signals = signals.at[:, 20:36].set(random.normal(k2, (L, 16)) * 0.01)

    def nn_forward_single(X, genome):
        """Single genome forward pass: (8,) → (6,)"""
        W1 = genome[:128].reshape(8, 16)
        B1 = genome[128:144]
        H = jnp.clip(2.0 * (X @ W1 + B1), -20.0, 20.0)
        ext = jnp.exp(H)
        H = (ext - 1.0) / (ext + 1.0)

        W2 = genome[144:240].reshape(16, 6)
        B2 = genome[240:246]
        Out = H @ W2 + B2
        Out = 1.0 / (1.0 + jnp.exp(jnp.clip(-Out, -20.0, 20.0)))
        return Out

    # Vectorize over population
    nn_forward_batch = vmap(nn_forward_single, in_axes=(0, 0))

    def step_fn(carry, sig):
        """Single timestep — used by lax.scan to fuse the loop."""
        pv, pp, mem0, mem1, pw, rets_cum, pk, mdd, genomes = carry

        dd = jnp.where(pp > 0, jnp.clip((pp - pv) / pp, 0, 1), 0.0)
        fir_pnl = jnp.clip((pv / jnp.maximum(pp, 1e-8) - 1.0) * 10.0, 0.0, 1.0)

        X = jnp.stack([
            jnp.full(POP, sig[0]),
            jnp.clip(dd * 5.0, 0, 1),
            jnp.zeros(POP),
            fir_pnl,
            jnp.full(POP, sig[1]),
            jnp.full(POP, sig[2]),
            mem0, mem1
        ], axis=1)

        emotions = nn_forward_batch(X, genomes)
        caution = emotions[:, 0]
        lev_base = 1.0 + emotions[:, 1] * 2.0
        maxw_base = (1.0 - caution * 0.8) * (0.2 + emotions[:, 3] * 0.8)
        mem0 = emotions[:, 4]
        mem1 = emotions[:, 5]

        # Allocate
        scores = jnp.clip(sig[4:4 + N_A], 0, None)
        w = jnp.broadcast_to(scores, (POP, N_A))
        tw = jnp.maximum(w.sum(axis=1, keepdims=True), 1e-8)
        w = w / tw
        w = jnp.minimum(w, maxw_base[:, None])
        tw2 = jnp.maximum(w.sum(axis=1, keepdims=True), 1e-8)
        w = w / tw2

        # Hysteresis
        w = jnp.where(jnp.abs(w - pw) < 0.05, pw, w)

        # Returns
        asset_rets = sig[20:20 + N_A]
        rr = (w * asset_rets).sum(axis=1)
        dr = rr * lev_base

        # Friction
        diff = jnp.abs(w - pw)
        friction = jnp.where(w < pw, 0.0002, 0.0004)
        dr = dr - (diff * lev_base[:, None] * friction).sum(axis=1)

        dr = dr - METABOLIC_RATE
        pv = pv * (1.0 + dr)
        pv = jnp.maximum(pv, 1e-8)
        pv = jnp.where(pv <= 0.05, 0.0001, pv)
        pp = jnp.maximum(pp, pv)

        rets_cum = rets_cum * (1.0 + dr)
        pk = jnp.maximum(pk, rets_cum)
        current_dd = (pk - rets_cum) / jnp.maximum(pk, 1e-8)
        mdd = jnp.maximum(mdd, current_dd)

        carry = (pv, pp, mem0, mem1, w, rets_cum, pk, mdd, genomes)
        return carry, None  # no per-step output needed

    @jit
    def evaluate(genomes):
        """Full evaluation using lax.scan to fuse the 5000-step loop."""
        pv = jnp.ones(POP)
        pp = jnp.ones(POP)
        mem0 = jnp.zeros(POP)
        mem1 = jnp.zeros(POP)
        pw = jnp.zeros((POP, N_A))
        rets_cum = jnp.ones(POP)
        pk = jnp.ones(POP)
        mdd = jnp.zeros(POP)

        init_carry = (pv, pp, mem0, mem1, pw, rets_cum, pk, mdd, genomes)
        (pv, pp, mem0, mem1, pw, rets_cum, pk, mdd, _), _ = lax.scan(step_fn, init_carry, signals)

        return jnp.where(mdd > 0.50, -9999.0, rets_cum)

    def evolve(genomes, fitness, key):
        """Simple tournament + mutation (not the bottleneck)."""
        ISLAND_SIZE = 1000
        new_parts = []
        for island in range(POP // ISLAND_SIZE):
            s, e = island * ISLAND_SIZE, (island + 1) * ISLAND_SIZE
            f = fitness[s:e]
            g = genomes[s:e]

            elite_idx = f.argmax()

            key, k1, k2, k3, k4 = random.split(key, 5)
            t1 = random.randint(k1, (ISLAND_SIZE - 1, 5), 0, ISLAND_SIZE)
            p1 = t1[jnp.arange(ISLAND_SIZE - 1), f[t1].argmax(axis=1)]
            t2 = random.randint(k2, (ISLAND_SIZE - 1, 5), 0, ISLAND_SIZE)
            p2 = t2[jnp.arange(ISLAND_SIZE - 1), f[t2].argmax(axis=1)]

            mask = random.uniform(k3, (ISLAND_SIZE - 1, N_PARAMS)) < 0.5
            children = jnp.where(mask, g[p1], g[p2])

            mut_r = random.uniform(k4, (ISLAND_SIZE - 1, N_PARAMS))
            key, k5, k6 = random.split(key, 3)
            children = children + jnp.where(mut_r < 0.05, random.normal(k5, children.shape) * 0.6, 0)
            children = children + jnp.where((mut_r >= 0.05) & (mut_r < 0.20), random.normal(k6, children.shape) * 0.03, 0)

            island_pop = jnp.concatenate([g[elite_idx:elite_idx + 1], children])
            new_parts.append(island_pop)

        return jnp.concatenate(new_parts)

    # Init
    key, subkey = random.split(key)
    genomes = random.normal(subkey, (POP, N_PARAMS)) * 1.5

    # Warmup (JIT compile — this is where lax.scan compilation happens)
    print("  Compiling with lax.scan (this may take a while)...")
    t_compile_start = time.perf_counter()
    for _ in range(WARMUP):
        fitness = evaluate(genomes)
        key, subkey = random.split(key)
        genomes = evolve(genomes, fitness, subkey)
    fitness.block_until_ready()
    t_compile = time.perf_counter() - t_compile_start
    print(f"  Compilation time: {t_compile:.1f}s")

    # Timed
    times = []
    for gen in range(NUM_GENS):
        t0 = time.perf_counter()
        fitness = evaluate(genomes)
        fitness.block_until_ready()
        t_eval = time.perf_counter() - t0

        t1 = time.perf_counter()
        key, subkey = random.split(key)
        genomes = evolve(genomes, fitness, subkey)
        genomes.block_until_ready()
        t_evolve = time.perf_counter() - t1

        total = time.perf_counter() - t0
        times.append(total)

        valid = fitness[fitness > -9000]
        best = float(valid.max()) if valid.size > 0 else -9999
        print(f"  Gen {gen} | Best: {best:.4f} | Eval: {t_eval:.1f}s | Evolve: {t_evolve:.1f}s | Total: {total:.1f}s")

    avg_time = sum(times) / len(times)
    gen_per_sec = 1.0 / avg_time

    print(f"\n  Avg time/gen: {avg_time:.2f}s")
    print(f"  Gen/sec:      {gen_per_sec:.4f}")
    return gen_per_sec


# ═══════════════════════════════════════════════════════════════════════
#  RUN ALL
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    rastrigin_gps = bench_rastrigin()
    financial_gps = bench_financial()

    print("\n" + "=" * 64)
    print("  SUMMARY — JAX Baseline")
    print("=" * 64)
    print(f"  Device:            {jax.devices()[0]}")
    print(f"  Rastrigin gen/sec: {rastrigin_gps:.1f}")
    print(f"  Financial gen/sec: {financial_gps:.4f}")
    print()
    print("  ┌──────────────┬────────────┬────────────┬────────────┬──────────┐")
    print("  │ Benchmark    │ NumPy CPU  │ JAX (CPU)  │ PyTorch GPU│ WebGPU   │")
    print("  ├──────────────┼────────────┼────────────┼────────────┼──────────┤")
    print(f"  │ Rastrigin    │ 3.68 g/s   │ {rastrigin_gps:>7.1f} g/s │  173.0 g/s │ 179.3 g/s│")
    print(f"  │ Financial    │ 0.056 g/s  │ {financial_gps:>7.4f} g/s│  0.29 g/s  │ ~24 g/s  │")
    print("  └──────────────┴────────────┴────────────┴────────────┴──────────┘")
    print()
    print("  Note: JAX runs on CPU only (no Metal/MPS backend on macOS).")
    print("  On CUDA hardware, JAX GPU results would differ.")
    print("=" * 64)
