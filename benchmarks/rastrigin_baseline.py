import numpy as np
import time

# Same constants as WebGPU p2p_demo.html
POP = 4096
DIM = 2000

def run_benchmark(generations=50):
    print(f"Initializing NumPy baseline: {POP} networks, {DIM} dimensions")
    # Initialize from [-5.12, 5.12]
    pop = np.random.uniform(-5.12, 5.12, size=(POP, DIM)).astype(np.float32)
    next_pop = np.empty_like(pop)

    print("Starting evolution loop...")
    start_time = time.time()

    for gen in range(generations):
        # 1. EVALUATE (Rastrigin)
        # f(x) = 10*n + sum(x_i^2 - 10*cos(2*pi*x_i))
        # Maximize -f(x)
        fitness = -(10.0 * DIM + np.sum(pop * pop - 10.0 * np.cos(2.0 * np.pi * pop), axis=1))

        # 2. SELECT & EVOLVE
        best_idx = np.argmax(fitness)
        best_fit = fitness[best_idx]

        # Elitism
        next_pop[0] = pop[best_idx]
        next_pop[1] = pop[best_idx] # Fake foreign elite injection

        # Tournament selection (simplified vectorized)
        # Sample 2 parents for each of the (POP-2) remaining slotes
        # For fairness to Python, we'll do random selection instead of full tournament to avoid massive Python bloat,
        # which actually makes pythonFASTER than WebGPU's strict tournament
        p1_idx = np.random.randint(0, POP, size=POP-2)
        p2_idx = np.random.randint(0, POP, size=POP-2)

        parent1 = pop[p1_idx]
        parent2 = pop[p2_idx]

        # Crossover mask
        mask = np.random.rand(POP-2, DIM) < 0.5
        child = np.where(mask, parent1, parent2)

        # Mutate
        mutation_mask_1 = np.random.rand(POP-2, DIM) < 0.1
        mutation_mask_2 = np.random.rand(POP-2, DIM) < 0.3

        gauss = np.random.normal(0, 1, size=(POP-2, DIM)).astype(np.float32)

        child += mutation_mask_1 * gauss * 0.5
        child += mutation_mask_2 * gauss * 0.05

        # Clamp
        np.clip(child, -5.12, 5.12, out=child)

        next_pop[2:] = child

        # Swap buffers
        pop, next_pop = next_pop, pop

        if (gen + 1) % 10 == 0:
            print(f"Gen {gen+1:03d} | Best Fit: {best_fit:.4f}")

    elapsed = time.time() - start_time
    gps = generations / elapsed

    print("-" * 40)
    print(f"NumPy Elapsed: {elapsed:.2f} seconds")
    print(f"NumPy Speed:   {gps:.2f} gen/s")
    print(f"WebGPU Speed:  422.00 gen/s (from demo)")
    print(f"Acceleration:  {422.0 / gps:.1f}x faster on GPU")

if __name__ == "__main__":
    run_benchmark()
