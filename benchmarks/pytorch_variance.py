#!/usr/bin/env python3
"""PyTorch MPS Rastrigin variance — N=10, POP=4096, DIM=2000."""
import torch
import time
import math

DEVICE = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
POP, DIM, PI = 4096, 2000, math.pi

print(f"PyTorch {torch.__version__} on {DEVICE}")
print("=" * 50)

runs = []
for i in range(10):
    pop = (torch.rand(POP, DIM, device=DEVICE) * 10.24 - 5.12).float()
    # Warmup
    for _ in range(10):
        fit = -(10.0 * DIM + (pop**2 - 10.0 * torch.cos(2.0 * PI * pop)).sum(dim=1))
        idx = torch.randint(0, POP, (POP, 5), device=DEVICE)
        winners = idx[torch.arange(POP, device=DEVICE), fit[idx].argmax(dim=1)]
        pop = pop[winners] + torch.randn_like(pop[winners]) * 0.3
        pop[0] = pop[fit.argmax()]
    if DEVICE.type == 'mps': torch.mps.synchronize()

    t0 = time.perf_counter()
    for _ in range(100):
        fit = -(10.0 * DIM + (pop**2 - 10.0 * torch.cos(2.0 * PI * pop)).sum(dim=1))
        idx = torch.randint(0, POP, (POP, 5), device=DEVICE)
        winners = idx[torch.arange(POP, device=DEVICE), fit[idx].argmax(dim=1)]
        pop = pop[winners] + torch.randn_like(pop[winners]) * 0.3
        pop[0] = pop[fit.argmax()]
    if DEVICE.type == 'mps': torch.mps.synchronize()
    elapsed = time.perf_counter() - t0
    gps = 100 / elapsed
    print(f"  Run {i+1}/10: {gps:.1f} gen/s")
    runs.append(gps)

import numpy as np
print(f"\n  PyTorch RESULT: {np.mean(runs):.1f} +/- {np.std(runs, ddof=1):.1f} gen/s (N=10)")
