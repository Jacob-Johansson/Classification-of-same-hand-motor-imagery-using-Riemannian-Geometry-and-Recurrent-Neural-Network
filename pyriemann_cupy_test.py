import pyriemann
import pyriemann_cupy
import numpy as np
import cupy as cp
from pyriemann.utils.mean import mean_riemann
import time


num_gpus = cp.cuda.runtime.getDeviceCount()
print(f"Number of GPUs available: {num_gpus}")
for i in range(num_gpus):
    props = cp.cuda.runtime.getDeviceProperties(i)
    print(f"GPU-{i}: {props['name']}")

# Distance to mean test
num_trials, num_channels, num_samples = 120, 42, 1000
covs = cp.zeros((num_trials, num_channels, num_channels))
for i in range(num_trials):
    covs[i, :, :] = cp.cov(cp.random.randn(num_channels, num_samples, dtype=cp.float64), bias=True)

from cupyx.profiler import benchmark

report = benchmark(pyriemann_cupy.mean_riemann, (covs, 1e-8, 100), n_repeat=42, n_warmup=10)
print(report)

start = time.time()
for _ in range(42):
    mean_py = mean_riemann(covs.get(), maxiter=100)
end = time.time()
print("Py mean Time:", end - start)

