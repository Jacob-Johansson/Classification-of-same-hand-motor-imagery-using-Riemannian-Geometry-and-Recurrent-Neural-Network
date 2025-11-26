import numpy as np
import dimension_reduction

x = np.random.randn(120, 32508)
y = np.random.randint(0, 2, 120)

reduction = dimension_reduction.DimensionReduction(2)

# Measure latency.
import time
start = time.time()
reduction.fit(x, y)
end = time.time()
print("Fit Time:", end - start)

start = time.time()
x_reduced = reduction.transform(x)
end = time.time()
print("Transform Time:", end - start)
print(x_reduced.shape)