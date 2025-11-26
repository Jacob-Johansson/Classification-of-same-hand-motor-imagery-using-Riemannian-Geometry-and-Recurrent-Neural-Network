import numpy as np
import feature_fusion

x = np.random.randn()

x = np.random.randn(120, 3, 12, 903)
y = np.random.randint(0, 2, 120)

fusion = feature_fusion.FeatureFusion((-1,))


# Measure latency.
import time
start = time.time()
x_fused = fusion.transform(x)
end = time.time()
print("Transform Time:", end - start)
print("Original shape:", x.shape)
print("Fused shape:", x_fused.shape)

# Test the order.
compare = np.zeros((120, 32508))
for w in range(x.shape[1]):
    for b in range(x.shape[2]):
        start = 903 * w * x.shape[2] +  903 * b
        indices = range(start, start + 903)
        print(indices)
        compare[:, indices] = x[:, w, b, :]

assert np.all(x_fused == compare)