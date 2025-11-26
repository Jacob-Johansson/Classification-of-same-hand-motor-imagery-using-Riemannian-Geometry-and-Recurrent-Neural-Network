import numpy as np
from mdrm import MDRM

mdrm = MDRM()
x = np.random.randn(120, 3, 12, 2)

# Measure latency.
import time
start = time.time()
dists = mdrm.fit_transform(x)
end = time.time()
print("Dists Time:", end - start)

start = time.time()
probs = mdrm.predict_probs(x)
end = time.time()
print("Prob Time:", end - start)

