import numpy as np
import riemann_distance_feature_extraction
import riemann_scm

x = np.random.randn(120, 3, 12, 42, 1000)
y = np.random.randint(0, 2, 120)
c = riemann_scm.RiemannSCM().fit_transform(x, y)

distance_feature_extraction = riemann_distance_feature_extraction.RiemannDistanceFeatureExtraction(num_jobs=4)

# Measure latency.
import time
start = time.time()
dists = distance_feature_extraction.fit_transform(c, y)
end = time.time()
print("Time:", end - start)