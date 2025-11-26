import numpy as np
import riemann_scm
import riemann_tangent_space_feature_extraction

x = np.random.randn(120, 3, 12, 42, 1000)
y = np.random.randint(0, 2, 120)
c = riemann_scm.RiemannSCM().fit_transform(x, y)

feature_extraction = riemann_tangent_space_feature_extraction.RiemannTangentSpaceFeatureExtraction(max_iterations=200, num_jobs=4)

# Measure latency.
import time
start = time.time()
feature_extraction.fit(c, y)
end = time.time()
print("Fit Time:", end - start)
start = time.time()
features = feature_extraction.transform(c)
end = time.time()
print("Transform Time:", end - start)