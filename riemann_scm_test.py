import numpy as np
import riemann_scm

# Ensure the output are positive definite symmetric matrices 
x = np.random.randn(120, 3, 12, 42, 1000)
riemannscm = riemann_scm.RiemannSCM()
c = riemannscm.fit_transform(x)

assert c.shape[0] == x.shape[0]
assert c.shape[1] == x.shape[1]
assert c.shape[2] == x.shape[2]
assert c.shape[3] == x.shape[3]
assert c.shape[4] == x.shape[3]

for trial in range(c.shape[0]):
    for window in range(c.shape[1]):
        for band in range(c.shape[2]):
            cov = c[trial, window, band]
            assert np.allclose(cov, cov.T)
            assert np.all(np.linalg.eigvalsh(cov) > 0)

# Measure latency 
import time
start = time.time()
c2 = riemannscm.fit_transform(x)
end = time.time()
print("Time:", end - start)