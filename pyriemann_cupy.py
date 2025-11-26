import numpy as np
import cupy as cp
import warnings

# A Cupy port of pyriemann.base._matrix_operator function.
def _matrix_operator(C, operator):
  if C.dtype.char in np.typecodes['AllFloat'] and (cp.isinf(C).any() or cp.isnan(C).any()):
    raise ValueError("Matrices must be positive definite. Add regularization to avoid this error.")
  eigvals, eigvecs = cp.linalg.eigh(C)
  eigvals = operator(eigvals)
  if C.ndim >= 3:
    eigvals = cp.expand_dims(eigvals, -2)
  D = (eigvecs * eigvals) @ cp.swapaxes(eigvecs.conj(), -2, -1)
  return D

# A Cupy port of pyriemann.base.logm function.
def logm(x):
  return _matrix_operator(x, cp.log)

# A Cupy port of pyriemann.base.expm function.
def expm(x):
  return _matrix_operator(x, cp.exp)

# A Cupy port of pyriemann.base.sqrtm function.
def sqrtm(x):
  return _matrix_operator(x, cp.sqrt)

# A Cupy port of pyriemann.base.invsqrtm function.
def invsqrtm(x):
  def isqrt(x): return 1. / cp.sqrt(x)
  return _matrix_operator(x, isqrt)

# A Cupy port of pyriemann.utils.mean.mean_riemann function.
def mean_riemann(x, threshold=10e-9, max_iterations=100):
  
  nu = 1.0
  tau = cp.finfo(cp.float64).max
  crit = cp.finfo(cp.float64).max

  sample_weight = cp.ones(x.shape[0])
  sample_weight /= cp.sum(sample_weight)

  M = cp.average(x, axis=0, weights=sample_weight)

  for _ in range(max_iterations):
    M12, Mm12 = sqrtm(M), invsqrtm(M)

    J = cp.einsum("a,abc->bc", sample_weight, logm(Mm12 @ x @ Mm12))
    M = M12 @ expm(nu * J) @ M12

    crit = cp.linalg.norm(J, ord="fro")
    h = nu * crit
    if h < tau:
        nu = 0.95 * nu
        tau = h
    else:
        nu = 0.5 * nu
    if crit <= threshold or nu <= threshold:
        break
  else:
      warnings.warn(f"Convergence not reached {crit}, {nu}")

  return M