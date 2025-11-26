import numpy as np
import cupy as cp
import math
import pyriemann.utils
import pyriemann.utils.covariance
import pyriemann
from pyriemann.utils.base import logm, invsqrtm, sqrtm, expm
from pyriemann.utils.tangentspace import upper
from pyriemann.utils.mean import mean_riemann
import torch

# From: https://stackoverflow.com/questions/73288332/is-there-a-way-to-compute-the-matrix-logarithm-of-a-pytorch-tensor
def component_logm(x:torch.tensor):
    eigvals, eigvecs = torch.linalg.eigh(x)
    eigvals_log = torch.log(torch.clamp(eigvals, min=1e-10))  # Avoid log(0)
    return eigvecs @ torch.diag_embed(eigvals_log) @ eigvecs.transpose(-1, -2)

# Converted from pyriemann.utils.base.sqrtm
def component_sqrtm(x:torch.tensor):
    eigvals, eigvecs = torch.linalg.eigh(x)
    eigvals = torch.sqrt(eigvals)
    return eigvecs @ torch.diag_embed(eigvals) @ eigvecs.transpose(-1, -2).to(device=x.device, dtype=x.dtype)

# Converted from pyriemann.utils.base.invsqrtm
def component_invsqrtm(x:torch.tensor) -> torch.tensor:
    eigvals, eigvecs = torch.linalg.eigh(x)
    eigvals = torch.div(1., torch.sqrt(eigvals))
    return (eigvecs @ torch.diag_embed(eigvals) @ eigvecs.transpose(-1, -2)).to(device=x.device, dtype=x.dtype)

def component_expm(x:torch.tensor):
    eigvals, eigvecs = torch.linalg.eigh(x)
    eigvals = torch.exp(eigvals)
    return eigvecs @ torch.diag_embed(eigvals) @ eigvecs.transpose(-1, -2).to(device=x.device, dtype=x.dtype)

def component_logm_cupy(x:cp.array):
    eigvals, eigvecs = cp.linalg.eigh(x)
    eigvals = cp.log(cp.maximum(eigvals, 1e-5))  # Avoid log(0)
    if x.ndim >= 3:
        eigvals = cp.expand_dims(eigvals, -2)
    return (eigvecs * eigvals) @ cp.swapaxes(eigvecs.conj(), -2, -1)

# Converted from pyriemann.utils.base.sqrtm
def component_sqrtm_cupy(x:cp.array):
    eigvals, eigvecs = cp.linalg.eigh(x)
    eigvals = cp.sqrt(eigvals)
    if x.ndim >= 3:
        eigvals = cp.expand_dims(eigvals, -2)
    return (eigvecs * eigvals) @ cp.swapaxes(eigvecs.conj(), -2, -1)

# Converted from pyriemann.utils.base.invsqrtm
def component_invsqrtm_cupy(x:cp.array):
    eigvals, eigvecs = cp.linalg.eigh(x)
    eigvals = 1.0 / cp.sqrt(eigvals)
    if x.ndim >= 3:
        eigvals = cp.expand_dims(eigvals, -2)
    return (eigvecs * eigvals) @ cp.swapaxes(eigvecs.conj(), -2, -1)

def component_expm_cupy(x:cp.array):
    eigvals, eigvecs = cp.linalg.eigh(x)
    eigvals = cp.exp(eigvals)
    if x.ndim >= 3:
        eigvals = cp.expand_dims(eigvals, -2)
    return (eigvecs * eigvals) @ cp.swapaxes(eigvecs.conj(), -2, -1)


def component_upper(x:torch.tensor):

    n = x.shape[-1]
    print(n)
    
    # Get the indices for the upper triangular part of the matrix
    idx = torch.triu_indices(n, n, offset=0, device=x.device)
    
    # Create the weight matrix: unity weight for the diagonal, sqrt(2) for the off-diagonal elements
    coeffs = (torch.sqrt(torch.tensor(2.0, device=x.device)) * torch.triu(torch.ones((n, n), device=x.device), diagonal=1) + torch.eye(n, n, device=x.device))
    coeffs = coeffs[idx[0], idx[1]]  # Extract the weights for the upper triangular part
    
    # Extract the upper triangular part and apply the weights
    T = coeffs * x[..., idx[0], idx[1]]
    
    return T

def is_positive_definite(x:torch.Tensor):
    eigvals, eigvecs = torch.linalg.eigh(x)
    return torch.all(eigvals > 0)


def component_riemannian_distance_to_mean(x:torch.Tensor, x_mean:torch.Tensor) -> torch.Tensor:
    """
    Component that computes the Riemannian distance between the covariance point on the manifold and the Riemannian mean.
    Input:
    - x: A tensor representing the covariance matrix on the manifold, with the shape (num_trials, num_channels, num_channels).
    - x_mean: The Riemannian mean to compute the riemannian distance to.
    Output:
    - The Riemannian distance between the covariance points on the manifold and the Riemannian mean.
    """

    x_mean_invsqrtm = component_invsqrtm(x_mean)
    return torch.linalg.matrix_norm(component_logm(x_mean_invsqrtm @ x @ x_mean_invsqrtm), ord='fro', dim=(-2, -1))

def component_riemannian_distance_to_mean_batched(x:torch.Tensor, x_means:torch.Tensor) -> torch.Tensor:
    """
    Component that computes the Riemannian distance between the covariance point on the manifold and the Riemannian mean.
    Input:
    - x: A tensor representing the covariance matrix on the manifold, with the shape (..., num_channels, num_channels).
    - x_means: The Riemannian means to compute the riemannian distance to, with the shape (..., num_means, num_channels, num_channels).
    Output:
    - The Riemannian distance between the covariance points on the manifold and the Riemannian mean, with the shape (..., num_means).
    """

    x_means_invsqrtm = component_invsqrtm(x_means)

    x_expanded = x.unsqueeze(-3)

    x_logms = component_logm(x_means_invsqrtm @ x_expanded @ x_means_invsqrtm)
    distances = torch.linalg.matrix_norm(x_logms, ord='fro', dim=(-2, -1))

    return distances

def component_riemannian_mean_manually(x:torch.Tensor, threshold:float=1e-6, max_iterations:int = 50) -> torch.Tensor:
    """
    Component that computes the Riemannian mean from the covariance matrices.
    Input:
    - x: A tensor of covariance matrices, with the shape (num_trials, num_channels, num_channels).
    - threshold: The threshold for the stopping criterion.
    - max_iterations: The maximum number of iterations (in case of no convergense).
    Output:
    - A tensor representing the Riemannian mean.
    """
   
    num_trials, num_channels, num_channels = x.shape

    # Initialize the mean
    mean = x.mean(dim=0, dtype=x.dtype)

    nu = 1.0
    tau = float('inf')
    crit = float('inf')

    # Iteratively search for the mean
    for _ in range(max_iterations):
        
        mean_sqrt = component_sqrtm(mean)
        mean_invsqrt = component_invsqrtm(mean)

        # Compute the tangent space mean
        logm_terms = [component_logm(mean_invsqrt @ xi @ mean_invsqrt) for xi in x]
        J = torch.mean(torch.stack(logm_terms), dim=0, dtype=x.dtype)

        # Update the mean
        mean = mean_sqrt @ component_expm(nu * J) @ mean_sqrt

        # Compute the convergence criterion
        crit = torch.linalg.matrix_norm(J, ord='fro')
        h = nu * crit
        if h < tau:
            nu *= 0.95
            tau = h
        else:
            nu *= 0.5
        if crit <= threshold or nu <= threshold:
            break
    else:
        print("Convergence not reached", crit, nu)
    
    return mean

def component_extract_covariance_matrices(x:np.ndarray, lambda_regulation=1e-6) -> np.ndarray:
    """
    Component that extracts the covariance matrices for each trial.
    Input:
    - x: An array of trials, with the shape (num_trials, num_channels, num_samples).
    Output:
    - An array of covariance matrices, with the shape (num_trials, num_channels, num_channels).
    """

    return np.array([pyriemann.utils.covariance.covariance_scm(trial, assume_centered=False) for trial in x])# + (lambda_regulation * np.eye(x.shape[1], x.shape[1])) for trial in x])
#------------------------------------------------------------

def component_covariance(x:torch.Tensor, lambda_regulation=1e-6) -> torch.Tensor:
    return torch.from_numpy(pyriemann.utils.covariance.covariance_scm(x.cpu().numpy(), assume_centered=False)).to(device=x.device, dtype=x.dtype) + (lambda_regulation * torch.eye(x.shape[0], x.shape[0], device=x.device, dtype=x.dtype))

def component_scm(x:torch.Tensor, lambda_regulation=1e-6) -> torch.Tensor:
    x_centered = x - x.mean(axis=2, keepdim=True)
    num_samples = x_centered.shape[2]
    return (torch.matmul(x_centered, x_centered.transpose(1, 2)) / (num_samples - 1)) + lambda_regulation*torch.eye(x_centered.shape[1], device=x.device, dtype=x.dtype).unsqueeze(0)
#----------------------------------------------

def component_scm_batched(x:torch.Tensor, lambda_regulation=1e-6) -> torch.Tensor:
    """
    Computes the sample covariance matrix (SCM) with regularization for batched input.

    Inputs:
    - x: Tensor with the shape (..., num_channels, num_samples), where `...` represents arbitrary batch dimensions.
    - lambda_regulation: Regularization term to handle numerical instability.

    Returns:
    - Tensor representing the SCMs with the shape (..., num_channels, num_channels).
    """
    
    x_centered = x - x.mean(dim=-1, keepdim=True)
    num_samples = x_centered.shape[-1]

    # Compute the covariance matrix
    covs = torch.matmul(
        x_centered, x_centered.transpose(-2, -1)
    ).div(num_samples - 1)

    # Add regularization
    identity = torch.eye(x_centered.shape[-2], device=x.device, dtype=x.dtype)
    covs += lambda_regulation * identity.unsqueeze(0).expand_as(covs)

    return covs

def component_riemannian_mean(x:np.ndarray, threshold=1e-6, max_iterations=50) -> np.ndarray:
    """
    Component that computes the Riemannian mean from the covariance matrices.
    Input:
    - x: A tensor of covariance matrices, with the shape (num_trials, num_channels, num_channels).
    - threshold: The threshold for the stopping criterion.
    - max_iterations: The maximum number of iterations (in case of no convergense).
    Output:
    - A tensor representing the Riemannian mean, with the shape (num_channels, num_channels).
    """
    
    return mean_riemann(x, tol=threshold, maxiter=max_iterations)

def component_compute_riemannian_distance_to_mean(x:np.ndarray, x_mean:np.ndarray) -> np.array:
    """
    Component that computes the Riemannian distance between the covariance point on the manifold and the Riemannian mean.
    Input:
    - x: An array of covariance points on the manifold, with the shape (num_trials, num_channels, num_channels).
    - x_mean: The Riemannian mean to compute the riemannian distance to.
    Output:
    - The Riemannian distance between the covariance points on the manifold and the Riemannian mean.
    """

    num_trials, num_channels, num_channels = x.shape
    
    output = np.zeros(num_trials)
    for trial in range(num_trials):
        s = logm(invsqrtm(x_mean) @ x[trial] @ invsqrtm(x_mean))

        output[trial] = np.linalg.norm(s, ord='fro')

    return output
#----------------------------------------------

def component_upper_batched(x:torch.Tensor) -> torch.Tensor:
    num_channels = x.shape[-1]
    indices = torch.triu_indices(num_channels, num_channels, offset=0, device=x.device)

    # Extract the upper triangular part of the matrix.
    upper_triangular_elements = x[..., indices[0], indices[1]]

    # Apply sqrt2 weight on the off-diagonal elements.
    upper_triangular_elements[..., indices[0] != indices[1]] *= torch.sqrt(torch.tensor(2.0, device=x.device, dtype=x.dtype))
    
    return upper_triangular_elements

def component_vectorize_covariance_matrices_batched(x:torch.Tensor, x_mean:torch.Tensor) -> torch.Tensor:
    
    invsqrt = component_invsqrtm(x_mean)

    tangent_matrices = component_logm(invsqrt @ x @ invsqrt)
    
    return component_upper_batched(tangent_matrices)


def component_vectorize_covariance_matrices(x:np.ndarray, x_mean:np.ndarray):
    """
    Component that maps the covariance matrices to the tangent space around the Riemannian mean, and vectorizes them.
    Input:
    - x: An array of covariance matrices, with the shape (num_trials, num_channels, num_channels).
    - x_mean: The Riemannian mean to vectorize the covariance matrices around.
    Output:
    - An array vectors representing the covariance matrices relative to the Riemannian mean, with the shape (num_trials, num_channels x (num_channels + 1) / 2)
    """
    invsqrt = invsqrtm(x_mean)

    tangent_matrices = np.array([
        logm(invsqrt @ c @ invsqrt) for c in x
    ])

    return upper(tangent_matrices)
#----------------------------------------------