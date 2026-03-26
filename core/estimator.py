"""
Unbiased Inner Product Estimator for TurboQuant

Implements the key mathematical result of TurboQuant: an unbiased estimator
for inner products using the two-stage quantization scheme.

Mathematical Foundation:
------------------------
Given vectors x, q ∈ ℝ^d:
  - Stage 1 produces x_hat (scalar quantization reconstruction)
  - Residual r = x - x_hat
  - Stage 2 encodes sign(R @ r) where R is Gaussian

The inner product decomposes as:
  <q, x> = <q, x_hat> + <q, r>

The QJL correction term estimates <q, r> via:
  <q, r> ≈ sqrt(π/2) * ||r|| / m * <R@q, sign(R@r)>

This estimator is UNBIASED: E[estimate] = <q, r>
with variance O(1/m) where m is the QJL output dimension.
"""

import torch
from torch import Tensor
from typing import Optional, Tuple
import math


def _qjl_correction_factor(m: int) -> float:
    """
    Compute the QJL scaling factor.
    
    For Gaussian projection matrix R with unit-norm rows:
      E[<Rq, sign(Rr)>] = sqrt(2/π) * m * <q, r> / ||r||
    
    Therefore:
      <q, r> ≈ sqrt(π/2) * ||r|| / m * <Rq, sign(Rr)>
    
    Args:
        m: QJL output dimension
        
    Returns:
        Scaling factor sqrt(π/2) / m
    """
    return math.sqrt(math.pi / 2) / m


def estimate_inner_product_unbiased(
    q: Tensor,
    x_hat: Tensor,
    q_projected: Tensor,
    r_signs: Tensor,
    r_norm: Tensor,
    m: int,
    eps: float = 1e-8
) -> Tensor:
    """
    Estimate inner product <q, x> using TurboQuant two-stage encoding.
    
    Formula:
      <q, x> ≈ <q, x_hat> + sqrt(π/2) * ||r|| / m * <R@q, sign(R@r)>
    
    Args:
        q: Query vectors (..., d)
        x_hat: Reconstructed vectors from Stage 1 (..., d)
        q_projected: Projected query R@q (..., m)
        r_signs: Sign vector sign(R@r) (..., m)
        r_norm: Residual norm ||r|| (..., 1)
        m: QJL output dimension
        eps: Numerical stability
        
    Returns:
        Estimated inner product (...)
    """
    # Stage 1: Base inner product from reconstruction
    base_dot = (q * x_hat).sum(dim=-1)
    
    # Stage 2: QJL correction term
    # <Rq, sign(Rr)>
    projected_dot = (q_projected * r_signs).sum(dim=-1)
    
    # Scaling: sqrt(π/2) * ||r|| / m
    scale = _qjl_correction_factor(m) * r_norm.squeeze(dim=-1)
    
    correction = scale * projected_dot
    
    return base_dot + correction


def estimate_inner_product_batch(
    queries: Tensor,
    keys_hat: Tensor,
    q_projected: Tensor,
    r_signs: Tensor,
    r_norms: Tensor,
    m: int,
    eps: float = 1e-8
) -> Tensor:
    """
    Batch estimate inner products between multiple queries and keys.
    
    Efficiently computes all pairwise inner products using matrix operations.
    
    Args:
        queries: Query vectors (n_q, d)
        keys_hat: Reconstructed key vectors (n_k, d)
        q_projected: Projected queries (n_q, m)
        r_signs: Sign vectors for keys (n_k, m)
        r_norms: Residual norms for keys (n_k, 1)
        m: QJL output dimension
        eps: Numerical stability
        
    Returns:
        Estimated inner products (n_q, n_k)
    """
    # Stage 1: All pairwise <q, k_hat>
    # (n_q, d) @ (n_k, d).T = (n_q, n_k)
    base_dots = queries @ keys_hat.T
    
    # Stage 2: QJL correction for all pairs
    # <Rq_i, sign_j> for all i, j
    # (n_q, m) @ (n_k, m).T = (n_q, n_k)
    projected_dots = q_projected @ r_signs.T
    
    # Scaling per key: sqrt(π/2) / m * ||r_j||
    scale = _qjl_correction_factor(m)  # scalar
    r_norms_flat = r_norms.squeeze(dim=-1)  # (n_k,)
    
    # Apply scaling: each column j scaled by ||r_j||
    correction = scale * projected_dots * r_norms_flat.unsqueeze(0)
    
    return base_dots + correction


class UnbiasedInnerProductEstimator:
    """
    Stateful unbiased inner product estimator for TurboQuant.
    
    Maintains the QJL projection and provides methods for
    single and batch inner product estimation.
    
    Attributes:
        qjl: QJLProjection instance
        input_dim: Input dimension d
        output_dim: QJL output dimension m
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        seed: int = 42,
        device: Optional[torch.device] = None
    ):
        """
        Initialize estimator.
        
        Args:
            input_dim: Input vector dimension d
            output_dim: QJL output dimension m
            seed: Random seed for QJL
            device: Target device
        """
        from .qjl_projection import QJLProjection
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        
        self.qjl = QJLProjection(input_dim, output_dim, seed, device)
    
    def encode_key(
        self,
        x: Tensor,
        x_hat: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Encode a key vector for inner product estimation.
        
        Args:
            x: Original key vector (..., d)
            x_hat: Reconstructed key vector (..., d)
            
        Returns:
            x_hat: Reconstructed vector (passed through)
            r_signs: QJL sign vector (..., m)
            r_norm: Residual norm (..., 1)
        """
        from .residual import encode_residual_qjl
        
        r_signs, r_norm = encode_residual_qjl(x, x_hat, self.qjl)
        return x_hat, r_signs, r_norm
    
    def encode_query(self, q: Tensor) -> Tensor:
        """
        Encode a query vector (just projection, no quantization).
        
        Args:
            q: Query vector (..., d)
            
        Returns:
            q_projected: R@q (..., m)
        """
        return self.qjl.project(q)
    
    def estimate(
        self,
        q: Tensor,
        x_hat: Tensor,
        r_signs: Tensor,
        r_norm: Tensor
    ) -> Tensor:
        """
        Estimate single inner product <q, x>.
        
        Args:
            q: Query vector (..., d)
            x_hat: Reconstructed key (..., d)
            r_signs: Encoded residual (..., m)
            r_norm: Residual norm (..., 1)
            
        Returns:
            Estimated inner product (...)
        """
        q_projected = self.encode_query(q)
        
        return estimate_inner_product_unbiased(
            q, x_hat, q_projected, r_signs, r_norm, self.output_dim
        )
    
    def estimate_batch(
        self,
        queries: Tensor,
        keys_hat: Tensor,
        r_signs: Tensor,
        r_norms: Tensor
    ) -> Tensor:
        """
        Estimate batch of inner products.
        
        Args:
            queries: Query vectors (n_q, d)
            keys_hat: Reconstructed keys (n_k, d)
            r_signs: Encoded residuals (n_k, m)
            r_norms: Residual norms (n_k, 1)
            
        Returns:
            Estimated inner products (n_q, n_k)
        """
        q_projected = self.encode_query(queries)
        
        return estimate_inner_product_batch(
            queries, keys_hat, q_projected, r_signs, r_norms, self.output_dim
        )
    
    def compute_correction_term(
        self,
        q: Tensor,
        r_signs: Tensor,
        r_norm: Tensor
    ) -> Tensor:
        """
        Compute only the QJL correction term (Stage 2 contribution).
        
        Useful for analyzing the contribution of each stage.
        
        Args:
            q: Query vector (..., d)
            r_signs: Encoded residual (..., m)
            r_norm: Residual norm (..., 1)
            
        Returns:
            Correction term (...)
        """
        q_projected = self.encode_query(q)
        
        projected_dot = (q_projected * r_signs).sum(dim=-1)
        scale = _qjl_correction_factor(self.output_dim) * r_norm.squeeze(-1)
        
        return scale * projected_dot


def validate_unbiasedness(
    estimator: UnbiasedInnerProductEstimator,
    x: Tensor,
    q: Tensor,
    x_hat: Tensor,
    num_samples: int = 1000
) -> Tuple[float, float]:
    """
    Validate that the estimator is unbiased via Monte Carlo.
    
    Runs multiple trials with different QJL seeds and checks that
    E[estimate - true] ≈ 0.
    
    Args:
        estimator: Estimator instance
        x: True vector (d,)
        q: Query vector (d,)
        x_hat: Reconstructed vector (d,)
        num_samples: Number of Monte Carlo samples
        
    Returns:
        mean_error: E[estimate - true]
        std_error: Std of estimation error
    """
    true_dot = (q * x).sum().item()
    
    errors = []
    for seed in range(num_samples):
        # Create estimator with different seed
        est = UnbiasedInnerProductEstimator(
            estimator.input_dim,
            estimator.output_dim,
            seed=seed,
            device=estimator.device
        )
        
        x_hat_enc, r_signs, r_norm = est.encode_key(x, x_hat)
        estimate = est.estimate(q, x_hat_enc, r_signs, r_norm).item()
        
        errors.append(estimate - true_dot)
    
    errors_tensor = torch.tensor(errors)
    mean_error = errors_tensor.mean().item()
    std_error = errors_tensor.std().item()
    
    return mean_error, std_error
