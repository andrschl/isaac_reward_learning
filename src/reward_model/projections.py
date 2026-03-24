"""Projection and proximal operators for reward model regularization.

References:
- L1 ball: Duchi et al., "Efficient Projections onto the L1-Ball for Learning in High Dimensions" (ICML 2008)
- L1 proximal: Parikh & Boyd, "Proximal Algorithms"
- Elastic net proximal: Zou & Hastie, "Regularization and Variable Selection Via the Elastic Net"
"""

from __future__ import annotations

import torch


def project_onto_l2_ball(x: torch.Tensor, radius: float) -> torch.Tensor:
    """Project x onto L2 ball of given radius (in-place style; returns projected tensor).

    min ||x - u||_2  s.t.  ||u||_2 <= radius

    Args:
        x: Tensor of shape [..., D]
        radius: Ball radius (> 0)

    Returns:
        Projected tensor, same shape as x.
    """
    if radius <= 0:
        raise ValueError(f"`radius` must be > 0, got {radius}.")
    norm = x.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-8)
    scale = (norm / radius).clamp(min=1.0)
    return x / scale


def project_onto_l1_ball(x: torch.Tensor, radius: float) -> torch.Tensor:
    """Project x onto L1 ball of given radius (Duchi et al. 2008).

    min ||x - u||_2  s.t.  ||u||_1 <= radius

    O(n log n) sort-based algorithm. Handles batched inputs.

    Args:
        x: Tensor of shape [..., D]
        radius: Ball radius (> 0)

    Returns:
        Projected tensor, same shape as x.
    """
    if radius <= 0:
        raise ValueError(f"`radius` must be > 0, got {radius}.")

    original_shape = x.shape
    x_flat = x.view(-1, x.shape[-1])
    l1_norm = x_flat.abs().sum(dim=1, keepdim=True)
    inside = (l1_norm <= radius).float()

    # Duchi et al.: sort |x| descending, find threshold theta
    mu, _ = torch.sort(x_flat.abs(), dim=1, descending=True)
    cumsum = mu.cumsum(dim=1)
    arange = torch.arange(1, x_flat.shape[1] + 1, device=x.device, dtype=x.dtype)
    cond = mu * arange > (cumsum - radius)
    rho = (cond * arange).max(dim=1).values
    rho = rho.clamp(min=1).long()

    # theta = (cumsum[rho-1] - radius) / rho
    batch_idx = torch.arange(x_flat.shape[0], device=x.device)
    rho_idx = (rho - 1).clamp(0, x_flat.shape[1] - 1)
    theta = (cumsum[batch_idx, rho_idx] - radius) / rho.to(x.dtype)

    # proj = sign(x) * max(0, |x| - theta)
    proj = (x_flat.abs() - theta.unsqueeze(1)).clamp(min=0) * x_flat.sign()

    out = inside * x_flat + (1 - inside) * proj
    return out.view(original_shape)


def soft_threshold(x: torch.Tensor, lam: float) -> torch.Tensor:
    """L1 proximal operator (soft-thresholding).

    prox_{lam ||·||_1}(v) = sign(v) * max(0, |v| - lam)

    Args:
        x: Tensor
        lam: Threshold (>= 0)

    Returns:
        Soft-thresholded tensor, same shape as x.
    """
    if lam < 0:
        raise ValueError(f"`lam` must be >= 0, got {lam}.")
    if lam == 0:
        return x
    return x.sign() * (x.abs() - lam).clamp(min=0)


def elastic_net_proximal(x: torch.Tensor, lam1: float, lam2: float) -> torch.Tensor:
    """Elastic net proximal operator.

    prox(v) = 1/(1+2*lam2) * soft_threshold(v, lam1)

    Args:
        x: Tensor
        lam1: L1 strength (>= 0)
        lam2: L2 strength (>= 0)

    Returns:
        Proximal result, same shape as x.
    """
    if lam1 < 0 or lam2 < 0:
        raise ValueError(f"`lam1` and `lam2` must be >= 0, got {lam1}, {lam2}.")
    scale = 1.0 / (1.0 + 2.0 * lam2)
    return scale * soft_threshold(x, lam1)
