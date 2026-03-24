from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from einops import rearrange

from reward_model.base import BaseRewardModel
from reward_model.projections import (
    elastic_net_proximal,
    project_onto_l1_ball,
    project_onto_l2_ball,
    soft_threshold,
)


_REGULARIZATION_OPTIONS = frozenset({"none", "l1", "l2", "elastic"})
_LINEAR_PROJECTION_OPTIONS = frozenset({"none", "l1_ball", "l2_ball"})


@dataclass(slots=True)
class RewardModelCfg:
    """
    Feature-based reward model config.

    The reward model consumes features directly (not observations/actions).

    MLP-only (when is_linear=False): hidden_dims, activation.
    Linear-only (when is_linear=True): linear_projection, linear_projection_radius.
    Both: regularization, regularization_strength, elastic_alpha.
    """
    num_features: int
    hidden_dims: tuple[int, ...] = (256, 256, 256)
    is_linear: bool = False
    activation: str = "elu"

    # Regularization (both MLP and linear)
    regularization: str = "none"  # "none" | "l1" | "l2" | "elastic"
    regularization_strength: float = 0.0
    elastic_alpha: float = 0.5  # L1 fraction when regularization=="elastic"

    # Linear-only: projection onto norm balls (applied after optimizer.step)
    linear_projection: str = "none"  # "none" | "l1_ball" | "l2_ball"
    linear_projection_radius: float = 1.0


class RewardModel(BaseRewardModel):
    """
    Reward model over features.

    Supported input shapes:
      - [N, D]     -> returns [N]
      - [B, T, D]  -> returns [B, T]

    `mask` is optional and used only for shape validation / optional zeroing.
    """

    def __init__(self, cfg: RewardModelCfg):
        super().__init__()
        self.cfg = cfg

        if cfg.num_features <= 0:
            raise ValueError(f"`num_features` must be > 0, got {cfg.num_features}")
        if cfg.regularization not in _REGULARIZATION_OPTIONS:
            raise ValueError(
                f"`regularization` must be one of {sorted(_REGULARIZATION_OPTIONS)}, "
                f"got {cfg.regularization!r}."
            )
        if cfg.linear_projection not in _LINEAR_PROJECTION_OPTIONS:
            raise ValueError(
                f"`linear_projection` must be one of {sorted(_LINEAR_PROJECTION_OPTIONS)}, "
                f"got {cfg.linear_projection!r}."
            )
        if not cfg.is_linear and cfg.linear_projection != "none":
            raise ValueError(
                "`linear_projection` is only valid when `is_linear` is True. "
                f"Got is_linear={cfg.is_linear}, linear_projection={cfg.linear_projection!r}."
            )
        if cfg.linear_projection_radius <= 0:
            raise ValueError(
                f"`linear_projection_radius` must be > 0, got {cfg.linear_projection_radius}."
            )

        self.reward = self._build_reward_network(cfg)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def _build_reward_network(self, cfg: RewardModelCfg) -> nn.Module:
        if cfg.is_linear:
            # Linear reward stays bias-free by design.
            return nn.Linear(cfg.num_features, 1, bias=False)

        hidden_dims = tuple(int(h) for h in cfg.hidden_dims)
        if len(hidden_dims) == 0:
            raise ValueError("`hidden_dims` must be non-empty for non-linear reward model.")

        layers: list[nn.Module] = []
        in_dim = cfg.num_features
        act = get_activation(cfg.activation)

        for h in hidden_dims:
            if h <= 0:
                raise ValueError(f"Hidden dims must be positive, got {hidden_dims}")
            layers.append(nn.Linear(in_dim, h))
            layers.append(act.__class__())  # fresh instance per layer
            in_dim = h

        layers.append(nn.Linear(in_dim, 1))
        return nn.Sequential(*layers)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def reset(self, dones: torch.Tensor | None = None) -> None:
        # Stateless reward model
        del dones

    @property
    def is_linear(self) -> bool:
        return bool(self.cfg.is_linear)

    def forward(self, feats: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Alias for compatibility with code calling `reward(feats, mask)`.
        """
        return self.get_reward_from_features(feats, mask)

    def get_reward_from_features(
        self,
        feats: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute per-step rewards from features.

        Args:
            feats:
                [N, D] or [B, T, D]
            mask:
                Optional boolean mask matching output shape ([N] or [B, T]).
                Used for validation and zeroing padded outputs.

        Returns:
            rewards with shape feats.shape[:-1]
        """
        if not isinstance(feats, torch.Tensor):
            feats = torch.as_tensor(feats)

        if feats.ndim not in (2, 3):
            raise ValueError(f"`feats` must be [N,D] or [B,T,D], got {tuple(feats.shape)}")

        if feats.ndim == 2:
            # feats: [N, D] -> rewards: [N]
            flat_rewards = self.reward(feats)
            rewards = rearrange(flat_rewards, "n 1 -> n")
        else:
            # feats: [B, T, D] -> rewards: [B, T]
            batch_size, traj_len, _ = feats.shape
            flat_feats = rearrange(feats, "b t d -> (b t) d")
            flat_rewards = self.reward(flat_feats)
            rewards = rearrange(flat_rewards, "(b t) 1 -> b t", b=batch_size, t=traj_len)

        if mask is not None:
            if not isinstance(mask, torch.Tensor):
                mask = torch.as_tensor(mask, device=rewards.device)
            rewards = rewards * mask.to(dtype=rewards.dtype, device=rewards.device)

        return rewards

    def get_regularization_loss(self) -> torch.Tensor:
        """L2 regularization penalty. Returns 0 when regularization is not L2."""
        cfg = self.cfg
        if cfg.regularization != "l2" or cfg.regularization_strength <= 0:
            return super().get_regularization_loss()
        total = torch.tensor(0.0, device=next(self.parameters()).device)
        for p in self.parameters():
            total = total + p.pow(2).sum()
        return cfg.regularization_strength * total

    def apply_proximal_step(self) -> None:
        """Apply L1 or elastic net proximal operator to all parameters."""
        cfg = self.cfg
        if cfg.regularization not in ("l1", "elastic") or cfg.regularization_strength <= 0:
            return
        strength = float(cfg.regularization_strength)
        if cfg.regularization == "l1":
            lam1, lam2 = strength, 0.0
        else:
            alpha = float(cfg.elastic_alpha)
            lam1 = alpha * strength
            lam2 = (1.0 - alpha) * strength
        with torch.no_grad():
            for p in self.parameters():
                if cfg.regularization == "l1":
                    p.data.copy_(soft_threshold(p.data, lam1))
                else:
                    p.data.copy_(elastic_net_proximal(p.data, lam1, lam2))

    def project_weights(self) -> None:
        """Project linear layer weight onto L1 or L2 ball. No-op for MLP."""
        cfg = self.cfg
        if not cfg.is_linear or cfg.linear_projection == "none":
            return
        radius = float(cfg.linear_projection_radius)
        linear_layer = self.reward
        if not isinstance(linear_layer, nn.Linear):
            return
        w = linear_layer.weight.data
        with torch.no_grad():
            if cfg.linear_projection == "l2_ball":
                projected = project_onto_l2_ball(w, radius)
            else:
                projected = project_onto_l1_ball(w, radius)
            linear_layer.weight.data.copy_(projected)

    @staticmethod
    def init_weights(sequential: nn.Sequential, scales: list[float]) -> None:
        """
        Optional helper (not automatically used).
        Applies orthogonal init to linear layers.
        """
        linear_layers = [m for m in sequential if isinstance(m, nn.Linear)]
        if len(scales) != len(linear_layers):
            raise ValueError(
                f"`scales` length ({len(scales)}) must match number of linear layers ({len(linear_layers)})."
            )
        for layer, gain in zip(linear_layers, scales):
            nn.init.orthogonal_(layer.weight, gain=gain)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)


def get_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "elu":
        return nn.ELU()
    if name == "selu":
        return nn.SELU()
    if name == "relu":
        return nn.ReLU()
    if name == "lrelu":
        return nn.LeakyReLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "sigmoid":
        return nn.Sigmoid()
    raise ValueError(f"Invalid activation function: {name}")
