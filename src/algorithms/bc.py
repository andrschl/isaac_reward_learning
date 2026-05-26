from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from storage.obs_action_storage import ObsActionBuffer

if TYPE_CHECKING:
    from rsl_rl.models import MLPModel


@dataclass(slots=True)
class BCCfg:
    """Behavioral cloning hyperparameters.

    Notes:
        - ``alpha`` is the convex mixing weight on the policy loss when BC is
          combined with PPO via :class:`algorithms.ppo_with_bc.PPOWithBC`.
          ``alpha=0`` disables BC; ``alpha=1`` makes the policy update
          BC-only.
        - This config holds no learning-rate / weight-decay / grad-clip
          settings: the actor optimizer is owned by PPO.
        - ``val_fraction`` carves out that fraction of expert *episodes* into a
          held-out validation set whose BC loss is logged (but never trained
          on) during validation passes. ``0`` disables the split.
    """

    alpha: float = 0.0
    loss_type: str = "nll"  # "nll" | "mse"
    batch_size: int = 256
    val_fraction: float = 0.0


class BC:
    """Compute a behavioral-cloning loss against an rsl_rl actor.

    BC is a *loss provider*, not a standalone trainer. It holds a reference to
    a shared :class:`storage.obs_action_storage.ObsActionBuffer` and exposes
    :meth:`compute_loss` so an outer optimizer (typically PPO's, via
    :class:`algorithms.ppo_with_bc.PPOWithBC`) can incorporate the BC gradient
    into its update.
    """

    def __init__(
        self,
        *,
        cfg: BCCfg,
        storage: ObsActionBuffer,
        val_storage: ObsActionBuffer | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        if cfg.loss_type not in {"nll", "mse"}:
            raise ValueError(
                f"`cfg.loss_type` must be 'nll' or 'mse', got {cfg.loss_type!r}."
            )
        if not (0.0 <= cfg.alpha <= 1.0):
            raise ValueError(f"`cfg.alpha` must be in [0, 1], got {cfg.alpha}.")
        if cfg.batch_size <= 0:
            raise ValueError(f"`cfg.batch_size` must be > 0, got {cfg.batch_size}.")

        self.cfg = cfg
        self.storage = storage
        # Held-out (obs, action) pairs for validation logging only; never
        # trained on. None when no train/val split was configured.
        self.val_storage = val_storage
        self.device = torch.device(device)

    @property
    def alpha(self) -> float:
        return self.cfg.alpha

    def _compute_loss(self, actor: MLPModel, storage: ObsActionBuffer) -> torch.Tensor:
        """Sample a minibatch from ``storage`` and return the BC loss."""
        obs_mb, expert_actions = storage.sample(self.cfg.batch_size, self.device)

        if self.cfg.loss_type == "nll":
            actor(obs_mb, stochastic_output=True)
            # GaussianDistribution.log_prob already sums over the action dim.
            log_prob = actor.get_output_log_prob(expert_actions)  # [B]
            return -log_prob.mean()

        mean_actions = actor(obs_mb)  # [B, A], deterministic
        return F.mse_loss(mean_actions, expert_actions)

    def compute_loss(self, actor: MLPModel) -> torch.Tensor:
        """Sample a training minibatch and return the BC loss.

        Args:
            actor: rsl_rl 5.x ``MLPModel`` exposing
                ``forward(obs, stochastic_output=True)`` (to populate the
                output distribution) and ``get_output_log_prob(actions)``.

        Shapes:
            sampled obs minibatch: nested dict; leaves ``[B, *leaf]``
            sampled expert actions: ``[B, A]``

        Returns:
            Scalar loss tensor (shape ``[]``). Gradients flow into
            ``actor.parameters()``; the caller owns
            ``zero_grad`` / ``backward`` / ``step``.
        """
        return self._compute_loss(actor, self.storage)

    def compute_validation_loss(self, actor: MLPModel) -> float | None:
        """Return the BC loss on the held-out validation set, or ``None``.

        ``None`` when no validation split was configured (``val_storage`` is
        empty/absent). Detached and run under ``torch.no_grad()``, so it never
        contributes a gradient. The caller is responsible for putting the actor
        in eval mode if desired.
        """
        if self.val_storage is None or len(self.val_storage) == 0:
            return None
        with torch.no_grad():
            return float(self._compute_loss(actor, self.val_storage).item())
