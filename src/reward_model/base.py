from __future__ import annotations

import torch
import torch.nn as nn


class BaseRewardModel(nn.Module):
    """Base reward API used by the training pipeline."""

    def reset(self, dones=None):
        del dones

    @property
    def is_linear(self) -> bool:
        """Whether reward evaluation is linear in features."""
        return False

    def get_reward_from_features(
        self,
        feats: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute per-step reward from features.

        Args:
            feats:
                Features with shape [N, D] or [B, T, D].
            mask:
                Optional mask with shape [N] or [B, T].
        """
        del feats, mask
        raise NotImplementedError

    def get_regularization_loss(self) -> torch.Tensor:
        """L2 regularization penalty (for differentiable reg). No-op by default."""
        try:
            device = next(self.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
        return torch.tensor(0.0, device=device)

    def apply_proximal_step(self) -> None:
        """Apply L1/elastic net proximal operator to parameters. No-op by default."""
        pass

    def project_weights(self) -> None:
        """Project linear weights onto norm ball (linear models only). No-op by default."""
        pass
