"""Human-readable console summaries for training/validation.

Kept separate from the runner so the training loop holds only learning logic;
formatting and stdout emission live here.
"""

from __future__ import annotations


class ConsoleReporter:
    """Print concise training and validation summaries to stdout."""

    def start(self, num_iterations: int) -> None:
        print(f"[INFO] Starting learning for {int(num_iterations)} iterations.")

    def iteration_summary(
        self,
        *,
        iteration: int,
        rl_metrics: dict[str, float],
        reward_metrics: dict[str, float],
    ) -> None:
        policy_loss = rl_metrics.get("RL/policy_loss", float("nan"))
        reward_loss = reward_metrics.get("IRL/reward_loss", float("nan"))
        feat_diff = reward_metrics.get("IRL/feature_exp_diff_norm", float("nan"))
        bc_loss = rl_metrics.get("BC/loss")
        bc_str = f"  bc_loss={bc_loss:.4f}" if bc_loss is not None else ""
        print(
            f"[INFO] iter {iteration + 1}  policy_loss={policy_loss:.4f}  "
            f"reward_loss={reward_loss:.4f}  feat_diff_norm={feat_diff:.4f}{bc_str}"
        )

    def validation_summary(
        self,
        *,
        iteration: int,
        metrics: dict[str, float],
        label: str = "",
    ) -> None:
        if not metrics:
            return
        summary = ", ".join(
            f"{key.rsplit('/', 1)[-1]}={value:.4f}" for key, value in metrics.items()
        )
        label_part = f"  {label}" if label else ""
        print(f"[INFO] validation summary iter {iteration + 1}{label_part}  {summary}")

    def validation_features(
        self,
        *,
        iteration: int,
        label: str,
        prefix: str,
        payload: dict[str, float],
    ) -> None:
        feature_items = [
            f"{key.removeprefix(prefix)}={value:.4f}"
            for key, value in payload.items()
            if key.startswith(prefix)
        ]
        if not feature_items:
            return
        print(
            f"[INFO] validation features iter {iteration + 1}  "
            f"{label}  " + ", ".join(feature_items)
        )
