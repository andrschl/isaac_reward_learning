from __future__ import annotations

import os
from typing import Any, Protocol


class MetricLogger(Protocol):
    """Small scalar logger API used by the runner."""

    def start(self) -> None:
        """Initialize external logging resources."""

    def log(self, payload: dict[str, float], step: int) -> None:
        """Log scalar payload at a training step."""

    def finish(self, exit_code: int = 0) -> None:
        """Flush/close external logging resources."""


class NoopMetricLogger:
    """Metric logger that intentionally does nothing."""

    def start(self) -> None:
        return

    def log(self, payload: dict[str, float], step: int) -> None:
        del payload, step

    def finish(self, exit_code: int = 0) -> None:
        del exit_code


class WandbMetricLogger:
    """Weights & Biases scalar logger.

    ``finish`` must be called explicitly because Isaac's ``SimulationApp.close``
    exits via ``os._exit`` and can skip Python atexit handlers.
    """

    def __init__(
        self,
        *,
        project: str,
        log_dir: str | None,
        run_name: str | None = None,
        group: str | None = None,
        tags: list[str] | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.project = str(project)
        self.log_dir = log_dir
        self.run_name = run_name
        self.group = group
        self.tags = list(tags) if tags else None
        self.config = dict(config) if config else None
        self._run: Any = None

    def start(self) -> None:
        if self._run is not None:
            return
        if self.log_dir is not None:
            os.makedirs(self.log_dir, exist_ok=True)
        try:
            import wandb

            self._run = wandb.init(
                project=self.project,
                dir=self.log_dir,
                name=self.run_name,
                group=self.group,
                tags=self.tags,
                config=self.config,
                reinit=True,
                settings=wandb.Settings(console="off"),
            )
        except Exception as exc:
            print(f"[WARN] wandb.init failed: {exc}")
            self._run = None

    def log(self, payload: dict[str, float], step: int) -> None:
        if not payload:
            return
        self.start()
        if self._run is not None:
            self._run.log(payload, step=step)

    def finish(self, exit_code: int = 0) -> None:
        if self._run is None:
            return
        try:
            import wandb

            wandb.finish(exit_code=exit_code)
        except Exception as exc:
            print(f"[WARN] wandb.finish failed: {exc}")
        finally:
            self._run = None
