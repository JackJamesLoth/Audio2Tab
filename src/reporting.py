"""Progress reporting, callbacks, and experiment tracking."""

import argparse
import os
import warnings
from typing import Any, Dict, List, Optional, Sequence


def emit_verbose_warning(verbose: bool, message: str) -> None:
    if verbose:
        warnings.warn(message)


def progress_iterable(items: Sequence[Any], description: str, total: Optional[int] = None) -> Sequence[Any]:
    try:
        from tqdm.auto import tqdm

        return tqdm(items, desc=description, total=total)
    except ImportError:
        return items


def build_loss_progress_callback(trainer_callback_cls: Any) -> Any:
    class LossProgressCallback(trainer_callback_cls):
        def __init__(self) -> None:
            self.progress_bar = None
            self.latest_train_loss: Optional[float] = None
            self.latest_val_loss: Optional[float] = None
            self.latest_step: int = 0
            self.latest_epoch: Optional[float] = None

        def _postfix(self) -> Dict[str, str]:
            return {
                "step": str(self.latest_step),
                "epoch": "NA" if self.latest_epoch is None else f"{self.latest_epoch:.2f}",
                "train_loss": "NA" if self.latest_train_loss is None else f"{self.latest_train_loss:.4f}",
                "eval_loss": "NA" if self.latest_val_loss is None else f"{self.latest_val_loss:.4f}",
            }

        def _refresh(self) -> None:
            if self.progress_bar is not None:
                self.progress_bar.set_postfix(self._postfix(), refresh=False)

        def on_train_begin(self, args, state, control, **kwargs):
            try:
                from tqdm.auto import tqdm

                total = getattr(state, "max_steps", None)
                if total is not None and total > 0:
                    self.progress_bar = tqdm(total=total, desc="Training", dynamic_ncols=True)
                    self._refresh()
            except ImportError:
                self.progress_bar = None
            return control

        def on_step_end(self, args, state, control, **kwargs):
            self.latest_step = int(getattr(state, "global_step", 0))
            epoch = getattr(state, "epoch", None)
            self.latest_epoch = None if epoch is None else float(epoch)
            if self.progress_bar is not None:
                self.progress_bar.n = self.latest_step
                self._refresh()
                self.progress_bar.refresh()
            return control

        def on_log(self, args, state, control, logs=None, **kwargs):
            self.latest_step = int(getattr(state, "global_step", 0))
            epoch = getattr(state, "epoch", None)
            self.latest_epoch = None if epoch is None else float(epoch)
            if logs:
                if "loss" in logs and logs["loss"] is not None:
                    self.latest_train_loss = float(logs["loss"])
                if "eval_loss" in logs and logs["eval_loss"] is not None:
                    self.latest_val_loss = float(logs["eval_loss"])
                self._refresh()
            return control

        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            self.latest_step = int(getattr(state, "global_step", 0))
            epoch = getattr(state, "epoch", None)
            self.latest_epoch = None if epoch is None else float(epoch)
            if metrics and metrics.get("eval_loss") is not None:
                self.latest_val_loss = float(metrics["eval_loss"])
                self._refresh()
            return control

        def on_train_end(self, args, state, control, **kwargs):
            self.latest_step = int(getattr(state, "global_step", 0))
            epoch = getattr(state, "epoch", None)
            self.latest_epoch = None if epoch is None else float(epoch)
            if self.progress_bar is not None:
                self.progress_bar.n = self.latest_step
                self._refresh()
                self.progress_bar.close()
                self.progress_bar = None
            return control

    return LossProgressCallback()


def resolve_report_to_targets(args: argparse.Namespace) -> List[str]:
    targets: List[str] = []
    seen = set()
    for target in args.report_to or []:
        normalized = str(target).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        targets.append(normalized)

    if args.use_wandb and "wandb" not in seen:
        targets.append("wandb")
    return targets


def ensure_wandb_available(args: argparse.Namespace, report_to: Sequence[str]) -> None:
    if not args.use_wandb and "wandb" not in report_to:
        return
    try:
        import wandb  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Weights & Biases logging was requested, but the 'wandb' package is not installed in this environment."
        ) from exc


def set_env_if_cli_provided(env_name: str, value: Optional[Any]) -> None:
    if value is None:
        return
    if isinstance(value, (list, tuple)):
        os.environ[env_name] = ",".join(str(item) for item in value if str(item).strip())
        return
    os.environ[env_name] = str(value)


def clear_wandb_environment() -> None:
    for env_name in (
        "WANDB_PROJECT",
        "WANDB_ENTITY",
        "WANDB_NAME",
        "WANDB_RUN_GROUP",
        "WANDB_JOB_TYPE",
        "WANDB_TAGS",
        "WANDB_MODE",
        "WANDB_DIR",
    ):
        os.environ.pop(env_name, None)
