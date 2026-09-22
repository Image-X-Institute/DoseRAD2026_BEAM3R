from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import os
import random
import re
import sys
import time
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Sequence

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
try:
    from monai.data import Dataset
except ImportError:
    from torch.utils.data import Dataset
from tqdm import tqdm

from beam_idd_validation import CtBeamIddAccumulator


def validate_stats_dict(stats: dict[str, Any]) -> None:
    """Require keys produced by ``compute_train_h5_stats`` / photon-style JSON."""
    for k in ("ct_min", "ct_max", "dose_scale"):
        if k not in stats:
            raise KeyError(f"stats JSON must contain '{k}'")
    if float(stats["dose_scale"]) <= 0:
        raise ValueError("stats['dose_scale'] must be positive")
    if float(stats["ct_max"]) <= float(stats["ct_min"]):
        raise ValueError("stats['ct_max'] must be greater than stats['ct_min']")


def load_stats_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as fh:
        return json.load(fh)


def save_stats_json(stats: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(stats, fh, indent=2)


@dataclass(frozen=True)
class TrainAmpConfig:
    """CUDA AMP settings for training loops (autocast + optional GradScaler)."""

    enabled: bool
    autocast_dtype: torch.dtype

    @property
    def uses_scaler(self) -> bool:
        return self.enabled and self.autocast_dtype == torch.float16


def add_amp_args(parser) -> None:
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Enable automatic mixed precision on CUDA (torch.amp.autocast + GradScaler for float16).",
    )
    parser.add_argument(
        "--amp-dtype",
        choices=("float16", "bfloat16"),
        default="float16",
        help="Autocast dtype when --amp is set (default: float16). bfloat16 often needs no GradScaler.",
    )


def add_scheduler_args(parser) -> None:
    parser.add_argument(
        "--scheduler",
        type=str,
        default=None,
        choices=["cosineannealing"],
        help="Learning rate scheduler (default: None, uses ReduceLROnPlateau).",
    )


SCHEDULEFREE_OPTIMIZER_NAME = "schedulefree_adamw"
_XLSTM_PARAM_PREFIXES = ("xlstm.", "xlstm2.", "xlstm3.")
_SCHEDULEFREE_OPTIMIZER_CLASS_NAMES = frozenset(
    {
        "AdamWScheduleFree",
        "AdamWScheduleFreeReference",
        "AdamWScheduleFreePaper",
    }
)


def add_optimizer_args(parser) -> None:
    """Shared optimizer CLI flags for DL-segment trainers."""
    parser.add_argument(
        "--optimizer",
        default=None,
        choices=("adam", SCHEDULEFREE_OPTIMIZER_NAME),
        help=(
            "Optimizer (default: adam). "
            f"{SCHEDULEFREE_OPTIMIZER_NAME} disables the LR scheduler."
        ),
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="AdamW / Schedule-Free weight decay (default: 0).",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Schedule-Free linear LR warmup in optimizer steps. "
            "For CNN-Mamba, --warmup-epochs is converted to steps when Schedule-Free is used."
        ),
    )


def optimizer_choice_from_args(args: Any) -> str:
    return getattr(args, "optimizer", None) or "adam"


def uses_schedule_free_optimizer(args: Any) -> bool:
    return optimizer_choice_from_args(args) == SCHEDULEFREE_OPTIMIZER_NAME


def config_uses_schedule_free(cfg: dict[str, Any]) -> bool:
    opt = cfg.get("optimizer")
    if opt == SCHEDULEFREE_OPTIMIZER_NAME:
        return True
    argv = cfg.get("argv") or []
    for i, token in enumerate(argv):
        if token == "--optimizer" and i + 1 < len(argv):
            return argv[i + 1] == SCHEDULEFREE_OPTIMIZER_NAME
    return False


def is_schedule_free_optimizer(optimizer: torch.optim.Optimizer) -> bool:
    return optimizer.__class__.__name__ in _SCHEDULEFREE_OPTIMIZER_CLASS_NAMES


def set_optimizer_train_mode(optimizer: torch.optim.Optimizer | None) -> None:
    if optimizer is not None and hasattr(optimizer, "train"):
        optimizer.train()


def set_optimizer_eval_mode(optimizer: torch.optim.Optimizer | None) -> None:
    if optimizer is not None and hasattr(optimizer, "eval"):
        optimizer.eval()


@contextlib.contextmanager
def optimizer_eval_weights_context(optimizer: torch.optim.Optimizer | None):
    """Switch Schedule-Free optimizers to eval weights before checkpoint export."""
    if optimizer is not None and is_schedule_free_optimizer(optimizer):
        optimizer.eval()
    try:
        yield
    finally:
        if optimizer is not None and is_schedule_free_optimizer(optimizer):
            optimizer.train()


def get_optimizer_lr_for_logging(optimizer: torch.optim.Optimizer) -> float:
    pg = optimizer.param_groups[0]
    return float(pg.get("scheduled_lr", pg["lr"]))


def format_optimizer_lrs(optimizer: torch.optim.Optimizer) -> str:
    if len(optimizer.param_groups) == 1:
        return f"{get_optimizer_lr_for_logging(optimizer):.2e}"
    base_lr = get_optimizer_lr_for_logging(optimizer)
    xlstm_pg = optimizer.param_groups[1]
    xlstm_lr = float(xlstm_pg.get("scheduled_lr", xlstm_pg["lr"]))
    return f"{base_lr:.2e} (xLSTM {xlstm_lr:.2e})"


def resolve_schedulefree_warmup_steps(args: Any, steps_per_epoch: int) -> int:
    warmup_steps = int(getattr(args, "warmup_steps", 0) or 0)
    if warmup_steps > 0:
        return warmup_steps
    warmup_epochs = int(getattr(args, "warmup_epochs", 0) or 0)
    if warmup_epochs > 0 and uses_schedule_free_optimizer(args):
        return warmup_epochs * int(steps_per_epoch)
    return 0


def build_xlstm_param_groups(
    model: nn.Module,
    base_lr: float,
    xlstm_lr: float,
) -> list[dict[str, Any]]:
    xlstm_params: list[torch.nn.Parameter] = []
    other_params: list[torch.nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith(_XLSTM_PARAM_PREFIXES):
            xlstm_params.append(param)
        else:
            other_params.append(param)
    if not xlstm_params:
        raise ValueError(
            "--xlstm-lr set but no xLSTM parameters found "
            "(expected modules xlstm, xlstm2, or xlstm3)."
        )
    return [
        {"params": other_params, "lr": base_lr},
        {"params": xlstm_params, "lr": xlstm_lr},
    ]


def build_training_optimizer(
    model: nn.Module,
    args: Any,
    *,
    param_groups: list[dict[str, Any]] | None = None,
    warmup_steps: int | None = None,
) -> torch.optim.Optimizer:
    """Build Adam (default) or Schedule-Free AdamW with optional param groups."""
    opt_name = optimizer_choice_from_args(args)
    lr = float(args.lr)
    weight_decay = float(getattr(args, "weight_decay", 0.0) or 0.0)
    if warmup_steps is None:
        warmup_steps = int(getattr(args, "warmup_steps", 0) or 0)

    params: Any
    if param_groups is not None:
        params = param_groups
    else:
        params = model.parameters()

    if opt_name == SCHEDULEFREE_OPTIMIZER_NAME:
        import schedulefree

        return schedulefree.AdamWScheduleFree(
            params,
            lr=lr,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
        )

    if opt_name != "adam":
        raise ValueError(f"Unknown optimizer: {opt_name!r}")

    if isinstance(params, list):
        return torch.optim.Adam(params)
    return torch.optim.Adam(model.parameters(), lr=lr)


def create_lr_scheduler(optimizer, args: Any):
    """Build LR scheduler: ReduceLROnPlateau (default) or CosineAnnealingLR."""
    if uses_schedule_free_optimizer(args):
        return None
    if getattr(args, "scheduler", None) == "cosineannealing":
        return CosineAnnealingLR(optimizer, T_max=int(args.epochs))
    return ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=int(args.lr_patience),
    )


def step_lr_scheduler(scheduler, val_loss: float) -> None:
    """Step training LR scheduler (plateau needs val_loss; cosine does not)."""
    if scheduler is None:
        return
    if isinstance(scheduler, CosineAnnealingLR):
        scheduler.step()
    else:
        scheduler.step(val_loss)


def add_training_performance_args(parser) -> None:
    """Opt-in speed experiments shared by DL-segment trainers."""
    parser.add_argument(
        "--compile-model",
        action="store_true",
        help="Wrap the model with torch.compile after device/memory-format setup.",
    )
    parser.add_argument(
        "--compile-mode",
        choices=("default", "reduce-overhead", "max-autotune"),
        default="default",
        help="torch.compile mode when --compile-model is enabled (default: default).",
    )
    parser.add_argument(
        "--compile-fullgraph",
        action="store_true",
        help="Require torch.compile to capture a full graph instead of allowing graph breaks.",
    )
    parser.add_argument(
        "--channels-last",
        action="store_true",
        help="Use channels-last memory format for 2-D encoder/decoder conv paths.",
    )
    parser.add_argument(
        "--channels-last-3d",
        action="store_true",
        help="Use channels-last-3d memory format for C3D conv paths.",
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=None,
        metavar="MAX_NORM",
        help=(
            "Clip gradient L2 norm to this value before each optimizer step "
            "(e.g. 1.0). Disabled by default. Unscales GradScaler before "
            "clipping when --amp float16 is active."
        ),
    )


def maybe_compile_model(model: nn.Module, args: Any) -> nn.Module:
    """Apply torch.compile when requested, leaving default training behavior unchanged."""
    if not bool(getattr(args, "compile_model", False)):
        return model
    if not hasattr(torch, "compile"):
        raise RuntimeError("--compile-model requires a PyTorch build with torch.compile")
    return torch.compile(
        model,
        mode=str(getattr(args, "compile_mode", "default")),
        fullgraph=bool(getattr(args, "compile_fullgraph", False)),
    )


def unwrap_compiled_model(model: nn.Module) -> nn.Module:
    """Return the original module behind torch.compile wrappers when present."""
    return getattr(model, "_orig_mod", model)


def resolve_train_amp(
    device: torch.device,
    *,
    use_amp: bool,
    amp_dtype: str = "float16",
) -> tuple[TrainAmpConfig, torch.amp.GradScaler | None]:
    """Build AMP config and GradScaler (fp16 only; bf16 typically runs without scaler)."""
    if not use_amp or device.type != "cuda":
        return TrainAmpConfig(enabled=False, autocast_dtype=torch.float16), None

    dtype = torch.bfloat16 if amp_dtype == "bfloat16" else torch.float16
    config = TrainAmpConfig(enabled=True, autocast_dtype=dtype)
    scaler = torch.amp.GradScaler("cuda") if config.uses_scaler else None
    return config, scaler


@contextlib.contextmanager
def train_autocast(amp: TrainAmpConfig):
    if amp.enabled:
        with torch.amp.autocast("cuda", dtype=amp.autocast_dtype):
            yield
    else:
        yield


def amp_backward_step(
    loss: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    *,
    parameters=None,
    max_grad_norm: float | None = None,
) -> None:
    if scaler is not None:
        scaler.scale(loss).backward()
        if max_grad_norm is not None and parameters is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        if max_grad_norm is not None and parameters is not None:
            torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm)
        optimizer.step()


def epoch_batch_progress(
    dataloader,
    desc: str,
    *,
    leave: bool = False,
):
    """Progress bar over batches within one train/val epoch."""
    return tqdm(
        dataloader,
        desc=desc,
        unit="batch",
        leave=leave,
        dynamic_ncols=True,
    )


TRAINING_STATE_JSON = "training_state.json"
LAST_CHECKPOINT_NAME = "last.pth"
_EPOCH_CKPT_RE = re.compile(r"^epoch_(\d+)\.pth$")


def add_resume_args(parser) -> None:
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume training from out-dir/model/: load last.pth (model + optimizer + "
            "scheduler + scaler) or the latest epoch_*.pth weights if last.pth is missing."
        ),
    )


def _epoch_number_from_checkpoint(path: Path) -> int | None:
    m = _EPOCH_CKPT_RE.match(path.name)
    return int(m.group(1)) if m else None


def _find_latest_epoch_checkpoint(model_dir: Path) -> tuple[Path, int] | None:
    """Return (path, 1-based epoch index from filename) for the latest epoch_*.pth."""
    best: tuple[int, Path] | None = None
    for p in model_dir.glob("epoch_*.pth"):
        n = _epoch_number_from_checkpoint(p)
        if n is None:
            continue
        if best is None or n > best[0]:
            best = (n, p)
    if best is None:
        return None
    return best[1], best[0]


def _write_training_state_json(
    model_dir: Path,
    *,
    next_epoch: int,
    best_val_loss: float,
    last_completed_epoch: int,
) -> None:
    state = {
        "next_epoch": next_epoch,
        "best_val_loss": best_val_loss,
        "last_completed_epoch": last_completed_epoch,
    }
    with (model_dir / TRAINING_STATE_JSON).open("w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2)


def _load_training_state_json(model_dir: Path) -> dict[str, Any] | None:
    path = model_dir / TRAINING_STATE_JSON
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def seed_dataloader_worker(worker_id: int) -> None:
    """Re-seed Python/NumPy RNGs per DataLoader worker for reproducibility.

    ``torch.initial_seed()`` already differs per worker (derived from the
    generator seed), so deriving the other RNGs from it keeps workers
    decorrelated yet deterministic across runs with the same training seed.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def seed_everything(seed: int) -> torch.Generator:
    """Seed Python, NumPy and torch RNGs and enable deterministic algorithms.

    Returns a seeded ``torch.Generator`` to hand to the training DataLoader so
    that shuffling order is reproducible. Determinism is best-effort: some CUDA
    ops have no deterministic implementation, so ``warn_only=True`` is used to
    avoid hard failures while still flagging nondeterministic kernels.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Required for deterministic cuBLAS GEMMs under use_deterministic_algorithms.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def load_init_from_checkpoint(
    model: nn.Module,
    init_from: str | Path,
    device: torch.device,
    *,
    allow_missing_prefixes: Sequence[str] = (),
) -> list[str]:
    """Warm-start ``model`` from a parent checkpoint, tolerating added modules.

    Unlike ``--resume`` this loads weights only -- no optimizer, scheduler or
    epoch counter -- so a fine-tune starts from the parent's parameters with a
    fresh optimizer state.

    Newly added parameters are permitted only when their names start with one of
    ``allow_missing_prefixes``; anything else missing, and any unexpected key,
    means the checkpoint does not match this architecture and is raised rather
    than silently ignored.
    """
    path = Path(init_from)
    if not path.is_file():
        raise FileNotFoundError(f"--init-from checkpoint not found: {path}")

    ckpt = torch.load(path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    incompatible = model.load_state_dict(state_dict, strict=False)

    if incompatible.unexpected_keys:
        raise RuntimeError(
            f"--init-from {path}: checkpoint has {len(incompatible.unexpected_keys)} "
            f"key(s) this model does not define, e.g. "
            f"{sorted(incompatible.unexpected_keys)[:5]}"
        )
    unexplained = [
        key
        for key in incompatible.missing_keys
        if not any(key.startswith(prefix) for prefix in allow_missing_prefixes)
    ]
    if unexplained:
        raise RuntimeError(
            f"--init-from {path}: {len(unexplained)} parameter(s) are missing from the "
            f"checkpoint and are not accounted for by a newly enabled feature, e.g. "
            f"{sorted(unexplained)[:5]}"
        )
    model.to(device)
    return sorted(incompatible.missing_keys)


def _load_state_dict_into_model(
    model: nn.Module, state_dict: dict[str, Any], device: torch.device
) -> None:
    model.load_state_dict(state_dict)
    model.to(device)


def try_resume_training(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any | None,
    scaler: torch.amp.GradScaler | None,
    model_dir: str | Path,
    device: torch.device,
    *,
    resume: bool,
) -> tuple[int, float, list[str]]:
    """
    If ``resume`` is True, restore from ``last.pth`` or latest ``epoch_*.pth``.

    Returns ``(start_epoch, best_val_loss, messages)`` where ``start_epoch`` is the
    0-based index of the next epoch to run (number of epochs already completed).
    """
    messages: list[str] = []
    if not resume:
        return 0, float("inf"), messages

    model_dir = Path(model_dir)
    if not model_dir.is_dir():
        messages.append(f"--resume: no model dir at {model_dir}; starting from epoch 1.")
        return 0, float("inf"), messages

    last_path = model_dir / LAST_CHECKPOINT_NAME
    if last_path.is_file():
        ckpt = torch.load(last_path, map_location=device, weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)
        _load_state_dict_into_model(model, state_dict, device)
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            set_optimizer_train_mode(optimizer)
        if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if scaler is not None and ckpt.get("scaler_state_dict") is not None:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = int(ckpt.get("next_epoch", 0))
        best_val_loss = float(ckpt.get("best_val_loss", float("inf")))
        messages.append(
            f"Resumed from {last_path} (next epoch {start_epoch + 1}, "
            f"best_val_loss={best_val_loss:.6f})"
        )
        return start_epoch, best_val_loss, messages

    latest = _find_latest_epoch_checkpoint(model_dir)
    if latest is not None:
        ckpt_path, epoch_num = latest
        state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
        _load_state_dict_into_model(model, state_dict, device)
        start_epoch = epoch_num
        state = _load_training_state_json(model_dir)
        best_val_loss = float(state["best_val_loss"]) if state else float("inf")
        messages.append(
            f"Resumed weights from {ckpt_path} (epochs 1–{epoch_num} done; "
            f"continuing at epoch {start_epoch + 1}). Optimizer/scheduler not restored "
            f"(no {LAST_CHECKPOINT_NAME})."
        )
        if state is None:
            messages.append(
                f"Note: {TRAINING_STATE_JSON} missing; best_val_loss reset to inf "
                "(best_model.pth may be overwritten)."
            )
        return start_epoch, best_val_loss, messages

    messages.append(f"--resume: no checkpoints in {model_dir}; starting from epoch 1.")
    return 0, float("inf"), messages


def save_training_checkpoints(
    model: nn.Module,
    model_dir: str | Path,
    epoch_index: int,
    val_loss: float,
    best_val_loss: float,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> tuple[float, list[str]]:
    """Save ``epoch_XXXX.pth`` every epoch; update ``best_model.pth`` on val improvement."""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    state_model = unwrap_compiled_model(model)
    with optimizer_eval_weights_context(optimizer):
        state_dict = state_model.state_dict()
    completed = epoch_index + 1
    epoch_path = model_dir / f"epoch_{completed:04d}.pth"
    torch.save(state_dict, epoch_path)
    messages = [f"Saved {epoch_path.name}"]

    if val_loss < best_val_loss:
        torch.save(state_dict, model_dir / "best_model.pth")
        messages.append("Updated best_model.pth")
        best_val_loss = val_loss

    next_epoch = epoch_index + 1
    _write_training_state_json(
        model_dir,
        next_epoch=next_epoch,
        best_val_loss=best_val_loss,
        last_completed_epoch=completed,
    )

    if optimizer is not None:
        last_payload: dict[str, Any] = {
            "next_epoch": next_epoch,
            "best_val_loss": best_val_loss,
            "last_completed_epoch": completed,
            "model_state_dict": state_dict,
            "optimizer_state_dict": optimizer.state_dict(),
        }
        if scheduler is not None:
            last_payload["scheduler_state_dict"] = scheduler.state_dict()
        if scaler is not None:
            last_payload["scaler_state_dict"] = scaler.state_dict()
        last_path = model_dir / LAST_CHECKPOINT_NAME
        torch.save(last_payload, last_path)
        messages.append(f"Saved {LAST_CHECKPOINT_NAME}")

    return best_val_loss, messages


LOSS_HISTORY_JSON = "loss_history.json"
LOSS_PLOT_FILENAME = "loss_curve.png"


@dataclass
class TrainingLossHistory:
    """Per-epoch train/val loss series persisted under the experiment out-dir."""

    epochs: list[int] = field(default_factory=list)
    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    learning_rates: list[float] = field(default_factory=list)

    def record(
        self,
        epoch_1based: int,
        train_loss: float,
        val_loss: float,
        *,
        lr: float | None = None,
    ) -> None:
        if self.epochs and epoch_1based <= self.epochs[-1]:
            idx = self.epochs.index(epoch_1based)
            self.train_losses[idx] = float(train_loss)
            self.val_losses[idx] = float(val_loss)
            if lr is not None:
                if idx < len(self.learning_rates):
                    self.learning_rates[idx] = float(lr)
                else:
                    self.learning_rates.append(float(lr))
            return
        self.epochs.append(int(epoch_1based))
        self.train_losses.append(float(train_loss))
        self.val_losses.append(float(val_loss))
        if lr is not None:
            self.learning_rates.append(float(lr))

    def to_dict(self) -> dict[str, Any]:
        return {
            "epochs": self.epochs,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "learning_rates": self.learning_rates,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainingLossHistory:
        return cls(
            epochs=[int(x) for x in data.get("epochs", [])],
            train_losses=[float(x) for x in data.get("train_losses", [])],
            val_losses=[float(x) for x in data.get("val_losses", [])],
            learning_rates=[float(x) for x in data.get("learning_rates", [])],
        )

    def save(self, out_dir: str | Path) -> Path:
        path = Path(out_dir) / LOSS_HISTORY_JSON
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)
        return path

    @classmethod
    def load(cls, out_dir: str | Path) -> TrainingLossHistory:
        path = Path(out_dir) / LOSS_HISTORY_JSON
        if not path.is_file():
            return cls()
        with path.open(encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))


def plot_training_losses(
    out_dir: str | Path,
    history: TrainingLossHistory,
) -> Path:
    """Write ``loss_curve.png`` (train + val loss vs epoch) to ``out_dir``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_path = out_dir / LOSS_PLOT_FILENAME

    if not history.epochs:
        return plot_path

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(
        history.epochs,
        history.train_losses,
        label="train",
        color="#1f77b4",
        linewidth=1.5,
        marker="o",
        markersize=3,
    )
    ax.plot(
        history.epochs,
        history.val_losses,
        label="val",
        color="#ff7f0e",
        linewidth=1.5,
        marker="o",
        markersize=3,
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and validation loss")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    return plot_path


def update_training_loss_plot(
    out_dir: str | Path,
    history: TrainingLossHistory,
    *,
    epoch_1based: int,
    train_loss: float,
    val_loss: float,
    lr: float | None = None,
) -> Path:
    """Append epoch losses, save ``loss_history.json``, and refresh ``loss_curve.png``."""
    history.record(epoch_1based, train_loss, val_loss, lr=lr)
    history.save(out_dir)
    return plot_training_losses(out_dir, history)


VAL_METRICS_HISTORY_JSON = "val_metrics_history.json"
BEV_TENSORBOARD_METRICS = ("masked_mae", "idd_distance", "idd_distance_floored", "mse")
CT_TENSORBOARD_METRICS = ("ct_masked_mae", "ct_mse")
CT_ROUNDTRIP_BEV_METRICS = (
    "bev_idd_from_ct",
    "bev_idd_from_ct_positive_clip",
)


def add_tensorboard_args(parser) -> None:
    parser.add_argument(
        "--tensorboard",
        action="store_true",
        help=(
            "Log fast BEV validation metrics (masked MAE, RMS IDD, MSE) to TensorBoard. "
            "With --ct-space-loss, also log CT-space masked MAE and MSE."
        ),
    )
    parser.add_argument(
        "--val-metrics-freq",
        type=int,
        default=1,
        help="Compute BEV validation metrics every N epochs when --tensorboard is set (default: 1).",
    )
    parser.add_argument(
        "--val-beam-idd",
        action="store_true",
        help=(
            "Also log the challenge beam-direction IDD curve distance per control "
            "point, averaged per patient, on --val-metrics-freq epochs. Needs the "
            "plan JSONs for the gantry angles, so --baseline-pb-dir must be set. "
            "This is the evaluator's metric, unlike the fixed-axis IDD in "
            "beam_level_metrics.py, which integrates the wrong axis for a coplanar "
            "beam (default: off)."
        ),
    )
    parser.add_argument(
        "--idd-floor-frac",
        type=float,
        default=0.005,
        help=(
            "Fraction of each BEV volume's own GT max used to floor both "
            "prediction and target (symmetrically) before computing the "
            "'idd_distance_floored' TensorBoard metric. Matches the fallback "
            "cutoff scripts/postprocess_eval.py uses for beams with no real "
            "submitted minimum_cutoff to read (default: 0.005)."
        ),
    )


def validate_val_metrics_args(args: Any) -> None:
    """Validate the beam IDD option while leaving its default path disabled."""
    if not bool(getattr(args, "val_beam_idd", False)):
        return
    if not bool(getattr(args, "tensorboard", False)):
        raise ValueError("--val-beam-idd requires --tensorboard")
    if int(getattr(args, "val_metrics_freq", 0)) <= 0:
        raise ValueError("--val-beam-idd requires --val-metrics-freq greater than 0")
    if not bool(getattr(args, "ct_space_loss", False)):
        # the metric is scored on the CT-space ROIs, which only that path builds
        raise ValueError("--val-beam-idd requires --ct-space-loss")
    if not val_beam_idd_plan_root(args):
        raise ValueError(
            "--val-beam-idd needs the plan JSONs that carry the beam geometry: "
            "set --data-dir (proton) or --baseline-pb-dir (photon)"
        )


def _dota_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _import_validation_metrics():
    repo_root = _dota_repo_root()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from validation import ValidationMetrics

    return ValidationMetrics


def default_bev_forward(
    model: nn.Module,
    batch,
    device: torch.device,
    *,
    materializer: Any | None = None,
    amp: TrainAmpConfig | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward pass for 2-D DL-segment models (CNN-Mamba, ConvLSTM, xLSTM, Attention1D)."""
    nb = device.type == "cuda"
    amp = amp or TrainAmpConfig(enabled=False, autocast_dtype=torch.float16)
    if materializer is not None:
        ct, proj, label = materializer.materialize(batch)
    else:
        ct = batch[0][0].to(device, non_blocking=nb)
        proj = batch[0][1].to(device, non_blocking=nb)
        label = batch[1].to(device, non_blocking=nb)
    with train_autocast(amp):
        output = model(ct, proj)
    return output, label


def bev_batch_meta_from_loader_batch(batch: Any, batch_size: int) -> dict[str, Any]:
    """Build the metadata dict expected by ``ValidationMetrics.update_batch``."""
    if isinstance(batch, dict):
        return batch
    return {
        "patient_id": ["val"] * batch_size,
        "beam_id": torch.zeros(batch_size, dtype=torch.long),
        "cp_id": torch.arange(batch_size, dtype=torch.long),
    }


def bev_metric_means_from_summary(summary: dict[str, dict[str, float]]) -> dict[str, float]:
    return {
        name: float(summary[name]["mean"])
        for name in BEV_TENSORBOARD_METRICS
        if name in summary and summary[name]["mean"] == summary[name]["mean"]
    }


class BevMetricsAccumulator:
    """Accumulate fast BEV and optional CT-space metrics during validation."""

    def __init__(
        self,
        device: torch.device,
        *,
        catalog: Any | None = None,
        gantry_lookup: dict[tuple[str, int, int], float] | None = None,
        idd_floor_frac: float = 0.005,
    ) -> None:
        ValidationMetrics = _import_validation_metrics()
        self._metrics = ValidationMetrics(
            metrics_mode="fast",
            compute_gradient_mae=False,
            use_gpu_metrics=device.type == "cuda",
            idd_floor_frac=idd_floor_frac,
        )
        self._bev_updates = 0
        self._ct_values: dict[str, list[torch.Tensor]] = {
            name: [] for name in (*CT_TENSORBOARD_METRICS, *CT_ROUNDTRIP_BEV_METRICS)
        }
        self._beam_idd = (
            CtBeamIddAccumulator(catalog, gantry_lookup,
                                 idd_floor_frac=idd_floor_frac)
            if catalog is not None and gantry_lookup is not None
            else None
        )

    def update(self, output: torch.Tensor, label: torch.Tensor, batch: Any) -> None:
        pred_bev = output.unsqueeze(1) if output.dim() == 4 else output
        target_bev = label.unsqueeze(1) if label.dim() == 4 else label
        if (
            pred_bev.ndim == 5
            and target_bev.ndim == 5
            and pred_bev.shape[:2] == target_bev.shape[:2]
            and pred_bev.shape[2] == 2 * target_bev.shape[2]
            and pred_bev.shape[3] == 2 * target_bev.shape[3]
            and pred_bev.shape[4] == target_bev.shape[4]
        ):
            # PixelShuffleDepthHeight predicts dose on a grid with half the voxel
            # spacing in D/H. Average each 2x2 physical cell back to the coarse
            # target grid before calculating BEV metrics.
            pred_bev = torch.nn.functional.avg_pool3d(
                pred_bev, kernel_size=(2, 2, 1), stride=(2, 2, 1)
            )
        batch_meta = bev_batch_meta_from_loader_batch(batch, int(pred_bev.shape[0]))
        self._metrics.update_batch(pred_bev, target_bev, batch_meta)
        self._bev_updates += 1

    def update_ct(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        valid: torch.Tensor,
        *,
        threshold: float = 0.10,
    ) -> None:
        """Accumulate normalized masked MAE and valid-voxel MSE in CT space."""
        if prediction.shape != target.shape or prediction.shape != valid.shape:
            raise ValueError(
                "CT metric tensors must have matching shapes, got "
                f"prediction={tuple(prediction.shape)}, target={tuple(target.shape)}, "
                f"valid={tuple(valid.shape)}"
            )
        if prediction.ndim != 4:
            raise ValueError(
                f"expected CT metric tensors (B,D,H,W), got {tuple(prediction.shape)}"
            )

        pred = prediction.detach().float()
        tgt = target.detach().float()
        mask = valid.detach().bool()
        reduce_dims = (1, 2, 3)

        valid_count = mask.sum(dim=reduce_dims)
        squared_error = (pred - tgt).square()
        mse = (
            (squared_error * mask.to(squared_error.dtype)).sum(dim=reduce_dims)
            / valid_count.clamp_min(1)
        )
        mse = torch.where(
            valid_count > 0, mse, torch.full_like(mse, float("nan"))
        )

        neg_inf = torch.full_like(tgt, float("-inf"))
        max_gt = torch.where(mask, tgt, neg_inf).amax(dim=reduce_dims)
        high_dose_mask = mask & (
            tgt >= threshold * max_gt.view(-1, 1, 1, 1)
        )
        high_dose_count = high_dose_mask.sum(dim=reduce_dims)
        absolute_error = (pred - tgt).abs()
        masked_mae = (
            (
                absolute_error
                * high_dose_mask.to(absolute_error.dtype)
            ).sum(dim=reduce_dims)
            / high_dose_count.clamp_min(1)
        ) / max_gt.clamp_min(torch.finfo(tgt.dtype).eps)
        masked_mae_valid = (valid_count > 0) & (high_dose_count > 0) & (max_gt > 0)
        masked_mae = torch.where(
            masked_mae_valid,
            masked_mae,
            torch.full_like(masked_mae, float("nan")),
        )

        self._ct_values["ct_masked_mae"].extend(masked_mae.unbind())
        self._ct_values["ct_mse"].extend(mse.unbind())

    def update_ct_beam_idd(
        self,
        sample: dict[str, Any],
        prediction: torch.Tensor,
        target: torch.Tensor,
        valid: torch.Tensor,
    ) -> None:
        """Score the evaluator's beam IDD for one control point, if enabled."""
        if self._beam_idd is None:
            return
        self._beam_idd.update(sample, prediction, target, valid)

    def update_ct_roundtrip_bev_idd(
        self,
        prediction_bev: torch.Tensor,
        prediction_bev_positive_clip: torch.Tensor,
        target_bev: torch.Tensor,
    ) -> None:
        """Accumulate BEV IDD after CT-space prediction is mapped back to BEV."""
        target = target_bev.detach().float()
        raw = prediction_bev.detach().float()
        clipped = prediction_bev_positive_clip.detach().float()
        if target.ndim == 3:
            target = target.unsqueeze(0)
        if raw.ndim == 3:
            raw = raw.unsqueeze(0)
        if clipped.ndim == 3:
            clipped = clipped.unsqueeze(0)
        if raw.shape != target.shape or clipped.shape != target.shape:
            raise ValueError(
                "round-trip BEV IDD tensors must have matching shapes, got "
                f"raw={tuple(raw.shape)}, clipped={tuple(clipped.shape)}, "
                f"target={tuple(target.shape)}"
            )
        self._ct_values["bev_idd_from_ct"].extend(
            idd_curve_distance_loss(raw, target, reduction="none").unbind()
        )
        self._ct_values["bev_idd_from_ct_positive_clip"].extend(
            idd_curve_distance_loss(clipped, target, reduction="none").unbind()
        )

    def summary_means(self) -> dict[str, float]:
        means = (
            bev_metric_means_from_summary(self._metrics.summary())
            if self._bev_updates
            else {}
        )
        for name, values in self._ct_values.items():
            if not values:
                continue
            stacked = torch.stack(values)
            finite = torch.isfinite(stacked)
            if finite.any():
                means[name] = float(stacked[finite].mean().cpu().item())
        if self._beam_idd is not None:
            means.update(self._beam_idd.summary_means())
        return means


def compute_bev_validation_means(
    model: nn.Module,
    val_loader,
    device: torch.device,
    *,
    forward_fn: Callable[..., tuple[torch.Tensor, torch.Tensor]] | None = None,
    materializer: Any | None = None,
    amp: TrainAmpConfig | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    desc: str = "val-metrics",
    idd_floor_frac: float = 0.005,
) -> dict[str, float]:
    """Accumulate fast BEV metrics on the validation loader (no gamma)."""
    forward = forward_fn or default_bev_forward
    accumulator = BevMetricsAccumulator(device, idd_floor_frac=idd_floor_frac)
    was_training = model.training
    model.eval()
    set_optimizer_eval_mode(optimizer)
    with torch.no_grad():
        for batch in epoch_batch_progress(val_loader, desc=desc, leave=False):
            output, label = forward(model, batch, device, materializer=materializer, amp=amp)
            accumulator.update(output, label, batch)
    if was_training:
        model.train()
        set_optimizer_train_mode(optimizer)
    return accumulator.summary_means()


@dataclass
class ValMetricsHistory:
    epochs: list[int] = field(default_factory=list)
    metrics: dict[str, list[float]] = field(default_factory=dict)

    def record(self, epoch_1based: int, means: dict[str, float]) -> None:
        if self.epochs and epoch_1based <= self.epochs[-1]:
            idx = self.epochs.index(epoch_1based)
            for name, value in means.items():
                self.metrics.setdefault(name, [float("nan")] * len(self.epochs))
                self.metrics[name][idx] = float(value)
            return
        self.epochs.append(int(epoch_1based))
        old_length = len(self.epochs) - 1
        for name in self.metrics.keys() | means.keys():
            self.metrics.setdefault(name, [float("nan")] * old_length)
            self.metrics[name].append(float(means.get(name, float("nan"))))

    def save(self, out_dir: str | Path) -> Path:
        path = Path(out_dir) / VAL_METRICS_HISTORY_JSON
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump({"epochs": self.epochs, "metrics": self.metrics}, fh, indent=2)
        return path

    @classmethod
    def load(cls, out_dir: str | Path) -> ValMetricsHistory:
        path = Path(out_dir) / VAL_METRICS_HISTORY_JSON
        if not path.is_file():
            return cls()
        with path.open(encoding="utf-8") as fh:
            data = json.load(fh)
        return cls(
            epochs=[int(x) for x in data.get("epochs", [])],
            metrics={k: [float(x) for x in v] for k, v in data.get("metrics", {}).items()},
        )


class BevMetricsLogger:
    """Log fast BEV validation metrics collected during the main validation pass."""

    def __init__(self, experiment_dir: str | Path, *, enabled: bool = True) -> None:
        self.enabled = bool(enabled)
        self.experiment_dir = Path(experiment_dir)
        self.history = ValMetricsHistory.load(self.experiment_dir)
        self._writer = None
        if self.enabled:
            from torch.utils.tensorboard import SummaryWriter

            tb_dir = self.experiment_dir / "tensorboard"
            self._writer = SummaryWriter(log_dir=str(tb_dir))
            print(f"TensorBoard BEV metrics: {tb_dir}")

    @staticmethod
    def should_collect(epoch_1based: int, freq: int, *, enabled: bool = True) -> bool:
        return bool(enabled) and freq > 0 and epoch_1based % freq == 0

    def create_accumulator(
        self,
        device: torch.device,
        *,
        catalog: Any | None = None,
        gantry_lookup: dict[tuple[str, int, int], float] | None = None,
        idd_floor_frac: float = 0.005,
    ) -> BevMetricsAccumulator:
        return BevMetricsAccumulator(
            device,
            catalog=catalog,
            gantry_lookup=gantry_lookup,
            idd_floor_frac=idd_floor_frac,
        )

    def record_losses(
        self,
        epoch_1based: int,
        train_loss: float,
        val_loss: float,
        *,
        lr: float | None = None,
    ) -> None:
        """Log per-epoch train/val loss and LR under the backfill's scalar tags.

        ``scripts/backfill_tensorboard.py`` writes ``loss/train``, ``loss/val`` and
        ``lr`` by replaying ``loss_history.json``; using the same tags here means a
        live run and a backfilled one overlay in TensorBoard. Unlike :meth:`record`
        this is not gated on ``--val-metrics-freq``: losses exist for every epoch,
        so they are logged for every epoch.
        """
        if self._writer is None:
            return
        self._writer.add_scalar("loss/train", float(train_loss), int(epoch_1based))
        self._writer.add_scalar("loss/val", float(val_loss), int(epoch_1based))
        if lr is not None:
            self._writer.add_scalar("lr", float(lr), int(epoch_1based))
        self._writer.flush()

    def record(self, epoch_1based: int, means: dict[str, float]) -> None:
        if not self.enabled or not means:
            return
        self.history.record(epoch_1based, means)
        self.history.save(self.experiment_dir)
        if self._writer is not None:
            for name, value in means.items():
                self._writer.add_scalar(f"val/{name}", value, epoch_1based)
            self._writer.flush()
        msg = "  [val metrics] " + "  ".join(f"{k}={v:.4g}" for k, v in means.items())
        print(msg, flush=True)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()

    def finalize(self) -> None:
        self.close()


# Backward-compatible alias for existing training scripts and docs.
AsyncBevMetricsLogger = BevMetricsLogger


_BIN_SAMPLE_RE = re.compile(r"^(?P<patient>.+)_\d+_CP\d{3}$")
_SAMPLE_ID_RE = re.compile(r"^(?P<patient>.+)_(?P<beam>\d+)_CP(?P<cp>\d{3})$")
_DEFAULT_BIN_SHAPES = ((256, 200, 200), (128, 192, 192))
BIN_FILE_DTYPES = ("float32", "float16", "float64")
MERGED_BIN_CHANNELS = ("ct", "proj", "dose")
MERGED_BIN_MANIFEST = "merged_bin_manifest.json"
MULTIMODAL_DATA_FORMATS = ("h5", "bin", "merged_bin", "otf_gpu")


def read_bev_bin_dtype_manifest(bins_root: str | Path) -> str | None:
    manifest = Path(bins_root) / "bev_bin_dtype.txt"
    if not manifest.is_file():
        return None
    line = manifest.read_text(encoding="utf-8").strip().splitlines()[0].strip()
    return line if line in BIN_FILE_DTYPES else None


def resolve_bin_dtype(bins_root: str | Path, dtype: str | None) -> str:
    if dtype is not None:
        return dtype
    return read_bev_bin_dtype_manifest(bins_root) or "float32"


def _infer_bin_layout_from_file_size(path: Path) -> tuple[tuple[int, int, int], np.dtype]:
    nbytes = path.stat().st_size
    for dt_name in ("float32", "float16", "float64"):
        itemsize = np.dtype(dt_name).itemsize
        if nbytes % itemsize != 0:
            continue
        n_elements = nbytes // itemsize
        for shp in _DEFAULT_BIN_SHAPES:
            if int(np.prod(shp)) == n_elements:
                return shp, np.dtype(dt_name)
    raise ValueError(
        f"Cannot infer shape/dtype from file size {nbytes} bytes for {path}. "
        f"Pass --bin-shape and --bin-dtype explicitly."
    )


def _patient_id_from_sample_id(sample_id: str) -> str:
    m = _BIN_SAMPLE_RE.match(sample_id)
    if m:
        return m.group("patient")
    return sample_id.split("_", 1)[0]


def _parse_doserad_sample_id(sample_id: str) -> tuple[str, int, int]:
    m = _SAMPLE_ID_RE.match(sample_id)
    if m is None:
        raise ValueError(
            f"Sample id {sample_id!r} must match '<patient>_<beam>_CPxxx' "
            "(for example 1ABB006_0_CP001)"
        )
    return m.group("patient"), int(m.group("beam")), int(m.group("cp"))


def _strip_bin_prefix(path: Path, prefix: str) -> str:
    name = path.name
    if not name.endswith(".bin"):
        raise ValueError(f"Expected .bin file, got: {path}")
    stem = name[:-4]
    if not stem.startswith(prefix):
        raise ValueError(f"Expected prefix '{prefix}' in filename: {path.name}")
    return stem[len(prefix) :]


def _load_bin_array(path: Path, shape: tuple[int, int, int], dtype: np.dtype) -> np.ndarray:
    arr = np.fromfile(path, dtype=dtype)
    expected = int(np.prod(shape))
    if arr.size != expected:
        raise ValueError(
            f"Unexpected element count in {path}: got {arr.size}, expected {expected} for shape={shape}"
        )
    return arr.reshape(shape)


def _infer_bin_shape_for_dtype(path: Path, dtype: np.dtype) -> tuple[int, int, int]:
    nbytes = path.stat().st_size
    if nbytes % dtype.itemsize != 0:
        raise ValueError(
            f"Cannot infer shape for {path}: file size {nbytes} is not divisible by "
            f"dtype itemsize {dtype.itemsize} ({dtype.name})"
        )
    n_elements = nbytes // dtype.itemsize
    for shp in _DEFAULT_BIN_SHAPES:
        if int(np.prod(shp)) == n_elements:
            return shp
    raise ValueError(
        f"Could not infer shape from {path}: {n_elements} {dtype.name} elements. "
        f"Tried shapes={list(_DEFAULT_BIN_SHAPES)}. Pass --bin-shape explicitly."
    )


def _infer_bin_shape_from_path(path: Path, dtype: np.dtype) -> tuple[int, int, int]:
    n_elements = np.fromfile(path, dtype=dtype).size
    for shp in _DEFAULT_BIN_SHAPES:
        if int(np.prod(shp)) == int(n_elements):
            return shp
    raise ValueError(
        f"Could not infer shape from element count {n_elements} in {path}. "
        f"Tried shapes={list(_DEFAULT_BIN_SHAPES)}"
    )


def _split_patients_fraction(
    patient_ids: list[str],
    validation_fraction: float,
    seed: int,
) -> tuple[set[str], set[str]]:
    rng = random.Random(seed)
    ids = list(patient_ids)
    rng.shuffle(ids)
    if len(ids) <= 1:
        return set(ids), set()
    n_val = int(round(validation_fraction * len(ids)))
    n_val = max(1, min(n_val, len(ids) - 1))
    val = set(ids[:n_val])
    train = set(ids[n_val:])
    return train, val


@dataclass(frozen=True)
class BinCatalog:
    """Discovered ct/proj/dose .bin files and train/validation sample ids."""

    ct_map: dict[str, Path]
    proj_map: dict[str, Path]
    dose_map: dict[str, Path]
    train_ids: list[str]
    val_ids: list[str]
    shape: tuple[int, int, int]
    storage_dtype: np.dtype
    test_ids: list[str] = field(default_factory=list)


def _looks_like_merged_bin_root(path: Path) -> Path | None:
    """Return merged pack root if ``path`` or its parent holds ``merged_bin_manifest.json``."""
    path = Path(path)
    if (path / MERGED_BIN_MANIFEST).is_file():
        return path
    if path.name in BIN_FILE_DTYPES and (path.parent / MERGED_BIN_MANIFEST).is_file():
        return path.parent
    return None


def discover_bin_catalog(
    bins_root: str | Path,
    *,
    patient_split_json: str | Path | None = None,
    validation_fraction: float = 0.2,
    seed: int = 333,
    shape: tuple[int, int, int] | None = None,
    dtype: str | None = None,
) -> BinCatalog:
    """List aligned BEV .bin samples under ``bins_root/{ct,proj,dose}/``."""
    root = Path(bins_root)
    merged_root = _looks_like_merged_bin_root(root)
    if merged_root is not None:
        raise FileNotFoundError(
            f"{root} is a merged_bin pack (found {MERGED_BIN_MANIFEST} under {merged_root}), "
            f"not separate ct/proj/dose folders. Use --data-format merged_bin."
        )
    resolved_dtype = resolve_bin_dtype(root, dtype)
    ct_dir = root / "ct"
    proj_dir = root / "proj"
    dose_dir = root / "dose"
    for d in (ct_dir, proj_dir, dose_dir):
        if not d.is_dir():
            raise FileNotFoundError(f"Missing required directory: {d}")

    ct_map = {_strip_bin_prefix(p, "ct_"): p for p in sorted(ct_dir.glob("ct_*.bin"))}
    proj_map = {_strip_bin_prefix(p, "proj_"): p for p in sorted(proj_dir.glob("proj_*.bin"))}
    dose_map = {_strip_bin_prefix(p, "dose_"): p for p in sorted(dose_dir.glob("dose_*.bin"))}
    sample_ids = sorted(set(ct_map) & set(proj_map) & set(dose_map))
    if not sample_ids:
        raise ValueError(f"No common sample IDs under {root}")

    in_dtype = np.dtype(resolved_dtype)
    first_ct = ct_map[sample_ids[0]]
    if shape is None:
        shape, inferred_dtype = _infer_bin_layout_from_file_size(first_ct)
        if in_dtype != inferred_dtype:
            raise ValueError(
                f"bin dtype {resolved_dtype!r} does not match {first_ct.name} "
                f"(file layout uses {inferred_dtype.name})"
            )
    else:
        expected = int(np.prod(shape))
        n_elements = first_ct.stat().st_size // in_dtype.itemsize
        if n_elements != expected:
            raise ValueError(
                f"--bin-shape {shape} incompatible with {first_ct.name} "
                f"for dtype {resolved_dtype} ({n_elements} elements)"
            )

    train_ids, val_ids, test_ids = _train_val_ids_from_samples(
        sample_ids,
        patient_split_json=patient_split_json,
        validation_fraction=validation_fraction,
        seed=seed,
    )

    sid0 = sample_ids[0]
    expected_bytes = int(np.prod(shape)) * in_dtype.itemsize
    for label, path in (
        ("ct", ct_map[sid0]),
        ("proj", proj_map[sid0]),
        ("dose", dose_map[sid0]),
    ):
        nbytes = path.stat().st_size
        if nbytes != expected_bytes:
            try:
                inferred_shape, inferred_dtype = _infer_bin_layout_from_file_size(path)
            except ValueError:
                inferred_dtype = None
                inferred_shape = None
            hint = ""
            if inferred_dtype is not None:
                hint = (
                    f" File looks like shape={inferred_shape} dtype={inferred_dtype.name}."
                )
            raise ValueError(
                f"{label} bin {path.name}: {nbytes} bytes, expected {expected_bytes} for "
                f"shape={shape} dtype={resolved_dtype}.{hint}"
            )

    return BinCatalog(
        ct_map=ct_map,
        proj_map=proj_map,
        dose_map=dose_map,
        train_ids=train_ids,
        val_ids=val_ids,
        test_ids=test_ids,
        shape=shape,
        storage_dtype=in_dtype,
    )


def _train_val_ids_from_samples(
    sample_ids: list[str],
    *,
    patient_split_json: str | Path | None,
    validation_fraction: float,
    seed: int,
) -> tuple[list[str], list[str], list[str]]:
    if patient_split_json:
        import sys

        preproc = Path(__file__).resolve().parent.parent / "preprocessing"
        sp = str(preproc)
        if sp not in sys.path:
            sys.path.insert(0, sp)
        from patient_split import h5_group_for_sample, load_patient_split

        split_data = load_patient_split(patient_split_json)
        train_ids: list[str] = []
        val_ids: list[str] = []
        test_ids: list[str] = []
        for sid in sample_ids:
            group = h5_group_for_sample(split_data, sid)
            if group == "train":
                train_ids.append(sid)
            elif group == "validation":
                val_ids.append(sid)
            elif group == "test":
                test_ids.append(sid)
    else:
        patients = sorted({_patient_id_from_sample_id(sid) for sid in sample_ids})
        train_patients, val_patients = _split_patients_fraction(
            patients, validation_fraction=validation_fraction, seed=seed
        )
        train_ids = [sid for sid in sample_ids if _patient_id_from_sample_id(sid) in train_patients]
        val_ids = [sid for sid in sample_ids if _patient_id_from_sample_id(sid) in val_patients]
        test_ids = []

    if not train_ids:
        raise ValueError("No training samples after split")
    if not val_ids:
        raise ValueError("No validation samples after split")
    return train_ids, val_ids, test_ids


@dataclass(frozen=True)
class MergedBinCatalog:
    """Packed merged BEV bins: ``{merged_root}/{dtype}/samples/<id>.bin``."""

    merged_root: Path
    samples_dir: Path
    train_ids: list[str]
    val_ids: list[str]
    shape_dhw: tuple[int, int, int]
    storage_dtype: np.dtype
    test_ids: list[str] = field(default_factory=list)

    @property
    def shape_cdhw(self) -> tuple[int, int, int, int]:
        return (len(MERGED_BIN_CHANNELS), *self.shape_dhw)


@dataclass(frozen=True)
class OtfGpuSampleRecord:
    """Dose-only on-disk sample plus patient CT / MAC / aperture paths."""

    sample_id: str
    patient_id: str
    beam_id: int
    cp_id: int
    dose_path: Path
    ct_path: Path
    mac_path: Path
    segment_path: Path
    ct_geometry_path: Path | None = None
    # Proton beamlets only: the ray this beamlet belongs to, its index within
    # that ray, the ray geometry, and the energy layer -- nominal energy,
    # energy spread and spot width, all read from beam_parameters.json.
    ray_id: int | None = None
    beamlet_id: int | None = None
    source_xyz_mm: tuple[float, float, float] | None = None
    target_xyz_mm: tuple[float, float, float] | None = None
    energy_mev: float | None = None
    sigma_energy_mev: float | None = None
    sigma_spot_mm: float | None = None

    @property
    def name(self) -> str:
        """Inference preprocessing task identifier."""
        return self.sample_id

    @property
    def mac_file(self) -> str:
        """Inference preprocessing task MAC path."""
        return str(self.mac_path)


@dataclass(frozen=True)
class OtfGpuCatalog:
    """Dose-only catalog for on-the-fly GPU BEV input generation."""

    data_root: Path
    baseline_pb_dir: Path
    baseline_split: str
    segment_mac_dir: Path
    train_ids: list[str]
    val_ids: list[str]
    records: dict[str, OtfGpuSampleRecord]
    shape: tuple[int, int, int]
    dose_dtype: np.dtype
    ct_name: str = "ct.mha"
    test_ids: list[str] = field(default_factory=list)
    otf_gpu_full: bool = False
    sct_dir: Path | None = None
    # Proton catalogs carry the beamlet energy table discovered from
    # beam_parameters.json: the sorted unique levels the energy token indexes
    # into, their spot sigmas, and each patient's energy span.
    modality: str = "photon"
    proton_energy_ranges: dict[str, tuple[float, float]] = field(default_factory=dict)
    proton_energy_levels: tuple[float, ...] = ()
    proton_energy_sigmas: tuple[float, ...] = ()

    @property
    def patient_ids(self) -> list[str]:
        return sorted(
            {
                self.records[sid].patient_id
                for sid in (self.train_ids + self.val_ids + self.test_ids)
            }
        )

    @property
    def patient_ct_map(self) -> dict[str, Path]:
        out: dict[str, Path] = {}
        for rec in self.records.values():
            out.setdefault(rec.patient_id, rec.ct_path)
        return out

    @property
    def patient_ct_geometry_map(self) -> dict[str, Path]:
        """Reference CT per patient whose header carries the true spacing/origin.

        Only populated for volumes read from ``--sct-dir``; empty otherwise.
        """
        out: dict[str, Path] = {}
        for rec in self.records.values():
            if rec.ct_geometry_path is not None:
                out.setdefault(rec.patient_id, rec.ct_geometry_path)
        return out


def read_merged_bin_manifest(merged_root: str | Path) -> tuple[tuple[int, int, int], tuple[str, ...]]:
    path = Path(merged_root) / MERGED_BIN_MANIFEST
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Pack merged bins first "
            f"(scripts/compare_merged_bin_precision_and_load.py --pack-merged)."
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    shape_dhw = tuple(int(x) for x in data["shape_dhw"])
    channels = tuple(data.get("channels", MERGED_BIN_CHANNELS))
    if tuple(channels) != MERGED_BIN_CHANNELS:
        raise ValueError(f"Expected channels {MERGED_BIN_CHANNELS}, got {channels}")
    return shape_dhw, channels


def resolve_merged_bin_paths(
    data_dir: str | Path,
    storage_dtype: str,
) -> tuple[Path, Path]:
    """Resolve merged root and ``samples/`` dir from ``--data-dir``.

    Accepts either the pack root (with ``merged_bin_manifest.json``) or
    ``<root>/float16`` / ``<root>/float32``.
    """
    if storage_dtype not in BIN_FILE_DTYPES:
        raise ValueError(f"storage_dtype must be one of {BIN_FILE_DTYPES}, got {storage_dtype!r}")

    data_dir = Path(data_dir)
    if (data_dir / MERGED_BIN_MANIFEST).is_file():
        merged_root = data_dir
        samples_dir = merged_root / storage_dtype / "samples"
    elif data_dir.name in BIN_FILE_DTYPES and (data_dir.parent / MERGED_BIN_MANIFEST).is_file():
        merged_root = data_dir.parent
        samples_dir = data_dir / "samples"
    else:
        raise FileNotFoundError(
            f"Could not find {MERGED_BIN_MANIFEST} under {data_dir} or its parent. "
            f"Pass the merged pack root (e.g. bev_merged_benchmark) or a dtype subdir."
        )
    if not samples_dir.is_dir():
        raise FileNotFoundError(
            f"Missing merged samples directory: {samples_dir}. "
            f"Run packing with storage dtype {storage_dtype!r}."
        )
    return merged_root, samples_dir


def discover_merged_bin_catalog(
    data_dir: str | Path,
    *,
    storage_dtype: str = "float16",
    patient_split_json: str | Path | None = None,
    validation_fraction: float = 0.2,
    seed: int = 333,
) -> MergedBinCatalog:
    merged_root, samples_dir = resolve_merged_bin_paths(data_dir, storage_dtype)
    shape_dhw, _channels = read_merged_bin_manifest(merged_root)
    in_dtype = np.dtype(storage_dtype)

    sample_ids = sorted(p.stem for p in samples_dir.glob("*.bin"))
    if not sample_ids:
        raise ValueError(f"No merged .bin samples under {samples_dir}")

    expected_bytes = int(np.prod((len(MERGED_BIN_CHANNELS), *shape_dhw))) * in_dtype.itemsize
    nbytes = (samples_dir / f"{sample_ids[0]}.bin").stat().st_size
    if nbytes != expected_bytes:
        raise ValueError(
            f"Merged sample size mismatch in {samples_dir}: got {nbytes} bytes, "
            f"expected {expected_bytes} for shape_cdhw={(len(MERGED_BIN_CHANNELS), *shape_dhw)} "
            f"dtype={storage_dtype}"
        )

    train_ids, val_ids, test_ids = _train_val_ids_from_samples(
        sample_ids,
        patient_split_json=patient_split_json,
        validation_fraction=validation_fraction,
        seed=seed,
    )
    return MergedBinCatalog(
        merged_root=merged_root,
        samples_dir=samples_dir,
        train_ids=train_ids,
        val_ids=val_ids,
        test_ids=test_ids,
        shape_dhw=shape_dhw,
        storage_dtype=in_dtype,
    )


def _require_otf_arg(value: str | Path | None, name: str) -> Path:
    if value is None or str(value) == "":
        raise ValueError(f"{name} is required when --data-format otf_gpu")
    return Path(value)


def _doserad_split_root(
    baseline_root: Path,
    baseline_split: str,
    *,
    modality: str = "photon",
) -> Path:
    return baseline_root / modality / baseline_split


def _doserad_dose_mha_path(patient_dir: Path, beam_id: int, cp_id: int) -> Path:
    return patient_dir / "dose" / f"Dose_B{beam_id}_CP{cp_id:03d}.mha"


def _collect_doserad_sample_ids_from_plans(split_root: Path) -> list[str]:
    sample_ids: list[str] = []
    if not split_root.is_dir():
        raise FileNotFoundError(f"Missing DoseRAD split directory: {split_root}")
    for patient_dir in sorted(split_root.iterdir()):
        if not patient_dir.is_dir():
            continue
        patient_id = patient_dir.name
        plan_path = patient_dir / f"{patient_id}.json"
        if not plan_path.is_file():
            continue
        plan_json = json.loads(plan_path.read_text(encoding="utf-8"))
        for beam in plan_json.get("beams", []):
            beam_id = int(beam["beam_idx"])
            for control_point in beam.get("control_points", []):
                cp_id = int(control_point["cp_idx"])
                sample_ids.append(f"{patient_id}_{beam_id}_CP{cp_id:03d}")
    return sorted(sample_ids)


def _default_otf_bev_shape(bev_grid: Any | None = None) -> tuple[int, int, int]:
    if bev_grid is not None:
        return bev_grid.shape_dhw
    return _DEFAULT_BIN_SHAPES[0]


def _load_bev_grid_config_module():
    module_name = "bev_grid_config"
    cached = sys.modules.get(module_name)
    if cached is not None:
        return cached

    bev_py = Path(__file__).resolve().parent.parent / "inference" / "bev_grid_config.py"
    spec = importlib.util.spec_from_file_location(module_name, bev_py)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load BEV grid config module from {bev_py}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_bev_grid_config(path: str | Path | None = None) -> Any:
    """Load ``BevGridConfig`` from JSON or return built-in defaults."""
    bev_module = _load_bev_grid_config_module()
    return bev_module.load_bev_grid_config(path)


def _build_mac_cache_entries(
    sample_ids: Iterable[str],
    records: dict[str, OtfGpuSampleRecord],
    pipeline: Any,
    bev_grid: Any,
) -> dict[str, dict[str, list[float]]]:
    crop = float(bev_grid.resolved_plane_origin_offset_mm)
    off = [0.0, -(bev_grid.ny // 2) + 0.5, -(bev_grid.nz // 2) + 0.5]
    out: dict[str, dict[str, list[float]]] = {}
    for sid in sample_ids:
        rec = records[sid]
        s, dx, dy = pipeline.extract_gps(str(rec.mac_path))
        U = pipeline._basis_np(dx, dy)
        src = np.asarray(s, dtype=np.float32) + crop * U[0]
        out[sid] = {
            "s": [float(v) for v in s],
            "dx": [float(v) for v in dx],
            "dy": [float(v) for v in dy],
            "U": U.reshape(-1).astype(np.float32).tolist(),
            "src": src.astype(np.float32).tolist(),
            "off": [float(v) for v in off],
            "NX": int(bev_grid.nx),
            "NY": int(bev_grid.ny),
            "NZ": int(bev_grid.nz),
            "z_align_mm": 0.0,
        }
    return out


def discover_otf_gpu_catalog(
    data_dir: str | Path | None,
    *,
    baseline_pb_dir: str | Path | None,
    segment_mac_dir: str | Path | None,
    baseline_split: str = "training",
    patient_split_json: str | Path | None = None,
    validation_fraction: float = 0.2,
    seed: int = 333,
    shape: tuple[int, int, int] | None = None,
    dose_dtype: str = "float16",
    ct_name: str = "ct.mha",
    otf_gpu_full: bool = False,
    bev_grid: Any | None = None,
    otf_modality: str = "photon",
    proton_beam_parameters: str | Path | None = None,
    sct_dir: str | Path | None = None,
) -> OtfGpuCatalog:
    """Discover otf_gpu samples and validate CT/MAC/aperture (+ dose) inputs."""
    if otf_modality not in ("photon", "proton"):
        raise ValueError(
            f"otf modality must be 'photon' or 'proton', got {otf_modality!r}"
        )
    if otf_modality == "proton":
        # Proton plans live under --data-dir with their own beamlet geometry,
        # and carry no MLC segments, so the photon discovery below does not
        # apply. Dose dtype is validated inside the proton builder.
        return _discover_proton_otf_gpu_catalog(
            data_dir,
            baseline_pb_dir=baseline_pb_dir,
            baseline_split=baseline_split,
            patient_split_json=patient_split_json,
            validation_fraction=validation_fraction,
            seed=seed,
            shape=shape,
            dose_dtype=np.dtype(dose_dtype),
            ct_name=ct_name,
            otf_gpu_full=otf_gpu_full,
            bev_grid=bev_grid,
            beam_parameters_path=proton_beam_parameters,
        )
    baseline_root = _require_otf_arg(baseline_pb_dir, "--baseline-pb-dir")
    sct_root: Path | None = None
    if sct_dir is not None and str(sct_dir) != "":
        sct_root = Path(sct_dir)
        if not sct_root.is_dir():
            raise FileNotFoundError(f"--sct-dir is not a directory: {sct_root}")
    segment_root = _require_otf_arg(segment_mac_dir, "--segment-mac-dir")
    if dose_dtype not in BIN_FILE_DTYPES:
        raise ValueError(f"otf dose dtype must be one of {BIN_FILE_DTYPES}, got {dose_dtype!r}")
    in_dtype = np.dtype(dose_dtype)

    mac_dir = segment_root / "mac"
    seg_dir = segment_root / "segments"
    for d in (mac_dir, seg_dir):
        if not d.is_dir():
            raise FileNotFoundError(f"Missing required directory for otf_gpu: {d}")

    split_root = _doserad_split_root(baseline_root, baseline_split)
    dose_map: dict[str, Path] = {}
    if otf_gpu_full:
        if data_dir is not None and str(data_dir) != "":
            root = Path(data_dir)
        else:
            root = baseline_root
        sample_ids = _collect_doserad_sample_ids_from_plans(split_root)
        if not sample_ids:
            raise ValueError(f"No control points found under plan JSONs in {split_root}")
        if shape is None:
            shape = _default_otf_bev_shape(bev_grid)
        expected_bytes = None
        for sid in sample_ids:
            try:
                patient_id, beam_id, cp_id = _parse_doserad_sample_id(sid)
            except ValueError:
                continue
            patient_dir = split_root / patient_id
            dose_map[sid] = _doserad_dose_mha_path(patient_dir, beam_id, cp_id)
    else:
        if data_dir is None or str(data_dir) == "":
            raise ValueError(
                "--data-dir is required for --data-format otf_gpu unless --otf-gpu-full is set"
            )
        root = Path(data_dir)
        dose_dir = root / "dose"
        if not dose_dir.is_dir():
            raise FileNotFoundError(f"Missing required directory for otf_gpu: {dose_dir}")
        dose_map = {_strip_bin_prefix(p, "dose_"): p for p in sorted(dose_dir.glob("dose_*.bin"))}
        sample_ids = sorted(dose_map)
        if not sample_ids:
            raise ValueError(f"No dose_*.bin files found under {dose_dir}")
        if shape is None:
            shape = _infer_bin_shape_for_dtype(dose_map[sample_ids[0]], in_dtype)
        expected_bytes = int(np.prod(shape)) * in_dtype.itemsize

    if bev_grid is not None and tuple(shape) != bev_grid.shape_dhw:
        raise ValueError(
            f"Catalog shape {shape} does not match --bev-grid-config shape_dhw {bev_grid.shape_dhw}"
        )

    train_ids, val_ids, test_ids = _train_val_ids_from_samples(
        sample_ids,
        patient_split_json=patient_split_json,
        validation_fraction=validation_fraction,
        seed=seed,
    )
    active_ids = sorted(set(train_ids) | set(val_ids) | set(test_ids))

    records: dict[str, OtfGpuSampleRecord] = {}
    errors: list[str] = []
    sct_path_by_patient: dict[str, Path | None] = {}
    for sid in active_ids:
        try:
            patient_id, beam_id, cp_id = _parse_doserad_sample_id(sid)
        except ValueError as e:
            errors.append(str(e))
            continue

        dose_path = dose_map[sid]
        patient_dir = split_root / patient_id
        ct_path = patient_dir / "image" / ct_name
        ct_geometry_path: Path | None = None
        if sct_root is not None:
            if patient_id not in sct_path_by_patient:
                sct_path_by_patient[patient_id] = _resolve_sct_path(sct_root, patient_id)
                if sct_path_by_patient[patient_id] is None:
                    errors.append(
                        f"Missing synthetic CT for patient {patient_id} under {sct_root}"
                    )
            sct_path = sct_path_by_patient[patient_id]
            if sct_path is not None:
                # the reference CT is still required: sCT exports carry no geometry
                if not ct_path.is_file():
                    errors.append(f"Missing reference CT for {sid}: {ct_path}")
                ct_geometry_path = ct_path
                ct_path = sct_path
        mac_path = mac_dir / f"{sid}.mac"
        segment_path = seg_dir / f"{sid}.bin"
        if expected_bytes is not None and dose_path.stat().st_size != expected_bytes:
            errors.append(
                f"{dose_path}: {dose_path.stat().st_size} bytes, expected {expected_bytes} "
                f"for shape={shape} dtype={in_dtype.name}"
            )
        if not dose_path.is_file():
            errors.append(f"Missing dose for {sid}: {dose_path}")
        if not ct_path.is_file():
            errors.append(f"Missing CT for {sid}: {ct_path}")
        if not mac_path.is_file():
            errors.append(f"Missing MAC for {sid}: {mac_path}")
        if not segment_path.is_file():
            errors.append(f"Missing aperture segment for {sid}: {segment_path}")

        records[sid] = OtfGpuSampleRecord(
            sample_id=sid,
            patient_id=patient_id,
            beam_id=beam_id,
            cp_id=cp_id,
            dose_path=dose_path,
            ct_path=ct_path,
            mac_path=mac_path,
            segment_path=segment_path,
            ct_geometry_path=ct_geometry_path,
        )

    if errors:
        preview = "\n  ".join(errors[:12])
        suffix = f"\n  ... {len(errors) - 12} more error(s)" if len(errors) > 12 else ""
        raise FileNotFoundError(f"Invalid otf_gpu catalog:\n  {preview}{suffix}")

    return OtfGpuCatalog(
        data_root=root,
        baseline_pb_dir=baseline_root,
        baseline_split=baseline_split,
        segment_mac_dir=segment_root,
        train_ids=train_ids,
        val_ids=val_ids,
        test_ids=test_ids,
        records=records,
        shape=shape,
        dose_dtype=in_dtype,
        ct_name=ct_name,
        otf_gpu_full=otf_gpu_full,
        sct_dir=sct_root,
        modality="photon",
    )


def _load_merged_bin_channels(
    path: Path,
    shape_dhw: tuple[int, int, int],
    storage_dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shape_cdhw = (len(MERGED_BIN_CHANNELS), *shape_dhw)
    n = int(np.prod(shape_cdhw))
    raw = np.fromfile(path, dtype=storage_dtype, count=n)
    if raw.size != n:
        raise ValueError(f"{path}: expected {n} elements, got {raw.size}")
    vol = raw.reshape(shape_cdhw)
    return vol[0], vol[1], vol[2]


def compute_train_bin_stats(catalog: BinCatalog) -> dict[str, float]:
    """Global dose scale from training-split .bin dose files."""
    dose_scale = 0.0
    for sid in tqdm(catalog.train_ids, desc="Computing dose_scale (train .bin)", leave=False):
        path = catalog.dose_map[sid]
        d = _load_bin_array(path, catalog.shape, catalog.storage_dtype)
        dose_scale = max(dose_scale, float(np.max(d)))
    if dose_scale <= 0:
        raise ValueError("Cannot compute dose_scale: all training dose volumes are non-positive")
    return {
        "ct_min": -1024.0,
        "ct_max": 3071.0,
        "dose_scale": dose_scale,
    }


def compute_train_merged_bin_stats(catalog: MergedBinCatalog) -> dict[str, float]:
    """Global dose scale from training-split merged dose channel."""
    dose_scale = 0.0
    for sid in tqdm(catalog.train_ids, desc="Computing dose_scale (train merged .bin)", leave=False):
        path = catalog.samples_dir / f"{sid}.bin"
        _ct, _proj, dose = _load_merged_bin_channels(
            path, catalog.shape_dhw, catalog.storage_dtype
        )
        dose_scale = max(dose_scale, float(np.max(dose)))
    if dose_scale <= 0:
        raise ValueError("Cannot compute dose_scale: all training dose volumes are non-positive")
    return {
        "ct_min": -1024.0,
        "ct_max": 3071.0,
        "dose_scale": dose_scale,
    }


def _load_dose_mha_max(path: Path) -> float:
    sitk = __import__("SimpleITK")
    arr = sitk.GetArrayFromImage(sitk.ReadImage(str(path)))
    return float(np.max(arr))


def _is_ct_anatomy_name(ct_name: str | Path) -> bool:
    """Return True when ``ct_name`` refers to a CT volume (default ``ct.mha`` / stem ``ct``)."""
    name = Path(ct_name).name
    stem = Path(ct_name).stem
    return name.lower() == "ct.mha" or stem.lower() == "ct"


def _catalog_ct_is_anatomy(catalog: OtfGpuCatalog) -> bool:
    """Return True when the catalog's network input is a CT-valued (HU) volume.

    Synthetic CTs from ``--sct-dir`` are HU volumes regardless of ``--ct-name``,
    which then only selects the reference volume used for geometry.
    """
    if catalog.sct_dir is not None:
        return True
    return _is_ct_anatomy_name(catalog.ct_name)


def _resolve_sct_path(sct_dir: Path, patient_id: str) -> Path | None:
    """Locate ``<sct_dir>/<patient_id>.<ext>`` (flat folder, one file per patient)."""
    for suffix in (".mha", ".mhd", ".nii.gz", ".nii", ".nrrd"):
        candidate = sct_dir / f"{patient_id}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def _read_otf_ct_volume(
    catalog: OtfGpuCatalog,
    patient_id: str,
    sitk: Any,
) -> tuple[np.ndarray, tuple[float, float, float], tuple[float, float, float]]:
    """Read a patient input volume as ``(x, y, z)`` float32 plus spacing/origin.

    Synthetic CTs are exported without geometry (identity spacing, zero origin),
    so with ``--sct-dir`` the spacing/origin are re-stamped from the real
    ``image/<ct-name>`` header of the same patient; without it every beam would
    be projected through the wrong part of the volume.
    """
    path = catalog.patient_ct_map[patient_id]
    img = sitk.ReadImage(str(path))
    arr = np.transpose(sitk.GetArrayFromImage(img), (2, 1, 0)).astype(np.float32, copy=False)
    spacing = tuple(float(v) for v in img.GetSpacing())
    origin = tuple(float(v) for v in img.GetOrigin())

    ref_path = catalog.patient_ct_geometry_map.get(patient_id)
    if ref_path is not None:
        reader = sitk.ImageFileReader()
        reader.SetFileName(str(ref_path))
        reader.ReadImageInformation()
        ref_size = tuple(int(v) for v in reader.GetSize())
        if tuple(int(v) for v in img.GetSize()) != ref_size:
            raise ValueError(
                f"Synthetic CT {path} has size {tuple(img.GetSize())} but reference CT "
                f"{ref_path} has size {ref_size}; both must be on the same voxel grid."
            )
        spacing = tuple(float(v) for v in reader.GetSpacing())
        origin = tuple(float(v) for v in reader.GetOrigin())
    return arr, spacing, origin


def _normalize_mri_volume_p01_p99(arr: np.ndarray) -> np.ndarray:
    """Per-volume robust MRI norm: map p01..p99 linearly to [0, 1] and clamp."""
    out = np.asarray(arr, dtype=np.float32)
    p1, p99 = np.quantile(out, [0.01, 0.99])
    return np.clip((out - p1) / (p99 - p1 + 1e-6), 0.0, 1.0).astype(np.float32, copy=False)


def compute_train_otf_gpu_stats(catalog: OtfGpuCatalog) -> dict[str, float]:
    """Global dose scale from training-split dose bins or dose MHA files.

    CT anatomy uses fixed HU bounds [-1024, 3071]. Non-CT (e.g. ``mr.mha``) uses
    placeholders [0, 1] because intensity is normalized per volume at load time.
    """
    dose_scale = 0.0
    desc = "Computing dose_scale (train otf dose mha)" if catalog.otf_gpu_full else (
        "Computing dose_scale (train otf dose)"
    )
    for sid in tqdm(catalog.train_ids, desc=desc, leave=False):
        rec = catalog.records[sid]
        if catalog.otf_gpu_full:
            dose_scale = max(dose_scale, _load_dose_mha_max(rec.dose_path))
        else:
            d = _load_bin_array(rec.dose_path, catalog.shape, catalog.dose_dtype)
            dose_scale = max(dose_scale, float(np.max(d)))
    if dose_scale <= 0:
        raise ValueError("Cannot compute dose_scale: all training dose volumes are non-positive")
    if _catalog_ct_is_anatomy(catalog):
        return {
            "ct_min": -1024.0,
            "ct_max": 3071.0,
            "dose_scale": dose_scale,
        }
    return {
        "ct_min": 0.0,
        "ct_max": 1.0,
        "dose_scale": dose_scale,
    }


def compute_train_stats(
    data_format: Literal["h5", "bin", "merged_bin", "otf_gpu"],
    data_dir: str | Path | None,
    *,
    patient_split_json: str | Path | None = None,
    validation_fraction: float = 0.2,
    seed: int = 333,
    bin_shape: tuple[int, int, int] | None = None,
    bin_dtype: str | None = None,
    merged_storage_dtype: str = "float16",
    baseline_pb_dir: str | Path | None = None,
    baseline_split: str = "training",
    segment_mac_dir: str | Path | None = None,
    otf_dose_dtype: str = "float16",
    ct_name: str = "ct.mha",
    otf_gpu_full: bool = False,
    bev_grid: Any | None = None,
    otf_modality: str = "photon",
    proton_beam_parameters: str | Path | None = None,
    sct_dir: str | Path | None = None,
) -> dict[str, float]:
    """Compute normalisation stats from the training split."""
    data_dir = resolve_multimodal_data_dir(
        data_format,
        data_dir,
        otf_gpu_full=otf_gpu_full,
        baseline_pb_dir=baseline_pb_dir,
    )
    if data_format == "h5":
        return compute_train_h5_stats(
            str(data_dir / "ct_dataset.h5"),
            str(data_dir / "dose_dataset.h5"),
            group_name="train",
        )
    if data_format == "bin":
        catalog = discover_bin_catalog(
            data_dir,
            patient_split_json=patient_split_json,
            validation_fraction=validation_fraction,
            seed=seed,
            shape=bin_shape,
            dtype=bin_dtype,
        )
        return compute_train_bin_stats(catalog)
    if data_format == "merged_bin":
        catalog = discover_merged_bin_catalog(
            data_dir,
            storage_dtype=merged_storage_dtype,
            patient_split_json=patient_split_json,
            validation_fraction=validation_fraction,
            seed=seed,
        )
        return compute_train_merged_bin_stats(catalog)
    if data_format == "otf_gpu":
        catalog = discover_otf_gpu_catalog(
            data_dir,
            baseline_pb_dir=baseline_pb_dir,
            baseline_split=baseline_split,
            segment_mac_dir=segment_mac_dir,
            patient_split_json=patient_split_json,
            validation_fraction=validation_fraction,
            seed=seed,
            shape=bin_shape,
            dose_dtype=otf_dose_dtype,
            ct_name=ct_name,
            otf_gpu_full=otf_gpu_full,
            bev_grid=bev_grid,
            otf_modality=otf_modality,
            proton_beam_parameters=proton_beam_parameters,
            sct_dir=sct_dir,
        )
        return compute_train_otf_gpu_stats(catalog)
    raise ValueError(f"data_format must be one of {MULTIMODAL_DATA_FORMATS}, got {data_format!r}")


def resolve_multimodal_data_dir(
    data_format: str,
    data_dir: str | Path | None,
    *,
    otf_gpu_full: bool = False,
    baseline_pb_dir: str | Path | None = None,
) -> Path:
    """Resolve the effective data root for multimodal loaders."""
    if data_format == "otf_gpu" and otf_gpu_full:
        if data_dir is not None and str(data_dir) != "":
            return Path(data_dir)
        if baseline_pb_dir is None or str(baseline_pb_dir) == "":
            raise ValueError("--baseline-pb-dir is required when --data-format otf_gpu --otf-gpu-full")
        return Path(baseline_pb_dir)
    if data_dir is None or str(data_dir) == "":
        raise ValueError(f"--data-dir is required for --data-format {data_format}")
    return Path(data_dir)


def _validate_proton_data_args(args: Any) -> None:
    """Checks that apply once --otf-modality proton is selected."""
    if getattr(args, "data_format", None) != "otf_gpu" or not bool(
        getattr(args, "otf_gpu_full", False)
    ):
        raise ValueError(
            "--otf-modality proton requires --data-format otf_gpu --otf-gpu-full"
        )
    beam_parameters = getattr(args, "proton_beam_parameters", None)
    if beam_parameters in (None, ""):
        raise ValueError("--otf-modality proton requires --proton-beam-parameters")
    if not Path(beam_parameters).is_file():
        raise FileNotFoundError(
            f"--proton-beam-parameters not found: {beam_parameters}"
        )
    if bool(getattr(args, "bev_depth_trim", False)):
        raise ValueError("--bev-depth-trim is not supported for proton otf_gpu")
    if int(getattr(args, "proton_fast_cache_patients", 4)) < 1:
        raise ValueError("--proton-fast-cache-patients must be at least 1")


def validate_multimodal_data_args(args: Any) -> None:
    """Validate CLI data arguments after ``parse_args()``."""
    otf_modality = str(getattr(args, "otf_modality", "photon"))
    if bool(getattr(args, "proton_energy_token", False)) and otf_modality != "proton":
        raise ValueError("--proton-energy-token requires --otf-modality proton")
    if otf_modality == "proton":
        _validate_proton_data_args(args)
    data_format = getattr(args, "data_format", "h5")
    data_dir = getattr(args, "data_dir", None)
    otf_gpu_full = bool(getattr(args, "otf_gpu_full", False))
    input_upscale_factor = int(
        getattr(args, "otf_input_upscale_factor", 1)
    )
    batched_affine = bool(
        getattr(args, "otf_batched_affine_preprocess", False)
    )
    if batched_affine and input_upscale_factor != 2:
        raise ValueError(
            "--otf-batched-affine-preprocess requires "
            "--otf-input-upscale-factor 2"
        )
    if int(getattr(args, "otf_batched_affine_cache_patients", 4)) < 1:
        raise ValueError(
            "--otf-batched-affine-cache-patients must be at least 1"
        )
    if input_upscale_factor > 1:
        if data_format != "otf_gpu" or not otf_gpu_full:
            raise ValueError(
                "--otf-input-upscale-factor 2 requires --data-format "
                "otf_gpu --otf-gpu-full"
            )
        if not bool(getattr(args, "ct_space_loss", False)):
            raise ValueError(
                "--otf-input-upscale-factor 2 requires --ct-space-loss"
            )
        if bool(getattr(args, "bev_depth_trim", False)):
            raise ValueError(
                "--otf-input-upscale-factor 2 is incompatible with "
                "--bev-depth-trim"
            )
        if bool(getattr(args, "ct_bev_idd_metrics", False)):
            raise ValueError(
                "--otf-input-upscale-factor 2 is incompatible with "
                "--ct-bev-idd-metrics"
            )
    sct_dir = getattr(args, "sct_dir", None)
    if sct_dir not in (None, ""):
        if data_format != "otf_gpu":
            raise ValueError("--sct-dir requires --data-format otf_gpu")
        if not Path(sct_dir).is_dir():
            raise ValueError(f"--sct-dir is not a directory: {sct_dir}")
    if bool(getattr(args, "ct_space_loss", False)):
        if data_format != "otf_gpu":
            raise ValueError("--ct-space-loss requires --data-format otf_gpu")
        if not otf_gpu_full:
            raise ValueError("--ct-space-loss requires --otf-gpu-full")
    if bool(getattr(args, "ct_space_bev_pixel_shuffle", False)):
        if not bool(getattr(args, "ct_space_loss", False)):
            raise ValueError("--ct-space-bev-pixel-shuffle requires --ct-space-loss")
    if bool(getattr(args, "ct_space_direct_spline_coefficients", False)):
        if not bool(getattr(args, "ct_space_loss", False)):
            raise ValueError(
                "--ct-space-direct-spline-coefficients requires --ct-space-loss"
            )
    if bool(getattr(args, "ct_bev_idd_metrics", False)):
        if not bool(getattr(args, "ct_space_loss", False)):
            raise ValueError("--ct-bev-idd-metrics requires --ct-space-loss")
    if data_format == "otf_gpu" and otf_gpu_full:
        if data_dir is not None and str(data_dir) != "":
            return
        if getattr(args, "baseline_pb_dir", None) in (None, ""):
            raise ValueError(
                "--baseline-pb-dir is required for --data-format otf_gpu with --otf-gpu-full "
                "when --data-dir is omitted"
            )
        return
    if data_dir is None or str(data_dir) == "":
        raise ValueError(f"--data-dir is required for --data-format {data_format}")
    bev_grid_path = getattr(args, "bev_grid_config", None)
    if bev_grid_path is not None and str(bev_grid_path) != "":
        path = Path(bev_grid_path)
        if not path.is_file():
            raise FileNotFoundError(f"--bev-grid-config not found: {path}")


def create_multimodal_train_val_datasets(
    data_format: Literal["h5", "bin", "merged_bin", "otf_gpu"],
    data_dir: str | Path | None,
    *,
    stats: dict[str, Any] | None = None,
    patient_split_json: str | Path | None = None,
    validation_fraction: float = 0.2,
    seed: int = 333,
    bin_shape: tuple[int, int, int] | None = None,
    bin_dtype: str | None = None,
    merged_storage_dtype: str = "float16",
    baseline_pb_dir: str | Path | None = None,
    baseline_split: str = "training",
    segment_mac_dir: str | Path | None = None,
    otf_dose_dtype: str = "float16",
    ct_name: str = "ct.mha",
    otf_gpu_full: bool = False,
    otf_stream_dose: bool = False,
    bev_grid: Any | None = None,
    otf_modality: str = "photon",
    proton_beam_parameters: str | Path | None = None,
    sct_dir: str | Path | None = None,
    input_transform=None,
    target_transform=None,
    train_augment: BEVAugmentConfig | None = None,
) -> tuple[
    "MultiModalHDF5Dataset | MultiModalBinDataset | MultiModalMergedBinDataset | OtfGpuDoseDataset",
    "MultiModalHDF5Dataset | MultiModalBinDataset | MultiModalMergedBinDataset | OtfGpuDoseDataset",
]:
    """Build train and validation datasets for CNN-Mamba / ConvLSTM / C3D base loaders."""
    data_dir = resolve_multimodal_data_dir(
        data_format,
        data_dir,
        otf_gpu_full=otf_gpu_full,
        baseline_pb_dir=baseline_pb_dir,
    )
    if data_format == "h5":
        ct_h5 = str(data_dir / "ct_dataset.h5")
        proj_h5 = str(data_dir / "proj_dataset.h5")
        dose_h5 = str(data_dir / "dose_dataset.h5")
        train_ds = MultiModalHDF5Dataset(
            ct_hdf_path=ct_h5,
            proj_hdf_path=proj_h5,
            dose_hdf_path=dose_h5,
            group_name="train",
            stats=stats,
            input_transform=input_transform,
            target_transform=target_transform,
            augment_config=train_augment,
        )
        val_ds = MultiModalHDF5Dataset(
            ct_hdf_path=ct_h5,
            proj_hdf_path=proj_h5,
            dose_hdf_path=dose_h5,
            group_name="validation",
            stats=stats,
            input_transform=input_transform,
            target_transform=target_transform,
            augment_config=None,
        )
        return train_ds, val_ds

    if data_format == "bin":
        catalog = discover_bin_catalog(
            data_dir,
            patient_split_json=patient_split_json,
            validation_fraction=validation_fraction,
            seed=seed,
            shape=bin_shape,
            dtype=bin_dtype,
        )
        train_ds = MultiModalBinDataset(
            catalog, group_name="train", stats=stats, augment_config=train_augment,
        )
        val_ds = MultiModalBinDataset(catalog, group_name="validation", stats=stats)
        return train_ds, val_ds

    if data_format == "merged_bin":
        catalog = discover_merged_bin_catalog(
            data_dir,
            storage_dtype=merged_storage_dtype,
            patient_split_json=patient_split_json,
            validation_fraction=validation_fraction,
            seed=seed,
        )
        train_ds = MultiModalMergedBinDataset(
            catalog, group_name="train", stats=stats, augment_config=train_augment,
        )
        val_ds = MultiModalMergedBinDataset(catalog, group_name="validation", stats=stats)
        return train_ds, val_ds

    if data_format == "otf_gpu":
        catalog = discover_otf_gpu_catalog(
            data_dir,
            baseline_pb_dir=baseline_pb_dir,
            baseline_split=baseline_split,
            segment_mac_dir=segment_mac_dir,
            patient_split_json=patient_split_json,
            validation_fraction=validation_fraction,
            seed=seed,
            shape=bin_shape,
            dose_dtype=otf_dose_dtype,
            ct_name=ct_name,
            otf_gpu_full=otf_gpu_full,
            bev_grid=bev_grid,
            otf_modality=otf_modality,
            proton_beam_parameters=proton_beam_parameters,
            sct_dir=sct_dir,
        )
        train_ds = OtfGpuDoseDataset(
            catalog, group_name="train", stream_full_dose=otf_stream_dose
        )
        val_ds = OtfGpuDoseDataset(
            catalog, group_name="validation", stream_full_dose=otf_stream_dose
        )
        return train_ds, val_ds

    raise ValueError(f"data_format must be one of {MULTIMODAL_DATA_FORMATS}, got {data_format!r}")


def add_multimodal_data_args(parser) -> None:
    """CLI flags shared by CNN-Mamba, CNN-ConvLSTM, and C3D trainers."""
    parser.add_argument(
        "--data-format",
        choices=MULTIMODAL_DATA_FORMATS,
        default="h5",
        help=(
            "Training data layout: 'h5' expects ct/proj/dose_dataset.h5 in --data-dir; "
            "'bin' expects --data-dir as BEV root with ct/, proj/, dose/ .bin folders; "
            "'merged_bin' expects a packed root from compare_merged_bin_precision_and_load.py "
            "(merged_bin_manifest.json and float16/ or float32/ samples/); "
            "'otf_gpu' reads dose/dose_*.bin by default and generates CT/proj on the training GPU; "
            "with --otf-gpu-full, dose BEV is also generated on GPU from baseline dose MHA files "
            "and --data-dir is optional."
        ),
    )
    parser.add_argument(
        "--otf-gpu-full",
        action="store_true",
        help=(
            "For --data-format otf_gpu: discover samples from --baseline-pb-dir plan JSON and "
            "dose/Dose_B*_CP*.mha, generate dose BEV on GPU (no dose/dose_*.bin under --data-dir). "
            "Requires --baseline-pb-dir and --segment-mac-dir; --data-dir is optional."
        ),
    )
    parser.add_argument(
        "--bev-grid-config",
        default=None,
        help=(
            "JSON file defining BEV cuboid shape_dhw, spacing_mm, sad_mm, and segment geometry. "
            "Default: built-in 256x200x200 @ 2 mm (see configs/bev_grid_default.json)."
        ),
    )
    parser.add_argument(
        "--patient-split-json",
        default=None,
        help=(
            "Patient split JSON (e.g. configs/bev_patient_split_seed333.json). "
            "Used for --data-format bin, merged_bin, or otf_gpu; ignored for h5 "
            "(split is in HDF5 groups)."
        ),
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.2,
        help=(
            "Patient-level val fraction for --data-format bin, merged_bin, or otf_gpu when "
            "--patient-split-json is omitted."
        ),
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=333,
        help=(
            "RNG seed for patient split when --data-format bin, merged_bin, or otf_gpu and no "
            "--patient-split-json."
        ),
    )
    parser.add_argument(
        "--merged-storage-dtype",
        default="float16",
        choices=BIN_FILE_DTYPES,
        help=(
            "On-disk dtype for --data-format merged_bin (float16/ or float32/ under --data-dir). "
            "Volumes are promoted to float32 in the dataset (default: float16)."
        ),
    )
    parser.add_argument(
        "--bin-shape",
        nargs=3,
        type=int,
        default=None,
        metavar=("Z", "X", "Y"),
        help="Volume shape for .bin files (default: infer from first CT bin).",
    )
    parser.add_argument(
        "--bin-dtype",
        default=None,
        choices=BIN_FILE_DTYPES,
        help=(
            "Raw dtype of .bin files (loaded as float32 for training). "
            "Default: bev_bin_dtype.txt under --data-dir if present, else float32."
        ),
    )
    parser.add_argument(
        "--baseline-pb-dir",
        default=None,
        help=(
            "DoseRAD root containing photon/<split>/<patient>/image/<ct-name>. "
            "Required for --data-format otf_gpu."
        ),
    )
    parser.add_argument(
        "--baseline-split",
        default="training",
        help="Split under --baseline-pb-dir/photon/ for --data-format otf_gpu (default: training).",
    )
    parser.add_argument(
        "--sct-dir",
        default=None,
        help=(
            "Optional flat directory of synthetic CTs (<patient-id>.mha, e.g. "
            ".../sCT/cGAN/photon) used as network input instead of "
            "--baseline-pb-dir/photon/<split>/<patient>/image/<ct-name>. Dose targets, "
            "plans, and apertures still come from --baseline-pb-dir, and spacing/origin "
            "are re-stamped from the real CT header (sCT exports carry no geometry). "
            "--data-format otf_gpu only."
        ),
    )
    parser.add_argument(
        "--segment-mac-dir",
        default=None,
        help=(
            "Directory containing mac/ and segments/ for --data-format otf_gpu."
        ),
    )
    parser.add_argument(
        "--otf-dose-dtype",
        default="float16",
        choices=BIN_FILE_DTYPES,
        help="On-disk dose dtype for --data-format otf_gpu (default: float16).",
    )
    parser.add_argument(
        "--otf-bev-mode",
        choices=("linear", "cubic"),
        default="cubic",
        help="CT interpolation mode for on-the-fly BEV input generation (default: cubic).",
    )
    parser.add_argument(
        "--ct-space-loss",
        action="store_true",
        help=(
            "Train against the original CT-space dose with differentiable cubic "
            "BEV-to-CT resampling. Requires otf_gpu, --otf-gpu-full, cubic mode, "
            "and a z-aligned BEV grid."
        ),
    )
    parser.add_argument(
        "--ct-space-bev-pixel-shuffle",
        action="store_true",
        help=(
            "With --ct-space-loss: model predicts dose on a finer BEV grid in depth "
            "and height using a phase-channel head and depth/height PixelShuffle. "
            "The factor is controlled by --ct-space-bev-upscale-factor. Requires "
            "--ct-space-loss."
        ),
    )
    parser.add_argument(
        "--ct-space-bev-upscale-factor",
        type=int,
        default=2,
        help=(
            "Depth/height upscale factor used by --ct-space-bev-pixel-shuffle "
            "(default: 2). Packed direct coefficients avoid materializing the "
            "resulting fine grid."
        ),
    )
    parser.add_argument(
        "--ct-space-iir-precision",
        choices=("fp32", "fp64"),
        default="fp32",
        help=(
            "Precision for the differentiable cubic IIR prefilter used by CT-space loss. "
            "Coefficients are sampled in float32 (default: fp32)."
        ),
    )
    parser.add_argument(
        "--ct-space-direct-spline-coefficients",
        action="store_true",
        help=(
            "Interpret the model output as cubic B-spline coefficients rather than "
            "grid samples. CT-space sampling then bypasses the differentiable IIR "
            "prefilter. Requires --ct-space-loss; combine with "
            "--ct-space-bev-pixel-shuffle for fine-grid coefficients."
        ),
    )
    parser.add_argument(
        "--ct-bev-idd-metrics",
        action="store_true",
        help=(
            "During validation, compute the RMS IDD of the CT-space prediction mapped "
            "back to BEV (logged as bev_idd_from_ct). Reconstructed from the CT-space "
            "prediction, so it works with the direct-coefficient heads where the standard "
            "BEV metrics do not. Requires --ct-space-loss."
        ),
    )
    parser.add_argument(
        "--ct-space-valid-voxel-mse",
        action="store_true",
        help=(
            "For CT-space training, compute MSE over valid sampler voxels only "
            "instead of retaining invalid zero-filled voxels in the ROI denominator."
        ),
    )
    parser.add_argument(
        "--ct-space-negative-dose-weight",
        type=float,
        default=0.0,
        help=(
            "Weight for mean squared negative sampled CT dose, evaluated over valid "
            "voxels. This constrains reconstructed dose, not spline coefficients "
            "(default: 0, disabled)."
        ),
    )
    parser.add_argument(
        "--otf-gpu-ct-cache-patients",
        type=int,
        default=0,
        help=(
            "Max patient CTs cached in VRAM for --data-format otf_gpu. "
            "0 means no LRU cap (cache all active patients; default)."
        ),
    )
    parser.add_argument(
        "--otf-collapsed-ct-dtype",
        choices=("float32", "float16", "bfloat16"),
        default="float32",
        help=(
            "Storage dtype for the z-collapsed CT planes held in each cached "
            "patient plan under --otf-batched-affine-preprocess (default: "
            "float32). float16 halves that cache's VRAM at ~1 HU quantisation; "
            "bfloat16 matches the inference default but quantises to ~8 HU."
        ),
    )
    parser.add_argument(
        "--otf-cache-dose",
        action="store_true",
        help=(
            "For --data-format otf_gpu --otf-gpu-full: keep loaded CT-coordinate dose MHA "
            "volumes in RAM across batches. Off by default to avoid unbounded RAM growth; "
            "dose is re-read from disk each time a sample is materialized."
        ),
    )
    parser.add_argument(
        "--otf-stream-dose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "For --data-format otf_gpu --otf-gpu-full: load CT-space dose MHA "
            "volumes in DataLoader workers so pin_memory can stage asynchronous "
            "GPU transfers (default: enabled; use --no-otf-stream-dose for the "
            "legacy synchronous main-process loader)."
        ),
    )
    parser.add_argument(
        "--ct-name",
        default="ct.mha",
        help="CT filename under each patient image/ directory for --data-format otf_gpu (default: ct.mha).",
    )
    parser.add_argument(
        "--bev-depth-trim",
        action="store_true",
        help=(
            "Option-2 BEV depth trim for --data-format otf_gpu: subgrid prepare_bev_input_volumes "
            "and dose[i0:i1+1] using per-sample cached depth ranges."
        ),
    )
    parser.add_argument(
        "--bev-depth-range-cache",
        default=None,
        help=(
            "JSON cache of per-sample BEV depth indices. Default: bev_depth_range.json under "
            "--out-dir or --data-dir when --bev-depth-trim is set."
        ),
    )
    parser.add_argument(
        "--bev-hu-thresh",
        type=float,
        default=-500.0,
        help="HU threshold for body mask when computing BEV depth range (default: -500).",
    )
    parser.add_argument(
        "--bev-depth-margin-slices",
        type=int,
        default=4,
        help="Extra BEV depth slices added on each side of projected body extent (default: 4).",
    )
    parser.add_argument(
        "--bev-depth-range-stride",
        type=int,
        default=8,
        help="Coarse CT subsampling stride for depth-range body mask (default: 8).",
    )
    parser.add_argument(
        "--bev-depth-range-build",
        action="store_true",
        help=(
            "Precompute --bev-depth-range-cache for the otf_gpu catalog (train+val+test) and exit. "
            "Requires --data-format otf_gpu."
        ),
    )

    # -- proton beamlet inputs ------------------------------------------
    parser.add_argument(
        "--otf-modality",
        choices=("photon", "proton"),
        default="photon",
        help=(
            "Geometry/input modality for --data-format otf_gpu "
            "(default: photon). Proton mode discovers ray/beamlet records "
            "from patient plan JSON files and requires --otf-gpu-full."
        ),
    )
    parser.add_argument(
        "--proton-beam-parameters",
        default=None,
        help=(
            "JSON containing proton.energy_table. Required with "
            "--otf-modality proton."
        ),
    )
    parser.add_argument(
        "--proton-input-upscale-factor",
        type=int,
        choices=(1, 2),
        default=2,
        help=(
            "Sample proton CT, fluence, RSP, WET and remaining range at this "
            "finer depth/height resolution, then inverse-pixel-shuffle onto "
            "the model lattice (default: 2)."
        ),
    )
    parser.add_argument(
        "--proton-density-calibration",
        choices=("legacy", "g4dcm", "g4dcm_rsp"),
        default="g4dcm_rsp",
        help="HU material calibration used for proton range conditioning.",
    )
    parser.add_argument(
        "--proton-conditioning",
        choices=("range_wet",),
        default="range_wet",
        help=(
            "Proton spatial conditioning supplied to the network. range_wet "
            "uses phase-packed fixed WET and remaining CSDA range channels."
        ),
    )
    parser.add_argument(
        "--proton-energy-token",
        action="store_true",
        help=(
            "Prepend one learned discrete-energy token to the temporal xLSTM "
            "sequence. The token index follows the sorted proton.energy_table "
            "in --proton-beam-parameters and complements the spatial WET/range "
            "conditioning (default: disabled)."
        ),
    )
    parser.add_argument(
        "--proton-body-threshold",
        type=float,
        default=0.1,
        help=(
            "Body-mask threshold on the normalised [0, 1] anatomy for "
            "--proton-range-source geometric (default: 0.1)."
        ),
    )
    parser.add_argument(
        "--proton-range-refiner",
        choices=("fixed",),
        default="fixed",
        help=(
            "Range-channel representation. fixed preserves the G4DCM RSP "
            "integral and empirical CSDA curve (default)."
        ),
    )
    parser.add_argument(
        "--proton-output-refiner",
        choices=("none", "bragg_residual"),
        default="none",
        help=(
            "Optional identity-initialized refinement of packed proton dose "
            "coefficients before CT-space spline sampling."
        ),
    )
    parser.add_argument(
        "--proton-fast-preprocess",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use patient-cached affine/Triton CT and fluence preprocessing for "
            "proton training (default: enabled)."
        ),
    )
    parser.add_argument(
        "--proton-fast-cache-patients",
        type=int,
        default=4,
        help="Maximum patient proton affine caches retained in VRAM (default: 4).",
    )
    parser.add_argument(
        "--proton-ray-aware-batches",
        action="store_true",
        help=(
            "Optionally keep energy layers from a proton ray in the same "
            "training batch so sampled CT and material state can be reused."
        ),
    )


def add_otf_input_upscale_arg(parser) -> None:
    """Add the CNN-xLSTM fine-input phase-packing option."""
    parser.add_argument(
        "--otf-input-upscale-factor",
        type=int,
        choices=(1, 2),
        default=1,
        help=(
            "For otf_gpu CNN-xLSTM training, sample CT and projected aperture "
            "at this finer depth/height resolution and inverse-pixel-shuffle "
            "the phases onto the configured BEV lattice. 1 preserves legacy "
            "single-channel inputs; 2 supplies four phase channels "
            "(default: 1)."
        ),
    )
    parser.add_argument(
        "--otf-batched-affine-preprocess",
        action="store_true",
        help=(
            "Generate fine CT/aperture inputs with the batched affine Triton "
            "path, without dense inverse-coordinate tensors. Requires "
            "--otf-input-upscale-factor 2 and aligned cubic CT preprocessing."
        ),
    )
    parser.add_argument(
        "--otf-batched-affine-cache-patients",
        type=int,
        default=4,
        help=(
            "Maximum number of patient affine preprocessing caches retained "
            "on GPU (default: 4)."
        ),
    )


def compute_train_h5_stats(
    ct_hdf_path: str,
    dose_hdf_path: str,
    group_name: str = "train",
) -> dict[str, float]:
    """Global dose scale from training split (max voxel over all train dose volumes).

    CT bounds match ``dota_torch.data.compute_photon_stats`` / ``normalise_ct``:
    clip to [ct_min, ct_max] then linear map to [0, 1] in the dataset when ``stats`` is set.
    """
    dose_scale = 0.0
    with h5py.File(dose_hdf_path, "r") as f_dose:
        if group_name not in f_dose:
            raise ValueError(f"Group '{group_name}' not found in {dose_hdf_path}")
        keys = sorted(f_dose[group_name].keys())
        if not keys:
            raise ValueError(f"No samples in '{group_name}' of {dose_hdf_path}")
        for k in tqdm(keys, desc="Computing dose_scale (train H5)", leave=False):
            d = f_dose[group_name][k][()]
            dose_scale = max(dose_scale, float(np.max(d)))
    if dose_scale <= 0:
        raise ValueError("Cannot compute dose_scale: all training dose volumes are non-positive")
    # ct_hdf_path unused but kept for API symmetry / future CT-based stats
    _ = ct_hdf_path
    return {
        "ct_min": -1024.0,
        "ct_max": 3071.0,
        "dose_scale": dose_scale,
    }


def _apply_stats_to_arrays(
    ct_data: np.ndarray,
    dose_data: np.ndarray,
    stats: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    ct_min = float(stats["ct_min"])
    ct_max = float(stats["ct_max"])
    dose_scale = float(stats["dose_scale"])
    if dose_scale <= 0:
        raise ValueError("stats['dose_scale'] must be positive")

    ct = torch.from_numpy(ct_data.astype(np.float32))
    ct = torch.clamp(ct, ct_min, ct_max)
    ct = (ct - ct_min) / (ct_max - ct_min)

    dose = torch.from_numpy(dose_data.astype(np.float32)) / dose_scale
    return ct, dose


@dataclass(frozen=True)
class BEVAugmentConfig:
    """Train-only random H/W flips for BEV volumes shaped ``(D, H, W)``."""

    p_h: float = 0.5
    p_w: float = 0.5

    def __post_init__(self) -> None:
        for name, value in (("p_h", self.p_h), ("p_w", self.p_w)):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"BEVAugmentConfig.{name} must be in [0, 1], got {value}")


def apply_random_bev_flip(
    ct: torch.Tensor,
    proj: torch.Tensor,
    dose: torch.Tensor,
    config: BEVAugmentConfig,
    *,
    rng: random.Random | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Flip axes 1 (H) and/or 2 (W) consistently across CT, projection, and dose."""
    rng = rng or random
    dims: list[int] = []
    if rng.random() < config.p_h:
        dims.append(1)
    if rng.random() < config.p_w:
        dims.append(2)
    if not dims:
        return ct, proj, dose
    return ct.flip(dims), proj.flip(dims), dose.flip(dims)


def build_bev_augment_config_from_args(args) -> BEVAugmentConfig | None:
    if not getattr(args, "aug_flip", False):
        return None
    return BEVAugmentConfig(p_h=args.aug_flip_p, p_w=args.aug_flip_p)


def add_augmentation_args(parser) -> None:
    parser.add_argument(
        "--aug-flip",
        action="store_true",
        help=(
            "Enable train-only random H/W flips (BEV axes 1 and 2) applied consistently "
            "to CT, projection, and dose."
        ),
    )
    parser.add_argument(
        "--aug-flip-p",
        type=float,
        default=0.5,
        help="Per-axis flip probability for H and W when --aug-flip is set (default: 0.5).",
    )


def masked_mae_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.10,
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    """Differentiable masked MAE matching ``validation.masked_mae`` (per volume in batch).

    For each batch item, let ``max_gt = max(target)``. The mask is
    ``target >= threshold * max_gt``. The loss term is
    ``mean(|pred - target| on mask) / max_gt``.

    Batch items with ``max_gt <= 0`` or an empty mask are skipped in the mean.
    If no batch item is valid, returns ``(pred * 0).sum()`` so the graph exists.

    Args:
        pred: Predictions, same shape as ``target`` (e.g. ``(B, T, H, W)``).
        target: Ground-truth dose.
        threshold: Fraction of per-item maximum defining the high-dose mask (default 0.1).
        reduction: ``"mean"`` (mean over valid batch items), ``"sum"``, or ``"none"``
            (per-item values, 0 where invalid).
    """
    if pred.shape != target.shape:
        raise ValueError(f"pred shape {pred.shape} != target shape {target.shape}")
    if threshold < 0:
        raise ValueError("threshold must be non-negative")

    if pred.ndim < 2:
        raise ValueError("expected pred with batch dimension (B, ...)")

    max_gt = target.amax(dim=tuple(range(1, pred.ndim)))  # (B,)
    expand = max_gt.view(-1, *([1] * (pred.ndim - 1)))
    thr_map = threshold * expand
    mask = (target >= thr_map) & (max_gt > 0).view(-1, *([1] * (pred.ndim - 1)))

    abs_err = (pred - target).abs()
    sum_e = (abs_err * mask.to(abs_err.dtype)).flatten(1).sum(1)
    cnt = mask.flatten(1).sum(1).to(dtype=pred.dtype)
    valid = (max_gt > 0) & (cnt > 0)
    per = torch.zeros(pred.shape[0], device=pred.device, dtype=pred.dtype)
    # amax/divisions may run in float32 under AMP; indexed put requires matching dtypes
    vals = (sum_e[valid] / cnt[valid]) / max_gt[valid]
    per[valid] = vals.to(dtype=per.dtype)

    if reduction == "mean":
        if not valid.any():
            return pred.sum() * 0.0
        return per[valid].mean()
    if reduction == "sum":
        return per[valid].sum() if valid.any() else pred.sum() * 0.0
    if reduction == "none":
        return per
    raise ValueError(f"reduction must be 'mean', 'sum', or 'none', got {reduction!r}")


def beam_gamma_hinge_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor | None = None,
    *,
    dose_percent: float = 0.01,
    margin: float = 0.8,
    temperature: float = 0.1,
    denominator_floor: float = 0.02,
    dose_weight_saturation: float = 0.10,
) -> torch.Tensor:
    """Smooth control-point surrogate for summed-beam dose agreement.

    The dimensionless error is ``abs(pred-target) / (dose_percent * denom)``,
    where ``denom=max(target, denominator_floor * max(target))`` per sample.
    A soft hinge starts at ``margin`` (0.8 therefore encourages error below
    0.8% for a 1% criterion). Voxels are weighted linearly by reference dose
    until ``dose_weight_saturation * max(target)`` and then receive unit weight.
    This deliberately includes low-dose control-point contributions which can
    accumulate above the evaluation cutoff after a beam is summed.

    This is a memory-bounded surrogate rather than a literal full-beam loss:
    ordinary shuffled training does not retain every control point in a beam.
    """
    if pred.shape != target.shape:
        raise ValueError(f"pred shape {pred.shape} != target shape {target.shape}")
    if pred.ndim < 2:
        raise ValueError("expected pred with batch dimension (B, ...)")
    if valid is not None and valid.shape != target.shape:
        raise ValueError(f"valid shape {valid.shape} != target shape {target.shape}")
    if dose_percent <= 0:
        raise ValueError("dose_percent must be positive")
    if margin < 0:
        raise ValueError("margin must be non-negative")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if denominator_floor <= 0:
        raise ValueError("denominator_floor must be positive")
    if dose_weight_saturation <= 0:
        raise ValueError("dose_weight_saturation must be positive")

    spatial_dims = tuple(range(1, target.ndim))
    max_gt = target.amax(dim=spatial_dims)
    expand = max_gt.view(-1, *([1] * (target.ndim - 1)))
    denom = torch.maximum(target.clamp_min(0.0), denominator_floor * expand)
    relative_error = (pred - target).abs() / (dose_percent * denom.clamp_min(1e-12))
    hinge = temperature * torch.nn.functional.softplus(
        (relative_error - margin) / temperature
    )
    weights = (target.clamp_min(0.0) / (dose_weight_saturation * expand).clamp_min(1e-12))
    weights = weights.clamp(max=1.0)
    weights = weights * (max_gt > 0).view(
        -1, *([1] * (target.ndim - 1))
    ).to(weights.dtype)
    if valid is not None:
        weights = weights * valid.to(weights.dtype)
    weight_sum = weights.flatten(1).sum(1)
    per = (hinge * weights).flatten(1).sum(1) / weight_sum.clamp_min(1.0)
    usable = weight_sum > 0
    if not usable.any():
        return pred.sum() * 0.0
    return per[usable].mean()


def _squeeze_channel_if_5d(pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Accept ``(B, T, H, W)`` or ``(B, 1, T, H, W)`` (validation layout)."""
    if pred.ndim == 5 and target.ndim == 5 and pred.shape[1] == 1 and target.shape[1] == 1:
        return pred.squeeze(1), target.squeeze(1)
    return pred, target


def idd_curve_distance_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    metric: Literal["rms", "mae"] = "rms",
    reduction: str = "mean",
    eps: float = 1e-12,
) -> torch.Tensor:
    """Differentiable IDD curve distance with GT-peak normalisation.

    Per batch item, sum dose over the transverse plane (last two spatial dims) to get
    IDD curves along depth (``T`` / axis 0 in ``(T, H, W)``).
    ``metric='rms'`` computes ``sqrt(mean((idd_pred - idd_gt)^2)) / peak_gt``.
    ``metric='mae'`` computes ``mean(abs(idd_pred/peak_gt - idd_gt/peak_gt))``.

    Items with ``peak_gt <= 0`` are skipped in ``mean`` / ``sum`` reductions.
    """
    pred, target = _squeeze_channel_if_5d(pred, target)
    if pred.shape != target.shape:
        raise ValueError(f"pred shape {pred.shape} != target shape {target.shape}")
    if pred.ndim < 3:
        raise ValueError("expected pred with shape (B, depth, H, W) or (B, 1, depth, H, W)")

    idd_pred = pred.sum(dim=(-2, -1))
    idd_gt = target.sum(dim=(-2, -1))
    peak_gt = idd_gt.amax(dim=-1)
    valid = peak_gt > 0

    denom = peak_gt.clamp_min(eps)
    if metric == "rms":
        mse_1d = (idd_pred - idd_gt).pow(2).mean(dim=-1)
        per = torch.sqrt(mse_1d + eps) / denom
    elif metric == "mae":
        per = (
            (idd_pred / denom.unsqueeze(-1)) - (idd_gt / denom.unsqueeze(-1))
        ).abs().mean(dim=-1)
    else:
        raise ValueError(f"metric must be 'rms' or 'mae', got {metric!r}")
    per = torch.where(valid, per, torch.zeros_like(per))

    if reduction == "mean":
        if not valid.any():
            return pred.sum() * 0.0
        return per[valid].mean()
    if reduction == "sum":
        return per[valid].sum() if valid.any() else pred.sum() * 0.0
    if reduction == "none":
        return per
    raise ValueError(f"reduction must be 'mean', 'sum', or 'none', got {reduction!r}")


def _gradient_along_dim(vol: torch.Tensor, dim: int) -> torch.Tensor:
    """Central differences matching ``numpy.gradient`` with unit spacing."""
    n = vol.size(dim)
    if n == 1:
        return torch.zeros_like(vol)
    if n == 2:
        d = torch.diff(vol, dim=dim)
        return torch.cat([d, d], dim=dim)

    lead = (vol.select(dim, 1) - vol.select(dim, 0)).unsqueeze(dim)
    trail = (vol.select(dim, n - 1) - vol.select(dim, n - 2)).unsqueeze(dim)
    interior = (vol.narrow(dim, 2, n - 2) - vol.narrow(dim, 0, n - 2)) * 0.5
    return torch.cat([lead, interior, trail], dim=dim)


def _grad_magnitude_3d(vol: torch.Tensor) -> torch.Tensor:
    """3-D gradient magnitude on the last three tensor dimensions (D, H, W)."""
    if vol.ndim < 3:
        raise ValueError(f"expected at least 3 spatial dims, got shape {vol.shape}")
    spatial_dims = tuple(range(vol.ndim - 3, vol.ndim))
    grads = [_gradient_along_dim(vol, dim) for dim in spatial_dims]
    return torch.sqrt(grads[0].pow(2) + grads[1].pow(2) + grads[2].pow(2))


def gradient_mae_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    """Differentiable gradient magnitude MAE matching ``validation.gradient_mae``.

    Per batch item, compute ``mean(| |∇pred| - |∇target| |)`` over all voxels.
    Central differences along depth and transverse axes match ``numpy.gradient``.
    Operates on normalized model dose when inputs are normalized.
    """
    pred, target = _squeeze_channel_if_5d(pred, target)
    if pred.shape != target.shape:
        raise ValueError(f"pred shape {pred.shape} != target shape {target.shape}")
    if pred.ndim < 4:
        raise ValueError("expected pred with shape (B, depth, H, W) or (B, 1, depth, H, W)")

    mag_pred = _grad_magnitude_3d(pred)
    mag_tgt = _grad_magnitude_3d(target)
    per = (mag_pred - mag_tgt).abs().flatten(1).mean(1)

    if reduction == "mean":
        return per.mean()
    if reduction == "sum":
        return per.sum()
    if reduction == "none":
        return per
    raise ValueError(f"reduction must be 'mean', 'sum', or 'none', got {reduction!r}")


class MaskedMAELoss(nn.Module):
    """``nn.Module`` wrapper around :func:`masked_mae_loss` (reduction ``mean``)."""

    def __init__(self, threshold: float = 0.10) -> None:
        super().__init__()
        self.threshold = threshold

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return masked_mae_loss(pred, target, self.threshold, reduction="mean")


def _squeeze_dose_channel(pred: torch.Tensor) -> torch.Tensor:
    """``(B, 1, D, H, W)`` or ``(B, D, H, W)`` -> ``(B, D, H, W)``."""
    if pred.ndim == 5 and pred.shape[1] == 1:
        return pred.squeeze(1)
    return pred


def make_primary_regression_loss(
    loss_type: Literal["mse", "mae", "huber"],
    *,
    huber_delta: float = 1.0,
) -> nn.Module:
    """Voxel-wise MSE, MAE, or Huber (Smooth L1) loss with mean reduction."""
    name = loss_type.lower()
    if name == "mse":
        return nn.MSELoss()
    if name == "mae":
        return nn.L1Loss()
    if name == "huber":
        return nn.HuberLoss(delta=float(huber_delta))
    raise ValueError(f"regression loss must be mse, mae, or huber, got {loss_type!r}")


def add_gradient_mae_loss_arg(parser) -> None:
    """Add ``--gradient-mae-weight`` for optional gradient magnitude MAE loss term."""
    parser.add_argument(
        "--gradient-mae-weight",
        type=float,
        default=0.0,
        help=(
            "If > 0, add this weight times gradient magnitude MAE "
            "(same definition as validation.gradient_mae) to the loss."
        ),
    )


def add_regression_loss_args(
    parser,
    *,
    default: Literal["mse", "mae"] = "mse",
    choices: Sequence[str] = ("mse", "huber"),
    include_primary_weight: bool = True,
) -> None:
    """Add ``--regression-loss``, optional ``--primary-loss-weight``, and ``--huber-delta``."""
    parser.add_argument(
        "--regression-loss",
        choices=tuple(choices),
        default=default,
        help=f"Primary voxel-wise loss (default: {default}).",
    )
    if include_primary_weight:
        parser.add_argument(
            "--primary-loss-weight",
            type=float,
            default=1.0,
            help=(
                "Weight for the --regression-loss primary voxel-wise term (default: 1.0). "
                "Set to 0 to omit it (requires another loss term with weight > 0)."
            ),
        )
    parser.add_argument(
        "--huber-delta",
        type=float,
        default=1.0,
        help="Delta for --regression-loss huber (PyTorch HuberLoss / Smooth L1).",
    )


def _regression_loss_label(loss_type: str, huber_delta: float) -> str:
    name = loss_type.upper()
    if loss_type.lower() == "huber":
        return f"{name}(delta={huber_delta:g})"
    return name


class RegressionCompositeLoss(nn.Module):
    """Primary regression + optional masked MAE, IDD distance, and gradient MAE terms."""

    def __init__(
        self,
        *,
        primary_loss: Literal["mse", "mae", "huber"] = "mse",
        primary_weight: float = 1.0,
        huber_delta: float = 1.0,
        masked_mae_weight: float = 0.0,
        masked_mae_threshold: float = 0.10,
        idd_weight: float = 0.0,
        idd_metric: Literal["rms", "mae"] = "rms",
        gradient_mae_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.primary = make_primary_regression_loss(primary_loss, huber_delta=huber_delta)
        self.primary_loss = primary_loss.lower()
        self.primary_weight = float(primary_weight)
        self.huber_delta = float(huber_delta)
        self.masked_mae_weight = float(masked_mae_weight)
        self.masked_mae_threshold = float(masked_mae_threshold)
        self.idd_weight = float(idd_weight)
        self.idd_metric = idd_metric
        self.gradient_mae_weight = float(gradient_mae_weight)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        terms: list[torch.Tensor] = []
        if self.primary_weight != 0.0:
            terms.append(self.primary_weight * self.primary(pred, target))
        if self.masked_mae_weight != 0.0:
            terms.append(
                self.masked_mae_weight
                * masked_mae_loss(
                    pred, target, self.masked_mae_threshold, reduction="mean"
                )
            )
        if self.idd_weight != 0.0:
            terms.append(
                self.idd_weight
                * idd_curve_distance_loss(
                    pred, target, metric=self.idd_metric, reduction="mean"
                )
            )
        if self.gradient_mae_weight != 0.0:
            terms.append(
                self.gradient_mae_weight
                * gradient_mae_loss(pred, target, reduction="mean")
            )
        if not terms:
            return pred.sum() * 0.0
        loss = terms[0]
        for term in terms[1:]:
            loss = loss + term
        return loss


class MSEMaskedMAECompositeLoss(RegressionCompositeLoss):
    """Backward-compatible alias: MSE primary term + optional masked MAE / IDD."""

    def __init__(
        self,
        mse_weight: float = 1.0,
        masked_mae_weight: float = 0.0,
        masked_mae_threshold: float = 0.10,
        idd_weight: float = 0.0,
    ) -> None:
        super().__init__(
            primary_loss="mse",
            primary_weight=mse_weight,
            masked_mae_weight=masked_mae_weight,
            masked_mae_threshold=masked_mae_threshold,
            idd_weight=idd_weight,
        )


def _primary_loss_weight_label(primary_weight: float, loss_label: str) -> str:
    if primary_weight == 1.0:
        return loss_label
    return f"{primary_weight:g} * {loss_label}"


def _validate_composite_loss_weights(
    *,
    primary_weight: float,
    masked_mae_weight: float,
    idd_weight: float,
    gradient_mae_weight: float,
) -> None:
    if (
        primary_weight == 0.0
        and masked_mae_weight <= 0.0
        and idd_weight <= 0.0
        and gradient_mae_weight <= 0.0
    ):
        raise SystemExit(
            "At least one loss term must have weight > 0 "
            "(--primary-loss-weight, --masked-mae-weight, --idd-weight, "
            "or --gradient-mae-weight)."
        )


def build_mamba_convlstm_criterion(args) -> tuple[nn.Module, str]:
    """Build loss module and human-readable description for CNN-Mamba / ConvLSTM."""
    loss_label = _regression_loss_label(args.regression_loss, args.huber_delta)
    primary_weight = float(getattr(args, "primary_loss_weight", 1.0))
    idd_weight = float(getattr(args, "idd_weight", 0.0))
    gradient_mae_weight = float(getattr(args, "gradient_mae_weight", 0.0))
    idd_metric = str(getattr(args, "idd_metric", "rms")).lower()
    use_composite = (
        primary_weight != 1.0
        or args.masked_mae_weight > 0
        or idd_weight > 0
        or gradient_mae_weight > 0
    )
    if use_composite:
        _validate_composite_loss_weights(
            primary_weight=primary_weight,
            masked_mae_weight=float(args.masked_mae_weight),
            idd_weight=idd_weight,
            gradient_mae_weight=gradient_mae_weight,
        )
        criterion = RegressionCompositeLoss(
            primary_loss=args.regression_loss,
            primary_weight=primary_weight,
            huber_delta=args.huber_delta,
            masked_mae_weight=args.masked_mae_weight,
            masked_mae_threshold=args.masked_mae_threshold,
            idd_weight=idd_weight,
            idd_metric=idd_metric,
            gradient_mae_weight=gradient_mae_weight,
        )
        parts: list[str] = []
        if primary_weight != 0.0:
            parts.append(_primary_loss_weight_label(primary_weight, loss_label))
        if args.masked_mae_weight > 0:
            parts.append(
                f"{args.masked_mae_weight} * masked_mae "
                f"(threshold={args.masked_mae_threshold})"
            )
        if idd_weight > 0:
            parts.append(f"{idd_weight} * idd_distance({idd_metric})")
        if gradient_mae_weight > 0:
            parts.append(f"{gradient_mae_weight} * gradient_mae")
        return criterion, " + ".join(parts)

    criterion = make_primary_regression_loss(
        args.regression_loss, huber_delta=args.huber_delta
    )
    return criterion, loss_label


class C3DDualHeadMAELoss(nn.Module):
    """Dual-head C3D loss on both outputs + optional masked MAE / IDD on ``output_B``."""

    def __init__(
        self,
        mae_b_weight: float = 1.0,
        mae_a_weight: float = 0.5,
        masked_mae_weight: float = 0.0,
        masked_mae_threshold: float = 0.10,
        idd_weight: float = 0.0,
        idd_metric: Literal["rms", "mae"] = "rms",
        gradient_mae_weight: float = 0.0,
        *,
        primary_loss: Literal["mae", "huber"] = "mae",
        huber_delta: float = 1.0,
    ) -> None:
        super().__init__()
        self.primary = make_primary_regression_loss(primary_loss, huber_delta=huber_delta)
        self.primary_loss = primary_loss.lower()
        self.huber_delta = float(huber_delta)
        self.mae_b_weight = float(mae_b_weight)
        self.mae_a_weight = float(mae_a_weight)
        self.masked_mae_weight = float(masked_mae_weight)
        self.masked_mae_threshold = float(masked_mae_threshold)
        self.idd_weight = float(idd_weight)
        self.idd_metric = idd_metric
        self.gradient_mae_weight = float(gradient_mae_weight)

    def forward(
        self,
        outputs: torch.Tensor | Sequence[torch.Tensor],
        target: torch.Tensor,
    ) -> torch.Tensor:
        if not isinstance(outputs, (list, tuple)) or len(outputs) != 2:
            raise TypeError("expected outputs as (output_A, output_B) from C3D Model")
        out_a = _squeeze_dose_channel(outputs[0])
        out_b = _squeeze_dose_channel(outputs[1])
        if out_a.shape != target.shape or out_b.shape != target.shape:
            raise ValueError(
                f"dose shape mismatch: A {out_a.shape} B {out_b.shape} target {target.shape}"
            )

        loss = torch.zeros((), device=target.device, dtype=target.dtype)
        if self.mae_b_weight != 0.0:
            loss = loss + self.mae_b_weight * self.primary(out_b, target)
        if self.mae_a_weight != 0.0:
            loss = loss + self.mae_a_weight * self.primary(out_a, target)
        if self.masked_mae_weight != 0.0:
            loss = loss + self.masked_mae_weight * masked_mae_loss(
                out_b, target, self.masked_mae_threshold, reduction="mean"
            )
        if self.idd_weight != 0.0:
            loss = loss + self.idd_weight * idd_curve_distance_loss(
                out_b, target, metric=self.idd_metric, reduction="mean"
            )
        if self.gradient_mae_weight != 0.0:
            loss = loss + self.gradient_mae_weight * gradient_mae_loss(
                out_b, target, reduction="mean"
            )
        return loss


def build_c3d_criterion(args) -> tuple[C3DDualHeadMAELoss, str]:
    """Build C3D dual-head loss and human-readable description."""
    loss_label = _regression_loss_label(args.regression_loss, args.huber_delta)
    idd_metric = str(getattr(args, "idd_metric", "rms")).lower()
    criterion = C3DDualHeadMAELoss(
        mae_b_weight=args.mae_b_weight,
        mae_a_weight=args.mae_a_weight,
        masked_mae_weight=args.masked_mae_weight,
        masked_mae_threshold=args.masked_mae_threshold,
        idd_weight=args.idd_weight,
        idd_metric=idd_metric,
        gradient_mae_weight=float(getattr(args, "gradient_mae_weight", 0.0)),
        primary_loss=args.regression_loss,
        huber_delta=args.huber_delta,
    )
    parts = [
        f"{args.mae_b_weight} * {loss_label}(output_B, dose)",
        f"{args.mae_a_weight} * {loss_label}(output_A, dose)",
    ]
    if args.masked_mae_weight > 0:
        parts.append(
            f"{args.masked_mae_weight} * masked_mae_B "
            f"(threshold={args.masked_mae_threshold})"
        )
    if args.idd_weight > 0:
        parts.append(f"{args.idd_weight} * idd_distance_B({idd_metric})")
    if getattr(args, "gradient_mae_weight", 0.0) > 0:
        parts.append(f"{args.gradient_mae_weight} * gradient_mae_B")
    return criterion, " + ".join(parts)


class MultiModalHDF5Dataset(Dataset):
    def __init__(self, ct_hdf_path: str, proj_hdf_path: str, dose_hdf_path: str,
                 group_name: str = 'train',
                 input_transform=None, target_transform=None,
                 stats: dict[str, Any] | None = None,
                 augment_config: BEVAugmentConfig | None = None):
        """
        Args:
            ct_hdf_path (str): Path to the CT HDF5 file.
            proj_hdf_path (str): Path to the projection HDF5 file.
            dose_hdf_path (str): Path to the dose HDF5 file.
            group_name (str): HDF5 group name to read ('train' or 'validation').
            input_transform (callable, optional): Transform applied to input data.
            target_transform (callable, optional): Transform applied to target data.
            stats (dict, optional): CT clip + [0,1] via ``ct_min``/``ct_max``; dose ``/ dose_scale``.
        """
        self.ct_hdf_path   = ct_hdf_path
        self.proj_hdf_path = proj_hdf_path
        self.dose_hdf_path = dose_hdf_path
        self.group_name      = group_name
        self.input_transform  = input_transform
        self.target_transform = target_transform
        self.stats = stats
        self.augment_config = augment_config

        # File handles, opened independently per DataLoader worker
        self.ct_file   = None
        self.proj_file = None
        self.dose_file = None

        # On init, retrieve all keys and verify consistency across files
        with h5py.File(self.ct_hdf_path, 'r') as f:
            if group_name not in f:
                raise ValueError(f"Group '{group_name}' not found in {self.ct_hdf_path}")
            self.ct_keys = sorted(list(f[group_name].keys()))

        with h5py.File(self.proj_hdf_path, 'r') as f:
            if group_name not in f:
                raise ValueError(f"Group '{group_name}' not found in {self.proj_hdf_path}")
            self.proj_keys = sorted(list(f[group_name].keys()))

        with h5py.File(self.dose_hdf_path, 'r') as f:
            if group_name not in f:
                raise ValueError(f"Group '{group_name}' not found in {self.dose_hdf_path}")
            self.dose_keys = sorted(list(f[group_name].keys()))

        # Verify that all three files have the same number of samples
        if not (len(self.ct_keys) == len(self.proj_keys) == len(self.dose_keys)):
            raise ValueError(
                f"Sample count mismatch in group '{group_name}':\n"
                f"  {self.ct_hdf_path} (CT): {len(self.ct_keys)} samples\n"
                f"  {self.proj_hdf_path} (Proj): {len(self.proj_keys)} samples\n"
                f"  {self.dose_hdf_path} (Dose): {len(self.dose_keys)} samples\n"
                "Ensure each file contains the same number of corresponding samples."
            )

        self.length = len(self.ct_keys)
        if self.length == 0:
            print(f"Warning: no data found in group '{group_name}'.")

    def _open_files(self):
        """Open HDF5 file handles for this worker if not already open."""
        if self.ct_file is None:
            self.ct_file   = h5py.File(self.ct_hdf_path,   'r')
            self.proj_file = h5py.File(self.proj_hdf_path, 'r')
            self.dose_file = h5py.File(self.dose_hdf_path, 'r')

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        self._open_files()

        if not (0 <= idx < self.length):
            raise IndexError(f"Index {idx} out of range (0 to {self.length - 1})")

        ct_key   = self.ct_keys[idx]
        proj_key = self.proj_keys[idx]
        dose_key = self.dose_keys[idx]

        try:
            ct_data   = self.ct_file[self.group_name][ct_key][()]
            proj_data = self.proj_file[self.group_name][proj_key][()]
            dose_data = self.dose_file[self.group_name][dose_key][()]
        except KeyError as e:
            print(f"KeyError at index {idx}:")
            print(f"  CT  : '{ct_key}'   from {self.ct_hdf_path}")
            print(f"  Proj: '{proj_key}' from {self.proj_hdf_path}")
            print(f"  Dose: '{dose_key}' from {self.dose_hdf_path}")
            print(f"  Error: {e}")
            for hdf_path, key in [
                (self.ct_hdf_path,   ct_key),
                (self.proj_hdf_path, proj_key),
                (self.dose_hdf_path, dose_key),
            ]:
                with h5py.File(hdf_path, 'r') as tmp:
                    if key not in tmp[self.group_name]:
                        print(f"  Confirmed missing: '{key}' in '{hdf_path}' group '{self.group_name}'")
                        break
            raise

        proj_tensor = torch.from_numpy(proj_data.astype(np.float32))
        if self.stats is not None:
            ct_tensor, dose_tensor = _apply_stats_to_arrays(ct_data, dose_data, self.stats)
        else:
            ct_tensor = torch.from_numpy(ct_data.astype(np.float32))
            dose_tensor = torch.from_numpy(dose_data.astype(np.float32))

        if self.augment_config is not None:
            ct_tensor, proj_tensor, dose_tensor = apply_random_bev_flip(
                ct_tensor, proj_tensor, dose_tensor, self.augment_config,
            )

        inputs = (ct_tensor, proj_tensor)
        if self.input_transform:
            inputs = self.input_transform(inputs)
        if self.target_transform:
            dose_tensor = self.target_transform(dose_tensor)

        return inputs, dose_tensor

    def close(self):
        """Close all open HDF5 file handles."""
        if self.ct_file:
            self.ct_file.close()
            self.ct_file = None
        if self.proj_file:
            self.proj_file.close()
            self.proj_file = None
        if self.dose_file:
            self.dose_file.close()
            self.dose_file = None

    def __del__(self):
        self.close()


class MultiModalBinDataset(Dataset):
    """Read BEV ct/proj/dose from ``bins_root/{ct,proj,dose}/*.bin`` (no HDF5 pack step)."""

    def __init__(
        self,
        catalog: BinCatalog,
        group_name: str = "train",
        *,
        stats: dict[str, Any] | None = None,
        input_transform=None,
        target_transform=None,
        augment_config: BEVAugmentConfig | None = None,
    ) -> None:
        if group_name == "train":
            self.sample_ids = list(catalog.train_ids)
        elif group_name == "validation":
            self.sample_ids = list(catalog.val_ids)
        elif group_name == "test":
            self.sample_ids = list(catalog.test_ids)
        else:
            raise ValueError(f"group_name must be 'train', 'validation', or 'test', got {group_name!r}")

        self.catalog = catalog
        self.group_name = group_name
        self.stats = stats
        self.dtype = catalog.storage_dtype
        self.input_transform = input_transform
        self.target_transform = target_transform
        self.augment_config = augment_config
        self.shape = catalog.shape

        if not self.sample_ids:
            print(f"Warning: no data found in bin group '{group_name}'.")

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        if not (0 <= idx < len(self.sample_ids)):
            raise IndexError(f"Index {idx} out of range (0 to {len(self.sample_ids) - 1})")

        sid = self.sample_ids[idx]
        ct_data = _load_bin_array(self.catalog.ct_map[sid], self.shape, self.dtype)
        proj_data = _load_bin_array(self.catalog.proj_map[sid], self.shape, self.dtype)
        dose_data = _load_bin_array(self.catalog.dose_map[sid], self.shape, self.dtype)

        proj_tensor = torch.from_numpy(proj_data.astype(np.float32))
        if self.stats is not None:
            ct_tensor, dose_tensor = _apply_stats_to_arrays(ct_data, dose_data, self.stats)
        else:
            ct_tensor = torch.from_numpy(ct_data.astype(np.float32))
            dose_tensor = torch.from_numpy(dose_data.astype(np.float32))

        if self.augment_config is not None:
            ct_tensor, proj_tensor, dose_tensor = apply_random_bev_flip(
                ct_tensor, proj_tensor, dose_tensor, self.augment_config,
            )

        inputs = (ct_tensor, proj_tensor)
        if self.input_transform:
            inputs = self.input_transform(inputs)
        if self.target_transform:
            dose_tensor = self.target_transform(dose_tensor)

        return inputs, dose_tensor


class MultiModalMergedBinDataset(Dataset):
    """Read packed merged BEV bins (one file per sample: ct, proj, dose stacked)."""

    def __init__(
        self,
        catalog: MergedBinCatalog,
        group_name: str = "train",
        *,
        stats: dict[str, Any] | None = None,
        input_transform=None,
        target_transform=None,
        augment_config: BEVAugmentConfig | None = None,
    ) -> None:
        if group_name == "train":
            self.sample_ids = list(catalog.train_ids)
        elif group_name == "validation":
            self.sample_ids = list(catalog.val_ids)
        elif group_name == "test":
            self.sample_ids = list(catalog.test_ids)
        else:
            raise ValueError(f"group_name must be 'train', 'validation', or 'test', got {group_name!r}")

        self.catalog = catalog
        self.group_name = group_name
        self.stats = stats
        self.input_transform = input_transform
        self.target_transform = target_transform
        self.augment_config = augment_config

        if not self.sample_ids:
            print(f"Warning: no data found in merged bin group '{group_name}'.")

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        if not (0 <= idx < len(self.sample_ids)):
            raise IndexError(f"Index {idx} out of range (0 to {len(self.sample_ids) - 1})")

        sid = self.sample_ids[idx]
        path = self.catalog.samples_dir / f"{sid}.bin"
        ct_data, proj_data, dose_data = _load_merged_bin_channels(
            path, self.catalog.shape_dhw, self.catalog.storage_dtype
        )

        proj_tensor = torch.from_numpy(proj_data.astype(np.float32))
        if self.stats is not None:
            ct_tensor, dose_tensor = _apply_stats_to_arrays(ct_data, dose_data, self.stats)
        else:
            ct_tensor = torch.from_numpy(ct_data.astype(np.float32))
            dose_tensor = torch.from_numpy(dose_data.astype(np.float32))

        if self.augment_config is not None:
            ct_tensor, proj_tensor, dose_tensor = apply_random_bev_flip(
                ct_tensor, proj_tensor, dose_tensor, self.augment_config,
            )

        inputs = (ct_tensor, proj_tensor)
        if self.input_transform:
            inputs = self.input_transform(inputs)
        if self.target_transform:
            dose_tensor = self.target_transform(dose_tensor)

        return inputs, dose_tensor


class OtfGpuDoseDataset(Dataset):
    """Dose-only dataset for ``otf_gpu``; CT/proj are generated later on the training GPU."""

    def __init__(
        self,
        catalog: OtfGpuCatalog,
        group_name: str = "train",
        *,
        stream_full_dose: bool = False,
    ) -> None:
        if group_name == "train":
            self.sample_ids = list(catalog.train_ids)
        elif group_name == "validation":
            self.sample_ids = list(catalog.val_ids)
        elif group_name == "test":
            self.sample_ids = list(catalog.test_ids)
        else:
            raise ValueError(f"group_name must be 'train', 'validation', or 'test', got {group_name!r}")
        self.catalog = catalog
        self.group_name = group_name
        self.stream_full_dose = bool(stream_full_dose)

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if not (0 <= idx < len(self.sample_ids)):
            raise IndexError(f"Index {idx} out of range (0 to {len(self.sample_ids) - 1})")

        sid = self.sample_ids[idx]
        rec = self.catalog.records[sid]
        out = {
            "sample_id": sid,
            "patient_id": rec.patient_id,
            "beam_id": rec.beam_id,
            "cp_id": rec.cp_id,
            "dose_path": str(rec.dose_path),
            "ct_path": str(rec.ct_path),
            "mac_path": str(rec.mac_path),
            "segment_path": str(rec.segment_path),
        }
        if self.catalog.modality == "proton":
            out.update({
                "ray_id": rec.ray_id,
                "beamlet_id": rec.beamlet_id,
                "energy_mev": rec.energy_mev,
                "sigma_energy_mev": rec.sigma_energy_mev,
                "sigma_spot_mm": rec.sigma_spot_mm,
            })
        if self.catalog.otf_gpu_full and self.stream_full_dose:
            sitk = __import__("SimpleITK")
            image = sitk.ReadImage(str(rec.dose_path))
            dose_data = np.ascontiguousarray(
                np.transpose(sitk.GetArrayFromImage(image), (2, 1, 0)),
                dtype=np.float32,
            )
            out["dose"] = torch.from_numpy(dose_data)
        elif not self.catalog.otf_gpu_full:
            dose_data = _load_bin_array(rec.dose_path, self.catalog.shape, self.catalog.dose_dtype)
            out["dose"] = torch.from_numpy(dose_data)
        return out


def collate_otf_gpu_batch(samples: list[dict[str, Any]]) -> dict[str, list[Any]]:
    """Keep variable-shape full-dose tensors as a pinnable per-sample list."""
    if not samples:
        return {}
    return {key: [sample[key] for sample in samples] for key in samples[0]}


def _load_inference_pipeline_module():
    module_name = "dl_segment_inference_pipeline"
    cached = sys.modules.get(module_name)
    if cached is not None:
        return cached

    inference_dir = Path(__file__).resolve().parent.parent / "inference"
    pipeline_py = inference_dir / "pipeline.py"
    inference_dir_str = str(inference_dir)
    if inference_dir_str not in sys.path:
        sys.path.insert(0, inference_dir_str)
    spec = importlib.util.spec_from_file_location(module_name, pipeline_py)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load inference pipeline from {pipeline_py}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_fast_preprocess_module():
    module_name = "dl_segment_fast_preprocess_triton"
    cached = sys.modules.get(module_name)
    if cached is not None:
        return cached

    inference_dir = Path(__file__).resolve().parent.parent / "inference"
    module_path = inference_dir / "fast_preprocess_triton.py"
    inference_dir_str = str(inference_dir)
    if inference_dir_str not in sys.path:
        sys.path.insert(0, inference_dir_str)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load affine preprocessing from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module



BEV_DEPTH_CACHE_VERSION = 2


@dataclass(frozen=True)
class BevDepthRangeParams:
    hu_thresh: float = -500.0
    margin_slices: int = 4
    stride: int = 8
    bev_grid: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, float | int]:
        out: dict[str, Any] = {
            "hu_thresh": float(self.hu_thresh),
            "margin_slices": int(self.margin_slices),
            "stride": int(self.stride),
        }
        if self.bev_grid is not None:
            out["bev_grid"] = self.bev_grid
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BevDepthRangeParams":
        return cls(
            hu_thresh=float(data.get("hu_thresh", -500.0)),
            margin_slices=int(data.get("margin_slices", 4)),
            stride=int(data.get("stride", 8)),
            bev_grid=data.get("bev_grid"),
        )


class BevDepthRangeCache:
    """Per-sample BEV depth index ranges ``[i0, i1]`` for option-2 trimming."""

    def __init__(
        self,
        path: str | Path | None,
        params: BevDepthRangeParams,
        *,
        auto_save_every: int = 0,
    ) -> None:
        self.path = Path(path) if path is not None else None
        self.params = params
        self.samples: dict[str, list[int]] = {}
        self._dirty = False
        self.auto_save_every = int(auto_save_every)
        self._sets_since_save = 0
        if self.path is not None and self.path.is_file():
            self._load()

    def get(self, sample_id: str) -> tuple[int, int] | None:
        row = self.samples.get(sample_id)
        if row is None:
            return None
        return int(row[0]), int(row[1])

    def set(self, sample_id: str, i0: int, i1: int) -> None:
        self.samples[sample_id] = [int(i0), int(i1)]
        self._dirty = True
        self._sets_since_save += 1
        if (
            self.auto_save_every > 0
            and self.path is not None
            and self._sets_since_save >= self.auto_save_every
        ):
            self.save()
            self._sets_since_save = 0

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": BEV_DEPTH_CACHE_VERSION,
            "params": self.params.as_dict(),
            "samples": self.samples,
        }
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self._dirty = False

    def _load(self) -> None:
        if self.path is None:
            return
        data = json.loads(self.path.read_text(encoding="utf-8"))
        loaded = BevDepthRangeParams.from_dict(data.get("params", {}))
        if loaded.as_dict() != self.params.as_dict():
            print(
                f"Warning: bev depth cache params mismatch at {self.path}; "
                f"file={loaded.as_dict()} cli={self.params.as_dict()} — "
                "missing sample_ids will be recomputed."
            )
        self.samples = {
            str(k): [int(v[0]), int(v[1])] for k, v in data.get("samples", {}).items()
        }


def resolve_bev_depth_range_cache_path(
    cache_path: str | Path | None,
    *,
    data_dir: str | Path | None = None,
    out_dir: str | Path | None = None,
) -> Path:
    if cache_path:
        return Path(cache_path)
    if out_dir:
        return Path(out_dir) / "bev_depth_range.json"
    if data_dir:
        return Path(data_dir) / "bev_depth_range.json"
    raise ValueError("bev depth cache path requires --bev-depth-range-cache, --out-dir, or --data-dir")


def _build_mac_cache_for_catalog(
    catalog: OtfGpuCatalog,
    pipeline: Any,
    bev_grid: Any | None = None,
) -> dict[str, dict[str, list[float]]]:
    if bev_grid is None:
        bev_grid = pipeline.BevGridConfig.default()
    return _build_mac_cache_entries(
        sorted(catalog.records),
        catalog.records,
        pipeline,
        bev_grid,
    )


def build_bev_depth_range_cache(
    catalog: OtfGpuCatalog,
    cache_path: str | Path,
    *,
    params: BevDepthRangeParams | None = None,
    device_index: int = 0,
    bev_grid: Any | None = None,
) -> BevDepthRangeCache:
    """Precompute ``(i0, i1)`` for every catalog sample (CT once per patient)."""
    params = params or BevDepthRangeParams()
    cache = BevDepthRangeCache(cache_path, params)
    pipeline = _load_inference_pipeline_module()
    if bev_grid is None:
        bev_grid = pipeline.BevGridConfig.default()
    cp = pipeline.cp
    cp.cuda.Device(device_index).use()
    mac_cache = _build_mac_cache_for_catalog(catalog, pipeline, bev_grid=bev_grid)

    patient_ct_gpu: dict[str, Any] = {}
    sitk = __import__("SimpleITK")
    for patient_id in catalog.patient_ids:
        arr, spacing, origin = _read_otf_ct_volume(catalog, patient_id, sitk)
        patient_ct_gpu[patient_id] = (cp.asarray(arr, cp.float32), spacing, origin)

    sample_ids = sorted(catalog.records)
    for sid in tqdm(sample_ids, desc="bev_depth_range"):
        rec = catalog.records[sid]
        ct_vol, spacing, origin = patient_ct_gpu[rec.patient_id]
        i0, i1 = pipeline.compute_bev_depth_range_from_ct_mac(
            ct_vol,
            spacing,
            origin,
            mac_cache,
            sid,
            hu_thresh=params.hu_thresh,
            margin_slices=params.margin_slices,
            stride=params.stride,
            nx=bev_grid.nx,
            bev_dx=bev_grid.spacing_dhw[0],
        )
        cache.set(sid, i0, i1)
    cache.save()
    return cache


def bev_depth_range_params_from_args(args: Any) -> BevDepthRangeParams:
    bev_grid = load_bev_grid_config(getattr(args, "bev_grid_config", None))
    return BevDepthRangeParams(
        hu_thresh=float(getattr(args, "bev_hu_thresh", -500.0)),
        margin_slices=int(getattr(args, "bev_depth_margin_slices", 4)),
        stride=int(getattr(args, "bev_depth_range_stride", 8)),
        bev_grid=bev_grid.to_dict(),
    )


def create_bev_depth_range_cache_from_args(
    args: Any,
    catalog: OtfGpuCatalog,
    *,
    out_dir: str | Path | None = None,
    bev_grid: Any | None = None,
) -> BevDepthRangeCache | None:
    if not getattr(args, "bev_depth_trim", False):
        return None
    path = resolve_bev_depth_range_cache_path(
        getattr(args, "bev_depth_range_cache", None),
        data_dir=getattr(args, "data_dir", None),
        out_dir=out_dir or getattr(args, "out_dir", None),
    )
    return BevDepthRangeCache(
        path,
        BevDepthRangeParams(
            hu_thresh=float(getattr(args, "bev_hu_thresh", -500.0)),
            margin_slices=int(getattr(args, "bev_depth_margin_slices", 4)),
            stride=int(getattr(args, "bev_depth_range_stride", 8)),
            bev_grid=(bev_grid.to_dict() if bev_grid is not None else None),
        ),
        auto_save_every=200,
    )


def run_bev_depth_range_build_if_requested(
    args: Any,
    catalog: OtfGpuCatalog,
    *,
    device_index: int = 0,
    out_dir: str | Path | None = None,
) -> None:
    if not getattr(args, "bev_depth_range_build", False):
        return
    if args.data_format != "otf_gpu":
        raise ValueError("--bev-depth-range-build requires --data-format otf_gpu")
    path = resolve_bev_depth_range_cache_path(
        getattr(args, "bev_depth_range_cache", None),
        data_dir=getattr(args, "data_dir", None),
        out_dir=out_dir or getattr(args, "out_dir", None),
    )
    params = bev_depth_range_params_from_args(args)
    bev_grid = load_bev_grid_config(getattr(args, "bev_grid_config", None))
    print(f"Building BEV depth range cache: {path}")
    build_bev_depth_range_cache(
        catalog, path, params=params, device_index=device_index, bev_grid=bev_grid
    )
    print(f"Wrote {len(catalog.records)} sample ranges to {path}")
    raise SystemExit(0)


def create_otf_gpu_batch_materializer(
    catalog: OtfGpuCatalog,
    device: torch.device | str,
    args: Any,
    *,
    stats: dict[str, Any] | None = None,
    out_dir: str | Path | None = None,
    bev_grid: Any | None = None,
) -> OtfGpuBatchMaterializer:
    if bev_grid is None:
        bev_grid = load_bev_grid_config(getattr(args, "bev_grid_config", None))
    ct_space_loss = bool(getattr(args, "ct_space_loss", False))
    ct_space_bev_pixel_shuffle = bool(getattr(args, "ct_space_bev_pixel_shuffle", False))
    ct_space_direct_spline_coefficients = bool(
        getattr(args, "ct_space_direct_spline_coefficients", False)
    )
    ct_space_direct_packed_coefficients = bool(
        getattr(args, "ct_space_direct_packed_coefficients", False)
    )
    ct_bev_idd_metrics = bool(getattr(args, "ct_bev_idd_metrics", False))
    ct_space_bev_upscale_factor = int(
        getattr(args, "ct_space_bev_upscale_factor", 2)
    )
    if ct_space_bev_upscale_factor < 2:
        raise ValueError("--ct-space-bev-upscale-factor must be at least 2")
    if ct_space_bev_pixel_shuffle and not ct_space_loss:
        raise ValueError("--ct-space-bev-pixel-shuffle requires --ct-space-loss")
    if ct_space_loss:
        if not catalog.otf_gpu_full:
            raise ValueError("--ct-space-loss requires --otf-gpu-full CT-space dose files")
        if args.otf_bev_mode != "cubic":
            raise ValueError("--ct-space-loss requires --otf-bev-mode cubic")
        if not (bev_grid.align_bev_z_to_ct_slices and bev_grid.bicubic_z_align):
            raise ValueError(
                "--ct-space-loss requires align_bev_z_to_ct_slices=true and "
                "bicubic_z_align=true in the BEV grid config"
            )
        if bool(getattr(args, "bev_depth_trim", False)):
            raise ValueError("--ct-space-loss is not compatible with --bev-depth-trim")
    depth_cache = create_bev_depth_range_cache_from_args(
        args,
        catalog,
        out_dir=out_dir,
        bev_grid=bev_grid,
    )
    # A proton catalog needs the beamlet materializer and its extra inputs:
    # spot fluence width, RSP calibration and the energy-token flag.
    proton = catalog.modality == "proton"
    materializer_type = (
        ProtonOtfGpuBatchMaterializer if proton else OtfGpuBatchMaterializer
    )
    proton_kwargs = {
        "proton_input_upscale_factor": int(
            getattr(args, "proton_input_upscale_factor", 2)
        ),
        "proton_density_calibration": str(
            getattr(args, "proton_density_calibration", "g4dcm_rsp")
        ),
        "proton_body_threshold": float(
            getattr(args, "proton_body_threshold", 0.1)
        ),
        "proton_energy_token": bool(getattr(args, "proton_energy_token", False)),
        "proton_fast_preprocess": bool(
            getattr(args, "proton_fast_preprocess", True)
        ),
        "proton_fast_cache_patients": int(
            getattr(args, "proton_fast_cache_patients", 4)
        ),
    } if proton else {}
    return materializer_type(
        catalog,
        device,
        stats=stats,
        mode=args.otf_bev_mode,
        gpu_ct_cache_patients=args.otf_gpu_ct_cache_patients,
        cache_dose=bool(getattr(args, "otf_cache_dose", False)),
        bev_depth_trim=bool(getattr(args, "bev_depth_trim", False)),
        depth_cache=depth_cache,
        bev_grid=bev_grid,
        ct_space_loss=ct_space_loss,
        ct_space_bev_pixel_shuffle=ct_space_bev_pixel_shuffle,
        ct_space_direct_spline_coefficients=ct_space_direct_spline_coefficients,
        ct_space_direct_packed_coefficients=ct_space_direct_packed_coefficients,
        ct_bev_idd_metrics=ct_bev_idd_metrics,
        ct_space_bev_upscale_factor=ct_space_bev_upscale_factor,
        ct_space_iir_precision=getattr(args, "ct_space_iir_precision", "fp32"),
        ct_space_valid_voxel_mse=bool(
            getattr(args, "ct_space_valid_voxel_mse", False)
        ),
        ct_space_negative_dose_weight=float(
            getattr(args, "ct_space_negative_dose_weight", 0.0)
        ),
        beam_gamma_hinge_weight=float(
            getattr(args, "beam_gamma_hinge_weight", 0.0)
        ),
        beam_gamma_hinge_dose_percent=float(
            getattr(args, "beam_gamma_hinge_dose_percent", 0.01)
        ),
        beam_gamma_hinge_margin=float(
            getattr(args, "beam_gamma_hinge_margin", 0.8)
        ),
        beam_gamma_hinge_temperature=float(
            getattr(args, "beam_gamma_hinge_temperature", 0.1)
        ),
        beam_gamma_hinge_denominator_floor=float(
            getattr(args, "beam_gamma_hinge_denominator_floor", 0.02)
        ),
        beam_gamma_hinge_dose_weight_saturation=float(
            getattr(args, "beam_gamma_hinge_dose_weight_saturation", 0.10)
        ),
        density_grad_weight=float(getattr(args, "density_grad_weight", 0.0)),
        density_grad_scale=float(getattr(args, "density_grad_scale", 0.15)),
        density_grad_cache_patients=int(
            getattr(args, "density_grad_cache_patients", 16)
        ),
        input_upscale_factor=int(
            getattr(args, "otf_input_upscale_factor", 1)
        ),
        batched_affine_preprocess=bool(
            getattr(args, "otf_batched_affine_preprocess", False)
        ),
        batched_affine_cache_patients=int(
            getattr(args, "otf_batched_affine_cache_patients", 4)
        ),
        collapsed_ct_dtype=str(
            getattr(args, "otf_collapsed_ct_dtype", "float32")
        ),
        **proton_kwargs,
    )


class OtfGpuBatchMaterializer:
    """Generate CT-BEV/projection tensors from dose-only batches in the main process."""

    def __init__(
        self,
        catalog: OtfGpuCatalog,
        device: torch.device | str,
        *,
        stats: dict[str, Any] | None = None,
        mode: str = "cubic",
        gpu_ct_cache_patients: int = 0,
        preload_cpu_ct: bool = True,
        cache_dose: bool = False,
        bev_depth_trim: bool = False,
        depth_cache: BevDepthRangeCache | None = None,
        bev_grid: Any | None = None,
        ct_space_loss: bool = False,
        ct_space_bev_pixel_shuffle: bool = False,
        ct_space_direct_spline_coefficients: bool = False,
        ct_space_direct_packed_coefficients: bool = False,
        ct_space_bev_upscale_factor: int = 2,
        ct_space_iir_precision: str = "fp32",
        ct_bev_idd_metrics: bool = False,
        ct_space_valid_voxel_mse: bool = False,
        ct_space_negative_dose_weight: float = 0.0,
        beam_gamma_hinge_weight: float = 0.0,
        beam_gamma_hinge_dose_percent: float = 0.01,
        beam_gamma_hinge_margin: float = 0.8,
        beam_gamma_hinge_temperature: float = 0.1,
        beam_gamma_hinge_denominator_floor: float = 0.02,
        beam_gamma_hinge_dose_weight_saturation: float = 0.10,
        density_grad_weight: float = 0.0,
        density_grad_scale: float = 0.15,
        density_grad_cache_patients: int = 16,
        input_upscale_factor: int = 1,
        batched_affine_preprocess: bool = False,
        batched_affine_cache_patients: int = 4,
        collapsed_ct_dtype: str = "float32",
    ) -> None:
        self.catalog = catalog
        self.device = torch.device(device)
        pipeline_mod = _load_inference_pipeline_module()
        self.bev_grid = bev_grid or pipeline_mod.BevGridConfig.default()
        self.input_upscale_factor = int(input_upscale_factor)
        if self.input_upscale_factor not in (1, 2):
            raise ValueError("input_upscale_factor must be 1 or 2")
        self.batched_affine_preprocess = bool(batched_affine_preprocess)
        self.batched_affine_cache_patients = int(
            batched_affine_cache_patients
        )
        if self.batched_affine_cache_patients < 1:
            raise ValueError("batched_affine_cache_patients must be at least 1")
        _collapsed_dtypes = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        if collapsed_ct_dtype not in _collapsed_dtypes:
            raise ValueError(
                "collapsed_ct_dtype must be one of "
                f"{sorted(_collapsed_dtypes)}, got {collapsed_ct_dtype!r}"
            )
        self.collapsed_ct_dtype = _collapsed_dtypes[collapsed_ct_dtype]
        self.ct_space_loss = bool(ct_space_loss)
        self.ct_space_bev_pixel_shuffle = bool(ct_space_bev_pixel_shuffle)
        self.ct_space_direct_spline_coefficients = bool(
            ct_space_direct_spline_coefficients
        )
        self.ct_space_direct_packed_coefficients = bool(
            ct_space_direct_packed_coefficients
        )
        self.ct_space_bev_upscale_factor = int(ct_space_bev_upscale_factor)
        if self.ct_space_bev_upscale_factor < 2:
            raise ValueError("ct_space_bev_upscale_factor must be at least 2")
        if self.ct_space_bev_pixel_shuffle and not self.ct_space_loss:
            raise ValueError("ct_space_bev_pixel_shuffle requires ct_space_loss")
        if self.ct_space_direct_spline_coefficients and not self.ct_space_loss:
            raise ValueError("ct_space_direct_spline_coefficients requires ct_space_loss")
        if self.ct_space_direct_packed_coefficients and not (
            self.ct_space_bev_pixel_shuffle
            and self.ct_space_direct_spline_coefficients
        ):
            raise ValueError(
                "ct_space_direct_packed_coefficients requires pixel shuffle geometry "
                "and direct spline coefficients"
            )
        if (
            self.ct_space_direct_packed_coefficients
            and self.bev_grid.bicubic_z_align_backend != "triton"
        ):
            raise ValueError(
                "ct_space_direct_packed_coefficients requires the Triton z-align backend"
            )
        if ct_space_iir_precision not in ("fp64", "fp32"):
            raise ValueError("ct_space_iir_precision must be 'fp64' or 'fp32'")
        self.ct_space_iir_precision = ct_space_iir_precision
        self.ct_bev_idd_metrics = bool(ct_bev_idd_metrics)
        self.ct_space_valid_voxel_mse = bool(ct_space_valid_voxel_mse)
        self.ct_space_negative_dose_weight = float(
            ct_space_negative_dose_weight
        )
        if self.ct_space_negative_dose_weight < 0:
            raise ValueError("ct_space_negative_dose_weight must be non-negative")
        if (
            self.ct_space_valid_voxel_mse
            or self.ct_space_negative_dose_weight > 0
        ) and not self.ct_space_loss:
            raise ValueError(
                "valid-voxel MSE and negative-dose penalty require ct_space_loss"
            )
        self.beam_gamma_hinge_weight = float(beam_gamma_hinge_weight)
        self.beam_gamma_hinge_dose_percent = float(beam_gamma_hinge_dose_percent)
        self.beam_gamma_hinge_margin = float(beam_gamma_hinge_margin)
        self.beam_gamma_hinge_temperature = float(beam_gamma_hinge_temperature)
        self.beam_gamma_hinge_denominator_floor = float(
            beam_gamma_hinge_denominator_floor
        )
        self.beam_gamma_hinge_dose_weight_saturation = float(
            beam_gamma_hinge_dose_weight_saturation
        )
        self.density_grad_weight = float(density_grad_weight)
        self.density_grad_scale = float(density_grad_scale)
        if self.density_grad_weight < 0:
            raise ValueError("density_grad_weight must be non-negative")
        if self.density_grad_scale <= 0:
            raise ValueError("density_grad_scale must be positive")
        if self.density_grad_weight > 0 and not self.ct_space_loss:
            raise ValueError("density-gradient weighting requires ct_space_loss")
        # Sized independently of the CT coefficient cache: an interface volume is
        # one float16 plane per voxel (~33 MB for a 274×260×228 CT), so holding
        # every training patient costs ~1 GB, whereas recomputing a full-volume
        # gradient on every cache miss is paid once per batch under shuffling.
        self.density_grad_cache_patients = max(1, int(density_grad_cache_patients))
        self.density_grad_cache: OrderedDict[str, Any] = OrderedDict()
        if self.beam_gamma_hinge_weight < 0:
            raise ValueError("beam_gamma_hinge_weight must be non-negative")
        if self.beam_gamma_hinge_weight > 0 and not self.ct_space_loss:
            raise ValueError("beam gamma hinge requires ct_space_loss")
        if self.ct_bev_idd_metrics and not self.ct_space_loss:
            raise ValueError("ct_bev_idd_metrics requires ct_space_loss")
        if self.input_upscale_factor > 1:
            if not self.catalog.otf_gpu_full or not self.ct_space_loss:
                raise ValueError(
                    "fine phase-packed OTF inputs require --otf-gpu-full "
                    "and --ct-space-loss"
                )
            if bev_depth_trim:
                raise ValueError(
                    "fine phase-packed OTF inputs are incompatible with "
                    "--bev-depth-trim"
                )
            if self.ct_bev_idd_metrics:
                raise ValueError(
                    "fine phase-packed OTF inputs do not support CT-to-BEV "
                    "round-trip IDD metrics"
                )
        if self.batched_affine_preprocess:
            if self.input_upscale_factor != 2:
                raise ValueError(
                    "batched affine preprocessing requires input_upscale_factor=2"
                )
            if mode != "cubic":
                raise ValueError(
                    "batched affine preprocessing requires cubic CT interpolation"
                )
            if not (
                self.bev_grid.align_bev_z_to_ct_slices
                and self.bev_grid.bicubic_z_align
                and self.bev_grid.bicubic_z_align_backend == "triton"
            ):
                raise ValueError(
                    "batched affine preprocessing requires the z-aligned "
                    "Triton bicubic backend"
                )
            if self.bev_grid.aperture_sample_on_unaligned_grid:
                raise ValueError(
                    "batched affine preprocessing requires aligned aperture sampling"
                )
            if not _catalog_ct_is_anatomy(self.catalog) and stats is None:
                # Non-HU inputs (p01-p99 normalised MR) are already in [0, 1] by
                # the time they reach the collapsed cache, so the fast path only
                # needs the out-of-volume fill to match. Without stats there is
                # no clamp at all and the fill would stay at the CT air value.
                raise ValueError(
                    "batched affine preprocessing on non-CT anatomy requires "
                    "--normalise (stats with ct_min/ct_max)"
                )
        if self.device.type != "cuda":
            raise RuntimeError("--data-format otf_gpu requires a CUDA torch device")
        self.device_index = self.device.index if self.device.index is not None else 0
        self.stats = stats
        self.mode = mode
        self.cache_dose = bool(cache_dose)
        self.bev_depth_trim = bool(bev_depth_trim)
        self.depth_cache = depth_cache
        if self.bev_depth_trim and self.depth_cache is None:
            raise ValueError("bev_depth_trim requires a BevDepthRangeCache instance")
        if mode not in ("linear", "cubic"):
            raise ValueError(f"otf BEV mode must be 'linear' or 'cubic', got {mode!r}")
        if gpu_ct_cache_patients < 0:
            raise ValueError("--otf-gpu-ct-cache-patients must be >= 0")
        self.gpu_ct_cache_patients = int(gpu_ct_cache_patients)

        self.pipeline = pipeline_mod
        self.fast_preprocess = (
            _load_fast_preprocess_module()
            if self.batched_affine_preprocess
            else None
        )
        self.cp = self.pipeline.cp
        self.cpndi = self.pipeline.cpndi
        self.sitk = __import__("SimpleITK")
        self.zoom = __import__("scipy.ndimage", fromlist=["zoom"]).zoom

        self.cp.cuda.Device(self.device_index).use()
        torch.cuda.set_device(self.device_index)
        self.dose_transfer_stream = torch.cuda.Stream(device=self.device)
        r = self.input_upscale_factor
        grid_type = type(self.bev_grid)
        self.input_grid = (
            self.bev_grid
            if r == 1
            else grid_type(
                shape_dhw=(
                    self.bev_grid.nx * r,
                    self.bev_grid.ny * r,
                    self.bev_grid.nz,
                ),
                spacing_dhw=(
                    float(self.bev_grid.spacing_dhw[0]) / r,
                    float(self.bev_grid.spacing_dhw[1]) / r,
                    float(self.bev_grid.spacing_dhw[2]),
                ),
                sad_mm=self.bev_grid.sad_mm,
                plane_origin_offset_mm=self.bev_grid.plane_origin_offset_mm,
                segment_native_size=self.bev_grid.segment_native_size,
                align_bev_z_to_ct_slices=self.bev_grid.align_bev_z_to_ct_slices,
                align_orient_tol=self.bev_grid.align_orient_tol,
                align_spacing_tol_mm=self.bev_grid.align_spacing_tol_mm,
                aperture_sample_on_unaligned_grid=(
                    self.bev_grid.aperture_sample_on_unaligned_grid
                ),
                bicubic_z_align=self.bev_grid.bicubic_z_align,
                bicubic_z_align_backend=(
                    self.bev_grid.bicubic_z_align_backend
                ),
            )
        )
        self.g_lin = self.pipeline.build_bev_index_grid(self.input_grid)
        self.mac_cache = self._build_mac_cache()
        self.input_mac_cache = (
            self.mac_cache
            if r == 1
            else _build_mac_cache_entries(
                sorted(self.catalog.records),
                self.catalog.records,
                self.pipeline,
                self.input_grid,
            )
        )
        self.cpu_ct_cache: dict[str, tuple[np.ndarray, tuple[float, float, float], tuple[float, float, float]]] = {}
        self.gpu_ct_cache: OrderedDict[str, tuple[Any, tuple[int, int, int], tuple[float, float, float], tuple[float, float, float]]] = OrderedDict()
        self.ct_meta_cache: dict[
            str, tuple[tuple[int, int, int], tuple[float, float, float], tuple[float, float, float]]
        ] = {}
        self.cpu_dose_cache: dict[str, np.ndarray] = {}
        self.segment_cache: dict[str, np.ndarray] = {}
        self.fast_preprocess_cache: OrderedDict[
            str, tuple[Any, dict[str, int]]
        ] = OrderedDict()
        # Affine/ROI construction is invariant across epochs.  Proton energy
        # layers on the same ray also share this entry.
        self.ct_loss_geometry_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
        self.prebuilt_ct_loss_geometry: dict[str, dict[str, Any]] = {}
        self.last_timings: dict[str, float] = {}
        self._ct_loss_samples: list[dict[str, Any]] = []

        if preload_cpu_ct:
            for patient_id in catalog.patient_ids:
                self._get_ct_cpu(patient_id)
        if self.gpu_ct_cache_patients == 0:
            for patient_id in catalog.patient_ids:
                self._get_ct_gpu(patient_id)

    def _anatomy_fill_clip(self) -> tuple[float, float, float]:
        """Out-of-field fill and clip bounds for the network input volume.

        CT anatomy keeps the HU conventions. A non-CT input (``--ct-name
        mr.mha``) has already been mapped to [0, 1] per volume by
        ``_normalize_mri_volume_p01_p99`` at load time, so out-of-field must
        read as 0 rather than -1024, and the clip must not reintroduce the HU
        range. Returns ``(cval, clip_min, clip_max)``.
        """
        if _catalog_ct_is_anatomy(self.catalog):
            return -1024.0, -1024.0, 3071.0
        return 0.0, 0.0, 1.0

    def _build_mac_cache(self) -> dict[str, dict[str, list[float]]]:
        sample_ids = sorted(set(self.catalog.train_ids) | set(self.catalog.val_ids))
        return _build_mac_cache_entries(
            sample_ids,
            self.catalog.records,
            self.pipeline,
            self.bev_grid,
        )

    def _get_ct_cpu(
        self, patient_id: str
    ) -> tuple[np.ndarray, tuple[float, float, float], tuple[float, float, float]]:
        if patient_id in self.cpu_ct_cache:
            return self.cpu_ct_cache[patient_id]

        arr, spacing, origin = _read_otf_ct_volume(self.catalog, patient_id, self.sitk)
        if not _catalog_ct_is_anatomy(self.catalog):
            arr = _normalize_mri_volume_p01_p99(arr)
        self.cpu_ct_cache[patient_id] = (arr, spacing, origin)
        return self.cpu_ct_cache[patient_id]

    def _get_ct_meta(
        self, patient_id: str
    ) -> tuple[tuple[int, int, int], tuple[float, float, float], tuple[float, float, float]]:
        """CT shape/spacing/origin without materialising coefficients in VRAM."""
        meta = self.ct_meta_cache.get(patient_id)
        if meta is not None:
            return meta

        ct_arr, spacing, origin = self._get_ct_cpu(patient_id)
        meta = (tuple(int(v) for v in ct_arr.shape), spacing, origin)
        self.ct_meta_cache[patient_id] = meta
        return meta

    def _get_ct_gpu(
        self, patient_id: str
    ) -> tuple[Any, tuple[int, int, int], tuple[float, float, float], tuple[float, float, float], bool]:
        if patient_id in self.gpu_ct_cache:
            self.gpu_ct_cache.move_to_end(patient_id)
            ct_coeff, ct_shape, spacing, origin = self.gpu_ct_cache[patient_id]
            return ct_coeff, ct_shape, spacing, origin, True

        ct_arr, spacing, origin = self._get_ct_cpu(patient_id)
        ct_vol = self.cp.asarray(ct_arr, self.cp.float32)
        if self.mode == "cubic":
            ct_coeff = self.cpndi.spline_filter(
                ct_vol, order=3, mode="mirror"
            ).astype(self.cp.float32, copy=False)
        else:
            ct_coeff = ct_vol
        ct_shape = tuple(int(v) for v in ct_vol.shape)
        self.gpu_ct_cache[patient_id] = (ct_coeff, ct_shape, spacing, origin)
        self.ct_meta_cache[patient_id] = (ct_shape, spacing, origin)
        evicted = False
        if self.gpu_ct_cache_patients > 0:
            while len(self.gpu_ct_cache) > self.gpu_ct_cache_patients:
                self.gpu_ct_cache.popitem(last=False)
                evicted = True
        if evicted:
            # Dropping the reference returns the block to CuPy's pool, not to
            # the driver; torch cannot reuse it until the pool releases it.
            self.cp.get_default_memory_pool().free_all_blocks()
        return ct_coeff, ct_shape, spacing, origin, False

    def _get_interface_weight(self, patient_id: str) -> torch.Tensor:
        """Per-voxel interface indicator in [0, 1] over the whole patient CT.

        Raw HU is mapped through a bilinear stoichiometric approximation to
        relative electron density before differencing. Bone spans three quarters
        of the HU range but saturates in density, so differencing HU directly
        would let bone edges dominate a term aimed at the lung/air interfaces the
        oracle ablation actually implicates.
        """
        cached = self.density_grad_cache.get(patient_id)
        if cached is not None:
            self.density_grad_cache.move_to_end(patient_id)
            return cached

        ct_arr, spacing, _ = self._get_ct_cpu(patient_id)
        hu = torch.as_tensor(
            np.ascontiguousarray(ct_arr), device=self.device, dtype=torch.float32
        )
        rho = torch.where(hu < 0.0, 1.0 + hu / 1000.0, 1.0 + hu / 1950.0).clamp_min_(0.0)
        del hu
        # Accumulated per axis rather than stacked: the transient for a full
        # patient volume is the dominant cost here.
        magnitude = torch.zeros_like(rho)
        for axis in range(rho.ndim):
            component = torch.gradient(rho, spacing=float(spacing[axis]), dim=axis)[0]
            magnitude += component.square()
            del component
        del rho
        weight = (
            (magnitude.sqrt_() / self.density_grad_scale).clamp_(0.0, 1.0).to(torch.float16)
        )
        self.density_grad_cache[patient_id] = weight
        while len(self.density_grad_cache) > self.density_grad_cache_patients:
            self.density_grad_cache.popitem(last=False)
        return weight

    def _interface_weight_for_sample(self, sample: dict[str, Any]) -> torch.Tensor:
        """Interface indicator cropped to a CT-space loss sample's ROI box."""
        patient_id = sample.get("patient_id")
        if patient_id is None:
            raise RuntimeError(
                "density-gradient weighting needs patient_id on the CT loss sample"
            )
        xs, xe, ys, ye, zs, ze = sample["roi_box"]
        weight = self._get_interface_weight(patient_id)[xs:xe, ys:ye, zs:ze]
        return weight.unsqueeze(0).float()

    def _get_fast_preprocess_cache(
        self,
        patient_id: str,
    ) -> tuple[Any, dict[str, int]]:
        cached = self.fast_preprocess_cache.get(patient_id)
        if cached is not None:
            self.fast_preprocess_cache.move_to_end(patient_id)
            return cached
        if self.fast_preprocess is None:
            raise RuntimeError("batched affine preprocessing is not enabled")

        ct_coeff, ct_shape, spacing, origin, _ = self._get_ct_gpu(patient_id)
        sample_ids = sorted(
            sid
            for sid, rec in self.catalog.records.items()
            if rec.patient_id == patient_id
        )
        tasks = [self.catalog.records[sid] for sid in sample_ids]
        ct_coeff_t = torch.from_dlpack(
            self.cp.ascontiguousarray(
                ct_coeff.astype(self.cp.float32, copy=False)
            )
        )
        plan_cache = self.fast_preprocess.build_fast_plan_inference_cache(
            tasks,
            self.input_mac_cache,
            self.input_grid,
            ct_coeff_t,
            ct_shape,
            spacing,
            origin,
            self.device,
            collapsed_dtype=self.collapsed_ct_dtype,
            include_backprojection=False,
        )
        result = (plan_cache, {sid: index for index, sid in enumerate(sample_ids)})
        self.fast_preprocess_cache[patient_id] = result
        while (
            len(self.fast_preprocess_cache)
            > self.batched_affine_cache_patients
        ):
            self.fast_preprocess_cache.popitem(last=False)
        return result

    def _fast_preprocess_inputs(
        self,
        sample_ids: Sequence[str],
        patient_ids: Sequence[str],
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Generate a model-input batch, grouping compatible patient CT shapes."""
        if self.fast_preprocess is None:
            raise RuntimeError("batched affine preprocessing is not enabled")

        grouped: defaultdict[
            tuple[int, int], list[tuple[int, Any, int]]
        ] = defaultdict(list)
        for batch_index, (sid, patient_id) in enumerate(
            zip(sample_ids, patient_ids)
        ):
            cache, sample_index = self._get_fast_preprocess_cache(patient_id)
            index = sample_index[sid]
            ct_xy = (
                int(cache.collapsed_ct.shape[1]),
                int(cache.collapsed_ct.shape[2]),
            )
            grouped[ct_xy].append((batch_index, cache, index))

        ct_items: list[torch.Tensor | None] = [None] * len(sample_ids)
        aperture_items: list[torch.Tensor | None] = [None] * len(sample_ids)
        for entries in grouped.values():
            collapsed_parts = []
            mask_parts = []
            geometry_parts: list[list[torch.Tensor]] = [[], [], [], [], []]
            for _, cache, index in entries:
                selection = slice(index, index + 1)
                collapsed_index = cache.collapsed_ct_index[selection]
                collapsed_parts.append(
                    cache.collapsed_ct.index_select(0, collapsed_index)
                )
                mask_parts.append(cache.masks[selection])
                for destination, value in zip(
                    geometry_parts, cache.preprocess_geometry(selection)
                ):
                    destination.append(value)

            geometry = tuple(
                torch.cat(parts, dim=0) for parts in geometry_parts
            )
            ct_group, aperture_group = self.fast_preprocess.fast_preprocess_batch(
                torch.cat(collapsed_parts, dim=0),
                torch.cat(mask_parts, dim=0),
                *geometry,
                output_shape=self.input_grid.shape_dhw,
                ct_min=(
                    float(self.stats["ct_min"])
                    if self.stats is not None
                    else -1024.0
                ),
                ct_max=(
                    float(self.stats["ct_max"])
                    if self.stats is not None
                    else 3071.0
                ),
                output_dtype=torch.float32,
                normalize_ct=self.stats is not None,
                fill_value=(
                    -1024.0 if _catalog_ct_is_anatomy(self.catalog) else 0.0
                ),
                spacing_yz=(
                    float(self.input_grid.spacing_dhw[1]),
                    float(self.input_grid.spacing_dhw[2]),
                ),
                sad_mm=float(self.input_grid.sad_mm),
            )
            for group_index, (batch_index, _, _) in enumerate(entries):
                ct_items[batch_index] = ct_group[group_index]
                aperture_items[batch_index] = aperture_group[group_index]

        if any(value is None for value in ct_items + aperture_items):
            raise RuntimeError("failed to populate batched affine inputs")
        ct_batch = torch.stack([value for value in ct_items if value is not None])
        aperture_batch = torch.stack(
            [value for value in aperture_items if value is not None]
        )
        return (
            self._pack_depth_height_phases(
                ct_batch, self.input_upscale_factor
            ),
            self._pack_depth_height_phases(
                aperture_batch, self.input_upscale_factor
            ),
            len(grouped),
        )

    def _get_dose_cpu(self, sample_id: str) -> np.ndarray:
        if self.cache_dose:
            cached = self.cpu_dose_cache.get(sample_id)
            if cached is not None:
                return cached
        rec = self.catalog.records[sample_id]
        img = self.sitk.ReadImage(str(rec.dose_path))
        arr = self.sitk.GetArrayFromImage(img)
        arr = np.transpose(arr, (2, 1, 0)).astype(np.float32, copy=False)
        if self.cache_dose:
            self.cpu_dose_cache[sample_id] = arr
        return arr

    def _prepare_dose_coeff_gpu(
        self, sample_id: str, dose_arr: np.ndarray | torch.Tensor | None = None
    ) -> Any:
        """Upload a CT-space dose and build its interpolation coefficients."""
        if dose_arr is None:
            dose_arr = self._get_dose_cpu(sample_id)
        if isinstance(dose_arr, torch.Tensor):
            if not dose_arr.is_cuda:
                raise ValueError("streamed dose tensor must already be on CUDA")
            dose_vol = self.cp.from_dlpack(dose_arr.detach().contiguous())
        else:
            dose_vol = self.cp.asarray(dose_arr, self.cp.float32)
        return self.pipeline.prepare_patient_volume_coeff_for_bev(
            dose_vol, self.mode, cval=0.0
        )

    def _get_segment_cpu(self, sample_id: str) -> np.ndarray:
        cached = self.segment_cache.get(sample_id)
        if cached is not None:
            return cached
        rec = self.catalog.records[sample_id]
        native = self.bev_grid.segment_native_size
        raw = np.fromfile(rec.segment_path, dtype=np.int8)
        if raw.size != native * native:
            raise ValueError(
                f"{rec.segment_path}: expected {native * native} int8 values, got {raw.size}"
            )
        seg = raw.reshape(native, native).astype(np.float32)
        zoom_y = float(self.input_grid.ny) / float(native)
        zoom_z = float(self.input_grid.nz) / float(native)
        # Rotate on native square first; rotating after zoom swaps H/W on non-square arrays.
        seg = np.flip(np.rot90(seg, 3), axis=0).copy()
        seg = self.zoom(seg, (zoom_y, zoom_z), order=1).astype(np.float32, copy=False)
        expected = (self.input_grid.ny, self.input_grid.nz)
        if seg.shape != expected:
            raise ValueError(
                f"{rec.segment_path}: processed segment shape {seg.shape}, expected {expected}"
            )
        self.segment_cache[sample_id] = seg
        return seg

    @staticmethod
    def _batch_list(batch: dict[str, Any], key: str) -> list[Any]:
        value = batch[key]
        if isinstance(value, (list, tuple)):
            return list(value)
        return [value]

    @staticmethod
    def _pack_depth_height_phases(
        volume: torch.Tensor,
        upscale_factor: int,
    ) -> torch.Tensor:
        """Pack fine depth/height into phase channels, with optional batch."""
        r = int(upscale_factor)
        if r == 1:
            return volume
        unbatched = volume.ndim == 3
        if unbatched:
            volume = volume.unsqueeze(0)
        elif volume.ndim != 4:
            raise ValueError(
                "expected fine BEV volume (D,H,W) or (B,D,H,W), got "
                f"{tuple(volume.shape)}"
            )
        batch, depth_fine, height_fine, width = volume.shape
        if depth_fine % r or height_fine % r:
            raise ValueError(
                f"fine BEV shape {tuple(volume.shape)} is not divisible by {r}"
            )
        depth = depth_fine // r
        height = height_fine // r
        packed = (
            volume.reshape(batch, depth, r, height, r, width)
            .permute(0, 1, 2, 4, 3, 5)
            .reshape(batch, depth, r * r, height, width)
        )
        return packed[0] if unbatched else packed

    def _ct_pad_value(self) -> float:
        if self.stats is not None:
            ct_min = float(self.stats["ct_min"])
            ct_max = float(self.stats["ct_max"])
            hu = min(-1024.0, ct_min)
            hu = max(hu, ct_min)
            return float((hu - ct_min) / (ct_max - ct_min))
        return -1024.0

    def _depth_range(self, sample_id: str, patient_id: str) -> tuple[int, int]:
        if self.depth_cache is None:
            raise RuntimeError("depth_cache is required for depth range lookup")
        cached = self.depth_cache.get(sample_id)
        if cached is not None:
            return cached
        ct_arr, spacing, origin = self._get_ct_cpu(patient_id)
        ct_vol = self.cp.asarray(ct_arr, self.cp.float32)
        params = self.depth_cache.params
        i0, i1 = self.pipeline.compute_bev_depth_range_from_ct_mac(
            ct_vol,
            spacing,
            origin,
            self.mac_cache,
            sample_id,
            hu_thresh=params.hu_thresh,
            margin_slices=params.margin_slices,
            stride=params.stride,
            nx=self.bev_grid.nx,
            bev_dx=self.bev_grid.spacing_dhw[0],
        )
        self.depth_cache.set(sample_id, i0, i1)
        return i0, i1

    @staticmethod
    def _pad_depth_cp(vol: Any, t_eff: int, t_max: int, pad_value: float) -> Any:
        if t_eff >= t_max:
            return vol
        pad = vol.__class__.full(
            (t_max - t_eff, vol.shape[1], vol.shape[2]),
            pad_value,
            dtype=vol.dtype,
        )
        return vol.__class__.concatenate([vol, pad], axis=0)

    @staticmethod
    def _pad_depth_torch(vol: torch.Tensor, t_eff: int, t_max: int, pad_value: float) -> torch.Tensor:
        if t_eff >= t_max:
            return vol
        pad = vol.new_full((t_max - t_eff, vol.shape[1], vol.shape[2]), pad_value)
        return torch.cat([vol, pad], dim=0)

    def save_depth_cache(self) -> None:
        if self.depth_cache is not None and self.depth_cache._dirty:
            self.depth_cache.save()

    def materialize(
        self,
        batch: dict[str, Any],
        *,
        augment: BEVAugmentConfig | None = None,
        need_bev_target: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Materialize CUDA inputs and optionally the BEV dose target."""
        if self.ct_space_loss and augment is not None:
            raise ValueError("--ct-space-loss is not compatible with BEV flip augmentation")
        if self.input_upscale_factor > 1 and need_bev_target:
            raise ValueError(
                "phase-packed fine OTF inputs currently support CT-space "
                "targets only; disable BEV/round-trip target metrics"
            )
        self.cp.cuda.Device(self.device_index).use()
        torch.cuda.set_device(self.device_index)
        timings: defaultdict[str, float] = defaultdict(float)
        t_total = time.perf_counter()
        self._ct_loss_samples = []
        sample_ids = [str(x) for x in self._batch_list(batch, "sample_id")]
        patient_ids = [str(x) for x in self._batch_list(batch, "patient_id")]
        if len(sample_ids) != len(patient_ids):
            raise ValueError(
                f"Batch sample_id/patient_id length mismatch: {len(sample_ids)} != {len(patient_ids)}"
            )

        streamed_dose_gpu: list[torch.Tensor] | None = None
        streamed_dose_events: list[torch.cuda.Event] | None = None
        if self.catalog.otf_gpu_full and "dose" in batch:
            dose_items = self._batch_list(batch, "dose")
            if len(dose_items) != len(sample_ids):
                raise ValueError(
                    f"Streamed dose batch length mismatch: {len(dose_items)} != {len(sample_ids)}"
                )
            streamed_dose_gpu = []
            streamed_dose_events = []
            t = time.perf_counter()
            with torch.cuda.stream(self.dose_transfer_stream):
                for dose_cpu in dose_items:
                    if not isinstance(dose_cpu, torch.Tensor):
                        raise TypeError("streamed dose entries must be torch tensors")
                    dose = dose_cpu.to(
                        self.device, dtype=torch.float32, non_blocking=True
                    )
                    event = torch.cuda.Event()
                    event.record(self.dose_transfer_stream)
                    streamed_dose_gpu.append(dose)
                    streamed_dose_events.append(event)
                    timings["dose_pinned"] += float(dose_cpu.is_pinned())
            timings["dose_transfer_enqueue"] += time.perf_counter() - t
            timings["dose_streamed"] = float(len(streamed_dose_gpu))

        dose_gpu: torch.Tensor | None = None
        if not self.catalog.otf_gpu_full:
            t = time.perf_counter()
            dose_cpu = batch["dose"]
            if dose_cpu.ndim == 3:
                dose_cpu = dose_cpu.unsqueeze(0)
            dose_gpu = dose_cpu.to(self.device, non_blocking=True).to(torch.float32)
            if self.stats is not None:
                dose_gpu = dose_gpu / float(self.stats["dose_scale"])
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            timings["dose_transfer"] += time.perf_counter() - t

        depth_ranges: list[tuple[int, int]] | None = None
        t_max = self.bev_grid.nx
        if self.bev_depth_trim:
            depth_ranges = [
                self._depth_range(sid, pid) for sid, pid in zip(sample_ids, patient_ids)
            ]
            t_max = max(i1 - i0 + 1 for i0, i1 in depth_ranges)
            timings["bev_depth_trim"] = 1.0
            timings["bev_depth_t_max"] = float(t_max)

        ct_pad = self._ct_pad_value()
        ct_tensors: list[torch.Tensor] = []
        proj_tensors: list[torch.Tensor] = []
        label_tensors: list[torch.Tensor] = []
        is_ct_anatomy = _catalog_ct_is_anatomy(self.catalog)
        if is_ct_anatomy:
            anatomy_cval = -1024.0
            anatomy_clip_min: float | None = -1024.0
        else:
            anatomy_cval = 0.0
            anatomy_clip_min = 0.0

        fast_ct_batch: torch.Tensor | None = None
        fast_proj_batch: torch.Tensor | None = None
        if self.batched_affine_preprocess:
            t = time.perf_counter()
            fast_ct_batch, fast_proj_batch, fast_groups = (
                self._fast_preprocess_inputs(sample_ids, patient_ids)
            )
            torch.cuda.synchronize(self.device)
            timings["prep.fast_affine"] += time.perf_counter() - t
            timings["prep.fast_affine_groups"] = float(fast_groups)
            timings["prep.use_bicubic_zcollapse"] = float(len(sample_ids))

        for b_idx, (sid, patient_id) in enumerate(zip(sample_ids, patient_ids)):
            rec = self.catalog.records[sid]
            i0, i1 = (0, self.bev_grid.nx - 1)
            if depth_ranges is not None:
                i0, i1 = depth_ranges[b_idx]
            t_eff = i1 - i0 + 1
            g_lin = self.g_lin
            nx = self.input_grid.nx
            if self.bev_depth_trim:
                g_lin = self.pipeline.slice_bev_index_grid(self.g_lin, i0, i1)
                nx = t_eff

            t = time.perf_counter()
            ct_coeff: Any = None
            if fast_ct_batch is not None and fast_proj_batch is not None:
                # The batched-affine path reads the per-patient plan cache, so
                # only CT geometry is needed here; skipping _get_ct_gpu keeps
                # evicted coefficient volumes from being spline-filtered again.
                ct_shape, spacing, origin = self._get_ct_meta(patient_id)
                hit = True
            else:
                ct_coeff, ct_shape, spacing, origin, hit = self._get_ct_gpu(
                    patient_id
                )
                self.cp.cuda.Stream.null.synchronize()
            timings["ct_cache_hit" if hit else "ct_cache_miss"] += 1.0
            timings["ct_cache_lookup"] += time.perf_counter() - t

            prep_times: dict[str, float] = {}
            coords = None
            ct_t: torch.Tensor | None = None
            proj_t: torch.Tensor | None = None
            if fast_ct_batch is not None and fast_proj_batch is not None:
                ct_t = fast_ct_batch[b_idx]
                proj_t = fast_proj_batch[b_idx]
            else:
                t = time.perf_counter()
                seg_cpu = self._get_segment_cpu(sid)
                timings["segment_cpu_cache"] += time.perf_counter() - t

                seg_cp = self.cp.asarray(seg_cpu, self.cp.float32)
                if self.catalog.otf_gpu_full:
                    need_input_coords = bool(
                        need_bev_target or self.ct_bev_idd_metrics
                    )
                    prepared = self.pipeline.prepare_bev_input_volumes(
                        patient_id,
                        str(rec.mac_path),
                        ct_coeff,
                        ct_shape,
                        spacing,
                        origin,
                        self.mode,
                        self.input_mac_cache,
                        nx,
                        self.input_grid.ny,
                        self.input_grid.nz,
                        g_lin,
                        seg_resized=seg_cp,
                        return_coords=need_input_coords,
                        grid=self.input_grid,
                        anatomy_cval=anatomy_cval,
                        anatomy_clip_min=anatomy_clip_min,
                    )
                    if need_input_coords:
                        bev_ct, seg_proj, prep_times, coords = prepared
                    else:
                        bev_ct, seg_proj, prep_times = prepared
                else:
                    bev_ct, seg_proj, prep_times = (
                        self.pipeline.prepare_bev_input_volumes(
                            patient_id,
                            str(rec.mac_path),
                            ct_coeff,
                            ct_shape,
                            spacing,
                            origin,
                            self.mode,
                            self.input_mac_cache,
                            nx,
                            self.input_grid.ny,
                            self.input_grid.nz,
                            g_lin,
                            seg_resized=seg_cp,
                            grid=self.input_grid,
                            anatomy_cval=anatomy_cval,
                            anatomy_clip_min=anatomy_clip_min,
                        )
                    )

                if self.stats is not None and is_ct_anatomy:
                    bev_ct = self.pipeline.apply_ct_stats_to_bev(
                        bev_ct, self.stats
                    )
                bev_ct = bev_ct.astype(self.cp.float32, copy=False)
                seg_proj = seg_proj.astype(self.cp.float32, copy=False)

            if self.catalog.otf_gpu_full:
                t = time.perf_counter()
                if streamed_dose_gpu is not None and streamed_dose_events is not None:
                    wait_start = time.perf_counter()
                    streamed_dose_events[b_idx].synchronize()
                    wait_time = time.perf_counter() - wait_start
                    timings["dose_transfer_wait"] += wait_time
                    timings["dose_transfer"] += wait_time
                    dose_arr: np.ndarray | torch.Tensor = streamed_dose_gpu[b_idx]
                else:
                    dose_arr = self._get_dose_cpu(sid)
                if need_bev_target:
                    if coords is None:
                        raise RuntimeError(
                            "BEV target materialization requires input coordinates"
                        )
                    dose_coeff = self._prepare_dose_coeff_gpu(sid, dose_arr)
                    self.cp.cuda.Stream.null.synchronize()
                    timings["dose_cache_lookup"] += time.perf_counter() - t
                    t = time.perf_counter()
                    bev_dose = self.pipeline.resample_volume_to_bev(
                        dose_coeff,
                        coords,
                        self.mode,
                        nx,
                        self.bev_grid.ny,
                        self.bev_grid.nz,
                        cval=0.0,
                        clip_min=0.0,
                        clip_max=None,
                        # Reuses the z_align decision prepare_bev_input_volumes made
                        # for this coords, so dose gets the same fast path as CT.
                        use_bicubic_zcollapse=bool(prep_times.get("prep.use_bicubic_zcollapse", 0.0)),
                        bicubic_z_align_backend=self.bev_grid.bicubic_z_align_backend,
                    )
                    self.cp.cuda.Stream.null.synchronize()
                    timings["dose_bev_map"] += time.perf_counter() - t
                    lbl = torch.from_dlpack(bev_dose)
                    if self.stats is not None:
                        lbl = lbl / float(self.stats["dose_scale"])
                else:
                    lbl = torch.empty(0, device=self.device, dtype=torch.float32)
                    timings["dose_bev_skipped"] += 1.0
                if self.ct_space_loss:
                    self._append_ct_loss_sample(
                        sid, rec, ct_shape, spacing, origin, dose_arr
                    )
                    if self.ct_bev_idd_metrics:
                        self._ct_loss_samples[-1].update(
                            {
                                "bev_coords": coords,
                                "use_bicubic_zcollapse": bool(
                                    prep_times.get(
                                        "prep.use_bicubic_zcollapse", 0.0
                                    )
                                ),
                            }
                        )
                if self.bev_depth_trim:
                    bev_ct = self._pad_depth_cp(bev_ct, t_eff, t_max, ct_pad)
                    seg_proj = self._pad_depth_cp(seg_proj, t_eff, t_max, 0.0)
                    if need_bev_target:
                        lbl = lbl[i0 : i1 + 1]
                        lbl = self._pad_depth_torch(lbl, t_eff, t_max, 0.0)
            elif self.bev_depth_trim:
                bev_ct = self._pad_depth_cp(bev_ct, t_eff, t_max, ct_pad)
                seg_proj = self._pad_depth_cp(seg_proj, t_eff, t_max, 0.0)
                lbl = dose_gpu[b_idx, i0 : i1 + 1]
                lbl = self._pad_depth_torch(lbl, t_eff, t_max, 0.0)
            else:
                lbl = dose_gpu[b_idx]

            if ct_t is None or proj_t is None:
                ct_t = torch.from_dlpack(bev_ct)
                proj_t = torch.from_dlpack(seg_proj)
                if self.input_upscale_factor > 1:
                    ct_t = self._pack_depth_height_phases(
                        ct_t, self.input_upscale_factor
                    )
                    proj_t = self._pack_depth_height_phases(
                        proj_t, self.input_upscale_factor
                    )
            if augment is not None:
                ct_t, proj_t, lbl = apply_random_bev_flip(ct_t, proj_t, lbl, augment)

            ct_tensors.append(ct_t)
            proj_tensors.append(proj_t)
            label_tensors.append(lbl)
            for key, value in prep_times.items():
                timings[key] += float(value)

        ct_batch = torch.stack(ct_tensors, dim=0)
        proj_batch = torch.stack(proj_tensors, dim=0)
        label = torch.stack(label_tensors, dim=0)
        timings["total"] = time.perf_counter() - t_total
        self.last_timings = dict(timings)
        return ct_batch, proj_batch, label

    def _append_ct_loss_sample(
        self,
        sample_id: str,
        rec: Any,
        ct_shape: tuple[int, int, int],
        spacing: tuple[float, float, float],
        origin: tuple[float, float, float],
        dose_xyz: np.ndarray | torch.Tensor,
        dose_roi_box: tuple[int, int, int, int, int, int] | None = None,
    ) -> None:
        """Keep the current batch's inverse coordinates and CT-space target on CUDA."""
        nx = int(self.bev_grid.nx)
        ny = int(self.bev_grid.ny)
        nz = int(self.bev_grid.nz)
        dx, dy, dz = (float(v) for v in self.bev_grid.spacing_dhw)
        if self.ct_space_bev_pixel_shuffle:
            # Fine dose head: r-times depth/height voxels at 1/r spacing;
            # the physical FOV remains the same as the coarse BEV.
            r = self.ct_space_bev_upscale_factor
            pred_nx, pred_ny, pred_nz = r * nx, r * ny, nz
            spacing_dhw = (dx / r, dy / r, dz)
        else:
            pred_nx, pred_ny, pred_nz = nx, ny, nz
            spacing_dhw = (dx, dy, dz)

        geometry_id: Any = sample_id
        if getattr(rec, "ray_id", None) is not None:
            geometry_id = (
                rec.patient_id,
                int(rec.beam_id),
                int(rec.ray_id),
            )
        geometry_key = (
            geometry_id,
            tuple(int(v) for v in ct_shape),
            tuple(float(v) for v in spacing),
            tuple(float(v) for v in origin),
            (pred_nx, pred_ny, pred_nz),
            tuple(float(v) for v in spacing_dhw),
            self.bev_grid.bicubic_z_align_backend,
        )
        cached_geometry = self.ct_loss_geometry_cache.get(geometry_key)
        ctx_kwargs = dict(
            mac_cache=self.mac_cache,
            NX=pred_nx,
            NY=pred_ny,
            NZ=pred_nz,
            spacing_dhw=spacing_dhw,
            plane_origin_offset_mm=self.bev_grid.resolved_plane_origin_offset_mm,
            align_bev_z_to_ct_slices=True,
            align_orient_tol=self.bev_grid.align_orient_tol,
            align_spacing_tol_mm=self.bev_grid.align_spacing_tol_mm,
        )
        use_affine = self.bev_grid.bicubic_z_align_backend == "triton"
        prebuilt_geometry = self.prebuilt_ct_loss_geometry.get(sample_id)
        if cached_geometry is not None:
            sample_geometry = cached_geometry["sample_geometry"]
            roi_shape = cached_geometry["roi_shape"]
            roi_box = cached_geometry["roi_box"]
        elif prebuilt_geometry is not None and use_affine:
            sample_geometry = {"affine": prebuilt_geometry["affine"]}
            roi_shape = prebuilt_geometry["roi_shape"]
            roi_box = prebuilt_geometry["roi_box"]
        elif use_affine:
            ctx = self.pipeline.build_back_projection_affine_ctx(
                str(rec.mac_path), ct_shape, spacing, origin, **ctx_kwargs
            )
            affine = torch.from_dlpack(
                self.cp.ascontiguousarray(ctx.affine)
            ).unsqueeze(0)
            w_row = affine[0, 2]
            z_alignment_error = torch.stack(
                (
                    w_row[0].abs(),
                    w_row[1].abs(),
                    (w_row[2].abs() - 1.0).abs(),
                    (w_row[3] - w_row[3].round()).abs(),
                )
            ).max()
            if float(z_alignment_error.item()) > 1e-3:
                raise RuntimeError(
                    f"{sample_id}: inverse BEV-to-CT affine is not z aligned"
                )
            sample_geometry = {"affine": affine}
            roi_shape = ctx.roi_shape
            roi_box = ctx.roi_box
        else:
            ctx = self.pipeline.build_back_projection_ctx(
                str(rec.mac_path),
                ct_shape,
                spacing,
                origin,
                interp=3,
                **ctx_kwargs,
            )
            q = torch.from_dlpack(self.cp.ascontiguousarray(ctx.map_idx)).reshape(
                3, *ctx.roi_shape
            )
            w_by_z = q[2, 0, 0]
            w_rounded = w_by_z.round()
            plane_error = (q[2] - w_by_z.view(1, 1, -1)).abs().max()
            integer_error = (w_by_z - w_rounded).abs().max()
            if float(torch.maximum(plane_error, integer_error).item()) > 1e-3:
                raise RuntimeError(
                    f"{sample_id}: inverse BEV-to-CT coordinates are not z aligned"
                )
            points = torch.stack((q[0], q[1]), dim=-1).reshape(1, -1, 2)
            valid = (
                (q[0] >= 0)
                & (q[0] <= pred_nx - 1)
                & (q[1] >= 0)
                & (q[1] <= pred_ny - 1)
                & (q[2] >= 0)
                & (q[2] <= pred_nz - 1)
            ).reshape(1, *ctx.roi_shape)
            points[..., 0].clamp_(0, pred_nx - 1)
            points[..., 1].clamp_(0, pred_ny - 1)
            nxr, nyr, nzr = ctx.roi_shape
            zq = torch.arange(
                nzr, device=self.device, dtype=torch.int32
            ).repeat(nxr * nyr)
            sample_geometry = {
                "points": points,
                "target_w_index": w_rounded.to(torch.int32),
                "zq": zq.unsqueeze(0),
                "valid": valid,
            }
            roi_shape = ctx.roi_shape
            roi_box = ctx.roi_box
        if cached_geometry is None:
            self.ct_loss_geometry_cache[geometry_key] = {
                "sample_geometry": sample_geometry,
                "roi_shape": roi_shape,
                "roi_box": roi_box,
            }
        xs, xe, ys, ye, zs, ze = roi_box
        if dose_roi_box is not None:
            supplied_box = tuple(int(value) for value in dose_roi_box)
            if supplied_box != tuple(int(value) for value in roi_box):
                raise ValueError(
                    f"{sample_id}: streamed dose ROI {supplied_box} does not "
                    f"match CT-loss ROI {roi_box}"
                )
            expected_shape = tuple(int(value) for value in roi_shape)
            if tuple(dose_xyz.shape) != expected_shape:
                raise ValueError(
                    f"{sample_id}: streamed ROI shape {tuple(dose_xyz.shape)} "
                    f"does not match {expected_shape}"
                )
            if isinstance(dose_xyz, torch.Tensor):
                target = dose_xyz.contiguous().unsqueeze(0)
            else:
                target = torch.as_tensor(
                    np.ascontiguousarray(dose_xyz),
                    device=self.device,
                    dtype=torch.float32,
                ).unsqueeze(0)
        elif isinstance(dose_xyz, torch.Tensor):
            target = dose_xyz[xs:xe, ys:ye, zs:ze].contiguous().unsqueeze(0)
        else:
            target = torch.as_tensor(
                np.ascontiguousarray(dose_xyz[xs:xe, ys:ye, zs:ze]),
                device=self.device,
                dtype=torch.float32,
            ).unsqueeze(0)
        if self.stats is not None:
            target = target / float(self.stats["dose_scale"])
        self._ct_loss_samples.append(
            {
                **sample_geometry,
                "sample_id": sample_id,
                "patient_id": rec.patient_id,
                "target": target,
                "roi_shape": roi_shape,
                "roi_box": roi_box,
                "ct_shape": tuple(int(v) for v in ct_shape),
                "spacing_xyz_mm": tuple(float(v) for v in spacing),
                "pred_shape_dhw": (pred_nx, pred_ny, pred_nz),
            }
        )

    def _ct_prediction_to_bev(
        self,
        prediction_roi: torch.Tensor,
        sample: dict[str, Any],
        *,
        positive_clip_before_interp: bool,
    ) -> torch.Tensor:
        """Cubic-interpolate one predicted CT ROI back onto the coarse BEV grid."""
        if "bev_coords" not in sample:
            raise RuntimeError("CT-to-BEV coordinates are missing for round-trip IDD")
        ct_volume = torch.zeros(
            sample["ct_shape"], device=self.device, dtype=torch.float32
        )
        xs, xe, ys, ye, zs, ze = (int(v) for v in sample["roi_box"])
        ct_volume[xs:xe, ys:ye, zs:ze] = prediction_roi[0].detach().float()
        if positive_clip_before_interp:
            ct_volume.clamp_min_(0.0)
        volume_cp = self.cp.from_dlpack(ct_volume.contiguous())
        coeff = self.pipeline.prepare_patient_volume_coeff_for_bev(
            volume_cp, self.mode, cval=0.0
        )
        bev_cp = self.pipeline.resample_volume_to_bev(
            coeff,
            sample["bev_coords"],
            self.mode,
            self.bev_grid.nx,
            self.bev_grid.ny,
            self.bev_grid.nz,
            cval=0.0,
            clip_min=None,
            clip_max=None,
            use_bicubic_zcollapse=bool(sample["use_bicubic_zcollapse"]),
            bicubic_z_align_backend=self.bev_grid.bicubic_z_align_backend,
        )
        self.cp.cuda.Stream.null.synchronize()
        return torch.from_dlpack(bev_cp)

    def loss(
        self,
        output: torch.Tensor | Sequence[torch.Tensor],
        bev_target: torch.Tensor,
        criterion: nn.Module,
        *,
        ct_metrics: BevMetricsAccumulator | None = None,
        ct_sample_callback: Callable[
            [dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor], None
        ]
        | None = None,
        include_aux_losses: bool = True,
    ) -> torch.Tensor:
        """Apply the configured objective and optionally collect CT validation metrics.

        ``include_aux_losses=False`` drops training-only terms (currently the
        density-gradient interface term) so that reported validation loss keeps
        the same definition across runs that enable them and runs that do not.
        """
        if not self.ct_space_loss:
            return criterion(output, bev_target)
        if not self._ct_loss_samples:
            raise RuntimeError("CT-space loss metadata is missing; call materialize first")

        model_dir = Path(__file__).resolve().parent.parent / "model"
        if str(model_dir) not in sys.path:
            sys.path.insert(0, str(model_dir))
        backend = self.bev_grid.bicubic_z_align_backend
        iir_dtype = (
            torch.float64
            if getattr(self, "ct_space_iir_precision", "fp32") == "fp64"
            else torch.float32
        )
        if backend == "triton":
            from patient_space_resample_triton import (
                bicubic_iir_sample_affine_from_coeff_triton,
                bicubic_iir_sample_affine_from_packed_coeff_triton,
                bicubic_iir_sample_from_coeff_triton,
                cubic_iir_prefilter_2d_triton,
            )
        else:
            from patient_space_resample import separable_cubic_prefilter
            from patient_space_resample_triton import bicubic_zcollapsed_sample

        outputs = tuple(output) if isinstance(output, (tuple, list)) else (output,)
        coeffs = []
        packed_direct = bool(
            getattr(self, "ct_space_direct_packed_coefficients", False)
        )
        coarse = self.bev_grid.shape_dhw
        upscale_factor = int(getattr(self, "ct_space_bev_upscale_factor", 2))
        if self.ct_space_bev_pixel_shuffle:
            expected = (
                upscale_factor * coarse[0],
                upscale_factor * coarse[1],
                coarse[2],
            )
        else:
            expected = coarse
        for tensor in outputs:
            if packed_direct:
                packed_expected = (
                    coarse[0], upscale_factor ** 2, coarse[1], coarse[2]
                )
                if (
                    tensor.ndim != 5
                    or tensor.shape[1] != packed_expected[0]
                    or tensor.shape[2] != packed_expected[1]
                    or tensor.shape[3] > packed_expected[2]
                    or tensor.shape[4] > packed_expected[3]
                ):
                    raise ValueError(
                        f"expected packed coefficients (B,{packed_expected[0]},"
                        f"{packed_expected[1]},"
                        f"{packed_expected[2]},{packed_expected[3]}), got "
                        f"{tuple(tensor.shape)}"
                    )
                pad_h = packed_expected[2] - tensor.shape[3]
                pad_w = packed_expected[3] - tensor.shape[4]
                if pad_h or pad_w:
                    tensor = torch.nn.functional.pad(
                        tensor,
                        (
                            pad_w // 2,
                            pad_w - pad_w // 2,
                            pad_h // 2,
                            pad_h - pad_h // 2,
                        ),
                    )
                coeffs.append(tensor)
                continue
            if tensor.ndim == 5 and tensor.shape[1] == 1:
                tensor = tensor[:, 0]
            if tensor.ndim != 4:
                raise ValueError(f"expected model dose output (B,D,H,W), got {tensor.shape}")
            if (
                tensor.shape[1] != expected[0]
                or tensor.shape[2] > expected[1]
                or tensor.shape[3] > expected[2]
            ):
                raise ValueError(
                    f"model dose shape {tuple(tensor.shape[1:])} is incompatible with "
                    f"expected pred grid {expected} "
                    f"(ct_space_bev_pixel_shuffle={self.ct_space_bev_pixel_shuffle})"
                )
            pad_h = expected[1] - tensor.shape[2]
            pad_w = expected[2] - tensor.shape[3]
            if pad_h or pad_w:
                tensor = torch.nn.functional.pad(
                    tensor,
                    (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2),
                )
            with torch.amp.autocast("cuda", enabled=False):
                if getattr(self, "ct_space_direct_spline_coefficients", False):
                    coeffs.append(tensor.float())
                else:
                    coeffs.append(
                        cubic_iir_prefilter_2d_triton(tensor.to(iir_dtype)).float()
                        if backend == "triton"
                        else separable_cubic_prefilter(tensor.float())
                    )

        losses = []
        for index, sample in enumerate(self._ct_loss_samples):
            predictions = []
            sample_valid = sample.get("valid")
            for coeff in coeffs:
                if packed_direct:
                    if "affine" not in sample:
                        raise RuntimeError(
                            "packed coefficient sampling requires affine z-align geometry"
                        )
                    pred, sample_valid = (
                        bicubic_iir_sample_affine_from_packed_coeff_triton(
                            coeff[index : index + 1],
                            sample["affine"],
                            sample["roi_shape"],
                        )
                    )
                elif backend == "triton" and "affine" in sample:
                    pred, sample_valid = bicubic_iir_sample_affine_from_coeff_triton(
                        coeff[index : index + 1],
                        sample["affine"],
                        sample["roi_shape"],
                    )
                elif backend == "triton":
                    pred = bicubic_iir_sample_from_coeff_triton(
                        coeff[index : index + 1],
                        sample["points"],
                        sample["target_w_index"],
                        sample["zq"],
                    )
                else:
                    pred = bicubic_zcollapsed_sample(
                        coeff[index : index + 1],
                        sample["points"],
                        sample["target_w_index"],
                        sample["zq"],
                        backend=backend,
                    )
                pred = pred.reshape(1, *sample["roi_shape"])
                predictions.append(pred.masked_fill(~sample_valid, 0.0))
            if sample_valid is None:
                raise RuntimeError("CT-space sampler did not provide a validity mask")
            target = sample["target"].masked_fill(~sample_valid, 0.0)
            if ct_sample_callback is not None:
                ct_sample_callback(
                    sample,
                    predictions[0].detach(),
                    sample["target"].detach(),
                    sample_valid.detach(),
                )
            if ct_metrics is not None:
                ct_metrics.update_ct(
                    predictions[0],
                    sample["target"],
                    sample_valid,
                )
                ct_metrics.update_ct_beam_idd(
                    sample,
                    predictions[0],
                    sample["target"],
                    sample_valid,
                )
                if bool(getattr(self, "ct_bev_idd_metrics", False)):
                    if bev_target.numel() == 0:
                        raise RuntimeError(
                            "round-trip BEV IDD requires materialize(..., "
                            "need_bev_target=True)"
                        )
                    pred_bev = self._ct_prediction_to_bev(
                        predictions[0], sample,
                        positive_clip_before_interp=False,
                    )
                    pred_bev_positive = self._ct_prediction_to_bev(
                        predictions[0], sample,
                        positive_clip_before_interp=True,
                    )
                    ct_metrics.update_ct_roundtrip_bev_idd(
                        pred_bev,
                        pred_bev_positive,
                        bev_target[index],
                    )
            use_valid_mse = bool(
                getattr(self, "ct_space_valid_voxel_mse", False)
            )
            if use_valid_mse:
                if len(predictions) != 1:
                    raise RuntimeError(
                        "valid-voxel CT MSE currently requires a single model output"
                    )
                squared_error = (predictions[0] - sample["target"]).square()
                sample_loss = squared_error[sample_valid].mean()
                masked_mae_weight = float(
                    getattr(criterion, "masked_mae_weight", 0.0)
                )
                if masked_mae_weight != 0.0:
                    valid_prediction = predictions[0].masked_fill(
                        ~sample_valid, 0.0
                    )
                    valid_target = sample["target"].masked_fill(
                        ~sample_valid, 0.0
                    )
                    sample_loss = sample_loss + (
                        masked_mae_weight
                        * masked_mae_loss(
                            valid_prediction,
                            valid_target,
                            float(getattr(criterion, "masked_mae_threshold", 0.10)),
                            reduction="mean",
                        )
                    )
            else:
                criterion_input = (
                    tuple(predictions) if len(predictions) > 1 else predictions[0]
                )
                sample_loss = criterion(criterion_input, target)
            negative_weight = float(
                getattr(self, "ct_space_negative_dose_weight", 0.0)
            )
            if negative_weight > 0:
                if len(predictions) != 1:
                    raise RuntimeError(
                        "negative sampled-dose penalty currently requires a single output"
                    )
                negative_squared = predictions[0].clamp_max(0.0).square()
                sample_loss = sample_loss + negative_weight * negative_squared[
                    sample_valid
                ].mean()
            if include_aux_losses and self.density_grad_weight > 0:
                if len(predictions) != 1:
                    raise RuntimeError(
                        "density-gradient weighting currently requires a single output"
                    )
                interface = self._interface_weight_for_sample(sample)
                squared = (predictions[0] - sample["target"]).square()
                masked_interface = interface * sample_valid
                # Squared error restricted to interface voxels, normalised by
                # their own mass so the term does not scale with how much of the
                # ROI happens to sit near an interface.
                sample_loss = sample_loss + self.density_grad_weight * (
                    (masked_interface * squared).sum()
                    / masked_interface.sum().clamp_min(1.0)
                )
            hinge_weight = float(getattr(self, "beam_gamma_hinge_weight", 0.0))
            if hinge_weight > 0:
                sample_loss = sample_loss + hinge_weight * beam_gamma_hinge_loss(
                    predictions[0],
                    sample["target"],
                    sample_valid,
                    dose_percent=float(
                        getattr(self, "beam_gamma_hinge_dose_percent", 0.01)
                    ),
                    margin=float(getattr(self, "beam_gamma_hinge_margin", 0.8)),
                    temperature=float(
                        getattr(self, "beam_gamma_hinge_temperature", 0.1)
                    ),
                    denominator_floor=float(
                        getattr(self, "beam_gamma_hinge_denominator_floor", 0.02)
                    ),
                    dose_weight_saturation=float(
                        getattr(
                            self,
                            "beam_gamma_hinge_dose_weight_saturation",
                            0.10,
                        )
                    ),
                )
            losses.append(sample_loss)
        self._ct_loss_samples = []
        return torch.stack(losses).mean()


def is_otf_gpu_dataset(ds: object) -> bool:
    return isinstance(ds, OtfGpuDoseDataset)


# -- proton on-the-fly training inputs ---------------------------------------
#
# The proton dose model trains through the same OTF-GPU pipeline as the
# photon one, with beamlets in place of control points: a Gaussian spot
# fluence, RSP/WET range conditioning and a discrete energy token per
# beamlet. Only the paths the shipped checkpoint was trained through are
# kept -- no analytic Bragg curve and no sCT overlay.

def val_beam_idd_plan_root(args: Any) -> str | None:
    """Plan root holding the gantry angles, or ``None`` when unavailable.

    Proton catalogs are discovered from ``--data-dir`` (falling back to
    ``--baseline-pb-dir``), photon ones from the baseline tree's
    ``photon/<split>/``.
    """
    if str(getattr(args, "otf_modality", "photon")) == "proton":
        return (
            getattr(args, "data_dir", None)
            or getattr(args, "baseline_pb_dir", None)
        )
    return getattr(args, "baseline_pb_dir", None)

def _nearest_proton_energy_row(
    energy_mev: float,
    energy_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    return min(
        energy_rows,
        key=lambda row: abs(float(row["energy_mev"]) - float(energy_mev)),
    )

def load_proton_energy_token_values(
    beam_parameters_path: str | Path,
) -> list[float]:
    """Return the stable sorted energy-to-token mapping from beam parameters."""
    path = Path(beam_parameters_path)
    parameters = json.loads(path.read_text(encoding="utf-8"))
    rows = list(parameters.get("proton", {}).get("energy_table", ()))
    levels = sorted({float(row["energy_mev"]) for row in rows})
    if not levels:
        raise ValueError(f"{path}: missing proton.energy_table")
    if len(levels) != len(rows):
        raise ValueError(f"{path}: proton.energy_table contains duplicate energies")
    return levels

def _proton_patient_dirs(
    root: Path,
    baseline_split: str,
) -> list[Path]:
    """Resolve either a DoseRAD proton split, a patient dir, or a flat root."""
    split_root = _doserad_split_root(
        root, baseline_split, modality="proton"
    )
    if split_root.is_dir():
        root = split_root
    if (root / f"{root.name}.json").is_file() and (root / "image").is_dir():
        return [root]
    return [
        patient_dir
        for patient_dir in sorted(root.iterdir())
        if patient_dir.is_dir()
        and (patient_dir / f"{patient_dir.name}.json").is_file()
        and (patient_dir / "image").is_dir()
        and (patient_dir / "dose").is_dir()
    ]

def _discover_proton_otf_gpu_catalog(
    data_dir: str | Path | None,
    *,
    baseline_pb_dir: str | Path | None,
    baseline_split: str,
    patient_split_json: str | Path | None,
    validation_fraction: float,
    seed: int,
    shape: tuple[int, int, int] | None,
    dose_dtype: np.dtype,
    ct_name: str,
    otf_gpu_full: bool,
    bev_grid: Any | None,
    beam_parameters_path: str | Path | None,
) -> OtfGpuCatalog:
    if not otf_gpu_full:
        raise ValueError("proton otf_gpu requires --otf-gpu-full")
    if beam_parameters_path in (None, ""):
        raise ValueError(
            "--proton-beam-parameters is required for proton otf_gpu"
        )
    parameters_path = Path(beam_parameters_path)
    if not parameters_path.is_file():
        raise FileNotFoundError(
            f"--proton-beam-parameters not found: {parameters_path}"
        )
    parameters = json.loads(parameters_path.read_text(encoding="utf-8"))
    energy_rows = list(parameters.get("proton", {}).get("energy_table", ()))
    if not energy_rows:
        raise ValueError(
            f"{parameters_path}: missing proton.energy_table"
        )
    energy_levels = tuple(
        sorted({float(row["energy_mev"]) for row in energy_rows})
    )
    if len(energy_levels) != len(energy_rows):
        raise ValueError(
            f"{parameters_path}: proton.energy_table contains duplicate energies"
        )
    _sigma_by_energy = {
        float(row["energy_mev"]): float(row["sigma_energy_mev"])
        for row in energy_rows
    }
    energy_sigmas = tuple(_sigma_by_energy[value] for value in energy_levels)

    if data_dir not in (None, ""):
        root = Path(data_dir)
    elif baseline_pb_dir not in (None, ""):
        root = Path(baseline_pb_dir)
    else:
        raise ValueError(
            "proton otf_gpu requires --data-dir or --baseline-pb-dir"
        )
    patient_dirs = _proton_patient_dirs(root, baseline_split)
    if not patient_dirs:
        raise FileNotFoundError(
            f"No proton patient directories found under {root}"
        )

    records: dict[str, OtfGpuSampleRecord] = {}
    errors: list[str] = []
    for patient_dir in patient_dirs:
        patient_id = patient_dir.name
        plan_path = patient_dir / f"{patient_id}.json"
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        ct_path = patient_dir / "image" / ct_name
        ct_geometry_path: Path | None = None
        if not ct_path.is_file():
            errors.append(f"Missing CT for {patient_id}: {ct_path}")
            continue
        for beam in plan.get("beams", ()):
            beam_id = int(beam["beam_idx"])
            for ray in beam.get("rays", ()):
                ray_id = int(ray["ray_idx"])
                source = tuple(float(v) for v in ray["ray_source"])
                target = tuple(float(v) for v in ray["ray_target"])
                for beamlet in ray.get("beamlets", ()):
                    beamlet_id = int(beamlet["beamlet_idx"])
                    energy = float(beamlet["energy"])
                    row = _nearest_proton_energy_row(energy, energy_rows)
                    if abs(float(row["energy_mev"]) - energy) > 1e-3:
                        errors.append(
                            f"{patient_id} B{beam_id} R{ray_id} L{beamlet_id}: "
                            f"energy {energy} is absent from {parameters_path}"
                        )
                        continue
                    sid = (
                        f"{patient_id}_{beam_id}_R{ray_id:02d}_L{beamlet_id}"
                    )
                    dose_path = (
                        patient_dir
                        / "dose"
                        / f"Dose_B{beam_id}_R{ray_id}_L{beamlet_id}.mha"
                    )
                    if not dose_path.is_file():
                        errors.append(f"Missing dose for {sid}: {dose_path}")
                        continue
                    # Proton geometry and Gaussian fluence are generated from
                    # metadata. These paths are stable identifiers only; no
                    # photon MAC or segment files are read.
                    virtual_mac = patient_dir / "mac" / f"{sid}.mac"
                    virtual_segment = patient_dir / "segments" / f"{sid}.bin"
                    records[sid] = OtfGpuSampleRecord(
                        sample_id=sid,
                        patient_id=patient_id,
                        beam_id=beam_id,
                        # Keep the legacy beam/cp validation interface unique
                        # within a proton beam while retaining explicit IDs.
                        cp_id=1000 * ray_id + beamlet_id,
                        dose_path=dose_path,
                        ct_path=ct_path,
                        mac_path=virtual_mac,
                        segment_path=virtual_segment,
                        ct_geometry_path=ct_geometry_path,
                        ray_id=ray_id,
                        beamlet_id=beamlet_id,
                        source_xyz_mm=source,
                        target_xyz_mm=target,
                        energy_mev=energy,
                        sigma_energy_mev=float(row["sigma_energy_mev"]),
                        sigma_spot_mm=float(row["sigma_spot_mm"]),
                    )
    if errors:
        preview = "\n  ".join(errors[:12])
        suffix = (
            f"\n  ... {len(errors) - 12} more error(s)"
            if len(errors) > 12
            else ""
        )
        raise FileNotFoundError(
            f"Invalid proton otf_gpu catalog:\n  {preview}{suffix}"
        )
    if not records:
        raise ValueError(f"No proton beamlets found under {root}")

    sample_ids = sorted(records)
    train_ids, val_ids, test_ids = _train_val_ids_from_samples(
        sample_ids,
        patient_split_json=patient_split_json,
        validation_fraction=validation_fraction,
        seed=seed,
    )
    if shape is None:
        shape = _default_otf_bev_shape(bev_grid)
    ranges = {
        key: (
            min(float(row[key]) for row in energy_rows),
            max(float(row[key]) for row in energy_rows),
        )
        for key in ("energy_mev", "sigma_energy_mev", "sigma_spot_mm")
    }
    return OtfGpuCatalog(
        data_root=root,
        baseline_pb_dir=(
            Path(baseline_pb_dir)
            if baseline_pb_dir not in (None, "")
            else root
        ),
        baseline_split=baseline_split,
        segment_mac_dir=root,
        train_ids=train_ids,
        val_ids=val_ids,
        test_ids=test_ids,
        records=records,
        shape=shape,
        dose_dtype=dose_dtype,
        ct_name=ct_name,
        otf_gpu_full=True,
        modality="proton",
        proton_energy_ranges=ranges,
        proton_energy_levels=energy_levels,
        proton_energy_sigmas=energy_sigmas,
    )

class ProtonRayBatchSampler:
    """Optionally batch all energy layers of a proton ray together."""

    def __init__(
        self,
        dataset: OtfGpuDoseDataset,
        batch_size: int,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        if dataset.catalog.modality != "proton":
            raise ValueError("ProtonRayBatchSampler requires a proton dataset")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.generator = generator
        grouped: OrderedDict[tuple[str, int, int], list[int]] = OrderedDict()
        for index, sample_id in enumerate(dataset.sample_ids):
            rec = dataset.catalog.records[sample_id]
            if rec.ray_id is None:
                raise ValueError(f"{sample_id}: missing proton ray id")
            key = (rec.patient_id, int(rec.beam_id), int(rec.ray_id))
            grouped.setdefault(key, []).append(index)
        self.batches: list[list[int]] = []
        batch: list[int] = []
        for group in grouped.values():
            if batch and len(batch) + len(group) > self.batch_size:
                self.batches.append(batch)
                batch = []
            if len(group) > self.batch_size:
                for start in range(0, len(group), self.batch_size):
                    self.batches.append(group[start : start + self.batch_size])
            else:
                batch.extend(group)
        if batch:
            self.batches.append(batch)

    def __iter__(self):
        order = torch.randperm(
            len(self.batches), generator=self.generator
        ).tolist()
        for batch_index in order:
            yield self.batches[batch_index]

    def __len__(self) -> int:
        return len(self.batches)


# -- proton batch materializer ------------------------------------------------
#
# Subclasses the photon OTF-GPU materializer: same CT cache, plan cache and
# CT-space loss machinery, with the proton inputs layered on -- a Gaussian
# spot fluence per beamlet, RSP/WET range conditioning on the fine grid, and
# a discrete energy-token index per sample.

class ProtonOtfGpuBatchMaterializer(OtfGpuBatchMaterializer):
    """OTF-full proton inputs with fine-grid phase packing and CT loss.

    The xLSTM remains on ``bev_grid``. CT, Gaussian fluence, RSP, WET and
    remaining range are evaluated on a finer depth/height grid and packed
    into phase channels before the model forward.
    """

    def __init__(
        self,
        catalog: OtfGpuCatalog,
        device: torch.device | str,
        *,
        proton_input_upscale_factor: int = 2,
        proton_density_calibration: str = "g4dcm_rsp",
        proton_body_threshold: float = 0.1,
        proton_energy_token: bool = False,
        proton_fast_preprocess: bool = True,
        proton_fast_cache_patients: int = 4,
        **kwargs: Any,
    ) -> None:
        if catalog.modality != "proton":
            raise ValueError("ProtonOtfGpuBatchMaterializer requires a proton catalog")
        self.proton_input_upscale_factor = int(proton_input_upscale_factor)
        if self.proton_input_upscale_factor not in (1, 2):
            raise ValueError("proton_input_upscale_factor must be 1 or 2")
        self.proton_density_calibration = str(proton_density_calibration)
        self.proton_body_threshold = float(proton_body_threshold)
        self.proton_energy_token = bool(proton_energy_token)
        self.proton_fast_preprocess_enabled = bool(proton_fast_preprocess)
        self.proton_fast_cache_patients = int(proton_fast_cache_patients)
        if self.proton_fast_cache_patients < 1:
            raise ValueError("proton_fast_cache_patients must be at least 1")
        if self.proton_density_calibration not in (
            "legacy",
            "g4dcm",
            "g4dcm_rsp",
        ):
            raise ValueError(
                "proton_density_calibration must be legacy, g4dcm, or g4dcm_rsp"
            )
        if not bool(kwargs.get("ct_space_loss", False)):
            raise ValueError("proton otf_gpu currently requires CT-space loss")
        if bool(kwargs.get("bev_depth_trim", False)):
            raise ValueError("proton otf_gpu does not support BEV depth trimming")
        super().__init__(catalog, device, **kwargs)
        if self.proton_fast_preprocess_enabled and not (
            self.mode == "cubic"
            and self.bev_grid.align_bev_z_to_ct_slices
            and self.bev_grid.bicubic_z_align
            and self.bev_grid.bicubic_z_align_backend == "triton"
        ):
            raise ValueError(
                "proton fast preprocessing requires aligned cubic Triton z-align"
            )

        r = self.proton_input_upscale_factor
        grid_type = type(self.bev_grid)
        self.input_grid = (
            self.bev_grid
            if r == 1
            else grid_type(
                shape_dhw=(
                    self.bev_grid.nx * r,
                    self.bev_grid.ny * r,
                    self.bev_grid.nz,
                ),
                spacing_dhw=(
                    float(self.bev_grid.spacing_dhw[0]) / r,
                    float(self.bev_grid.spacing_dhw[1]) / r,
                    float(self.bev_grid.spacing_dhw[2]),
                ),
                sad_mm=self.bev_grid.sad_mm,
                plane_origin_offset_mm=self.bev_grid.plane_origin_offset_mm,
                segment_native_size=self.bev_grid.segment_native_size,
                align_bev_z_to_ct_slices=self.bev_grid.align_bev_z_to_ct_slices,
                bicubic_z_align=self.bev_grid.bicubic_z_align,
                bicubic_z_align_backend=(
                    self.bev_grid.bicubic_z_align_backend
                ),
            )
        )
        self.g_lin = self.pipeline.build_bev_index_grid(self.input_grid)

        model_dir = Path(__file__).resolve().parent.parent / "model"
        if str(model_dir) not in sys.path:
            sys.path.insert(0, str(model_dir))
        from physics_conditioning import BEVMaterialConditioning

        self.material_conditioning = BEVMaterialConditioning(
            dz_mm=float(self.input_grid.spacing_dhw[0]),
            include_class=False,
            density_calibration=self.proton_density_calibration,
        ).to(self.device)
        self.model_conditioning: torch.Tensor | None = None
        self.model_energy_index: torch.Tensor | None = None
        self.proton_energy_levels = tuple(catalog.proton_energy_levels)
        self.proton_energy_sigmas = tuple(catalog.proton_energy_sigmas)
        if self.proton_energy_token and not self.proton_energy_levels:
            raise ValueError(
                "proton energy-token conditioning requires catalog energy levels"
            )
        self.proton_fast_module = (
            _load_fast_preprocess_module()
            if self.proton_fast_preprocess_enabled
            else None
        )
        self.proton_fast_cache: OrderedDict[
            str, tuple[Any, dict[str, int]]
        ] = OrderedDict()
        # Cumulative since process start; ``materialize`` reports per-batch
        # deltas so the startup warm-up in prepare_streaming_dose_roi_boxes
        # is not charged to the first training batch.
        self.proton_fast_cache_hit_count = 0
        self.proton_fast_cache_miss_count = 0
        self.proton_fast_cache_build_seconds = 0.0
        self.proton_fast_cache_evictions = 0

    @staticmethod
    def _ray_basis(rec: OtfGpuSampleRecord) -> np.ndarray:
        if rec.source_xyz_mm is None or rec.target_xyz_mm is None:
            raise ValueError(f"{rec.sample_id}: missing proton ray geometry")
        source = np.asarray(rec.source_xyz_mm, dtype=np.float64)
        target = np.asarray(rec.target_xyz_mm, dtype=np.float64)
        depth = target - source
        norm = float(np.linalg.norm(depth))
        if norm <= 0:
            raise ValueError(f"{rec.sample_id}: source and target coincide")
        depth /= norm
        axial = np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
        horizontal = np.cross(axial, depth)
        horizontal_norm = float(np.linalg.norm(horizontal))
        if horizontal_norm < 1e-6:
            raise ValueError(
                f"{rec.sample_id}: ray is parallel to the CT axial direction"
            )
        horizontal /= horizontal_norm
        axial = np.cross(depth, horizontal)
        axial /= np.linalg.norm(axial)
        return np.stack((depth, horizontal, axial)).astype(np.float32)

    def _build_mac_cache_for_geometry(
        self, nx: int, ny: int, nz: int, dx: float
    ) -> dict[str, dict[str, list[float]]]:
        out: dict[str, dict[str, list[float]]] = {}
        sample_ids = sorted(
            set(self.catalog.train_ids)
            | set(self.catalog.val_ids)
            | set(self.catalog.test_ids)
        )
        for sid in sample_ids:
            rec = self.catalog.records[sid]
            basis = self._ray_basis(rec)
            source = np.asarray(rec.source_xyz_mm, dtype=np.float32)
            target = np.asarray(rec.target_xyz_mm, dtype=np.float32)
            plane_origin = target - (0.5 * nx * dx) * basis[0]
            out[sid] = {
                "s": source.tolist(),
                "dx": basis[0].tolist(),
                "dy": basis[1].tolist(),
                "U": basis.reshape(-1).tolist(),
                "src": plane_origin.tolist(),
                "off": [0.0, -(ny // 2) + 0.5, -(nz // 2) + 0.5],
                "NX": nx,
                "NY": ny,
                "NZ": nz,
                "z_align_mm": 0.0,
            }
        return out

    def _build_mac_cache(self) -> dict[str, dict[str, list[float]]]:
        # Called by the base constructor before input_grid is created.
        r = self.proton_input_upscale_factor
        return self._build_mac_cache_for_geometry(
            int(self.bev_grid.nx) * r,
            int(self.bev_grid.ny) * r,
            int(self.bev_grid.nz),
            float(self.bev_grid.spacing_dhw[0]) / r,
        )

    def _get_segment_cpu(self, sample_id: str) -> np.ndarray:
        cached = self.segment_cache.get(sample_id)
        if cached is not None:
            return cached
        rec = self.catalog.records[sample_id]
        if rec.sigma_spot_mm is None:
            raise ValueError(f"{sample_id}: missing proton spot width")
        y = (
            np.arange(self.input_grid.ny, dtype=np.float32)
            - self.input_grid.ny // 2
            + 0.5
        ) * float(self.input_grid.spacing_dhw[1])
        z = (
            np.arange(self.input_grid.nz, dtype=np.float32)
            - self.input_grid.nz // 2
            + 0.5
        ) * float(self.input_grid.spacing_dhw[2])
        radius2 = y[:, None] ** 2 + z[None, :] ** 2
        sigma2 = float(rec.sigma_spot_mm) ** 2
        spot = np.exp(-0.5 * radius2 / sigma2).astype(np.float32)
        self.segment_cache[sample_id] = spot
        return spot

    @staticmethod
    def _pack_phases(
        volume: torch.Tensor, upscale_factor: int
    ) -> torch.Tensor:
        r = int(upscale_factor)
        if r == 1:
            return volume.unsqueeze(1)
        depth_fine, height_fine, width = volume.shape
        if depth_fine % r or height_fine % r:
            raise ValueError(
                f"fine volume {tuple(volume.shape)} is not divisible by {r}"
            )
        depth = depth_fine // r
        height = height_fine // r
        return (
            volume.reshape(depth, r, height, r, width)
            .permute(0, 1, 3, 2, 4)
            .reshape(depth, r * r, height, width)
        )

    def _batch_energy_index(self, sample_ids: Sequence[str]) -> torch.Tensor:
        """Map each sample's beamlet energy onto its table level."""
        indices: list[int] = []
        for sample_id in sample_ids:
            energy = self.catalog.records[sample_id].energy_mev
            if energy is None:
                raise ValueError(f"{sample_id}: missing proton energy")
            index = min(
                range(len(self.proton_energy_levels)),
                key=lambda value: abs(
                    self.proton_energy_levels[value] - float(energy)
                ),
            )
            if abs(self.proton_energy_levels[index] - float(energy)) > 1e-3:
                raise ValueError(
                    f"{sample_id}: energy {energy} is absent from the energy table"
                )
            indices.append(index)
        return torch.tensor(indices, device=self.device, dtype=torch.long)

    def _range_conditioning(
        self,
        anatomy_fine: torch.Tensor,
        energy_mev: float,
        energy_index: int | None = None,
    ) -> torch.Tensor:
        """Packed WET / remaining-range planes for one fine anatomy volume."""
        mc = self.material_conditioning
        anatomy_fine = anatomy_fine.float()
        wet, remaining = mc.fixed_range_wet(
            anatomy_fine, energy_mev, normalisation_mm=300.0
        )
        wet = wet.clamp(0.0, 2.0)
        remaining = remaining.clamp(-2.0, 1.0)
        r = self.proton_input_upscale_factor
        planes = [self._pack_phases(wet, r), self._pack_phases(remaining, r)]
        return torch.cat(planes, dim=1)

    @staticmethod
    def _ray_key(rec: OtfGpuSampleRecord) -> tuple[str, int, int]:
        if rec.ray_id is None:
            raise ValueError(f"{rec.sample_id}: missing proton ray id")
        return rec.patient_id, int(rec.beam_id), int(rec.ray_id)

    def _range_conditioning_batch(
        self,
        anatomy_unique: torch.Tensor,
        inverse_tensor: torch.Tensor,
        energies_mev: torch.Tensor,
        energy_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Packed range planes for a batch, classified once per unique ray.

        Material classification is energy independent, so it runs on the
        unique-ray volumes and is expanded to the batch before the
        energy-dependent evaluation. Several energy layers of one ray share a
        single classification pass.
        """
        mc = self.material_conditioning
        state_unique = mc.hu_to_rsp_state(anatomy_unique)
        state = tuple(
            value.index_select(0, inverse_tensor) for value in state_unique
        )
        wet, remaining = mc.fixed_range_wet_from_rsp_state(
            *state, energies_mev, normalisation_mm=300.0
        )

        wet = wet.clamp(0.0, 2.0)
        remaining = remaining.clamp(-2.0, 1.0)
        r = self.proton_input_upscale_factor
        packed_wet = torch.stack([self._pack_phases(value, r) for value in wet])
        packed_remaining = torch.stack(
            [self._pack_phases(value, r) for value in remaining]
        )
        planes = [packed_wet, packed_remaining]
        return torch.cat(planes, dim=2)

    def _get_proton_fast_cache(
        self, patient_id: str
    ) -> tuple[Any, dict[str, int]]:
        cached = self.proton_fast_cache.get(patient_id)
        if cached is not None:
            self.proton_fast_cache.move_to_end(patient_id)
            self.proton_fast_cache_hit_count += 1
            return cached
        if self.proton_fast_module is None:
            raise RuntimeError("proton fast preprocessing is disabled")
        self.proton_fast_cache_miss_count += 1
        t_build = time.perf_counter()
        ct_coeff, ct_shape, spacing, origin, _ = self._get_ct_gpu(patient_id)
        sample_ids = sorted(
            sid for sid, rec in self.catalog.records.items()
            if rec.patient_id == patient_id
        )
        tasks = [self.catalog.records[sid] for sid in sample_ids]
        ct_coeff_t = torch.from_dlpack(
            self.cp.ascontiguousarray(ct_coeff.astype(self.cp.float32, copy=False))
        )
        cache = self.proton_fast_module.build_fast_proton_plan_cache(
            tasks,
            self.mac_cache,
            self.input_grid,
            self.bev_grid,
            ct_coeff_t,
            ct_shape,
            spacing,
            origin,
            self.device,
            output_upscale_factor=self.ct_space_bev_upscale_factor,
            collapsed_dtype=torch.float32,
        )
        for index, sid in enumerate(sample_ids):
            context = cache.backprojection_contexts[index]
            self.prebuilt_ct_loss_geometry[sid] = {
                "affine": cache.inverse_affines[index],
                "roi_shape": context.roi_shape,
                "roi_box": context.roi_box,
            }
        result = cache, {sid: index for index, sid in enumerate(sample_ids)}
        self.proton_fast_cache[patient_id] = result
        while len(self.proton_fast_cache) > self.proton_fast_cache_patients:
            self.proton_fast_cache.popitem(last=False)
            self.proton_fast_cache_evictions += 1
        self.proton_fast_cache_build_seconds += time.perf_counter() - t_build
        return result

    def prepare_streaming_dose_roi_boxes(
        self,
    ) -> dict[str, tuple[int, int, int, int, int, int]]:
        """Build plain CPU ROI metadata before DataLoader workers start.

        The same proton fast-plan construction also warms the patient caches
        used during training, so this does not duplicate steady-state work.
        """
        if not self.proton_fast_preprocess_enabled:
            raise RuntimeError(
                "worker-side proton dose cropping requires proton fast preprocessing"
            )
        for patient_id in self.catalog.patient_ids:
            self._get_proton_fast_cache(patient_id)
        active_ids = sorted(
            set(self.catalog.train_ids)
            | set(self.catalog.val_ids)
            | set(self.catalog.test_ids)
        )
        boxes = {
            sample_id: tuple(
                int(value)
                for value in self.prebuilt_ct_loss_geometry[sample_id]["roi_box"]
            )
            for sample_id in active_ids
        }
        if len(boxes) != len(active_ids):
            raise RuntimeError("failed to prepare every proton dose ROI")
        return boxes

    def _fast_proton_inputs(
        self,
        sample_ids: Sequence[str],
        patient_ids: Sequence[str],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
        """Batch affine preprocessing and reuse CT/material state by ray."""
        if self.proton_fast_module is None:
            raise RuntimeError("proton fast preprocessing is disabled")
        ct_items: list[torch.Tensor | None] = [None] * len(sample_ids)
        fluence_items: list[torch.Tensor | None] = [None] * len(sample_ids)
        conditioning_items: list[torch.Tensor | None] = [None] * len(sample_ids)
        grouped: defaultdict[str, list[int]] = defaultdict(list)
        for batch_index, patient_id in enumerate(patient_ids):
            grouped[patient_id].append(batch_index)
        unique_ray_count = 0

        for patient_id, batch_indices in grouped.items():
            cache, sample_index = self._get_proton_fast_cache(patient_id)
            cache_indices = torch.as_tensor(
                [sample_index[sample_ids[index]] for index in batch_indices],
                device=self.device,
                dtype=torch.long,
            )
            ray_to_unique: dict[tuple[str, int, int], int] = {}
            representatives: list[int] = []
            inverse: list[int] = []
            for local_index, batch_index in enumerate(batch_indices):
                key = self._ray_key(self.catalog.records[sample_ids[batch_index]])
                unique_index = ray_to_unique.get(key)
                if unique_index is None:
                    unique_index = len(representatives)
                    ray_to_unique[key] = unique_index
                    representatives.append(local_index)
                inverse.append(unique_index)
            unique_ray_count += len(representatives)
            representative_indices = cache_indices.index_select(
                0,
                torch.as_tensor(representatives, device=self.device, dtype=torch.long),
            )
            anatomy_cval, anatomy_clip_min, anatomy_clip_max = (
                self._anatomy_fill_clip()
            )
            model_ct_unique, anatomy_unique = (
                self.proton_fast_module.fast_preprocess_ct_batch(
                    cache,
                    representative_indices,
                    output_shape=self.input_grid.shape_dhw,
                    ct_min=(
                        float(self.stats["ct_min"])
                        if self.stats
                        else anatomy_clip_min
                    ),
                    ct_max=(
                        float(self.stats["ct_max"])
                        if self.stats
                        else anatomy_clip_max
                    ),
                    normalize_ct=self.stats is not None,
                    anatomy_cval=anatomy_cval,
                    anatomy_clip=(anatomy_clip_min, anatomy_clip_max),
                    output_dtype=torch.float32,
                )
            )
            fluence = self.proton_fast_module.fast_preprocess_aperture_batch(
                cache,
                cache_indices,
                output_shape=self.input_grid.shape_dhw,
                spacing_yz=(
                    float(self.input_grid.spacing_dhw[1]),
                    float(self.input_grid.spacing_dhw[2]),
                ),
                sad_mm=float(self.input_grid.sad_mm),
                output_dtype=torch.float32,
            )
            inverse_tensor = torch.as_tensor(
                inverse, device=self.device, dtype=torch.long
            )
            model_ct = model_ct_unique.index_select(0, inverse_tensor)
            energies = torch.as_tensor(
                [
                    float(self.catalog.records[sample_ids[index]].energy_mev)
                    for index in batch_indices
                ],
                device=self.device,
                dtype=torch.float32,
            )
            conditioning = self._range_conditioning_batch(
                anatomy_unique,
                inverse_tensor,
                energies,
                energy_index=None,
            )
            for local_index, batch_index in enumerate(batch_indices):
                ct_items[batch_index] = self._pack_phases(
                    model_ct[local_index], self.proton_input_upscale_factor
                )
                fluence_items[batch_index] = self._pack_phases(
                    fluence[local_index], self.proton_input_upscale_factor
                )
                conditioning_items[batch_index] = conditioning[local_index]

        if any(value is None for value in ct_items + fluence_items + conditioning_items):
            raise RuntimeError("failed to populate proton fast-preprocess batch")
        return (
            torch.stack([value for value in ct_items if value is not None]),
            torch.stack([value for value in fluence_items if value is not None]),
            torch.stack([value for value in conditioning_items if value is not None]),
            len(grouped),
            unique_ray_count,
        )

    def materialize(
        self,
        batch: dict[str, Any],
        *,
        augment: BEVAugmentConfig | None = None,
        need_bev_target: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if augment is not None:
            raise ValueError("proton CT-space training does not support BEV flips")
        self.cp.cuda.Device(self.device_index).use()
        torch.cuda.set_device(self.device_index)
        t_total = time.perf_counter()
        timings: defaultdict[str, float] = defaultdict(float)
        self._ct_loss_samples = []
        sample_ids = [
            str(x) for x in self._batch_list(batch, "sample_id")
        ]
        patient_ids = [
            str(x) for x in self._batch_list(batch, "patient_id")
        ]
        if len(sample_ids) != len(patient_ids):
            raise ValueError("sample_id and patient_id batch lengths differ")

        dose_items = (
            self._batch_list(batch, "dose") if "dose" in batch else None
        )
        dose_roi_boxes = (
            self._batch_list(batch, "dose_roi_box")
            if "dose_roi_box" in batch
            else None
        )
        streamed_dose_gpu: list[torch.Tensor] | None = None
        streamed_events: list[torch.cuda.Event] | None = None
        if dose_items is not None:
            streamed_dose_gpu = []
            streamed_events = []
            timings["dose_streamed_mib"] = sum(
                int(dose.numel()) * int(dose.element_size())
                for dose in dose_items
            ) / float(1 << 20)
            timings["dose_roi_streamed"] = float(
                len(dose_items) if dose_roi_boxes is not None else 0
            )
            with torch.cuda.stream(self.dose_transfer_stream):
                for dose_cpu in dose_items:
                    dose = dose_cpu.to(
                        self.device, dtype=torch.float32, non_blocking=True
                    )
                    event = torch.cuda.Event()
                    event.record(self.dose_transfer_stream)
                    streamed_dose_gpu.append(dose)
                    streamed_events.append(event)

        ct_tensors: list[torch.Tensor] = []
        fluence_tensors: list[torch.Tensor] = []
        conditioning_tensors: list[torch.Tensor] = []
        bev_targets: list[torch.Tensor] = []
        anatomy_cval, anatomy_clip_min, anatomy_clip_max = self._anatomy_fill_clip()
        fast_ct_batch: torch.Tensor | None = None
        fast_fluence_batch: torch.Tensor | None = None
        fast_conditioning_batch: torch.Tensor | None = None
        if self.proton_fast_preprocess_enabled:
            t = time.perf_counter()
            hits_before = self.proton_fast_cache_hit_count
            misses_before = self.proton_fast_cache_miss_count
            build_before = self.proton_fast_cache_build_seconds
            evictions_before = self.proton_fast_cache_evictions
            (
                fast_ct_batch,
                fast_fluence_batch,
                fast_conditioning_batch,
                patient_groups,
                unique_rays,
            ) = self._fast_proton_inputs(sample_ids, patient_ids)
            torch.cuda.synchronize(self.device)
            timings["prep.fast_affine"] = time.perf_counter() - t
            timings["prep.fast_affine_groups"] = float(patient_groups)
            timings["prep.unique_rays"] = float(unique_rays)
            timings["prep.ray_reuse"] = float(len(sample_ids) - unique_rays)
            # Plan-cache rebuilds are the dominant cost when
            # --proton-fast-cache-patients is smaller than the patient count;
            # prep.fast_cache_build is the share of prep.fast_affine spent
            # rebuilding rather than doing per-batch affine work.
            timings["proton_fast_cache_hit"] = float(
                self.proton_fast_cache_hit_count - hits_before
            )
            timings["proton_fast_cache_miss"] = float(
                self.proton_fast_cache_miss_count - misses_before
            )
            timings["proton_fast_cache_evicted"] = float(
                self.proton_fast_cache_evictions - evictions_before
            )
            timings["prep.fast_cache_build"] = (
                self.proton_fast_cache_build_seconds - build_before
            )
            timings["proton_fast_cache_resident"] = float(
                len(self.proton_fast_cache)
            )
        for index, (sid, patient_id) in enumerate(
            zip(sample_ids, patient_ids)
        ):
            rec = self.catalog.records[sid]
            ct_coeff, ct_shape, spacing, origin, hit = self._get_ct_gpu(
                patient_id
            )
            timings["ct_cache_hit" if hit else "ct_cache_miss"] += 1.0
            prep_times: dict[str, float] = {}
            if (
                fast_ct_batch is not None
                and fast_fluence_batch is not None
                and fast_conditioning_batch is not None
            ):
                ct_tensors.append(fast_ct_batch[index])
                fluence_tensors.append(fast_fluence_batch[index])
                conditioning_tensors.append(fast_conditioning_batch[index])
            else:
                spot = self.cp.asarray(self._get_segment_cpu(sid), self.cp.float32)
                bev_ct, fluence, prep_times = (
                    self.pipeline.prepare_bev_input_volumes(
                        patient_id,
                        str(rec.mac_path),
                        ct_coeff,
                        ct_shape,
                        spacing,
                        origin,
                        self.mode,
                        self.mac_cache,
                        self.input_grid.nx,
                        self.input_grid.ny,
                        self.input_grid.nz,
                        self.g_lin,
                        seg_resized=spot,
                        grid=self.input_grid,
                        anatomy_cval=anatomy_cval,
                        anatomy_clip_min=anatomy_clip_min,
                    )
                )
                bev_ct = self.cp.clip(
                    bev_ct, anatomy_clip_min, anatomy_clip_max
                ).astype(self.cp.float32, copy=False)
                anatomy_fine = torch.from_dlpack(bev_ct)
                if self.stats is not None:
                    normalized_ct = (
                        anatomy_fine - float(self.stats["ct_min"])
                    ) / (
                        float(self.stats["ct_max"])
                        - float(self.stats["ct_min"])
                    )
                else:
                    normalized_ct = anatomy_fine
                r = self.proton_input_upscale_factor
                ct_tensors.append(self._pack_phases(normalized_ct, r))
                fluence_tensors.append(
                    self._pack_phases(torch.from_dlpack(fluence), r)
                )
                if rec.energy_mev is None:
                    raise ValueError(f"{sid}: missing proton energy")
                conditioning_tensors.append(
                    self._range_conditioning(
                        anatomy_fine,
                        rec.energy_mev,
                        energy_index=None,
                    )
                )

            if streamed_dose_gpu is not None and streamed_events is not None:
                streamed_events[index].synchronize()
                dose_xyz: np.ndarray | torch.Tensor = streamed_dose_gpu[index]
            else:
                dose_xyz = self._get_dose_cpu(sid)
            self._append_ct_loss_sample(
                sid,
                rec,
                ct_shape,
                spacing,
                origin,
                dose_xyz,
                dose_roi_box=(
                    tuple(int(value) for value in dose_roi_boxes[index])
                    if dose_roi_boxes is not None
                    else None
                ),
            )
            if need_bev_target:
                raise RuntimeError(
                    "proton BEV targets require the CT round-trip IDD metrics, "
                    "which this repository does not ship"
                )
                coords, use_bicubic_zcollapse = (
                    self._prepare_proton_roundtrip_bev_geometry(
                        patient_id,
                        rec,
                        ct_coeff,
                        ct_shape,
                        spacing,
                        origin,
                    )
                )
                sample = self._ct_loss_samples[-1]
                sample.update(
                    {
                        "bev_coords": coords,
                        "use_bicubic_zcollapse": use_bicubic_zcollapse,
                    }
                )
                # The streamed target may contain only the CT loss ROI. Rebuild
                # the full CT volume with zeros outside that same ROI before
                # mapping it to the coarse BEV grid, exactly as for prediction.
                bev_targets.append(
                    self._ct_prediction_to_bev(
                        sample["target"],
                        sample,
                        positive_clip_before_interp=True,
                    )
                )
            for key, value in prep_times.items():
                timings[key] += float(value)

        self.model_conditioning = torch.stack(conditioning_tensors, dim=0)
        self.model_energy_index = (
            self._batch_energy_index(sample_ids)
            if self.proton_energy_token
            else None
        )
        ct_batch = torch.stack(ct_tensors, dim=0)
        fluence_batch = torch.stack(fluence_tensors, dim=0)
        label = (
            torch.stack(bev_targets, dim=0)
            if bev_targets
            else torch.empty(
                (len(sample_ids), 0), device=self.device, dtype=torch.float32
            )
        )
        timings["total"] = time.perf_counter() - t_total
        self.last_timings = dict(timings)
        return ct_batch, fluence_batch, label
