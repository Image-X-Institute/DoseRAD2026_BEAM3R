from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm
from monai.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "model"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "model", "CNN_Mamba"))

from physics_conditioning import ProtonBraggResidualHead
from utils import (
    ProtonRayBatchSampler,
    TrainAmpConfig,
    load_proton_energy_token_values,
    BEVAugmentConfig,
    add_amp_args,
    add_augmentation_args,
    add_resume_args,
    add_multimodal_data_args,
    add_otf_input_upscale_arg,
    add_regression_loss_args,
    add_gradient_mae_loss_arg,
    add_training_performance_args,
    add_scheduler_args,
    add_optimizer_args,
    build_training_optimizer,
    get_optimizer_lr_for_logging,
    resolve_schedulefree_warmup_steps,
    set_optimizer_eval_mode,
    set_optimizer_train_mode,
    uses_schedule_free_optimizer,
    add_tensorboard_args,
    create_lr_scheduler,
    step_lr_scheduler,
    amp_backward_step,
    AsyncBevMetricsLogger,
    BevMetricsAccumulator,
    build_bev_augment_config_from_args,
    build_mamba_convlstm_criterion,
    compute_train_stats,
    create_multimodal_train_val_datasets,
    collate_otf_gpu_batch,
    validate_multimodal_data_args,
    load_stats_json,
    maybe_compile_model,
    resolve_train_amp,
    save_stats_json,
    epoch_batch_progress,
    load_init_from_checkpoint,
    save_training_checkpoints,
    seed_dataloader_worker,
    seed_everything,
    try_resume_training,
    train_autocast,
    TrainingLossHistory,
    OtfGpuBatchMaterializer,
    create_otf_gpu_batch_materializer,
    run_bev_depth_range_build_if_requested,
    plot_training_losses,
    update_training_loss_plot,
    validate_stats_dict,
    validate_val_metrics_args,
    load_bev_grid_config,
)
from plan_validation import build_plan_gantry_lookup
from model import CNN_Mamba2_L2_SpatialMix, MambaDoseArchConfig, validate_mamba_dose_arch_config


def enable_fast_cudnn() -> None:
    """Prefer fast cuDNN kernels over repeatable CUDA execution."""
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def _skip_standard_bev_metrics(materializer) -> bool:
    """Standard BEV metrics need a dose-sample output volume. The direct-coefficient
    heads emit spline/packed coefficients instead (packed also has an incompatible
    shape), so skip them and rely on the CT-space metrics plus --ct-bev-idd-metrics
    (bev_idd_from_ct), which are reconstructed from the CT-space prediction inside
    ``materializer.loss``.
    """
    if materializer is None:
        return False
    return bool(
        getattr(materializer, "ct_space_direct_spline_coefficients", False)
        or getattr(materializer, "ct_space_direct_packed_coefficients", False)
    )


def _model_conditioning(materializer) -> torch.Tensor | None:
    """Proton range conditioning for the current batch (``None`` for photons).

    ``ProtonOtfGpuBatchMaterializer.materialize`` publishes the phase-packed
    WET / remaining-range planes it built for the batch; photon materializers
    and pre-materialized data formats have no such attribute.
    """
    if materializer is None:
        return None
    return getattr(materializer, "model_conditioning", None)

def _model_energy_index(materializer) -> torch.Tensor | None:
    if materializer is None:
        return None
    return getattr(materializer, "model_energy_index", None)

def _forward_model(model, ct, proj, materializer):
    kwargs = {"conditioning": _model_conditioning(materializer)}
    energy_index = _model_energy_index(materializer)
    if energy_index is not None:
        kwargs["energy_index"] = energy_index
    return model(ct, proj, **kwargs)

def _refine_proton_output(
    model: nn.Module,
    output: torch.Tensor,
    conditioning: torch.Tensor | None,
    fluence: torch.Tensor,
) -> torch.Tensor:
    refiner = getattr(model, "proton_output_refiner", None)
    if refiner is None:
        return output
    if conditioning is None:
        raise RuntimeError("proton output refiner requires range conditioning")
    return refiner(
        output,
        conditioning.to(output.dtype),
        fluence.to(output.dtype),
    )

def train_one_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
    device,
    *,
    desc: str = "train",
    amp: TrainAmpConfig | None = None,
    scaler: torch.amp.GradScaler | None = None,
    materializer: OtfGpuBatchMaterializer | None = None,
    train_augment: BEVAugmentConfig | None = None,
    max_grad_norm: float | None = None,
    max_batches: int = 0,
):
    model.train()
    set_optimizer_train_mode(optimizer)
    epoch_loss = 0.0
    nb = device.type == "cuda"
    amp = amp or TrainAmpConfig(enabled=False, autocast_dtype=torch.float16)
    progress = epoch_batch_progress(dataloader, desc=desc)
    for batch_index, batch in enumerate(progress):
        # Smoke-test escape hatch: 0 means run the whole epoch.
        if max_batches and batch_index >= max_batches:
            break
        if materializer is not None:
            ct, proj, label = materializer.materialize(
                batch,
                augment=train_augment,
                need_bev_target=not materializer.ct_space_loss,
            )
            t = materializer.last_timings
            progress.set_postfix(
                otf=f"{t.get('total', 0.0):.3f}s",
                ct=f"{t.get('prep.ct_map', 0.0):.3f}s",
                proj=f"{t.get('prep.projection', 0.0):.3f}s",
                dose=f"{t.get('dose_transfer', 0.0):.3f}s",
                hit=int(t.get('ct_cache_hit', 0.0)),
                miss=int(t.get('ct_cache_miss', 0.0)),
                refresh=False,
            )
        else:
            ct = batch[0][0].to(device, non_blocking=nb)
            proj = batch[0][1].to(device, non_blocking=nb)
            label = batch[1].to(device, non_blocking=nb)

        optimizer.zero_grad(set_to_none=True)
        with train_autocast(amp):
            output = _forward_model(model, ct, proj, materializer)
            output = _refine_proton_output(
                model, output, _model_conditioning(materializer), proj
            )
            loss = (
                materializer.loss(output, label, criterion)
                if materializer is not None
                else criterion(output, label)
            )
        amp_backward_step(
            loss, optimizer, scaler,
            parameters=model.parameters() if max_grad_norm else None,
            max_grad_norm=max_grad_norm,
        )
        epoch_loss += loss.detach().float().item()
    return epoch_loss / len(dataloader)


def validate_one_epoch(
    model,
    dataloader,
    criterion,
    device,
    *,
    optimizer=None,
    desc: str = "val",
    amp: TrainAmpConfig | None = None,
    materializer: OtfGpuBatchMaterializer | None = None,
    bev_metrics: BevMetricsAccumulator | None = None,
    max_batches: int = 0,
):
    model.eval()
    set_optimizer_eval_mode(optimizer)
    epoch_loss = 0.0
    nb = device.type == "cuda"
    amp = amp or TrainAmpConfig(enabled=False, autocast_dtype=torch.float16)
    with torch.no_grad():
        progress = epoch_batch_progress(dataloader, desc=desc)
        for batch_index, batch in enumerate(progress):
            if max_batches and batch_index >= max_batches:
                break
            if materializer is not None:
                # The direct/packed coefficient heads have no dose-sample BEV
                # output, so the standard BEV metrics are skipped and a BEV
                # target would be built for nothing.
                collect_bev_sample_metrics = (
                    bev_metrics is not None
                    and not _skip_standard_bev_metrics(materializer)
                )
                ct, proj, label = materializer.materialize(
                    batch,
                    need_bev_target=(
                        not materializer.ct_space_loss
                        or collect_bev_sample_metrics
                        or (
                            bev_metrics is not None
                            and bool(
                                getattr(materializer, "ct_bev_idd_metrics", False)
                            )
                        )
                    ),
                )
                t = materializer.last_timings
                progress.set_postfix(
                    otf=f"{t.get('total', 0.0):.3f}s",
                    ct=f"{t.get('prep.ct_map', 0.0):.3f}s",
                    proj=f"{t.get('prep.projection', 0.0):.3f}s",
                    dose=f"{t.get('dose_transfer', 0.0):.3f}s",
                    hit=int(t.get('ct_cache_hit', 0.0)),
                    miss=int(t.get('ct_cache_miss', 0.0)),
                    refresh=False,
                )
            else:
                ct = batch[0][0].to(device, non_blocking=nb)
                proj = batch[0][1].to(device, non_blocking=nb)
                label = batch[1].to(device, non_blocking=nb)

            with train_autocast(amp):
                output = _forward_model(model, ct, proj, materializer)
                output = _refine_proton_output(
                    model, output, _model_conditioning(materializer), proj
                )
                loss = (
                    materializer.loss(
                        output,
                        label,
                        criterion,
                        ct_metrics=bev_metrics,
                        # keeps val loss comparable across arms that enable
                        # training-only terms and arms that do not
                        include_aux_losses=False,
                    )
                    if materializer is not None
                    else criterion(output, label)
                )
            if bev_metrics is not None and not _skip_standard_bev_metrics(materializer):
                bev_metrics.update(output, label, batch)
            epoch_loss += loss.detach().float().item()
    return epoch_loss / len(dataloader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train CNN-Mamba dose prediction model')
    parser.add_argument(
        '--data-dir',
        default=None,
        help=(
            "H5: directory with ct_dataset.h5, proj_dataset.h5, dose_dataset.h5. "
            "Bin (--data-format bin): BEV root with ct/, proj/, dose/ .bin folders. "
            "Optional for --data-format otf_gpu when --otf-gpu-full is set."
        ),
    )
    add_multimodal_data_args(parser)
    add_otf_input_upscale_arg(parser)
    add_augmentation_args(parser)
    add_regression_loss_args(parser, default="mse", choices=("mse", "huber"))
    add_gradient_mae_loss_arg(parser)
    add_amp_args(parser)
    add_resume_args(parser)
    add_training_performance_args(parser)
    parser.add_argument(
        '--fast-cudnn',
        action='store_true',
        help=(
            'Enable cuDNN benchmarking and allow nondeterministic CUDA kernels. '
            'This overrides deterministic CUDA execution requested by a training seed.'
        ),
    )
    parser.add_argument(
        '--ct-space-direct-packed-coefficients',
        action='store_true',
        help=(
            'Keep the fine direct-spline head as packed coarse-lattice phase channels '
            'and sample them directly in Triton, avoiding PixelShuffle materialisation.'
        ),
    )
    add_scheduler_args(parser)
    add_optimizer_args(parser)
    parser.add_argument(
        '--max-grad-norm',
        type=float,
        default=1.0,
        help=(
            'Clip global gradient norm to this value each optimizer step '
            '(<=0 disables). Recommended for Mamba3, which can spike catastrophically '
            'without it. Default: 1.0. Overridden by --grad-clip when that flag is set '
            '(shared xLSTM-compatible alias from training performance args).'
        ),
    )
    parser.add_argument(
        '--warmup-epochs',
        type=int,
        default=0,
        help=(
            'Linear LR warmup over this many epochs (base LR reached at the end of '
            'warmup); the main scheduler takes over afterwards. 0 disables. Default: 0.'
        ),
    )
    parser.add_argument(
        '--training-seed',
        type=int,
        default=None,
        help=(
            'If set, seed Python/NumPy/torch RNGs, the DataLoader generator, and '
            'DataLoader workers, and request deterministic CUDA algorithms '
            '(default: None, no explicit seeding). Required for short ablation '
            'arms to be comparable; --fast-cudnn overrides the determinism.'
        ),
    )
    parser.add_argument(
        '--init-from',
        default=None,
        metavar='CHECKPOINT',
        help=(
            'Warm-start model weights from a parent checkpoint (weights only; no '
            'optimizer, scheduler or epoch counter). Applied only when --out-dir '
            'has no checkpoint of its own to resume from. Newly enabled features '
            'are allowed to be absent from the parent; any other missing or '
            'unexpected key is an error.'
        ),
    )
    parser.add_argument(
        '--freeze-modules',
        default=None,
        metavar='PREFIX,PREFIX,...',
        help=(
            'Comma-separated top-level module name prefixes to exclude from '
            'training (e.g. "decoder" or "decoder,slice_scaler"). Sets '
            'requires_grad_(False) on matching parameters after weights are '
            'loaded (--init-from or --resume) and before the optimizer is built; '
            'standard optimizers skip params with no gradient, so no other '
            'change is needed. Freezing a module does not skip its backward '
            'pass if anything upstream of it is still trainable -- only its own '
            'weight update and optimizer state are saved.'
        ),
    )
    parser.add_argument(
        '--unet-skips',
        action='store_true',
        help=(
            'Add ReZero-gated encoder→decoder skip connections. The bottleneck is '
            'stride-8 laterally, so without them the decoder rebuilds every '
            'lateral edge - body contour, lung/tissue interface - from a 25×25 '
            'code. Identity at initialisation.'
        ),
    )
    parser.add_argument(
        '--depth-film',
        action='store_true',
        help=(
            'FiLM-condition the bottleneck and first decoder stage on physical '
            'depth along the beam axis. The per-slice 2D blocks are otherwise '
            'shared across all depths with no depth input. Identity at '
            'initialisation.'
        ),
    )
    parser.add_argument(
        '--density-grad-weight',
        type=float,
        default=0.0,
        help=(
            'If > 0, add this weight times CT-space squared error restricted to '
            'density-interface voxels. Training-only: validation loss keeps the '
            'unweighted definition. Requires --ct-space-loss. Default: 0.0.'
        ),
    )
    parser.add_argument(
        '--density-grad-scale',
        type=float,
        default=0.15,
        help=(
            'Relative-electron-density gradient magnitude (per mm) that saturates '
            'the interface indicator used by --density-grad-weight (default: 0.15, '
            'which a sharp lung/soft-tissue boundary reaches under the central '
            'differences used here at 2 mm CT spacing). Raise it to weight only '
            'the sharpest interfaces.'
        ),
    )
    parser.add_argument(
        '--density-grad-cache-patients',
        type=int,
        default=16,
        help=(
            'Patients whose interface-weight volumes stay resident on the GPU for '
            '--density-grad-weight (default: 16, roughly 33 MB each). Too small and '
            'a full-volume gradient is recomputed on nearly every batch under '
            'shuffling. Result-neutral: this is a cache size only.'
        ),
    )
    add_tensorboard_args(parser)
    parser.add_argument('--out-dir', default=None,
                        help='Output directory (default: ./results/<timestamp>_Mamba)')
    parser.add_argument('--device', default='cuda:0',
                        help='PyTorch device string (default: cuda:0)')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=0,
        help=(
            "Stop each training epoch after N batches (0 = whole epoch). "
            "Smoke-test tooling only -- it truncates the epoch, so any run "
            "using it is not a real training run."
        ),
    )
    parser.add_argument(
        "--max-val-batches",
        type=int,
        default=0,
        help="Stop each validation pass after N batches (0 = whole pass).",
    )
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr-patience', type=int, default=15)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument(
        '--normalise',
        action='store_true',
        help=(
            "Apply photon-style stats: CT clip to [ct_min, ct_max] then scale to [0, 1]; "
            "dose divided by global dose_scale from training H5 (or from --stats)."
        ),
    )
    parser.add_argument(
        '--stats',
        default=None,
        help=(
            "Path to JSON with ct_min, ct_max, dose_scale. "
            "Used with --normalise; if omitted, stats are computed from train split in H5."
        ),
    )
    parser.add_argument(
        '--masked-mae-weight',
        type=float,
        default=0.0,
        help=(
            "If > 0, add this weight times masked high-dose MAE (same definition as "
            "validation.masked_mae) to the MSE loss."
        ),
    )
    parser.add_argument(
        '--masked-mae-threshold',
        type=float,
        default=0.10,
        help="High-dose mask fraction of per-sample max(target) (default: 0.10).",
    )
    parser.add_argument(
        '--idd-weight',
        type=float,
        default=0.0,
        help=(
            "If > 0, add this weight times IDD curve RMS distance (same definition as "
            "validation.idd_curve_distance) to the loss."
        ),
    )
    parser.add_argument(
        '--idd-metric',
        choices=('rms', 'mae'),
        default='rms',
        help=(
            "IDD distance metric when --idd-weight > 0: 'rms' matches validation.idd_curve_distance; "
            "'mae' is mean absolute difference between IDD curves after normalising both by GT IDD peak."
        ),
    )
    parser.add_argument(
        '--decoder',
        choices=('conv_transpose', 'bilinear_conv'),
        default='conv_transpose',
        help=(
            "Decoder head: conv_transpose (original ConvTranspose stack) or "
            "bilinear_conv (bilinear upsample ×2 + Conv after each stage)."
        ),
    )
    parser.add_argument(
        '--window-attention',
        action='store_true',
        help=(
            "After Mamba temporal mixing, apply spatial self-attention on each "
            "slice feature map before spatial_post."
        ),
    )
    parser.add_argument(
        '--window-attn-heads',
        type=int,
        default=4,
        help="Number of attention heads for --window-attention (default: 4).",
    )
    parser.add_argument(
        '--window-size',
        type=int,
        default=5,
        help=(
            "Window size stored on SimpleWindowAttention (default: 5); "
            "current implementation uses full spatial attention."
        ),
    )
    parser.add_argument(
        '--spatial-mid',
        action='store_true',
        help=(
            'Insert spatial_mid (dw 7×7 + pw 1×1) between two Mamba stacks of '
            '--layers blocks each (default: off, single Mamba stack only).'
        ),
    )
    parser.add_argument(
        '--bidirectional-mamba',
        action='store_true',
        help=(
            'Run a second Mamba scan over the depth-reversed sequence with '
            'independent backward blocks; fuse with the forward pass by addition.'
        ),
    )
    parser.add_argument(
        '--bwd-gate-lr-mult',
        type=float,
        default=50.0,
        help=(
            'LR multiplier for the zero-initialised bidirectional fusion gates '
            '(default: 50). They start at exactly 0 so the model matches the '
            'unidirectional one at step 0; at the base LR a zero scalar barely '
            'moves, and the run can finish with the backward scan still off.'
        ),
    )
    parser.add_argument(
        '--temporal-conv',
        action='store_true',
        help=(
            'Add non-causal depthwise TemporalConv1d along depth after Mamba '
            '(default: off).'
        ),
    )
    parser.add_argument(
        '--temporal-kernel-size',
        type=int,
        default=5,
        help='Kernel size for --temporal-conv (default: 5).',
    )
    parser.add_argument(
        '--mamba-core',
        choices=('mamba2', 'mamba3'),
        default='mamba2',
        help='Temporal SSM core in each bottleneck block (default: mamba2).',
    )
    parser.add_argument(
        '--mamba-d',
        type=int,
        default=32,
        metavar='D',
        help=(
            'Mamba bottleneck width d after in_proj (default: 32). '
            'For --mamba-core mamba3 with default --mamba3-headdim 64, use d>=64 '
            '(e.g. --mamba-d 64) so d_inner=128 and nheads>=2.'
        ),
    )
    parser.add_argument(
        '--model-capacity-scale',
        type=float,
        default=1.0,
        metavar='S',
        help=(
            'Uniformly scale encoder, decoder, Mamba bottleneck, and '
            'aperture-prefix hidden width (default: 1.0; 1.5 maps encoder '
            'width 64 to 96 and --mamba-d 32 to 48).'
        ),
    )
    parser.add_argument(
        '--mamba3-headdim',
        type=int,
        default=64,
        help='Mamba3 headdim when --mamba-core mamba3 (default: 64).',
    )
    parser.add_argument(
        '--mamba3-chunk-size',
        type=int,
        default=64,
        help='Mamba3 chunk_size when --mamba-core mamba3 (default: 64).',
    )
    parser.add_argument(
        '--n-prefix',
        type=int,
        default=0,
        metavar='K',
        help='Number of learnable prefix tokens prepended before each Mamba pass (default: 0).',
    )
    parser.add_argument(
        '--prefix-mode',
        default='constant',
        choices=('constant', 'aperture'),
        help=(
            "Prefix mechanism used by --n-prefix: 'constant' is a single learned "
            "per-channel token shared across the batch (default); 'aperture' is a "
            "spatially-varying prefix gated by the beam aperture (x2's first depth slice)."
        ),
    )
    parser.add_argument(
        '--prefix-gate-hidden',
        type=int,
        default=16,
        metavar='H',
        help="Hidden width of the aperture gate encoder, only used when --prefix-mode aperture (default: 16).",
    )
    parser.add_argument(
        '--use-scaler',
        action='store_true',
        help='Add a per-slice scalar gate before the decoder (tanh+1, zero-init).',
    )
    parser.add_argument(
        '--scaler-pool',
        default='max',
        choices=('avg', 'max'),
        help='Pooling type for the slice scaler (default: max).',
    )
    parser.add_argument(
        '--scaler-layers',
        type=int,
        default=3,
        help='Number of layers in the slice scaler MLP (1, 2, or 3; default: 3).',
    )
    parser.add_argument(
        '--decoder-final-relu',
        action='store_true',
        help='Use nn.ReLU instead of LeakyReLU after the last decoder upsampling stage.',
    )
    parser.add_argument(
        '--output-softplus',
        action='store_true',
        help='Apply nn.Softplus to model output as an optional final layer.',
    )
    parser.add_argument(
        '--output-relu',
        action='store_true',
        help='Apply nn.ReLU to model output as an optional final layer.',
    )
    args = parser.parse_args()
    validate_multimodal_data_args(args)
    # Token levels drive the embedding and must stay 0 when the token is off.
    try:
        args.proton_energy_token_values = (
            load_proton_energy_token_values(args.proton_beam_parameters)
            if args.proton_energy_token
            else []
        )
        args.proton_energy_token_levels = len(args.proton_energy_token_values)
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit(
            f"invalid proton energy-token configuration: {exc}"
        ) from exc
    try:
        validate_val_metrics_args(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if int(getattr(args, 'otf_input_upscale_factor', 1)) > 1:
        # the round-trip IDD metrics rebuild a BEV on the coarse lattice, which
        # no longer matches the fine sampling grid once phases are packed
        if getattr(args, 'ct_bev_idd_metrics', False) or int(
            getattr(args, 'ct_roundtrip_idd_freq', 0)
        ) > 0:
            raise SystemExit(
                "--otf-input-upscale-factor 2 does not currently support "
                "CT-to-BEV round-trip IDD validation (--ct-bev-idd-metrics / "
                "--ct-roundtrip-idd-freq)"
            )
    if args.output_softplus and args.output_relu:
        raise SystemExit("--output-softplus and --output-relu are mutually exclusive")
    loader_generator = None
    if args.training_seed is not None:
        loader_generator = seed_everything(args.training_seed)
        print(f"Training seed: {args.training_seed} (deterministic algorithms requested)")
    if args.fast_cudnn:
        enable_fast_cudnn()
        print("Fast cuDNN enabled (benchmark=True, deterministic=False)")
        if args.training_seed is not None:
            print(
                "Warning: --fast-cudnn overrides the determinism requested by "
                "--training-seed; shuffling stays reproducible but CUDA kernels "
                "do not. Drop it when arms must be comparable."
            )
    if args.ct_space_direct_packed_coefficients:
        if not (
            args.ct_space_loss
            and args.ct_space_bev_pixel_shuffle
            and args.ct_space_direct_spline_coefficients
        ):
            raise SystemExit(
                '--ct-space-direct-packed-coefficients requires --ct-space-loss, '
                '--ct-space-bev-pixel-shuffle, and '
                '--ct-space-direct-spline-coefficients'
            )

    bev_grid = load_bev_grid_config(getattr(args, "bev_grid_config", None))
    wants_cuda = str(args.device).lower().startswith("cuda")
    if wants_cuda and not torch.cuda.is_available():
        raise SystemExit(
            "Requested CUDA device but torch.cuda.is_available() is False. "
            "CNN-Mamba uses mamba_ssm Triton kernels that require GPU tensors. "
            "Run inside `docker run --gpus all ...` and verify CUDA from the same environment."
        )
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        n = torch.cuda.device_count()
        idx = device.index if device.index is not None else 0
        if idx >= n:
            raise SystemExit(
                f"Requested {device} but only {n} CUDA device(s) visible "
                f"(check CUDA_VISIBLE_DEVICES and --gpus)."
            )
        # mamba_ssm Triton uses the current CUDA device; without this, cuda:N tensors + default
        # context on GPU 0 can raise: ValueError: Pointer argument ... (cpu tensor?)
        torch.cuda.set_device(idx)
        print(
            f"Using device {device} (torch.cuda.current_device()={torch.cuda.current_device()})"
        )
    else:
        print(
            "WARNING: CPU training; mamba_ssm Triton ops typically require CUDA and may fail."
        )

    # Proton runs pack r**2 phases per modality and add two broadcast
    # conditioning planes (WET, remaining range); photon runs add none.
    proton_otf = str(getattr(args, "otf_modality", "photon")) == "proton"
    input_phase_channels = (
        int(args.proton_input_upscale_factor) ** 2
        if proton_otf
        else int(args.otf_input_upscale_factor) ** 2
    )
    conditioning_channels = 2 * input_phase_channels if proton_otf else 0
    mamba_arch = MambaDoseArchConfig(
        decoder=args.decoder,
        input_h=bev_grid.ny,
        input_w=bev_grid.nz,
        d=args.mamba_d,
        use_window_attention=args.window_attention,
        window_attn_heads=args.window_attn_heads,
        window_size=args.window_size,
        output_softplus=args.output_softplus,
        output_relu=args.output_relu,
        decoder_final_relu=args.decoder_final_relu,
        depth_height_pixel_shuffle=args.ct_space_bev_pixel_shuffle,
        return_packed_dh_coefficients=args.ct_space_direct_packed_coefficients,
        spatial_mid=args.spatial_mid,
        bidirectional_mamba=args.bidirectional_mamba,
        use_temporal_conv=args.temporal_conv,
        temporal_kernel_size=args.temporal_kernel_size,
        mamba_core=args.mamba_core,
        mamba3_headdim=args.mamba3_headdim,
        mamba3_chunk_size=args.mamba3_chunk_size,
        n_prefix=args.n_prefix,
        prefix_mode=args.prefix_mode,
        prefix_gate_hidden=args.prefix_gate_hidden,
        use_scaler=args.use_scaler,
        scaler_pool=args.scaler_pool,
        scaler_layers=args.scaler_layers,
        model_capacity_scale=args.model_capacity_scale,
        input_phase_channels=input_phase_channels,
        conditioning_channels=conditioning_channels,
        energy_token_levels=int(getattr(args, "proton_energy_token_levels", 0)),
        unet_skips=args.unet_skips,
        depth_film=args.depth_film,
        film_sad_mm=bev_grid.sad_mm,
        film_depth_spacing_mm=bev_grid.spacing_dhw[0],
    )
    try:
        validate_mamba_dose_arch_config(mamba_arch)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    model = CNN_Mamba2_L2_SpatialMix(
        mamba_arch,
        use_channels_last=args.channels_last,
    ).to(device)
    if proton_otf and args.proton_output_refiner == "bragg_residual":
        if not args.ct_space_direct_packed_coefficients:
            raise SystemExit(
                "--proton-output-refiner bragg_residual requires "
                "--ct-space-direct-packed-coefficients"
            )
        model.proton_output_refiner = ProtonBraggResidualHead(
            input_phase_channels
        ).to(device)
        print(
            f"Attached ProtonBraggResidualHead "
            f"(phase_channels={input_phase_channels}, zero-init identity)"
        )
    if args.channels_last:
        # Convert only 4-D conv weights: prefix tokens (--n-prefix) are 5-D
        # (1, K, C, 1, 1) and Module.to(memory_format=...) rejects them.
        model._apply(
            lambda tensor: tensor.contiguous(memory_format=torch.channels_last)
            if tensor.ndim == 4 else tensor
        )
    print(f"Mamba base architecture: {asdict(mamba_arch)}")
    print(f"Mamba effective architecture: {asdict(model.arch)}")

    current_time  = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    experiment_dir = args.out_dir or os.path.join('results', f'{current_time}_Mamba')

    bin_shape = tuple(args.bin_shape) if args.bin_shape else None
    data_kw = dict(
        patient_split_json=args.patient_split_json,
        validation_fraction=args.validation_fraction,
        seed=args.split_seed,
        bin_shape=bin_shape,
        bin_dtype=args.bin_dtype,
        merged_storage_dtype=args.merged_storage_dtype,
        baseline_pb_dir=args.baseline_pb_dir,
        baseline_split=args.baseline_split,
        segment_mac_dir=args.segment_mac_dir,
        otf_dose_dtype=args.otf_dose_dtype,
        ct_name=args.ct_name,
        sct_dir=args.sct_dir,
        otf_gpu_full=args.otf_gpu_full,
        bev_grid=bev_grid,
        otf_modality=args.otf_modality,
        proton_beam_parameters=args.proton_beam_parameters,
    )

    stats = None
    stats_computed = False
    if args.normalise:
        if args.stats:
            stats_path = Path(args.stats)
            if not stats_path.is_file():
                raise FileNotFoundError(f"--stats file not found: {stats_path}")
            stats = load_stats_json(stats_path)
            validate_stats_dict(stats)
            print(f"Loaded normalisation stats from {stats_path}")
            print(
                f"  ct_min={stats['ct_min']} ct_max={stats['ct_max']} "
                f"dose_scale={float(stats['dose_scale']):.6g}"
            )
        else:
            print("Computing normalisation stats from training split (dose global max)...")
            stats = compute_train_stats(args.data_format, args.data_dir, **data_kw)
            stats_computed = True
            validate_stats_dict(stats)
            print(
                f"  ct_min={stats['ct_min']} ct_max={stats['ct_max']} "
                f"dose_scale={stats['dose_scale']:.6g}"
            )

    train_augment = build_bev_augment_config_from_args(args)
    if train_augment is not None:
        print(f"Train augmentation: random H/W flips (p={train_augment.p_h})")

    print(f"Data format: {args.data_format}  data_dir={args.data_dir}")
    train_ds, val_ds = create_multimodal_train_val_datasets(
        args.data_format,
        args.data_dir,
        stats=stats,
        train_augment=train_augment,
        otf_stream_dose=args.otf_stream_dose,
        **data_kw,
    )
    otf_materializer = None
    if args.data_format == "otf_gpu":
        run_bev_depth_range_build_if_requested(
            args,
            train_ds.catalog,
            device_index=device.index if device.index is not None else 0,
            out_dir=experiment_dir,
        )
        otf_materializer = create_otf_gpu_batch_materializer(
            train_ds.catalog,
            device,
            args,
            stats=stats,
            out_dir=experiment_dir,
            bev_grid=bev_grid,
        )
    log_dir    = os.path.join(experiment_dir, 'log')
    model_dir  = os.path.join(experiment_dir, 'model')
    script_dir = os.path.join(experiment_dir, 'script')
    os.makedirs(log_dir,    exist_ok=True)
    os.makedirs(model_dir,  exist_ok=True)
    os.makedirs(script_dir, exist_ok=True)
    shutil.copy(os.path.abspath(__file__), os.path.join(script_dir, os.path.basename(__file__)))

    if stats_computed:
        stats_out = os.path.join(experiment_dir, 'dl_segment_stats.json')
        save_stats_json(stats, stats_out)
        print(f"Wrote computed stats to {stats_out}")

    config_hyper = dict(vars(args))
    config_hyper["bev_grid"] = bev_grid.to_dict()
    config_hyper["mamba_arch"] = mamba_arch.to_json_dict()
    config_hyper["argv"] = sys.argv
    config_hyper["torch_device"] = str(device)
    config_hyper["cuda_available"] = bool(torch.cuda.is_available())
    config_hyper["experiment_dir_abs"] = os.path.abspath(experiment_dir)
    config_hyper["stats_computed_in_run"] = stats_computed
    if stats is not None:
        config_hyper["normalisation_stats"] = {
            "ct_min": float(stats["ct_min"]),
            "ct_max": float(stats["ct_max"]),
            "dose_scale": float(stats["dose_scale"]),
        }
    else:
        config_hyper["normalisation_stats"] = None
    if device.type == "cuda":
        config_hyper["cuda_current_device_index"] = torch.cuda.current_device()
    else:
        config_hyper["cuda_current_device_index"] = None
    config_hyper_path = os.path.join(script_dir, "config_hyper.json")
    with open(config_hyper_path, "w", encoding="utf-8") as f_cfg:
        json.dump(config_hyper, f_cfg, indent=2, sort_keys=True)
    print(f"Wrote config_hyper to {config_hyper_path}")

    pin = device.type == "cuda"
    otf_collate = (
        collate_otf_gpu_batch
        if args.data_format == "otf_gpu"
        and args.otf_gpu_full
        and args.otf_stream_dose
        else None
    )
    train_loader_kwargs = dict(
        num_workers=args.num_workers,
        pin_memory=pin,
        persistent_workers=(args.num_workers > 0),
        collate_fn=otf_collate,
        generator=loader_generator,
        worker_init_fn=seed_dataloader_worker if loader_generator is not None else None,
    )
    if proton_otf and args.proton_ray_aware_batches:
        # Keeping a ray's energy layers in one batch lets the materializer
        # compute that ray's CT and geometry once instead of per layer.
        ray_batch_sampler = ProtonRayBatchSampler(train_ds, args.batch_size)
        train_loader = DataLoader(
            train_ds,
            batch_sampler=ray_batch_sampler,
            **train_loader_kwargs,
        )
        print(
            "Proton ray-aware batching enabled: "
            f"{len(ray_batch_sampler)} batches"
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            **train_loader_kwargs,
        )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin,
        persistent_workers=(args.num_workers > 0),
        collate_fn=otf_collate,
    )

    criterion, loss_desc = build_mamba_convlstm_criterion(args)
    print(f"Loss: {loss_desc}")
    amp_config, grad_scaler = resolve_train_amp(
        device, use_amp=args.amp, amp_dtype=args.amp_dtype
    )
    if args.amp and device.type != "cuda":
        print("Warning: --amp ignored (CUDA not available); training in full precision.")
    elif amp_config.enabled:
        scaler_note = "with GradScaler" if grad_scaler is not None else "without GradScaler"
        print(f"AMP enabled: {amp_config.autocast_dtype} ({scaler_note})")
    # Bidirectional fusion gates are zero-initialised for warm-start parity, so
    # they need a faster group than the rest of the network or they never leave
    # zero. ReduceLROnPlateau scales every group by the same factor, so the
    # multiplier survives decays.
    _gate_names = {"bwd_gate", "bwd_gate2"}
    _gates = [p for n, p in model.named_parameters() if n in _gate_names]
    _param_groups = None
    if _gates:
        _others = [p for n, p in model.named_parameters() if n not in _gate_names]
        _gate_lr = float(args.lr) * float(args.bwd_gate_lr_mult)
        _param_groups = [
            {"params": _others, "lr": float(args.lr)},
            {"params": _gates, "lr": _gate_lr},
        ]
        print(
            f"Bidirectional gates: {len(_gates)} scalar(s) at lr={_gate_lr:g} "
            f"({args.bwd_gate_lr_mult:g}x base {args.lr:g})"
        )
    optimizer = build_training_optimizer(
        model,
        args,
        param_groups=_param_groups,
        warmup_steps=resolve_schedulefree_warmup_steps(args, len(train_loader)),
    )
    if uses_schedule_free_optimizer(args):
        print(
            f"Optimizer: Schedule-Free AdamW lr={args.lr:g}, "
            f"weight_decay={args.weight_decay:g}, "
            f"warmup_steps={resolve_schedulefree_warmup_steps(args, len(train_loader))}"
        )
    scheduler = create_lr_scheduler(optimizer, args)

    # --grad-clip (shared with xLSTM via add_training_performance_args) overrides
    # the Mamba-specific --max-grad-norm default when provided.
    clip_val = args.grad_clip if args.grad_clip is not None else args.max_grad_norm
    max_grad_norm = clip_val if clip_val and clip_val > 0 else None
    print(
        f"Gradient clipping: max_norm={max_grad_norm}"
        if max_grad_norm is not None
        else "Gradient clipping: disabled"
    )
    warmup_epochs = 0 if uses_schedule_free_optimizer(args) else max(int(args.warmup_epochs), 0)
    base_lr = float(args.lr)
    if warmup_epochs > 0:
        print(f"LR warmup: linear over {warmup_epochs} epoch(s) to base LR {base_lr:.2e}")

    start_epoch, best_val_loss, resume_msgs = try_resume_training(
        model, optimizer, scheduler, grad_scaler, model_dir, device, resume=args.resume
    )
    for msg in resume_msgs:
        print(msg)
    if args.init_from:
        if start_epoch > 0:
            print(
                f"Ignoring --init-from: resumed from {model_dir} at epoch "
                f"{start_epoch + 1}, so the run's own checkpoint takes precedence."
            )
        else:
            allow_missing = []
            if args.unet_skips:
                allow_missing += ["skip_proj_", "skip_gate_"]
            if args.depth_film:
                allow_missing += ["film_bottleneck.", "film_decoder."]
            added = load_init_from_checkpoint(
                model, args.init_from, device, allow_missing_prefixes=allow_missing
            )
            print(f"Warm-started weights from {args.init_from}")
            if added:
                print(
                    f"  {len(added)} newly initialised parameter(s) from enabled "
                    f"features: {added}"
                )
            else:
                print("  all parameters loaded from the parent checkpoint")
    if args.freeze_modules:
        # After weights load, before compile: torch.compile specialises on which
        # tensors require grad, and freezing does not depend on how the weights
        # got here (--init-from or --resume), so it must be re-applied every
        # process start regardless -- requires_grad is not part of a state_dict.
        prefixes = tuple(p.strip() for p in args.freeze_modules.split(",") if p.strip())
        frozen = trainable = 0
        for name, p in model.named_parameters():
            if name.split(".", 1)[0] in prefixes:
                p.requires_grad_(False)
                frozen += p.numel()
            else:
                trainable += p.numel()
        if frozen == 0:
            raise ValueError(
                f"--freeze-modules {args.freeze_modules!r} matched no top-level "
                f"module names; available: "
                f"{sorted({n.split('.', 1)[0] for n, _ in model.named_parameters()})}"
            )
        print(
            f"Frozen: {prefixes} -- {frozen:,} parameter(s) excluded, "
            f"{trainable:,} ({trainable / (frozen + trainable) * 100:.1f}%) still trainable"
        )
    if args.compile_model:
        print(
            f"Compiling model with torch.compile(mode={args.compile_mode!r}, "
            f"fullgraph={args.compile_fullgraph})"
        )
    model = maybe_compile_model(model, args)

    loss_history = TrainingLossHistory.load(experiment_dir)
    if loss_history.epochs:
        plot_training_losses(experiment_dir, loss_history)

    metrics_logger = AsyncBevMetricsLogger(experiment_dir, enabled=args.tensorboard)

    log_file = os.path.join(log_dir, 'run.log')
    log_mode = 'a' if start_epoch > 0 and os.path.isfile(log_file) else 'w'
    with open(log_file, log_mode) as f:
        if start_epoch > 0:
            f.write("\n--- resumed training ---\n")
        epoch_iter = range(start_epoch, args.epochs)
        for epoch in tqdm(
            epoch_iter, desc="epochs", unit="epoch", initial=start_epoch, total=args.epochs
        ):
            ep = epoch + 1
            in_warmup = warmup_epochs > 0 and ep <= warmup_epochs
            if in_warmup:
                warmup_lr = base_lr * ep / warmup_epochs
                for pg in optimizer.param_groups:
                    pg['lr'] = warmup_lr
            train_loss = train_one_epoch(
                model, train_loader, optimizer, criterion, device,
                desc=f"train {ep}/{args.epochs}",
                amp=amp_config,
                scaler=grad_scaler,
                materializer=otf_materializer,
                train_augment=train_augment,
                max_batches=args.max_train_batches,
                max_grad_norm=max_grad_norm,
            )
            collect_standard_metrics = AsyncBevMetricsLogger.should_collect(
                ep, args.val_metrics_freq, enabled=metrics_logger.enabled
            )
            bev_metrics = (
                metrics_logger.create_accumulator(
                    device,
                    # h5 datasets carry no sample catalog; beam IDD needs the
                    # otf_gpu records for the beam/cp join anyway
                    catalog=getattr(val_ds, "catalog", None),
                    # lru_cached, so the plan tree is walked once per process
                    gantry_lookup=(
                        build_plan_gantry_lookup(
                            args.baseline_pb_dir, args.baseline_split
                        )
                        if args.val_beam_idd
                        else None
                    ),
                    idd_floor_frac=args.idd_floor_frac,
                )
                if collect_standard_metrics
                else None
            )
            val_loss = validate_one_epoch(
                model, val_loader, criterion, device,
                optimizer=optimizer,
                desc=f"val {ep}/{args.epochs}",
                amp=amp_config,
                materializer=otf_materializer,
                bev_metrics=bev_metrics,
                max_batches=args.max_val_batches,
            )
            current_lr = get_optimizer_lr_for_logging(optimizer)
            msg = (f"Epoch {epoch+1}/{args.epochs}, LR: {current_lr:.2e}, "
                   f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            tqdm.write(msg)
            f.write(msg + '\n')

            if otf_materializer is not None:
                otf_materializer.save_depth_cache()

            # Hold off the main scheduler until warmup completes so it doesn't
            # react to (or reduce) the ramped-up LR mid-warmup.
            if not in_warmup:
                step_lr_scheduler(scheduler, val_loss)

            best_val_loss, ckpt_msgs = save_training_checkpoints(
                model, model_dir, epoch, val_loss, best_val_loss,
                optimizer=optimizer, scheduler=scheduler, scaler=grad_scaler,
            )
            for ckpt_msg in ckpt_msgs:
                tqdm.write(ckpt_msg)
                f.write(ckpt_msg + "\n")

            plot_path = update_training_loss_plot(
                experiment_dir, loss_history,
                epoch_1based=ep, train_loss=train_loss, val_loss=val_loss, lr=current_lr,
            )
            plot_msg = f"Updated loss plot: {plot_path}"
            tqdm.write(plot_msg)
            f.write(plot_msg + "\n")

            metrics_logger.record_losses(
                ep, train_loss, val_loss, lr=current_lr
            )

            if bev_metrics is not None:
                metrics_logger.record(ep, bev_metrics.summary_means())
            f.flush()

    metrics_logger.finalize()
