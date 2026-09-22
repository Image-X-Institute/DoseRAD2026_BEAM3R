"""Inference-only fusions for the fixed-shape CNN-Mamba submission model.

The checkpoint is loaded into the original module hierarchy first.  These
patches are then installed in place, preserving all checkpoint parameter names:

* Mamba block ``LayerNorm`` uses the equivalent fused Triton implementation
  shipped with ``mamba_ssm``.
* Mamba3 projection preparation is collapsed into one Triton pass. Its x/z
  outputs remain aligned views of a minimally padded projection and are read
  directly by the forward kernel instead of copied into contiguous tensors.
* The frozen Mamba3 recurrence prepares Q/K per chunk on chip, avoiding the
  training kernel's backward-only Q/K and scalar scratch tensors.
* Decoder transposed convolutions retain their registered bias parameter but
  omit the standalone cuDNN bias kernel.  The following ReLU/LeakyReLU module
  adds that bias and applies the activation in one channels-last Triton pass.

They are deliberately not used by training: the decoder activation is in-place
and the primary purpose here is to minimize inference memory traffic.
"""

from __future__ import annotations

import os
import weakref
import types

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _bias_activation_inplace_kernel(
    x,
    bias,
    n_elements,
    channels: tl.constexpr,
    spatial_elements: tl.constexpr,
    channels_last: tl.constexpr,
    negative_slope: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    if channels_last:
        channel = offsets % channels
    else:
        channel = (offsets // spatial_elements) % channels
    value = tl.load(x + offsets, mask=mask).to(tl.float32)
    value += tl.load(bias + channel, mask=mask).to(tl.float32)
    value = tl.where(value >= 0.0, value, value * negative_slope)
    tl.store(x + offsets, value, mask=mask)


@triton.jit
def _bidirectional_merge_kernel(
    forward,
    backward_reversed,
    residual,
    gate,
    output,
    n_elements,
    depth,
    channels,
    height,
    width,
    forward_stride_b,
    forward_stride_t,
    forward_stride_c,
    forward_stride_h,
    forward_stride_w,
    backward_stride_b,
    backward_stride_t,
    backward_stride_c,
    backward_stride_h,
    backward_stride_w,
    residual_stride_b,
    residual_stride_t,
    residual_stride_c,
    residual_stride_h,
    residual_stride_w,
    output_stride_b,
    output_stride_t,
    output_stride_c,
    output_stride_h,
    output_stride_w,
    has_gate: tl.constexpr,
    subtract_residual: tl.constexpr,
    output_fp32: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    index = offsets
    w_index = index % width
    index //= width
    h_index = index % height
    index //= height
    c_index = index % channels
    index //= channels
    t_index = index % depth
    b_index = index // depth

    forward_offset = (
        b_index * forward_stride_b
        + t_index * forward_stride_t
        + c_index * forward_stride_c
        + h_index * forward_stride_h
        + w_index * forward_stride_w
    )
    backward_offset = (
        b_index * backward_stride_b
        + (depth - 1 - t_index) * backward_stride_t
        + c_index * backward_stride_c
        + h_index * backward_stride_h
        + w_index * backward_stride_w
    )
    forward_value = tl.load(forward + forward_offset, mask=mask).to(tl.float32)
    backward_value = tl.load(
        backward_reversed + backward_offset, mask=mask
    ).to(tl.float32)
    if subtract_residual:
        residual_offset = (
            b_index * residual_stride_b
            + t_index * residual_stride_t
            + c_index * residual_stride_c
            + h_index * residual_stride_h
            + w_index * residual_stride_w
        )
        residual_value = tl.load(
            residual + residual_offset, mask=mask
        ).to(tl.float32)
        # Match eager BF16 subtraction before the gate multiplication.
        backward_value = (backward_value - residual_value).to(tl.bfloat16)
    if has_gate:
        gate_value = tl.load(gate).to(tl.float32)
        # A one-element FP32 parameter promotes this multiplication and the
        # subsequent addition to FP32 in eager PyTorch.
        result = forward_value + backward_value * gate_value
    else:
        result = forward_value + backward_value
    if not output_fp32:
        result = result.to(tl.bfloat16)
    output_offset = (
        b_index * output_stride_b
        + t_index * output_stride_t
        + c_index * output_stride_c
        + h_index * output_stride_h
        + w_index * output_stride_w
    )
    tl.store(output + output_offset, result, mask=mask)


def _fused_bidirectional_merge(
    forward: torch.Tensor,
    backward_reversed: torch.Tensor,
    residual: torch.Tensor,
    gate: torch.Tensor | None,
    *,
    subtract_residual: bool,
) -> torch.Tensor:
    """Reverse and gated-merge one backward stack while retaining its layout."""
    if (
        not forward.is_cuda
        or forward.dtype != torch.bfloat16
        or backward_reversed.dtype != forward.dtype
        or residual.dtype != forward.dtype
    ):
        backward = torch.flip(backward_reversed, dims=[1])
        if gate is None:
            return forward + backward
        delta = backward - residual if subtract_residual else backward
        return forward + gate * delta
    if forward.ndim != 5 or forward.shape != backward_reversed.shape:
        raise ValueError("bidirectional merge expects matching five-dimensional tensors")
    if residual.shape != forward.shape:
        raise ValueError("bidirectional residual shape does not match scan output")
    output_dtype = (
        torch.promote_types(forward.dtype, gate.dtype)
        if gate is not None
        else forward.dtype
    )
    # Retain the scan's B,T,C,H,W view over channels-last storage.  The
    # following spatial block relies on this layout to stay on its established
    # cuDNN path; standard contiguous output is both slower and less numerically
    # stable after the convolution.
    output = torch.empty_strided(
        forward.shape,
        forward.stride(),
        device=forward.device,
        dtype=output_dtype,
    )
    batch, depth, channels, height, width = forward.shape
    n_elements = forward.numel()
    gate_arg = forward if gate is None else gate
    _bidirectional_merge_kernel[(triton.cdiv(n_elements, 1024),)](
        forward,
        backward_reversed,
        residual,
        gate_arg,
        output,
        n_elements,
        depth,
        channels,
        height,
        width,
        *forward.stride(),
        *backward_reversed.stride(),
        *residual.stride(),
        *output.stride(),
        has_gate=gate is not None,
        subtract_residual=bool(subtract_residual),
        output_fp32=output_dtype == torch.float32,
        BLOCK=1024,
    )
    return output


@triton.jit
def _mamba3_prepare_inference_kernel(
    projection,
    q,
    k,
    adt,
    dt,
    trap,
    angles,
    b_weight,
    c_weight,
    dt_bias,
    rows,
    sequence,
    projection_stride,
    d_inner: tl.constexpr,
    state: tl.constexpr,
    heads: tl.constexpr,
    angle_dim: tl.constexpr,
    eps: tl.constexpr,
    a_floor: tl.constexpr,
    write_angles: tl.constexpr,
    rows_per_program: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Prepare the projected Mamba3 inputs without copying the x/z views."""
    row_ids = (
        tl.program_id(0) * rows_per_program + tl.arange(0, rows_per_program)
    )
    columns = tl.arange(0, BLOCK)
    row_mask = row_ids[:, None] < rows
    source_row = row_ids[:, None] * projection_stride

    state_mask = row_mask & (columns[None, :] < state)
    b_offset: tl.constexpr = 2 * d_inner
    c_offset: tl.constexpr = b_offset + state
    b_value = tl.load(
        projection + source_row + b_offset + columns[None, :],
        mask=state_mask,
        other=0.0,
    ).to(tl.float32)
    c_value = tl.load(
        projection + source_row + c_offset + columns[None, :],
        mask=state_mask,
        other=0.0,
    ).to(tl.float32)
    b_rms = tl.rsqrt(tl.sum(b_value * b_value, axis=1) / state + eps)
    c_rms = tl.rsqrt(tl.sum(c_value * c_value, axis=1) / state + eps)
    b_value *= b_rms[:, None] * tl.load(
        b_weight + columns[None, :], mask=state_mask
    )
    c_value *= c_rms[:, None] * tl.load(
        c_weight + columns[None, :], mask=state_mask
    )
    state_output = row_ids[:, None] * state + columns[None, :]
    tl.store(k + state_output, b_value, mask=state_mask)
    tl.store(q + state_output, c_value, mask=state_mask)

    head_mask = row_mask & (columns[None, :] < heads)
    dt_offset: tl.constexpr = c_offset + state
    a_offset: tl.constexpr = dt_offset + heads
    trap_offset: tl.constexpr = a_offset + heads
    raw_dt = tl.load(
        projection + source_row + dt_offset + columns[None, :],
        mask=head_mask,
        other=0.0,
    ).to(tl.float32)
    raw_dt += tl.load(dt_bias + columns[None, :], mask=head_mask)
    raw_a = tl.load(
        projection + source_row + a_offset + columns[None, :],
        mask=head_mask,
        other=0.0,
    ).to(tl.float32)
    # Stable softplus, matching the two torch softplus operations in Mamba3.
    dt_value = tl.where(
        raw_dt > 20.0, raw_dt, tl.log(1.0 + tl.exp(raw_dt))
    )
    softplus_a = tl.where(
        raw_a > 20.0, raw_a, tl.log(1.0 + tl.exp(raw_a))
    )
    a_value = tl.minimum(-softplus_a, -a_floor)
    batch_index = row_ids[:, None] // sequence
    sequence_index = row_ids[:, None] % sequence
    scalar_output = (
        (batch_index * heads + columns[None, :]) * sequence + sequence_index
    )
    tl.store(dt + scalar_output, dt_value, mask=head_mask)
    tl.store(adt + scalar_output, a_value * dt_value, mask=head_mask)
    trap_value = tl.load(
        projection + source_row + trap_offset + columns[None, :],
        mask=head_mask,
    )
    tl.store(trap + scalar_output, trap_value, mask=head_mask)

    if write_angles:
        angle_count: tl.constexpr = heads * angle_dim
        angle_mask = row_mask & (columns[None, :] < angle_count)
        angle_offset: tl.constexpr = trap_offset + heads
        angle_source = columns[None, :] % angle_dim
        angle_value = tl.load(
            projection + source_row + angle_offset + angle_source,
            mask=angle_mask,
        ).to(tl.float32)
        angle_output = row_ids[:, None] * angle_count + columns[None, :]
        tl.store(angles + angle_output, angle_value, mask=angle_mask)


def _mamba3_siso_forward_strided(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    adt: torch.Tensor,
    dt: torch.Tensor,
    trap: torch.Tensor,
    q_bias: torch.Tensor,
    k_bias: torch.Tensor,
    angles: torch.Tensor,
    d: torch.Tensor,
    z: torch.Tensor,
    *,
    chunk_size: int,
) -> torch.Tensor:
    """Inference-only Mamba3 SISO launch which preserves aligned x/z views."""
    from mamba_ssm.ops.triton.mamba3.angle_dt import angle_dt_fwd
    from mamba_ssm.ops.triton.mamba3.mamba3_siso_fwd import (
        mamba3_siso_fwd_kernel,
    )

    batch, sequence, q_heads, q_dim = q.shape
    _, _, heads, value_dim = v.shape
    angles_cumsum = angle_dt_fwd(
        angles,
        dt,
        chunk_size=chunk_size,
        return_output_state=False,
    )

    output = torch.empty(
        (batch, sequence, heads, value_dim), device=v.device, dtype=v.dtype
    )
    # The forward kernel uses these as shared-memory spill buffers even when no
    # autograd state is requested.
    q_store = torch.empty(
        (batch, sequence, heads, q_dim), device=q.device, dtype=q.dtype
    )
    k_store = torch.empty_like(q_store)
    qk_store = torch.empty(
        (batch, heads, sequence), device=q.device, dtype=torch.float32
    )
    scale_store = torch.empty_like(qk_store)
    gamma_store = torch.empty_like(qk_store)

    head_dim_qk = triton.next_power_of_2(q_dim)
    head_dim_v = triton.next_power_of_2(value_dim)
    mamba3_siso_fwd_kernel[(heads, batch)](
        # Inputs; recurrent/variable-length inputs are unused for full-sequence
        # inference and are compiled out by the flags below.
        q,
        k,
        v,
        adt,
        dt,
        trap,
        q_bias,
        k_bias,
        angles_cumsum,
        d,
        z,
        None,
        None,
        None,
        None,
        # Outputs.
        output,
        None,
        None,
        None,
        None,
        q_store,
        k_store,
        qk_store,
        scale_store,
        gamma_store,
        None,
        None,
        # Input strides.
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        adt.stride(0), adt.stride(1), adt.stride(2),
        dt.stride(0), dt.stride(1), dt.stride(2),
        trap.stride(0), trap.stride(1), trap.stride(2),
        q_bias.stride(0), q_bias.stride(1),
        k_bias.stride(0), k_bias.stride(1),
        angles_cumsum.stride(0),
        angles_cumsum.stride(1),
        angles_cumsum.stride(2),
        angles_cumsum.stride(3),
        d.stride(0),
        z.stride(0), z.stride(1), z.stride(2), z.stride(3),
        # Initial-state and cu_seqlens strides.
        0, 0, 0, 0,
        0, 0, 0,
        0, 0, 0,
        0,
        # Output strides.
        output.stride(0),
        output.stride(1),
        output.stride(2),
        output.stride(3),
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0,
        0, 0, 0,
        q_store.stride(0),
        q_store.stride(1),
        q_store.stride(2),
        q_store.stride(3),
        k_store.stride(0),
        k_store.stride(1),
        k_store.stride(2),
        k_store.stride(3),
        qk_store.stride(0), qk_store.stride(1), qk_store.stride(2),
        scale_store.stride(0), scale_store.stride(1), scale_store.stride(2),
        gamma_store.stride(0), gamma_store.stride(1), gamma_store.stride(2),
        0, 0, 0, 0,
        0, 0, 0, 0,
        # Dimensions and compile-time constants.
        sequence,
        q_heads,
        q_dim,
        value_dim,
        angles_cumsum.shape[-1],
        chunk_size,
        head_dim_qk,
        head_dim_v,
        STORE_SSM_STATES_ADT_OUTV=False,
        HAS_INITIAL_STATES=False,
        RETURN_FINAL_STATES=False,
        HAS_D=True,
        HAS_Z=True,
        IS_VARLEN=False,
    )
    return output


def _scratch_free_mamba3_enabled() -> bool:
    # Opt in only after benchmarking the complete architecture/chunk-size pair.
    # It helps the chunk-16 bidirectional candidate, but regresses the shipped
    # chunk-32 unidirectional model despite reducing recurrent scratch traffic.
    value = os.environ.get("DOSERAD_MAMBA3_SCRATCH_FREE", "0").strip().lower()
    return value not in {"0", "false", "no", "off"}


def _fused_mamba3_angle_enabled() -> bool:
    value = os.environ.get("DOSERAD_MAMBA3_FUSED_ANGLE", "0").strip().lower()
    return value not in {"0", "false", "no", "off"}


def _bidirectional_merge_enabled() -> bool:
    # This boundary-only fusion is retained for profiling, but is deliberately
    # opt-in: it was neutral on the full bidirectional checkpoint and tiny
    # first-stack rounding changes can be amplified by the following BF16 conv.
    value = os.environ.get(
        "DOSERAD_MAMBA_BIDIRECTIONAL_MERGE", "0"
    ).strip().lower()
    return value not in {"0", "false", "no", "off"}


def _run_mamba_temporal_with_fused_merge(
    self,
    x_5d: torch.Tensor,
    blocks: nn.ModuleList,
    *,
    prefix: torch.Tensor | None = None,
    blocks_bwd: nn.ModuleList | None = None,
    bwd_gate: torch.Tensor | None = None,
) -> torch.Tensor:
    fwd = self._mamba_scan(x_5d, blocks, prefix=prefix)
    if blocks_bwd is None:
        return fwd
    x_reversed = torch.flip(x_5d, dims=[1])
    bwd_reversed = self._mamba_scan(
        x_reversed, blocks_bwd, prefix=prefix
    )
    pass_through = isinstance(self.in_proj, nn.Identity) and isinstance(
        self.out_proj, nn.Identity
    )
    return _fused_bidirectional_merge(
        fwd,
        bwd_reversed,
        x_5d,
        bwd_gate,
        subtract_residual=bool(bwd_gate is not None and pass_through),
    )


def install_bidirectional_merge_fusion(model: nn.Module) -> bool:
    """Install the reverse/gate merge only on compatible bidirectional models."""
    if not _bidirectional_merge_enabled():
        return False
    if getattr(model, "_mamba_bidirectional_merge_fused", False):
        return False
    if not callable(getattr(model, "_run_mamba_temporal", None)):
        return False
    if getattr(model, "blocks_bwd", None) is None:
        return False
    object.__setattr__(
        model,
        "_unfused_run_mamba_temporal",
        model._run_mamba_temporal,
    )
    object.__setattr__(model, "_mamba_bidirectional_merge_fused", True)
    model._run_mamba_temporal = types.MethodType(
        _run_mamba_temporal_with_fused_merge, model
    )
    return True


def _mamba3_siso_forward_scratch_free(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    adt: torch.Tensor,
    dt: torch.Tensor,
    trap: torch.Tensor,
    q_bias: torch.Tensor,
    k_bias: torch.Tensor,
    angles: torch.Tensor,
    d: torch.Tensor,
    z: torch.Tensor,
    *,
    chunk_size: int,
    fused_angle: bool,
) -> torch.Tensor:
    """Lazy package import keeps CPU-only structural tests dependency-light."""
    from .mamba3_scratch_free import mamba3_siso_forward_scratch_free

    if fused_angle:
        prepared_angles = angles
    else:
        from mamba_ssm.ops.triton.mamba3.angle_dt import angle_dt_fwd

        prepared_angles = angle_dt_fwd(
            angles,
            dt,
            chunk_size=chunk_size,
            return_output_state=False,
        )
    return mamba3_siso_forward_scratch_free(
        q,
        k,
        v,
        adt,
        dt,
        trap,
        q_bias,
        k_bias,
        prepared_angles,
        d,
        z,
        chunk_size=chunk_size,
        fused_angle=fused_angle,
    )


def _mamba3_prepared_forward(
    self,
    inputs: torch.Tensor,
    seq_idx=None,
    cu_seqlens=None,
    inference_params=None,
) -> torch.Tensor:
    """Mamba3 full-sequence forward specialized for frozen SISO inference."""
    if (
        not inputs.is_cuda
        or seq_idx is not None
        or cu_seqlens is not None
        or inference_params is not None
    ):
        return self._unfused_inference_forward(
            inputs, seq_idx, cu_seqlens, inference_params
        )

    batch, sequence, _ = inputs.shape
    padded_projection = F.linear(inputs, self._inference_padded_in_proj_weight)
    projection = padded_projection[..., : self._inference_projection_width]
    z = projection[..., : self.d_inner].view(
        batch, sequence, self.nheads, self.headdim
    )
    x = projection[..., self.d_inner : 2 * self.d_inner].view(
        batch, sequence, self.nheads, self.headdim
    )

    qk = torch.empty(
        (2, batch, sequence, self.num_bc_heads, self.d_state),
        device=inputs.device,
        dtype=projection.dtype,
    )
    adt_dt = torch.empty(
        (2, batch, self.nheads, sequence),
        device=inputs.device,
        dtype=torch.float32,
    )
    trap = torch.empty(
        (batch, self.nheads, sequence),
        device=inputs.device,
        dtype=projection.dtype,
    )
    angle_offset = (
        2 * self.d_inner + 2 * self.d_state + 3 * self.nheads
    )
    raw_angles = projection[
        ..., angle_offset : angle_offset + self.num_rope_angles
    ]
    angles = (
        raw_angles
        if self._mamba3_scratch_free and self._mamba3_fused_angle
        else torch.empty(
            (batch, sequence, self.nheads, self.num_rope_angles),
            device=inputs.device,
            dtype=torch.float32,
        )
    )
    rows = batch * sequence
    rows_per_program = 2
    _mamba3_prepare_inference_kernel[
        (triton.cdiv(rows, rows_per_program),)
    ](
        projection,
        qk[1],
        qk[0],
        adt_dt[0],
        adt_dt[1],
        trap,
        angles,
        self.B_norm.weight,
        self.C_norm.weight,
        self.dt_bias,
        rows,
        sequence,
        padded_projection.shape[-1],
        d_inner=self.d_inner,
        state=self.d_state,
        heads=self.nheads,
        angle_dim=self.num_rope_angles,
        eps=float(self.B_norm.eps),
        a_floor=float(self.A_floor),
        write_angles=not (
            self._mamba3_scratch_free and self._mamba3_fused_angle
        ),
        rows_per_program=rows_per_program,
        BLOCK=256,
        num_warps=4,
    )
    forward_impl = (
        _mamba3_siso_forward_scratch_free
        if self._mamba3_scratch_free
        else _mamba3_siso_forward_strided
    )
    forward_kwargs = {"chunk_size": self.chunk_size}
    if self._mamba3_scratch_free:
        forward_kwargs["fused_angle"] = self._mamba3_fused_angle
    output = forward_impl(
        qk[1],
        qk[0],
        x,
        adt_dt[0],
        adt_dt[1],
        trap,
        self.C_bias.squeeze(1),
        self.B_bias.squeeze(1),
        angles,
        self.D,
        z,
        **forward_kwargs,
    )
    return self.out_proj(
        output.reshape(batch, sequence, self.d_inner).to(x.dtype)
    )


class FusedInferenceLayerNorm(nn.Module):
    """Checkpoint-compatible wrapper around mamba_ssm's Triton LayerNorm."""

    def __init__(self, original: nn.LayerNorm) -> None:
        super().__init__()
        if len(original.normalized_shape) != 1:
            raise ValueError(
                "fused inference LayerNorm requires one normalized dimension"
            )
        # Register under the same names as nn.LayerNorm so state_dict keys do
        # not change if the already-loaded inference model is inspected.
        self.weight = original.weight
        self.bias = original.bias
        self.eps = float(original.eps)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if not inputs.is_cuda:
            return F.layer_norm(
                inputs,
                (inputs.shape[-1],),
                self.weight,
                self.bias,
                self.eps,
            )
        from mamba_ssm.ops.triton.layer_norm import layer_norm_fn

        return layer_norm_fn(
            inputs,
            self.weight,
            self.bias,
            eps=self.eps,
        )


class BiaslessInferenceConvTranspose2d(nn.ConvTranspose2d):
    """ConvTranspose2d retaining ``bias`` for state compatibility but not adding it."""

    def forward(
        self,
        inputs: torch.Tensor,
        output_size: list[int] | None = None,
    ) -> torch.Tensor:
        if self.padding_mode != "zeros":
            raise ValueError("Only zero padding is supported for ConvTranspose2d")
        output_padding = self._output_padding(
            inputs,
            output_size,
            self.stride,
            self.padding,
            self.kernel_size,
            2,
            self.dilation,
        )
        return F.conv_transpose2d(
            inputs,
            self.weight,
            None,
            self.stride,
            self.padding,
            output_padding,
            self.groups,
            self.dilation,
        )


class FusedInferenceBiasActivation(nn.Module):
    """Add a preceding convolution's bias and activate in one in-place pass."""

    def __init__(
        self,
        convolution: BiaslessInferenceConvTranspose2d,
        *,
        negative_slope: float,
    ) -> None:
        super().__init__()
        if convolution.bias is None:
            raise ValueError("fused decoder activation requires a convolution bias")
        # A weak reference avoids registering the convolution (and its
        # parameters) a second time below the activation in state_dict.
        object.__setattr__(self, "_convolution", weakref.ref(convolution))
        self.negative_slope = float(negative_slope)

    @property
    def bias(self) -> torch.Tensor:
        convolution = self._convolution()
        if convolution is None or convolution.bias is None:
            raise RuntimeError("fused decoder convolution is no longer available")
        return convolution.bias

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if not inputs.is_cuda:
            return F.leaky_relu(
                inputs + self.bias.view(1, -1, 1, 1),
                negative_slope=self.negative_slope,
                inplace=False,
            )
        if inputs.is_contiguous(memory_format=torch.channels_last):
            channels_last = True
        elif inputs.is_contiguous():
            channels_last = False
        else:
            # This is not expected from the production cuDNN decoder, but keep
            # a correct fallback instead of silently treating arbitrary strides
            # as contiguous storage.
            return F.leaky_relu(
                inputs + self.bias.view(1, -1, 1, 1),
                negative_slope=self.negative_slope,
                inplace=False,
            )
        n_elements = inputs.numel()
        spatial_elements = inputs.shape[2] * inputs.shape[3]
        _bias_activation_inplace_kernel[(triton.cdiv(n_elements, 1024),)](
            inputs,
            self.bias,
            n_elements,
            channels=inputs.shape[1],
            spatial_elements=spatial_elements,
            channels_last=channels_last,
            negative_slope=self.negative_slope,
            BLOCK=1024,
        )
        return inputs


def install_mamba3_preparation_fusions(
    model: nn.Module,
    *,
    scratch_free: bool | None = None,
    fused_angle: bool | None = None,
) -> int:
    """Install the aligned projection and strided Mamba3 inference forward."""
    if scratch_free is None:
        scratch_free = _scratch_free_mamba3_enabled()
    if fused_angle is None:
        fused_angle = _fused_mamba3_angle_enabled()
    replacement_count = 0
    for module in model.modules():
        if module.__class__.__name__ != "Mamba3":
            continue
        if getattr(module, "_mamba3_inference_preparation_fused", False):
            continue
        if module.is_mimo or module.is_outproj_norm:
            # The submission checkpoint uses the SISO path without output
            # normalization. Preserve the upstream implementation otherwise.
            continue
        if module.num_bc_heads != 1 or module.mimo_rank != 1:
            continue
        if module.in_proj.bias is not None:
            continue

        projection_width = module.in_proj.weight.shape[0]
        # The Mamba3 forward kernel loads x/z through TMA. Padding the GEMM row
        # to a 16-byte boundary lets it consume projected views directly rather
        # than materializing two large contiguous copies.
        alignment_elements = max(1, 16 // module.in_proj.weight.element_size())
        padded_width = (
            (projection_width + alignment_elements - 1) // alignment_elements
        ) * alignment_elements
        # Autocast emits BF16 even when checkpoint weights are FP32, for which
        # eight elements are required to retain the same byte alignment.
        padded_width = ((padded_width + 7) // 8) * 8
        padded_weight = F.pad(
            module.in_proj.weight.detach(),
            (0, 0, 0, padded_width - projection_width),
        )
        # object.__setattr__ deliberately avoids adding checkpoint/state_dict
        # keys: installation happens only after strict loading and freezing.
        object.__setattr__(module, "_inference_padded_in_proj_weight", padded_weight)
        object.__setattr__(module, "_inference_projection_width", projection_width)
        object.__setattr__(module, "_unfused_inference_forward", module.forward)
        object.__setattr__(module, "_mamba3_scratch_free", bool(scratch_free))
        object.__setattr__(module, "_mamba3_fused_angle", bool(fused_angle))
        object.__setattr__(module, "_mamba3_inference_preparation_fused", True)
        module.forward = types.MethodType(_mamba3_prepared_forward, module)
        replacement_count += 1
    return replacement_count


def install_mamba_inference_fusions(
    model: nn.Module,
    *,
    scratch_free_mamba3: bool | None = None,
    fused_mamba3_angle: bool | None = None,
) -> tuple[int, int]:
    """Install fusions after strict checkpoint loading; return legacy counts."""
    if model.training:
        raise ValueError("Mamba inference fusions require model.eval()")
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("Mamba inference fusions require frozen parameters")

    install_mamba3_preparation_fusions(
        model,
        scratch_free=scratch_free_mamba3,
        fused_angle=fused_mamba3_angle,
    )
    install_bidirectional_merge_fusion(model)

    norm_count = 0
    for stack_name in ("blocks", "blocks2", "blocks_bwd", "blocks2_bwd"):
        blocks = getattr(model, stack_name, None)
        if blocks is None:
            continue
        for block in blocks:
            if isinstance(block.norm, FusedInferenceLayerNorm):
                continue
            if not isinstance(block.norm, nn.LayerNorm):
                raise TypeError(
                    f"{stack_name} contains unsupported norm {type(block.norm)!r}"
                )
            block.norm = FusedInferenceLayerNorm(block.norm)
            norm_count += 1

    decoder = getattr(model, "decoder", None)
    if not isinstance(decoder, nn.Sequential):
        raise TypeError("fused Mamba inference requires a sequential decoder")
    activation_count = 0
    for index in range(len(decoder) - 1):
        convolution = decoder[index]
        activation = decoder[index + 1]
        if isinstance(activation, FusedInferenceBiasActivation):
            continue
        if not isinstance(convolution, nn.ConvTranspose2d):
            continue
        if isinstance(activation, nn.LeakyReLU):
            negative_slope = float(activation.negative_slope)
        elif isinstance(activation, nn.ReLU):
            negative_slope = 0.0
        else:
            continue
        # Safe in-place class specialization: no parameters or attributes move,
        # so original checkpoint keys such as decoder.0.bias remain unchanged.
        convolution.__class__ = BiaslessInferenceConvTranspose2d
        decoder[index + 1] = FusedInferenceBiasActivation(
            convolution,
            negative_slope=negative_slope,
        )
        activation_count += 1

    return norm_count, activation_count
