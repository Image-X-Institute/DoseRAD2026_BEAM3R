"""Scratch-free Mamba3 SISO inference recurrence.

This is a fixed full-sequence inference specialization of the upstream
``mamba3_siso_fwd_kernel``.  The training kernel has a preprocessing phase
which writes head-expanded Q/K and scalar intermediates to global memory, then
reads them back during the recurrent phase.  Those stores are useful to the
backward implementation but are unnecessary for frozen inference.

Here each program prepares one chunk and consumes it immediately while Q/K,
scale and gamma remain on chip.  The mathematical operation and BF16 Q/K
rounding are deliberately retained so this can be selected after strict
checkpoint loading without changing the model architecture or state dict.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _cos_approx(x):
    return tl.inline_asm_elementwise(
        "cos.approx.f32 $0, $1;",
        constraints="=f,f",
        args=[x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _sin_approx(x):
    return tl.inline_asm_elementwise(
        "sin.approx.f32 $0, $1;",
        constraints="=f,f",
        args=[x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _tanh_approx(x):
    return tl.inline_asm_elementwise(
        "tanh.approx.f32 $0, $1;",
        constraints="=f,f",
        args=[x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _sigmoid(x):
    return tl.sigmoid(x)


@triton.jit
def _silu(x):
    return x * tl.sigmoid(x)


@triton.autotune(
    configs=[
        triton.Config({}, num_stages=stages, num_warps=warps, maxnreg=maxnreg)
        for stages in (1, 2, 3)
        for warps in (2, 4, 8)
        for maxnreg in (None, 128, 256)
    ],
    key=["CHUNK_SIZE", "HEADDIM_QK", "HEADDIM_V", "FUSE_ANGLE"],
)
@triton.jit
def _mamba3_siso_scratch_free_kernel(
    q,
    k,
    v,
    adt,
    dt,
    trap,
    q_bias,
    k_bias,
    angles,
    d,
    z,
    output,
    stride_q_batch,
    stride_q_seqlen,
    stride_q_head,
    stride_q_qkdim,
    stride_k_batch,
    stride_k_seqlen,
    stride_k_head,
    stride_k_qkdim,
    stride_v_batch,
    stride_v_seqlen,
    stride_v_head,
    stride_v_vdim,
    stride_adt_batch,
    stride_adt_head,
    stride_adt_seqlen,
    stride_dt_batch,
    stride_dt_head,
    stride_dt_seqlen,
    stride_trap_batch,
    stride_trap_head,
    stride_trap_seqlen,
    stride_q_bias_head,
    stride_q_bias_qkdim,
    stride_k_bias_head,
    stride_k_bias_qkdim,
    stride_angles_batch,
    stride_angles_seqlen,
    stride_angles_head,
    stride_angles_qkdim,
    stride_d_head,
    stride_z_batch,
    stride_z_seqlen,
    stride_z_head,
    stride_z_vdim,
    stride_output_batch,
    stride_output_seqlen,
    stride_output_head,
    stride_output_vdim,
    seqlen,
    nheads_qk,
    headdim_qk,
    headdim_v,
    headdim_angles,
    CHUNK_SIZE: tl.constexpr,
    HEADDIM_QK: tl.constexpr,
    HEADDIM_V: tl.constexpr,
    FUSE_ANGLE: tl.constexpr,
):
    pid_head = tl.program_id(0)
    pid_batch = tl.program_id(1)
    nheads = tl.num_programs(0)
    head_idx_qk = pid_head // (nheads // nheads_qk)
    num_chunks = tl.cdiv(seqlen, CHUNK_SIZE)

    q_ptr = q + pid_batch * stride_q_batch + head_idx_qk * stride_q_head
    k_ptr = k + pid_batch * stride_k_batch + head_idx_qk * stride_k_head
    v_ptr = v + pid_batch * stride_v_batch + pid_head * stride_v_head
    adt_ptr = adt + pid_batch * stride_adt_batch + pid_head * stride_adt_head
    dt_ptr = dt + pid_batch * stride_dt_batch + pid_head * stride_dt_head
    trap_ptr = trap + pid_batch * stride_trap_batch + pid_head * stride_trap_head
    q_bias_ptr = q_bias + pid_head * stride_q_bias_head
    k_bias_ptr = k_bias + pid_head * stride_k_bias_head
    angle_ptr = angles + pid_batch * stride_angles_batch
    if not FUSE_ANGLE:
        angle_ptr += pid_head * stride_angles_head
    z_ptr = z + pid_batch * stride_z_batch + pid_head * stride_z_head
    output_ptr = output + pid_batch * stride_output_batch + pid_head * stride_output_head

    q_desc = tl.make_tensor_descriptor(
        q_ptr,
        shape=[seqlen, headdim_qk],
        strides=[stride_q_seqlen, stride_q_qkdim],
        block_shape=[CHUNK_SIZE, HEADDIM_QK],
    )
    k_desc = tl.make_tensor_descriptor(
        k_ptr,
        shape=[seqlen, headdim_qk],
        strides=[stride_k_seqlen, stride_k_qkdim],
        block_shape=[CHUNK_SIZE, HEADDIM_QK],
    )
    v_desc = tl.make_tensor_descriptor(
        v_ptr,
        shape=[seqlen, headdim_v],
        strides=[stride_v_seqlen, stride_v_vdim],
        block_shape=[CHUNK_SIZE, HEADDIM_V],
    )
    z_desc = tl.make_tensor_descriptor(
        z_ptr,
        shape=[seqlen, headdim_v],
        strides=[stride_z_seqlen, stride_z_vdim],
        block_shape=[CHUNK_SIZE, HEADDIM_V],
    )
    output_desc = tl.make_tensor_descriptor(
        output_ptr,
        shape=[seqlen, headdim_v],
        strides=[stride_output_seqlen, stride_output_vdim],
        block_shape=[CHUNK_SIZE, HEADDIM_V],
    )

    q_bias_block = tl.load(
        q_bias_ptr + tl.arange(0, HEADDIM_QK) * stride_q_bias_qkdim,
        mask=tl.arange(0, HEADDIM_QK) < headdim_qk,
    )
    k_bias_block = tl.load(
        k_bias_ptr + tl.arange(0, HEADDIM_QK) * stride_k_bias_qkdim,
        mask=tl.arange(0, HEADDIM_QK) < headdim_qk,
    )
    d_value = tl.load(d + pid_head * stride_d_head).to(tl.float32)
    acc_ssm_states = tl.zeros([HEADDIM_V, HEADDIM_QK], dtype=tl.float32)
    if FUSE_ANGLE:
        angle_state = tl.zeros([HEADDIM_QK // 2], dtype=tl.float32)
        pi = 3.141592653589793
        two_pi = 2.0 * pi

    for chunk_idx in range(num_chunks):
        chunk_start = chunk_idx * CHUNK_SIZE
        offsets = chunk_start + tl.arange(0, CHUNK_SIZE)
        valid = offsets < seqlen

        # Load the shared Q/K inputs and make the same head-specific transforms
        # as the upstream phase-one kernel. They are cast to V's BF16 dtype
        # before the dot products, matching the former global scratch stores.
        q_pre = q_desc.load([chunk_start, 0])
        k_pre = k_desc.load([chunk_start, 0])
        v_block = v_desc.load([chunk_start, 0])
        z_block = z_desc.load([chunk_start, 0])
        q_pre += q_bias_block[None, :]
        k_pre += k_bias_block[None, :]

        dt_value = tl.load(
            dt_ptr + offsets * stride_dt_seqlen,
            mask=valid,
            other=0.0,
        ).to(tl.float32)
        dt_shifted = tl.load(
            dt_ptr + (offsets + 1) * stride_dt_seqlen,
            mask=(offsets + 1) < seqlen,
            other=0.0,
        ).to(tl.float32)
        trap_value = _sigmoid(
            tl.load(
                trap_ptr + offsets * stride_trap_seqlen,
                mask=valid,
                other=0.0,
            ).to(tl.float32)
        )
        trap_shifted = _sigmoid(
            tl.load(
                trap_ptr + (offsets + 1) * stride_trap_seqlen,
                mask=(offsets + 1) < seqlen,
                other=0.0,
            ).to(tl.float32)
        )
        gamma = dt_value * trap_value
        scale = dt_shifted * (1.0 - trap_shifted) + gamma

        qk_dot = tl.dot(
            q_pre * k_pre,
            tl.full([HEADDIM_QK, 1], 1, dtype=q_pre.dtype),
        ).to(q_pre.dtype)
        qk_dot = (qk_dot.reshape(CHUNK_SIZE) * gamma).to(tl.float32)

        angle_offsets = tl.arange(0, HEADDIM_QK // 2)
        angle_block = tl.load(
            angle_ptr
            + offsets[:, None] * stride_angles_seqlen
            + angle_offsets[None, :] * stride_angles_qkdim,
            mask=valid[:, None] & (angle_offsets[None, :] < headdim_angles),
            other=0.0,
        ).to(tl.float32)
        if FUSE_ANGLE:
            angle_delta = _tanh_approx(angle_block) * pi * dt_value[:, None]
            angle_block = tl.cumsum(angle_delta, axis=0) + angle_state[None, :]
            angle_block -= two_pi * tl.floor(angle_block / two_pi)
            angle_state += tl.sum(angle_delta, axis=0)
            angle_state -= two_pi * tl.floor(angle_state / two_pi)
        cosine = _cos_approx(angle_block)
        sine = _sin_approx(angle_block)

        k0, k1 = tl.split(
            tl.reshape(k_pre, [CHUNK_SIZE, HEADDIM_QK // 2, 2])
        )
        k_rotated = tl.reshape(
            tl.join(k0 * cosine - k1 * sine, k0 * sine + k1 * cosine),
            [CHUNK_SIZE, HEADDIM_QK],
        )
        q0, q1 = tl.split(
            tl.reshape(q_pre, [CHUNK_SIZE, HEADDIM_QK // 2, 2])
        )
        q_rotated = tl.reshape(
            tl.join(q0 * cosine - q1 * sine, q0 * sine + q1 * cosine),
            [CHUNK_SIZE, HEADDIM_QK],
        )
        # Invalid TMA rows would previously have been discarded by the
        # descriptor store and returned as zero during phase two.
        q_block = tl.where(valid[:, None], q_rotated, 0.0).to(v_block.dtype)
        k_block = tl.where(
            valid[:, None], k_rotated * scale[:, None], 0.0
        ).to(v_block.dtype)

        decay = tl.load(
            adt_ptr + offsets * stride_adt_seqlen,
            mask=valid,
            other=0.0,
        ).to(tl.float32)
        decay *= 1.44269504089
        decay_cumsum = tl.cumsum(decay)
        decay_sum = tl.sum(decay)

        output_accumulator = tl.dot(
            q_block, tl.trans(acc_ssm_states).to(q_block.dtype)
        )
        output_accumulator *= tl.math.exp2(decay_cumsum)[:, None]

        attention = tl.dot(q_block, tl.trans(k_block))
        attention *= tl.math.exp2(
            tl.minimum(
                decay_cumsum[:, None] - decay_cumsum[None, :], 0.0
            )
        )
        attention = tl.where(
            tl.arange(0, CHUNK_SIZE)[:, None]
            > tl.arange(0, CHUNK_SIZE)[None, :],
            attention,
            0.0,
        )
        output_accumulator += tl.dot(attention.to(v_block.dtype), v_block)
        output_accumulator += (d_value + qk_dot)[:, None] * v_block
        output_accumulator *= _silu(z_block.to(tl.float32))
        output_desc.store([chunk_start, 0], output_accumulator)

        reverse_decay = decay_sum - decay_cumsum
        scaled_v = v_block * tl.math.exp2(reverse_decay)[:, None]
        acc_ssm_states = (
            acc_ssm_states * tl.math.exp2(decay_sum)
            + tl.dot(tl.trans(scaled_v).to(k_block.dtype), k_block)
        )


def _alloc_tma_descriptor(
    size: int, alignment: int, stream: Optional[int]
) -> torch.Tensor:
    del alignment, stream
    return torch.empty(size, device="cuda", dtype=torch.int8)


# Matches the allocator installed by the upstream Mamba3 forward module, but
# keeps this local inference kernel independent of importing its training path.
triton.set_allocator(_alloc_tma_descriptor)


def mamba3_siso_forward_scratch_free(
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
    """Run the fixed full-sequence SISO recurrence without Q/K scratch."""
    batch, sequence, q_heads, q_dim = q.shape
    _, _, heads, value_dim = v.shape
    output = torch.empty(
        (batch, sequence, heads, value_dim),
        device=v.device,
        dtype=v.dtype,
    )
    _mamba3_siso_scratch_free_kernel[(heads, batch)](
        q,
        k,
        v,
        adt,
        dt,
        trap,
        q_bias,
        k_bias,
        angles,
        d,
        z,
        output,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        adt.stride(0), adt.stride(1), adt.stride(2),
        dt.stride(0), dt.stride(1), dt.stride(2),
        trap.stride(0), trap.stride(1), trap.stride(2),
        q_bias.stride(0), q_bias.stride(1),
        k_bias.stride(0), k_bias.stride(1),
        angles.stride(0),
        angles.stride(1),
        0 if fused_angle else angles.stride(2),
        angles.stride(2) if fused_angle else angles.stride(3),
        d.stride(0),
        z.stride(0), z.stride(1), z.stride(2), z.stride(3),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        output.stride(3),
        sequence,
        q_heads,
        q_dim,
        value_dim,
        angles.shape[-1],
        CHUNK_SIZE=chunk_size,
        HEADDIM_QK=triton.next_power_of_2(q_dim),
        HEADDIM_V=triton.next_power_of_2(value_dim),
        FUSE_ANGLE=bool(fused_angle),
    )
    return output
