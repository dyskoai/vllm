# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from functools import cache

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.turboquant_kv_cache import (
    TURBOQUANT_QJL_SCALE,
    _apply_mse_inverse_transform,
    _apply_qjl_inverse_transform,
    apply_turboquant_query_transforms,
    dequantize_turboquant_vectors,
    get_turboquant_layout,
)

PREFILL_QUERY_CHUNK_SIZE = 128
TURBOQUANT_DECODE_BLOCK_N = 32
TURBOQUANT_GROUP0_DIM = 32
TURBOQUANT_GROUP1_DIM = 96
TURBOQUANT_GROUP1_PADDED = 128


@cache
def _norm_lut(device_type: str, device_index: int | None) -> torch.Tensor:
    device = torch.device(device_type, device_index)
    values = torch.arange(1 << 16, dtype=torch.int32, device=device)
    return values.to(torch.int16).view(torch.float16).to(torch.float32)


def get_turboquant_norm_lut(device: torch.device) -> torch.Tensor:
    return _norm_lut(device.type, device.index)


@triton.jit
def _tanh(x):
    return 2 * tl.sigmoid(2 * x) - 1


@triton.jit
def _load_norm_from_lut(norm_lut_ptr, packed_ptr, token_base, offset):
    low = tl.load(packed_ptr + token_base + offset).to(tl.int32)
    high = tl.load(packed_ptr + token_base + offset + 1).to(tl.int32)
    idx = low | (high << 8)
    return tl.load(norm_lut_ptr + idx)


@triton.jit
def _turboquant_decode_q1_kernel(
    q_rot0_ptr,
    q_qjl0_ptr,
    q_rot1_ptr,
    q_qjl1_ptr,
    key_cache_ptr,
    value_cache_ptr,
    block_table_ptr,
    seq_lens_ptr,
    kv_head_for_query_head_ptr,
    centroids2_ptr,
    centroids1_ptr,
    norm_lut_ptr,
    out_g0_mse_ptr,
    out_g0_qjl_ptr,
    out_g1_mse_ptr,
    out_g1_qjl_ptr,
    q_rot0_stride_0,
    q_rot0_stride_1,
    q_qjl0_stride_0,
    q_qjl0_stride_1,
    q_rot1_stride_0,
    q_rot1_stride_1,
    q_qjl1_stride_0,
    q_qjl1_stride_1,
    k_stride_0,
    k_stride_1,
    k_stride_2,
    k_stride_3,
    v_stride_0,
    v_stride_1,
    v_stride_2,
    v_stride_3,
    block_table_stride,
    out_g0_mse_stride_0,
    out_g0_mse_stride_1,
    out_g0_qjl_stride_0,
    out_g0_qjl_stride_1,
    out_g1_mse_stride_0,
    out_g1_mse_stride_1,
    out_g1_qjl_stride_0,
    out_g1_qjl_stride_1,
    softmax_scale,
    logits_soft_cap,
    qjl_scale0,
    qjl_scale1,
    block_size: tl.constexpr,
    num_heads: tl.constexpr,
    block_n: tl.constexpr,
    group0_dim: tl.constexpr,
    group1_dim: tl.constexpr,
    group1_padded: tl.constexpr,
):
    seq_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    kv_head_idx = tl.load(kv_head_for_query_head_ptr + head_idx)
    seq_len = tl.load(seq_lens_ptr + seq_idx)

    offs_n = tl.arange(0, block_n)
    offs_d0 = tl.arange(0, group0_dim)
    offs_d1 = tl.arange(0, group1_padded)
    mask_d1 = offs_d1 < group1_dim

    q_rot0 = tl.load(
        q_rot0_ptr
        + seq_idx * q_rot0_stride_0
        + head_idx * q_rot0_stride_1
        + offs_d0,
    ).to(tl.float32)
    q_qjl0 = tl.load(
        q_qjl0_ptr
        + seq_idx * q_qjl0_stride_0
        + head_idx * q_qjl0_stride_1
        + offs_d0,
    ).to(tl.float32)
    q_rot1 = tl.load(
        q_rot1_ptr
        + seq_idx * q_rot1_stride_0
        + head_idx * q_rot1_stride_1
        + offs_d1,
        mask=mask_d1,
        other=0.0,
    ).to(tl.float32)
    q_qjl1 = tl.load(
        q_qjl1_ptr
        + seq_idx * q_qjl1_stride_0
        + head_idx * q_qjl1_stride_1
        + offs_d1,
        mask=mask_d1,
        other=0.0,
    ).to(tl.float32)

    m_i = -float("inf")
    l_i = 0.0
    acc_g0_mse = tl.zeros([group0_dim], dtype=tl.float32)
    acc_g0_qjl = tl.zeros([group0_dim], dtype=tl.float32)
    acc_g1_mse = tl.zeros([group1_padded], dtype=tl.float32)
    acc_g1_qjl = tl.zeros([group1_padded], dtype=tl.float32)

    for start_n in range(0, seq_len, block_n):
        tok_idx = start_n + offs_n
        tok_mask = tok_idx < seq_len
        block_ids = tl.load(
            block_table_ptr + seq_idx * block_table_stride + tok_idx // block_size,
            mask=tok_mask,
            other=0,
        ).to(tl.int64)
        tok_in_block = tok_idx % block_size

        key_base = (
            block_ids * k_stride_0 + tok_in_block * k_stride_1 + kv_head_idx * k_stride_2
        )

        g0_mse_bytes = tl.load(
            key_cache_ptr + key_base[:, None] + (offs_d0 // 4)[None, :],
            mask=tok_mask[:, None],
            other=0,
        ).to(tl.int32)
        g0_mse_idx = (g0_mse_bytes >> (((offs_d0 % 4) * 2)[None, :])) & 0x3
        g0_centroids = tl.load(centroids2_ptr + g0_mse_idx, mask=tok_mask[:, None], other=0.0)
        g0_qjl_bytes = tl.load(
            key_cache_ptr + key_base[:, None] + (8 + offs_d0 // 8)[None, :],
            mask=tok_mask[:, None],
            other=0,
        ).to(tl.int32)
        g0_qjl_signs = (
            (((g0_qjl_bytes >> ((offs_d0 % 8)[None, :])) & 0x1).to(tl.float32) * 2.0)
            - 1.0
        )
        g0_vec_norm = _load_norm_from_lut(norm_lut_ptr, key_cache_ptr, key_base, 12)
        g0_res_norm = _load_norm_from_lut(norm_lut_ptr, key_cache_ptr, key_base, 14)

        g1_mse_bytes = tl.load(
            key_cache_ptr + key_base[:, None] + (16 + offs_d1 // 8)[None, :],
            mask=tok_mask[:, None] & mask_d1[None, :],
            other=0,
        ).to(tl.int32)
        g1_mse_idx = (g1_mse_bytes >> ((offs_d1 % 8)[None, :])) & 0x1
        g1_centroids = tl.load(
            centroids1_ptr + g1_mse_idx,
            mask=tok_mask[:, None] & mask_d1[None, :],
            other=0.0,
        )
        g1_qjl_bytes = tl.load(
            key_cache_ptr + key_base[:, None] + (28 + offs_d1 // 8)[None, :],
            mask=tok_mask[:, None] & mask_d1[None, :],
            other=0,
        ).to(tl.int32)
        g1_qjl_signs = (
            (((g1_qjl_bytes >> ((offs_d1 % 8)[None, :])) & 0x1).to(tl.float32) * 2.0)
            - 1.0
        )
        g1_vec_norm = _load_norm_from_lut(norm_lut_ptr, key_cache_ptr, key_base, 40)
        g1_res_norm = _load_norm_from_lut(norm_lut_ptr, key_cache_ptr, key_base, 42)

        score_g0 = g0_vec_norm * (
            tl.sum(g0_centroids * q_rot0[None, :], axis=1)
            + g0_res_norm * qjl_scale0 * tl.sum(g0_qjl_signs * q_qjl0[None, :], axis=1)
        )
        score_g1 = g1_vec_norm * (
            tl.sum(g1_centroids * q_rot1[None, :], axis=1)
            + g1_res_norm * qjl_scale1 * tl.sum(g1_qjl_signs * q_qjl1[None, :], axis=1)
        )

        if logits_soft_cap > 0:
            scores = logits_soft_cap * _tanh(
                ((score_g0 + score_g1) * softmax_scale) / logits_soft_cap
            )
        else:
            scores = (score_g0 + score_g1) * softmax_scale
        scores = tl.where(tok_mask, scores, float("-inf"))

        m_ij = tl.maximum(m_i, tl.max(scores, axis=0))
        alpha = tl.exp(m_i - m_ij)
        probs = tl.exp(scores - m_ij)
        probs = tl.where(tok_mask, probs, 0.0)

        acc_g0_mse *= alpha
        acc_g0_qjl *= alpha
        acc_g1_mse *= alpha
        acc_g1_qjl *= alpha

        value_base = (
            block_ids * v_stride_0 + tok_in_block * v_stride_1 + kv_head_idx * v_stride_2
        )

        vg0_mse_bytes = tl.load(
            value_cache_ptr + value_base[:, None] + (offs_d0 // 4)[None, :],
            mask=tok_mask[:, None],
            other=0,
        ).to(tl.int32)
        vg0_mse_idx = (vg0_mse_bytes >> (((offs_d0 % 4) * 2)[None, :])) & 0x3
        vg0_centroids = tl.load(
            centroids2_ptr + vg0_mse_idx,
            mask=tok_mask[:, None],
            other=0.0,
        )
        vg0_qjl_bytes = tl.load(
            value_cache_ptr + value_base[:, None] + (8 + offs_d0 // 8)[None, :],
            mask=tok_mask[:, None],
            other=0,
        ).to(tl.int32)
        vg0_qjl_signs = (
            (((vg0_qjl_bytes >> ((offs_d0 % 8)[None, :])) & 0x1).to(tl.float32) * 2.0)
            - 1.0
        )
        vg0_vec_norm = _load_norm_from_lut(norm_lut_ptr, value_cache_ptr, value_base, 12)
        vg0_res_norm = _load_norm_from_lut(norm_lut_ptr, value_cache_ptr, value_base, 14)

        vg1_mse_bytes = tl.load(
            value_cache_ptr + value_base[:, None] + (16 + offs_d1 // 8)[None, :],
            mask=tok_mask[:, None] & mask_d1[None, :],
            other=0,
        ).to(tl.int32)
        vg1_mse_idx = (vg1_mse_bytes >> ((offs_d1 % 8)[None, :])) & 0x1
        vg1_centroids = tl.load(
            centroids1_ptr + vg1_mse_idx,
            mask=tok_mask[:, None] & mask_d1[None, :],
            other=0.0,
        )
        vg1_qjl_bytes = tl.load(
            value_cache_ptr + value_base[:, None] + (28 + offs_d1 // 8)[None, :],
            mask=tok_mask[:, None] & mask_d1[None, :],
            other=0,
        ).to(tl.int32)
        vg1_qjl_signs = (
            (((vg1_qjl_bytes >> ((offs_d1 % 8)[None, :])) & 0x1).to(tl.float32) * 2.0)
            - 1.0
        )
        vg1_vec_norm = _load_norm_from_lut(norm_lut_ptr, value_cache_ptr, value_base, 40)
        vg1_res_norm = _load_norm_from_lut(norm_lut_ptr, value_cache_ptr, value_base, 42)

        weight_g0 = probs * vg0_vec_norm
        weight_g1 = probs * vg1_vec_norm
        res_weight_g0 = weight_g0 * vg0_res_norm
        res_weight_g1 = weight_g1 * vg1_res_norm

        acc_g0_mse += tl.sum(weight_g0[:, None] * vg0_centroids, axis=0)
        acc_g0_qjl += tl.sum(res_weight_g0[:, None] * vg0_qjl_signs, axis=0)
        acc_g1_mse += tl.sum(
            weight_g1[:, None] * vg1_centroids,
            axis=0,
        )
        acc_g1_qjl += tl.sum(
            res_weight_g1[:, None] * vg1_qjl_signs,
            axis=0,
        )

        l_i = l_i * alpha + tl.sum(probs, axis=0)
        m_i = m_ij

    out_g0_mse_ptrs = (
        out_g0_mse_ptr + seq_idx * out_g0_mse_stride_0 + head_idx * out_g0_mse_stride_1 + offs_d0
    )
    out_g0_qjl_ptrs = (
        out_g0_qjl_ptr + seq_idx * out_g0_qjl_stride_0 + head_idx * out_g0_qjl_stride_1 + offs_d0
    )
    out_g1_mse_ptrs = (
        out_g1_mse_ptr + seq_idx * out_g1_mse_stride_0 + head_idx * out_g1_mse_stride_1 + offs_d1
    )
    out_g1_qjl_ptrs = (
        out_g1_qjl_ptr + seq_idx * out_g1_qjl_stride_0 + head_idx * out_g1_qjl_stride_1 + offs_d1
    )

    tl.store(out_g0_mse_ptrs, acc_g0_mse / l_i)
    tl.store(out_g0_qjl_ptrs, acc_g0_qjl / l_i)
    tl.store(out_g1_mse_ptrs, acc_g1_mse / l_i, mask=mask_d1)
    tl.store(out_g1_qjl_ptrs, acc_g1_qjl / l_i, mask=mask_d1)


def _turboquant_decode_q1_fused(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    key_group_indices: tuple[torch.Tensor, torch.Tensor],
    value_group_indices: tuple[torch.Tensor, torch.Tensor],
    key_rotations: tuple[torch.Tensor, torch.Tensor],
    key_qjl_matrices: tuple[torch.Tensor, torch.Tensor],
    value_rotations: tuple[torch.Tensor, torch.Tensor],
    value_qjl_matrices: tuple[torch.Tensor, torch.Tensor],
    centroids: dict[int, torch.Tensor],
    norm_lut: torch.Tensor,
    softmax_scale: float,
    kv_head_for_query_head: torch.Tensor,
    key_query_group_indices: tuple[torch.Tensor, torch.Tensor] | None,
    value_query_group_indices: tuple[torch.Tensor, torch.Tensor] | None,
    logits_soft_cap: float,
    out: torch.Tensor | None,
) -> torch.Tensor:
    (q_rot_groups, q_qjl_groups) = apply_turboquant_query_transforms(
        query,
        key_group_indices,
        key_rotations,
        key_qjl_matrices,
        kv_head_for_query_head=kv_head_for_query_head,
        per_query_group_indices=key_query_group_indices,
    )

    q_rot0 = q_rot_groups[0].contiguous()
    q_qjl0 = q_qjl_groups[0].contiguous()
    q_rot1 = q_rot_groups[1].contiguous()
    q_qjl1 = q_qjl_groups[1].contiguous()
    seq_lens = seq_lens.to(device=query.device, dtype=torch.int32, non_blocking=True)
    block_table = block_table.to(device=query.device, non_blocking=True)
    kv_head_for_query_head = kv_head_for_query_head.to(
        device=query.device, dtype=torch.int64, non_blocking=True
    )

    num_tokens, num_heads, head_size = query.shape
    assert head_size == 128

    out_g0_mse = torch.empty(
        (num_tokens, num_heads, TURBOQUANT_GROUP0_DIM),
        dtype=torch.float32,
        device=query.device,
    )
    out_g0_qjl = torch.empty_like(out_g0_mse)
    out_g1_mse = torch.empty(
        (num_tokens, num_heads, TURBOQUANT_GROUP1_DIM),
        dtype=torch.float32,
        device=query.device,
    )
    out_g1_qjl = torch.empty_like(out_g1_mse)

    centroids2 = centroids[2].contiguous()
    centroids1 = centroids[1].contiguous()
    qjl_scale0 = TURBOQUANT_QJL_SCALE / TURBOQUANT_GROUP0_DIM
    qjl_scale1 = TURBOQUANT_QJL_SCALE / TURBOQUANT_GROUP1_DIM

    grid = (num_tokens, num_heads)
    _turboquant_decode_q1_kernel[grid](
        q_rot0,
        q_qjl0,
        q_rot1,
        q_qjl1,
        key_cache,
        value_cache,
        block_table,
        seq_lens,
        kv_head_for_query_head,
        centroids2,
        centroids1,
        norm_lut,
        out_g0_mse,
        out_g0_qjl,
        out_g1_mse,
        out_g1_qjl,
        q_rot0.stride(0),
        q_rot0.stride(1),
        q_qjl0.stride(0),
        q_qjl0.stride(1),
        q_rot1.stride(0),
        q_rot1.stride(1),
        q_qjl1.stride(0),
        q_qjl1.stride(1),
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        key_cache.stride(3),
        value_cache.stride(0),
        value_cache.stride(1),
        value_cache.stride(2),
        value_cache.stride(3),
        block_table.stride(0),
        out_g0_mse.stride(0),
        out_g0_mse.stride(1),
        out_g0_qjl.stride(0),
        out_g0_qjl.stride(1),
        out_g1_mse.stride(0),
        out_g1_mse.stride(1),
        out_g1_qjl.stride(0),
        out_g1_qjl.stride(1),
        softmax_scale,
        logits_soft_cap,
        qjl_scale0,
        qjl_scale1,
        block_size=key_cache.shape[1],
        num_heads=num_heads,
        block_n=TURBOQUANT_DECODE_BLOCK_N,
        group0_dim=TURBOQUANT_GROUP0_DIM,
        group1_dim=TURBOQUANT_GROUP1_DIM,
        group1_padded=TURBOQUANT_GROUP1_PADDED,
    )

    group0 = _apply_mse_inverse_transform(out_g0_mse, value_rotations[0]) + (
        _apply_qjl_inverse_transform(out_g0_qjl, value_qjl_matrices[0])
        * (TURBOQUANT_QJL_SCALE / TURBOQUANT_GROUP0_DIM)
    )
    group1 = _apply_mse_inverse_transform(out_g1_mse, value_rotations[1]) + (
        _apply_qjl_inverse_transform(out_g1_qjl, value_qjl_matrices[1])
        * (TURBOQUANT_QJL_SCALE / TURBOQUANT_GROUP1_DIM)
    )

    if value_query_group_indices is None:
        value_query_group_indices = tuple(
            group.index_select(0, kv_head_for_query_head) for group in value_group_indices
        )

    output = torch.empty_like(query) if out is None else out
    output.zero_()
    output_fp32 = output.to(torch.float32)
    output_fp32.scatter_add_(
        -1,
        value_query_group_indices[0].unsqueeze(0).expand(num_tokens, -1, -1),
        group0,
    )
    output_fp32.scatter_add_(
        -1,
        value_query_group_indices[1].unsqueeze(0).expand(num_tokens, -1, -1),
        group1,
    )
    output.copy_(output_fp32.to(output.dtype))
    return output


def _use_fused_decode_q1_path(
    query: torch.Tensor,
    kv_cache_dtype: str,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    causal: bool,
    sliding_window: tuple[int, int],
    sinks: torch.Tensor | None,
    mm_prefix_range: torch.Tensor | None,
) -> bool:
    if query.device.type != "cuda":
        return False
    if kv_cache_dtype != "turboquant_3_2":
        return False
    if query.shape[-1] != 128:
        return False
    if not causal:
        return False
    if sliding_window != (-1, -1):
        return False
    if sinks is not None or mm_prefix_range is not None:
        return False
    if query.shape[0] != seq_lens.numel():
        return False
    query_lens = query_start_loc[1:] - query_start_loc[:-1]
    return bool(torch.all(query_lens == 1).item())


def _turboquant_decode_attention_fallback(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    key_group_indices: tuple[torch.Tensor, torch.Tensor],
    value_group_indices: tuple[torch.Tensor, torch.Tensor],
    key_rotations: tuple[torch.Tensor, torch.Tensor],
    key_qjl_matrices: tuple[torch.Tensor, torch.Tensor],
    value_rotations: tuple[torch.Tensor, torch.Tensor],
    value_qjl_matrices: tuple[torch.Tensor, torch.Tensor],
    centroids: dict[int, torch.Tensor],
    softmax_scale: float,
    kv_cache_dtype: str,
    kv_head_for_query_head: torch.Tensor | None = None,
    causal: bool = True,
    sliding_window: tuple[int, int] = (-1, -1),
    sinks: torch.Tensor | None = None,
    logits_soft_cap: float = 0.0,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    layout = get_turboquant_layout(kv_cache_dtype, query.shape[-1])
    if kv_head_for_query_head is None:
        kv_group_num = query.shape[1] // key_cache.shape[2]
        kv_head_for_query_head = (
            torch.arange(query.shape[1], device=query.device, dtype=torch.int64)
            // kv_group_num
        )

    output = torch.empty_like(query) if out is None else out
    output.zero_()
    query_lens = query_start_loc[1:] - query_start_loc[:-1]

    for seq_idx, seq_len in enumerate(seq_lens.tolist()):
        q_start = int(query_start_loc[seq_idx].item())
        q_len = int(query_lens[seq_idx].item())
        q_end = q_start + q_len
        num_blocks = (seq_len + key_cache.shape[1] - 1) // key_cache.shape[1]
        block_ids = block_table[seq_idx, :num_blocks].to(torch.int64)
        seq_key_cache = key_cache.index_select(0, block_ids).reshape(
            num_blocks * key_cache.shape[1], key_cache.shape[2], layout.packed_dim
        )[:seq_len]
        seq_value_cache = value_cache.index_select(0, block_ids).reshape(
            num_blocks * value_cache.shape[1], value_cache.shape[2], layout.packed_dim
        )[:seq_len]

        seq_key = dequantize_turboquant_vectors(
            seq_key_cache,
            kv_cache_dtype,
            query.shape[-1],
            key_rotations,
            key_qjl_matrices,
            centroids,
            key_group_indices,
            query.dtype,
        )
        seq_value = dequantize_turboquant_vectors(
            seq_value_cache,
            kv_cache_dtype,
            query.shape[-1],
            value_rotations,
            value_qjl_matrices,
            centroids,
            value_group_indices,
            query.dtype,
        )

        seq_query = query[q_start:q_end]
        context_len = max(seq_len - q_len, 0) if causal else 0
        key_positions = torch.arange(seq_len, device=query.device, dtype=torch.int32)
        if causal:
            query_positions = torch.arange(
                context_len,
                context_len + q_len,
                device=query.device,
                dtype=torch.int32,
            )
            allowed = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
            left_window, right_window = sliding_window
            if left_window != -1:
                allowed &= key_positions.unsqueeze(0) >= (
                    query_positions.unsqueeze(1) - left_window
                )
            if right_window != -1:
                allowed &= key_positions.unsqueeze(0) <= (
                    query_positions.unsqueeze(1) + right_window
                )
        else:
            allowed = torch.ones((q_len, seq_len), dtype=torch.bool, device=query.device)

        q_states = seq_query.permute(1, 0, 2).to(torch.float32)
        k_states = (
            seq_key.permute(1, 0, 2).index_select(0, kv_head_for_query_head).to(torch.float32)
        )
        v_states = (
            seq_value.permute(1, 0, 2).index_select(0, kv_head_for_query_head).to(torch.float32)
        )
        if sinks is not None:
            sink_logits = sinks[:, None, None].to(torch.float32)
            zero_value = torch.zeros(
                (query.shape[1], 1, query.shape[2]),
                dtype=torch.float32,
                device=query.device,
            )
            v_states = torch.cat((v_states, zero_value), dim=1)

        output_chunks: list[torch.Tensor] = []
        for chunk_start in range(0, q_len, PREFILL_QUERY_CHUNK_SIZE):
            chunk_end = min(chunk_start + PREFILL_QUERY_CHUNK_SIZE, q_len)
            q_chunk = q_states[:, chunk_start:chunk_end, :]
            allowed_chunk = allowed[chunk_start:chunk_end]

            logits = torch.einsum("hqd,hkd->hqk", q_chunk, k_states) * softmax_scale
            if logits_soft_cap > 0:
                logits = logits_soft_cap * torch.tanh(logits / logits_soft_cap)
            logits = logits.masked_fill(~allowed_chunk.unsqueeze(0), float("-inf"))
            if sinks is not None:
                sink_chunk = sink_logits.expand(-1, chunk_end - chunk_start, 1)
                logits = torch.cat((logits, sink_chunk), dim=-1)
            attn = torch.softmax(logits, dim=-1)
            output_chunks.append(torch.einsum("hqk,hkd->hqd", attn, v_states))

        seq_output = torch.cat(output_chunks, dim=1)
        output[q_start:q_end].copy_(seq_output.permute(1, 0, 2).to(output.dtype))

    return output


@torch.inference_mode()
def turboquant_decode_attention_fwd(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    key_group_indices: tuple[torch.Tensor, torch.Tensor],
    value_group_indices: tuple[torch.Tensor, torch.Tensor],
    key_rotations: tuple[torch.Tensor, torch.Tensor],
    key_qjl_matrices: tuple[torch.Tensor, torch.Tensor],
    value_rotations: tuple[torch.Tensor, torch.Tensor],
    value_qjl_matrices: tuple[torch.Tensor, torch.Tensor],
    centroids: dict[int, torch.Tensor],
    norm_lut: torch.Tensor,
    softmax_scale: float,
    kv_cache_dtype: str,
    token_seq_ids: torch.Tensor | None = None,
    token_kv_lens: torch.Tensor | None = None,
    token_query_positions: torch.Tensor | None = None,
    kv_head_for_query_head: torch.Tensor | None = None,
    key_query_group_indices: tuple[torch.Tensor, torch.Tensor] | None = None,
    value_query_group_indices: tuple[torch.Tensor, torch.Tensor] | None = None,
    value_mse_inverse_matrices: tuple[torch.Tensor, torch.Tensor] | None = None,
    value_qjl_inverse_matrices: tuple[torch.Tensor, torch.Tensor] | None = None,
    causal: bool = True,
    sliding_window: tuple[int, int] = (-1, -1),
    sinks: torch.Tensor | None = None,
    mm_prefix_range: torch.Tensor | None = None,
    logits_soft_cap: float = 0.0,
    output_lse: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    del token_seq_ids
    del token_kv_lens
    del token_query_positions
    del value_mse_inverse_matrices
    del value_qjl_inverse_matrices
    del output_lse

    if query.ndim != 3:
        raise ValueError(f"Expected query shape [T, H, D], got {query.shape}")

    if kv_head_for_query_head is None:
        kv_group_num = query.shape[1] // key_cache.shape[2]
        kv_head_for_query_head = (
            torch.arange(query.shape[1], device=query.device, dtype=torch.int64)
            // kv_group_num
        )

    if _use_fused_decode_q1_path(
        query,
        kv_cache_dtype,
        query_start_loc,
        seq_lens,
        causal,
        sliding_window,
        sinks,
        mm_prefix_range,
    ):
        return _turboquant_decode_q1_fused(
            query,
            key_cache,
            value_cache,
            block_table,
            seq_lens,
            key_group_indices,
            value_group_indices,
            key_rotations,
            key_qjl_matrices,
            value_rotations,
            value_qjl_matrices,
            centroids,
            norm_lut,
            softmax_scale,
            kv_head_for_query_head,
            key_query_group_indices,
            value_query_group_indices,
            logits_soft_cap,
            out,
        )

    return _turboquant_decode_attention_fallback(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        block_table=block_table,
        query_start_loc=query_start_loc,
        seq_lens=seq_lens,
        key_group_indices=key_group_indices,
        value_group_indices=value_group_indices,
        key_rotations=key_rotations,
        key_qjl_matrices=key_qjl_matrices,
        value_rotations=value_rotations,
        value_qjl_matrices=value_qjl_matrices,
        centroids=centroids,
        softmax_scale=softmax_scale,
        kv_cache_dtype=kv_cache_dtype,
        kv_head_for_query_head=kv_head_for_query_head,
        causal=causal,
        sliding_window=sliding_window,
        sinks=sinks,
        logits_soft_cap=logits_soft_cap,
        out=out,
    )
