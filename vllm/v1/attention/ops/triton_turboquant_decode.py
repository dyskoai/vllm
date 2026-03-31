# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from functools import cache

import torch

from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.ops.turboquant_kv_cache import (
    dequantize_turboquant_vectors,
    get_turboquant_layout,
)

PREFILL_QUERY_CHUNK_SIZE = 128


@cache
def _norm_lut(device_type: str, device_index: int | None) -> torch.Tensor:
    device = torch.device(device_type, device_index)
    values = torch.arange(1 << 16, dtype=torch.int32, device=device)
    return values.to(torch.int16).view(torch.float16).to(torch.float32)


def get_turboquant_norm_lut(device: torch.device) -> torch.Tensor:
    return _norm_lut(device.type, device.index)


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
    del norm_lut
    del token_seq_ids
    del token_kv_lens
    del token_query_positions
    del key_query_group_indices
    del value_query_group_indices
    del value_mse_inverse_matrices
    del value_qjl_inverse_matrices
    del output_lse
    del mm_prefix_range

    if query.ndim != 3:
        raise ValueError(f"Expected query shape [T, H, D], got {query.shape}")

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
            allowed = torch.ones(
                (q_len, seq_len), dtype=torch.bool, device=query.device
            )

        q_states = seq_query.permute(1, 0, 2).to(torch.float32)
        k_states = (
            seq_key.permute(1, 0, 2)
            .index_select(0, kv_head_for_query_head)
            .to(torch.float32)
        )
        v_states = (
            seq_value.permute(1, 0, 2)
            .index_select(0, kv_head_for_query_head)
            .to(torch.float32)
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
            logits = logits.masked_fill(~allowed_chunk.unsqueeze(0), float("-inf"))
            if logits_soft_cap > 0:
                logits = logits_soft_cap * torch.tanh(logits / logits_soft_cap)
            if sinks is not None:
                sink_chunk = sink_logits.expand(-1, chunk_end - chunk_start, 1)
                logits = torch.cat((logits, sink_chunk), dim=-1)
            attn = torch.softmax(logits, dim=-1)
            output_chunks.append(torch.einsum("hqk,hkd->hqd", attn, v_states))

        seq_output = torch.cat(output_chunks, dim=1)
        output[q_start:q_end].copy_(seq_output.permute(1, 0, 2).to(output.dtype))

    return output
