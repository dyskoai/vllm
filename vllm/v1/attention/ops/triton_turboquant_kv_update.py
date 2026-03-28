# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import torch

from vllm.v1.attention.ops.turboquant_kv_cache import (
    TurboQuantLayout,
    quantize_turboquant_vectors,
)


def turboquant_write_packed_kv(
    x: torch.Tensor,
    cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    layout: TurboQuantLayout,
    group_indices: tuple[torch.Tensor, torch.Tensor],
    mse_transform_matrices: tuple[torch.Tensor, torch.Tensor],
    qjl_transform_matrices: tuple[torch.Tensor, torch.Tensor],
    mse_to_qjl_matrices: tuple[torch.Tensor, torch.Tensor],
    centroids: dict[int, torch.Tensor],
) -> None:
    del mse_to_qjl_matrices

    if x.numel() == 0:
        return
    if cache.dtype != torch.uint8:
        raise ValueError("TurboQuant KV cache update expects uint8 cache storage.")
    if x.ndim != 3:
        raise ValueError(f"Expected input shape [T, H, D], got {x.shape}")
    if cache.ndim != 4:
        raise ValueError(
            "Expected cache shape [num_blocks, block_size, num_kv_heads, packed_dim], "
            f"got {cache.shape}"
        )
    if cache.shape[2] != x.shape[1]:
        raise ValueError("TurboQuant cache head count does not match the input.")
    if cache.shape[3] != layout.packed_dim:
        raise ValueError("TurboQuant cache packed_dim does not match the layout.")
    if slot_mapping.ndim != 1 or slot_mapping.shape[0] != x.shape[0]:
        raise ValueError("slot_mapping must be a 1D tensor aligned with input tokens.")

    valid = slot_mapping >= 0
    if not valid.any():
        return

    packed = quantize_turboquant_vectors(
        x[valid],
        "turboquant_3_2",
        mse_transform_matrices,
        qjl_transform_matrices,
        centroids,
        group_indices,
    )
    slots = slot_mapping[valid].to(torch.int64)
    block_size = cache.shape[1]
    block_idx = torch.div(slots, block_size, rounding_mode="floor")
    block_offset = slots % block_size
    cache[block_idx, block_offset] = packed
