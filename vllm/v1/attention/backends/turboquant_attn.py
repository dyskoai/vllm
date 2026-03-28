# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TurboQuant attention backend for CUDA H100."""

from __future__ import annotations

from typing import ClassVar

import torch

from vllm import _custom_ops as ops
from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadataBuilder,
    AttentionType,
)
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionImpl,
    FlashAttentionMetadata,
    FlashAttentionMetadataBuilder,
)
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    TURBOQUANT_VALUE_GROUP_SIZE,
    turboquant_entry_size_bytes,
)

logger = init_logger(__name__)


def _unpack_2bit(packed: torch.Tensor, count: int) -> torch.Tensor:
    shifts = torch.tensor([0, 2, 4, 6], device=packed.device, dtype=torch.uint8)
    unpacked = ((packed.unsqueeze(-1) >> shifts) & 0x3).reshape(*packed.shape[:-1], -1)
    return unpacked[..., :count]


def _unpack_1bit(packed: torch.Tensor, count: int) -> torch.Tensor:
    shifts = torch.arange(8, device=packed.device, dtype=torch.uint8)
    unpacked = ((packed.unsqueeze(-1) >> shifts) & 0x1).reshape(*packed.shape[:-1], -1)
    return unpacked[..., :count]


def _dequantize_turboquant_cache(
    kv_cache: torch.Tensor,
    *,
    head_size: int,
    target_dtype: torch.dtype,
    codebook: torch.Tensor,
    value_group_size: int = TURBOQUANT_VALUE_GROUP_SIZE,
) -> torch.Tensor:
    k_lm_bytes = head_size // 4
    k_qjl_bytes = head_size // 8
    gamma_end = k_lm_bytes + k_qjl_bytes + 2
    v_packed_end = gamma_end + head_size // 4

    key_low = _unpack_2bit(kv_cache[..., :k_lm_bytes], head_size)
    key_high = _unpack_1bit(kv_cache[..., k_lm_bytes : k_lm_bytes + k_qjl_bytes], head_size)
    gamma_bytes = kv_cache[..., k_lm_bytes + k_qjl_bytes : gamma_end].to(torch.int32)
    gamma = (gamma_bytes[..., 0] | (gamma_bytes[..., 1] << 8)).to(torch.float32) / 1024.0
    key_levels = (key_low | (key_high << 2)).to(torch.float32)
    key = ((key_levels / 7.0) * 2.0 - 1.0) * gamma.unsqueeze(-1)

    value_idx = _unpack_2bit(kv_cache[..., gamma_end:v_packed_end], head_size).to(torch.long)
    num_groups = head_size // value_group_size
    scale_bytes = kv_cache[..., v_packed_end:].to(torch.int32).view(
        *kv_cache.shape[:-1], num_groups, 2
    )
    scales = (scale_bytes[..., 0] | (scale_bytes[..., 1] << 8)).to(torch.float32) / 1024.0
    value = codebook.to(torch.float32)[value_idx]
    value = value.view(*value.shape[:-1], num_groups, value_group_size)
    value = value * scales.unsqueeze(-1)
    value = value.reshape(*value.shape[:-2], head_size)

    return torch.stack(
        (key.to(target_dtype), value.to(target_dtype)),
        dim=0,
    )


class TurboQuantMetadataBuilder(FlashAttentionMetadataBuilder):
    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.kv_cache_dtype = vllm_config.model_config.dtype


class TurboQuantAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["turboquant_3_2"]
    forward_includes_kv_cache_update: bool = False

    @staticmethod
    def get_name() -> str:
        return "TURBOQUANT"

    @staticmethod
    def get_impl_cls() -> type["TurboQuantAttentionImpl"]:
        return TurboQuantAttentionImpl

    @staticmethod
    def get_builder_cls() -> type["AttentionMetadataBuilder"]:
        return TurboQuantMetadataBuilder

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int]:
        return [16]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [128]

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        del cache_dtype_str
        if block_size != 16:
            raise ValueError("TurboQuant requires block size 16.")
        return (
            num_blocks,
            block_size,
            num_kv_heads,
            turboquant_entry_size_bytes(head_size),
        )

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        if include_num_layers_dimension:
            return (0, 1, 2, 3, 4)
        return (0, 1, 2, 3)

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability >= DeviceCapability(9, 0)

    @classmethod
    def supports_combination(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: CacheDType | None,
        block_size: int | None,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        device_capability: DeviceCapability,
    ) -> str | None:
        del dtype, use_mla, has_sink, use_sparse, device_capability
        if kv_cache_dtype != "turboquant_3_2":
            return "TurboQuant backend requires kv_cache_dtype=turboquant_3_2"
        if head_size != 128:
            return "TurboQuant currently only supports head_size 128"
        if block_size not in (None, 16):
            return "TurboQuant currently only supports block_size 16"
        return None


class TurboQuantAttentionImpl(AttentionImpl[FlashAttentionMetadata]):
    can_return_lse_for_decode: bool = False

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        sinks: torch.Tensor | None = None,
    ) -> None:
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError("TurboQuant only supports decoder attention.")
        if sliding_window is not None:
            raise NotImplementedError("TurboQuant does not support sliding window attention.")
        if sinks is not None:
            raise NotImplementedError("TurboQuant does not support attention sinks.")
        if kv_cache_dtype != "turboquant_3_2":
            raise ValueError(
                f"TurboQuant backend requires kv_cache_dtype=turboquant_3_2, got {kv_cache_dtype}."
            )

        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.alibi_slopes = None if alibi_slopes is None else torch.tensor(alibi_slopes)
        self.kv_cache_dtype = kv_cache_dtype
        self.logits_soft_cap = 0 if logits_soft_cap is None else logits_soft_cap
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name
        self.attn_type = attn_type
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.supports_quant_query_input = False
        self._flash_fallback_impl = FlashAttentionImpl(
            num_heads=num_heads,
            head_size=head_size,
            scale=scale,
            num_kv_heads=num_kv_heads,
            alibi_slopes=alibi_slopes,
            sliding_window=sliding_window,
            kv_cache_dtype="auto",
            logits_soft_cap=logits_soft_cap,
            attn_type=attn_type,
            kv_sharing_target_layer_name=kv_sharing_target_layer_name,
            sinks=None,
        )
        logger.info_once(
            "Using TurboQuant reference backend for native packed KV cache.",
            scope="local",
        )

    def _build_flash_fallback_cache(
        self,
        layer: AttentionLayer,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        target_dtype: torch.dtype,
    ) -> torch.Tensor:
        full_kv_cache = _dequantize_turboquant_cache(
            kv_cache,
            head_size=self.head_size,
            target_dtype=target_dtype,
            codebook=layer._turboquant_codebook,
            value_group_size=layer._turboquant_value_group_size,
        )
        key_cache, value_cache = full_kv_cache.unbind(0)
        ops.reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            "auto",
            layer._k_scale,
            layer._v_scale,
        )
        return full_kv_cache

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert output is not None, "Output tensor must be provided."

        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "TurboQuant does not support fused output quantization."
            )

        if attn_metadata is None:
            return output.fill_(0)

        num_actual_tokens = attn_metadata.num_actual_tokens
        if (
            attn_metadata.max_query_len == 1
            and not attn_metadata.use_cascade
            and self._flash_fallback_impl.dcp_world_size == 1
        ):
            ops.turboquant_paged_attention(
                output[:num_actual_tokens],
                query[:num_actual_tokens],
                kv_cache,
                self.num_kv_heads,
                self.scale,
                attn_metadata.block_table,
                attn_metadata.seq_lens,
                kv_cache.shape[1],
                attn_metadata.max_seq_len,
                layer._turboquant_rotation,
                layer._turboquant_qjl_state,
                layer._turboquant_codebook,
            )
            return output

        full_kv_cache = self._build_flash_fallback_cache(
            layer,
            key,
            value,
            kv_cache,
            attn_metadata.slot_mapping,
            target_dtype=query.dtype,
        )
        return self._flash_fallback_impl.forward(
            layer,
            query,
            key,
            value,
            full_kv_cache,
            attn_metadata,
            output=output,
            output_scale=output_scale,
            output_block_scale=output_block_scale,
        )

    def do_kv_cache_update(
        self,
        layer: AttentionLayer,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        ops.reshape_and_cache_turboquant(
            key,
            value,
            kv_cache,
            slot_mapping,
            layer._turboquant_rotation,
            layer._turboquant_qjl_state,
            layer._turboquant_codebook,
            layer._turboquant_value_group_size,
        )
