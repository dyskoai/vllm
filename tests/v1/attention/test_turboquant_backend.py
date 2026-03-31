# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import patch

import pytest
import torch

from vllm.config import CacheConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.attention import Attention
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backends.triton_attn import TritonAttentionBackend
from vllm.v1.attention.ops.turboquant_kv_cache import get_turboquant_packed_dim
from vllm.v1.attention.selector import _cached_get_attn_backend, get_attn_backend
from vllm.v1.kv_cache_interface import TurboQuantAttentionSpec

try:
    from vllm.platforms.cuda import CudaPlatform
except (ImportError, ModuleNotFoundError):
    CudaPlatform = None

from vllm.platforms.cpu import CpuPlatform


pytestmark = pytest.mark.skip_global_cleanup


@pytest.fixture(autouse=True)
def clear_backend_cache():
    _cached_get_attn_backend.cache_clear()


def test_cache_config_accepts_turboquant_dtype():
    cache_config = CacheConfig(
        cache_dtype="turboquant_3_2",
        block_size=16,
        enable_turboquant=True,
    )
    assert cache_config.cache_dtype == "turboquant_3_2"
    assert cache_config.enable_turboquant is True


def test_turboquant_attention_spec_page_size_bytes():
    spec = TurboQuantAttentionSpec(
        block_size=16,
        num_kv_heads=8,
        head_size=128,
        dtype=torch.uint8,
        cache_dtype_str="turboquant_3_2",
    )

    packed_dim = get_turboquant_packed_dim(128, "turboquant_3_2")
    assert spec.page_size_bytes == 2 * 16 * 8 * packed_dim
    assert spec.real_page_size_bytes == spec.page_size_bytes


def test_triton_backend_shape_and_validation():
    packed_dim = get_turboquant_packed_dim(128, "turboquant_3_2")
    assert TritonAttentionBackend.get_kv_cache_shape(
        5, 16, 4, 128, cache_dtype_str="turboquant_3_2"
    ) == (5, 2, 16, 4, packed_dim)

    valid = TritonAttentionBackend.validate_configuration(
        head_size=128,
        dtype=torch.float16,
        kv_cache_dtype="turboquant_3_2",
        block_size=16,
        use_mla=False,
        has_sink=False,
        use_sparse=False,
        use_mm_prefix=False,
        use_per_head_quant_scales=False,
        device_capability=DeviceCapability(9, 0),
        attn_type="decoder",
    )
    assert valid == []

    invalid_head_size = TritonAttentionBackend.validate_configuration(
        head_size=64,
        dtype=torch.float16,
        kv_cache_dtype="turboquant_3_2",
        block_size=32,
        use_mla=False,
        has_sink=False,
        use_sparse=False,
        use_mm_prefix=False,
        use_per_head_quant_scales=False,
        device_capability=DeviceCapability(8, 0),
        attn_type="decoder",
    )
    assert invalid_head_size == ["TurboQuant currently requires head_size=128"]

    invalid_block_size = TritonAttentionBackend.validate_configuration(
        head_size=128,
        dtype=torch.float16,
        kv_cache_dtype="turboquant_3_2",
        block_size=32,
        use_mla=False,
        has_sink=False,
        use_sparse=False,
        use_mm_prefix=False,
        use_per_head_quant_scales=False,
        device_capability=DeviceCapability(9, 0),
        attn_type="decoder",
    )
    assert invalid_block_size == ["TurboQuant currently requires block_size=16"]

    invalid_sm = TritonAttentionBackend.validate_configuration(
        head_size=128,
        dtype=torch.float16,
        kv_cache_dtype="turboquant_3_2",
        block_size=16,
        use_mla=False,
        has_sink=False,
        use_sparse=False,
        use_mm_prefix=False,
        use_per_head_quant_scales=False,
        device_capability=DeviceCapability(8, 0),
        attn_type="decoder",
    )
    assert invalid_sm == ["TurboQuant currently requires sm90/H100"]


def test_attention_layer_returns_turboquant_spec():
    cache_config = CacheConfig(
        cache_dtype="turboquant_3_2",
        block_size=16,
        enable_turboquant=True,
    )
    vllm_config = VllmConfig(cache_config=cache_config)

    with set_current_vllm_config(vllm_config):
        layer = Attention(
            num_heads=32,
            head_size=128,
            scale=0.1,
            num_kv_heads=8,
            cache_config=cache_config,
            prefix="layers.0.self_attn",
            attn_backend=TritonAttentionBackend,
        )
        spec = layer.get_kv_cache_spec(vllm_config)

    assert isinstance(spec, TurboQuantAttentionSpec)
    assert layer._turboquant_layer_name == "layers.0.self_attn"
    assert layer._turboquant_model_name is None


def test_selector_picks_triton_backend_on_supported_cuda():
    if CudaPlatform is None:
        pytest.skip("CudaPlatform not available")

    cache_config = CacheConfig(
        cache_dtype="turboquant_3_2",
        block_size=16,
        enable_turboquant=True,
    )
    vllm_config = VllmConfig(cache_config=cache_config)

    with (
        set_current_vllm_config(vllm_config),
        patch("vllm.platforms.current_platform", CudaPlatform()),
        patch.object(
            CudaPlatform,
            "get_device_capability",
            return_value=DeviceCapability(9, 0),
        ),
    ):
        backend = get_attn_backend(
            head_size=128,
            dtype=torch.float16,
            kv_cache_dtype="turboquant_3_2",
        )

    assert backend is TritonAttentionBackend


def test_selector_rejects_turboquant_on_cpu():
    cache_config = CacheConfig(
        cache_dtype="turboquant_3_2",
        block_size=16,
        enable_turboquant=True,
    )
    vllm_config = VllmConfig(cache_config=cache_config)

    with (
        set_current_vllm_config(vllm_config),
        patch("vllm.platforms.current_platform", CpuPlatform()),
        pytest.raises(ValueError, match="TurboQuant backend requires CUDA"),
    ):
        get_attn_backend(
            head_size=128,
            dtype=torch.float16,
            kv_cache_dtype="turboquant_3_2",
        )
