# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from unittest.mock import patch

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.config import CacheConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.attention import Attention
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backends.triton_attn import TritonAttentionBackend
from vllm.v1.attention.ops.triton_turboquant_kv_update import (
    turboquant_write_packed_kv,
)
from vllm.v1.attention.ops.turboquant_kv_cache import (
    get_turboquant_centroids,
    get_turboquant_layout,
    get_turboquant_mse_transform_matrix,
    get_turboquant_packed_dim,
    get_turboquant_qjl_matrix,
    get_turboquant_rotation,
    quantize_turboquant_vectors,
)
from vllm.v1.attention.ops.triton_turboquant_decode import (
    _native_q1_decode_enabled,
    _turboquant_decode_q1_fused,
    get_turboquant_norm_lut,
    turboquant_decode_attention_fwd,
)
from vllm.v1.attention.ops.turboquant_metadata import (
    build_default_turboquant_metadata,
)
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


def test_turboquant_write_packed_kv_requires_sign_vectors():
    device = torch.device("cpu")
    head_size = 128
    num_kv_heads = 4
    layout = get_turboquant_layout("turboquant_3_2", head_size)
    metadata = build_default_turboquant_metadata(
        recipe="turboquant_3_2",
        head_size=head_size,
        num_kv_heads=num_kv_heads,
        layer_names=["attn"],
    )
    group_indices = metadata.get_layer("attn").key.get_group_indices(
        device, head_size, "turboquant_3_2"
    )
    rotations = tuple(
        get_turboquant_rotation(device, group.dim, idx)
        for idx, group in enumerate(layout.groups)
    )
    qjl_matrices = tuple(
        get_turboquant_qjl_matrix(device, group.dim, idx)
        for idx, group in enumerate(layout.groups)
    )
    centroids = {
        group.mse_bits: get_turboquant_centroids(device, group.dim, group.mse_bits)
        for group in layout.groups
        if group.mse_bits > 0
    }
    x = torch.randn(3, num_kv_heads, head_size, dtype=torch.float32)
    cache = torch.zeros(1, 16, num_kv_heads, layout.packed_dim, dtype=torch.uint8)
    slot_mapping = torch.tensor([0, 1, 2], dtype=torch.int32)
    unused_mse_to_qjl = tuple(torch.empty(0) for _ in layout.groups)

    turboquant_write_packed_kv(
        x,
        cache,
        slot_mapping,
        layout,
        group_indices,
        rotations,
        qjl_matrices,
        unused_mse_to_qjl,
        centroids,
    )

    assert torch.count_nonzero(cache[0, :3]) > 0

    mse_matrices = tuple(
        get_turboquant_mse_transform_matrix(device, group.dim, idx)
        for idx, group in enumerate(layout.groups)
    )
    with pytest.raises(ValueError, match="1D sign vectors"):
        turboquant_write_packed_kv(
            x,
            cache,
            slot_mapping,
            layout,
            group_indices,
            mse_matrices,
            qjl_matrices,
            unused_mse_to_qjl,
            centroids,
        )


@pytest.fixture
def turboquant_decode_fixture():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for TurboQuant decode tests")

    device = torch.device("cuda")
    dtype = torch.bfloat16
    kv_cache_dtype = "turboquant_3_2"
    head_size = 128
    num_heads = 4
    num_kv_heads = 2
    num_tokens = 2
    block_size = 16
    seq_lens = torch.tensor([17, 11], device=device, dtype=torch.int32)
    max_seq_len = int(seq_lens.max().item())
    blocks_per_seq = (max_seq_len + block_size - 1) // block_size
    total_blocks = num_tokens * blocks_per_seq

    metadata = build_default_turboquant_metadata(
        recipe=kv_cache_dtype,
        head_size=head_size,
        num_kv_heads=num_kv_heads,
        layer_names=["attn"],
    )
    layer_metadata = metadata.get_layer("attn")
    key_group_indices = layer_metadata.key.get_group_indices(
        device, head_size, kv_cache_dtype
    )
    value_group_indices = layer_metadata.value.get_group_indices(
        device, head_size, kv_cache_dtype
    )
    layout = get_turboquant_layout(kv_cache_dtype, head_size)

    key_rotations = tuple(
        get_turboquant_rotation(device, group.dim, idx)
        for idx, group in enumerate(layout.groups)
    )
    key_qjl_matrices = tuple(
        get_turboquant_qjl_matrix(device, group.dim, idx)
        for idx, group in enumerate(layout.groups)
    )
    value_rotations = key_rotations
    value_qjl_matrices = key_qjl_matrices
    centroids = {
        group.mse_bits: get_turboquant_centroids(device, group.dim, group.mse_bits)
        for group in layout.groups
        if group.mse_bits > 0
    }
    norm_lut = get_turboquant_norm_lut(device)
    kv_head_for_query_head = torch.tensor(
        [0, 0, 1, 1], device=device, dtype=torch.int64
    )

    torch.manual_seed(0)
    query = torch.randn(num_tokens, num_heads, head_size, device=device, dtype=dtype)
    key_states = torch.randn(
        num_tokens, max_seq_len, num_kv_heads, head_size, device=device, dtype=dtype
    )
    value_states = torch.randn_like(key_states)

    packed_dim = layout.packed_dim
    key_cache = torch.zeros(
        total_blocks, block_size, num_kv_heads, packed_dim, device=device, dtype=torch.uint8
    )
    value_cache = torch.zeros_like(key_cache)
    block_table = torch.zeros(num_tokens, blocks_per_seq, device=device, dtype=torch.int32)
    for seq_idx, seq_len in enumerate(seq_lens.tolist()):
        key_packed = quantize_turboquant_vectors(
            key_states[seq_idx, :seq_len],
            kv_cache_dtype,
            key_rotations,
            key_qjl_matrices,
            centroids,
            key_group_indices,
        )
        value_packed = quantize_turboquant_vectors(
            value_states[seq_idx, :seq_len],
            kv_cache_dtype,
            value_rotations,
            value_qjl_matrices,
            centroids,
            value_group_indices,
        )
        for block_idx in range((seq_len + block_size - 1) // block_size):
            dst_block = seq_idx * blocks_per_seq + block_idx
            src_start = block_idx * block_size
            src_end = min(src_start + block_size, seq_len)
            token_count = src_end - src_start
            key_cache[dst_block, :token_count] = key_packed[src_start:src_end]
            value_cache[dst_block, :token_count] = value_packed[src_start:src_end]
            block_table[seq_idx, block_idx] = dst_block

    query_start_loc = torch.arange(
        0, num_tokens + 1, device=device, dtype=torch.int32
    )
    softmax_scale = head_size**-0.5

    return {
        "query": query,
        "key_cache": key_cache,
        "value_cache": value_cache,
        "block_table": block_table,
        "query_start_loc": query_start_loc,
        "seq_lens": seq_lens,
        "key_group_indices": key_group_indices,
        "value_group_indices": value_group_indices,
        "key_rotations": key_rotations,
        "key_qjl_matrices": key_qjl_matrices,
        "value_rotations": value_rotations,
        "value_qjl_matrices": value_qjl_matrices,
        "centroids": centroids,
        "norm_lut": norm_lut,
        "softmax_scale": softmax_scale,
        "kv_cache_dtype": kv_cache_dtype,
        "kv_head_for_query_head": kv_head_for_query_head,
    }


def test_turboquant_native_q1_dispatch_uses_native_when_enabled(
    turboquant_decode_fixture,
):
    fixture = turboquant_decode_fixture
    _native_q1_decode_enabled.cache_clear()

    def fake_native(*args, **kwargs):
        args[0].zero_()
        args[1].zero_()
        args[2].zero_()
        args[3].zero_()

    with (
        patch.dict(os.environ, {"VLLM_TURBOQUANT_NATIVE_Q1": "1"}),
        patch.object(ops, "turboquant_decode_q1_paged", side_effect=fake_native) as native_op,
    ):
        output = _turboquant_decode_q1_fused(
            fixture["query"],
            fixture["key_cache"],
            fixture["value_cache"],
            fixture["block_table"],
            fixture["seq_lens"],
            fixture["key_group_indices"],
            fixture["value_group_indices"],
            fixture["key_rotations"],
            fixture["key_qjl_matrices"],
            fixture["value_rotations"],
            fixture["value_qjl_matrices"],
            fixture["centroids"],
            fixture["norm_lut"],
            fixture["softmax_scale"],
            fixture["kv_head_for_query_head"],
            None,
            None,
            0.0,
            None,
        )

    _native_q1_decode_enabled.cache_clear()
    native_op.assert_called_once()
    assert torch.count_nonzero(output) == 0


def test_turboquant_native_q1_dispatch_falls_back_when_disabled(
    turboquant_decode_fixture,
):
    fixture = turboquant_decode_fixture
    _native_q1_decode_enabled.cache_clear()

    with (
        patch.dict(os.environ, {"VLLM_TURBOQUANT_NATIVE_Q1": "0"}),
        patch.object(ops, "turboquant_decode_q1_paged") as native_op,
    ):
        output = _turboquant_decode_q1_fused(
            fixture["query"],
            fixture["key_cache"],
            fixture["value_cache"],
            fixture["block_table"],
            fixture["seq_lens"],
            fixture["key_group_indices"],
            fixture["value_group_indices"],
            fixture["key_rotations"],
            fixture["key_qjl_matrices"],
            fixture["value_rotations"],
            fixture["value_qjl_matrices"],
            fixture["centroids"],
            fixture["norm_lut"],
            fixture["softmax_scale"],
            fixture["kv_head_for_query_head"],
            None,
            None,
            0.0,
            None,
        )

    _native_q1_decode_enabled.cache_clear()
    native_op.assert_not_called()
    assert torch.count_nonzero(output) > 0


def test_turboquant_native_q1_matches_triton_reference(turboquant_decode_fixture):
    fixture = turboquant_decode_fixture
    _native_q1_decode_enabled.cache_clear()

    baseline = _turboquant_decode_q1_fused(
        fixture["query"],
        fixture["key_cache"],
        fixture["value_cache"],
        fixture["block_table"],
        fixture["seq_lens"],
        fixture["key_group_indices"],
        fixture["value_group_indices"],
        fixture["key_rotations"],
        fixture["key_qjl_matrices"],
        fixture["value_rotations"],
        fixture["value_qjl_matrices"],
        fixture["centroids"],
        fixture["norm_lut"],
        fixture["softmax_scale"],
        fixture["kv_head_for_query_head"],
        None,
        None,
        0.0,
        None,
    )

    native_success = {"value": False}
    real_native = ops.turboquant_decode_q1_paged

    def wrapped_native(*args, **kwargs):
        real_native(*args, **kwargs)
        native_success["value"] = True

    with (
        patch.dict(os.environ, {"VLLM_TURBOQUANT_NATIVE_Q1": "1"}),
        patch.object(ops, "turboquant_decode_q1_paged", side_effect=wrapped_native),
    ):
        _native_q1_decode_enabled.cache_clear()
        native = _turboquant_decode_q1_fused(
            fixture["query"],
            fixture["key_cache"],
            fixture["value_cache"],
            fixture["block_table"],
            fixture["seq_lens"],
            fixture["key_group_indices"],
            fixture["value_group_indices"],
            fixture["key_rotations"],
            fixture["key_qjl_matrices"],
            fixture["value_rotations"],
            fixture["value_qjl_matrices"],
            fixture["centroids"],
            fixture["norm_lut"],
            fixture["softmax_scale"],
            fixture["kv_head_for_query_head"],
            None,
            None,
            0.0,
            None,
        )

    _native_q1_decode_enabled.cache_clear()
    assert native_success["value"] is True
    torch.testing.assert_close(native, baseline, atol=1e-3, rtol=1e-3)


def test_turboquant_native_q1_falls_back_for_unsupported_query_shape(
    turboquant_decode_fixture,
):
    fixture = turboquant_decode_fixture
    _native_q1_decode_enabled.cache_clear()

    unsupported_query = fixture["query"][:1].repeat_interleave(2, dim=0)
    unsupported_query_start_loc = torch.tensor(
        [0, 2], device=unsupported_query.device, dtype=torch.int32
    )
    unsupported_seq_lens = fixture["seq_lens"][:1]
    unsupported_block_table = fixture["block_table"][:1]

    with (
        patch.dict(os.environ, {"VLLM_TURBOQUANT_NATIVE_Q1": "1"}),
        patch.object(ops, "turboquant_decode_q1_paged") as native_op,
    ):
        output = turboquant_decode_attention_fwd(
            query=unsupported_query,
            key_cache=fixture["key_cache"],
            value_cache=fixture["value_cache"],
            block_table=unsupported_block_table,
            query_start_loc=unsupported_query_start_loc,
            seq_lens=unsupported_seq_lens,
            key_group_indices=fixture["key_group_indices"],
            value_group_indices=fixture["value_group_indices"],
            key_rotations=fixture["key_rotations"],
            key_qjl_matrices=fixture["key_qjl_matrices"],
            value_rotations=fixture["value_rotations"],
            value_qjl_matrices=fixture["value_qjl_matrices"],
            centroids=fixture["centroids"],
            norm_lut=fixture["norm_lut"],
            softmax_scale=fixture["softmax_scale"],
            kv_cache_dtype=fixture["kv_cache_dtype"],
            kv_head_for_query_head=fixture["kv_head_for_query_head"],
            logits_soft_cap=0.0,
            out=None,
        )

    _native_q1_decode_enabled.cache_clear()
    native_op.assert_not_called()
    assert torch.count_nonzero(output) > 0


def test_turboquant_native_q1_rejects_unsupported_block_size():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for TurboQuant native decode validation")

    device = torch.device("cuda")
    out_g0_mse = torch.empty((1, 1, 32), device=device, dtype=torch.float32)
    out_g0_qjl = torch.empty_like(out_g0_mse)
    out_g1_mse = torch.empty((1, 1, 96), device=device, dtype=torch.float32)
    out_g1_qjl = torch.empty_like(out_g1_mse)
    q_rot0 = torch.randn((1, 1, 32), device=device, dtype=torch.float32)
    q_qjl0 = torch.randn_like(q_rot0)
    q_rot1 = torch.randn((1, 1, 96), device=device, dtype=torch.float32)
    q_qjl1 = torch.randn_like(q_rot1)
    key_cache = torch.zeros((1, 32, 1, 44), device=device, dtype=torch.uint8)
    value_cache = torch.zeros_like(key_cache)
    block_table = torch.zeros((1, 1), device=device, dtype=torch.int32)
    seq_lens = torch.ones((1,), device=device, dtype=torch.int32)
    kv_head_for_query_head = torch.zeros((1,), device=device, dtype=torch.int64)
    centroids2 = torch.randn((4,), device=device, dtype=torch.float32)
    centroids1 = torch.randn((2,), device=device, dtype=torch.float32)
    norm_lut = torch.zeros((1 << 16,), device=device, dtype=torch.float32)

    with pytest.raises(RuntimeError, match="block_size=16"):
        ops.turboquant_decode_q1_paged(
            out_g0_mse,
            out_g0_qjl,
            out_g1_mse,
            out_g1_qjl,
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
            1.0,
            0.0,
        )
