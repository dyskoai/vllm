#include "cache.h"

#include <torch/all.h>

using namespace at::indexing;

namespace {

constexpr double kTurboQuantScale = 1024.0;

torch::Tensor pack_2bit(const torch::Tensor& values) {
  auto grouped = values.to(torch::kLong).view(
      {values.size(0), values.size(1), values.size(2) / 4, 4});
  auto shifts = torch::arange(
      0, 8, 2, grouped.options().dtype(torch::kLong).device(grouped.device()));
  return torch::bitwise_left_shift(grouped, shifts)
      .sum(-1)
      .to(torch::kUInt8);
}

torch::Tensor pack_1bit(const torch::Tensor& values) {
  auto grouped = values.to(torch::kLong).view(
      {values.size(0), values.size(1), values.size(2) / 8, 8});
  auto shifts = torch::arange(
      0, 8, 1, grouped.options().dtype(torch::kLong).device(grouped.device()));
  return torch::bitwise_left_shift(grouped, shifts)
      .sum(-1)
      .to(torch::kUInt8);
}

torch::Tensor quantize_to_u16_bytes(const torch::Tensor& values) {
  auto q = torch::clamp((values * kTurboQuantScale).round(), 0, 65535)
               .to(torch::kLong);
  auto lo = torch::bitwise_and(q, 0xFF).to(torch::kUInt8);
  auto hi = torch::bitwise_right_shift(q, 8).to(torch::kUInt8);
  return torch::stack({lo, hi}, -1);
}

}  // namespace

void reshape_and_cache_turboquant(torch::Tensor& key, torch::Tensor& value,
                                  torch::Tensor& kv_cache,
                                  torch::Tensor& slot_mapping,
                                  torch::Tensor& rotation,
                                  torch::Tensor& qjl_state,
                                  torch::Tensor& codebook,
                                  int64_t value_group_size) {
  TORCH_CHECK(key.is_cuda(), "TurboQuant key must be a CUDA tensor.");
  TORCH_CHECK(value.is_cuda(), "TurboQuant value must be a CUDA tensor.");
  TORCH_CHECK(kv_cache.is_cuda(), "TurboQuant kv_cache must be a CUDA tensor.");
  TORCH_CHECK(kv_cache.dtype() == torch::kUInt8,
              "TurboQuant kv_cache must use uint8 storage.");
  TORCH_CHECK(key.dim() == 3 && value.dim() == 3,
              "TurboQuant expects key/value tensors shaped [tokens, heads, dim].");
  TORCH_CHECK(kv_cache.dim() == 4,
              "TurboQuant kv_cache must have shape [blocks, block, heads, entry_bytes].");

  const auto num_tokens = slot_mapping.size(0);
  if (num_tokens == 0) {
    return;
  }

  auto valid_mask = slot_mapping >= 0;
  if (valid_mask.sum().item<int64_t>() == 0) {
    return;
  }

  auto valid_indices = torch::nonzero(valid_mask).squeeze(-1);
  auto key_actual = key.index_select(0, valid_indices).contiguous();
  auto value_actual = value.index_select(0, valid_indices).contiguous();
  auto slots = slot_mapping.index_select(0, valid_indices).to(torch::kLong);

  const auto block_size = kv_cache.size(1);
  const auto num_heads = key_actual.size(1);
  const auto head_size = key_actual.size(2);
  TORCH_CHECK(head_size == 128,
              "TurboQuant initial landing only supports head_size 128.");
  TORCH_CHECK(block_size == 16,
              "TurboQuant initial landing only supports block_size 16.");
  TORCH_CHECK(head_size % value_group_size == 0,
              "TurboQuant head_size must be divisible by value_group_size.");

  auto gamma = torch::amax(torch::abs(key_actual), -1).clamp_min(1e-6);
  auto key_levels =
      torch::clamp((((key_actual / gamma.unsqueeze(-1)) + 1.0) * 0.5 * 7.0).round(),
                   0, 7)
          .to(torch::kUInt8);
  auto key_low = torch::bitwise_and(key_levels, 0x3);
  auto key_high = torch::bitwise_and(
      torch::bitwise_right_shift(key_levels.to(torch::kLong), 2), 0x1)
                      .to(torch::kUInt8);

  auto num_groups = head_size / value_group_size;
  auto value_grouped = value_actual.view(
      {value_actual.size(0), num_heads, num_groups, value_group_size});
  auto value_scales =
      torch::amax(torch::abs(value_grouped), -1).div(1.5).clamp_min(1e-6);
  auto codebook_view = codebook.to(value.device(), torch::kFloat)
                           .view({1, 1, 1, 1, codebook.size(0)});
  auto normalized_values = value_grouped / value_scales.unsqueeze(-1);
  auto value_levels = torch::argmin(
                          torch::abs(normalized_values.unsqueeze(-1) - codebook_view),
                          -1)
                          .to(torch::kUInt8);
  auto value_levels_flat =
      value_levels.view({value_actual.size(0), num_heads, head_size});

  auto packed_entries = torch::cat(
      {
          pack_2bit(key_low),
          pack_1bit(key_high),
          quantize_to_u16_bytes(gamma),
          pack_2bit(value_levels_flat),
          quantize_to_u16_bytes(value_scales)
              .view({value_actual.size(0), num_heads, num_groups * 2}),
      },
      -1);

  auto flat_cache = kv_cache.view(
      {kv_cache.size(0) * kv_cache.size(1), kv_cache.size(2), kv_cache.size(3)});
  flat_cache.index_put_({slots}, packed_entries);

  (void)rotation;
  (void)qjl_state;
}
