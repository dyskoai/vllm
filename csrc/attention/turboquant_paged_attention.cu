#include "ops.h"

#include <torch/all.h>

using namespace at::indexing;

namespace {

constexpr double kTurboQuantScale = 1024.0;

torch::Tensor unpack_2bit(const torch::Tensor& packed, int64_t count) {
  auto shifts = torch::arange(
      0, 8, 2, packed.options().dtype(torch::kLong).device(packed.device()));
  auto unpacked = torch::bitwise_and(
                      torch::bitwise_right_shift(packed.unsqueeze(-1).to(torch::kLong),
                                                 shifts),
                      0x3)
                      .reshape({packed.size(0), packed.size(1), -1});
  return unpacked.index({Slice(), Slice(), Slice(None, count)}).to(torch::kLong);
}

torch::Tensor unpack_1bit(const torch::Tensor& packed, int64_t count) {
  auto shifts = torch::arange(
      0, 8, 1, packed.options().dtype(torch::kLong).device(packed.device()));
  auto unpacked = torch::bitwise_and(
                      torch::bitwise_right_shift(packed.unsqueeze(-1).to(torch::kLong),
                                                 shifts),
                      0x1)
                      .reshape({packed.size(0), packed.size(1), -1});
  return unpacked.index({Slice(), Slice(), Slice(None, count)}).to(torch::kLong);
}

std::pair<torch::Tensor, torch::Tensor> dequantize_entries(
    const torch::Tensor& entries, const torch::Tensor& codebook,
    int64_t head_size, int64_t value_group_size) {
  const int64_t k_lm_bytes = head_size / 4;
  const int64_t k_qjl_bytes = head_size / 8;
  const int64_t gamma_end = k_lm_bytes + k_qjl_bytes + 2;
  const int64_t v_packed_end = gamma_end + head_size / 4;
  const int64_t num_groups = head_size / value_group_size;

  auto key_low = unpack_2bit(entries.index({Slice(), Slice(), Slice(None, k_lm_bytes)}),
                             head_size);
  auto key_high = unpack_1bit(
      entries.index({Slice(), Slice(), Slice(k_lm_bytes, k_lm_bytes + k_qjl_bytes)}),
      head_size);
  auto gamma_bytes =
      entries.index({Slice(), Slice(), Slice(k_lm_bytes + k_qjl_bytes, gamma_end)})
          .to(torch::kLong);
  auto gamma = (gamma_bytes.index({Slice(), Slice(), 0}) |
                torch::bitwise_left_shift(gamma_bytes.index({Slice(), Slice(), 1}), 8))
                   .to(torch::kFloat) /
               kTurboQuantScale;
  auto key_levels = key_low | torch::bitwise_left_shift(key_high, 2);
  auto key = ((key_levels.to(torch::kFloat) / 7.0) * 2.0 - 1.0) *
             gamma.unsqueeze(-1);

  auto value_levels = unpack_2bit(
      entries.index({Slice(), Slice(), Slice(gamma_end, v_packed_end)}), head_size);
  auto scale_bytes = entries.index({Slice(), Slice(), Slice(v_packed_end, None)})
                         .reshape({entries.size(0), entries.size(1), num_groups, 2})
                         .to(torch::kLong);
  auto scales = (scale_bytes.index({Slice(), Slice(), Slice(), 0}) |
                 torch::bitwise_left_shift(
                     scale_bytes.index({Slice(), Slice(), Slice(), 1}), 8))
                    .to(torch::kFloat) /
                kTurboQuantScale;

  auto codebook_values =
      codebook.to(entries.device(), torch::kFloat)
          .index({value_levels.reshape({-1})})
          .view({entries.size(0), entries.size(1), num_groups, value_group_size});
  auto values = codebook_values;
  values = values * scales.unsqueeze(-1);
  auto value = values.reshape({entries.size(0), entries.size(1), head_size});
  return {key, value};
}

}  // namespace

void turboquant_paged_attention(torch::Tensor& out, torch::Tensor& query,
                                torch::Tensor& kv_cache, int64_t num_kv_heads,
                                double scale, torch::Tensor& block_tables,
                                torch::Tensor& seq_lens, int64_t block_size,
                                int64_t max_seq_len, torch::Tensor& rotation,
                                torch::Tensor& qjl_state,
                                torch::Tensor& codebook) {
  TORCH_CHECK(out.is_cuda(), "TurboQuant output must be CUDA.");
  TORCH_CHECK(query.is_cuda(), "TurboQuant query must be CUDA.");
  TORCH_CHECK(kv_cache.is_cuda(), "TurboQuant kv_cache must be CUDA.");
  TORCH_CHECK(kv_cache.dim() == 4,
              "TurboQuant kv_cache must have shape [blocks, block, heads, entry].");

  const auto batch_size = query.size(0);
  const auto num_heads = query.size(1);
  const auto head_size = query.size(2);
  TORCH_CHECK(head_size == 128,
              "TurboQuant initial landing only supports head_size 128.");
  TORCH_CHECK(block_size == 16,
              "TurboQuant initial landing only supports block_size 16.");
  TORCH_CHECK(batch_size == seq_lens.size(0),
              "TurboQuant decode expects one query token per request.");
  TORCH_CHECK(num_heads % num_kv_heads == 0,
              "TurboQuant decode requires num_heads divisible by num_kv_heads.");

  for (int64_t req_idx = 0; req_idx < batch_size; ++req_idx) {
    const auto seq_len = seq_lens.index({req_idx}).item<int64_t>();
    if (seq_len <= 0) {
      out.index_put_({req_idx}, torch::zeros_like(out.index({req_idx})));
      continue;
    }

    const auto num_blocks = (seq_len + block_size - 1) / block_size;
    auto block_ids =
        block_tables.index({req_idx, Slice(None, num_blocks)}).to(torch::kLong);
    auto pages = kv_cache.index_select(0, block_ids).reshape(
        {num_blocks * block_size, kv_cache.size(2), kv_cache.size(3)});
    auto entries = pages.index({Slice(None, seq_len)});

    auto [keys, values] =
        dequantize_entries(entries, codebook, head_size, /*value_group_size=*/64);
    if (num_heads != num_kv_heads) {
      const auto repeat = num_heads / num_kv_heads;
      keys = keys.repeat_interleave(repeat, 1);
      values = values.repeat_interleave(repeat, 1);
    }

    auto q = query.index({req_idx}).to(torch::kFloat);
    auto scores = torch::einsum("hd,thd->ht", {q, keys.to(torch::kFloat)}) * scale;
    auto probs = torch::softmax(scores, -1);
    auto result =
        torch::einsum("ht,thd->hd", {probs, values.to(torch::kFloat)}).to(out.dtype());
    out.index_put_({req_idx}, result);
  }

  (void)max_seq_len;
  (void)rotation;
  (void)qjl_state;
}
