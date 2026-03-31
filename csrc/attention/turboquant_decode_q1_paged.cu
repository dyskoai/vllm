#include "ops.h"

#include <torch/all.h>

void turboquant_decode_q1_paged(
    torch::Tensor& out_g0_mse, torch::Tensor& out_g0_qjl,
    torch::Tensor& out_g1_mse, torch::Tensor& out_g1_qjl,
    torch::Tensor& q_rot0, torch::Tensor& q_qjl0, torch::Tensor& q_rot1,
    torch::Tensor& q_qjl1, torch::Tensor& key_cache, torch::Tensor& value_cache,
    torch::Tensor& block_tables, torch::Tensor& seq_lens,
    torch::Tensor& kv_head_for_query_head, torch::Tensor& centroids2,
    torch::Tensor& centroids1, torch::Tensor& norm_lut, double scale,
    double logits_soft_cap) {
  TORCH_CHECK(out_g0_mse.is_cuda(), "out_g0_mse must be CUDA.");
  TORCH_CHECK(out_g0_qjl.is_cuda(), "out_g0_qjl must be CUDA.");
  TORCH_CHECK(out_g1_mse.is_cuda(), "out_g1_mse must be CUDA.");
  TORCH_CHECK(out_g1_qjl.is_cuda(), "out_g1_qjl must be CUDA.");
  TORCH_CHECK(q_rot0.is_cuda(), "q_rot0 must be CUDA.");
  TORCH_CHECK(q_qjl0.is_cuda(), "q_qjl0 must be CUDA.");
  TORCH_CHECK(q_rot1.is_cuda(), "q_rot1 must be CUDA.");
  TORCH_CHECK(q_qjl1.is_cuda(), "q_qjl1 must be CUDA.");
  TORCH_CHECK(key_cache.is_cuda(), "key_cache must be CUDA.");
  TORCH_CHECK(value_cache.is_cuda(), "value_cache must be CUDA.");
  TORCH_CHECK(block_tables.is_cuda(), "block_tables must be CUDA.");
  TORCH_CHECK(seq_lens.is_cuda(), "seq_lens must be CUDA.");
  TORCH_CHECK(kv_head_for_query_head.is_cuda(),
              "kv_head_for_query_head must be CUDA.");
  TORCH_CHECK(centroids2.is_cuda(), "centroids2 must be CUDA.");
  TORCH_CHECK(centroids1.is_cuda(), "centroids1 must be CUDA.");
  TORCH_CHECK(norm_lut.is_cuda(), "norm_lut must be CUDA.");

  TORCH_CHECK(q_rot0.dim() == 3 && q_qjl0.dim() == 3 && q_rot1.dim() == 3 &&
                  q_qjl1.dim() == 3,
              "TurboQuant q1 decode expects transformed query tensors shaped "
              "[tokens, heads, dim].");
  TORCH_CHECK(key_cache.dim() == 4 && value_cache.dim() == 4,
              "TurboQuant q1 decode expects packed caches shaped "
              "[blocks, block_size, kv_heads, packed_dim].");

  (void)scale;
  (void)logits_soft_cap;

  TORCH_CHECK(
      false,
      "turboquant_decode_q1_paged is an experimental native entrypoint. "
      "The CUDA kernel is not implemented yet. Keep using the Triton "
      "TurboQuant decode path until this op is filled in.");
}
