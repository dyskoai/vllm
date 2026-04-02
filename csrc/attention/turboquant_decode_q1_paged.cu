#include "ops.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/all.h>

#include <cfloat>
#include <cmath>

namespace {

constexpr int kHeadSize = 128;
constexpr int kBlockSize = 16;
constexpr int kThreads = 128;
constexpr int kGroup0Dim = 32;
constexpr int kGroup1Dim = 96;
constexpr int kGroup0MseOutputs = 32;
constexpr int kGroup0QjlOutputs = 32;
constexpr int kGroup1MseOutputs = 96;
constexpr int kGroup1QjlOutputs = 96;
constexpr int kPackedOutputs =
    kGroup0MseOutputs + kGroup0QjlOutputs + kGroup1MseOutputs + kGroup1QjlOutputs;

constexpr int kGroup0MseBytes = 8;
constexpr int kGroup0QjlBytes = 4;
constexpr int kGroup0VecNormOffset = 12;
constexpr int kGroup0ResNormOffset = 14;
constexpr int kGroup0PackedBytes = 16;

constexpr int kGroup1MseBytes = 12;
constexpr int kGroup1QjlBytes = 12;
constexpr int kGroup1VecNormOffset = 24;
constexpr int kGroup1ResNormOffset = 26;

constexpr float kTurboQuantQjlScale = 1.2533141373155001f;  // sqrt(pi / 2)
constexpr float kQjlScale0 = kTurboQuantQjlScale / static_cast<float>(kGroup0Dim);
constexpr float kQjlScale1 = kTurboQuantQjlScale / static_cast<float>(kGroup1Dim);

__device__ inline float tanh_fast(float x) { return tanhf(x); }

__device__ inline int read_norm_index(const uint8_t* base, int offset) {
  return static_cast<int>(base[offset]) |
         (static_cast<int>(base[offset + 1]) << 8);
}

__device__ inline float read_norm(const uint8_t* base, int offset,
                                  const float* norm_lut) {
  return norm_lut[read_norm_index(base, offset)];
}

__device__ inline float read_group0_mse_centroid(const uint8_t* base, int dim_idx,
                                                 const float* centroids2) {
  const uint8_t packed = base[dim_idx >> 2];
  const int shift = (dim_idx & 0x3) << 1;
  const int centroid_idx = (packed >> shift) & 0x3;
  return centroids2[centroid_idx];
}

__device__ inline float read_group0_qjl_sign(const uint8_t* base, int dim_idx) {
  const uint8_t packed = base[kGroup0MseBytes + (dim_idx >> 3)];
  return ((packed >> (dim_idx & 0x7)) & 0x1) ? 1.0f : -1.0f;
}

__device__ inline float read_group1_mse_centroid(const uint8_t* base, int dim_idx,
                                                 const float* centroids1) {
  const uint8_t packed = base[dim_idx >> 3];
  const int centroid_idx = (packed >> (dim_idx & 0x7)) & 0x1;
  return centroids1[centroid_idx];
}

__device__ inline float read_group1_qjl_sign(const uint8_t* base, int dim_idx) {
  const uint8_t packed = base[kGroup1MseBytes + (dim_idx >> 3)];
  return ((packed >> (dim_idx & 0x7)) & 0x1) ? 1.0f : -1.0f;
}

__device__ inline float decode_value_component(
    const uint8_t* value_base, int output_idx, const float* centroids2,
    const float* centroids1, const float* norm_lut) {
  const uint8_t* group0_ptr = value_base;
  const uint8_t* group1_ptr = value_base + kGroup0PackedBytes;
  const float group0_vec_norm = read_norm(group0_ptr, kGroup0VecNormOffset, norm_lut);
  const float group0_res_norm = read_norm(group0_ptr, kGroup0ResNormOffset, norm_lut);
  const float group1_vec_norm = read_norm(group1_ptr, kGroup1VecNormOffset, norm_lut);
  const float group1_res_norm = read_norm(group1_ptr, kGroup1ResNormOffset, norm_lut);

  if (output_idx < kGroup0MseOutputs) {
    return group0_vec_norm *
           read_group0_mse_centroid(group0_ptr, output_idx, centroids2);
  }
  if (output_idx < (kGroup0MseOutputs + kGroup0QjlOutputs)) {
    const int dim_idx = output_idx - kGroup0MseOutputs;
    return group0_vec_norm * group0_res_norm *
           read_group0_qjl_sign(group0_ptr, dim_idx);
  }
  if (output_idx < (kGroup0MseOutputs + kGroup0QjlOutputs + kGroup1MseOutputs)) {
    const int dim_idx = output_idx - (kGroup0MseOutputs + kGroup0QjlOutputs);
    return group1_vec_norm *
           read_group1_mse_centroid(group1_ptr, dim_idx, centroids1);
  }
  const int dim_idx =
      output_idx - (kGroup0MseOutputs + kGroup0QjlOutputs + kGroup1MseOutputs);
  return group1_vec_norm * group1_res_norm *
         read_group1_qjl_sign(group1_ptr, dim_idx);
}

__global__ void turboquant_decode_q1_paged_kernel(
    float* out_g0_mse, float* out_g0_qjl, float* out_g1_mse, float* out_g1_qjl,
    const float* q_rot0, const float* q_qjl0, const float* q_rot1,
    const float* q_qjl1, const uint8_t* key_cache, const uint8_t* value_cache,
    const int32_t* block_tables, const int32_t* seq_lens,
    const int64_t* kv_head_for_query_head, const float* centroids2,
    const float* centroids1, const float* norm_lut,
    int64_t q_rot0_stride_0, int64_t q_rot0_stride_1, int64_t q_qjl0_stride_0,
    int64_t q_qjl0_stride_1, int64_t q_rot1_stride_0, int64_t q_rot1_stride_1,
    int64_t q_qjl1_stride_0, int64_t q_qjl1_stride_1, int64_t cache_stride_0,
    int64_t cache_stride_1, int64_t cache_stride_2, int64_t block_table_stride,
    int64_t out_g0_mse_stride_0, int64_t out_g0_mse_stride_1,
    int64_t out_g0_qjl_stride_0, int64_t out_g0_qjl_stride_1,
    int64_t out_g1_mse_stride_0, int64_t out_g1_mse_stride_1,
    int64_t out_g1_qjl_stride_0, int64_t out_g1_qjl_stride_1, float scale,
    float logits_soft_cap) {
  const int head_idx = blockIdx.x;
  const int seq_idx = blockIdx.y;
  const int tid = threadIdx.x;

  __shared__ float s_q_rot0[kGroup0Dim];
  __shared__ float s_q_qjl0[kGroup0Dim];
  __shared__ float s_q_rot1[kGroup1Dim];
  __shared__ float s_q_qjl1[kGroup1Dim];
  __shared__ float s_reduce0[kThreads];
  __shared__ float s_reduce1[kThreads];
  __shared__ float s_alpha;
  __shared__ float s_prob;
  __shared__ float s_l;
  __shared__ float s_m;

  if (tid < kGroup0Dim) {
    s_q_rot0[tid] =
        q_rot0[seq_idx * q_rot0_stride_0 + head_idx * q_rot0_stride_1 + tid];
    s_q_qjl0[tid] =
        q_qjl0[seq_idx * q_qjl0_stride_0 + head_idx * q_qjl0_stride_1 + tid];
  }
  if (tid < kGroup1Dim) {
    s_q_rot1[tid] =
        q_rot1[seq_idx * q_rot1_stride_0 + head_idx * q_rot1_stride_1 + tid];
    s_q_qjl1[tid] =
        q_qjl1[seq_idx * q_qjl1_stride_0 + head_idx * q_qjl1_stride_1 + tid];
  }
  if (tid == 0) {
    s_l = 0.0f;
    s_m = -FLT_MAX;
  }
  __syncthreads();

  const int64_t kv_head_idx = kv_head_for_query_head[head_idx];
  const int32_t seq_len = seq_lens[seq_idx];

  float acc0 = 0.0f;
  float acc1 = 0.0f;
  const int out_idx0 = tid;
  const int out_idx1 = tid + kThreads;

  for (int token_idx = 0; token_idx < seq_len; ++token_idx) {
    const int32_t block_id =
        block_tables[seq_idx * block_table_stride + token_idx / kBlockSize];
    const int32_t token_in_block = token_idx % kBlockSize;
    const int64_t token_base = static_cast<int64_t>(block_id) * cache_stride_0 +
                               static_cast<int64_t>(token_in_block) * cache_stride_1 +
                               kv_head_idx * cache_stride_2;

    const uint8_t* key_ptr = key_cache + token_base;
    float local_g0 = 0.0f;
    float local_g1 = 0.0f;

    if (tid < kGroup0Dim) {
      local_g0 =
          read_group0_mse_centroid(key_ptr, tid, centroids2) * s_q_rot0[tid] +
          read_norm(key_ptr, kGroup0ResNormOffset, norm_lut) * kQjlScale0 *
              read_group0_qjl_sign(key_ptr, tid) * s_q_qjl0[tid];
    }
    if (tid < kGroup1Dim) {
      const uint8_t* key_group1_ptr = key_ptr + kGroup0PackedBytes;
      local_g1 =
          read_group1_mse_centroid(key_group1_ptr, tid, centroids1) *
              s_q_rot1[tid] +
          read_norm(key_group1_ptr, kGroup1ResNormOffset, norm_lut) *
              kQjlScale1 * read_group1_qjl_sign(key_group1_ptr, tid) *
              s_q_qjl1[tid];
    }

    s_reduce0[tid] = local_g0;
    s_reduce1[tid] = local_g1;
    __syncthreads();

    for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        s_reduce0[tid] += s_reduce0[tid + stride];
        s_reduce1[tid] += s_reduce1[tid + stride];
      }
      __syncthreads();
    }

    if (tid == 0) {
      const float group0_vec_norm = read_norm(key_ptr, kGroup0VecNormOffset, norm_lut);
      const float group1_vec_norm =
          read_norm(key_ptr + kGroup0PackedBytes, kGroup1VecNormOffset, norm_lut);
      float score = group0_vec_norm * s_reduce0[0] + group1_vec_norm * s_reduce1[0];
      score *= scale;
      if (logits_soft_cap > 0.0f) {
        score = logits_soft_cap * tanh_fast(score / logits_soft_cap);
      }

      const float prev_m = s_m;
      const float prev_l = s_l;
      const float next_m = fmaxf(prev_m, score);
      const float alpha =
          (prev_l == 0.0f) ? 0.0f : expf(prev_m - next_m);
      const float prob = expf(score - next_m);
      s_alpha = alpha;
      s_prob = prob;
      s_m = next_m;
      s_l = prev_l * alpha + prob;
    }
    __syncthreads();

    const uint8_t* value_ptr = value_cache + token_base;
    acc0 *= s_alpha;
    if (out_idx0 < kPackedOutputs) {
      acc0 += s_prob * decode_value_component(value_ptr, out_idx0, centroids2,
                                              centroids1, norm_lut);
    }
    acc1 *= s_alpha;
    if (out_idx1 < kPackedOutputs) {
      acc1 += s_prob * decode_value_component(value_ptr, out_idx1, centroids2,
                                              centroids1, norm_lut);
    }
    __syncthreads();
  }

  const float inv_l = (s_l > 0.0f) ? (1.0f / s_l) : 0.0f;

  if (out_idx0 < kGroup0MseOutputs) {
    out_g0_mse[seq_idx * out_g0_mse_stride_0 + head_idx * out_g0_mse_stride_1 +
               out_idx0] = acc0 * inv_l;
  } else if (out_idx0 < (kGroup0MseOutputs + kGroup0QjlOutputs)) {
    const int offset = out_idx0 - kGroup0MseOutputs;
    out_g0_qjl[seq_idx * out_g0_qjl_stride_0 + head_idx * out_g0_qjl_stride_1 +
               offset] = acc0 * inv_l;
  } else if (out_idx0 <
             (kGroup0MseOutputs + kGroup0QjlOutputs + kGroup1MseOutputs)) {
    const int offset = out_idx0 - (kGroup0MseOutputs + kGroup0QjlOutputs);
    out_g1_mse[seq_idx * out_g1_mse_stride_0 + head_idx * out_g1_mse_stride_1 +
               offset] = acc0 * inv_l;
  } else {
    const int offset =
        out_idx0 - (kGroup0MseOutputs + kGroup0QjlOutputs + kGroup1MseOutputs);
    out_g1_qjl[seq_idx * out_g1_qjl_stride_0 + head_idx * out_g1_qjl_stride_1 +
               offset] = acc0 * inv_l;
  }

  if (out_idx1 < (kGroup0MseOutputs + kGroup0QjlOutputs + kGroup1MseOutputs)) {
    const int offset = out_idx1 - (kGroup0MseOutputs + kGroup0QjlOutputs);
    out_g1_mse[seq_idx * out_g1_mse_stride_0 + head_idx * out_g1_mse_stride_1 +
               offset] = acc1 * inv_l;
  } else if (out_idx1 < kPackedOutputs) {
    const int offset =
        out_idx1 - (kGroup0MseOutputs + kGroup0QjlOutputs + kGroup1MseOutputs);
    out_g1_qjl[seq_idx * out_g1_qjl_stride_0 + head_idx * out_g1_qjl_stride_1 +
               offset] = acc1 * inv_l;
  }
}

void check_tensor_cuda(torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA.");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous.");
}

}  // namespace

void turboquant_decode_q1_paged(
    torch::Tensor& out_g0_mse, torch::Tensor& out_g0_qjl,
    torch::Tensor& out_g1_mse, torch::Tensor& out_g1_qjl,
    torch::Tensor& q_rot0, torch::Tensor& q_qjl0, torch::Tensor& q_rot1,
    torch::Tensor& q_qjl1, torch::Tensor& key_cache, torch::Tensor& value_cache,
    torch::Tensor& block_tables, torch::Tensor& seq_lens,
    torch::Tensor& kv_head_for_query_head, torch::Tensor& centroids2,
    torch::Tensor& centroids1, torch::Tensor& norm_lut, double scale,
    double logits_soft_cap) {
  check_tensor_cuda(out_g0_mse, "out_g0_mse");
  check_tensor_cuda(out_g0_qjl, "out_g0_qjl");
  check_tensor_cuda(out_g1_mse, "out_g1_mse");
  check_tensor_cuda(out_g1_qjl, "out_g1_qjl");
  check_tensor_cuda(q_rot0, "q_rot0");
  check_tensor_cuda(q_qjl0, "q_qjl0");
  check_tensor_cuda(q_rot1, "q_rot1");
  check_tensor_cuda(q_qjl1, "q_qjl1");
  check_tensor_cuda(key_cache, "key_cache");
  check_tensor_cuda(value_cache, "value_cache");
  check_tensor_cuda(block_tables, "block_tables");
  check_tensor_cuda(seq_lens, "seq_lens");
  check_tensor_cuda(kv_head_for_query_head, "kv_head_for_query_head");
  check_tensor_cuda(centroids2, "centroids2");
  check_tensor_cuda(centroids1, "centroids1");
  check_tensor_cuda(norm_lut, "norm_lut");

  TORCH_CHECK(q_rot0.scalar_type() == torch::kFloat32,
              "q_rot0 must be float32.");
  TORCH_CHECK(q_qjl0.scalar_type() == torch::kFloat32,
              "q_qjl0 must be float32.");
  TORCH_CHECK(q_rot1.scalar_type() == torch::kFloat32,
              "q_rot1 must be float32.");
  TORCH_CHECK(q_qjl1.scalar_type() == torch::kFloat32,
              "q_qjl1 must be float32.");
  TORCH_CHECK(key_cache.scalar_type() == torch::kUInt8,
              "key_cache must be uint8.");
  TORCH_CHECK(value_cache.scalar_type() == torch::kUInt8,
              "value_cache must be uint8.");
  TORCH_CHECK(block_tables.scalar_type() == torch::kInt32,
              "block_tables must be int32.");
  TORCH_CHECK(seq_lens.scalar_type() == torch::kInt32,
              "seq_lens must be int32.");
  TORCH_CHECK(kv_head_for_query_head.scalar_type() == torch::kInt64,
              "kv_head_for_query_head must be int64.");

  TORCH_CHECK(q_rot0.dim() == 3 && q_qjl0.dim() == 3 && q_rot1.dim() == 3 &&
                  q_qjl1.dim() == 3,
              "TurboQuant q1 decode expects transformed query tensors shaped "
              "[tokens, heads, dim].");
  TORCH_CHECK(key_cache.dim() == 4 && value_cache.dim() == 4,
              "TurboQuant q1 decode expects packed caches shaped "
              "[blocks, block_size, kv_heads, packed_dim].");
  TORCH_CHECK(q_rot0.size(2) == kGroup0Dim && q_qjl0.size(2) == kGroup0Dim,
              "TurboQuant native q1 decode requires group0 dim 32.");
  TORCH_CHECK(q_rot1.size(2) == kGroup1Dim && q_qjl1.size(2) == kGroup1Dim,
              "TurboQuant native q1 decode requires group1 dim 96.");
  TORCH_CHECK(key_cache.size(1) == kBlockSize && value_cache.size(1) == kBlockSize,
              "TurboQuant native q1 decode requires block_size=16.");
  TORCH_CHECK(key_cache.size(3) == kGroup0PackedBytes + 28 &&
                  value_cache.size(3) == kGroup0PackedBytes + 28,
              "TurboQuant native q1 decode requires the existing packed_dim=44 "
              "layout for head_size=128.");
  TORCH_CHECK(q_rot0.size(0) == seq_lens.size(0),
              "TurboQuant native q1 decode expects one transformed query token "
              "per sequence.");
  TORCH_CHECK(q_rot0.size(1) == kv_head_for_query_head.size(0),
              "TurboQuant native q1 decode expects kv_head mapping per query head.");
  TORCH_CHECK(out_g0_mse.size(0) == q_rot0.size(0) &&
                  out_g0_mse.size(1) == q_rot0.size(1) &&
                  out_g0_mse.size(2) == kGroup0Dim,
              "out_g0_mse has an unexpected shape.");
  TORCH_CHECK(out_g0_qjl.size(0) == q_rot0.size(0) &&
                  out_g0_qjl.size(1) == q_rot0.size(1) &&
                  out_g0_qjl.size(2) == kGroup0Dim,
              "out_g0_qjl has an unexpected shape.");
  TORCH_CHECK(out_g1_mse.size(0) == q_rot0.size(0) &&
                  out_g1_mse.size(1) == q_rot0.size(1) &&
                  out_g1_mse.size(2) == kGroup1Dim,
              "out_g1_mse has an unexpected shape.");
  TORCH_CHECK(out_g1_qjl.size(0) == q_rot0.size(0) &&
                  out_g1_qjl.size(1) == q_rot0.size(1) &&
                  out_g1_qjl.size(2) == kGroup1Dim,
              "out_g1_qjl has an unexpected shape.");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(q_rot0));
  const auto num_tokens = q_rot0.size(0);
  const auto num_heads = q_rot0.size(1);
  const dim3 grid(num_heads, num_tokens);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(q_rot0.get_device());

  turboquant_decode_q1_paged_kernel<<<grid, kThreads, 0, stream>>>(
      out_g0_mse.data_ptr<float>(), out_g0_qjl.data_ptr<float>(),
      out_g1_mse.data_ptr<float>(), out_g1_qjl.data_ptr<float>(),
      q_rot0.data_ptr<float>(), q_qjl0.data_ptr<float>(), q_rot1.data_ptr<float>(),
      q_qjl1.data_ptr<float>(), key_cache.data_ptr<uint8_t>(),
      value_cache.data_ptr<uint8_t>(), block_tables.data_ptr<int32_t>(),
      seq_lens.data_ptr<int32_t>(), kv_head_for_query_head.data_ptr<int64_t>(),
      centroids2.data_ptr<float>(), centroids1.data_ptr<float>(),
      norm_lut.data_ptr<float>(), q_rot0.stride(0), q_rot0.stride(1),
      q_qjl0.stride(0), q_qjl0.stride(1), q_rot1.stride(0), q_rot1.stride(1),
      q_qjl1.stride(0), q_qjl1.stride(1), key_cache.stride(0), key_cache.stride(1),
      key_cache.stride(2), block_tables.stride(0), out_g0_mse.stride(0),
      out_g0_mse.stride(1), out_g0_qjl.stride(0), out_g0_qjl.stride(1),
      out_g1_mse.stride(0), out_g1_mse.stride(1), out_g1_qjl.stride(0),
      out_g1_qjl.stride(1), static_cast<float>(scale),
      static_cast<float>(logits_soft_cap));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
