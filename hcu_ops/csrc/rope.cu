// Single-launch rotary embedding for HCU.
//
// sgl_kernel.rotary_embedding launches one CTA per token with
// min(nq * rot_dim / 2, 512) threads, but its HCU build is compiled with a
// 256-thread launch bound, so sglang splits MLA's 64 query heads into 8 launches
// (~5.4 us each on gfx938, i.e. ~43 us per layer).  This kernel tiles heads over
// gridDim.y instead and covers q and k in one launch.  The per-element math is the
// same expression on the same scalar type as the vLLM/sgl_kernel kernel, so the
// output is bit-identical.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>

namespace {
constexpr int kThreads = 256;

template <typename scalar_t, bool kNeox>
__device__ __forceinline__ void rotate(scalar_t* __restrict__ arr, const scalar_t* __restrict__ cos_ptr,
                                       const scalar_t* __restrict__ sin_ptr, int rot_offset, int embed_dim) {
  int x_index, y_index;
  scalar_t cos, sin;
  if (kNeox) {
    x_index = rot_offset;
    y_index = embed_dim + rot_offset;
    cos = cos_ptr[x_index];
    sin = sin_ptr[x_index];
  } else {
    x_index = 2 * rot_offset;
    y_index = 2 * rot_offset + 1;
    cos = cos_ptr[x_index / 2];
    sin = sin_ptr[x_index / 2];
  }
  const scalar_t x = arr[x_index];
  const scalar_t y = arr[y_index];
  arr[x_index] = x * cos - y * sin;
  arr[y_index] = y * cos + x * sin;
}

// rotate() reading the pair from src and writing it to dst (same expression).
template <typename scalar_t, bool kNeox>
__device__ __forceinline__ void rotate_to(const scalar_t* __restrict__ src, scalar_t* __restrict__ dst,
                                          const scalar_t* __restrict__ cos_ptr,
                                          const scalar_t* __restrict__ sin_ptr, int rot_offset, int embed_dim) {
  int x_index, y_index;
  scalar_t cos, sin;
  if (kNeox) {
    x_index = rot_offset;
    y_index = embed_dim + rot_offset;
    cos = cos_ptr[x_index];
    sin = sin_ptr[x_index];
  } else {
    x_index = 2 * rot_offset;
    y_index = 2 * rot_offset + 1;
    cos = cos_ptr[x_index / 2];
    sin = sin_ptr[x_index / 2];
  }
  const scalar_t x = src[x_index];
  const scalar_t y = src[y_index];
  dst[x_index] = x * cos - y * sin;
  dst[y_index] = y * cos + x * sin;
}

// grid = (tokens, ceil((nq + nk) / heads_per_cta)); thread i of a CTA handles
// (head h0 + i / embed_dim, pair i % embed_dim), embed_dim = rot_dim / 2.
template <typename scalar_t, typename pos_t, bool kNeox>
__global__ __launch_bounds__(kThreads) void rope_kernel(
    const pos_t* __restrict__ positions, scalar_t* __restrict__ q, scalar_t* __restrict__ k,
    const scalar_t* __restrict__ cache, int rot_dim, int nq, int nk, int64_t q_tok, int64_t q_head,
    int64_t k_tok, int64_t k_head, int heads_per_cta) {
  const int64_t t = blockIdx.x;
  const int embed_dim = rot_dim / 2;
  const scalar_t* cos_ptr = cache + static_cast<int64_t>(positions[t]) * rot_dim;
  const scalar_t* sin_ptr = cos_ptr + embed_dim;
  const int h0 = blockIdx.y * heads_per_cta;
  for (int i = threadIdx.x; i < heads_per_cta * embed_dim; i += kThreads) {
    const int h = h0 + i / embed_dim;
    const int r = i % embed_dim;
    if (h < nq) {
      rotate<scalar_t, kNeox>(q + t * q_tok + h * q_head, cos_ptr, sin_ptr, r, embed_dim);
    } else if (h < nq + nk) {
      rotate<scalar_t, kNeox>(k + t * k_tok + (h - nq) * k_head, cos_ptr, sin_ptr, r, embed_dim);
    }
  }
}

// rope_kernel with the rotated query written to qo (MLA decode: straight into the
// rope tail of the attention's concatenated q) instead of in place; k stays in place.
template <typename scalar_t, typename pos_t, bool kNeox>
__global__ __launch_bounds__(kThreads) void rope_qout_kernel(
    const pos_t* __restrict__ positions, const scalar_t* __restrict__ q, scalar_t* __restrict__ qo,
    scalar_t* __restrict__ k, const scalar_t* __restrict__ cache, int rot_dim, int nq, int nk, int64_t q_tok,
    int64_t q_head, int64_t qo_tok, int64_t qo_head, int64_t k_tok, int64_t k_head, int heads_per_cta) {
  const int64_t t = blockIdx.x;
  const int embed_dim = rot_dim / 2;
  const scalar_t* cos_ptr = cache + static_cast<int64_t>(positions[t]) * rot_dim;
  const scalar_t* sin_ptr = cos_ptr + embed_dim;
  const int h0 = blockIdx.y * heads_per_cta;
  for (int i = threadIdx.x; i < heads_per_cta * embed_dim; i += kThreads) {
    const int h = h0 + i / embed_dim;
    const int r = i % embed_dim;
    if (h < nq) {
      rotate_to<scalar_t, kNeox>(q + t * q_tok + h * q_head, qo + t * qo_tok + h * qo_head, cos_ptr, sin_ptr, r,
                                 embed_dim);
    } else if (h < nq + nk) {
      rotate<scalar_t, kNeox>(k + t * k_tok + (h - nq) * k_head, cos_ptr, sin_ptr, r, embed_dim);
    }
  }
}

// [T, n * head_size] or [T, n, head_size] view with a unit last stride ->
// (n, token stride, head stride).
void head_view(const torch::Tensor& x, int64_t tokens, int64_t head_size, int& n, int64_t& tok,
               int64_t& head) {
  TORCH_CHECK(x.stride(-1) == 1, "rope: last dim must be contiguous");
  TORCH_CHECK(x.size(0) == tokens, "rope: token count mismatch");
  if (x.dim() == 2) {
    n = x.size(1) / head_size;
    tok = x.stride(0);
    head = head_size;
  } else {
    TORCH_CHECK(x.dim() == 3 && x.size(2) == head_size, "rope: expected [T, n, head_size]");
    n = x.size(1);
    tok = x.stride(0);
    head = x.stride(1);
  }
}

template <typename scalar_t, typename pos_t>
void launch(const torch::Tensor& positions, torch::Tensor& query, const c10::optional<torch::Tensor>& key,
            const torch::Tensor& cache, int rot_dim, int nq, int nk, int64_t q_tok, int64_t q_head,
            int64_t k_tok, int64_t k_head, bool is_neox) {
  const int heads_per_cta = std::max(1, kThreads / (rot_dim / 2));
  const dim3 grid(positions.numel(), (nq + nk + heads_per_cta - 1) / heads_per_cta);
  const auto stream = at::cuda::getCurrentCUDAStream().stream();
  auto pp = positions.data_ptr<pos_t>();
  auto qp = reinterpret_cast<scalar_t*>(query.data_ptr());
  auto kp = key.has_value() ? reinterpret_cast<scalar_t*>(key->data_ptr()) : nullptr;
  auto cp = reinterpret_cast<const scalar_t*>(cache.data_ptr());
  if (is_neox)
    rope_kernel<scalar_t, pos_t, true><<<grid, kThreads, 0, stream>>>(
        pp, qp, kp, cp, rot_dim, nq, nk, q_tok, q_head, k_tok, k_head, heads_per_cta);
  else
    rope_kernel<scalar_t, pos_t, false><<<grid, kThreads, 0, stream>>>(
        pp, qp, kp, cp, rot_dim, nq, nk, q_tok, q_head, k_tok, k_head, heads_per_cta);
}

template <typename scalar_t, typename pos_t>
void launch_qout(const torch::Tensor& positions, const torch::Tensor& query, torch::Tensor& q_out,
                 const c10::optional<torch::Tensor>& key, const torch::Tensor& cache, int rot_dim, int nq, int nk,
                 int64_t q_tok, int64_t q_head, int64_t qo_tok, int64_t qo_head, int64_t k_tok, int64_t k_head,
                 bool is_neox) {
  const int heads_per_cta = std::max(1, kThreads / (rot_dim / 2));
  const dim3 grid(positions.numel(), (nq + nk + heads_per_cta - 1) / heads_per_cta);
  const auto stream = at::cuda::getCurrentCUDAStream().stream();
  auto pp = positions.data_ptr<pos_t>();
  auto qp = reinterpret_cast<const scalar_t*>(query.data_ptr());
  auto op = reinterpret_cast<scalar_t*>(q_out.data_ptr());
  auto kp = key.has_value() ? reinterpret_cast<scalar_t*>(key->data_ptr()) : nullptr;
  auto cp = reinterpret_cast<const scalar_t*>(cache.data_ptr());
  if (is_neox)
    rope_qout_kernel<scalar_t, pos_t, true><<<grid, kThreads, 0, stream>>>(
        pp, qp, op, kp, cp, rot_dim, nq, nk, q_tok, q_head, qo_tok, qo_head, k_tok, k_head, heads_per_cta);
  else
    rope_qout_kernel<scalar_t, pos_t, false><<<grid, kThreads, 0, stream>>>(
        pp, qp, op, kp, cp, rot_dim, nq, nk, q_tok, q_head, qo_tok, qo_head, k_tok, k_head, heads_per_cta);
}
}  // namespace

void rope_inplace(const torch::Tensor& positions, torch::Tensor& query, const c10::optional<torch::Tensor>& key,
                  int64_t head_size, const torch::Tensor& cos_sin_cache, bool is_neox) {
  const int64_t tokens = positions.numel();
  if (tokens == 0) return;
  TORCH_CHECK(positions.is_contiguous(), "rope: positions must be contiguous");
  TORCH_CHECK(cos_sin_cache.is_contiguous() && cos_sin_cache.scalar_type() == query.scalar_type(),
              "rope: cos_sin_cache must be contiguous with the query dtype");
  const int rot_dim = cos_sin_cache.size(1);
  int nq, nk = 0;
  int64_t q_tok, q_head, k_tok = 0, k_head = 0;
  head_view(query, tokens, head_size, nq, q_tok, q_head);
  if (key.has_value()) {
    TORCH_CHECK(key->scalar_type() == query.scalar_type(), "rope: q/k dtype mismatch");
    head_view(*key, tokens, head_size, nk, k_tok, k_head);
  }
  const bool pos64 = positions.scalar_type() == at::kLong;
  TORCH_CHECK(pos64 || positions.scalar_type() == at::kInt, "rope: positions must be int32/int64");
#define L(T) (pos64 ? launch<T, int64_t> : launch<T, int32_t>)(positions, query, key, cos_sin_cache, \
      rot_dim, nq, nk, q_tok, q_head, k_tok, k_head, is_neox)
  if (query.scalar_type() == at::kBFloat16) L(c10::BFloat16);
  else if (query.scalar_type() == at::kHalf) L(c10::Half);
  else TORCH_CHECK(false, "rope: query must be bf16/fp16");
#undef L
}

void rope_qout(const torch::Tensor& positions, const torch::Tensor& query, torch::Tensor& q_out,
               const c10::optional<torch::Tensor>& key, int64_t head_size, const torch::Tensor& cos_sin_cache,
               bool is_neox) {
  const int64_t tokens = positions.numel();
  if (tokens == 0) return;
  TORCH_CHECK(positions.is_contiguous(), "rope: positions must be contiguous");
  TORCH_CHECK(cos_sin_cache.is_contiguous() && cos_sin_cache.scalar_type() == query.scalar_type(),
              "rope: cos_sin_cache must be contiguous with the query dtype");
  TORCH_CHECK(q_out.scalar_type() == query.scalar_type() && q_out.sizes() == query.sizes(),
              "rope: q_out must match the query shape / dtype");
  const int rot_dim = cos_sin_cache.size(1);
  int nq, nqo, nk = 0;
  int64_t q_tok, q_head, qo_tok, qo_head, k_tok = 0, k_head = 0;
  head_view(query, tokens, head_size, nq, q_tok, q_head);
  head_view(q_out, tokens, head_size, nqo, qo_tok, qo_head);
  if (key.has_value()) {
    TORCH_CHECK(key->scalar_type() == query.scalar_type(), "rope: q/k dtype mismatch");
    head_view(*key, tokens, head_size, nk, k_tok, k_head);
  }
  const bool pos64 = positions.scalar_type() == at::kLong;
  TORCH_CHECK(pos64 || positions.scalar_type() == at::kInt, "rope: positions must be int32/int64");
#define L(T) (pos64 ? launch_qout<T, int64_t> : launch_qout<T, int32_t>)(positions, query, q_out, key, \
      cos_sin_cache, rot_dim, nq, nk, q_tok, q_head, qo_tok, qo_head, k_tok, k_head, is_neox)
  if (query.scalar_type() == at::kBFloat16) L(c10::BFloat16);
  else if (query.scalar_type() == at::kHalf) L(c10::Half);
  else TORCH_CHECK(false, "rope: query must be bf16/fp16");
#undef L
}
