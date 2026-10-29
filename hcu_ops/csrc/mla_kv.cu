// MLA decode front end in one launch (sglang hook: SGLANG_HCU_MLA_KV_FUSED=1): the q rope of
// rope_qout, kv_a_layernorm of the latent and the DSA fp8 KV-cache store, which sglang runs as
// lightop rms_norm_opt + rope + lightop fused_quantize_and_store_mla_kv_cache (3 launches).
//
// grid = (T, ceil(nq / 8) + 1), 256 threads.  The q tiles are rope_qout's (8 heads x 32 pairs).
// The last CTA of token t:
//  * wave 0, 64 lanes x 8 elements: lightop rms_kernel_small_hidden_multi_token<512, 64> on
//    kv_c[t] (sequential sum of squares, shfl_down 32..1, c10 compat rsqrt on lane 0,
//    bf16(x * rstd * w)), written back in place, then the lightop store's fp8 group quant of
//    that bf16 row (amax over 128, scale = fmaxf(amax, eps) / 448, cvt_pk_fp8 of x / scale)
//    into kv_buffer[loc[t]]: 512 fp8 bytes, 4 fp32 scales at +512;
//  * lanes 0..31 of wave 1: the k rope pair (rope.cu's expression), in place in k_pe and
//    copied as bf16 to +528.
// Rows t >= loc.numel() are normed and rotated but not stored, like the unfused ops; the slot
// is not range checked (lightop doesn't either).
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAMathCompat.h>
#include <c10/util/BFloat16.h>
#include <torch/extension.h>

#include <cstdint>

namespace {

using bf16 = c10::BFloat16;
constexpr int kThreads = 256;
constexpr int kNope = 512;
constexpr int kRot = 64;
constexpr int kVec = 8;
constexpr int kScaleOff = kNope;           // 4 fp32 group scales
constexpr int kRopeOff = kNope + 4 * 4;    // 64 bf16
constexpr int kSlotBytes = kRopeOff + 2 * kRot;
constexpr int kHeadsPerCta = kThreads / (kRot / 2);

__device__ __forceinline__ float bf16_to_float(uint16_t h) { return __uint_as_float(uint32_t(h) << 16); }

// c10::BFloat16(float) on ROCm: round to nearest even, NaN -> 0x7FC0
__device__ __forceinline__ uint16_t float_to_bf16(float f) {
  if (f != f) return 0x7FC0;
  const uint32_t u = __float_as_uint(f);
  return static_cast<uint16_t>((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}

__device__ __forceinline__ uint32_t pack_fp8x4(float a, float b, float c, float d) {
  int32_t r = 0;
  r = __builtin_hcu_cvt_pk_fp8_f32(a, b, r, false);
  r = __builtin_hcu_cvt_pk_fp8_f32(c, d, r, true);
  return static_cast<uint32_t>(r);
}

// norm_quant.cu's rms_rstd (RSQRT 0 = lightop's c10::cuda::compat::rsqrt)
template <int RSQRT>
__device__ __forceinline__ float rms_rstd(float ss, int cols, float eps) {
  if constexpr (RSQRT == 0) return c10::cuda::compat::rsqrt(ss / cols + eps);
  if constexpr (RSQRT == 1) return __builtin_amdgcn_rsqf(ss / cols + eps);
  return 1.0f / sqrtf(ss / cols + eps);
}

// rope.cu rotate(): rotated pair r of src -> (xi, ox), (yi, oy).  No __restrict__: q may be
// rotated in place (src == dst).
template <bool kNeox>
__device__ __forceinline__ void rope_pair(const bf16* src, const bf16* cos_ptr, const bf16* sin_ptr, int r,
                                          int& xi, int& yi, bf16& ox, bf16& oy) {
  constexpr int embed_dim = kRot / 2;
  bf16 cos, sin;
  if (kNeox) {
    xi = r;
    yi = embed_dim + r;
    cos = cos_ptr[xi];
    sin = sin_ptr[xi];
  } else {
    xi = 2 * r;
    yi = 2 * r + 1;
    cos = cos_ptr[xi / 2];
    sin = sin_ptr[xi / 2];
  }
  const bf16 x = src[xi];
  const bf16 y = src[yi];
  ox = x * cos - y * sin;
  oy = y * cos + x * sin;
}

template <typename pos_t, bool kNeox, int RSQRT>
__global__ __launch_bounds__(kThreads) void rope_qout_kv_store_kernel(
    const pos_t* __restrict__ positions, const bf16* q, bf16* qo, int nq, int64_t q_tok, int64_t q_head,
    int64_t qo_tok, int64_t qo_head, uint16_t* __restrict__ kv_c, int64_t kv_tok, const uint16_t* __restrict__ gamma,
    float norm_eps, bf16* __restrict__ k_pe, int64_t k_tok, const bf16* __restrict__ cache,
    uint8_t* __restrict__ kv_buffer, int64_t slot_stride, const int64_t* __restrict__ loc, int n_loc,
    float store_eps) {
  const int64_t t = blockIdx.x;
  const bf16* cos_ptr = cache + static_cast<int64_t>(positions[t]) * kRot;
  const bf16* sin_ptr = cos_ptr + kRot / 2;
  const int tid = threadIdx.x;
  int xi, yi;
  bf16 ox, oy;

  if (blockIdx.y + 1 < gridDim.y) {  // q heads
    const int h = blockIdx.y * kHeadsPerCta + tid / (kRot / 2);
    if (h < nq) {
      rope_pair<kNeox>(q + t * q_tok + h * q_head, cos_ptr, sin_ptr, tid % (kRot / 2), xi, yi, ox, oy);
      bf16* dst = qo + t * qo_tok + h * qo_head;
      dst[xi] = ox;
      dst[yi] = oy;
    }
    return;
  }

  const bool store = t < n_loc;
  uint8_t* slot = store ? kv_buffer + loc[t] * slot_stride : nullptr;
  if (tid < 64) {
    const int lane = tid;
    uint16_t* row = kv_c + t * kv_tok + lane * kVec;
    const uint4 rx = *reinterpret_cast<const uint4*>(row);
    const uint4 rw = *reinterpret_cast<const uint4*>(gamma + lane * kVec);
    const uint16_t* hx = reinterpret_cast<const uint16_t*>(&rx);
    const uint16_t* hw = reinterpret_cast<const uint16_t*>(&rw);
    float x[kVec];
    float sum = 0.0f;
#pragma unroll
    for (int ii = 0; ii < kVec; ++ii) {
      x[ii] = bf16_to_float(hx[ii]);
      sum += x[ii] * x[ii];
    }
#pragma unroll
    for (int o = 32; o > 0; o >>= 1) sum += __shfl_down(sum, o);
    float rstd = 0.0f;
    if (lane == 0) rstd = rms_rstd<RSQRT>(sum, kNope, norm_eps);
    rstd = __shfl(rstd, 0);

    uint4 ry;
    uint16_t* hy = reinterpret_cast<uint16_t*>(&ry);
    float y[kVec];
    float amax = 0.0f;
#pragma unroll
    for (int ii = 0; ii < kVec; ++ii) {
      hy[ii] = float_to_bf16(x[ii] * rstd * bf16_to_float(hw[ii]));
      y[ii] = bf16_to_float(hy[ii]);
      amax = fmaxf(amax, fabsf(y[ii]));
    }
    *reinterpret_cast<uint4*>(row) = ry;
    // 128-element groups = 16 lanes
#pragma unroll
    for (int o = 8; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor(amax, o));
    if (store) {
      amax = fmaxf(amax, store_eps);
      const float scale = amax / 448.0f;
      const float inv = scale == 0.0f ? 0.0f : 1.0f / scale;
      if ((lane & 15) == 0) reinterpret_cast<float*>(slot + kScaleOff)[lane >> 4] = scale;
      uint2 qv;
      qv.x = pack_fp8x4(y[0] * inv, y[1] * inv, y[2] * inv, y[3] * inv);
      qv.y = pack_fp8x4(y[4] * inv, y[5] * inv, y[6] * inv, y[7] * inv);
      *reinterpret_cast<uint2*>(slot + lane * kVec) = qv;
    }
  } else if (tid < 64 + kRot / 2) {
    bf16* kr = k_pe + t * k_tok;
    rope_pair<kNeox>(kr, cos_ptr, sin_ptr, tid - 64, xi, yi, ox, oy);
    kr[xi] = ox;
    kr[yi] = oy;
    if (store) {
      bf16* cr = reinterpret_cast<bf16*>(slot + kRopeOff);
      cr[xi] = ox;
      cr[yi] = oy;
    }
  }
}

bool aligned16(const void* p) { return reinterpret_cast<uintptr_t>(p) % 16 == 0; }

}  // namespace

// positions [T]; q_pe [T, nq, 64] -> q_out (same shape, unit last stride; may alias q_pe);
// kv_c [T, 512] and k_pe [T, 64] (or [T, 1, *]) bf16 row-strided views, normed / rotated in
// place; kv_buffer uint8 [slots, >= 656] (or any view whose dim 0 is the slot), loc int64.
void rope_qout_kv_store(const torch::Tensor& positions, const torch::Tensor& q_pe, torch::Tensor& q_out,
                        torch::Tensor& kv_c, const torch::Tensor& kv_weight, double norm_eps, torch::Tensor& k_pe,
                        const torch::Tensor& cos_sin_cache, bool is_neox, torch::Tensor& kv_buffer,
                        const torch::Tensor& loc, double store_eps, int64_t rsqrt_mode) {
  const int64_t T = positions.numel();
  TORCH_CHECK(positions.is_contiguous() &&
                  (positions.scalar_type() == at::kLong || positions.scalar_type() == at::kInt),
              "rope_qout_kv_store: positions must be contiguous int32/int64");
  TORCH_CHECK(q_pe.scalar_type() == at::kBFloat16 && q_out.scalar_type() == at::kBFloat16 &&
                  kv_c.scalar_type() == at::kBFloat16 && kv_weight.scalar_type() == at::kBFloat16 &&
                  k_pe.scalar_type() == at::kBFloat16 && cos_sin_cache.scalar_type() == at::kBFloat16,
              "rope_qout_kv_store: bf16 only");
  TORCH_CHECK(q_pe.dim() == 3 && q_pe.size(0) == T && q_pe.size(2) == kRot && q_pe.stride(2) == 1 &&
                  q_out.sizes() == q_pe.sizes() && q_out.stride(2) == 1,
              "rope_qout_kv_store: q_pe / q_out must be [T, nq, 64] with a unit last stride");
  TORCH_CHECK(cos_sin_cache.is_contiguous() && cos_sin_cache.dim() == 2 && cos_sin_cache.size(1) == kRot,
              "rope_qout_kv_store: cos_sin_cache must be a contiguous [max_pos, 64]");
  TORCH_CHECK(kv_c.numel() == T * kNope && kv_c.size(-1) == kNope && kv_c.stride(-1) == 1 &&
                  (kv_c.dim() == 2 || (kv_c.dim() == 3 && kv_c.size(1) == 1)),
              "rope_qout_kv_store: kv_c must be [T, 512] / [T, 1, 512] with a unit last stride");
  TORCH_CHECK(k_pe.numel() == T * kRot && k_pe.size(-1) == kRot && k_pe.stride(-1) == 1 &&
                  (k_pe.dim() == 2 || (k_pe.dim() == 3 && k_pe.size(1) == 1)),
              "rope_qout_kv_store: k_pe must be [T, 64] / [T, 1, 64] with a unit last stride");
  TORCH_CHECK(kv_weight.is_contiguous() && kv_weight.numel() == kNope, "rope_qout_kv_store: weight must be [512]");
  TORCH_CHECK(kv_buffer.scalar_type() == at::kByte && kv_buffer.dim() >= 2 &&
                  kv_buffer.numel() / kv_buffer.size(0) >= kSlotBytes,
              "rope_qout_kv_store: kv_buffer must be uint8 with >= 656 bytes a slot");
  TORCH_CHECK(loc.scalar_type() == at::kLong && loc.is_contiguous() && loc.numel() <= T,
              "rope_qout_kv_store: loc must be a contiguous int64 of at most T slots");
  const int64_t kv_tok = kv_c.stride(0), k_tok = k_pe.stride(0), slot_stride = kv_buffer.stride(0);
  TORCH_CHECK(aligned16(kv_c.data_ptr()) && kv_tok % 8 == 0 && aligned16(kv_weight.data_ptr()) &&
                  aligned16(kv_buffer.data_ptr()) && slot_stride % 16 == 0,
              "rope_qout_kv_store: kv_c rows / weight / kv_buffer slots not 16 B aligned");
  if (T == 0) return;
  const c10::cuda::CUDAGuard guard(q_pe.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int nq = q_pe.size(1);
  const dim3 grid(T, (nq + kHeadsPerCta - 1) / kHeadsPerCta + 1);
  const auto* qp = reinterpret_cast<const bf16*>(q_pe.data_ptr());
  auto* op = reinterpret_cast<bf16*>(q_out.data_ptr());
  auto* kvp = reinterpret_cast<uint16_t*>(kv_c.data_ptr());
  const auto* wp = reinterpret_cast<const uint16_t*>(kv_weight.data_ptr());
  auto* kp = reinterpret_cast<bf16*>(k_pe.data_ptr());
  const auto* cp = reinterpret_cast<const bf16*>(cos_sin_cache.data_ptr());
  auto* bp = kv_buffer.data_ptr<uint8_t>();
  const auto* lp = loc.data_ptr<int64_t>();
  const int n_loc = static_cast<int>(loc.numel());
#define HCU_KVS(P, N, R)                                                                                    \
  rope_qout_kv_store_kernel<P, N, R><<<grid, kThreads, 0, stream>>>(                                        \
      positions.data_ptr<P>(), qp, op, nq, q_pe.stride(0), q_pe.stride(1), q_out.stride(0), q_out.stride(1), \
      kvp, kv_tok, wp, float(norm_eps), kp, k_tok, cp, bp, slot_stride, lp, n_loc, float(store_eps))
#define HCU_KVS_R(P, N)              \
  switch (rsqrt_mode) {              \
    case 1: HCU_KVS(P, N, 1); break; \
    case 2: HCU_KVS(P, N, 2); break; \
    default: HCU_KVS(P, N, 0);       \
  }
  if (positions.scalar_type() == at::kLong) {
    if (is_neox) { HCU_KVS_R(int64_t, true) } else { HCU_KVS_R(int64_t, false) }
  } else {
    if (is_neox) { HCU_KVS_R(int32_t, true) } else { HCU_KVS_R(int32_t, false) }
  }
#undef HCU_KVS_R
#undef HCU_KVS
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
