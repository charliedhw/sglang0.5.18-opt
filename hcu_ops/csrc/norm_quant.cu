// Fusions of a bf16 producer with the per-token fp8 quant of the CHANNEL-fp8 linear that reads
// its output (compressed-tensors FP8_DYNAMIC: scaled_fp8_quant(x, use_per_token_if_dynamic)
// -> lightop per_token_quant_fp8), bit-exact with the unfused ops:
//
//  * fused_add_rms_quant: lightop op.fused_add_rms_norm_opt (in place: residual = x + residual,
//    x = rms_norm(residual) * w) + per_token_quant_fp8(x).  Same kernel shape as lightop's
//    fused_add_rms_kernel_opt<bf16, float, 8, BLOCK> (one CTA per token, 8 elements a thread,
//    BlockReduceSum_NEW tree: shfl_down 32..1, then wave 0 over the per-wave sums), so the
//    sum of squares, rsqrt and the bf16 results are the same bits.
//  * silu_mul_quant: lightop fuse_silu_and_mul (bf16 silu with exp2, bf16 product) +
//    per_token_quant_fp8, without the bf16 activation round trip.
//
// per_token_quant_fp8: amax = max |x| (fmaxf, order free), scale = amax / 448,
// q = cvt_pk_fp8(x * (amax == 0 ? 0 : 448 / amax)) four elements at a time.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAMathCompat.h>
#include <torch/extension.h>

#include <cstdint>

namespace {

constexpr int kVec = 8;
constexpr float kFp8Max = 448.0f;
constexpr float kLog2e = 1.44269504089f;  // lightop silu_kernel_opt

__device__ __forceinline__ float bf16_to_float(uint16_t h) { return __uint_as_float(uint32_t(h) << 16); }

// c10::BFloat16(float) on ROCm: round to nearest even, NaN -> 0x7FC0
__device__ __forceinline__ uint16_t float_to_bf16(float f) {
  if (f != f) return 0x7FC0;
  const uint32_t u = __float_as_uint(f);
  return static_cast<uint16_t>((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}

__device__ __forceinline__ float bf16_round(float f) { return bf16_to_float(float_to_bf16(f)); }

// gfx938 v_cvt_bf16_f32: bf16_round for every non-NaN input (bf16_round_check in moe_ops.cu)
__device__ __forceinline__ float bf16_round_hw(float f) {
  return __uint_as_float(uint32_t(static_cast<uint16_t>(__builtin_hcu_cvt_bf16_f32(f, false, false))) << 16);
}

__device__ __forceinline__ uint32_t pack_fp8x4(float a, float b, float c, float d) {
  int32_t r = 0;
  r = __builtin_hcu_cvt_pk_fp8_f32(a, b, r, false);
  r = __builtin_hcu_cvt_pk_fp8_f32(c, d, r, true);
  return static_cast<uint32_t>(r);
}

__device__ __forceinline__ uint2 quant8(const float (&y)[kVec], float inv) {
  uint2 q;
  q.x = pack_fp8x4(y[0] * inv, y[1] * inv, y[2] * inv, y[3] * inv);
  q.y = pack_fp8x4(y[4] * inv, y[5] * inv, y[6] * inv, y[7] * inv);
  return q;
}

// rsqrt(mean square + eps) as lightop computes it (RSQRT 0, c10::cuda::compat::rsqrt); the
// other variants are only for tests that pin down how lightop was built
template <int RSQRT>
__device__ __forceinline__ float rms_rstd(float ss, int cols, float eps) {
  if constexpr (RSQRT == 0) return c10::cuda::compat::rsqrt(ss / cols + eps);
  if constexpr (RSQRT == 1) return __builtin_amdgcn_rsqf(ss / cols + eps);
  return 1.0f / sqrtf(ss / cols + eps);
}

template <int BLOCK, int RSQRT>
__global__ __launch_bounds__(BLOCK) void add_rms_quant_kernel(uint16_t* __restrict__ x, uint16_t* __restrict__ residual,
                                                              const uint16_t* __restrict__ gamma,
                                                              uint8_t* __restrict__ q, float* __restrict__ scales,
                                                              int rows, int cols, float eps) {
  constexpr int kWaves = BLOCK / 64;
  __shared__ float s_sum[kWaves];
  __shared__ float s_max[kWaves];
  __shared__ float s_rstd;
  const int j = threadIdx.x, lane = j % 64, wid = j / 64;
  const int tcol = cols / kVec;
  const bool active = j < tcol;
  float w[kVec];
  if (active) {
    const uint4 rw = *reinterpret_cast<const uint4*>(gamma + j * kVec);
    const uint16_t* hw = reinterpret_cast<const uint16_t*>(&rw);
#pragma unroll
    for (int ii = 0; ii < kVec; ++ii) w[ii] = bf16_to_float(hw[ii]);
  }
  for (int i = blockIdx.x; i < rows; i += gridDim.x) {
    const int64_t idx = (int64_t(i) * tcol + j) * kVec;
    float sum[kVec];
    float val = 0.0f;
    if (active) {
      const uint4 rx = *reinterpret_cast<const uint4*>(x + idx);
      const uint4 rr = *reinterpret_cast<const uint4*>(residual + idx);
      const uint16_t* hx = reinterpret_cast<const uint16_t*>(&rx);
      const uint16_t* hr = reinterpret_cast<const uint16_t*>(&rr);
#pragma unroll
      for (int ii = 0; ii < kVec; ++ii) {
        sum[ii] = bf16_to_float(hx[ii]) + bf16_to_float(hr[ii]);
        val += sum[ii] * sum[ii];
      }
    }
    // lightop BlockReduceSum_NEW; only thread 0's total is used
#pragma unroll
    for (int o = 32; o > 0; o >>= 1) val += __shfl_down(val, o);
    if (lane == 0) s_sum[wid] = val;
    __syncthreads();
    if (wid == 0 && lane < kWaves) {
      val = s_sum[lane];
#pragma unroll
      for (int o = kWaves / 2; o > 0; o >>= 1) val += __shfl_down(val, o);
      if (lane == 0) s_rstd = rms_rstd<RSQRT>(val, cols, eps);
    }
    __syncthreads();
    const float rstd = s_rstd;

    float y[kVec];
    float amax = 0.0f;
    if (active) {
      uint4 ox, orr;
      uint16_t* px = reinterpret_cast<uint16_t*>(&ox);
      uint16_t* pr = reinterpret_cast<uint16_t*>(&orr);
#pragma unroll
      for (int ii = 0; ii < kVec; ++ii) {
        px[ii] = float_to_bf16(sum[ii] * rstd * w[ii]);
        pr[ii] = float_to_bf16(sum[ii]);
        y[ii] = bf16_to_float(px[ii]);
        amax = fmaxf(amax, fabsf(y[ii]));  // fmaxf skips NaN, as lightop's max does
      }
      *reinterpret_cast<uint4*>(x + idx) = ox;
      *reinterpret_cast<uint4*>(residual + idx) = orr;
    }
#pragma unroll
    for (int o = 32; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor(amax, o));
    if (lane == 0) s_max[wid] = amax;
    __syncthreads();  // also orders this token's s_sum / s_rstd reads before the next token's writes
#pragma unroll
    for (int k = 0; k < kWaves; ++k) amax = fmaxf(amax, s_max[k]);
    if (active) {
      const float inv = amax == 0.0f ? 0.0f : kFp8Max / amax;
      if (j == 0) scales[i] = amax / kFp8Max;
      *reinterpret_cast<uint2*>(q + idx) = quant8(y, inv);
    }
  }
}

// d = NJ * 2048 at most; thread t handles 8-element chunks c = j * 256 + t of gate and up
template <int NJ>
__global__ __launch_bounds__(256) void silu_mul_quant_kernel(const uint16_t* __restrict__ in,
                                                             uint8_t* __restrict__ q, float* __restrict__ scales,
                                                             int rows, int d) {
  __shared__ float s_max[2][4];
  const int t = threadIdx.x, lane = t % 64, wid = t / 64;
  const int chunks = d / kVec;
  int it = 0;
  for (int r = blockIdx.x; r < rows; r += gridDim.x, ++it) {
    const uint16_t* src = in + int64_t(r) * (2 * d);
    float y[NJ][kVec];
    float amax = 0.0f;
#pragma unroll
    for (int j = 0; j < NJ; ++j) {
      const int c = j * 256 + t;
      if (c < chunks) {
        const uint4 rg = *reinterpret_cast<const uint4*>(src + c * kVec);
        const uint4 ru = *reinterpret_cast<const uint4*>(src + d + c * kVec);
        const uint16_t* hg = reinterpret_cast<const uint16_t*>(&rg);
        const uint16_t* hu = reinterpret_cast<const uint16_t*>(&ru);
#pragma unroll
        for (int ii = 0; ii < kVec; ++ii) {
          const float g = bf16_to_float(hg[ii]);
          // bf16(silu) as lightop silu_kernel_opt; a NaN only feeds the product, which
          // bf16_round maps to 0x7FC0 like c10, so hardware rounding is exact here
          const float s = bf16_round_hw(g / (1.0f + __builtin_amdgcn_exp2f(-g * kLog2e)));
          y[j][ii] = bf16_round(s * bf16_to_float(hu[ii]));
          amax = fmaxf(amax, fabsf(y[j][ii]));
        }
      }
    }
#pragma unroll
    for (int o = 32; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor(amax, o));
    if (lane == 0) s_max[it & 1][wid] = amax;
    __syncthreads();
#pragma unroll
    for (int k = 0; k < 4; ++k) amax = fmaxf(amax, s_max[it & 1][k]);
    const float inv = amax == 0.0f ? 0.0f : kFp8Max / amax;
    if (t == 0) scales[r] = amax / kFp8Max;
    uint8_t* dst = q + int64_t(r) * d;
#pragma unroll
    for (int j = 0; j < NJ; ++j) {
      const int c = j * 256 + t;
      if (c < chunks) *reinterpret_cast<uint2*>(dst + c * kVec) = quant8(y[j], inv);
    }
  }
}

bool aligned16(const torch::Tensor& t) { return reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0; }

void check_quant_out(const torch::Tensor& q, const torch::Tensor& s, int64_t rows, int64_t cols, const char* op) {
  TORCH_CHECK(q.dim() == 2 && q.size(0) == rows && q.size(1) == cols && q.is_contiguous() && q.element_size() == 1,
              op, ": q must be a contiguous 1-byte [rows, cols] tensor");
  TORCH_CHECK(s.numel() == rows && s.is_contiguous() && s.scalar_type() == at::kFloat, op,
              ": s must be a contiguous fp32 tensor of rows elements");
  TORCH_CHECK(aligned16(q), op, ": q not 16 B aligned");
}

}  // namespace

// x, residual: bf16 [rows, H] contiguous, modified in place exactly like lightop
// op.fused_add_rms_norm_opt; q [rows, H] fp8 e4m3, s [rows] / [rows, 1] fp32 written.
// H must take lightop's fused_add_rms_kernel_opt<8, 512 | 1024> path (H % 16 == 0,
// 2048 < H < 8192, H != 4096; 4096 / 8192 use a tiled kernel with another reduction).
void fused_add_rms_quant(torch::Tensor& x, torch::Tensor& residual, const torch::Tensor& weight, double eps,
                         torch::Tensor& q, torch::Tensor& s, int64_t rsqrt_mode) {
  TORCH_CHECK(x.dim() == 2 && x.is_contiguous() && residual.is_contiguous() && weight.is_contiguous() &&
                  residual.sizes() == x.sizes() && weight.dim() == 1 && weight.numel() == x.size(1),
              "fused_add_rms_quant: x / residual [rows, H] and weight [H] must be contiguous");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && residual.scalar_type() == at::kBFloat16 &&
                  weight.scalar_type() == at::kBFloat16,
              "fused_add_rms_quant: bf16 only");
  const int64_t rows = x.size(0), cols = x.size(1);
  TORCH_CHECK(cols % 16 == 0 && cols > 2048 && cols < 8192 && cols != 4096,
              "fused_add_rms_quant: unsupported hidden size ", cols);
  TORCH_CHECK(aligned16(x) && aligned16(residual) && aligned16(weight), "fused_add_rms_quant: inputs not 16 B aligned");
  TORCH_CHECK(rows < (int64_t(1) << 31), "fused_add_rms_quant: too many rows");
  check_quant_out(q, s, rows, cols, "fused_add_rms_quant");
  if (rows == 0) return;
  const c10::cuda::CUDAGuard guard(x.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto* xp = reinterpret_cast<uint16_t*>(x.data_ptr());
  auto* rp = reinterpret_cast<uint16_t*>(residual.data_ptr());
  const auto* wp = reinterpret_cast<const uint16_t*>(weight.data_ptr());
  auto* qp = reinterpret_cast<uint8_t*>(q.data_ptr());
  float* sp = s.data_ptr<float>();
  const dim3 grid(std::min<int64_t>(rows, 1024));
  const float feps = static_cast<float>(eps);  // lightop: T_ACC eps = epsilon
#define HCU_ARQ(B, R) \
  add_rms_quant_kernel<B, R><<<grid, B, 0, stream>>>(xp, rp, wp, qp, sp, int(rows), int(cols), feps)
  if (cols > 4096) {
    switch (rsqrt_mode) {
      case 1: HCU_ARQ(1024, 1); break;
      case 2: HCU_ARQ(1024, 2); break;
      default: HCU_ARQ(1024, 0);
    }
  } else {
    switch (rsqrt_mode) {
      case 1: HCU_ARQ(512, 1); break;
      case 2: HCU_ARQ(512, 2); break;
      default: HCU_ARQ(512, 0);
    }
  }
#undef HCU_ARQ
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// gate_up: bf16 [rows, 2 d] contiguous (gate | up); q [rows, d] fp8 e4m3, s [rows] fp32
void silu_mul_quant(const torch::Tensor& gate_up, torch::Tensor& q, torch::Tensor& s) {
  TORCH_CHECK(gate_up.dim() == 2 && gate_up.is_contiguous() && gate_up.scalar_type() == at::kBFloat16,
              "silu_mul_quant: gate_up must be a contiguous bf16 [rows, 2 d]");
  const int64_t rows = gate_up.size(0), d = gate_up.size(1) / 2;
  TORCH_CHECK(gate_up.size(1) == 2 * d && d % kVec == 0 && d > 0 && d <= 8 * 2048,
              "silu_mul_quant: d must be a multiple of 8 up to 16384, got ", d);
  TORCH_CHECK(aligned16(gate_up), "silu_mul_quant: gate_up not 16 B aligned");
  TORCH_CHECK(rows < (int64_t(1) << 31), "silu_mul_quant: too many rows");
  check_quant_out(q, s, rows, d, "silu_mul_quant");
  if (rows == 0) return;
  const c10::cuda::CUDAGuard guard(gate_up.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const auto* ip = reinterpret_cast<const uint16_t*>(gate_up.data_ptr());
  auto* qp = reinterpret_cast<uint8_t*>(q.data_ptr());
  float* sp = s.data_ptr<float>();
  const dim3 grid(std::min<int64_t>(rows, 2048));
  switch ((d + 2047) / 2048) {
#define HCU_SMQ(NJ)                                                                            \
  case NJ:                                                                                     \
    silu_mul_quant_kernel<NJ><<<grid, 256, 0, stream>>>(ip, qp, sp, int(rows), int(d)); \
    break;
    HCU_SMQ(1)
    HCU_SMQ(2)
    HCU_SMQ(3)
    HCU_SMQ(4)
    HCU_SMQ(5)
    HCU_SMQ(6)
    HCU_SMQ(7)
    HCU_SMQ(8)
#undef HCU_SMQ
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
