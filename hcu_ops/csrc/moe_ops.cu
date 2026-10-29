// MoE router for sigmoid + correction-bias top-k over a single expert group (GLM-5.x,
// Kimi-K2 style): a drop-in for lightop moe_fused_gate (moe_fused_gate_kernel_gourp1,
// same numerics and tie-breaking) fused with sglang's HCU DeepEP id postprocess (rows
// >= num_token_non_padded get id -1, ids are int64), which otherwise is a separate
// compiled kernel per MoE layer.
//
// 16 lanes (one DPP row) per token, E / 16 experts per lane.  Each of the topk rounds
// takes the lane-local best (strict >, so the lowest index among equal scores) and
// max-reduces (score, -index) keys over the row with DPP, no LDS round trips; the
// winner's entry is cleared with unrolled compares instead of a dynamic index (which
// put lightop's per-lane arrays in scratch).
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cfloat>
#include <cstdint>

namespace {

constexpr int kLanes = 16;
constexpr int kRowsPerBlock = 4;              // one wave per block
constexpr float kLog2e = 1.44269504089f;      // lightop exp_func on gfx938

// bf16 is passed as uint16_t; conversions to float are exact
template <typename T, int N>
__device__ __forceinline__ void load_float(const T* __restrict__ p, float (&v)[N]) {
  constexpr int kChunks = N * sizeof(T) / 16;
  static_assert(N * sizeof(T) % 16 == 0, "16 B aligned lane slices only");
  uint4 raw[kChunks];
#pragma unroll
  for (int i = 0; i < kChunks; ++i) raw[i] = reinterpret_cast<const uint4*>(p)[i];
  if constexpr (sizeof(T) == 4) {
    const float* f = reinterpret_cast<const float*>(raw);
#pragma unroll
    for (int i = 0; i < N; ++i) v[i] = f[i];
  } else {
    const uint16_t* h = reinterpret_cast<const uint16_t*>(raw);
#pragma unroll
    for (int i = 0; i < N; ++i) v[i] = __uint_as_float(uint32_t(h[i]) << 16);
  }
}

// order-preserving float -> uint32 (+0 and -0 compare equal in lightop, so tie them too)
__device__ __forceinline__ uint32_t ord_key(float f) {
  if (f == 0.0f) f = 0.0f;
  const uint32_t u = __float_as_uint(f);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

template <int Ctrl>
__device__ __forceinline__ uint32_t dpp(uint32_t v) {
  return static_cast<uint32_t>(__builtin_amdgcn_update_dpp(0, static_cast<int>(v), Ctrl, 0xF, 0xF, false));
}

// (hi, lo) = max over the pair of lanes Ctrl connects, carrying the winner's prob along
template <int Ctrl>
__device__ __forceinline__ void max_step(uint32_t& hi, uint32_t& lo, float& p) {
  const uint32_t ohi = dpp<Ctrl>(hi), olo = dpp<Ctrl>(lo);
  const float op = __uint_as_float(dpp<Ctrl>(__float_as_uint(p)));
  const uint64_t mine = (uint64_t(hi) << 32) | lo, other = (uint64_t(ohi) << 32) | olo;
  if (other > mine) {
    hi = ohi;
    lo = olo;
    p = op;
  }
}

template <typename TL, typename TB, int E, int K>
__global__ __launch_bounds__(kLanes * kRowsPerBlock) void moe_gate_topk_kernel(
    const TL* __restrict__ logits, const TB* __restrict__ bias, float* __restrict__ weights,
    int64_t* __restrict__ ids, int64_t rows, float route_scale,
    const void* __restrict__ non_padded, int non_padded_bytes) {
  constexpr int V = E / kLanes;
  const int lane = threadIdx.x % kLanes;
  const int64_t row = int64_t(blockIdx.x) * kRowsPerBlock + threadIdx.x / kLanes;
  // rows past the end recompute row 0 instead of exiting, so every DPP row stays full
  const int64_t r = row < rows ? row : 0;
  const int start = lane * V;

  float score[V], prob[V];
  {
    float x[V], b[V];
    load_float(logits + r * E + start, x);
    load_float(bias + start, b);
#pragma unroll
    for (int i = 0; i < V; ++i) {
      prob[i] = 1.0f / (1.0f + __builtin_amdgcn_exp2f(-x[i] * kLog2e));
      score[i] = b[i] + prob[i];
    }
  }

  float sum = 0.0f, my_prob = 0.0f;
  int my_id = 0;
#pragma unroll
  for (int k = 0; k < K; ++k) {
    float best = -FLT_MAX, best_prob = prob[0];
    int best_id = start;
#pragma unroll
    for (int i = 0; i < V; ++i) {
      if (score[i] > best) {
        best = score[i];
        best_id = start + i;
        best_prob = prob[i];
      }
    }
    uint32_t hi = ord_key(best), lo = ~uint32_t(best_id);
    max_step<0x128>(hi, lo, best_prob);  // row_ror:8
    max_step<0x124>(hi, lo, best_prob);  // row_ror:4
    max_step<0x4E>(hi, lo, best_prob);   // quad_perm [2,3,0,1]
    max_step<0xB1>(hi, lo, best_prob);   // quad_perm [1,0,3,2]
    const int winner = static_cast<int>(~lo);
    const int off = winner - start;
#pragma unroll
    for (int i = 0; i < V; ++i) score[i] = off == i ? -FLT_MAX : score[i];
    sum += best_prob;
    if (lane == k) {
      my_prob = best_prob;
      my_id = winner;
    }
  }

  if (row < rows && lane < K) {
    const int64_t o = row * K + lane;
    weights[o] = (my_prob / sum) * route_scale;
    bool pad = false;
    if (non_padded_bytes == 4) {
      pad = row >= *static_cast<const int32_t*>(non_padded);
    } else if (non_padded_bytes == 8) {
      pad = row >= *static_cast<const int64_t*>(non_padded);
    }
    ids[o] = pad ? -1 : my_id;
  }
}

template <typename TL, typename TB, int E>
void launch_e(const torch::Tensor& logits, const torch::Tensor& bias, torch::Tensor& w, torch::Tensor& ids,
              int64_t topk, float route_scale, const void* np, int np_bytes, cudaStream_t stream) {
  const int64_t rows = logits.size(0);
  const dim3 grid((rows + kRowsPerBlock - 1) / kRowsPerBlock), block(kLanes * kRowsPerBlock);
  const auto* x = reinterpret_cast<const TL*>(logits.data_ptr());
  const auto* b = reinterpret_cast<const TB*>(bias.data_ptr());
  float* wp = w.data_ptr<float>();
  int64_t* ip = ids.data_ptr<int64_t>();
  switch (topk) {
#define HCU_GATE_K(K)                                                                                    \
  case K:                                                                                                \
    moe_gate_topk_kernel<TL, TB, E, K><<<grid, block, 0, stream>>>(x, b, wp, ip, rows, route_scale, np, \
                                                                   np_bytes);                           \
    break;
    HCU_GATE_K(6)
    HCU_GATE_K(8)
#undef HCU_GATE_K
    default:
      TORCH_CHECK(false, "moe_gate_topk: unsupported topk ", topk);
  }
}

template <typename TL, typename TB>
void launch_t(const torch::Tensor& logits, const torch::Tensor& bias, torch::Tensor& w, torch::Tensor& ids,
              int64_t topk, float route_scale, const void* np, int np_bytes, cudaStream_t stream) {
  switch (logits.size(1)) {
    case 128: launch_e<TL, TB, 128>(logits, bias, w, ids, topk, route_scale, np, np_bytes, stream); break;
    case 256: launch_e<TL, TB, 256>(logits, bias, w, ids, topk, route_scale, np, np_bytes, stream); break;
    case 384: launch_e<TL, TB, 384>(logits, bias, w, ids, topk, route_scale, np, np_bytes, stream); break;
    case 512: launch_e<TL, TB, 512>(logits, bias, w, ids, topk, route_scale, np, np_bytes, stream); break;
    default: TORCH_CHECK(false, "moe_gate_topk: unsupported expert count ", logits.size(1));
  }
}

}  // namespace

std::vector<torch::Tensor> moe_gate_topk(const torch::Tensor& logits, const torch::Tensor& bias, int64_t topk,
                                         double routed_scaling_factor, bool apply_routed_scaling_factor_on_output,
                                         const c10::optional<torch::Tensor>& num_token_non_padded) {
  TORCH_CHECK(logits.dim() == 2 && logits.is_contiguous() && bias.dim() == 1 && bias.is_contiguous() &&
                  bias.numel() == logits.size(1),
              "moe_gate_topk: logits [T, E] and bias [E] must be contiguous");
  TORCH_CHECK(logits.scalar_type() == at::kFloat || logits.scalar_type() == at::kBFloat16,
              "moe_gate_topk: logits must be fp32 or bf16");
  TORCH_CHECK(bias.scalar_type() == at::kFloat || bias.scalar_type() == at::kBFloat16,
              "moe_gate_topk: bias must be fp32 or bf16");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(logits.data_ptr()) % 16 == 0 &&
                  reinterpret_cast<uintptr_t>(bias.data_ptr()) % 16 == 0,
              "moe_gate_topk: logits / bias must be 16 B aligned");
  const c10::cuda::CUDAGuard guard(logits.device());
  const int64_t rows = logits.size(0);
  auto w = torch::empty({rows, topk}, logits.options().dtype(torch::kFloat32));
  auto ids = torch::empty({rows, topk}, logits.options().dtype(torch::kInt64));
  if (rows == 0) return {w, ids};

  const void* np = nullptr;
  int np_bytes = 0;
  if (num_token_non_padded.has_value()) {
    const auto& t = *num_token_non_padded;
    TORCH_CHECK(t.numel() >= 1 && t.device() == logits.device() &&
                    (t.scalar_type() == at::kInt || t.scalar_type() == at::kLong),
                "moe_gate_topk: num_token_non_padded must be an int32 / int64 device scalar");
    np = t.data_ptr();
    np_bytes = t.element_size();
  }
  // lightop: scale on output only when asked and the factor is positive
  const float route_scale = apply_routed_scaling_factor_on_output && routed_scaling_factor > 0.0
                                ? static_cast<float>(routed_scaling_factor)
                                : 1.0f;
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const bool xb = logits.scalar_type() == at::kBFloat16, bb = bias.scalar_type() == at::kBFloat16;
  if (xb && bb) {
    launch_t<uint16_t, uint16_t>(logits, bias, w, ids, topk, route_scale, np, np_bytes, stream);
  } else if (xb) {
    launch_t<uint16_t, float>(logits, bias, w, ids, topk, route_scale, np, np_bytes, stream);
  } else if (bb) {
    launch_t<float, uint16_t>(logits, bias, w, ids, topk, route_scale, np, np_bytes, stream);
  } else {
    launch_t<float, float>(logits, bias, w, ids, topk, route_scale, np, np_bytes, stream);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {w, ids};
}

// Masked SwiGLU + per-token fp8 e4m3 quant between the two DeepGEMM masked grouped GEMMs of
// the DeepEP low-latency MoE: a drop-in for lightop fuse_silu_mul_fp8_quant_ep(x, 0,
// masked_m, limit=None) on the rows that matter (t < masked_m[e]; the other rows stay
// uninitialised in both).  Same numerics: bf16 silu with exp2, bf16 product, row abs-max,
// scale = max(amax / 448, 1 / (448 * 512)), q = cvt_pk_fp8(y * (1 / scale)).
// lightop launches E x 128 CTAs of 256 threads with one row each (in decode nearly all of
// them only read masked_m and exit; at large m CTAs also redo later experts' rows).  Here
// the valid rows are numbered globally through a prefix sum of masked_m and dealt out to a
// fixed grid, a row split over WPR waves of a 4-wave CTA (cross-wave max through LDS): the
// op is VALU-latency bound (~30 VALU per element, exp / rcp at quarter rate), so few rows
// are spread over 4 or 2 waves to reach every SIMD, many rows take one wave each (no
// cross-wave sync).  WPR 0 picks this per launch from the row count, so one config fits
// every batch size under CUDA graphs.
namespace {

constexpr int kActMaxExperts = 256;

// c10::BFloat16(float): round to nearest even, NaN -> 0x7FC0; returned widened to float
__device__ __forceinline__ float bf16_round(float f) {
  if (f != f) return __uint_as_float(0x7FC00000u);
  const uint32_t u = __float_as_uint(f);
  return __uint_as_float((u + 0x7FFFu + ((u >> 16) & 1u)) & 0xFFFF0000u);
}

// gfx938 v_cvt_bf16_f32: one VALU for bf16_round's three, the same bits for every non-NaN
// f32 (test_moe_ops.py checks all 2^32 with bf16_round_check), any NaN for a NaN
__device__ __forceinline__ float bf16_round_hw(float f) {
  return __uint_as_float(uint32_t(static_cast<uint16_t>(__builtin_hcu_cvt_bf16_f32(f, false, false))) << 16);
}

// lightop silu_kernel<bf16>: bf16(x / (1 + 2^(-x log2 e))).  A NaN here only feeds a NaN
// product, which bf16_round turns into 0x7FC0 as in lightop, so hardware rounding is exact.
// (x * rcp(den), with or without a Newton step, misrounds a few of the 65536 bf16 inputs:
// -0, -87.5 .. -88.5 where rcp flushes, 5.9375 for plain rcp; and it only pays off for
// thousands of rows, so the IEEE division stays.)
__device__ __forceinline__ float silu_bf16(float x) {
  return bf16_round_hw(x / (1.0f + __builtin_amdgcn_exp2f(-x * kLog2e)));
}

__device__ __forceinline__ uint32_t pack_fp8x4(float a, float b, float c, float d) {
  int32_t r = 0;
  r = __builtin_hcu_cvt_pk_fp8_f32(a, b, r, false);
  r = __builtin_hcu_cvt_pk_fp8_f32(c, d, r, true);
  return static_cast<uint32_t>(r);
}

// The CTA's share of rows [0, total), row r of expert e where s_end[e - 1] <= r < s_end[e].
// d = NJW * WPR * 512.  Wave w of a row covers elements [w * NJW * 512, (w + 1) * NJW * 512),
// lane l elements j * 512 + l * 8 + [0, 8) of that, gate and up halves alike.
template <int NJW, int WPR>
__device__ __forceinline__ void act_rows(const uint16_t* __restrict__ in, uint8_t* __restrict__ out,
                                         float* __restrict__ scales, const int* s_end, float (*s_max)[4],
                                         int num_experts, int cap, int total) {
  constexpr int D = NJW * WPR * 512, RPB = 4 / WPR;
  const int wave = threadIdx.x / 64, lane = threadIdx.x % 64, part = wave % WPR;
  int it = 0;
  for (int r0 = blockIdx.x * RPB; r0 < total; r0 += gridDim.x * RPB, ++it) {
    const int r = r0 + wave / WPR;
    const bool active = r < total;  // uniform per wave
    int64_t row = 0;
    float y[NJW][8];
    float amax = 0.0f;
    if (active) {
      int e = 0;
      for (int c = 0; c < num_experts; c += 64) {
        e += __popcll(__ballot(c + lane < num_experts && s_end[c + lane] <= r));
      }
      row = int64_t(e) * cap + (r - (e ? s_end[e - 1] : 0));
      const uint16_t* src = in + row * (2 * D) + part * (NJW * 512) + lane * 8;
      uint4 rg[NJW], ru[NJW];
#pragma unroll
      for (int j = 0; j < NJW; ++j) {
        rg[j] = *reinterpret_cast<const uint4*>(src + j * 512);
        ru[j] = *reinterpret_cast<const uint4*>(src + D + j * 512);
      }
#pragma unroll
      for (int j = 0; j < NJW; ++j) {
        const uint16_t* hg = reinterpret_cast<const uint16_t*>(&rg[j]);
        const uint16_t* hu = reinterpret_cast<const uint16_t*>(&ru[j]);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          const float x = __uint_as_float(uint32_t(hg[i]) << 16);
          const float u = __uint_as_float(uint32_t(hu[i]) << 16);
          y[j][i] = bf16_round(silu_bf16(x) * u);
          amax = fmaxf(amax, fabsf(y[j][i]));  // fmaxf skips NaN, as lightop's max does
        }
      }
#pragma unroll
      for (int o = 32; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor(amax, o));
    }
    if constexpr (WPR > 1) {
      if (lane == 0) s_max[it & 1][wave] = amax;
      __syncthreads();
#pragma unroll
      for (int k = 0; k < WPR; ++k) amax = fmaxf(amax, s_max[it & 1][wave - part + k]);
    }
    if (active) {
      const float scale = fmaxf(amax / 448.0f, 1.0f / (448.0f * 512.0f));
      const float inv = 1.0f / scale;
      if (part == 0 && lane == 0) scales[row] = scale;
      uint8_t* dst = out + row * D + part * (NJW * 512) + lane * 8;
#pragma unroll
      for (int j = 0; j < NJW; ++j) {
        uint2 q;
        q.x = pack_fp8x4(y[j][0] * inv, y[j][1] * inv, y[j][2] * inv, y[j][3] * inv);
        q.y = pack_fp8x4(y[j][4] * inv, y[j][5] * inv, y[j][6] * inv, y[j][7] * inv);
        *reinterpret_cast<uint2*>(dst + j * 512) = q;
      }
    }
  }
}

// d = NJ * 512.  WPR 0: the split with the least work per SIMD, rows * w waves (1 / w of a row
// each) over 4 * CUs SIMDs, the smaller w on ties (fewer syncs); CTA-uniform.  (Holding all three
// row loops it takes 64 VGPRs at d = 2048; capping that at 48 spills and only slows the 1-wave
// loop, and below ~200 rows a CU runs at most 3 CTAs anyway.)
template <int NJ, int WPR>
__global__ __launch_bounds__(256) void silu_mul_fp8_quant_masked_kernel(
    const uint16_t* __restrict__ in, uint8_t* __restrict__ out, float* __restrict__ scales,
    const int32_t* __restrict__ masked_m, int num_experts, int cap, float inv_simds) {
  __shared__ int s_end[kActMaxExperts];  // inclusive prefix sums of the clamped row counts
  __shared__ float s_max[2][4];
  if (threadIdx.x < 64) {
    const int lane = threadIdx.x;
    int base = 0;
    for (int c = 0; c < num_experts; c += 64) {
      int v = c + lane < num_experts ? min(max(masked_m[c + lane], 0), cap) : 0;
#pragma unroll
      for (int o = 1; o < 64; o <<= 1) {
        const int n = __shfl_up(v, o);
        if (lane >= o) v += n;
      }
      s_end[c + lane] = base + v;
      base += __shfl(v, 63);
    }
  }
  __syncthreads();
  const int total = s_end[num_experts - 1];
  if (int(blockIdx.x) >= total) return;  // no row for this CTA at any split
  if constexpr (WPR != 0) {
    act_rows<NJ / WPR, WPR>(in, out, scales, s_end, s_max, num_experts, cap, total);
  } else {
    // quarter rows per SIMD (float: an integer division costs a long scalar chain; a choice
    // off by rounding is only slower, never wrong)
    const float q = total * inv_simds - 1e-4f;
    const float l1 = 4.0f * ceilf(q), l2 = 2.0f * ceilf(2.0f * q), l4 = ceilf(4.0f * q);
    if constexpr (NJ % 4 == 0) {
      if (l4 < l2 && l4 < l1) return act_rows<NJ / 4, 4>(in, out, scales, s_end, s_max, num_experts, cap, total);
    }
    if constexpr (NJ % 2 == 0) {
      if (l2 < l1) return act_rows<NJ / 2, 2>(in, out, scales, s_end, s_max, num_experts, cap, total);
    }
    act_rows<NJ, 1>(in, out, scales, s_end, s_max, num_experts, cap, total);
  }
}

// bf16_round_hw vs bf16_round on every f32 (NaN inputs: NaN-ness only);
// res = {mismatches, smallest mismatching input}
__global__ void bf16_round_check_kernel(unsigned long long* __restrict__ res) {
  unsigned long long bad = 0, first = ~0ull;
  for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (1ull << 32); i += gridDim.x * blockDim.x) {
    const float f = __uint_as_float(uint32_t(i));
    const float h = bf16_round_hw(f);
    const bool ok = f != f ? h != h : __float_as_uint(h) == __float_as_uint(bf16_round(f));
    if (!ok) {
      ++bad;
      first = min(first, i);
    }
  }
  if (bad) {
    atomicAdd(&res[0], bad);
    atomicMin(&res[1], first);
  }
}

template <int NJ>
void launch_act(int64_t wpr, dim3 grid, cudaStream_t stream, const uint16_t* xp, uint8_t* op, float* sp,
                const int32_t* mp, int E, int cap, float inv_simds) {
  TORCH_CHECK(NJ % std::max<int64_t>(wpr, 1) == 0, "silu_mul_fp8_quant_masked: d / 512 = ", NJ,
              " not divisible by ", wpr, " waves per row");
  switch (wpr) {
#define HCU_ACT_WPR(W)                                                                                \
  case W:                                                                                             \
    if constexpr (W == 0 || NJ % W == 0)                                                              \
      silu_mul_fp8_quant_masked_kernel<NJ, W><<<grid, 256, 0, stream>>>(xp, op, sp, mp, E, cap, inv_simds); \
    break;
    HCU_ACT_WPR(0)
    HCU_ACT_WPR(1)
    HCU_ACT_WPR(2)
    HCU_ACT_WPR(4)
#undef HCU_ACT_WPR
  }
}

}  // namespace

std::vector<torch::Tensor> silu_mul_fp8_quant_masked(const torch::Tensor& x, const torch::Tensor& masked_m,
                                                     int64_t ctas, int64_t waves_per_row) {
  TORCH_CHECK(x.dim() == 3 && x.is_contiguous() && x.scalar_type() == at::kBFloat16,
              "silu_mul_fp8_quant_masked: x must be a contiguous bf16 [E, cap, 2 * d]");
  const int64_t E = x.size(0), cap = x.size(1), d = x.size(2) / 2;
  TORCH_CHECK(d % 512 == 0 && d >= 512 && d <= 4096 && x.size(2) == 2 * d,
              "silu_mul_fp8_quant_masked: d must be a multiple of 512 up to 4096, got ", d);
  TORCH_CHECK(E <= kActMaxExperts && E * cap < (int64_t(1) << 31),
              "silu_mul_fp8_quant_masked: at most 256 experts and 2^31 rows");
  TORCH_CHECK(masked_m.dim() == 1 && masked_m.numel() == E && masked_m.scalar_type() == at::kInt &&
                  masked_m.is_contiguous() && masked_m.device() == x.device(),
              "silu_mul_fp8_quant_masked: masked_m must be int32 [E] on the same device");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0, "silu_mul_fp8_quant_masked: x not 16 B aligned");
  TORCH_CHECK(ctas >= 1 && ctas <= 65535 &&
                  (waves_per_row == 0 || waves_per_row == 1 || waves_per_row == 2 || waves_per_row == 4),
              "silu_mul_fp8_quant_masked: bad launch shape");
  const c10::cuda::CUDAGuard guard(x.device());
  auto out = torch::empty({E, cap, d}, x.options().dtype(at::kFloat8_e4m3fn));
  auto scales = torch::empty({E, cap, 1}, x.options().dtype(torch::kFloat32));
  if (E == 0 || cap == 0) return {out, scales};
  // no more CTAs than row groups could ever need
  const int64_t rpb = waves_per_row ? 4 / waves_per_row : 1;
  const dim3 grid(std::min<int64_t>(ctas, (E * cap + rpb - 1) / rpb));
  const float inv_simds = 1.0f / (4 * at::cuda::getCurrentDeviceProperties()->multiProcessorCount);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const auto* xp = reinterpret_cast<const uint16_t*>(x.data_ptr());
  auto* op = reinterpret_cast<uint8_t*>(out.data_ptr());
  float* sp = scales.data_ptr<float>();
  const int32_t* mp = masked_m.data_ptr<int32_t>();
  switch (d / 512) {
#define HCU_ACT_NJ(NJ)                                                                   \
  case NJ:                                                                               \
    launch_act<NJ>(waves_per_row, grid, stream, xp, op, sp, mp, int(E), int(cap), inv_simds); \
    break;
    HCU_ACT_NJ(1)
    HCU_ACT_NJ(2)
    HCU_ACT_NJ(3)
    HCU_ACT_NJ(4)
    HCU_ACT_NJ(5)
    HCU_ACT_NJ(6)
    HCU_ACT_NJ(7)
    HCU_ACT_NJ(8)
#undef HCU_ACT_NJ
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out, scales};
}

torch::Tensor bf16_round_check(const torch::Tensor& like) {
  const c10::cuda::CUDAGuard guard(like.device());
  auto res = torch::tensor({int64_t(0), int64_t(-1)}, torch::kInt64).to(like.device());
  bf16_round_check_kernel<<<4096, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<unsigned long long*>(res.data_ptr()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return res;
}
