// Skinny (decode) bf16 GEMM with an NN weight on gfx938: Y[M,N] = X[M,K] @ W[K,N], W
// row-major [K, N] (n contiguous) - the lm_head weight after SGLANG_HCU_LM_HEAD_NN's
// relayout (lm_head.weight.T) and the draft vocab-parallel shard (weight_shard_t).
//
// mmac wants each lane's A operand as 4 consecutive k of one output feature n, but NN
// rows hold n contiguously.  Instead of a cross-lane transpose, every lane loads 4 k rows
// of the same 8 n (16 B each) and builds the 8 A operands (n = 8r + j, j < 8) with
// v_perm in registers:
//   lane l (g = l/16, r = l%16) loads W[kb + 4H*g + i][n0 + 8r .. +8], i < 4H,
// so one wave instruction reads 4 rows x 256 B (the pattern that reaches ~1.6-1.8 TB/s on
// BW1100, probe_bw2.hip; 16 rows x 64 B caps at ~1.1).  mmac j (k slice h) then covers
// the 16 features n0 + 8r + j with lane group g supplying k = kb + 4H*g + 4h .. +4 and X
// (B operand, token m = r) the same k from one 8 * H byte load.  Result element e of lane
// l is C[n0 + 8r + j][m = e*4 + g], so the 8 accumulators of a lane are 8 consecutive n
// of one token row: one 16 B bf16 store each.
//
// A CTA owns 128 n; its NW waves split K (chunks of 16H k rows, PF chunks prefetched) and
// reduce through LDS in a fixed order (deterministic).  N % 8 == 0 (the last 128-n tile
// may be partial), M <= 16 * MT (MT <= 4: M <= 64).
#include <ATen/hip/HIPContext.h>
#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>
#include <torch/extension.h>

namespace {

using floatx4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;
using intx2 = __attribute__((__vector_size__(2 * sizeof(int)))) int;
using shortx4 = __attribute__((__vector_size__(4 * sizeof(short)))) short;
using uintx2 = __attribute__((__vector_size__(2 * sizeof(unsigned)))) unsigned;
using uintx4 = __attribute__((__vector_size__(4 * sizeof(unsigned)))) unsigned;

__device__ __forceinline__ uintx4 ld16(const uint8_t* p, bool stream) {
  return stream ? __builtin_nontemporal_load(reinterpret_cast<const uintx4*>(p))
                : *reinterpret_cast<const uintx4*>(p);
}

__device__ __forceinline__ unsigned bf16x2(float lo, float hi) {
  const unsigned short a = __builtin_bit_cast(unsigned short, __float2bfloat16(lo));
  const unsigned short b = __builtin_bit_cast(unsigned short, __float2bfloat16(hi));
  return a | ((unsigned)b << 16);
}

// H: k slices (of 4 rows) per lane per chunk; X dwords per chunk = 2H
template <int H, int MT>
struct NNChunk {
  uintx4 w[4 * H];
  unsigned x[MT][2 * H];
};

template <int NW, int MT, int H, int PF>
__global__ __launch_bounds__(NW * 64) void skinny_nn_kernel(
    const uint8_t* __restrict__ W, const uint8_t* __restrict__ X, __hip_bfloat16* __restrict__ Y,
    int M, int N, int K, int ldw, int ldx, int ldy, bool stream_w) {
  constexpr int S = PF + 1, KC = 16 * H;  // k rows per chunk
  const int lane = threadIdx.x & 63, w = threadIdx.x >> 6;
  const int r = lane & 15, g = lane >> 4;
  const int n = blockIdx.x * 128 + 8 * r;
  const bool nok = n < N;
  const int chunks = K / KC;
  const int per = (chunks + NW - 1) / NW;
  const int c0 = min(w * per, chunks), c1 = min(c0 + per, chunks);

  // lanes past N / M read row 0 / column 0 instead: their A rows / B columns only feed
  // outputs that are never stored
  const uint8_t* wp = W + (size_t)(4 * H * g) * ldw + (size_t)(nok ? n : 0) * 2;
  const uint8_t* xp[MT];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt) {
    const int m = mt * 16 + r;
    xp[mt] = X + (size_t)(m < M ? m : 0) * ldx + g * 8 * H;
  }
  floatx4 acc[MT][8];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int j = 0; j < 8; ++j) acc[mt][j] = floatx4{0, 0, 0, 0};

  auto load = [&](NNChunk<H, MT>& b, int c) {
    const int kb = c * KC;
#pragma unroll
    for (int i = 0; i < 4 * H; ++i) b.w[i] = ld16(wp + (size_t)(kb + i) * ldw, stream_w);
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
      if constexpr (H == 2) {
        const uintx4 v = *reinterpret_cast<const uintx4*>(xp[mt] + kb * 2);
#pragma unroll
        for (int d = 0; d < 4; ++d) b.x[mt][d] = v[d];
      } else {
        const uintx2 v = *reinterpret_cast<const uintx2*>(xp[mt] + kb * 2);
        b.x[mt][0] = v[0];
        b.x[mt][1] = v[1];
      }
    }
  };
  auto consume = [&](const NNChunk<H, MT>& b) {
#pragma unroll
    for (int h = 0; h < H; ++h) {
      shortx4 xb[MT];
#pragma unroll
      for (int mt = 0; mt < MT; ++mt)
        xb[mt] = __builtin_bit_cast(shortx4, intx2{(int)b.x[mt][2 * h], (int)b.x[mt][2 * h + 1]});
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        // (row 4h + 0/1 | 2/3) halves of n = 8r + j: low (j even) or high (j odd) bf16
        const unsigned sel = (j & 1) ? 0x07060302u : 0x05040100u;
        const int d = j >> 1;
        const unsigned lo = __builtin_amdgcn_perm(b.w[4 * h + 1][d], b.w[4 * h][d], sel);
        const unsigned hi = __builtin_amdgcn_perm(b.w[4 * h + 3][d], b.w[4 * h + 2][d], sel);
        const shortx4 a = __builtin_bit_cast(shortx4, intx2{(int)lo, (int)hi});
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
          acc[mt][j] = __builtin_hcu_mmac_f32_16x16x16_bf16_lit_lts(a, xb[mt], acc[mt][j], false, false);
      }
    }
  };

  NNChunk<H, MT> buf[S];
#pragma unroll
  for (int p = 0; p < PF; ++p)
    if (c0 + p < c1) load(buf[p], c0 + p);
  for (int c = c0; c < c1; c += S) {
#pragma unroll
    for (int s = 0; s < S; ++s) {
      if (c + s + PF < c1) load(buf[(s + PF) % S], c + s + PF);
      if (c + s < c1) consume(buf[s]);
    }
  }

  if constexpr (NW > 1) {
    // one pass per (mt, 4 of the 8 j): red[w][j][e][lane] holds the waves' partial sums
    // (NW * 4 KB of LDS), summed over w in order; each item stores 4 bf16 (8 B)
    __shared__ float red[NW * 4 * 4 * 64];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int jp = 0; jp < 2; ++jp) {
        if (mt | jp) __syncthreads();
#pragma unroll
        for (int jj = 0; jj < 4; ++jj)
#pragma unroll
          for (int e = 0; e < 4; ++e) red[((w * 4 + jj) * 4 + e) * 64 + lane] = acc[mt][jp * 4 + jj][e];
        __syncthreads();
        for (int i = threadIdx.x; i < 4 * 64; i += NW * 64) {
          const int l2 = i & 63, e = i >> 6;
          float v[4];
#pragma unroll
          for (int jj = 0; jj < 4; ++jj) {
            float s = red[(jj * 4 + e) * 64 + l2];
#pragma unroll
            for (int ww = 1; ww < NW; ++ww) s += red[((ww * 4 + jj) * 4 + e) * 64 + l2];
            v[jj] = s;
          }
          const int m = mt * 16 + e * 4 + (l2 >> 4), nn = blockIdx.x * 128 + 8 * (l2 & 15);
          if (m < M && nn < N)
            *reinterpret_cast<uintx2*>(Y + (size_t)m * ldy + nn + 4 * jp) =
                uintx2{bf16x2(v[0], v[1]), bf16x2(v[2], v[3])};
        }
      }
  } else {
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int m = mt * 16 + e * 4 + g;
        const floatx4* a = acc[mt];
        if (m < M && nok)
          *reinterpret_cast<uintx4*>(Y + (size_t)m * ldy + n) =
              uintx4{bf16x2(a[0][e], a[1][e]), bf16x2(a[2][e], a[3][e]), bf16x2(a[4][e], a[5][e]),
                     bf16x2(a[6][e], a[7][e])};
      }
  }
}

template <int NW, int MT, int H, int PF>
void nn_launch(const torch::Tensor& x, const torch::Tensor& w, torch::Tensor& y, bool stream_w, hipStream_t st) {
  const int M = x.size(0), K = x.size(1), N = w.size(1);
  skinny_nn_kernel<NW, MT, H, PF><<<(N + 127) / 128, NW * 64, 0, st>>>(
      (const uint8_t*)w.data_ptr(), (const uint8_t*)x.data_ptr(), (__hip_bfloat16*)y.data_ptr(), M, N, K,
      w.stride(0) * 2, x.stride(0) * 2, y.stride(0), stream_w);
}

template <int NW, int MT>
void nn_launch_hp(int h, int pf, const torch::Tensor& x, const torch::Tensor& w, torch::Tensor& y, bool stream_w,
                  hipStream_t st) {
  if (h == 2) {
    if (pf == 1) nn_launch<NW, MT, 2, 1>(x, w, y, stream_w, st);
    else nn_launch<NW, MT, 2, 2>(x, w, y, stream_w, st);
  } else {
    if (pf == 1) nn_launch<NW, MT, 1, 1>(x, w, y, stream_w, st);
    else if (pf == 2) nn_launch<NW, MT, 1, 2>(x, w, y, stream_w, st);
    else nn_launch<NW, MT, 1, 3>(x, w, y, stream_w, st);
  }
}

template <int NW>
void nn_launch_m(int mt, int h, int pf, const torch::Tensor& x, const torch::Tensor& w, torch::Tensor& y,
                 bool stream_w, hipStream_t st) {
  if (mt == 1) nn_launch_hp<NW, 1>(h, pf, x, w, y, stream_w, st);
  else if (mt == 2) nn_launch_hp<NW, 2>(h, pf, x, w, y, stream_w, st);
  else if (mt == 3) nn_launch_hp<NW, 3>(h, pf, x, w, y, stream_w, st);
  else nn_launch_hp<NW, 4>(h, pf, x, w, y, stream_w, st);
}

}  // namespace

void skinny_nn(const torch::Tensor& x, const torch::Tensor& w, torch::Tensor& y, int64_t nw, int64_t h, int64_t pf,
               bool stream_w) {
  TORCH_CHECK(x.scalar_type() == at::ScalarType::BFloat16 && w.scalar_type() == at::ScalarType::BFloat16 &&
                  y.scalar_type() == at::ScalarType::BFloat16, "skinny_nn: bf16 only");
  TORCH_CHECK(x.dim() == 2 && w.dim() == 2 && y.dim() == 2, "skinny_nn: 2-D tensors expected");
  TORCH_CHECK(x.stride(1) == 1 && w.stride(1) == 1 && y.stride(1) == 1, "skinny_nn: inner dim must be contiguous");
  const int M = x.size(0), K = x.size(1), N = w.size(1);
  TORCH_CHECK(w.size(0) == K && y.size(0) == M && y.size(1) == N, "skinny_nn: shape mismatch");
  TORCH_CHECK(M >= 1 && M <= 64, "skinny_nn: M must be in [1, 64]");
  TORCH_CHECK(h == 1 || h == 2, "skinny_nn: h must be 1/2");
  TORCH_CHECK(pf >= 1 && pf <= (h == 1 ? 3 : 2), "skinny_nn: pf must be 1..3 (h 1) / 1..2 (h 2)");
  TORCH_CHECK(N % 8 == 0 && K % (16 * h) == 0, "skinny_nn: N % 8 and K % (16 h) must be 0");
  TORCH_CHECK((x.stride(0) | w.stride(0) | y.stride(0)) % 8 == 0, "skinny_nn: rows must be 16 B aligned");
  TORCH_CHECK(((uintptr_t)x.data_ptr() | (uintptr_t)w.data_ptr() | (uintptr_t)y.data_ptr()) % 16 == 0,
              "skinny_nn: 16 B alignment");
  const int mt = (M + 15) / 16;
  hipStream_t st = at::hip::getCurrentHIPStream();
  switch (nw) {
    case 1: nn_launch_m<1>(mt, h, pf, x, w, y, stream_w, st); break;
    case 2: nn_launch_m<2>(mt, h, pf, x, w, y, stream_w, st); break;
    case 4: nn_launch_m<4>(mt, h, pf, x, w, y, stream_w, st); break;
    case 8: nn_launch_m<8>(mt, h, pf, x, w, y, stream_w, st); break;
    default: TORCH_CHECK(false, "skinny_nn: nw must be 1/2/4/8");
  }
}
