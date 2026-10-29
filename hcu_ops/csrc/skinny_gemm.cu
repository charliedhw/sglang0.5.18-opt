// Skinny (decode) dense GEMM for GLM-5.3 on gfx938: Y[M,N] = X[M,K] * W[N,K]^T.
//
// Decode runs these linears with M = 6 * requests-per-rank rows, where hipBLASLt's
// heuristic tiles only reach ~1 TB/s.  Here each CTA owns NT 16-row tiles of W for
// the full K; its NW waves split K, stream W straight from HBM (16 B per lane per
// 64-wide k step), keep the tiny X operand in L1/L2, and reduce the partial sums
// through LDS in a fixed order (deterministic).
//
// mmac layout (probed, probe_mmac.hip): lane l supplies A row l%16 and B col l%16
// with k = (l/16)*8 .. +8; result element e of lane l is C[row l%16][col e*4 + l/16].
// A = W tile (16 output features), B = X tile (16 tokens).
//
//   fp8:  W/X fp8 e4m3fn, per-row fp32 scales ws[N], xs[M]  (w8a8 per-channel/per-token)
//   bf16: W/X bf16, no scales
#include <ATen/hip/HIPContext.h>
#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>
#include <torch/extension.h>

#include <algorithm>

namespace {

using floatx4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;
using intx2 = __attribute__((__vector_size__(2 * sizeof(int)))) int;
using shortx4 = __attribute__((__vector_size__(4 * sizeof(short)))) short;
using uintx4 = __attribute__((__vector_size__(4 * sizeof(unsigned)))) unsigned;

template <bool kFp8>
__device__ __forceinline__ void mma2(const uintx4& a, const uintx4& b, floatx4& c) {
  // one 16-byte lane load = two mmac k-slices (low / high 8 bytes)
  if constexpr (kFp8) {
    c = __builtin_hcu_mmac_f32_16x16x32_fp8_fp8_lit_lts(intx2{(int)a[0], (int)a[1]}, intx2{(int)b[0], (int)b[1]}, c, false, false);
    c = __builtin_hcu_mmac_f32_16x16x32_fp8_fp8_lit_lts(intx2{(int)a[2], (int)a[3]}, intx2{(int)b[2], (int)b[3]}, c, false, false);
  } else {
    const shortx4 a0 = __builtin_bit_cast(shortx4, intx2{(int)a[0], (int)a[1]});
    const shortx4 a1 = __builtin_bit_cast(shortx4, intx2{(int)a[2], (int)a[3]});
    const shortx4 b0 = __builtin_bit_cast(shortx4, intx2{(int)b[0], (int)b[1]});
    const shortx4 b1 = __builtin_bit_cast(shortx4, intx2{(int)b[2], (int)b[3]});
    c = __builtin_hcu_mmac_f32_16x16x16_bf16_lit_lts(a0, b0, c, false, false);
    c = __builtin_hcu_mmac_f32_16x16x16_bf16_lit_lts(a1, b1, c, false, false);
  }
}

__device__ __forceinline__ uintx4 ld16(const uint8_t* p) { return *reinterpret_cast<const uintx4*>(p); }
__device__ __forceinline__ uintx4 ld16_stream(const uint8_t* p) {
  return __builtin_nontemporal_load(reinterpret_cast<const uintx4*>(p));
}

// kStep: bytes of K per row covered by one 16 B lane load across the 4 lane groups (64)
template <bool kFp8, int NW, int NT, int MT, int U>
__global__ __launch_bounds__(NW * 64) void skinny_gemm_kernel(
    const uint8_t* __restrict__ W, const uint8_t* __restrict__ X,
    const float* __restrict__ ws, const float* __restrict__ xs,
    __hip_bfloat16* __restrict__ Y, int M, int N, int Kb, int ldw, int ldx, int ldy,
    bool stream_w) {
  constexpr int kStep = 64;  // bytes of K per row per load step
  const int lane = threadIdx.x & 63, w = threadIdx.x >> 6;
  const int r = lane & 15, g = lane >> 4;
  const int n_base = blockIdx.x * (16 * NT);
  const int steps = Kb / kStep;
  const int per = (steps + NW - 1) / NW;
  const int s0 = min(w * per, steps), s1 = min(s0 + per, steps);

  const uint8_t* wp[NT];
#pragma unroll
  for (int t = 0; t < NT; ++t) wp[t] = W + (size_t)(n_base + t * 16 + r) * ldw + g * 16;
  const uint8_t* xp[MT];
  bool xok[MT];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt) {
    const int m = mt * 16 + r;
    xok[mt] = m < M;
    xp[mt] = X + (size_t)(xok[mt] ? m : 0) * ldx + g * 16;
  }
  floatx4 acc[NT][MT];
#pragma unroll
  for (int t = 0; t < NT; ++t)
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) acc[t][mt] = floatx4{0, 0, 0, 0};

  int s = s0;
  for (; s + U <= s1; s += U) {
    uintx4 wv[U][NT], xv[U][MT];
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const int k = (s + u) * kStep;
#pragma unroll
      for (int t = 0; t < NT; ++t) wv[u][t] = stream_w ? ld16_stream(wp[t] + k) : ld16(wp[t] + k);
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) xv[u][mt] = xok[mt] ? ld16(xp[mt] + k) : uintx4{0, 0, 0, 0};
    }
#pragma unroll
    for (int u = 0; u < U; ++u)
#pragma unroll
      for (int t = 0; t < NT; ++t)
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) mma2<kFp8>(wv[u][t], xv[u][mt], acc[t][mt]);
  }
  for (; s < s1; ++s) {
    const int k = s * kStep;
#pragma unroll
    for (int t = 0; t < NT; ++t) {
      const uintx4 wv = stream_w ? ld16_stream(wp[t] + k) : ld16(wp[t] + k);
#pragma unroll
      for (int mt = 0; mt < MT; ++mt)
        mma2<kFp8>(wv, xok[mt] ? ld16(xp[mt] + k) : uintx4{0, 0, 0, 0}, acc[t][mt]);
    }
  }

  constexpr int kTiles = NT * MT;
  if constexpr (NW > 1) {
    __shared__ floatx4 red[NW * kTiles * 64];
#pragma unroll
    for (int t = 0; t < NT; ++t)
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) red[(w * kTiles + t * MT + mt) * 64 + lane] = acc[t][mt];
    __syncthreads();
    for (int i = threadIdx.x; i < kTiles * 64; i += NW * 64) {
      floatx4 sum = red[i];
#pragma unroll
      for (int ww = 1; ww < NW; ++ww) sum += red[ww * kTiles * 64 + i];
      const int tm = i >> 6, l2 = i & 63;
      const int t = tm / MT, mt = tm % MT;
      const int n = n_base + t * 16 + (l2 & 15);
      const float wsc = kFp8 ? ws[n] : 1.f;
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int m = mt * 16 + e * 4 + (l2 >> 4);
        if (m < M) Y[(size_t)m * ldy + n] = __float2bfloat16(kFp8 ? sum[e] * wsc * xs[m] : sum[e]);
      }
    }
  } else {
#pragma unroll
    for (int t = 0; t < NT; ++t) {
      const int n = n_base + t * 16 + r;
      const float wsc = kFp8 ? ws[n] : 1.f;
#pragma unroll
      for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int m = mt * 16 + e * 4 + g;
          if (m < M) Y[(size_t)m * ldy + n] = __float2bfloat16(kFp8 ? acc[t][mt][e] * wsc * xs[m] : acc[t][mt][e]);
        }
    }
  }
}

// v2: loading W directly in mmac layout touches 16 rows x 64 B per wave instruction,
// which caps HBM at ~1.1 TB/s on BW1100 (probe_bw2.hip); 256 B per row reaches
// ~1.6-1.8 TB/s.  So each wave loads its 16-row W tile as 4 rows x 256 B per
// instruction into a wave-private LDS slab and reads it back in mmac layout (wave
// local: LDS ops of one wave execute in order, no CTA barrier).  X stays a direct
// L2 load in mmac layout.  W and X of chunk c+PF are prefetched together before
// chunk c is consumed so the in-order vmcnt never waits on the prefetch.
constexpr int kSeg = 256, kPitch = kSeg + 16, kTileLds = 16 * kPitch;

template <int NT, int MT>
struct Chunk {
  uintx4 w[NT][4];  // instr i: rows 4i + lane/16, bytes (lane%16)*16
  uintx4 x[MT][4];  // k step j (64 B): mmac layout
};

template <bool kFp8, int NW, int NT, int MT, int PF>
__global__ __launch_bounds__(NW * 64) void skinny_gemm_lds_kernel(
    const uint8_t* __restrict__ W, const uint8_t* __restrict__ X,
    const float* __restrict__ ws, const float* __restrict__ xs,
    __hip_bfloat16* __restrict__ Y, int M, int N, int Kb, int ldw, int ldx, int ldy) {
  constexpr int kWaveLds = NT * kTileLds;  // also holds the NT*MT KB reduce slab (MT <= 4)
  constexpr int S = PF + 1;
  __shared__ __attribute__((aligned(16))) uint8_t lds[NW * kWaveLds];
  const int lane = threadIdx.x & 63, w = threadIdx.x >> 6;
  const int r = lane & 15, g = lane >> 4;
  const int n_base = blockIdx.x * (16 * NT);
  const int chunks = Kb / kSeg;
  const int per = (chunks + NW - 1) / NW;
  const int c0 = min(w * per, chunks), c1 = min(c0 + per, chunks);
  uint8_t* my = lds + w * kWaveLds;

  const uint8_t* wl[NT];
#pragma unroll
  for (int t = 0; t < NT; ++t) wl[t] = W + (size_t)(n_base + t * 16 + g) * ldw + r * 16;
  const uint8_t* xp[MT];
  bool xok[MT];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt) {
    const int m = mt * 16 + r;
    xok[mt] = m < M;
    xp[mt] = X + (size_t)(xok[mt] ? m : 0) * ldx + g * 16;
  }
  floatx4 acc[NT][MT];
#pragma unroll
  for (int t = 0; t < NT; ++t)
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) acc[t][mt] = floatx4{0, 0, 0, 0};

  auto load = [&](Chunk<NT, MT>& b, int c) {
    const int kb = c * kSeg;
#pragma unroll
    for (int t = 0; t < NT; ++t)
#pragma unroll
      for (int i = 0; i < 4; ++i) b.w[t][i] = ld16(wl[t] + (size_t)i * 4 * ldw + kb);
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int j = 0; j < 4; ++j)
        b.x[mt][j] = xok[mt] ? ld16(xp[mt] + kb + j * 64) : uintx4{0, 0, 0, 0};
  };
  auto consume = [&](const Chunk<NT, MT>& b) {
#pragma unroll
    for (int t = 0; t < NT; ++t)
#pragma unroll
      for (int i = 0; i < 4; ++i)
        *reinterpret_cast<uintx4*>(my + t * kTileLds + (4 * i + g) * kPitch + r * 16) = b.w[t][i];
    __asm__ volatile("" ::: "memory");
#pragma unroll
    for (int t = 0; t < NT; ++t)
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const uintx4 a = *reinterpret_cast<const uintx4*>(my + t * kTileLds + r * kPitch + j * 64 + g * 16);
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) mma2<kFp8>(a, b.x[mt][j], acc[t][mt]);
      }
    __asm__ volatile("" ::: "memory");
  };

  Chunk<NT, MT> buf[S];
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

  constexpr int kTiles = NT * MT;
  if constexpr (NW > 1) {
    floatx4* red = reinterpret_cast<floatx4*>(lds);
    floatx4* mine = reinterpret_cast<floatx4*>(my);
#pragma unroll
    for (int t = 0; t < NT; ++t)
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) mine[(t * MT + mt) * 64 + lane] = acc[t][mt];
    __syncthreads();
    constexpr int kStride = kWaveLds / 16;  // floatx4 per wave slab
    for (int i = threadIdx.x; i < kTiles * 64; i += NW * 64) {
      floatx4 sum = red[i];
#pragma unroll
      for (int ww = 1; ww < NW; ++ww) sum += red[ww * kStride + i];
      const int tm = i >> 6, l2 = i & 63;
      const int t = tm / MT, mt = tm % MT;
      const int n = n_base + t * 16 + (l2 & 15);
      const float wsc = kFp8 ? ws[n] : 1.f;
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int m = mt * 16 + e * 4 + (l2 >> 4);
        if (m < M) Y[(size_t)m * ldy + n] = __float2bfloat16(kFp8 ? sum[e] * wsc * xs[m] : sum[e]);
      }
    }
  } else {
#pragma unroll
    for (int t = 0; t < NT; ++t) {
      const int n = n_base + t * 16 + r;
      const float wsc = kFp8 ? ws[n] : 1.f;
#pragma unroll
      for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int m = mt * 16 + e * 4 + g;
          if (m < M) Y[(size_t)m * ldy + n] = __float2bfloat16(kFp8 ? acc[t][mt][e] * wsc * xs[m] : acc[t][mt][e]);
        }
    }
  }
}

// v4: the mmac-layout X loads of v2 (16 rows x 64 B per instruction, re-read by every
// CTA from L2) grow with M and dominate once M > ~8 (diag_skinny.py).  Here X is
// staged through LDS too: each chunk's X[0:M, 256 B] is loaded row-contiguously
// (4 rows x 256 B per instruction, only ceil(M/4) instructions) and read back in
// mmac layout once per chunk for all NT tiles.  W tiles share WS <= 2 slabs (LDS
// ops of a wave run in order, so tile t may overwrite tile t-2's slab once its reads
// are issued), and LDS is sized at launch for the actual M.  X slab rows past the
// loaded ones are never read; loaded rows >= M duplicate row M-1 - a B column only
// feeds its own (unstored) output column.
template <int NT, int XI>
struct Chunk4 {
  uintx4 w[NT][4];  // instr i: rows 4i + lane/16, bytes (lane%16)*16
  uintx4 x[XI];     // instr i: X rows 4i + lane/16, bytes (lane%16)*16
};

template <bool kFp8, int NW, int NT, int MT, int PF>
__global__ __launch_bounds__(NW * 64) void skinny_gemm_v4_kernel(
    const uint8_t* __restrict__ W, const uint8_t* __restrict__ X,
    const float* __restrict__ ws, const float* __restrict__ xs,
    __hip_bfloat16* __restrict__ Y, int M, int N, int Kb, int ldw, int ldx, int ldy, int wave_lds) {
  constexpr int XI = MT * 4, WS = NT < 2 ? NT : 2, S = PF + 1;
  extern __shared__ __attribute__((aligned(16))) uint8_t lds4[];
  const int lane = threadIdx.x & 63, w = threadIdx.x >> 6;
  const int r = lane & 15, g = lane >> 4;
  const int n_base = blockIdx.x * (16 * NT);
  const int chunks = Kb / kSeg;
  const int per = (chunks + NW - 1) / NW;
  const int c0 = min(w * per, chunks), c1 = min(c0 + per, chunks);
  const int xi_n = (M + 3) >> 2;  // live X load instructions per chunk
  uint8_t* wslab = lds4 + w * wave_lds;
  uint8_t* xslab = wslab + WS * kTileLds;

  const uint8_t* wl[NT];
#pragma unroll
  for (int t = 0; t < NT; ++t) wl[t] = W + (size_t)(n_base + t * 16 + g) * ldw + r * 16;
  int xo[XI], xrd[MT];
#pragma unroll
  for (int i = 0; i < XI; ++i) xo[i] = min(4 * i + g, M - 1) * ldx + r * 16;
#pragma unroll
  for (int mt = 0; mt < MT; ++mt) xrd[mt] = min(mt * 16 + r, 4 * xi_n - 1) * kPitch + g * 16;
  floatx4 acc[NT][MT];
#pragma unroll
  for (int t = 0; t < NT; ++t)
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) acc[t][mt] = floatx4{0, 0, 0, 0};

  auto load = [&](Chunk4<NT, XI>& b, int c) {
    const int kb = c * kSeg;
#pragma unroll
    for (int i = 0; i < XI; ++i)
      if (i < xi_n) b.x[i] = ld16(X + xo[i] + kb);
#pragma unroll
    for (int t = 0; t < NT; ++t)
#pragma unroll
      for (int i = 0; i < 4; ++i) b.w[t][i] = ld16(wl[t] + (size_t)i * 4 * ldw + kb);
  };
  auto put_w = [&](const Chunk4<NT, XI>& b, int t) {
#pragma unroll
    for (int i = 0; i < 4; ++i)
      *reinterpret_cast<uintx4*>(wslab + (t % WS) * kTileLds + (4 * i + g) * kPitch + r * 16) = b.w[t][i];
  };
  auto consume = [&](const Chunk4<NT, XI>& b) {
#pragma unroll
    for (int i = 0; i < XI; ++i)
      if (i < xi_n) *reinterpret_cast<uintx4*>(xslab + (4 * i + g) * kPitch + r * 16) = b.x[i];
#pragma unroll
    for (int t = 0; t < WS; ++t) put_w(b, t);
    __asm__ volatile("" ::: "memory");
    uintx4 xb[MT][4];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int j = 0; j < 4; ++j) xb[mt][j] = *reinterpret_cast<const uintx4*>(xslab + xrd[mt] + j * 64);
#pragma unroll
    for (int t = 0; t < NT; ++t) {
      if (t >= WS) {
        __asm__ volatile("" ::: "memory");
        put_w(b, t);
        __asm__ volatile("" ::: "memory");
      }
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const uintx4 a = *reinterpret_cast<const uintx4*>(wslab + (t % WS) * kTileLds + r * kPitch + j * 64 + g * 16);
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) mma2<kFp8>(a, xb[mt][j], acc[t][mt]);
      }
    }
    __asm__ volatile("" ::: "memory");
  };

  Chunk4<NT, XI> buf[S];
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

  constexpr int kTiles = NT * MT;
  if constexpr (NW > 1) {
    floatx4* red = reinterpret_cast<floatx4*>(lds4);
    floatx4* mine = reinterpret_cast<floatx4*>(wslab);
    __asm__ volatile("" ::: "memory");
#pragma unroll
    for (int t = 0; t < NT; ++t)
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) mine[(t * MT + mt) * 64 + lane] = acc[t][mt];
    __syncthreads();
    const int stride = wave_lds / 16;  // floatx4 per wave slab
    for (int i = threadIdx.x; i < kTiles * 64; i += NW * 64) {
      floatx4 sum = red[i];
#pragma unroll
      for (int ww = 1; ww < NW; ++ww) sum += red[ww * stride + i];
      const int tm = i >> 6, l2 = i & 63;
      const int t = tm / MT, mt = tm % MT;
      const int n = n_base + t * 16 + (l2 & 15);
      const float wsc = kFp8 ? ws[n] : 1.f;
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int m = mt * 16 + e * 4 + (l2 >> 4);
        if (m < M) Y[(size_t)m * ldy + n] = __float2bfloat16(kFp8 ? sum[e] * wsc * xs[m] : sum[e]);
      }
    }
  } else {
#pragma unroll
    for (int t = 0; t < NT; ++t) {
      const int n = n_base + t * 16 + r;
      const float wsc = kFp8 ? ws[n] : 1.f;
#pragma unroll
      for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int m = mt * 16 + e * 4 + g;
          if (m < M) Y[(size_t)m * ldy + n] = __float2bfloat16(kFp8 ? acc[t][mt][e] * wsc * xs[m] : acc[t][mt][e]);
        }
    }
  }
}

// Batched bf16 variant for the MLA absorb bmm's (decode): Y[b] = X[b] * W[b]^T with
// per-head W[b] = [N, K] row-major (w_kc: N 512 x K 192, w_vc: N 256 x K 512, 64 heads)
// and strided X / Y (q_nope / attn_output head slices, transposed outputs).  K is short,
// so there is no K split: each wave owns one 16-row W tile for the whole K, walking it
// in CW-byte chunks with one chunk of prefetch.  kLds: a chunk of the tile is loaded as
// row-major 16 B pieces (lane-contiguous, 1 KB per instruction; the whole tile is one
// contiguous block for w_kc) into a wave-private LDS slab and read back in mmac layout;
// otherwise W is loaded straight in mmac layout (16 rows x 64 B per instruction).
template <int CW, int MT>
struct BChunk {
  uintx4 w[CW / 64];
  uintx4 x[MT][CW / 64];
};

template <int CW, int MT, bool kLds>
__global__ __launch_bounds__(256) void bmm_skinny_kernel(
    const uint8_t* __restrict__ W, const uint8_t* __restrict__ X, __hip_bfloat16* __restrict__ Y,
    int M, int N, int Kb, int64_t sw, int ldw, int64_t sx, int ldx, int64_t sy, int ldy) {
  constexpr int J = CW / 64, kPitchB = CW + 16, kPerRow = CW / 16;
  extern __shared__ __attribute__((aligned(16))) uint8_t bslab[];
  const int lane = threadIdx.x & 63, w = threadIdx.x >> 6;
  const int r = lane & 15, g = lane >> 4;
  const int n0 = (blockIdx.x * (blockDim.x >> 6) + w) * 16;
  if (n0 >= N) return;  // waves are independent (no CTA barrier)
  const int64_t b = blockIdx.y;
  W += b * sw + (int64_t)n0 * ldw;
  X += b * sx;
  Y += b * sy + n0;
  uint8_t* slab = bslab + w * 16 * kPitchB;

  int woff[J], loff[J];
#pragma unroll
  for (int i = 0; i < J; ++i) {
    if constexpr (kLds) {
      const int p = i * 64 + lane, row = p / kPerRow, col = p % kPerRow;
      woff[i] = row * ldw + col * 16;
      loff[i] = row * kPitchB + col * 16;
    } else {
      woff[i] = r * ldw + i * 64 + g * 16;
      loff[i] = 0;
    }
  }
  int xoff[MT];
  bool xok[MT];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt) {
    const int m = mt * 16 + r;
    xok[mt] = m < M;
    xoff[mt] = (xok[mt] ? m : 0) * ldx + g * 16;
  }
  floatx4 acc[MT];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt) acc[mt] = floatx4{0, 0, 0, 0};

  auto load = [&](BChunk<CW, MT>& bf, int c) {
    const int kb = c * CW;
#pragma unroll
    for (int i = 0; i < J; ++i) bf.w[i] = ld16(W + woff[i] + kb);
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int j = 0; j < J; ++j) bf.x[mt][j] = xok[mt] ? ld16(X + xoff[mt] + kb + j * 64) : uintx4{0, 0, 0, 0};
  };
  auto consume = [&](const BChunk<CW, MT>& bf) {
    if constexpr (kLds) {
#pragma unroll
      for (int i = 0; i < J; ++i) *reinterpret_cast<uintx4*>(slab + loff[i]) = bf.w[i];
      __asm__ volatile("" ::: "memory");
#pragma unroll
      for (int j = 0; j < J; ++j) {
        const uintx4 a = *reinterpret_cast<const uintx4*>(slab + r * kPitchB + j * 64 + g * 16);
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) mma2<false>(a, bf.x[mt][j], acc[mt]);
      }
      __asm__ volatile("" ::: "memory");
    } else {
#pragma unroll
      for (int j = 0; j < J; ++j)
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) mma2<false>(bf.w[j], bf.x[mt][j], acc[mt]);
    }
  };

  const int chunks = Kb / CW;
  BChunk<CW, MT> buf[2];
  load(buf[0], 0);
  for (int c = 0; c < chunks; c += 2) {
    if (c + 1 < chunks) load(buf[1], c + 1);
    consume(buf[0]);
    if (c + 1 < chunks) {
      if (c + 2 < chunks) load(buf[0], c + 2);
      consume(buf[1]);
    }
  }

#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const int m = mt * 16 + e * 4 + g;
      if (m < M) Y[(int64_t)m * ldy + r] = __float2bfloat16(acc[mt][e]);
    }
}

template <int CW, int MT>
void bmm_launch(bool lds, int nwv, const torch::Tensor& x, const torch::Tensor& w, torch::Tensor& y,
                hipStream_t st) {
  const int B = x.size(0), M = x.size(1), N = w.size(1), Kb = x.size(2) * 2;
  const dim3 grid((N / 16 + nwv - 1) / nwv, B);
  const auto* wp = (const uint8_t*)w.data_ptr();
  const auto* xp = (const uint8_t*)x.data_ptr();
  auto* yp = (__hip_bfloat16*)y.data_ptr();
  const int64_t sw = w.stride(0) * 2, sx = x.stride(0) * 2, sy = y.stride(0);
  const int ldw = w.stride(1) * 2, ldx = x.stride(1) * 2, ldy = y.stride(1);
  if (lds)
    bmm_skinny_kernel<CW, MT, true><<<grid, nwv * 64, nwv * 16 * (CW + 16), st>>>(
        wp, xp, yp, M, N, Kb, sw, ldw, sx, ldx, sy, ldy);
  else
    bmm_skinny_kernel<CW, MT, false><<<grid, nwv * 64, 0, st>>>(wp, xp, yp, M, N, Kb, sw, ldw, sx, ldx, sy, ldy);
}

template <int CW>
void bmm_launch_m(int mt, bool lds, int nwv, const torch::Tensor& x, const torch::Tensor& w, torch::Tensor& y,
                  hipStream_t st) {
  if (mt == 1) bmm_launch<CW, 1>(lds, nwv, x, w, y, st);
  else bmm_launch<CW, 2>(lds, nwv, x, w, y, st);
}

template <bool kFp8, int NW, int NT, int MT>
void launch3(int ver, const torch::Tensor& x, const torch::Tensor& xs, const torch::Tensor& w,
             const torch::Tensor& ws, torch::Tensor& y, int Kb, bool stream_w, hipStream_t st) {
  const int M = x.size(0), N = w.size(0), esz = kFp8 ? 1 : 2;
  const uint8_t* wp = (const uint8_t*)w.data_ptr();
  const uint8_t* xp = (const uint8_t*)x.data_ptr();
  const float* wsp = kFp8 ? ws.data_ptr<float>() : nullptr;
  const float* xsp = kFp8 ? xs.data_ptr<float>() : nullptr;
  auto* yp = (__hip_bfloat16*)y.data_ptr();
  const int ldw = w.stride(0) * esz, ldx = x.stride(0) * esz, ldy = y.stride(0);
  if (ver >= 4) {  // ver 4 / 5: prefetch 1 / 2 chunks ahead (5 only while registers allow)
    TORCH_CHECK(Kb % kSeg == 0, "skinny_gemm v4: K bytes must be a multiple of 256");
    constexpr int WS = NT < 2 ? NT : 2;
    int wave_lds = WS * kTileLds + 4 * ((M + 3) / 4) * kPitch;
    if (NW > 1) wave_lds = std::max(wave_lds, NT * MT * 1024);
    const int smem = NW * wave_lds;
    TORCH_CHECK(smem <= 64 * 1024, "skinny_gemm v4: nw too large for LDS at this M");
    if constexpr (NW * NT > 16) {
      TORCH_CHECK(false, "skinny_gemm v4: nw * nt must be <= 16");
    } else if (ver == 5 && NT + MT <= 4) {
      skinny_gemm_v4_kernel<kFp8, NW, NT, MT, 2><<<N / (16 * NT), NW * 64, smem, st>>>(
          wp, xp, wsp, xsp, yp, M, N, Kb, ldw, ldx, ldy, wave_lds);
    } else {
      skinny_gemm_v4_kernel<kFp8, NW, NT, MT, 1><<<N / (16 * NT), NW * 64, smem, st>>>(
          wp, xp, wsp, xsp, yp, M, N, Kb, ldw, ldx, ldy, wave_lds);
    }
    return;
  }
  if (ver >= 2) {
    TORCH_CHECK(Kb % kSeg == 0, "skinny_gemm v2: K bytes must be a multiple of 256");
    if constexpr (NW * NT * kTileLds > 64 * 1024) {
      TORCH_CHECK(false, "skinny_gemm v2: nw * nt too large for LDS");
    } else if (ver == 2 || MT >= 3) {  // ver 3: prefetch 2 chunks ahead
      skinny_gemm_lds_kernel<kFp8, NW, NT, MT, 1><<<N / (16 * NT), NW * 64, 0, st>>>(
          wp, xp, wsp, xsp, yp, M, N, Kb, ldw, ldx, ldy);
    } else {
      skinny_gemm_lds_kernel<kFp8, NW, NT, MT, 2><<<N / (16 * NT), NW * 64, 0, st>>>(
          wp, xp, wsp, xsp, yp, M, N, Kb, ldw, ldx, ldy);
    }
    return;
  }
  if constexpr (NW * NT * MT > 32) {  // LDS reduce buffer would exceed 32 KB
    TORCH_CHECK(false, "skinny_gemm: nw * nt * m_tiles must be <= 32");
  } else {
    constexpr int U = (NT * MT >= 8) ? 2 : 4;
    skinny_gemm_kernel<kFp8, NW, NT, MT, U><<<N / (16 * NT), NW * 64, 0, st>>>(
        wp, xp, wsp, xsp, yp, M, N, Kb, ldw, ldx, ldy, stream_w);
  }
}

template <bool kFp8, int NW, int NT>
void launch2(int ver, int mt, const torch::Tensor& x, const torch::Tensor& xs, const torch::Tensor& w,
             const torch::Tensor& ws, torch::Tensor& y, int Kb, bool stream_w, hipStream_t st) {
  if (mt == 1) launch3<kFp8, NW, NT, 1>(ver, x, xs, w, ws, y, Kb, stream_w, st);
  else if (mt == 2) launch3<kFp8, NW, NT, 2>(ver, x, xs, w, ws, y, Kb, stream_w, st);
  else if (mt == 3) launch3<kFp8, NW, NT, 3>(ver, x, xs, w, ws, y, Kb, stream_w, st);
  else launch3<kFp8, NW, NT, 4>(ver, x, xs, w, ws, y, Kb, stream_w, st);
}

template <bool kFp8, int NW>
void launch1(int ver, int nt, int mt, const torch::Tensor& x, const torch::Tensor& xs, const torch::Tensor& w,
             const torch::Tensor& ws, torch::Tensor& y, int Kb, bool stream_w, hipStream_t st) {
  if (nt == 1) launch2<kFp8, NW, 1>(ver, mt, x, xs, w, ws, y, Kb, stream_w, st);
  else if (nt == 2) launch2<kFp8, NW, 2>(ver, mt, x, xs, w, ws, y, Kb, stream_w, st);
  else launch2<kFp8, NW, 4>(ver, mt, x, xs, w, ws, y, Kb, stream_w, st);
}

}  // namespace

// Default launch shape: one 16-row tile per CTA unless that leaves the K split
// too thin; NW waves split K so each wave streams >= ~8 steps of 64 B per row
// (v1) or >= 4 chunks of 256 B (v2, whose staging slab caps nw * nt at 8).
static void pick_cfg(int ver, int M, int N, int Kb, int mt, int& nw, int& nt) {
  if (nt <= 0) nt = 1;
  if (nw > 0) return;
  nw = 1;
  if (ver >= 2) {
    const int chunks = Kb / kSeg, cap = ver >= 4 ? 16 : 8;
    while (nw < 8 && chunks / (nw * 2) >= 4 && nw * 2 * nt <= cap) nw *= 2;
    // v4 LDS per wave: W slabs + X slab for M rows (see launch3)
    const int wave_lds = std::min(nt, 2) * kTileLds + 4 * ((M + 3) / 4) * kPitch;
    while (ver >= 4 && nw > 1 && nw * std::max(wave_lds, nt * mt * 1024) > 64 * 1024) nw /= 2;
  } else {
    const int steps = Kb / 64;
    while (nw < 16 && steps / (nw * 2) >= 8 && nw * 2 * nt * mt <= 32) nw *= 2;
  }
}

void skinny_gemm(const torch::Tensor& x, const torch::Tensor& x_scale, const torch::Tensor& w,
                 const torch::Tensor& w_scale, torch::Tensor& y, int64_t nw, int64_t nt, bool stream_w,
                 int64_t ver) {
  const bool fp8 = w.scalar_type() == at::ScalarType::Float8_e4m3fn;
  TORCH_CHECK(fp8 || w.scalar_type() == at::ScalarType::BFloat16, "w must be fp8_e4m3fn or bf16");
  TORCH_CHECK(x.scalar_type() == w.scalar_type(), "x/w dtype mismatch");
  TORCH_CHECK(y.scalar_type() == at::ScalarType::BFloat16, "y must be bf16");
  TORCH_CHECK(x.dim() == 2 && w.dim() == 2 && y.dim() == 2, "2-D tensors expected");
  TORCH_CHECK(x.stride(1) == 1 && w.stride(1) == 1 && y.stride(1) == 1, "inner dim must be contiguous");
  const int M = x.size(0), K = x.size(1), N = w.size(0), esz = fp8 ? 1 : 2;
  TORCH_CHECK(w.size(1) == K && y.size(0) == M && y.size(1) == N, "shape mismatch");
  TORCH_CHECK(M >= 1 && M <= 64, "M must be in [1, 64]");
  const int Kb = K * esz;
  TORCH_CHECK(Kb % 64 == 0, "K bytes must be a multiple of 64");
  TORCH_CHECK((x.stride(0) * esz) % 16 == 0 && (w.stride(0) * esz) % 16 == 0, "rows must be 16 B aligned");
  TORCH_CHECK(((uintptr_t)x.data_ptr() | (uintptr_t)w.data_ptr()) % 16 == 0, "16 B alignment");
  if (fp8) {
    TORCH_CHECK(x_scale.scalar_type() == at::ScalarType::Float && x_scale.numel() == M && x_scale.is_contiguous(), "x_scale");
    TORCH_CHECK(w_scale.scalar_type() == at::ScalarType::Float && w_scale.numel() == N && w_scale.is_contiguous(), "w_scale");
  }
  const int mt = (M + 15) / 16;
  int inw = nw, int_nt = nt;
  pick_cfg(ver, M, N, Kb, mt, inw, int_nt);
  TORCH_CHECK(N % (16 * int_nt) == 0, "N must be a multiple of 16*nt");
  hipStream_t st = at::hip::getCurrentHIPStream();
#define SG_NW(NWV)                                                                                \
  if (inw == NWV) {                                                                               \
    if (fp8) launch1<true, NWV>(ver, int_nt, mt, x, x_scale, w, w_scale, y, Kb, stream_w, st);   \
    else launch1<false, NWV>(ver, int_nt, mt, x, x_scale, w, w_scale, y, Kb, stream_w, st);      \
    return;                                                                                       \
  }
  SG_NW(1) SG_NW(2) SG_NW(4) SG_NW(8) SG_NW(16)
#undef SG_NW
  TORCH_CHECK(false, "nw must be 1/2/4/8/16");
}

void bmm_skinny(const torch::Tensor& x, const torch::Tensor& w, torch::Tensor& y, int64_t cw, int64_t nwv,
                bool lds) {
  TORCH_CHECK(x.scalar_type() == at::ScalarType::BFloat16 && w.scalar_type() == at::ScalarType::BFloat16 &&
                  y.scalar_type() == at::ScalarType::BFloat16, "bmm_skinny: bf16 only");
  TORCH_CHECK(x.dim() == 3 && w.dim() == 3 && y.dim() == 3, "bmm_skinny: 3-D tensors expected");
  TORCH_CHECK(x.stride(2) == 1 && w.stride(2) == 1 && y.stride(2) == 1, "bmm_skinny: inner dim must be contiguous");
  const int B = x.size(0), M = x.size(1), K = x.size(2), N = w.size(1);
  TORCH_CHECK(w.size(0) == B && w.size(2) == K && y.size(0) == B && y.size(1) == M && y.size(2) == N,
              "bmm_skinny: shape mismatch");
  TORCH_CHECK(M >= 1 && M <= 32, "bmm_skinny: M must be in [1, 32]");
  TORCH_CHECK(N % 16 == 0, "bmm_skinny: N must be a multiple of 16");
  const int Kb = K * 2;
  if (cw <= 0) cw = Kb % 384 == 0 ? 384 : Kb % 256 == 0 ? 256 : 128;
  TORCH_CHECK(Kb % cw == 0, "bmm_skinny: K bytes must be a multiple of the chunk width");
  TORCH_CHECK((x.stride(0) | x.stride(1) | w.stride(0) | w.stride(1)) % 8 == 0, "bmm_skinny: 16 B aligned rows");
  TORCH_CHECK(((uintptr_t)x.data_ptr() | (uintptr_t)w.data_ptr()) % 16 == 0, "bmm_skinny: 16 B alignment");
  TORCH_CHECK(nwv == 1 || nwv == 2 || nwv == 4, "bmm_skinny: nwv must be 1/2/4");
  const int mt = (M + 15) / 16;
  hipStream_t st = at::hip::getCurrentHIPStream();
  switch (cw) {
    case 128: bmm_launch_m<128>(mt, lds, nwv, x, w, y, st); break;
    case 256: bmm_launch_m<256>(mt, lds, nwv, x, w, y, st); break;
    case 384: bmm_launch_m<384>(mt, lds, nwv, x, w, y, st); break;
    case 512: bmm_launch_m<512>(mt, lds, nwv, x, w, y, st); break;
    default: TORCH_CHECK(false, "bmm_skinny: cw must be 128/256/384/512");
  }
}
