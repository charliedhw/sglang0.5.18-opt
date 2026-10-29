// Exact Top-2048 + page-table transform over LightOp sparse paged-MQA logits.
//
// Drop-in replacement for lightop fast_topk_transform_sparse_mask (decode
// layout: one page-table row per score row).  Selects the same top-2048 set
// (threshold ties resolved towards lower token index) and writes entries
// strictly above the threshold in ascending token order followed by the
// threshold ties; the layout is deterministic across graph replays.
//
// LightOp runs 4 rounds of 8-bit radix over the row with heavily contended
// shared atomics (positive logits share the top key byte) plus two extra
// scan passes, all with a serial 16-token bit loop.  This kernel instead
// streams float4 + mask with every wave owning a contiguous segment:
//   pass 1: 12-bit histogram (key[31:20]) + per-wave positive/zero counts;
//           if positives <= 2048 <= positives + zeros the threshold is +0
//           and passes 2-3 are skipped
//   pass 2: 12-bit histogram (key[19:8]) of the threshold bin
//   pass 3: per-wave 8-bit histograms (key[7:0]) of the 24-bit threshold
//           prefix + per-wave "strictly above" counts
//   pass 4: ballot-prefix compaction of the selected token indices into LDS,
//           then one coalesced page-table gather.
// All per-token work is branch-free: tokens outside a histogram add into a
// per-lane trash slot (per-token exec-mask branches cost ~4x otherwise).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <algorithm>
#include <cstdint>

namespace {

constexpr int kTopK = 2048;
constexpr int kThreads = 1024;
constexpr int kWave = 64;
constexpr int kWaves = kThreads / kWave;
constexpr int kChunk = kWave * 4;  // tokens per wave iteration (float4/lane)
constexpr int kUnroll = 4;
constexpr int kBins = 4096;
constexpr uint32_t kZeroKey = 0x80000000u;

__device__ __forceinline__ uint32_t to_key(float x) {
  const uint32_t b = __float_as_uint(x);
  return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

__device__ __forceinline__ int lane_id() { return threadIdx.x & (kWave - 1); }

__device__ __forceinline__ int wave_sum(int x) {
#pragma unroll
  for (int o = kWave / 2; o > 0; o >>= 1) x += __shfl_xor(x, o);
  return x;
}

// Exclusive prefix (in lane order) of per-lane counts in [0, 7].
__device__ __forceinline__ int wave_prefix_small(int c, int& total) {
  const uint64_t b0 = __ballot(c & 1);
  const uint64_t b1 = __ballot(c & 2);
  const uint64_t b2 = __ballot(c & 4);
  const int lane = lane_id();
  const uint64_t lt = lane == 0 ? 0ull : (~0ull >> (64 - lane));
  total = __popcll(b0) + 2 * __popcll(b1) + 4 * __popcll(b2);
  return __popcll(b0 & lt) + 2 * __popcll(b1 & lt) + 4 * __popcll(b2 & lt);
}

// Raw loads of tokens [idx0, idx0 + 4); branch-free so that all loads of an
// unrolled group are in flight together.  Out-of-range groups read token 0.
template <bool kVec>
__device__ __forceinline__ void load_raw(
    const float* __restrict__ vals, const int16_t* __restrict__ mask,
    int idx0, int lim, float (&v)[4], uint32_t& mraw) {
  const int safe = idx0 < lim ? idx0 : 0;
  mraw = static_cast<uint16_t>(mask[safe >> 4]);
  if constexpr (kVec) {
    // kVec guarantees width % 4 == 0, so the float4 stays inside the row.
    const float4 f = *reinterpret_cast<const float4*>(vals + safe);
    v[0] = f.x; v[1] = f.y; v[2] = f.z; v[3] = f.w;
  } else {
#pragma unroll
    for (int j = 0; j < 4; ++j)
      v[j] = vals[safe + j < lim ? safe + j : lim - 1];
  }
}

// Converts raw loads to keys.  Tokens whose nonzero-mask bit is clear are
// implicit +0 (their score storage is never written).  Returns the bitmask
// of tokens < lim.
__device__ __forceinline__ uint32_t to_keys(
    int idx0, int lim, const float (&v)[4], uint32_t mraw, uint32_t (&k)[4]) {
  const uint32_t m = (mraw >> (idx0 & 15)) & 0xFu;
#pragma unroll
  for (int j = 0; j < 4; ++j) k[j] = ((m >> j) & 1u) ? to_key(v[j]) : kZeroKey;
  const int rem = lim - idx0;
  const int nv = rem <= 0 ? 0 : (rem < 4 ? rem : 4);
  return (1u << nv) - 1u;
}

// Finds the bin t with suffix(t) >= need > suffix(t + 1); writes t and
// suffix(t + 1).  Requires sum(hist) >= need >= 1.
template <int NB>
__device__ __forceinline__ void find_threshold(
    const int* hist, int need, int* s_wtot, int* s_res) {
  constexpr int BPT = NB >= kThreads ? NB / kThreads : 1;
  constexpr int ACTIVE = NB / BPT;
  const int tid = threadIdx.x;
  const int lane = lane_id();
  const int w = tid / kWave;
  int h[BPT];
  int s = 0;
#pragma unroll
  for (int j = 0; j < BPT; ++j) {
    h[j] = tid < ACTIVE ? hist[tid * BPT + j] : 0;
    s += h[j];
  }
  int x = s;
#pragma unroll
  for (int o = 1; o < kWave; o <<= 1) {
    const int y = __shfl_down(x, o);
    if (lane + o < kWave) x += y;
  }
  if (lane == 0) s_wtot[w] = x;
  __syncthreads();
  int after = 0;
  for (int v = w + 1; v < kWaves; ++v) after += s_wtot[v];
  const int suf = x + after;
  const int suf_next = suf - s;
  if (tid < ACTIVE && suf >= need && suf_next < need) {
    int acc = suf_next;
#pragma unroll
    for (int j = BPT - 1; j >= 0; --j) {
      if (acc + h[j] >= need) {
        s_res[0] = tid * BPT + j;
        s_res[1] = acc;
        break;
      }
      acc += h[j];
    }
  }
  __syncthreads();
}

template <bool kVec>
__global__ __launch_bounds__(kThreads) void sparse_mask_topk_decode_kernel(
    const float* __restrict__ score, int64_t score_stride, int width,
    const int16_t* __restrict__ nonzero_mask, int64_t mask_stride,
    const int32_t* __restrict__ lengths,
    const int32_t* __restrict__ src_page_table, int64_t src_stride,
    int32_t* __restrict__ dst_page_table) {
  __shared__ int hist[kBins + kWave];  // + per-lane trash slots
  __shared__ int whist[kWaves][256];
  __shared__ int s_trash[kWave];
  __shared__ int s_out[kTopK + kWave];  // selected token indices + trash
  __shared__ int s_wtot[kWaves];
  __shared__ int s_res[2];
  __shared__ int s_above[kWaves];
  __shared__ int s_ties[kWaves];
  __shared__ int s_zero[kWaves];

  const int64_t row = blockIdx.x;
  const int tid = threadIdx.x;
  const int lane = lane_id();
  const int w = tid / kWave;
  int len = lengths[row];
  len = len < width ? len : width;
  const int32_t* src = src_page_table + row * src_stride;
  int32_t* dst = dst_page_table + row * kTopK;

  if (len <= kTopK) {
    for (int i = tid; i < kTopK; i += kThreads) dst[i] = i < len ? src[i] : -1;
    return;
  }

  const float* vals = score + row * score_stride;
  const int16_t* mask = nonzero_mask + row * mask_stride;
  const int seg = (len + kWaves * kChunk - 1) / (kWaves * kChunk) * kChunk;
  const int seg_begin = w * seg;
  const int seg_end = seg_begin + seg < len ? seg_begin + seg : len;
  const int trash = kBins + lane;

  // Visits every token slot of this wave's segment as (`valid`, `key`, `idx`);
  // loads of kUnroll chunks are issued before any use.
#define FOR_EACH_GROUP(...)                                                 \
  for (int c = seg_begin; c < seg_end; c += kChunk * kUnroll) {             \
    float rv[kUnroll][4];                                                   \
    uint32_t rm[kUnroll];                                                   \
    _Pragma("unroll") for (int u = 0; u < kUnroll; ++u)                     \
        load_raw<kVec>(vals, mask, c + u * kChunk + lane * 4, seg_end,      \
                       rv[u], rm[u]);                                       \
    _Pragma("unroll") for (int u = 0; u < kUnroll; ++u) {                   \
      const int i0 = c + u * kChunk + lane * 4;                             \
      uint32_t kk[4];                                                       \
      const uint32_t vv = to_keys(i0, seg_end, rv[u], rm[u], kk);           \
      _Pragma("unroll") for (int j = 0; j < 4; ++j) {                       \
        const bool valid = (vv >> j) & 1u;                                  \
        const uint32_t key = kk[j];                                         \
        __VA_ARGS__                                                         \
      }                                                                     \
    }                                                                       \
  }

  // ---- pass 1: key[31:20] of nonzero tokens + positive / zero counts ----
  for (int i = tid; i < kBins; i += kThreads) hist[i] = 0;
  __syncthreads();
  {
    int zc = 0;
    int pc = 0;
    FOR_EACH_GROUP({
      const bool nz = key != kZeroKey;
      zc += valid & !nz;
      pc += valid & (key > kZeroKey);
      atomicAdd(&hist[valid & nz ? static_cast<int>(key >> 20) : trash], 1);
    })
    zc = wave_sum(zc);
    pc = wave_sum(pc);
    if (lane == 0) {
      if (zc) atomicAdd(&hist[kZeroKey >> 20], zc);
      s_above[w] = pc;
      s_ties[w] = zc;
      s_zero[w] = zc;
    }
  }
  __syncthreads();
  int positives = 0;
  int zeros = 0;
#pragma unroll
  for (int v = 0; v < kWaves; ++v) {
    positives += s_above[v];
    zeros += s_zero[v];
  }

  uint32_t thr;
  int remaining;
  if (positives <= kTopK && positives + zeros >= kTopK) {
    // Sparse fast path: every positive plus the lowest-index zeros.
    thr = kZeroKey;
    remaining = kTopK - positives;
  } else {
    find_threshold<kBins>(hist, kTopK, s_wtot, s_res);
    const uint32_t t1 = static_cast<uint32_t>(s_res[0]);
    const int need1 = kTopK - s_res[1];

    // ---- pass 2: key[19:8] within bin t1 (zeros added from pass 1) ----
    for (int i = tid; i < kBins; i += kThreads) hist[i] = 0;
    __syncthreads();
    if (tid == 0 && t1 == (kZeroKey >> 20))
      atomicAdd(&hist[(kZeroKey >> 8) & 0xFFFu], zeros);
    FOR_EACH_GROUP({
      const bool cand = valid & ((key >> 20) == t1) & (key != kZeroKey);
      atomicAdd(&hist[cand ? static_cast<int>((key >> 8) & 0xFFFu) : trash], 1);
    })
    __syncthreads();
    find_threshold<kBins>(hist, need1, s_wtot, s_res);
    const uint32_t prefix24 = (t1 << 12) | static_cast<uint32_t>(s_res[0]);
    const int need2 = need1 - s_res[1];

    // ---- pass 3: per-wave key[7:0] within prefix24 + per-wave above ----
    for (int i = tid; i < kWaves * 256; i += kThreads) (&whist[0][0])[i] = 0;
    __syncthreads();
    {
      int above = 0;
      FOR_EACH_GROUP({
        const uint32_t hi = key >> 8;
        above += valid & (hi > prefix24);
        const bool cand = valid & (hi == prefix24) & (key != kZeroKey);
        atomicAdd(cand ? &whist[w][key & 0xFFu] : &s_trash[lane], 1);
      })
      above = wave_sum(above);
      if (lane == 0) {
        if (prefix24 == (kZeroKey >> 8))
          atomicAdd(&whist[w][kZeroKey & 0xFFu], s_zero[w]);
        s_above[w] = above;
      }
    }
    __syncthreads();
    if (tid < 256) {
      int s = 0;
#pragma unroll
      for (int v = 0; v < kWaves; ++v) s += whist[v][tid];
      hist[tid] = s;
    }
    __syncthreads();
    find_threshold<256>(hist, need2, s_wtot, s_res);
    const uint32_t t3 = static_cast<uint32_t>(s_res[0]);
    thr = (prefix24 << 8) | t3;
    remaining = need2 - s_res[1];
    {
      int a = 0;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const uint32_t d = lane * 4 + j;
        if (d > t3) a += whist[w][d];
      }
      a = wave_sum(a);
      if (lane == 0) {
        s_above[w] += a;
        s_ties[w] = whist[w][t3];
      }
    }
    __syncthreads();
  }
  const int tie_base = kTopK - remaining;

  // ---- pass 4: compact selected token indices into s_out ----
  int above_pos = 0;
  int tie_pos = 0;
  for (int v = 0; v < w; ++v) {
    above_pos += s_above[v];
    tie_pos += s_ties[v];
  }
  for (int c = seg_begin; c < seg_end; c += kChunk * kUnroll) {
    float rv[kUnroll][4];
    uint32_t rm[kUnroll];
#pragma unroll
    for (int u = 0; u < kUnroll; ++u)
      load_raw<kVec>(vals, mask, c + u * kChunk + lane * 4, seg_end, rv[u],
                     rm[u]);
#pragma unroll
    for (int u = 0; u < kUnroll; ++u) {
      const int i0 = c + u * kChunk + lane * 4;
      uint32_t kk[4];
      const uint32_t vv = to_keys(i0, seg_end, rv[u], rm[u], kk);
      bool sa[4], st[4];
      int na = 0, nt = 0;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const bool valid = (vv >> j) & 1u;
        sa[j] = valid & (kk[j] > thr);
        st[j] = valid & (kk[j] == thr);
        na += sa[j];
        nt += st[j];
      }
      int ta, tt;
      int pa = above_pos + wave_prefix_small(na, ta);
      int pt = tie_pos + wave_prefix_small(nt, tt);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int tgt = sa[j] ? pa
                        : (st[j] & (pt < remaining)) ? tie_base + pt
                                                     : kTopK + lane;
        s_out[tgt] = i0 + j;
        pa += sa[j];
        pt += st[j];
      }
      above_pos += ta;
      tie_pos += tt;
    }
  }
#undef FOR_EACH_GROUP
  __syncthreads();
  for (int i = tid; i < kTopK; i += kThreads) dst[i] = src[s_out[i]];
}

// ---------------------------------------------------------------------------
// Split variant: one row is spread over up to S CTAs (each a `slice` of
// tokens), so small decode batches use the whole GPU instead of one CU per
// row.  The slicing is derived on device from the row length (the score width
// is the padded page-table capacity under CUDA graphs).
//   K1 split_hist:    per-slice 12-bit histogram + positive/zero counts,
//                     merged into a per-row global histogram.
//   K2 split_compact: each CTA derives the bin threshold t1 and compacts, in
//                     token order, the tokens certainly selected (list A) and
//                     the threshold-bin candidates (list C, with keys); also a
//                     global 12-bit histogram of the candidates' key[19:8].
//   K3 split_refine:  each CTA resolves the 24-bit prefix and writes its
//                     slice's count above it + 8-bit histogram of key[7:0].
//   K4 split_select:  each CTA resolves the exact threshold from all slices'
//                     histograms and writes its slice's share of A, then C
//                     above threshold, then C ties (lowest index first)
//                     through the page table.
// ---------------------------------------------------------------------------
constexpr int kBlockTok = kThreads * 4;  // tokens per block step (float4/thread)
constexpr int kSplitUnroll = 2;
constexpr int kStepTok = kBlockTok * kSplitUnroll;
constexpr int kMaxSplit = 64;
constexpr int kMeta = 8;           // per-row meta: fast, t1, need1, prefix24, need2
constexpr int kRefine = 1 + 256;   // per-slice: count above prefix24 + hist
// Per-slice list capacities.  |A| < 2048 per row by construction; a slice
// whose candidate list C overflows re-scans its scores in K3/K4 instead (the
// score width is the page-table capacity, ~1M tokens, so lists cannot be
// sized by width).
constexpr int kCapA = kTopK;
constexpr int kCapC = 2048;

// CTAs are slice-major (blockIdx = s * rows + row): the dispatcher deals CTAs
// to shader engines round-robin, so a row's idle tail slices must not sit next
// to each other.  Only one 1024-thread CTA fits per CU, so the host picks S
// such that rows * S stays within one wave of CTAs.
// Tokens per CTA for a row of `len` tokens spread over at most S CTAs.
__device__ __forceinline__ int row_slice(int len, int S) {
  const int per = (len + S - 1) / S;
  return (per + kStepTok - 1) / kStepTok * kStepTok;
}

// Block-wide walk over tokens [begin, end), one float4 group per thread per
// step; f(i0, valid_bits, keys) is called uniformly by every thread.
template <bool kVec, typename F>
__device__ __forceinline__ void for_each_block(
    const float* __restrict__ vals, const int16_t* __restrict__ mask,
    int begin, int end, F&& f) {
  for (int base = begin; base < end; base += kBlockTok) {
    const int i0 = base + static_cast<int>(threadIdx.x) * 4;
    float v[4];
    uint32_t mr;
    load_raw<kVec>(vals, mask, i0, end, v, mr);
    uint32_t k[4];
    const uint32_t vv = to_keys(i0, end, v, mr, k);
    f(i0, vv, k);
  }
}

template <typename T>
__device__ __forceinline__ T block_excl_scan(T x, T& total, T* s_w) {
  const int lane = lane_id();
  const int w = threadIdx.x / kWave;
  T inc = x;
#pragma unroll
  for (int o = 1; o < kWave; o <<= 1) {
    const T y = __shfl_up(inc, o);
    if (lane >= o) inc += y;
  }
  if (lane == kWave - 1) s_w[w] = inc;
  __syncthreads();
  T before = 0;
  T tot = 0;
#pragma unroll
  for (int v = 0; v < kWaves; ++v) {
    const T t = s_w[v];
    before += v < w ? t : T(0);
    tot += t;
  }
  __syncthreads();
  total = tot;
  return before + inc - x;
}

// Visits this CTA's slice [begin, end) block-stride: thread tid handles
// tokens base + u * kBlockTok + tid * 4 + j, i.e. ascending token order is
// (base, u, tid, j).
#define FOR_EACH_STEP(PER_STEP_PRE, PER_TOKEN, PER_STEP_POST)                 \
  for (int base = begin; base < end; base += kStepTok) {                      \
    float rv[kSplitUnroll][4];                                                \
    uint32_t rm[kSplitUnroll];                                                \
    _Pragma("unroll") for (int u = 0; u < kSplitUnroll; ++u)                  \
        load_raw<kVec>(vals, mask, base + u * kBlockTok + tid * 4, end,       \
                       rv[u], rm[u]);                                         \
    PER_STEP_PRE                                                              \
    _Pragma("unroll") for (int u = 0; u < kSplitUnroll; ++u) {                \
      const int i0 = base + u * kBlockTok + tid * 4;                          \
      uint32_t kk[4];                                                         \
      const uint32_t vv = to_keys(i0, end, rv[u], rm[u], kk);                 \
      _Pragma("unroll") for (int j = 0; j < 4; ++j) {                         \
        const bool valid = (vv >> j) & 1u;                                    \
        const uint32_t key = kk[j];                                           \
        PER_TOKEN                                                             \
      }                                                                       \
    }                                                                         \
    PER_STEP_POST                                                             \
  }

template <bool kVec>
__global__ __launch_bounds__(kThreads) void split_hist_kernel(
    const float* __restrict__ score, int64_t score_stride, int width,
    const int16_t* __restrict__ nonzero_mask, int64_t mask_stride,
    const int32_t* __restrict__ lengths, int S,
    int* __restrict__ ghist, int* __restrict__ gcnt) {
  __shared__ int hist[kBins + kWave];
  const int row = blockIdx.x % (gridDim.x / S);
  const int s = blockIdx.x / (gridDim.x / S);
  const int tid = threadIdx.x;
  const int lane = lane_id();
  int len = lengths[row];
  len = len < width ? len : width;
  if (len <= kTopK) return;
  const int slice = row_slice(len, S);
  const int begin = s * slice;
  if (begin >= len) return;
  const int end = begin + slice < len ? begin + slice : len;
  const float* vals = score + row * score_stride;
  const int16_t* mask = nonzero_mask + row * mask_stride;
  const int trash = kBins + lane;

  for (int i = tid; i < kBins; i += kThreads) hist[i] = 0;
  __syncthreads();
  int zc = 0;
  int pc = 0;
  FOR_EACH_STEP(, {
    const bool nz = key != kZeroKey;
    zc += valid & !nz;
    pc += valid & (key > kZeroKey);
    atomicAdd(&hist[valid & nz ? static_cast<int>(key >> 20) : trash], 1);
  }, )
  zc = wave_sum(zc);
  pc = wave_sum(pc);
  if (lane == 0) {
    if (zc) {
      atomicAdd(&hist[kZeroKey >> 20], zc);
      atomicAdd(&gcnt[row * 2 + 1], zc);
    }
    if (pc) atomicAdd(&gcnt[row * 2], pc);
  }
  __syncthreads();
  int* gh = ghist + static_cast<int64_t>(row) * kBins;
  for (int i = tid; i < kBins; i += kThreads) {
    const int h = hist[i];
    if (h) atomicAdd(&gh[i], h);
  }
}

template <bool kVec>
__global__ __launch_bounds__(kThreads) void split_compact_kernel(
    const float* __restrict__ score, int64_t score_stride, int width,
    const int16_t* __restrict__ nonzero_mask, int64_t mask_stride,
    const int32_t* __restrict__ lengths, int S,
    const int* __restrict__ ghist1, const int* __restrict__ gcnt,
    int* __restrict__ ghist2, int* __restrict__ gmeta,
    int* __restrict__ gcnt_ac, int* __restrict__ ent_a,
    int2* __restrict__ ent_c) {
  __shared__ int hist[kBins + kWave];
  __shared__ uint64_t s_w64[kWaves];
  __shared__ int s_wtot[kWaves];
  __shared__ int s_res[2];
  const int row = blockIdx.x % (gridDim.x / S);
  const int s = blockIdx.x / (gridDim.x / S);
  const int tid = threadIdx.x;
  const int lane = lane_id();
  int len = lengths[row];
  len = len < width ? len : width;
  if (len <= kTopK) return;
  const int slice = row_slice(len, S);
  const int begin = s * slice;
  if (begin >= len) return;
  int* cnt_a = gcnt_ac + (row * 2) * S;
  int* cnt_c = gcnt_ac + (row * 2 + 1) * S;
  const int end = begin + slice < len ? begin + slice : len;
  const float* vals = score + row * score_stride;
  const int16_t* mask = nonzero_mask + row * mask_stride;
  const int trash = kBins + lane;

  const int positives = gcnt[row * 2];
  const int zeros = gcnt[row * 2 + 1];
  // Sparse fast path: threshold is exactly +0, list C holds the zero ties.
  const bool fast = positives <= kTopK && positives + zeros >= kTopK;
  uint32_t t1 = 0;
  int need1 = 0;
  int cap_c = 0x7FFFFFFF;
  if (fast) {
    cap_c = kTopK - positives;
  } else {
    find_threshold<kBins>(ghist1 + static_cast<int64_t>(row) * kBins, kTopK,
                          s_wtot, s_res);
    t1 = static_cast<uint32_t>(s_res[0]);
    need1 = kTopK - s_res[1];
    for (int i = tid; i < kBins; i += kThreads) hist[i] = 0;
  }
  if (s == 0 && tid == 0) {
    gmeta[row * kMeta + 0] = fast;
    gmeta[row * kMeta + 1] = static_cast<int>(t1);
    gmeta[row * kMeta + 2] = need1;
  }
  __syncthreads();

  // Slice s owns kCapA / kCapC list entries.
  const int64_t slot = static_cast<int64_t>(row) * S + s;
  int* ea = ent_a + slot * kCapA;
  int2* ec = ent_c + slot * kCapC;
  const int lim_c = cap_c < kCapC ? cap_c : kCapC;
  int off_a = 0;
  int off_c = 0;
  // Only keys are kept across the block scan; flags are recomputed after it
  // (storing them as lane masks spilled).
  uint32_t ks[kSplitUnroll][4];
  int na[kSplitUnroll], nc[kSplitUnroll];
  FOR_EACH_STEP(
      {
        _Pragma("unroll") for (int u = 0; u < kSplitUnroll; ++u) na[u] = nc[u] = 0;
      },
      {
        const uint32_t hi = key >> 20;
        const bool a = valid & (fast ? key > kZeroKey : hi > t1);
        const bool c = valid & (fast ? key == kZeroKey : hi == t1);
        ks[u][j] = key;
        na[u] += a;
        nc[u] += c;
        if (!fast)
          atomicAdd(&hist[c ? static_cast<int>((key >> 8) & 0xFFFu) : trash], 1);
      },
      {
        const uint64_t packed =
            static_cast<uint64_t>(na[0]) | (static_cast<uint64_t>(nc[0]) << 16) |
            (static_cast<uint64_t>(na[1]) << 32) | (static_cast<uint64_t>(nc[1]) << 48);
        uint64_t tot;
        const uint64_t ex = block_excl_scan<uint64_t>(packed, tot, s_w64);
        const int tot_a0 = static_cast<int>(tot & 0xFFFF);
        const int tot_c0 = static_cast<int>((tot >> 16) & 0xFFFF);
        const int tot_a1 = static_cast<int>((tot >> 32) & 0xFFFF);
        const int tot_c1 = static_cast<int>(tot >> 48);
        _Pragma("unroll") for (int u = 0; u < kSplitUnroll; ++u) {
          int pa = off_a + (u ? tot_a0 + static_cast<int>((ex >> 32) & 0xFFFF)
                              : static_cast<int>(ex & 0xFFFF));
          int pc = off_c + (u ? tot_c0 + static_cast<int>(ex >> 48)
                              : static_cast<int>((ex >> 16) & 0xFFFF));
          const int i0 = base + u * kBlockTok + tid * 4;
          _Pragma("unroll") for (int j = 0; j < 4; ++j) {
            const uint32_t key = ks[u][j];
            const bool valid = i0 + j < end;
            const bool a = valid & (fast ? key > kZeroKey : (key >> 20) > t1);
            const bool c = valid & (fast ? key == kZeroKey : (key >> 20) == t1);
            if (a && pa < kCapA) ea[pa] = i0 + j;
            if (c) {
              if (pc < lim_c) ec[pc] = make_int2(i0 + j, static_cast<int>(key));
              ++pc;
            }
            pa += a;
          }
        }
        off_a += tot_a0 + tot_a1;
        off_c += tot_c0 + tot_c1;
      })
  if (tid == 0) {
    cnt_a[s] = off_a;
    cnt_c[s] = off_c < cap_c ? off_c : cap_c;
  }
  if (!fast) {
    __syncthreads();
    int* gh = ghist2 + static_cast<int64_t>(row) * kBins;
    for (int i = tid; i < kBins; i += kThreads) {
      const int h = hist[i];
      if (h) atomicAdd(&gh[i], h);
    }
  }
}
#undef FOR_EACH_STEP

template <bool kVec>
__global__ __launch_bounds__(kThreads) void split_refine_kernel(
    const float* __restrict__ score, int64_t score_stride, int width,
    const int16_t* __restrict__ nonzero_mask, int64_t mask_stride,
    const int32_t* __restrict__ lengths, int S,
    const int* __restrict__ ghist2, int* __restrict__ gmeta,
    const int* __restrict__ gcnt_ac, const int2* __restrict__ ent_c,
    int* __restrict__ gref) {
  __shared__ int hist[256 + kWave];
  __shared__ int s_wtot[kWaves];
  __shared__ int s_res[2];
  __shared__ int s_gt;
  const int row = blockIdx.x % (gridDim.x / S);
  const int s = blockIdx.x / (gridDim.x / S);
  const int tid = threadIdx.x;
  const int lane = lane_id();
  int len = lengths[row];
  len = len < width ? len : width;
  if (len <= kTopK) return;
  const int slice = row_slice(len, S);
  const int begin = s * slice;
  if (begin >= len || gmeta[row * kMeta + 0]) return;  // fast path: thr is +0
  const int end = begin + slice < len ? begin + slice : len;

  const uint32_t t1 = static_cast<uint32_t>(gmeta[row * kMeta + 1]);
  const int need1 = gmeta[row * kMeta + 2];
  find_threshold<kBins>(ghist2 + static_cast<int64_t>(row) * kBins, need1,
                        s_wtot, s_res);
  const uint32_t prefix24 = (t1 << 12) | static_cast<uint32_t>(s_res[0]);
  if (s == 0 && tid == 0) {
    gmeta[row * kMeta + 3] = static_cast<int>(prefix24);
    gmeta[row * kMeta + 4] = need1 - s_res[1];
  }
  if (tid < 256) hist[tid] = 0;
  if (tid == 0) s_gt = 0;
  __syncthreads();
  const int n_c = gcnt_ac[(row * 2 + 1) * S + s];
  int gt = 0;
  if (n_c <= kCapC) {
    const int2* ec = ent_c + (static_cast<int64_t>(row) * S + s) * kCapC;
    for (int e = tid; e < n_c; e += kThreads) {
      const uint32_t key = static_cast<uint32_t>(ec[e].y);
      const uint32_t hi = key >> 8;
      gt += hi > prefix24;
      atomicAdd(&hist[hi == prefix24 ? static_cast<int>(key & 0xFFu) : 256 + lane], 1);
    }
  } else {
    // List C overflowed: re-scan the slice for the threshold-bin tokens.
    for_each_block<kVec>(
        score + row * score_stride, nonzero_mask + row * mask_stride, begin, end,
        [&](int, uint32_t vv, const uint32_t(&k)[4]) {
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            const bool c = ((vv >> j) & 1u) & ((k[j] >> 20) == t1);
            const uint32_t hi = k[j] >> 8;
            gt += c & (hi > prefix24);
            atomicAdd(&hist[c & (hi == prefix24) ? static_cast<int>(k[j] & 0xFFu)
                                                 : 256 + lane], 1);
          }
        });
  }
  gt = wave_sum(gt);
  if (lane == 0 && gt) atomicAdd(&s_gt, gt);
  __syncthreads();
  int* out = gref + (static_cast<int64_t>(row) * S + s) * kRefine;
  if (tid < 256) out[1 + tid] = hist[tid];
  if (tid == 0) out[0] = s_gt;
}

template <bool kVec>
__global__ __launch_bounds__(kThreads) void split_select_kernel(
    const float* __restrict__ score, int64_t score_stride, int width,
    const int16_t* __restrict__ nonzero_mask, int64_t mask_stride,
    const int32_t* __restrict__ lengths,
    const int32_t* __restrict__ src_page_table, int64_t src_stride,
    int32_t* __restrict__ dst_page_table, int S,
    const int* __restrict__ gmeta, const int* __restrict__ gcnt_ac,
    const int* __restrict__ ent_a, const int2* __restrict__ ent_c,
    const int* __restrict__ gref) {
  __shared__ int hist[256];
  __shared__ int s_wtot[kWaves];
  __shared__ int s_res[2];
  __shared__ int s_red[4];
  __shared__ int s_tie_before;
  const int row = blockIdx.x % (gridDim.x / S);
  const int s = blockIdx.x / (gridDim.x / S);
  const int tid = threadIdx.x;
  const int lane = lane_id();
  int len = lengths[row];
  len = len < width ? len : width;
  const int32_t* src = src_page_table + row * src_stride;
  int32_t* dst = dst_page_table + static_cast<int64_t>(row) * kTopK;
  if (len <= kTopK) {
    if (s == 0)
      for (int i = tid; i < kTopK; i += kThreads) dst[i] = i < len ? src[i] : -1;
    return;
  }
  const int slice = row_slice(len, S);
  const int begin = s * slice;
  if (begin >= len) return;
  const int end = begin + slice < len ? begin + slice : len;
  const int n_slices = (len + slice - 1) / slice;

  const int* cnt_a = gcnt_ac + (row * 2) * S;
  const int* cnt_c = gcnt_ac + (row * 2 + 1) * S;
  int n_a = 0;
  int pre_a = 0;
  int pre_c = 0;
  for (int r = 0; r < n_slices; ++r) {
    const int a = cnt_a[r];
    n_a += a;
    pre_a += r < s ? a : 0;
    pre_c += r < s ? cnt_c[r] : 0;
  }
  const int my_a = cnt_a[s];
  const int my_c = cnt_c[s];
  const int64_t slot = static_cast<int64_t>(row) * S + s;
  const int* ea = ent_a + slot * kCapA;
  const int2* ec = ent_c + slot * kCapC;
  for (int i = tid; i < my_a; i += kThreads) dst[pre_a + i] = src[ea[i]];

  if (gmeta[row * kMeta + 0]) {
    // Fast path: list C is the zero ties in token order (never overflows).
    const int take = kTopK - n_a - pre_c;
    for (int i = tid; i < my_c && i < take; i += kThreads)
      dst[n_a + pre_c + i] = src[ec[i].x];
    return;
  }

  const uint32_t t1 = static_cast<uint32_t>(gmeta[row * kMeta + 1]);
  const uint32_t prefix24 = static_cast<uint32_t>(gmeta[row * kMeta + 3]);
  const int need2 = gmeta[row * kMeta + 4];
  // Row-wide key[7:0] histogram + this slice's predecessors' counts per bin.
  const int* ref = gref + static_cast<int64_t>(row) * S * kRefine;
  int pre_bin = 0;
  int pre_gt = 0;
  if (tid < 256) {
    int tot = 0;
    for (int r = 0; r < n_slices; ++r) {
      const int h = ref[r * kRefine + 1 + tid];
      tot += h;
      pre_bin += r < s ? h : 0;
    }
    hist[tid] = tot;
  }
  for (int r = 0; r < s; ++r) pre_gt += ref[r * kRefine];
  __syncthreads();
  find_threshold<256>(hist, need2, s_wtot, s_res);
  const uint32_t t3 = static_cast<uint32_t>(s_res[0]);
  const uint32_t thr = (prefix24 << 8) | t3;
  const int remaining = need2 - s_res[1];
  const int tie_base = kTopK - remaining;
  {
    int above = tid < 256 && static_cast<uint32_t>(tid) > t3 ? pre_bin : 0;
    above = wave_sum(above);
    if (tid < 256 && lane == 0) s_red[tid / kWave] = above;
    if (static_cast<uint32_t>(tid) == t3) s_tie_before = pre_bin;
  }
  __syncthreads();
  int gt_pos = n_a + pre_gt + s_red[0] + s_red[1] + s_red[2] + s_red[3];
  int tie_rank = s_tie_before;
  if (gt_pos >= tie_base && tie_rank >= remaining) return;

  if (my_c <= kCapC) {
    for (int c0 = 0; c0 < my_c; c0 += kThreads) {
      const int e = c0 + tid;
      const int2 ent = e < my_c ? ec[e] : make_int2(0, 0);
      const uint32_t key = static_cast<uint32_t>(ent.y);
      const bool gt = e < my_c && key > thr;
      const bool eq = e < my_c && key == thr;
      int tot;
      const int ex = block_excl_scan<int>(gt | (eq << 16), tot, s_wtot);
      if (gt) dst[gt_pos + (ex & 0xFFFF)] = src[ent.x];
      if (eq) {
        const int rk = tie_rank + (ex >> 16);
        if (rk < remaining) dst[tie_base + rk] = src[ent.x];
      }
      gt_pos += tot & 0xFFFF;
      tie_rank += tot >> 16;
    }
    return;
  }
  // List C overflowed: ordered compaction straight from the scores.
  for_each_block<kVec>(
      score + row * score_stride, nonzero_mask + row * mask_stride, begin, end,
      [&](int i0, uint32_t vv, const uint32_t(&k)[4]) {
        int ng = 0;
        int ne = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const bool valid = (vv >> j) & 1u;
          ng += valid & ((k[j] >> 20) == t1) & (k[j] > thr);
          ne += valid & (k[j] == thr);
        }
        int tot;
        const int ex = block_excl_scan<int>(ng | (ne << 16), tot, s_wtot);
        int pg = gt_pos + (ex & 0xFFFF);
        int pq = tie_rank + (ex >> 16);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const bool valid = (vv >> j) & 1u;
          if (valid & ((k[j] >> 20) == t1) & (k[j] > thr)) dst[pg++] = src[i0 + j];
          if (valid & (k[j] == thr)) {
            if (pq < remaining) dst[tie_base + pq] = src[i0 + j];
            ++pq;
          }
        }
        gt_pos += tot & 0xFFFF;
        tie_rank += tot >> 16;
      });
}

}  // namespace

static void check_inputs(
    const torch::Tensor& score, const torch::Tensor& nonzero_mask,
    const torch::Tensor& lengths, const torch::Tensor& page_table_size_1,
    const torch::Tensor& cu_seqlens_q, int64_t topk) {
  TORCH_CHECK(topk == kTopK, "sparse-mask TopK requires topk=2048");
  TORCH_CHECK(score.is_cuda() && score.scalar_type() == at::kFloat &&
                  score.dim() == 2 && score.stride(1) == 1,
              "score must be row-contiguous float32 [rows, width]");
  const int64_t rows = score.size(0);
  TORCH_CHECK(nonzero_mask.scalar_type() == at::kShort &&
                  nonzero_mask.dim() == 2 && nonzero_mask.stride(1) == 1 &&
                  nonzero_mask.size(0) == rows &&
                  nonzero_mask.size(1) == (score.size(1) + 15) / 16,
              "nonzero_mask must be row-contiguous int16 [rows, ceil(width/16)]");
  TORCH_CHECK(lengths.scalar_type() == at::kInt && lengths.dim() == 1 &&
                  lengths.is_contiguous() && lengths.size(0) == rows,
              "lengths must be contiguous int32 [rows]");
  TORCH_CHECK(page_table_size_1.scalar_type() == at::kInt &&
                  page_table_size_1.dim() == 2 &&
                  page_table_size_1.stride(1) == 1 &&
                  page_table_size_1.size(0) == rows,
              "page_table_size_1 must be row-contiguous int32 [rows, *]");
  TORCH_CHECK(cu_seqlens_q.size(0) == rows + 1,
              "only the decode layout (one page-table row per score row) is supported");
}

static bool use_vec(const torch::Tensor& score) {
  return score.size(1) % 4 == 0 && score.stride(0) % 4 == 0 &&
         reinterpret_cast<uintptr_t>(score.data_ptr<float>()) % 16 == 0;
}

static void launch_single(
    const torch::Tensor& score, const torch::Tensor& nonzero_mask,
    const torch::Tensor& lengths, const torch::Tensor& page_table_size_1,
    torch::Tensor& dst, hipStream_t stream) {
  const dim3 grid(static_cast<uint32_t>(score.size(0)));
  const dim3 block(kThreads);
#define LAUNCH(V)                                                           \
  sparse_mask_topk_decode_kernel<V><<<grid, block, 0, stream>>>(            \
      score.data_ptr<float>(), score.stride(0),                             \
      static_cast<int>(score.size(1)), nonzero_mask.data_ptr<int16_t>(),    \
      nonzero_mask.stride(0), lengths.data_ptr<int32_t>(),                  \
      page_table_size_1.data_ptr<int32_t>(), page_table_size_1.stride(0),   \
      dst.data_ptr<int32_t>())
  if (use_vec(score)) LAUNCH(true);
  else LAUNCH(false);
#undef LAUNCH
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor sparse_mask_topk_transform(
    const torch::Tensor& score, const torch::Tensor& nonzero_mask,
    const torch::Tensor& lengths, const torch::Tensor& page_table_size_1,
    const torch::Tensor& cu_seqlens_q, int64_t topk) {
  check_inputs(score, nonzero_mask, lengths, page_table_size_1, cu_seqlens_q, topk);
  const int64_t rows = score.size(0);
  const c10::cuda::CUDAGuard guard(score.device());
  auto dst = torch::empty({rows, kTopK}, score.options().dtype(at::kInt));
  if (rows == 0) return dst;
  launch_single(score, nonzero_mask, lengths, page_table_size_1, dst,
                at::cuda::getCurrentCUDAStream().stream());
  return dst;
}

torch::Tensor sparse_mask_topk_transform_split(
    const torch::Tensor& score, const torch::Tensor& nonzero_mask,
    const torch::Tensor& lengths, const torch::Tensor& page_table_size_1,
    const torch::Tensor& cu_seqlens_q, int64_t topk, int64_t max_splits) {
  check_inputs(score, nonzero_mask, lengths, page_table_size_1, cu_seqlens_q, topk);
  const int64_t rows = score.size(0);
  const c10::cuda::CUDAGuard guard(score.device());
  auto dst = torch::empty({rows, kTopK}, score.options().dtype(at::kInt));
  if (rows == 0) return dst;
  const auto stream = at::cuda::getCurrentCUDAStream().stream();
  const int width = static_cast<int>(score.size(1));
  // One 1024-thread CTA per CU: split only while rows * S fits one wave of
  // CTAs (S is fixed by the row count, so this is graph-capture safe).  With
  // S == 1 the single-CTA kernel is faster.  Each row uses <= S CTAs of whole
  // kStepTok blocks (row_slice), surplus CTAs exit at once.
  const int64_t n_cu = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  const int64_t S = std::min<int64_t>(
      std::min<int64_t>(std::max<int64_t>((width + kStepTok - 1) / kStepTok, 1),
                        n_cu / rows),
      std::min<int64_t>(max_splits, kMaxSplit));
  if (S <= 1) {
    launch_single(score, nonzero_mask, lengths, page_table_size_1, dst, stream);
    return dst;
  }
  const auto iopt = score.options().dtype(at::kInt);
  // [ghist1 rows*4096 | ghist2 rows*4096 | cnt rows*2] must start zeroed.
  auto zws = torch::zeros({rows * (2 * kBins + 2)}, iopt);
  auto meta = torch::empty({rows * kMeta}, iopt);
  auto cnt_ac = torch::empty({rows * 2 * S}, iopt);
  auto ref = torch::empty({rows * S * kRefine}, iopt);
  auto ent_a = torch::empty({rows * S * kCapA}, iopt);
  auto ent_c = torch::empty({rows * S * kCapC * 2}, iopt);
  int* ghist1 = zws.data_ptr<int>();
  int* ghist2 = ghist1 + rows * kBins;
  int* gcnt = ghist2 + rows * kBins;
  const bool vec = use_vec(score);
  const dim3 grid(static_cast<uint32_t>(rows * S));
  const dim3 block(kThreads);
  const int Si = static_cast<int>(S);
  int2* ec = reinterpret_cast<int2*>(ent_c.data_ptr<int>());
#define LAUNCH_SPLIT(V)                                                       \
  split_hist_kernel<V><<<grid, block, 0, stream>>>(                           \
      score.data_ptr<float>(), score.stride(0), width,                        \
      nonzero_mask.data_ptr<int16_t>(), nonzero_mask.stride(0),               \
      lengths.data_ptr<int32_t>(), Si, ghist1, gcnt);                         \
  split_compact_kernel<V><<<grid, block, 0, stream>>>(                        \
      score.data_ptr<float>(), score.stride(0), width,                        \
      nonzero_mask.data_ptr<int16_t>(), nonzero_mask.stride(0),               \
      lengths.data_ptr<int32_t>(), Si, ghist1, gcnt, ghist2,                  \
      meta.data_ptr<int>(), cnt_ac.data_ptr<int>(), ent_a.data_ptr<int>(),    \
      ec);                                                                    \
  split_refine_kernel<V><<<grid, block, 0, stream>>>(                         \
      score.data_ptr<float>(), score.stride(0), width,                        \
      nonzero_mask.data_ptr<int16_t>(), nonzero_mask.stride(0),               \
      lengths.data_ptr<int32_t>(), Si, ghist2, meta.data_ptr<int>(),          \
      cnt_ac.data_ptr<int>(), ec, ref.data_ptr<int>());                       \
  split_select_kernel<V><<<grid, block, 0, stream>>>(                         \
      score.data_ptr<float>(), score.stride(0), width,                        \
      nonzero_mask.data_ptr<int16_t>(), nonzero_mask.stride(0),               \
      lengths.data_ptr<int32_t>(), page_table_size_1.data_ptr<int32_t>(),     \
      page_table_size_1.stride(0), dst.data_ptr<int32_t>(), Si,               \
      meta.data_ptr<int>(), cnt_ac.data_ptr<int>(), ent_a.data_ptr<int>(),    \
      ec, ref.data_ptr<int>())
  if (vec) {
    LAUNCH_SPLIT(true);
  } else {
    LAUNCH_SPLIT(false);
  }
#undef LAUNCH_SPLIT
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return dst;
}

// sparse_mask_topk_transform_split with pass 1 already done by
// mqa_logits_hist(hist_lens=lengths): zws is its [ghist1 | ghist2 | cnt]
// workspace for these rows / lengths / width.  Same splits, same output.
torch::Tensor sparse_mask_topk_transform_split_pre(
    const torch::Tensor& score, const torch::Tensor& nonzero_mask,
    const torch::Tensor& lengths, const torch::Tensor& page_table_size_1,
    const torch::Tensor& cu_seqlens_q, int64_t topk, int64_t max_splits,
    const torch::Tensor& zws) {
  check_inputs(score, nonzero_mask, lengths, page_table_size_1, cu_seqlens_q, topk);
  const int64_t rows = score.size(0);
  const c10::cuda::CUDAGuard guard(score.device());
  auto dst = torch::empty({rows, kTopK}, score.options().dtype(at::kInt));
  if (rows == 0) return dst;
  const auto stream = at::cuda::getCurrentCUDAStream().stream();
  const int width = static_cast<int>(score.size(1));
  const int64_t n_cu = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  const int64_t S = std::min<int64_t>(
      std::min<int64_t>(std::max<int64_t>((width + kStepTok - 1) / kStepTok, 1),
                        n_cu / rows),
      std::min<int64_t>(max_splits, kMaxSplit));
  if (S <= 1) {
    launch_single(score, nonzero_mask, lengths, page_table_size_1, dst, stream);
    return dst;
  }
  TORCH_CHECK(zws.scalar_type() == at::kInt && zws.is_contiguous() &&
                  zws.numel() == rows * (2 * kBins + 2),
              "zws must be the int32 [rows * (2 * 4096 + 2)] pass-1 workspace");
  const auto iopt = score.options().dtype(at::kInt);
  auto meta = torch::empty({rows * kMeta}, iopt);
  auto cnt_ac = torch::empty({rows * 2 * S}, iopt);
  auto ref = torch::empty({rows * S * kRefine}, iopt);
  auto ent_a = torch::empty({rows * S * kCapA}, iopt);
  auto ent_c = torch::empty({rows * S * kCapC * 2}, iopt);
  int* ghist1 = const_cast<int*>(zws.data_ptr<int>());
  int* ghist2 = ghist1 + rows * kBins;
  int* gcnt = ghist2 + rows * kBins;
  const bool vec = use_vec(score);
  const dim3 grid(static_cast<uint32_t>(rows * S));
  const dim3 block(kThreads);
  const int Si = static_cast<int>(S);
  int2* ec = reinterpret_cast<int2*>(ent_c.data_ptr<int>());
#define LAUNCH_SPLIT_PRE(V)                                                   \
  split_compact_kernel<V><<<grid, block, 0, stream>>>(                        \
      score.data_ptr<float>(), score.stride(0), width,                        \
      nonzero_mask.data_ptr<int16_t>(), nonzero_mask.stride(0),               \
      lengths.data_ptr<int32_t>(), Si, ghist1, gcnt, ghist2,                  \
      meta.data_ptr<int>(), cnt_ac.data_ptr<int>(), ent_a.data_ptr<int>(),    \
      ec);                                                                    \
  split_refine_kernel<V><<<grid, block, 0, stream>>>(                         \
      score.data_ptr<float>(), score.stride(0), width,                        \
      nonzero_mask.data_ptr<int16_t>(), nonzero_mask.stride(0),               \
      lengths.data_ptr<int32_t>(), Si, ghist2, meta.data_ptr<int>(),          \
      cnt_ac.data_ptr<int>(), ec, ref.data_ptr<int>());                       \
  split_select_kernel<V><<<grid, block, 0, stream>>>(                         \
      score.data_ptr<float>(), score.stride(0), width,                        \
      nonzero_mask.data_ptr<int16_t>(), nonzero_mask.stride(0),               \
      lengths.data_ptr<int32_t>(), page_table_size_1.data_ptr<int32_t>(),     \
      page_table_size_1.stride(0), dst.data_ptr<int32_t>(), Si,               \
      meta.data_ptr<int>(), cnt_ac.data_ptr<int>(), ent_a.data_ptr<int>(),    \
      ec, ref.data_ptr<int>())
  if (vec) {
    LAUNCH_SPLIT_PRE(true);
  } else {
    LAUNCH_SPLIT_PRE(false);
  }
#undef LAUNCH_SPLIT_PRE
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return dst;
}
