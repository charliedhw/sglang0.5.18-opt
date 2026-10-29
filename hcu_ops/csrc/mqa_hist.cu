// GLM-5.3 DSA indexer logits for the sparse-mask TopK: drop-in for lightop
// page_mqa_logits_sparse_mask[_grouped] (native fp8 path, same bits), with an
// optional fused pass 1 of sparse_mask_topk_transform_split.
//
// lightop's grouped native-fp8 kernel stages every 16-token KV tile through
// LDS with a full vmcnt(0) wait per tile and reads a page's block-table entry
// only when it starts on that page.  With decode grids of 1-2 requests per rank
// (<= 4 waves / CU) it is latency bound: ~94 us for a 64K context, against
// ~10 us of KV traffic.  Here every wave loads the same operand fragments
// straight from global memory into VGPRs and prefetches the next page (KV +
// scales one page ahead, block-table entry two pages ahead).  The MMAC order,
// reduction order and mask / logits writes are lightop's:
//   acc[row][q_iter] over k_block (front, rear 8 bytes)
//   sum += fmaxf(acc, 0) * w over (q_iter, item); shfl_down 16, 32; * kv_scale
//   mask bit = token < len && bits != 0; logits written only where the bit is set
// so logits (at set bits) and mask are bit-identical.
//
// kHist: also builds what split_hist_kernel computes for rows with
// min(hist_len, width) > 2048: the 12-bit key[31:20] histogram (ghist1) and the
// positive / zero counts (gcnt).  Per-CTA counts live in 16-bit packed LDS
// words (host keeps tokens per CTA-row < 2^16) and are flushed with global
// atomics.  mqa_init_kernel replaces the two memsets (mask, TopK workspace) and
// preloads the zero bin / zero count with the row length, from which every
// CTA subtracts its nonzero tokens.  sparse_mask_topk_transform_split_pre then
// runs passes 2-4 only.
#include <ATen/hip/HIPContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <hip/hip_runtime.h>
#include <torch/extension.h>

#include <algorithm>
#include <cstdint>

namespace {

using floatx4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;
using intx2 = __attribute__((__vector_size__(2 * sizeof(int)))) int;
using uintx4 = __attribute__((__vector_size__(4 * sizeof(unsigned)))) unsigned;

constexpr int kHeadDim = 128;
constexpr int kHeads = 32;
constexpr int kPage = 64;
constexpr int kTile = 16;
constexpr int kWaves = 4;
constexpr int kThreads = kWaves * 64;
constexpr int kTokPerCta = kWaves * kPage;
constexpr int kTopK = 2048;
constexpr int kBins = 4096;
constexpr int kHistWords = kBins / 2;
constexpr int kZeroBin = 0x800;  // kZeroKey >> 20
constexpr int kMaxCtaRowTokens = 65535;

__device__ __forceinline__ uint32_t to_key(float x) {
  const uint32_t b = __float_as_uint(x);
  return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

__device__ __forceinline__ int wave_sum(int x) {
#pragma unroll
  for (int o = 32; o > 0; o >>= 1) x += __shfl_xor(x, o);
  return x;
}

// one 16-byte fragment = two mmac k-slices (front / rear 8 bytes), A = KV, B = Q
__device__ __forceinline__ void mma2(const uintx4& kv, const uintx4& q, floatx4& c) {
  c = __builtin_hcu_mmac_f32_16x16x32_fp8_fp8_lit_lts(intx2{(int)kv[0], (int)kv[1]}, intx2{(int)q[0], (int)q[1]}, c,
                                                      false, false);
  c = __builtin_hcu_mmac_f32_16x16x32_fp8_fp8_lit_lts(intx2{(int)kv[2], (int)kv[3]}, intx2{(int)q[2], (int)q[3]}, c,
                                                      false, false);
}

struct PageFrag {
  uintx4 kv[4][2];  // [16-token tile][k_block]
  float scale[4];
};

// Lane l: token tile*16 + l%16, bytes (l/16)*16 + k_block*64 .. +16 (the LDS
// fragment lightop reads); scale of token tile*16 + l%16.
__device__ __forceinline__ void load_page(const uint8_t* __restrict__ kv, int64_t kv_stride, int phys, int lane,
                                          PageFrag& f) {
  const uint8_t* page = kv + static_cast<int64_t>(phys) * kv_stride;
  const uint8_t* p = page + (lane & 15) * kHeadDim + (lane >> 4) * 16;
#pragma unroll
  for (int t = 0; t < 4; ++t) {
#pragma unroll
    for (int kb = 0; kb < 2; ++kb) f.kv[t][kb] = *reinterpret_cast<const uintx4*>(p + t * kTile * kHeadDim + kb * 64);
  }
  const float* sc = reinterpret_cast<const float*>(page + kPage * kHeadDim);
#pragma unroll
  for (int t = 0; t < 4; ++t) f.scale[t] = sc[t * kTile + (lane & 15)];
}

// grid: pctas * requests * (G / R) CTAs, ordered (request, cta_seed, pair) as in
// lightop; wave w of CTA seed c owns pages (c + k * pctas) * 4 + w.  All G rows
// of a request share the first row's block table.
template <int G, int R, bool kHist>
__global__ __launch_bounds__(kThreads) void mqa_logits_kernel(
    const uint8_t* __restrict__ q, const uint8_t* __restrict__ kv, int64_t kv_stride,
    const float* __restrict__ w, int64_t logits_stride, int64_t bt_stride,
    const int* __restrict__ ctx_lens, float* __restrict__ logits, const int* __restrict__ bt,
    int16_t* __restrict__ mask, int pctas, const int* __restrict__ hist_lens,
    int* __restrict__ ghist1, int* __restrict__ gcnt) {
  static_assert(G % R == 0, "rows per CTA must divide the group");
  constexpr int P = G / R;
  extern __shared__ uint32_t s_hist[];  // kHist: [R][kHistWords], two 16-bit bins per word
  const int tid = threadIdx.x;
  const int wave = tid >> 6;
  const int lane = tid & 63;
  const int bpr = pctas * P;
  const int request = blockIdx.x / bpr;
  const int rb = blockIdx.x - request * bpr;
  const int cta_seed = rb / P;
  const int pair = rb - cta_seed * P;
  const int group_row = request * G;
  const int row0 = group_row + pair * R;
  const int limit = static_cast<int>(logits_stride);

  int len[R];
  int hlim[R];
  int max_len = 0;
#pragma unroll
  for (int r = 0; r < R; ++r) {
    const int l = ctx_lens[row0 + r];
    len[r] = l < 0 ? 0 : (l > limit ? limit : l);
    max_len = len[r] > max_len ? len[r] : max_len;
    hlim[r] = 0;
    if constexpr (kHist) {
      int h = hist_lens[row0 + r];
      h = h < limit ? h : limit;
      hlim[r] = h > kTopK ? h : 0;
    }
  }
  if (cta_seed * kTokPerCta >= max_len) return;  // uniform, before any barrier

  if constexpr (kHist) {
    for (int i = tid; i < R * kHistWords; i += kThreads) s_hist[i] = 0;
    __syncthreads();
  }

  // Q / weight fragments in lightop's layout: lane l holds head q_iter*16 + l%16,
  // bytes (l/16)*16 + k_block*64 .. +16, and the weights of heads
  // q_iter*16 + item*4 + l/16 (mmac result element item of lane l).
  uintx4 rq[R][2][2];
  float rw[R][8];
  const int q_lane = (lane & 15) * kHeadDim + (lane >> 4) * 16;
#pragma unroll
  for (int r = 0; r < R; ++r) {
    const uint8_t* qr = q + static_cast<int64_t>(row0 + r) * (kHeads * kHeadDim) + q_lane;
#pragma unroll
    for (int qi = 0; qi < 2; ++qi) {
#pragma unroll
      for (int kb = 0; kb < 2; ++kb)
        rq[r][qi][kb] = *reinterpret_cast<const uintx4*>(qr + qi * kTile * kHeadDim + kb * 64);
#pragma unroll
      for (int item = 0; item < 4; ++item)
        rw[r][qi * 4 + item] = w[(row0 + r) * kHeads + qi * kTile + item * 4 + (lane >> 4)];
    }
  }

  const int n_pages = (max_len + kPage - 1) / kPage;
  const int step = pctas * kWaves;
  const int* btr = bt + static_cast<int64_t>(group_row) * bt_stride;
  const int64_t mask_stride = (logits_stride + kTile - 1) / kTile;
  int nz[R];
  int pos[R];
#pragma unroll
  for (int r = 0; r < R; ++r) nz[r] = pos[r] = 0;

  int pg = cta_seed * kWaves + wave;
  if (pg < n_pages) {
    PageFrag cur;
    load_page(kv, kv_stride, btr[pg], lane, cur);
    int nxt_pg = pg + step;
    int nxt_phys = btr[nxt_pg < n_pages ? nxt_pg : pg];
    while (true) {
      // Branch-free prefetch: past the end the current page is re-read.
      const int nn_pg = nxt_pg + step;
      const int nn_phys = btr[nn_pg < n_pages ? nn_pg : pg];
      PageFrag nxt;
      load_page(kv, kv_stride, nxt_phys, lane, nxt);

#pragma unroll
      for (int t = 0; t < 4; ++t) {
        floatx4 acc[R][2];
#pragma unroll
        for (int r = 0; r < R; ++r) acc[r][0] = acc[r][1] = floatx4{0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
        for (int kb = 0; kb < 2; ++kb) {
#pragma unroll
          for (int r = 0; r < R; ++r) {
#pragma unroll
            for (int qi = 0; qi < 2; ++qi) mma2(cur.kv[t][kb], rq[r][qi][kb], acc[r][qi]);
          }
        }
        float s[R];
#pragma unroll
        for (int r = 0; r < R; ++r) {
          s[r] = 0.0f;
#pragma unroll
          for (int qi = 0; qi < 2; ++qi) {
#pragma unroll
            for (int item = 0; item < 4; ++item) s[r] += fmaxf(acc[r][qi][item], 0.0f) * rw[r][qi * 4 + item];
          }
          s[r] += __shfl_down(s[r], 16);
          s[r] += __shfl_down(s[r], 32);
          s[r] *= cur.scale[t];
        }
        const int tile_off = pg * kPage + t * kTile;
        const int token = tile_off + lane;
#pragma unroll
        for (int r = 0; r < R; ++r) {
          const bool keep = lane < kTile && token < len[r] && __float_as_uint(s[r]) != 0u;
          const uint64_t bits = __ballot(keep);
          if (lane == 0 && tile_off < limit)
            mask[static_cast<int64_t>(row0 + r) * mask_stride + tile_off / kTile] =
                static_cast<int16_t>(bits & 0xffffu);
          if (keep) {
            logits[static_cast<int64_t>(row0 + r) * logits_stride + token] = s[r];
            if constexpr (kHist) {
              if (token < hlim[r]) {
                const uint32_t key = to_key(s[r]);
                atomicAdd(&s_hist[r * kHistWords + (key >> 21)], 1u << ((key >> 16) & 16u));
                ++nz[r];
                pos[r] += key > 0x80000000u;
              }
            }
          }
        }
      }

      pg = nxt_pg;
      if (pg >= n_pages) break;
      cur = nxt;
      nxt_pg = nn_pg;
      nxt_phys = nn_phys;
    }
  }

  if constexpr (kHist) {
    __shared__ int s_cnt[2 * R];
    if (tid < 2 * R) s_cnt[tid] = 0;
    __syncthreads();
#pragma unroll
    for (int r = 0; r < R; ++r) {
      const int a = wave_sum(nz[r]);
      const int b = wave_sum(pos[r]);
      if (lane == 0 && a) {
        atomicAdd(&s_cnt[2 * r], a);
        atomicAdd(&s_cnt[2 * r + 1], b);
      }
    }
    __syncthreads();
#pragma unroll
    for (int r = 0; r < R; ++r) {
      if (hlim[r] == 0) continue;
      int* gh = ghist1 + static_cast<int64_t>(row0 + r) * kBins;
      for (int i = tid; i < kHistWords; i += kThreads) {
        const uint32_t v = s_hist[r * kHistWords + i];
        if (v & 0xffffu) atomicAdd(&gh[2 * i], static_cast<int>(v & 0xffffu));
        if (v >> 16) atomicAdd(&gh[2 * i + 1], static_cast<int>(v >> 16));
      }
      if (tid == 0) {
        const int n = s_cnt[2 * r];
        const int p = s_cnt[2 * r + 1];
        if (n) {
          atomicSub(&gh[kZeroBin], n);
          atomicSub(&gcnt[2 * (row0 + r) + 1], n);
        }
        if (p) atomicAdd(&gcnt[2 * (row0 + r)], p);
      }
    }
  }
}

// One CTA per row: zero the mask row; with ghist1, initialise the TopK
// workspace [ghist1 | ghist2 | cnt] as split pass 1 expects it before its
// nonzero tokens are added (all tokens counted as zeros).
__global__ __launch_bounds__(256) void mqa_init_kernel(
    int16_t* __restrict__ mask, int64_t mask_stride, const int* __restrict__ hist_lens, int width,
    int* __restrict__ ghist1, int* __restrict__ ghist2, int* __restrict__ gcnt) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  int16_t* m = mask + row * mask_stride;
  if (mask_stride % 8 == 0) {
    int4* m4 = reinterpret_cast<int4*>(m);
    for (int64_t i = tid; i < mask_stride / 8; i += 256) m4[i] = make_int4(0, 0, 0, 0);
  } else {
    for (int64_t i = tid; i < mask_stride; i += 256) m[i] = 0;
  }
  if (ghist1 == nullptr) return;
  int h = hist_lens[row];
  h = h < width ? h : width;
  const int zeros = h > kTopK ? h : 0;
  int4* g1 = reinterpret_cast<int4*>(ghist1 + static_cast<int64_t>(row) * kBins);
  int4* g2 = reinterpret_cast<int4*>(ghist2 + static_cast<int64_t>(row) * kBins);
  for (int i = tid; i < kBins / 4; i += 256) {
    g1[i] = make_int4(i == kZeroBin / 4 ? zeros : 0, 0, 0, 0);
    g2[i] = make_int4(0, 0, 0, 0);
  }
  if (tid == 0) {
    gcnt[2 * row] = 0;
    gcnt[2 * row + 1] = zeros;
  }
}

template <int G, int R>
void launch_logits(bool hist, int64_t blocks, hipStream_t stream, const uint8_t* q, const uint8_t* kv,
                   int64_t kv_stride, const float* w, int64_t logits_stride, int64_t bt_stride, const int* ctx,
                   float* logits, const int* bt, int16_t* mask, int pctas, const int* hl, int* ghist1, int* gcnt) {
  const dim3 grid(static_cast<uint32_t>(blocks));
  if (hist) {
    mqa_logits_kernel<G, R, true><<<grid, kThreads, R * kHistWords * sizeof(uint32_t), stream>>>(
        q, kv, kv_stride, w, logits_stride, bt_stride, ctx, logits, bt, mask, pctas, hl, ghist1, gcnt);
  } else {
    mqa_logits_kernel<G, R, false><<<grid, kThreads, 0, stream>>>(
        q, kv, kv_stride, w, logits_stride, bt_stride, ctx, logits, bt, mask, pctas, nullptr, nullptr, nullptr);
  }
}

}  // namespace

// Returns {logits [B, max_ctx] f32, mask [B, ceil(max_ctx/16)] i16, zws}; zws is
// the split-TopK workspace with pass 1 done when hist_lens is given (else empty).
std::vector<torch::Tensor> mqa_logits_hist(
    const torch::Tensor& q, const torch::Tensor& kv_cache, const torch::Tensor& weights,
    const torch::Tensor& context_lens, const torch::Tensor& block_table, int64_t max_context_len,
    int64_t group_size, int64_t persistent_ctas, int64_t rows_per_cta,
    const c10::optional<torch::Tensor>& hist_lens) {
  TORCH_CHECK(q.is_cuda() && q.scalar_type() == at::kFloat8_e4m3fn && q.dim() == 4 && q.size(1) == 1 &&
                  q.size(2) == kHeads && q.size(3) == kHeadDim && q.is_contiguous(),
              "q must be contiguous fp8 e4m3fn [B, 1, 32, 128]");
  const int64_t B = q.size(0);
  TORCH_CHECK(kv_cache.scalar_type() == at::kByte && kv_cache.dim() == 4 && kv_cache.size(1) == kPage &&
                  kv_cache.size(2) == 1 && kv_cache.size(3) == kHeadDim + 4 &&
                  kv_cache.stride(1) == kHeadDim + 4 && kv_cache.stride(3) == 1 && kv_cache.stride(0) % 16 == 0,
              "kv_cache must be packed fp8+scale uint8 [blocks, 64, 1, 132]");
  TORCH_CHECK(weights.scalar_type() == at::kFloat && weights.dim() == 2 && weights.size(0) == B &&
                  weights.size(1) == kHeads && weights.is_contiguous(),
              "weights must be contiguous float32 [B, 32]");
  TORCH_CHECK(context_lens.scalar_type() == at::kInt && context_lens.dim() == 1 && context_lens.size(0) == B &&
                  context_lens.is_contiguous(),
              "context_lens must be contiguous int32 [B]");
  TORCH_CHECK(block_table.scalar_type() == at::kInt && block_table.dim() == 2 && block_table.size(0) == B &&
                  block_table.stride(1) == 1,
              "block_table must be row-contiguous int32 [B, pages]");
  TORCH_CHECK(max_context_len > 0 && max_context_len == block_table.size(1) * kPage,
              "max_context_len must equal block_table capacity");
  const int G = static_cast<int>(group_size);
  const int R = static_cast<int>(rows_per_cta);
  TORCH_CHECK((G == 1 && R == 1) || (G == 6 && (R == 1 || R == 2 || R == 3)),
              "supported (group_size, rows_per_cta): (1, 1), (6, 1/2/3)");
  TORCH_CHECK(B > 0 && B % G == 0, "batch must be a positive multiple of group_size");
  const bool hist = hist_lens.has_value();
  if (hist) {
    TORCH_CHECK(hist_lens->scalar_type() == at::kInt && hist_lens->dim() == 1 && hist_lens->size(0) == B &&
                    hist_lens->is_contiguous(),
                "hist_lens must be contiguous int32 [B]");
  }
  const c10::cuda::CUDAGuard guard(q.device());
  const auto stream = at::cuda::getCurrentCUDAStream().stream();

  auto logits = torch::empty({B, max_context_len}, q.options().dtype(at::kFloat));
  const int64_t mask_stride = (max_context_len + kTile - 1) / kTile;
  auto mask = torch::empty({B, mask_stride}, q.options().dtype(at::kShort));
  torch::Tensor zws;
  int* ghist1 = nullptr;
  int* ghist2 = nullptr;
  int* gcnt = nullptr;
  if (hist) {
    zws = torch::empty({B * (2 * kBins + 2)}, q.options().dtype(at::kInt));
    ghist1 = zws.data_ptr<int>();
    ghist2 = ghist1 + B * kBins;
    gcnt = ghist2 + B * kBins;
  } else {
    zws = torch::empty({0}, q.options().dtype(at::kInt));
  }
  mqa_init_kernel<<<static_cast<uint32_t>(B), 256, 0, stream>>>(
      mask.data_ptr<int16_t>(), mask_stride, hist ? hist_lens->data_ptr<int>() : nullptr,
      static_cast<int>(max_context_len), ghist1, ghist2, gcnt);

  const int64_t max_chunks = (max_context_len + kTokPerCta - 1) / kTokPerCta;
  int64_t pctas = persistent_ctas <= 0 ? std::min<int64_t>(max_chunks, 32)
                                       : std::min<int64_t>(persistent_ctas, max_chunks);
  if (hist) {
    // 16-bit LDS counters: a CTA sees <= ceil(chunks / pctas) * 256 tokens per row
    while ((max_chunks + pctas - 1) / pctas * kTokPerCta > kMaxCtaRowTokens) ++pctas;
  }
  const int64_t blocks = pctas * (B / G) * (G / R);
  TORCH_CHECK(blocks <= INT32_MAX, "grid too large");
  const auto* qp = reinterpret_cast<const uint8_t*>(q.data_ptr());
  const auto* kvp = kv_cache.data_ptr<uint8_t>();
  const int64_t kv_stride = kv_cache.stride(0);
#define LAUNCH(GG, RR)                                                                                             \
  launch_logits<GG, RR>(hist, blocks, stream, qp, kvp, kv_stride, weights.data_ptr<float>(), logits.stride(0),   \
                        block_table.stride(0), context_lens.data_ptr<int>(), logits.data_ptr<float>(),            \
                        block_table.data_ptr<int>(), mask.data_ptr<int16_t>(), static_cast<int>(pctas),           \
                        hist ? hist_lens->data_ptr<int>() : nullptr, ghist1, gcnt)
  if (G == 1) LAUNCH(1, 1);
  else if (R == 1) LAUNCH(6, 1);
  else if (R == 2) LAUNCH(6, 2);
  else LAUNCH(6, 3);
#undef LAUNCH
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {logits, mask, zws};
}
