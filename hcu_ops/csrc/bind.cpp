#include <torch/extension.h>

torch::Tensor sparse_mask_topk_transform(
    const torch::Tensor& score, const torch::Tensor& nonzero_mask,
    const torch::Tensor& lengths, const torch::Tensor& page_table_size_1,
    const torch::Tensor& cu_seqlens_q, int64_t topk);
torch::Tensor sparse_mask_topk_transform_split(
    const torch::Tensor& score, const torch::Tensor& nonzero_mask,
    const torch::Tensor& lengths, const torch::Tensor& page_table_size_1,
    const torch::Tensor& cu_seqlens_q, int64_t topk, int64_t max_splits);
void skinny_gemm(const torch::Tensor& x, const torch::Tensor& x_scale, const torch::Tensor& w,
                 const torch::Tensor& w_scale, torch::Tensor& y, int64_t nw, int64_t nt, bool stream_w,
                 int64_t ver);
void rope_inplace(const torch::Tensor& positions, torch::Tensor& query, const c10::optional<torch::Tensor>& key,
                  int64_t head_size, const torch::Tensor& cos_sin_cache, bool is_neox);
void rope_qout(const torch::Tensor& positions, const torch::Tensor& query, torch::Tensor& q_out,
               const c10::optional<torch::Tensor>& key, int64_t head_size, const torch::Tensor& cos_sin_cache,
               bool is_neox);
std::vector<torch::Tensor> moe_gate_topk(const torch::Tensor& logits, const torch::Tensor& bias, int64_t topk,
                                         double routed_scaling_factor, bool apply_routed_scaling_factor_on_output,
                                         const c10::optional<torch::Tensor>& num_token_non_padded);
void bmm_skinny(const torch::Tensor& x, const torch::Tensor& w, torch::Tensor& y, int64_t cw, int64_t nwv,
                bool lds);
std::vector<torch::Tensor> silu_mul_fp8_quant_masked(const torch::Tensor& x, const torch::Tensor& masked_m,
                                                     int64_t ctas, int64_t waves_per_row);
torch::Tensor bf16_round_check(const torch::Tensor& like);
void skinny_nn(const torch::Tensor& x, const torch::Tensor& w, torch::Tensor& y, int64_t nw, int64_t h, int64_t pf,
               bool stream_w);
void fused_add_rms_quant(torch::Tensor& x, torch::Tensor& residual, const torch::Tensor& weight, double eps,
                         torch::Tensor& q, torch::Tensor& s, int64_t rsqrt_mode);
void silu_mul_quant(const torch::Tensor& gate_up, torch::Tensor& q, torch::Tensor& s);
void rope_qout_kv_store(const torch::Tensor& positions, const torch::Tensor& q_pe, torch::Tensor& q_out,
                        torch::Tensor& kv_c, const torch::Tensor& kv_weight, double norm_eps, torch::Tensor& k_pe,
                        const torch::Tensor& cos_sin_cache, bool is_neox, torch::Tensor& kv_buffer,
                        const torch::Tensor& loc, double store_eps, int64_t rsqrt_mode);
torch::Tensor sparse_mask_topk_transform_split_pre(
    const torch::Tensor& score, const torch::Tensor& nonzero_mask,
    const torch::Tensor& lengths, const torch::Tensor& page_table_size_1,
    const torch::Tensor& cu_seqlens_q, int64_t topk, int64_t max_splits, const torch::Tensor& zws);
std::vector<torch::Tensor> mqa_logits_hist(const torch::Tensor& q, const torch::Tensor& kv_cache,
                                           const torch::Tensor& weights, const torch::Tensor& context_lens,
                                           const torch::Tensor& block_table, int64_t max_context_len,
                                           int64_t group_size, int64_t persistent_ctas, int64_t rows_per_cta,
                                           const c10::optional<torch::Tensor>& hist_lens);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sparse_mask_topk_transform", &sparse_mask_topk_transform,
        "Exact top-2048 + page-table transform over sparse-mask MQA logits");
  m.def("sparse_mask_topk_transform_split", &sparse_mask_topk_transform_split,
        "Same as sparse_mask_topk_transform; few rows are split over several "
        "CTAs (falls back to one CTA per row once rows fill the GPU)");
  m.def("skinny_gemm", &skinny_gemm,
        "y[M,N] = x[M,K] @ w[N,K]^T for M <= 64 (fp8 w8a8 per-channel or bf16)");
  m.def("rope_inplace", &rope_inplace,
        "sgl_kernel.rotary_embedding drop-in (same args, in place) in one launch for any head count");
  m.def("rope_qout", &rope_qout,
        "rope_inplace with the rotated query written to q_out (key still in place), same numerics");
  m.def("moe_gate_topk", &moe_gate_topk,
        "lightop moe_fused_gate (1 group, sigmoid + bias, renormalized top-k) with int64 ids, "
        "padded rows -> -1");
  m.def("bmm_skinny", &bmm_skinny,
        "y[b] = x[b] @ w[b]^T, bf16, M <= 32, strided batch / rows (MLA absorb bmm's in decode)");
  m.def("silu_mul_fp8_quant_masked", &silu_mul_fp8_quant_masked,
        "lightop fuse_silu_mul_fp8_quant_ep(x, 0, masked_m) drop-in (rows t < masked_m[e], same numerics)");
  m.def("bf16_round_check", &bf16_round_check,
        "v_cvt_bf16_f32 vs software RNE on all 2^32 floats: {mismatches, first}, for tests");
  m.def("skinny_nn", &skinny_nn,
        "y[M,N] = x[M,K] @ w[K,N] for M <= 64, bf16, w n-contiguous (NN lm_head / vocab-parallel shard)");
  m.def("fused_add_rms_quant", &fused_add_rms_quant,
        "lightop op.fused_add_rms_norm_opt (in place) + per_token_quant_fp8 of the normed x, same bits");
  m.def("silu_mul_quant", &silu_mul_quant,
        "lightop fuse_silu_and_mul + per_token_quant_fp8 of the product, same bits");
  m.def("rope_qout_kv_store", &rope_qout_kv_store,
        "MLA decode: rope_qout + kv_a rms_norm_opt (in place) + k rope (in place) + lightop "
        "fused_quantize_and_store_mla_kv_cache fp8 store in one launch, same bits");
  m.def("sparse_mask_topk_transform_split_pre", &sparse_mask_topk_transform_split_pre,
        "sparse_mask_topk_transform_split with pass 1 (histogram) taken from mqa_logits_hist's workspace");
  m.def("mqa_logits_hist", &mqa_logits_hist,
        "lightop page_mqa_logits_sparse_mask[_grouped] native fp8 drop-in (same bits) -> {logits, mask, zws}; "
        "with hist_lens also split-TopK pass 1 into zws");
}
