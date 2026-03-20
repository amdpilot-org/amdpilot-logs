# Learned Insights

- **Trial 1**: The dynamic boolean mask `page_table_1[page_table_1 != -1]` produces variable-length output that breaks CUDA graph capture; replace with pre-computed CSR indptr (sum+cumsum) and a Triton scatter kernel for fixed-shape extraction
- **Trial 1**: Scalar tensor assignment like `kv_indptr[0] = 0` is disallowed during HIP stream capture; use torch.zeros pre-initialization instead
- **Trial 1**: mla_decode_fwd on aiter requires keyword arguments for sm_scale and does not support a logit_cap parameter
- **Trial 1**: The aiter prefill for NSA can reuse the existing tilelang_sparse_fwd kernel which is already AMD GPU compatible
- **Trial 1**: When debugging server startup failures, always check the full server log tail (`cat /tmp/sglang_server.log | tail -150`) to find the actual traceback
