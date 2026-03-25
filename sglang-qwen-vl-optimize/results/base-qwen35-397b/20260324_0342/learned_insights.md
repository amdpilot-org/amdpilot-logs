# Learned Insights

- **Trial 1**: On AMD MI355X with Qwen3-VL (Lk=128, GQA kv_group_num=4), increasing BLOCK_N from 32 to 64 in triton grouped decode attention gives ~18.7% throughput improvement
- **Trial 1**: waves_per_eu=2 and num_stages=2 (up from 1) in triton decode attention improve MI355X occupancy, contributing ~15% throughput gain
- **Trial 1**: The key optimization file is /sgl-workspace/sglang/python/sglang/srt/layers/attention/triton_ops/decode_attention.py, specifically _decode_grouped_att_m_fwd and _fwd_grouped_kernel_stage1
- **Trial 1**: Increasing triton_attention_num_kv_splits to 32 slightly hurts performance on this workload — avoid
- **Trial 1**: Baseline SGLang triton attention on MI355X: 1288 tok/s, TPOT 12.21ms. After BLOCK_N+occupancy tuning: 1722 tok/s, TPOT 8.80ms
- **Trial 2**: torch.compile on ROCm with SGLang causes server startup timeout due to long compilation — avoid this approach
- **Trial 2**: Attention kernels account for only ~41% of GPU time in Qwen3-VL serving; GEMM/MLP is ~40%, so attention-only optimization has a ceiling
- **Trial 2**: bench_config.env can set EXTRA_SERVER_ARGS but NOT override the attention-backend which is hardcoded in the benchmark script
- **Trial 2**: After BLOCK_N=64 + waves_per_eu=2 + num_stages=2 tuning, TPOT improved from 12.21ms to 8.80ms — already better than vLLM's 9.09ms
- **Trial 3**: Reducing triton_attention_num_kv_splits from 16 to 4 provides ~7% additional throughput improvement on MI355X with Qwen3-VL (1723→1845 tok/s)
- **Trial 3**: Run-to-run variance on the bench_qwen_vl.sh benchmark is approximately ±15-20 tok/s
- **Trial 3**: Combined BLOCK_N=64 + waves_per_eu=2 + num_stages=2 + num_kv_splits=4 achieves ~43% throughput improvement over SGLang triton baseline on MI355X
- **Trial 4**: Combined BLOCK_N=64 + waves_per_eu=2 + num_stages=2 + num_kv_splits=4 achieves 1853 tok/s on MI355X with Qwen3-VL-8B — 43% above SGLang triton baseline and 12% above vLLM
- **Trial 4**: For GQA models with kv_group_num=4 and head_dim=128 on MI355X, BLOCK_N=64 is optimal in triton grouped decode attention (BLOCK_N=32 default is too small)
- **Trial 4**: Reducing triton_attention_num_kv_splits from 8 to 4 on HIP improves throughput ~5-7% for this workload — fewer splits reduce reduction overhead
- **Trial 4**: TPOT improved from 12.21ms (baseline) to ~8.12ms (optimized), beating vLLM's 9.09ms
- **Trial 4**: Run-to-run variance on bench_qwen_vl.sh is approximately ±15-20 tok/s
