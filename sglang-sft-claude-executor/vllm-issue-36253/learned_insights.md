# Learned Insights

- **Trial 1**: aiter's biased_grouped_topk kernel has a hard limit of 32 experts per group; for 256-expert models like MiniMax-M2.5, num_expert_group=8 is the smallest valid divisor (256/8=32)
- **Trial 1**: Setting topk_group = num_expert_group makes grouped top-k semantically equivalent to global top-k with bias correction
- **Trial 1**: The aiter fast path gate: rocm_aiter_ops.is_fused_moe_enabled() must return True, scoring_func must be 'sigmoid', and e_score_correction_bias must be provided
- **Trial 1**: The aiter biased_grouped_topk kernel internally dispatches to biased_grouped_topk_hip or moe_fused_gate based on token count
- **Trial 1**: When running test harnesses on this system, use PATH=/usr/bin:$PATH so python3 resolves to python3.12 which has vllm installed
