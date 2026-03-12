# Task: Optimize See task_description.md on AMD GPU -- Stage1 Diagnose And Infra

## Starting Point

You are starting from a **clean environment**.
- Base image: `amdpilot-eval-sglang-kimi-moe-tune`
- Repo cloned at `/workspace/` from `https://github.com/sgl-project/sglang.git`
- **CHECK FIRST**: Run `ls /workspace/` to see if a previous trial left benchmark scripts or results. Reuse them if they exist.

## Task Description

# Optimize Kimi K2.5 fused_moe_triton Performance

## Context

Kimi K2.5 is a multimodal encoder-decoder model (architectures: KimiK25ForConditionalGeneration, based on DeepseekV3) that uses Mixture of Experts (MoE).

When serving Kimi K2.5 with sglang, the fused_moe_triton kernel uses default config, resulting in poor performance. The model uses int4_w4a16 quantization with 384 experts.

Profiling shows the fused_moe kernel is a significant bottleneck:
- Prefill first MoE: ~9.11ms, second MoE: ~4.28ms
- Decode first MoE: ~501us, second MoE: ~180us

The fused_moe_triton kernel has a config lookup mechanism that loads tuned configurations based on model parameters (E, N, dtype), but the appropriate tuned configs may be missing for this model's specific configuration.

## Task

Investigate and resolve the poor fused_moe_triton performance for Kimi K2.5. The solution should ensure the kernel uses optimized configurations rather than defaults.

## Key Model Parameters

- `num_local_experts: 384` (E=384)
- `moe_intermediate_size: 2048`
- Quantization: int4_w4a16 (4-bit weights, group_size=32)
- Served with TP=8

## Required Approach

**You MUST follow this workflow — do not skip steps:**

1. **Read the MoE config tuning reference**: Read `/workspace/skills/amd-kernel-optimization/references/moe-config-tuning.md` — it explains how config lookup works, what parameters mean, and the correct tuning workflow.

2. **Understand the config lookup mechanism**: Read `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_config.py` to understand how configs are loaded. Check what config files exist and what's missing.

3. **Read the existing tuning infrastructure**: The `benchmark/kernels/fused_moe_triton/` directory contains:
   - `tuning_fused_moe_triton.py` — main tuning script with ray-based config search
   - `tuning_fused_moe_triton_sep.py` — separate up/down projection tuning
   - `common_utils.py` — shared utilities for config generation
   - `README.md` — usage instructions

   **Read these files BEFORE writing any configs.** They contain the correct methodology.

4. **Run systematic config benchmarking**: Either use the existing tuning script or write a systematic config search that tests multiple configs per batch size. Do NOT fabricate config values — benchmark each config to find the actual best.

5. **Benchmark with exclusive GPU access**: When running final benchmarks, ensure no other GPU-intensive processes are running.

## Environment

- Repository: sgl-project/sglang (code at `/workspace/sglang`)
- Docker container with ROCm, PyTorch, AMD GPU (8x MI300X)
- Use `/opt/venv/bin/python3` for all commands
- Model weights at `/sgl-workspace/models/models--moonshotai--Kimi-K2.5/`
- The benchmark/kernels/fused_moe_triton/ directory contains tuning scripts and utilities

## Verification

Run the test harness after applying your fix:
```bash
cd /workspace/sglang && /opt/venv/bin/python3 /workspace/test_harness.py
```

## Execution Environment

You are running inside a Docker container with ROCm, PyTorch, and GPU access.
All commands run directly in this container — no `docker exec` needed.
- Repo at `/workspace/`

## Goal

Achieve **>=999.0 score** for `` on AMD GPU.
Run the verification command: `cd /workspace/sglang && /opt/venv/bin/python3 /workspace/test_harness.py`

## Workflow

1. **Read skills**: Read the optimization skill at `/workspace/skills/amd-kernel-optimization/SKILL.md` FIRST
2. **Investigate**: Read the task description, repo structure, and find the code to optimize
3. **Environment probe**: Check PyTorch version (`/opt/venv/bin/python3 -c "import torch; print(torch.__version__)"`), GPU type, available AMD libs (aiter, CK)
4. **Baseline**: Run the benchmark command to establish the starting metric. Record it accurately
5. **Profile (MANDATORY — do this BEFORE any code changes)**:
   Run at least ONE of these profiling methods and analyze the output:

   Option A — rocprof kernel trace (best for kernel-level bottleneck analysis):
   ```
   rocprof --stats /opt/venv/bin/python3 your_benchmark_script.py 2>&1 | tail -40
   ```
   This prints per-kernel GPU time. Look for which kernels dominate total time.

   Option B — torch.profiler with GPU activities:
   ```python
   import torch
   with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
       # run the workload once
       pass
   print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
   ```

   **You MUST include the profiling output in your response before proceeding to step 6.**
   The supervisor will reject your trial if no profiling evidence is found.
   Use the profile to determine whether each bottleneck is compute-bound or memory-bound,
   and choose optimization strategies accordingly.

6. **Optimize**: Apply optimizations ONE AT A TIME, benchmark after EACH change. Revert immediately if performance regresses. Try optimizations in this order:
   - AMD-specific hardware intrinsics (if applicable to the bottleneck)
   - Memory access pattern improvements (coalescing, shared memory usage)
   - Reduce synchronization overhead
   - Template specialization for common cases
   - Instruction-level optimization (prefetching, compute/memory overlap)
7. **Save before experimenting**: Before trying a risky config change, save a backup of your current best config files (`cp config.json config.json.best`). If the change regresses, restore from the backup immediately.
8. **Update state**: Write results to `/workspace/optimization_state.json` after each successful optimization
9. **Verify**: Run the benchmark 3x and report the median. The median must consistently beat the target

## CRITICAL RULES

- Do NOT modify the benchmark script or test harness. Any modified benchmark result is INVALID.
- Do NOT reduce warmup iterations or measurement iterations.
- After making CUDA/HIP kernel changes, REBUILD the project: `cd /sgl-workspace/aiter && /opt/venv/bin/python3 setup.py develop`
- If a kernel change causes compilation errors, check the AMD-specific syntax carefully. AMD wavefronts are 64-wide (not 32).
- ALWAYS profile before optimizing. Blind trial-and-error wastes time.

## Integrity Rules (MANDATORY)

- Do NOT modify `test_harness.py` or the benchmark script.
- Do NOT change benchmark parameters (warmup, iterations, batch size, input dimensions).
- Do NOT redirect or filter benchmark output (no `grep`, no `| tail`).
- The orchestrator independently re-runs the benchmark to verify your results. Any discrepancy between your claimed result and the verification run causes the trial to be marked as FAILED.
- Achieve improvements through legitimate code changes only.

## CRITICAL: Output Format

Your LAST action before finishing MUST be running the benchmark command **directly** —
do NOT pipe through `grep` or redirect output. The FULL benchmark output must appear
in your output so the orchestrator can capture the metric via regex: `SCORE:\s+([\d.]+)`

The output must include a line matching: `score: <value>`

Without this line, the trial is recorded as a FAILURE even if you achieved great results.

## State File

After completing work, update `/workspace/optimization_state.json`:
```json
{
  "applied_optimizations": [
    {"name": "<technique>", "impact_ms": -5.0, "files_changed": ["<path>"], "description": "<what>"}
  ],
  "attempted_but_failed": [
    {"name": "<technique>", "reason": "<why>", "error": "<snippet>"}
  ],
  "profiling_summary": {
    "top_bottlenecks": ["<kernel> <pct>"],
    "gpu_utilization": 0.72
  },
  "environment_changes": ["pip install <pkg>"]
}
```
This file is read by the orchestrator and injected into the next stage's prompt.
