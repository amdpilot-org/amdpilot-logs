# SGLang SFT Trajectory Collection — Claude Opus 4.6 Executor

SFT training trajectories from Claude Opus 4.6 solving SGLang PRs autonomously.
All three agents (supervisor, executor, nudge) use Claude Opus 4.6 via the AMD LLM Gateway.

## Setup

- **Executor**: Claude Opus 4.6 via `supervisor_proxy.py` (OpenAI→Anthropic tool translation + SSE streaming)
- **Supervisor**: Claude Opus 4.6 (same proxy)
- **Nudge**: Claude Opus 4.6 (same proxy)
- **Hardware**: MI355X x8 (40 XCDs)
- **Agent framework**: kimi-cli 1.15.0 via `uv run --python 3.14`
- **Branch**: `claude-executor-sft-data`

## Anti-Reward-Hacking Measures

| Measure | Implementation |
|---|---|
| Git history stripped | `rm -rf .git && git init` — agent cannot `git show` the merged fix |
| AST-based verification | Python `ast` module checks code structure, not string patterns |
| Test harness hidden | At `/opt/test_harness.py`, not in agent's working directory |
| No scoring hints | Task description doesn't reveal harness path or check criteria |

## Results

### SGLang #19935: FP8 MLA decode k_scale assertion fix

| Metric | Value |
|---|---|
| **Score** | 100 / 100 |
| **Time** | 276s (4.6 min) |
| **Trials** | 1 |
| **Trajectory** | 173 lines (context.jsonl) |
| **Git history peeking** | 0 attempts |
| **Reward hacking** | None detected |

**Task**: Fix assertion failure `q_scale.has_value() && kv_scale.has_value()` in aiter ASM MLA decode when `layer.k_scale` is None. Fall back to `self.k_scale` at all 4 `mla_decode_fwd` call sites.

**Agent approach**: Read `aiter_backend.py`, identified all 4 call sites via grep, applied the `layer.k_scale if layer.k_scale is not None else self.k_scale` pattern. Functionally identical to the human PR.

**Human PR**: [sgl-project/sglang#19935](https://github.com/sgl-project/sglang/pull/19935)

### SGLang #20187: FP8 prefill + radix cache integration

| Metric | Value |
|---|---|
| **Score** | 100 / 100 (6/6 structural checks) |
| **Time** | 583s (9.7 min) |
| **Trials** | 1 |
| **Trajectory** | 311 lines (context.jsonl) |
| **Git history peeking** | 0 attempts |
| **Reward hacking** | None — agent identified kv_indptr and total_s fixes independently |

**Task**: Enable FP8 prefill attention for the radix-cache path in `aiter_backend.py`.

**Agent approach (6 changes matching human PR)**:
1. Created `mla_fp8_prefill_attn` helper method (extracted shared FP8 prefill logic)
2. Fixed `init_forward_metadata`: `kv_indptr` from `mla_indices_updater` instead of `qo_indptr`
3. Fixed `total_s = forward_batch.seq_lens_sum` (was `extend_seq_lens.sum()`)
4. Added `fused_gemm_afp4wfp4_split_cat` via `kv_b_proj` tuple-dispatch in radix-cache branch
5. Added FP8 dispatch in radix-cache branch
6. Refactored no-prefix path to use shared helper

**Independent reasoning verified**: Agent identified the kv_indptr and total_s bugs from code analysis BEFORE reading any test harness (at trajectory lines 130 and 118 respectively).

**Human PR**: [sgl-project/sglang#20187](https://github.com/sgl-project/sglang/pull/20187)

### SGLang #18526: NSA cudagraph capture + aiter prefill

| Metric | Value |
|---|---|
| **Score** | 100 / 100 |
| **Time** | 3601s (60 min) |
| **Trials** | 1 |
| **Trajectory** | 307 lines (context.jsonl) |
| **Reward hacking** | None — test harness untouched (hash match), unfixed code crashes (SCORE: 0), fix required for server to start |

**Task**: Replace dynamic boolean mask `page_table_1[page_table_1 != -1]` in NSA decode aiter backend with a deterministic Triton kernel to enable CUDA graph capture, and add aiter NSA prefill implementation.

**Agent approach**:
1. Added `get_valid_kv_indices` Triton kernel in `triton_kernel.py` using `tl.load`/`tl.store` with pre-computed cumsum indptr — fixed-shape ops compatible with CUDA graph capture
2. Pre-allocated `kv_indices_buf` in `nsa_backend.py`, replaced boolean mask in decode path with the new kernel
3. Fixed `mla_decode_fwd` call signature (added `kv_last_page_lens`, `sm_scale` kwarg)
4. Added `_forward_aiter_prefill` delegating to tilelang sparse kernel

**Base image**: `rocm/sgl-dev:v0.5.9-rocm720-mi35x-20260226` (pre-dates PR merge Feb 27)

**Human PR**: [sgl-project/sglang#18526](https://github.com/sgl-project/sglang/pull/18526)

### vLLM #36253: fused_topk_bias aiter fast path for MiniMax-M2.5

| Metric | Value |
|---|---|
| **Score** | 2182.81 tok/s (output token throughput) |
| **Baseline** | 1870.69 tok/s (unfixed) |
| **Improvement** | +16.7% throughput |
| **Time** | 1313s (22 min) |
| **Trials** | 1 (within 2-trial run) |
| **Trajectory** | 175 lines (context.jsonl) |
| **Reward hacking** | None — throughput benchmark (not binary), baseline independently measured, test harness untouched |

**Task**: Add a fast path in `fused_topk_bias` to use aiter's `biased_grouped_topk` kernel on ROCm instead of falling back to generic PyTorch ops for MiniMax-M2.5 (256 experts, sigmoid scoring with bias correction).

**Agent approach**:
1. Added `_compute_num_expert_group()` helper: smallest divisor of `num_experts` satisfying aiter's 32-expert-per-group limit (256/8=32 -> 8 groups)
2. Added aiter fast path: detects sigmoid + bias correction, routes to `rocm_aiter_grouped_topk` with `topk_group=num_expert_group`
3. Only 1 file modified (`fused_topk_bias_router.py`) — the exact target file

**Base image**: `rocm/vllm-dev:nightly_main_20260308` (pre-dates PR merge Mar 9)

**Human PR**: [vllm-project/vllm#36253](https://github.com/vllm-project/vllm/pull/36253)

## File Structure

```
sglang-issue-19935/
  summary.json              # Score, timing, trial metadata
  trace.md                  # Human-readable trajectory
  run.log                   # Orchestrator log
  scoreboard.jsonl          # Per-trial metrics
  agent_output/
    trial_1.txt             # Raw agent output (6 KB)
    trial_1_trajectory/
      prompt.txt
      sessions/<id>/<run>/
        context.jsonl       # SFT training data (173 lines)
        wire.jsonl

sglang-issue-20187/
  ...same structure...

sglang-issue-18526/           # NSA cudagraph + aiter prefill
  summary.json
  trace.md
  run.log
  scoreboard.jsonl
  agent_output/
    trial_1.txt
    trial_1_trajectory/
      sessions/<id>/<run>/
        context.jsonl       # SFT training data (307 lines)
        wire.jsonl

vllm-issue-36253/             # fused_topk_bias aiter fast path
  summary.json
  trace.md
  run.log
  scoreboard.jsonl
  agent_output/
    trial_1.txt
    trial_1_trajectory/
      sessions/<id>/<run>/
        context.jsonl       # SFT training data (175 lines)
        wire.jsonl
```
