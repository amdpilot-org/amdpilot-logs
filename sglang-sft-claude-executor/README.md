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
  summary.json
  trace.md
  run.log
  scoreboard.jsonl
  agent_output/
    trial_1.txt             # Raw agent output (20 KB)
    trial_1_trajectory/
      prompt.txt
      sessions/<id>/<run>/
        context.jsonl       # SFT training data (311 lines)
        wire.jsonl
```
