# LoRA SFT Dataset: AMD GPU Kernel Agent Trajectories

Curated training data for LoRA/SFT fine-tuning from AMD GPU agent systems.
Each example is a full multi-turn agent trajectory (system/user/assistant/tool) where the agent
reads code, edits files, runs tests, and iterates — producing verified AMD GPU kernel fixes and optimizations.

## Quick Start

```python
import json

# Load all 125 examples
examples = []
with open("combined/train.jsonl") as f:
    for line in f:
        examples.append(json.loads(line))

# Each example has: {id, source, messages, score, num_turns, num_tool_calls, ...}
# messages is a list of {role, content, tool_calls?, tool_call_id?} dicts
print(f"{len(examples)} examples, {sum(len(e['messages']) for e in examples)} total messages")
```

## Dataset Summary

| Metric | Value |
|---|---|
| Total examples | 125 |
| Total messages | 14,507 |
| Total tool calls | 7,356 |
| Score range | 50.0 -- 100.0 |
| Mean score | 69.3 |
| Hardware | AMD Instinct MI355X (gfx950) |
| Backend | Triton (ROCm 7.2) / SGLang / aiter / HIP / CK |

### By Source

| Source | Examples | Task Type | Executor Model | Score Range |
|---|---|---|---|---|
| KernelBench Phase 3 | 94 | optimize | Qwen3.5-397B-A17B | 60.0 -- 100.0 |
| AMD Kernel Pipeline | 29 | bugfix | Claude Opus 4.6 | 50.0 -- 100.0 |
| SGLang #19935 | 1 | bugfix | Claude Opus 4.6 | 100.0 |
| SGLang #20187 | 1 | feature | Claude Opus 4.6 | 100.0 |

### By Task Type

| Type | Count | Description |
|---|---|---|
| optimize | 94 | Triton kernel optimization for AMD MI355X (speedup over PyTorch) |
| bugfix | 30 | Real PR fixes across ROCm/aiter, HIP, HIPIFY, CK, PyTorch, rocm-libraries |
| feature | 1 | FP8 prefill + radix cache integration in SGLang aiter backend |

### By Repository

| Repository | Examples | Source |
|---|---|---|
| KernelBench (Triton) | 94 | amdpilot KernelBench Phase 3 |
| ROCm/HIPIFY | 8 | AMD Kernel Pipeline |
| ROCm/composable_kernel | 6 | AMD Kernel Pipeline |
| ROCm/HIP | 5 | AMD Kernel Pipeline |
| pytorch/pytorch | 4 | AMD Kernel Pipeline |
| ROCm/aiter | 3 | AMD Kernel Pipeline |
| ROCm/rocm-libraries | 3 | AMD Kernel Pipeline |
| SGLang (amdpilot) | 2 | Claude Opus 4.6 executor |

## Directory Structure

```
lora-sft-dataset/
  README.md                            # This file
  convert_context_jsonl.py             # Conversion script (kimi-cli -> HF messages)

  kernelbench/                         # 94 examples — Triton kernel optimization
    train.jsonl                        # All 94 examples in HF messages format
    metadata.json                      # Filtering criteria, model info, stats
    by_level/
      level1.jsonl                     # 28 examples (simpler kernels)
      level2.jsonl                     # 52 examples (intermediate)
      level3.jsonl                     # 14 examples (complex kernels)
    by_quality/
      tier1_score_ge75.jsonl           # 6 highest-quality (score >= 75)
      tier2_score_ge60.jsonl           # All 94 (score >= 60)

  amd-kernel-pipeline/                 # 29 examples — real AMD PR fixes
    train.jsonl                        # 20 PASS + 9 PARTIAL from 6 AMD repos
    metadata.json                      # Repos, validation tiers, quality scores

  sglang-bugfix/                       # 1 example — SGLang #19935
    train.jsonl                        # Converted to HF messages format
    metadata.json                      # Source PR, verification details
    raw/
      context.jsonl                    # Original kimi-cli trajectory
      summary.json                     # Orchestrator run summary
      trial_1.txt                      # Agent stdout (tool usage log)

  sglang-feature/                      # 1 example — SGLang #20187
    train.jsonl
    metadata.json
    raw/
      context.jsonl
      summary.json
      trial_1.txt

  combined/                            # All 125 examples merged
    train.jsonl                        # Ready for training
    stats.json                         # Aggregate statistics
```

## Data Sources

### 1. KernelBench (94 examples)

Triton kernel optimization trajectories from KernelBench Phase 3. Each example shows the agent
converting a PyTorch operation into an optimized Triton kernel for AMD MI355X GPUs.

- **Executor**: Qwen3.5-397B-A17B (self-hosted, thinking model)
- **Supervisor**: Claude Opus 4.6 (review and stage management)
- **Filtering**: Verified correct, score >= 60 (speedup over PyTorch), real `@triton.jit` kernels, no `torch.compile` hacks
- **AMD-specific patterns**: manual tanh (94), wavefront 64 (81), gfx950 target (94), explicit fp32 casts (94)

### 2. AMD Kernel Pipeline (29 examples)

Real AMD GPU kernel PR fixes solved by Claude Opus 4.6 via the OpenHands agent framework.
Each trajectory shows the agent reading the PR description, exploring the codebase, implementing
the fix, and running tests.

- **Executor**: Claude Opus 4.6 (via OpenHands)
- **Repos**: ROCm/HIPIFY (8), ROCm/composable_kernel (6), ROCm/HIP (5), pytorch/pytorch (4), ROCm/aiter (3), ROCm/rocm-libraries (3)
- **Validation**: 20 PASS (test suite green), 9 PARTIAL (some tests pass)
- **Quality scores**: 4-5 (human-rated PR difficulty/quality)
- **Source**: `/home/jinpan12/amd-kernel-sft-pipeline/results/`

### 3. SGLang #19935 — FP8 MLA Decode k_scale Fix (1 example)

The agent fixes an assertion failure in the aiter MLA decode kernel when `layer.k_scale` is None.
The fix adds a fallback to `self.k_scale` at all 4 `mla_decode_fwd` call sites.

- **Executor**: Claude Opus 4.6
- **Human PR**: [sgl-project/sglang#19935](https://github.com/sgl-project/sglang/pull/19935)
- **Score**: 100 (AST-verified: all 4 call sites fixed)
- **Trajectory**: 75 messages, 41 tool calls, 276s runtime
- **Anti-hacking**: Git history stripped, AST verification, no reward hacking detected

### 4. SGLang #20187 — FP8 Prefill + Radix Cache (1 example)

The agent integrates FP8 prefill attention into the radix-cache code path, including
creating a helper method, fixing metadata setup, and adding fused GEMM dispatch.

- **Executor**: Claude Opus 4.6
- **Human PR**: [sgl-project/sglang#20187](https://github.com/sgl-project/sglang/pull/20187)
- **Score**: 100 (6/6 structural checks: helper method, kv_indptr fix, total_s fix, fused GEMM, radix FP8 branch, no-prefix refactor)
- **Trajectory**: 126 messages, 63 tool calls, 583s runtime
- **Anti-hacking**: Git history stripped, AST verification, agent identified kv_indptr/total_s fixes independently

## Message Schema

Each example is a JSON object with this structure:

```json
{
  "id": "kernelbench-L1-P100-trial1",
  "source": "amdpilot-kernelbench-phase3",
  "score": 100.0,
  "messages": [
    {"role": "system", "content": "You are an AMD GPU kernel optimization expert..."},
    {"role": "user", "content": "<task prompt>"},
    {"role": "assistant", "content": "", "tool_calls": [
      {"type": "function", "id": "call_abc", "function": {"name": "bash", "arguments": "{\"command\": \"ls\"}"}}
    ]},
    {"role": "tool", "content": "file1.py\nfile2.py", "tool_call_id": "call_abc"},
    {"role": "assistant", "content": "<reasoning + next action>"},
    ...
  ]
}
```

### Roles

| Role | Description |
|---|---|
| `system` | System prompt with environment info and task instructions |
| `user` | Task prompt (first turn) or nudge messages (mid-trajectory) |
| `assistant` | Agent reasoning + tool call decisions |
| `tool` | Tool execution results (file contents, shell output, etc.) |

### Additional Fields

| Field | Type | Description |
|---|---|---|
| `id` | string | Unique example identifier |
| `source` | string | Data source (`amdpilot-kernelbench-phase3` or `sglang-sft-claude-executor`) |
| `score` | float | Verification score (60-100 for kernelbench, 100 for sglang) |
| `task_type` | string | `optimize`, `bugfix`, or `feature` (sglang examples only) |
| `num_turns` | int | Number of user + assistant messages |
| `num_tool_calls` | int | Total tool calls across all assistant turns |
| `level` | int | KernelBench difficulty level 1-3 (kernelbench only) |
| `problem_id` | int | KernelBench problem ID (kernelbench only) |
| `triton_kernel_count` | int | Number of `@triton.jit` kernels in solution (kernelbench only) |
| `amd_specific_fixes` | list | AMD-specific patterns used (kernelbench only) |

## Quality Assurance

| Measure | KernelBench | SGLang Issues |
|---|---|---|
| Verified correct | Yes (benchmark score) | Yes (AST structural checks) |
| Compared to human fix | N/A (novel optimizations) | Yes (matches human PR changes) |
| Reward hacking check | torch.compile hacks removed | Git history stripped, AST checks |
| Score threshold | >= 60 (speedup over PyTorch) | 100/100 |
| Real code output | @triton.jit required | Functional code changes verified |

## Training Notes

- The dataset uses OpenAI-compatible chat format with tool calls
- Compatible with: HuggingFace TRL `SFTTrainer`, axolotl, LLaMA-Factory
- For LoRA training, the `messages` field maps directly to multi-turn chat templates
- Consider filtering by `score >= 75` for highest quality (6 kernelbench + 22 pipeline/sglang = 28 examples)
- The `amd-kernel-pipeline` examples with `validation_status: "PASS"` are the most reliable (20 examples)
- The `tool_calls` field uses the OpenAI function-calling format (`type: "function"`)
