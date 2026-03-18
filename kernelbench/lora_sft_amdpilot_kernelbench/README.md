# lora_sft_amdpilot_kernelbench

SFT training data from KernelBench x amdpilot experiment.

## Overview

- **Source**: amdpilot Phase 3 (full pipeline) trajectories
- **Hardware**: AMD Instinct MI355X (gfx950, CDNA4)
- **Models**: Qwen3.5-397B-A17B (executor) + Claude Opus 4.6 (supervisor/nudge)
- **Backend**: Triton (ROCm 7.2, Triton 3.6.0)
- **Format**: HuggingFace messages (OpenAI chat format with tool calls)

## Stats

- Total examples: 94
- By level: L1=28, L2=52, L3=14
- Tier 1 (score >= 75): 6
- Tier 2 (score >= 60): 94
- Average score: 63.9

## Quality Filtering

- Verified correct only (verified=true in summary.json)
- Score >= 60.0 (correct AND faster than PyTorch)
- 10 torch.compile hacks removed
- Real @triton.jit kernel required
- Clean trajectories only (no server errors, no infinite loops)
- Best trial per task

## Files

- `train.jsonl` -- all examples
- `by_level/level{1,2,3}.jsonl` -- split by KernelBench level
- `by_quality/tier1_score_ge75.jsonl` -- highest quality
- `by_quality/tier2_score_ge60.jsonl` -- all qualifying
- `metadata.json` -- dataset statistics
