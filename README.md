# amdpilot-logs

Experiment results and agent trajectories from the amdpilot multi-agent system for AMD GPU kernel optimization.

This repo stores structured logs from eval runs, intended for:
- Ablation study analysis
- High-quality trajectory collection for SFT/RL training data
- Reproducibility and audit trail

## Structure

```
<eval-task>/
  <phase-name>/
    summary.json          # Structured results (scores, trials, supervisor decisions)
    scoreboard.jsonl      # Per-trial metrics
    trace.md              # Human-readable full trajectory
    optimization_state.json
    agent_output/
      trial_1.txt         # Raw agent output per trial
      trial_2.txt
      ...
    *.log                 # Orchestrator log
```

## Eval Tasks

### sglang-pr-19228: Kimi K2.5 fused_moe_triton optimization

Source PR: [sgl-project/sglang#19228](https://github.com/sgl-project/sglang/pull/19228)

6-phase ablation study investigating why the agent couldn't learn profiling-driven optimization.

| Phase | Directory | Best Verified | Profiled? | Key Fix |
|-------|-----------|-------------|-----------|---------|
| 1a | `phase1_exp_a_post_trial_hardcheck` | 74.4 | No | Post-trial profiling check |
| 1b | `phase1_exp_b_claude_nudge` | 70.4 | Yes (once) | Claude mid-trial nudge |
| 3 | `phase3_config_fix_8h` | 73.2 | No | `_build_stages()` config fix, 8h budget |
| 4 | `phase4_higher_targets_nudge_fix` | 74.2 | No | Targets 40/70/85, nudge delivery fix |
| 5 | `phase5_build_stages_fix` | 78.2 | No | `_build_stages()` alias fix (target/duration/description) |
| 6 | `phase6_skills_tier0_perfect_score` | **98.7** | **Every trial** | moe-config-tuning.md + rocprof docs + Tier 0 test harness |

Key finding: The original conclusion ("model capability is the bottleneck") was wrong. The real bottleneck was missing skill documentation and test harness incentives. Same Qwen 3.5 model went from 74.2 to 98.7 with infrastructure fixes only.

Full analysis: [Notion -- Ablation Study](https://www.notion.so/31f651cb22e58081a596cda3d7a315f4)
