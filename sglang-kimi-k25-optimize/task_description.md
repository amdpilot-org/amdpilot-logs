# Kimi-K2.5 Decode Latency Optimization

Optimize the decode latency of the Kimi-K2.5 (1T MoE) model on 8x AMD MI355X GPUs using SGLang.
The single metric that matters is the output of `/workspace/bench_kimi_k25.sh`. Lower is better.

## Prerequisites

Download the model weights before running (requires HuggingFace access):
```bash
huggingface-cli download moonshotai/Kimi-K2.5
```
Ensure the HuggingFace cache directory is mounted into the container at `/root/.cache/huggingface`.
Set `HF_HOME` on the host to point to your cache directory, or bind-mount it directly.

## Environment

- **Installed SGLang runtime**: `/sgl-workspace/sglang/` — on `sys.path`, used by `python3 -m sglang.*`. **Edit files HERE to modify SGLang behavior.**
- **SGLang reference checkout**: `/workspace/sglang/` — fresh `git clone` for reference only. Changes here do NOT affect the runtime.
- **AITER library**: `/sgl-workspace/aiter/` — AMD inference acceleration library. May need modifications for Kimi-K2.5 support.
- **Model weights**: `moonshotai/Kimi-K2.5` cached at `/root/.cache/huggingface`.
- **Benchmark script**: `/workspace/bench_kimi_k25.sh` — runs `sglang.bench_one_batch` with fixed workload params.

## Model Architecture

Kimi-K2.5 is a 1T-parameter MoE model (DeepSeek V3 architecture):
- 384 experts, 8 active per token, 1 shared expert
- MLA (Multi-head Latent Attention) with 64 attention heads
- hidden_size=7168, MoE hidden dim per expert=2048
- 61 layers (including 1 dense layer), SwiGLU activation
- Requires `--trust-remote-code`

## Benchmark

The benchmark runs `sglang.bench_one_batch` with a long-context workload and prints:
  `Decode median (ms): <value> | tp=8 batch=1 in=8192 out=2048 decode=<backend>`

Fixed parameters: tp=8, batch_size=1, input_len=8192, output_len=2048.
First run takes a long time (model loading + CUDA graph compilation).
Set `timeout: 1200` when running it. If it times out, kill leftover sglang processes
(`ps aux | grep sglang | grep -v grep | awk '{print $2}' | xargs -r kill -9`) before retrying.

### Backend configuration

The benchmark defaults to `triton` decode / `aiter` prefill attention backends.
To change backends, write to `/workspace/bench_config.env`:
```bash
export DECODE_ATTENTION_BACKEND=aiter
export PREFILL_ATTENTION_BACKEND=aiter
export SGLANG_ROCM_FUSED_DECODE_MLA=1
```
This ensures the verification run uses the same configuration as your run.

Read the benchmark output logs carefully to identify which backends are active (attention,
MoE, all-reduce) before optimizing. Only optimize backends that are actually in use.

## Optimization Objective

Your single goal: **minimize the decode median latency reported by `bench_kimi_k25.sh`.**

There are NO restrictions on approach. You are free to:
- Switch between any available attention backends (triton, aiter, flashinfer, or any combination)
- Tune MoE kernel configurations (GEMM block sizes, waves_per_eu, etc.)
- Tune GEMM configs for small-M shapes (batch=1 decode)
- Fix bugs in any backend to unlock better performance
- Modify source code in `/sgl-workspace/sglang/` and `/sgl-workspace/aiter/`
- Try any optimization technique: kernel config tuning, torch.compile, graph capture, scheduling, backend switching, etc.

The only constraint: the final metric must come from `bench_kimi_k25.sh` with CUDA graphs enabled (no `--disable-cuda-graph`).

## Rules

- Do NOT use `pkill -f` — it kills your own shell. Use targeted `kill <PID>`.
- Read error messages carefully and fix the root cause.
- Final metrics must use CUDA graphs (no `--disable-cuda-graph`).
- Run `bench_kimi_k25.sh` as your LAST command to capture the final metric.
