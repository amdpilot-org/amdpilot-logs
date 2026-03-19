#!/usr/bin/env python3
"""Convert kimi-cli context.jsonl to HuggingFace messages format.

Reads a raw context.jsonl (with _checkpoint, _usage metadata lines) and
outputs a single-line JSON object matching the KernelBench SFT schema.
"""

import json
import sys
from pathlib import Path

MAX_TOOL_OUTPUT_LEN = 8000


def convert(context_path: str, example_id: str, source: str,
            task_type: str, score: float) -> dict:
    with open(context_path) as f:
        raw_lines = [json.loads(line.strip()) for line in f if line.strip()]

    messages = []
    for entry in raw_lines:
        role = entry.get("role", entry.get("type", ""))
        if role in ("_checkpoint", "_usage"):
            continue

        content = entry.get("content", "")
        tool_calls = entry.get("tool_calls", [])
        tool_call_id = entry.get("tool_call_id", "")

        if isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, dict):
                    t = part.get("text", part.get("content", ""))
                    if t:
                        text_parts.append(str(t))
                elif isinstance(part, str):
                    text_parts.append(part)
            content = "\n".join(text_parts).strip()

        if isinstance(content, str) and len(content) > MAX_TOOL_OUTPUT_LEN and role == "tool":
            content = content[:MAX_TOOL_OUTPUT_LEN] + "\n[truncated]"

        msg: dict = {"role": role, "content": content}

        if tool_calls:
            normalized_tc = []
            for tc in tool_calls:
                fn = tc.get("function", {})
                normalized_tc.append({
                    "type": tc.get("type", "function"),
                    "id": tc.get("id", ""),
                    "function": {
                        "name": fn.get("name", ""),
                        "arguments": fn.get("arguments", "{}"),
                    },
                })
            msg["tool_calls"] = normalized_tc
            if not content:
                msg["content"] = ""

        if tool_call_id:
            msg["tool_call_id"] = tool_call_id

        messages.append(msg)

    num_tool_calls = sum(len(m.get("tool_calls", [])) for m in messages)
    num_turns = sum(1 for m in messages if m["role"] in ("user", "assistant"))

    return {
        "id": example_id,
        "source": source,
        "task_type": task_type,
        "score": score,
        "num_turns": num_turns,
        "num_tool_calls": num_tool_calls,
        "messages": messages,
    }


def main():
    configs = [
        {
            "context": "sglang-bugfix/raw/context.jsonl",
            "output": "sglang-bugfix/train.jsonl",
            "id": "sglang-19935-fp8-mla-decode-kscale-fix",
            "source": "sglang-sft-claude-executor",
            "task_type": "bugfix",
            "score": 100.0,
        },
        {
            "context": "sglang-feature/raw/context.jsonl",
            "output": "sglang-feature/train.jsonl",
            "id": "sglang-20187-fp8-prefill-radix-cache",
            "source": "sglang-sft-claude-executor",
            "task_type": "feature",
            "score": 100.0,
        },
    ]

    base = Path(__file__).parent
    for cfg in configs:
        ctx_path = base / cfg["context"]
        out_path = base / cfg["output"]
        example = convert(
            str(ctx_path), cfg["id"], cfg["source"],
            cfg["task_type"], cfg["score"],
        )
        with open(out_path, "w") as f:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

        meta = {
            "source_pr": f"sgl-project/sglang#{cfg['id'].split('-')[1]}",
            "task_type": cfg["task_type"],
            "score": cfg["score"],
            "executor_model": "Claude Opus 4.6",
            "num_turns": example["num_turns"],
            "num_tool_calls": example["num_tool_calls"],
            "num_messages": len(example["messages"]),
            "anti_reward_hacking": {
                "git_history_stripped": True,
                "ast_verification": True,
                "verified_against_human_pr": True,
            },
        }
        meta_path = base / cfg["output"].replace("train.jsonl", "metadata.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        print(f"Converted {cfg['id']}: {len(example['messages'])} messages, "
              f"{example['num_tool_calls']} tool calls -> {out_path}")


if __name__ == "__main__":
    main()
