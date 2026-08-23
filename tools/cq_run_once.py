#!/usr/bin/env python3
"""One-shot cq (local Qwen via litellm) invocation with real-time visible logging.
Usage: python3 cq_run_once.py <prompt_file> <log_file> [timeout_seconds]

Streams claude's --output-format stream-json events and writes a compact,
human-readable, line-flushed log so `tail -f` shows live progress (tool
calls, tool results, final answer, cost/turn summary).
"""
import json
import os
import subprocess
import sys
import time


def trunc(s, n=220):
    s = str(s)
    s = s.replace("\n", "\\n")
    return s if len(s) <= n else s[:n] + f"...[{len(s)}chars]"


def fmt_event(d):
    t = d.get("type")
    if t == "system":
        if d.get("subtype") == "thinking_tokens":
            return None  # too noisy, skip
        if d.get("subtype") == "init":
            return f"[init] model={d.get('model')} cwd={d.get('cwd')}"
        return f"[system:{d.get('subtype')}]"
    if t == "assistant":
        msg = d.get("message", {})
        for block in msg.get("content", []):
            bt = block.get("type")
            if bt == "thinking":
                continue  # skip verbose reasoning
            if bt == "tool_use":
                return f"[TOOL_CALL] {block.get('name')} {trunc(block.get('input'))}"
            if bt == "text":
                txt = block.get("text", "").strip()
                if txt:
                    limit = 4000 if ("DONE:" in txt or "FAILED:" in txt) else 400
                    return f"[ASSISTANT] {trunc(txt, limit)}"
        return None
    if t == "user":
        msg = d.get("message", {})
        for block in msg.get("content", []):
            if block.get("type") == "tool_result":
                err = block.get("is_error", False)
                tag = "TOOL_ERROR" if err else "TOOL_RESULT"
                return f"[{tag}] {trunc(block.get('content'))}"
        return None
    if t == "result":
        return (f"[RESULT] error={d.get('is_error')} turns={d.get('num_turns')} "
                f"cost=${d.get('total_cost_usd')} duration_api_ms={d.get('duration_api_ms')} "
                f"usage={d.get('usage')}")
    return None


def main():
    prompt_file = sys.argv[1]
    log_file = sys.argv[2]
    timeout = int(sys.argv[3]) if len(sys.argv) > 3 else 3600

    with open(prompt_file, "r") as f:
        prompt = f.read()

    env = os.environ.copy()
    # Make sure cq's Bash tool calls can find conda-installed CLI tools
    # (mdbook, tectonic, dot, node, gh, ...) by bare name.
    env["PATH"] = "/home/smhan/miniconda3/bin:" + env.get("PATH", "")
    env.update({
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        "ANTHROPIC_BASE_URL": "http://localhost:4000",
        "ANTHROPIC_AUTH_TOKEN": "sk-local-litellm-key",
        "ANTHROPIC_MODEL": "qwen3.8-27b",
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": "qwen3.8-27b",
    })

    cmd = [
        "claude", "-p", prompt,
        "--dangerously-skip-permissions",
        "--no-session-persistence",
        "--output-format", "stream-json",
        "--verbose",
    ]

    t0 = time.time()
    lf = open(log_file, "w", buffering=1)
    lf.write(f"=== cq run started {time.ctime()} ===\n")

    proc = subprocess.Popen(
        cmd, cwd="/home/smhan/book-ml", env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    )

    rc = None
    result_line = None
    try:
        while True:
            if timeout and (time.time() - t0) > timeout:
                proc.kill()
                lf.write(f"\n=== TIMEOUT after {timeout}s, killed ===\n")
                rc = -1
                break
            line = proc.stdout.readline()
            if line == "" and proc.poll() is not None:
                rc = proc.returncode
                break
            line = line.strip()
            if not line:
                continue
            if line.startswith("{"):
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    lf.write(f"[RAW] {trunc(line)}\n")
                    continue
                out = fmt_event(d)
                if out:
                    lf.write(out + "\n")
                if d.get("type") == "result":
                    result_line = d
            else:
                lf.write(f"[STDERR/OTHER] {trunc(line)}\n")
    except KeyboardInterrupt:
        proc.kill()
        rc = -2

    dt = time.time() - t0
    lf.write(f"\n=== cq run ended {time.ctime()} (elapsed {dt:.1f}s, rc={rc}) ===\n")
    lf.close()
    print(f"rc={rc} elapsed={dt:.1f}s log={log_file}")
    if result_line:
        print(json.dumps(result_line))


if __name__ == "__main__":
    main()
