#!/usr/bin/env python3
"""One-shot cq (local Qwen via litellm) invocation with real-time visible logging.
Usage: python3 cq_run_once.py <prompt_file> <log_file> [timeout_seconds]

Streams claude's --output-format stream-json events and writes a compact,
human-readable, line-flushed log so `tail -f` shows live progress (tool
calls, tool results, final answer, cost/turn summary).

Two hardening fixes over the original version (both were real ways this
could hang past its timeout instead of terminating):

1. The read loop used to be a blocking `proc.stdout.readline()` -- if
   claude went completely silent (no more stream-json events, e.g. stuck
   inside a long Bash tool call), the loop never got back to the top to
   re-check the deadline, so the timeout never fired. Now every read is
   gated by `select.select(..., timeout=1s)` so the loop always revisits
   the deadline check at least once a second no matter what claude is
   doing.
2. `proc.kill()` only signals the immediate `claude` child; anything
   *claude* itself spawned (e.g. a `jupyter nbconvert --execute` from a
   Bash tool call) is in the same process group and doesn't get killed
   with it -- it's reparented to init and keeps running as an orphan
   forever. The child is now started with `start_new_session=True` (its
   own process group) and killed via `os.killpg(...)` so the whole tree
   dies together.
"""
import json
import os
import select
import signal
import subprocess
import sys
import time


def trunc(s, n=220):
    s = str(s)
    s = s.replace("\n", "\\n")
    return s if len(s) <= n else s[:n] + f"...[{len(s)}chars]"


def _handle_line(line, lf):
    """Format+write one line of claude's stream-json output to the log.
    Returns the parsed `result` event dict if this line was one, else None."""
    if line.startswith("{"):
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            lf.write(f"[RAW] {trunc(line)}\n")
            return None
        out = fmt_event(d)
        if out:
            lf.write(out + "\n")
        if d.get("type") == "result":
            return d
        return None
    lf.write(f"[STDERR/OTHER] {trunc(line)}\n")
    return None


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
        start_new_session=True,  # own process group -> kill_tree() can reap everything it spawns
    )

    def kill_tree(p):
        # SIGKILL the whole process group (p itself + anything it spawned,
        # e.g. a Bash-tool child that's still running). ProcessLookupError
        # means it's already gone -- fine, that's the goal anyway.
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass

    rc = None
    result_line = None
    try:
        while True:
            elapsed = time.time() - t0
            if timeout and elapsed > timeout:
                kill_tree(proc)
                lf.write(f"\n=== TIMEOUT after {timeout}s, killed (process group) ===\n")
                rc = -1
                break
            if proc.poll() is not None:
                # process exited; drain whatever's left in the pipe without blocking
                for line in proc.stdout:
                    line = line.strip()
                    if not line:
                        continue
                    res = _handle_line(line, lf)
                    if res:
                        result_line = res
                rc = proc.returncode
                break
            # Never block on readline() longer than 1s (or the remaining
            # budget if shorter) -- this is what makes the timeout check
            # above actually get revisited even if claude goes completely
            # silent (e.g. stuck inside a long-running Bash tool call).
            wait = min(1.0, timeout - elapsed) if timeout else 1.0
            ready, _, _ = select.select([proc.stdout], [], [], max(wait, 0))
            if not ready:
                continue
            line = proc.stdout.readline()
            if line == "":
                continue  # EOF race with poll(); loop will catch the exit above
            line = line.strip()
            if not line:
                continue
            res = _handle_line(line, lf)
            if res:
                result_line = res
    except KeyboardInterrupt:
        kill_tree(proc)
        rc = -2

    dt = time.time() - t0
    lf.write(f"\n=== cq run ended {time.ctime()} (elapsed {dt:.1f}s, rc={rc}) ===\n")
    lf.close()
    print(f"rc={rc} elapsed={dt:.1f}s log={log_file}")
    if result_line:
        print(json.dumps(result_line))


if __name__ == "__main__":
    main()
