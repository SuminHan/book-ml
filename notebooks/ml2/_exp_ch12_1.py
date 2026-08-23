"""One-shot verification for 12.1 (behavior cloning / DAgger) numbers."""
import math

# ---------- grid world ----------
# rows: 0 = top corridor, 1 = middle (under construction), 2 = bottom
# cols: 0..8, goal = any row col 8, dead = (1,3),(1,4)
DEAD = {(1, 3), (1, 4)}
GOAL_COL = 8

def expert_policy(s):
    r, c = s
    if c >= GOAL_COL:
        return None
    if r != 0:
        return "up"          # not on top corridor -> get back to it
    return "right"

def step(s, a):
    r, c = s
    if a == "right": return (r, c + 1)
    if a == "left":  return (r, c - 1)
    if a == "up":    return (r - 1, c)
    if a == "down":  return (r + 1, c)

def demo_from(s0):
    s, acts, path = s0, [], [s0]
    while s[1] < GOAL_COL:
        a = expert_policy(s)
        s = step(s, a)
        acts.append(a)
        path.append(s)
        assert s not in DEAD, "expert crashed?!"
    return list(zip(path[:-1], acts)), path

demo, demo_path = demo_from((0, 0))
print("DEMO (state: action):")
for s, a in demo:
    print(f"  {s}: {a}")
print("DEMO PATH:", " -> ".join(str(x) for x in demo_path))

# ---------- BC = lookup table + 1-NN fallback ----------
table = dict(demo)

def nn_action(s, table):
    r, c = s
    best = min(table.items(), key=lambda kv: (abs(kv[0][0] - r) + abs(kv[0][1] - c), kv[0][1]))
    return best[1]

def bc_policy(s):
    return table.get(s, nn_action(s, table))

def rollout(policy, s0, max_steps=50):
    s, path, taken = s0, [s0], []
    for _ in range(max_steps):
        if s in DEAD:
            return path, taken, "crash"
        if s[1] >= GOAL_COL:
            return path, taken, "goal"
        a = policy(s)
        taken.append((s, a))
        s = step(s, a)
        path.append(s)
    return path, taken, "timeout"

print("\n--- deploy from (1,1) with initial BC (demo only) ---")
for s in [(1, 1), (1, 2), (1, 0), (1, 5)]:
    nearest = min(table.items(), key=lambda kv: abs(kv[0][0] - s[0]) + abs(kv[0][1] - s[1]))
    print(f"  state {s}: NN demo state={nearest[0]} action={nearest[1]}")
p, t, res = rollout(bc_policy, (1, 1))
print("BC path:", " -> ".join(str(x) for x in p), "=>", res)
print("BC actions:", t)

# more expert demos from same (or other top-row) starts -> does it help?
for extra_start in [(0, 0), (0, 1), (0, 2)]:
    d, _ = demo_from(extra_start)
    for s, a in d:
        table.setdefault(s, a)   # table already has them; NN table unchanged in effect
p, t, res = rollout(bc_policy, (1, 1))
print("after extra top-row demos, from (1,1):", " -> ".join(str(x) for x in p), "=>", res)

# ---------- DAgger ----------
print("\n--- DAgger ---")
D = dict(demo)
for rnd in range(1, 4):
    s, path, taken = (1, 1), [(1, 1)], []
    while s not in DEAD and s[1] < GOAL_COL:
        a = D.get(s, nn_action(s, D))     # agent follows its OWN policy
        D[s] = expert_policy(s)           # expert labels the VISITED state
        path.append(step(s, a))
        s = path[-1]
    result = "crash" if s in DEAD else "goal"
    print(f"round {rnd}: agent path = {' -> '.join(str(x) for x in path)} => {result}")
    if result == "goal":
        break
print("dataset size:", len(D))

# ---------- compounding error ----------
print("\n--- compounding error: P(at least one of H steps is wrong) = 1-(1-eps)^H ---")
for eps in (0.02, 0.05):
    line = f"eps={eps}: "
    for H in (1, 10, 50, 100, 500):
        line += f" H={H}: {1 - (1 - eps) ** H:.4f}"
    print(" ", line)
