"""One-shot verification + experiment for section 12.3 (BC vs RL).

(a) compounding-error closed form: P(all T steps correct) = p^T
    -> saves ch12_3_compounding_error.svg, prints table numbers
(b) 5x5 gridworld, three arms:
      A: pure BC   (expert demos only; random off-path)
      B: pure RL   (Q-learning from scratch)
      C: BC-init + RL (Q-learning warm-started from BC policy)
    - expert takes a safe but suboptimal 10-step detour (optimum = 8 steps)
    - pitfall cell (2,2) = reward -50, episode ends
    - training/deployment env has 10% disturbance (random adjacent move),
      so a pure-BC agent gets knocked off the demo path and acts OOD
    -> saves ch12_3_bc_rl_comparison.svg, prints final returns + pitfall counts
"""
import os
import math
import random

import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt

kr = [f.name for f in font_manager.fontManager.ttflist if "Noto Sans CJK KR" in f.name]
if kr:
    plt.rcParams["font.sans-serif"] = [kr[0]]
plt.rcParams["axes.unicode_minus"] = False

IMG = "/home/smhan/book-ml/kor/src/images"
if not os.path.isdir(IMG):
    IMG = "/tmp"

# ---------------------------------------------------------------- (a)
print("=== (a) compounding error: P(no mistake in T steps) = p^T ===")
for p in (0.98, 0.99, 0.995):
    row = "  ".join(f"T={T}: {p**T:.3f}" for T in (50, 100, 200, 300))
    print(f"p={p}: {row}")
for p in (0.98, 0.99, 0.995):
    t50 = math.log(0.5) / math.log(p)
    t05 = math.log(0.05) / math.log(p)
    print(f"p={p}: 50% survival at T={t50:.0f}, 5% survival at T={t05:.0f}")

T_grid = np.arange(1, 501)
fig, ax = plt.subplots(figsize=(8, 5))
for p, c in ((0.98, "tab:red"), (0.99, "tab:blue"), (0.995, "tab:green")):
    ax.plot(T_grid, p ** T_grid, lw=2, color=c, label=f"per-step success rate p = {p}")
ax.axhline(0.5, color="gray", ls="--", lw=1)
ax.axhline(0.05, color="gray", ls=":", lw=1)
ax.text(12, 0.53, "50%", fontsize=9, color="gray")
ax.text(12, 0.02, "5%", fontsize=9, color="gray")
ax.set_xlabel("episode length T (number of steps)")
ax.set_ylabel("P(no mistake in all T steps)  $p^T$")
ax.set_title("Compounding error: even a 99% per-step expert-level policy\n"
             "has only 5% chance of a flawless long episode")
ax.set_ylim(0, 1.05)
ax.legend(fontsize=9, loc="center right")
plt.tight_layout()
fa = os.path.join(IMG, "ch12_3_compounding_error.svg")
plt.savefig(fa, bbox_inches="tight")
plt.close()
print("saved", fa)

# ---------------------------------------------------------------- (b)
S = 5
START, GOAL, PIT = (4, 0), (0, 4), (2, 2)
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
DISTURB = 0.05  # per step: actual move = random adjacent cell (not commanded)
EP_MAX = 60
N_EP = 500

# expert path: safe but suboptimal (10 steps; optimum is 8)
EXPERT_PATH = [(4, 0), (3, 0), (3, 1), (2, 1), (1, 1),
               (1, 0), (0, 0), (0, 1), (0, 2), (0, 3), (0, 4)]
assert (2, 2) not in EXPERT_PATH and EXPERT_PATH[-1] == GOAL
expert_action = {}
for s, ns in zip(EXPERT_PATH, EXPERT_PATH[1:]):
    d = (ns[0] - s[0], ns[1] - s[1])
    expert_action[s] = ACTIONS.index(d)
PATH_STATES = set(EXPERT_PATH[:-1])
print(f"expert path length = {len(EXPERT_PATH) - 1} (return -{len(EXPERT_PATH)-1}); optimum = 8")

def neighbors(s):
    r, c = s
    out = []
    for a in range(4):
        nr = min(max(r + ACTIONS[a][0], 0), S - 1)
        nc = min(max(c + ACTIONS[a][1], 0), S - 1)
        out.append((nr, nc))
    return out

def step(s, a, rng):
    if rng.random() < DISTURB:
        ns = rng.choice(neighbors(s))      # disturbance: knocked one cell randomly
    else:
        r, c = s
        nr = min(max(r + ACTIONS[a][0], 0), S - 1)
        nc = min(max(c + ACTIONS[a][1], 0), S - 1)
        ns = (nr, nc)
    if ns == PIT:
        return ns, -50.0, True
    return ns, -1.0, (ns == GOAL)

# --- arm A: pure behavior cloning from clean expert demos (deterministic)
bc_policy = dict(expert_action)

def run_bc(n_ep, seed):
    rng = random.Random(seed)
    rets, pit_at = [], []
    for _ in range(n_ep):
        s, ret, done = START, 0.0, False
        for _t in range(EP_MAX):
            a = bc_policy[s] if s in bc_policy else rng.randrange(4)  # OOD -> random
            ns, r, done = step(s, a, rng)
            ret += r
            if ns == PIT:
                pit_at.append(_ + 1)
            s = ns
            if done:
                break
        rets.append(ret)
    return rets, pit_at

# --- Q-learning helper (arms B, C)
states_all = [(r, c) for r in range(S) for c in range(S) if (r, c) != PIT]

def q_learning(n_ep, q0, seed, eps0=0.5, eps_min=0.05, alpha=0.1, gamma=0.95):
    rng = random.Random(seed)
    rets, pit_at = [], []
    for ep in range(n_ep):
        eps = max(eps_min, eps0 * 0.97 ** ep)
        s, ret, done = START, 0.0, False
        for _t in range(EP_MAX):
            a = rng.randrange(4) if rng.random() < eps else max(range(4), key=lambda x: q0[s][x])
            q_old = q0[s][a]
            ns, r, done = step(s, a, rng)
            target = r if done else r + gamma * max(q0[ns])
            q0[s][a] = q_old + alpha * (target - q_old)
            ret += r
            if ns == PIT:
                pit_at.append(ep + 1)
            s = ns
            if done:
                break
        rets.append(ret)
    return rets, pit_at

rets_bc, pit_bc = run_bc(N_EP, seed=0)
q_scratch = {s: [0.0] * 4 for s in states_all}
rets_rl, pit_rl = q_learning(N_EP, q_scratch, seed=0, eps0=0.5)
# warm start = "BC gives an accurate initial value function":
# Q(s, expert_action) = -remaining_steps (true cost-to-go along the demo path)
# Q(s, other) = one step worse; off-path states stay 0 (to be discovered)
q_warm = {s: [0.0] * 4 for s in states_all}
for i, s in enumerate(EXPERT_PATH[:-1]):
    remaining = (len(EXPERT_PATH) - 1) - i   # 10, 9, ..., 1
    for a in range(4):
        q_warm[s][a] = -(remaining) if a == expert_action[s] else -(remaining) + 1.0
rets_warm, pit_warm = q_learning(N_EP, q_warm, seed=0, eps0=0.2)
pits_bc, pits_rl, pits_warm = len(pit_bc), len(pit_rl), len(pit_warm)
print("pitfalls in first 50 episodes (safety of the early phase):")
print(f"  pure BC: {sum(1 for e in pit_bc if e <= 50)},  pure RL: {sum(1 for e in pit_rl if e <= 50)},  "
      f"BC-init+RL: {sum(1 for e in pit_warm if e <= 50)}")

def moving_avg(x, w=20):
    return [sum(x[max(0, i - w + 1):i + 1]) / min(i + 1, w) for i in range(len(x))]

def first_beat_expert(rets, thr=-9.5, w=20):
    """first episode where w-ep moving average is better than expert level"""
    m = moving_avg(rets, w)
    for i, v in enumerate(m):
        if i + 1 >= w and v > thr:
            return i + 1
    return None

bc_mean = np.mean(rets_bc[-20:])
print(f"pure BC:      last-20 mean return = {bc_mean:.2f}, pitfalls = {pits_bc}/{N_EP}, "
      f"beats-expert episode = {first_beat_expert(rets_bc)}")
print(f"pure RL:      last-20 mean return = {np.mean(rets_rl[-20:]):.2f}, pitfalls = {pits_rl}/{N_EP}, "
      f"beats-expert episode = {first_beat_expert(rets_rl)}")
print(f"BC-init + RL: last-20 mean return = {np.mean(rets_warm[-20:]):.2f}, pitfalls = {pits_warm}/{N_EP}, "
      f"beats-expert episode = {first_beat_expert(rets_warm)}")

x = np.arange(1, N_EP + 1)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.5))
ax1.plot(x, moving_avg(rets_bc), color="tab:gray", lw=1.8, label="Pure BC (expert demos only)")
ax1.plot(x, moving_avg(rets_rl), color="tab:red", lw=1.4, alpha=0.9, label="Pure RL (from scratch)")
ax1.plot(x, moving_avg(rets_warm), color="tab:blue", lw=1.8, label="BC-init + RL (warm start)")
ax1.axhline(-8, color="green", ls="--", lw=1, label="optimal path = -8 (no disturbance)")
ax1.set_xlabel("episode")
ax1.set_ylabel("mean return (20-ep moving average)")
ax1.set_title("(a) final performance: RL reaches the 8-step optimum\n"
              "(beats the expert's 10-step detour); pure BC is capped")
ax1.legend(fontsize=8, loc="center right")

arms = ["pure BC", "pure RL", "BC-init + RL"]
cnt = [pits_bc, pits_rl, pits_warm]
bars = ax2.bar(arms, cnt, color=["tab:gray", "tab:red", "tab:blue"], alpha=0.85)
for b, v in zip(bars, cnt):
    ax2.text(b.get_x() + b.get_width() / 2, v + max(cnt) * 0.01, str(v), ha="center", fontsize=9)
ax2.set_ylabel("pitfalls (dangerous cells, reward -50)\nduring 500 episodes")
ax2.set_title("(b) safety cost: number of dangerous actions tried\nwhile learning in the environment")
plt.tight_layout()
fb = os.path.join(IMG, "ch12_3_bc_rl_comparison.svg")
plt.savefig(fb, bbox_inches="tight")
plt.close()
print("saved", fb)
