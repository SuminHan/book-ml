"""One-shot verification + experiment for section 9.2.

(a) hand-checkable numbers (error amplification, soft-update lag)
(b) real small-network DQN, with vs without target network, on a 1D random walk
    -> saves the learning-curve SVG and prints the key numbers.
"""
import os
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

IMG = "/home/smhan/book-ml/kor/src/images"
if not os.path.isdir(IMG):
    IMG = "/tmp"

X_MAX = 12
GAMMA = 0.99
STEP_R = -1.0        # per-step cost (survival cost)
GOAL_R = +10.0       # reach x=12 (goal)
HIT_R = -10.0        # reach x=0  (hazard)
DRIFT = 0.9          # prob of moving in the action's intended direction


# ---------------- stochastic transition model (shared by env + true value) ----
def trans(x, a):
    """return aggregated [(x2, r, done, prob)] for a single action in state x."""
    if a == 0:  # left: mostly x-1
        cands = [(max(0, x - 1), DRIFT), (min(X_MAX, x + 1), 1.0 - DRIFT)]
    else:       # right: mostly x+1
        cands = [(min(X_MAX, x + 1), DRIFT), (max(0, x - 1), 1.0 - DRIFT)]
    agg = defaultdict(float)
    for x2, p in cands:
        agg[x2] += p
    out = []
    for x2, p in agg.items():
        done = (x2 == 0) or (x2 == X_MAX)
        r = STEP_R + (GOAL_R if x2 == X_MAX else HIT_R if x2 == 0 else 0.0)
        out.append((x2, r, done, p))
    return out


def env_step(x, a):
    """sample a concrete transition (for on-policy rollout)."""
    table = trans(x, a)
    xs = [t[0] for t in table]
    ps = [t[3] for t in table]
    x2 = random.choices(xs, weights=ps, k=1)[0]
    done = (x2 == 0) or (x2 == X_MAX)
    r = STEP_R + (GOAL_R if x2 == X_MAX else HIT_R if x2 == 0 else 0.0)
    return x2, r, done


# ---------------- (a) hand numbers -------------------------------------------
print("=== (a) hand-checkable numbers ===")
for g in [0.9, 0.99]:
    print(f"gamma={g}: 1/(1-gamma) = {1/(1-g):.1f} steps (error half-life ~{np.log(0.5)/np.log(g):.1f})")
# soft update effective lag
for tau in [0.005, 0.01]:
    print(f"soft update tau={tau}: effective lag ~1/tau = {1/tau:.0f} steps")
# hard update
print("hard update C=1000: target can be up to 1000 steps stale")
# existing by-hand example (keep, re-verify)
for th in [0.0, 0.5, 0.9]:
    print(f"no-target example: 1+0.99*{th} = {1+0.99*th:.3f}")


# ---------------- true value via value iteration -----------------------------
def true_values():
    Q = np.zeros((X_MAX + 1, 2))
    for _ in range(5000):
        nQ = Q.copy()
        for x in range(1, X_MAX):  # non-terminal only
            for a in range(2):
                val = 0.0
                for x2, r, done, p in trans(x, a):
                    nxt = 0.0 if done else GAMMA * np.max(Q[x2])
                    val += p * (r + nxt)
                nQ[x, a] = val
        Q = nQ
    return Q


Qtrue = true_values()
probe_x = 8
print(f"\n=== true values (value iteration) ===")
print(f"Q*({probe_x}, left) = {Qtrue[probe_x,0]:.4f}   Q*({probe_x}, right) = {Qtrue[probe_x,1]:.4f}")
print(f"argmax at {probe_x} = {'right' if Qtrue[probe_x,1] > Qtrue[probe_x,0] else 'left'}")


# ---------------- (b) DQN experiment -----------------------------------------
class QNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 2))

    def forward(self, x):
        return self.net(x)


def run(use_target, seed=0, steps=25000, lr=1e-2, sync=200, eps=0.1, max_ep=40):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    net = QNet()
    tnet = None
    if use_target:
        tnet = QNet()
        tnet.load_state_dict(net.state_dict())
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    probe_steps, probe_qr, probe_ql, loss_vals = [], [], [], []
    step_id = 0
    while step_id < steps:
        x = random.randint(1, X_MAX - 1)
        done = False
        ep = 0
        while not done and ep < max_ep:
            s1 = torch.tensor([[x / X_MAX]], dtype=torch.float32)
            with torch.no_grad():
                q = net(s1)
            a = random.randrange(2) if random.random() < eps else int(q.argmax().item())
            x2, r, done = env_step(x, a)
            s2 = torch.tensor([[x2 / X_MAX]], dtype=torch.float32)
            with torch.no_grad():
                tgt_mask = 0.0 if done else 1.0
            if use_target:
                with torch.no_grad():
                    tgt = r + GAMMA * tnet(s2).max(1).values * tgt_mask
            else:
                tgt = r + GAMMA * net(s2).max(1).values * tgt_mask  # moving target (in-graph)
            pred = net(s1).gather(1, torch.tensor([[a]])).squeeze()
            loss = (pred - tgt) ** 2
            opt.zero_grad()
            loss.backward()
            opt.step()
            step_id += 1
            if step_id % 25 == 0:
                with torch.no_grad():
                    qv = net(torch.tensor([[probe_x / X_MAX]])).numpy()[0]
                probe_steps.append(step_id)
                probe_qr.append(qv[1])
                probe_ql.append(qv[0])
                loss_vals.append(loss.item())
            if use_target and step_id % sync == 0:
                tnet.load_state_dict(net.state_dict())
            x = x2
            ep += 1
    return np.array(probe_steps), np.array(probe_qr), np.array(probe_ql), np.array(loss_vals)


def moving_avg(y, w=40):
    k = np.ones(w) / w
    return np.convolve(y, k, mode="same")


print("\n=== (b) running experiments (this may take ~1 min) ===")
stA, qrA, qlA, loA = run(use_target=False)
stB, qrB, qlB, loB = run(use_target=True)

print(f"\nno-target  : final Q({probe_x},right)={qrA[-1]:.3f}  std(last 100)={qrA[-100:].std():.3f}  mean_loss(last100)={loA[-100:].mean():.3f}")
print(f"with-target: final Q({probe_x},right)={qrB[-1]:.3f}  std(last 100)={qrB[-100:].std():.3f}  mean_loss(last100)={loB[-100:].mean():.3f}")
print(f"true Q*({probe_x},right) = {Qtrue[probe_x,1]:.3f}")
for tag, st, qr, lo in [("no-target", stA, qrA, loA), ("with-target", stB, qrB, loB)]:
    for idx in [len(st) // 4, len(st) // 2, 3 * len(st) // 4]:
        print(f"  {tag} step {int(st[idx])}: Q={qr[idx]:.3f}  loss={lo[idx]:.3f}")


# ---------------- figure -------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.0))
ax1.plot(stA / 1000, moving_avg(qrA), color="#e6550d", lw=2.2, label="no target net (moving target)")
ax1.plot(stB / 1000, moving_avg(qrB), color="#3182bd", lw=2.2, label="target net (hard, C=200)")
ax1.axhline(Qtrue[probe_x, 1], color="black", ls="--", lw=1.5, label=f"true $Q^* = {Qtrue[probe_x,1]:.2f}$")
ax1.set_xlabel(r"training step ($\times 10^3$)")
ax1.set_ylabel(f"Q({probe_x}, right)")
ax1.set_title("(a) probe-state Q-value")
ax1.legend(fontsize=8)
ax1.grid(alpha=0.3)
ax2.plot(stA / 1000, moving_avg(loA), color="#e6550d", lw=2.2, label="no target net")
ax2.plot(stB / 1000, moving_avg(loB), color="#3182bd", lw=2.2, label="target net")
ax2.set_yscale("log")
ax2.set_xlabel(r"training step ($\times 10^3$)")
ax2.set_ylabel("TD loss (log)")
ax2.set_title("(b) training loss")
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3, which="both")
fig.suptitle("DQN target network: moving vs. fixed target", y=1.02, fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(IMG, "ch09_2_target_net_effect.svg"), bbox_inches="tight")
print(f"\nsaved figure -> {os.path.join(IMG, 'ch09_2_target_net_effect.svg')}")
