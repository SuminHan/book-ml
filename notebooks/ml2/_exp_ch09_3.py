"""One-shot verification + figures for section 9.3 (experience replay & CartPole DQN curve).

Run once, adopt the numbers. Prints every figure the book text quotes and
saves two SVGs into kor/src/images/.
"""
import random
from collections import Counter, deque

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt

kr = [f.name for f in font_manager.fontManager.ttflist if "Noto Sans CJK KR" in f.name]
if kr:
    plt.rcParams["font.sans-serif"] = [kr[0]]
plt.rcParams["axes.unicode_minus"] = False

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

IMG = "/home/smhan/book-ml/kor/src/images"

# ---------------------------------------------------------------- 1. FIFO demo
print("=== (1) ReplayBuffer FIFO (capacity 4) ===")
buf = deque(maxlen=4)
for a in ["e1", "e2", "e3", "e4", "e5"]:
    buf.append(a)
    print("push", a, "->", list(buf))
r42 = random.Random(42)
print("random.sample(list(buf), 3), seed 42:", r42.sample(list(buf), 3))
r42 = random.Random(42)
draws = [r42.choice(list(buf)) for _ in range(12)]
print("12 with-replacement draws:", draws)
print("counts:", dict(sorted(Counter(draws).items())))

# ---------------------------------------------------------------- 2. env sanity
env = gym.make("CartPole-v1")
obs, _ = env.reset(seed=0)
obs, r, term, trunc, _ = env.step(0)
print("=== (2) CartPole sanity ===")
print("obs0:", np.round(obs, 4), "r:", r, "max_episode_steps:", env.spec.max_episode_steps)
env.close()

# ---------------------------------------------------------------- 3. DQN
GAMMA = 0.99
BATCH = 32
WARMUP = 1000
CAPACITY = 10000
EPS_START, EPS_END, EPS_STEP_DECAY = 1.0, 0.05, 0.9997
LR = 1e-3
EPISODES = 300
UPDATE_FREQ = 100

class QNetwork(nn.Module):
    def __init__(self, state_dim, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, n_actions),
        )
    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def push(self, t):
        self.buffer.append(t)
    def __len__(self):
        return len(self.buffer)
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

def dqn_loss(Q_net, target_net, s, a, r, s_next, dones, gamma):
    q_pred = Q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        q_next = target_net(s_next).max(dim=1).values
        target = r + gamma * q_next * (1 - dones)
    return F.smooth_l1_loss(q_pred, target)

def train_dqn(seed, episodes=EPISODES):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    env = gym.make("CartPole-v1")
    Q_net = QNetwork(4, 2)
    target_net = QNetwork(4, 2)
    target_net.load_state_dict(Q_net.state_dict())
    opt = torch.optim.Adam(Q_net.parameters(), lr=LR)
    buffer = ReplayBuffer(CAPACITY)
    rets, losses, total_step = [], [], 0
    best = 0.0
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + 1000 * ep)
        done, ret = False, 0.0
        while not done:
            eps = max(EPS_END, EPS_START * EPS_STEP_DECAY ** total_step)
            if random.random() < eps:
                a = random.randrange(env.action_space.n)   # reproducible (random.seed)
            else:
                a = int(Q_net(torch.tensor(obs, dtype=torch.float32)).argmax())
            ns, r, term, trunc, _ = env.step(a)
            buffer.push((obs, a, r, ns, float(term)))
            done = term or trunc
            obs = ns
            ret += r
            total_step += 1
            if len(buffer) >= WARMUP:
                batch = buffer.sample(BATCH)
                s = torch.tensor(np.array([b[0] for b in batch], dtype=np.float32))
                at = torch.tensor([b[1] for b in batch])
                rr = torch.tensor([b[2] for b in batch], dtype=torch.float32)
                ns_ = torch.tensor(np.array([b[3] for b in batch], dtype=np.float32))
                dn = torch.tensor([b[4] for b in batch], dtype=torch.float32)
                loss = dqn_loss(Q_net, target_net, s, at, rr, ns_, dn, GAMMA)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(Q_net.parameters(), 10.0)
                opt.step()
                losses.append(loss.item())
            if total_step % UPDATE_FREQ == 0:
                target_net.load_state_dict(Q_net.state_dict())
        rets.append(ret)
        best = max(best, ret)
    env.close()
    return dict(rets=np.array(rets), losses=np.array(losses), best=best)

results = {}
for seed in (0, 1, 2):
    results[seed] = train_dqn(seed)
    res = results[seed]
    print(f"=== DQN seed {seed} ===")
    print("first20 mean:", round(float(np.mean(res["rets"][:20])), 2))
    print("last20 mean :", round(float(np.mean(res["rets"][-20:])), 2))
    print("best        :", res["best"])
    for e in (0, 50, 100, 150, 200, 250, 299):
        print(f"  episode {e}: {res['rets'][e]:.0f}")
    print("loss first5 mean:", round(float(np.mean(res["losses"][:5])), 4))
    print("loss last50 mean:", round(float(np.mean(res["losses"][-50:])), 4))
np.save(IMG.replace("/images", "") + "/.ch09_3_returns.npy", results[0]["rets"])

# ---------------------------------------------------------------- 4. learning curve (3 seeds)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
ax = axes[0]
for seed, color in ((0, "#1f77b4"), (1, "#d62728"), (2, "#2ca02c")):
    rets = results[seed]["rets"]
    sm = np.convolve(rets, np.ones(10) / 10, mode="valid")
    ax.plot(np.arange(1, 301), rets, color=color, alpha=0.15, lw=0.8)
    ax.plot(np.arange(10, 301), sm, color=color, lw=1.8, label=f"시드 {seed} (10-에피소드 이동평균)")
ax.axhline(500, color="k", ls=":", lw=1)
ax.text(299, 460, "500 (만점)", ha="right", fontsize=8, color="k")
ax.set_xlabel("에피소드")
ax.set_ylabel("에피소드 리턴 (스텝 수)")
ax.set_title("CartPole DQN 학습곡선 — 시드 3개")
ax.legend(fontsize=8, loc="lower right")
ax.set_xlim(0, 301)
ax.set_ylim(0, 520)

ax = axes[1]
res0 = results[0]
losses = res0["losses"]
w = 200
smloss = np.convolve(losses, np.ones(w) / w, mode="valid")
ax.plot(np.arange(len(losses)), losses, color="#7f7f7f", alpha=0.15, lw=0.6)
ax.plot(np.arange(w, len(losses) + 1), smloss, color="#8c564b", lw=1.8,
        label="TD 손실 (200-스텝 이동평균, 시드 0)")
ax.set_yscale("log")
ax.set_xlabel("학습 스텝 (버퍼 채움 이후)")
ax.set_ylabel("TD 손실 (log)")
ax.set_title("DQN 손실 곡선")
ax.legend(fontsize=8, loc="upper right")
fig.tight_layout()
fig.savefig(IMG + "/ch09_3_dqn_cartpole_curve.svg", bbox_inches="tight")
plt.close(fig)
print("saved", IMG + "/ch09_3_dqn_cartpole_curve.svg")

# ---------------------------------------------------------------- 5. buffer cycle diagram
fig, ax = plt.subplots(figsize=(9.5, 5.2))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")

def box(x0, y0, w, h, title, body, fc, ec):
    ax.add_patch(plt.Rectangle((x0, y0), w, h, fc=fc, ec=ec, lw=1.6, zorder=2))
    ax.text(x0 + w / 2, y0 + h - 0.42, title, ha="center", va="top", fontsize=11,
            fontweight="bold", zorder=3)
    ax.text(x0 + w / 2, y0 + h - 1.05, body, ha="center", va="top", fontsize=8.5, zorder=3)

def arrow(p0, p1, text, tpos=None, color="k"):
    ax.annotate("", xy=p1, xytext=p0,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5), zorder=1)
    if text:
        mx, my = tpos if tpos else ((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2)
        ax.text(mx, my, text, ha="center", va="center", fontsize=9, color=color, zorder=3,
                bbox=dict(fc="white", ec="none", pad=1.5))

box(0.4, 6.8, 3.6, 2.4, "에이전트 (환경)",
    "ε-greedy 정책으로 행동\n(s, a, r, s', done)\n스텝 진행", "#e8f0fe", "#1f77b4")
box(0.4, 1.0, 4.2, 3.0, "재현 버퍼 (capacity 10,000)",
    "FIFO: 가득 차면\n가장 오래된 항목이\n밀려나고 새 항목이\n끝에 추가됨", "#fff3e0", "#e67e22")
box(5.6, 1.0, 4.0, 3.0, "훈련 미니배치",
    "버퍼에서 무작위로\n32개 비복원 추출\n(s, a, r, s', done)", "#eafaf1", "#2ca02c")
box(5.6, 6.8, 4.0, 2.4, "Q-network  →  타겟 network",
    "TD 오차 손실(Huber) + 역전파\n100 스텝마다 $\\theta^- \\leftarrow \\theta$\n(하드 동기화)", "#fdeef4", "#c2185b")

arrow((2.2, 6.8), (2.2, 4.05), "① push\n매 스텝 1개", (1.35, 5.45), "#1f77b4")
arrow((4.65, 2.5), (5.55, 2.5), "② sample\n배치 32개", (5.1, 3.15), "#2ca02c")
arrow((7.6, 4.05), (7.6, 6.75), "③ 손실·역전파\n(배치마다 1회)", (8.9, 5.4), "#c2185b")
arrow((5.55, 7.9), (4.05, 7.9), "④ 정책 개선", (4.8, 8.3), "#1f77b4")
ax.text(0.4, 0.35, "① ② ③ ④가 학습 루프의 한 바퀴 — ①은 매 스텝마다, ②③은 버퍼에 1,000개(워밍업)가 쌓인 뒤 매 스텝, ④는 100스텝 주기로.",
        fontsize=9, ha="left", va="bottom", style="italic")
fig.savefig(IMG + "/ch09_3_replay_buffer_cycle.svg", bbox_inches="tight")
plt.close(fig)
print("saved", IMG + "/ch09_3_replay_buffer_cycle.svg")
print("=== DONE ===")
