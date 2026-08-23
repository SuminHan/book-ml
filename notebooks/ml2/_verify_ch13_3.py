"""13.3 검증 (최종): Reacher-v5에서 PPO — 단일 검증 스크립트.
obs[8:10] = fingertip - target (진짜 오차 벡터)를 쓰는 비례제어기 베이스라인 포함.
실행: /home/smhan/miniconda3/envs/bookml/bin/python _verify_ch13_3.py
"""
import math, time, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt
kr = [f.name for f in font_manager.fontManager.ttflist if "Noto Sans CJK KR" in f.name]
if kr: plt.rcParams["font.sans-serif"] = [kr[0]]
plt.rcParams["axes.unicode_minus"] = False
import torch, torch.nn as nn
from torch.distributions import Normal
import gymnasium as gym

IMG = "/home/smhan/book-ml/kor/src/images"
torch.manual_seed(42); np.random.seed(42)

env = gym.make("Reacher-v5")

# ---------- 0) 목표 지점 분포 (반지름 상한 확인) ----------
targets = []
for i in range(100):
    s, _ = env.reset(seed=i)
    targets.append(np.linalg.norm(s[4:6]))
targets = np.array(targets)
print(f"목표 반지름: min={targets.min():.3f} max={targets.max():.3f}  (상한 0.2)")

def run_policy(policy_fn, n_ep=10, seed0=100):
    rets = []
    for ep in range(n_ep):
        s, _ = env.reset(seed=seed0 + ep)
        tot = 0.0
        for _ in range(50):
            a = policy_fn(s)
            s, r, term, trunc, _ = env.step(a)
            tot += r
            if term or trunc:
                break
        rets.append(tot)
    return rets

zero = run_policy(lambda s: np.zeros(2), seed0=200)
rand = run_policy(lambda s: env.action_space.sample(), seed0=42)
prop05 = run_policy(lambda s: np.clip(-0.5 * s[8:10], -1, 1), seed0=100)
prop20 = run_policy(lambda s: np.clip(-2.0 * s[8:10], -1, 1), seed0=100)
print(f"무력(0) 정책:          {np.mean(zero):8.2f}  {[round(float(x),1) for x in zero]}")
print(f"무작위 정책:           {np.mean(rand):8.2f}  {[round(float(x),1) for x in rand]}")
print(f"비례제어기 k=0.5:       {np.mean(prop05):8.2f}  {[round(float(x),1) for x in prop05]}")
print(f"비례제어기 k=2.0:       {np.mean(prop20):8.2f}  {[round(float(x),1) for x in prop20]}")
env.close()

# ---------- PPO (11.2와 동일한 구조) ----------
class GaussianPolicy(nn.Module):
    def __init__(self, state_dim, act_dim, act_high=1.0):
        super().__init__()
        self.act_high = act_high
        self.net = nn.Sequential(nn.Linear(state_dim, 64), nn.Tanh(),
                                 nn.Linear(64, 64), nn.Tanh())
        self.mu = nn.Linear(64, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim) - 0.5)
    def forward(self, x):
        h = self.net(x)
        return torch.tanh(self.mu(h)) * self.act_high, torch.exp(self.log_std)

class ValueNet(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, 64), nn.Tanh(),
                                 nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))
    def forward(self, x):
        return self.net(x).squeeze(-1)

def gae(rewards, values, dones, gamma, lam):
    adv, g = np.zeros_like(rewards), 0.0
    for t in reversed(range(len(rewards))):
        last = 1.0 if t == len(rewards) - 1 else 0.0
        delta = rewards[t] + gamma * values[t + 1] * (1 - last) - values[t]
        g = delta + gamma * lam * (1 - last) * g
        adv[t] = g
    return adv, adv + values[:-1]

def evaluate(net, deterministic, seed0, n_ep=10):
    e = gym.make("Reacher-v5"); rets = []
    for ep in range(n_ep):
        s, _ = e.reset(seed=seed0 + ep); tot = 0.0
        for _ in range(50):
            st = torch.tensor(np.asarray(s, dtype=np.float32))
            with torch.no_grad():
                mu, std = net(st)
                a = mu.numpy() if deterministic else Normal(mu, std).sample().numpy()
            s, r, t_, tr_, _ = e.step(np.clip(a, -1, 1))
            tot += r
            if t_ or tr_:
                break
        rets.append(tot)
    e.close()
    return float(np.mean(rets))

env = gym.make("Reacher-v5")
actor, critic = GaussianPolicy(10, 2, 1.0), ValueNet(10)
opt = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=3e-4)
EPS, GAMMA, LAM, N_EPOCH, MB = 0.2, 0.99, 0.95, 4, 64
N_ITER, STEPS = 1000, 1024   # 총 1,024,000 스텝

curves_s, curves_d, logstd_hist = [], [], []
t0 = time.time()
for it in range(N_ITER):
    s, _ = env.reset()
    buf = []
    for _ in range(STEPS):
        s_old = np.asarray(s, dtype=np.float32)
        st = torch.tensor(s_old)
        with torch.no_grad():
            mu, std = actor(st)
            d = Normal(mu, std)
            a_t = d.sample()
            lp = d.log_prob(a_t).sum().item()
            v = critic(st).item()
        s, r, term, trunc, _ = env.step(np.clip(a_t.numpy(), -1.0, 1.0))
        buf.append((s_old, a_t.numpy(), r, float(term or trunc), lp, v))
        if term or trunc:
            s, _ = env.reset()
    states = np.stack([b[0] for b in buf])
    acts = torch.tensor(np.stack([b[1] for b in buf]))
    rews = np.array([b[2] for b in buf]); dones = np.array([b[3] for b in buf])
    lpo = torch.tensor(np.array([b[4] for b in buf]))
    vs = np.append(np.array([b[5] for b in buf]), 0.0)
    adv, rets = gae(rews, vs, dones, GAMMA, LAM)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    Rb = torch.tensor(rets, dtype=torch.float32)
    Ab = torch.tensor(adv, dtype=torch.float32)
    Sb = torch.tensor(states, dtype=torch.float32)
    perm = np.random.permutation(len(buf))
    for _ in range(N_EPOCH):
        for start in range(0, len(buf), MB):
            sel = perm[start:start + MB]
            mu, std = actor(Sb[sel]); d = Normal(mu, std)
            ratio = torch.exp(d.log_prob(acts[sel]).sum(dim=1) - lpo[sel])
            surr = torch.min(ratio * Ab[sel], torch.clamp(ratio, 1 - EPS, 1 + EPS) * Ab[sel])
            v = critic(Sb[sel])
            loss = -surr.mean() + 0.5 * nn.functional.mse_loss(v, Rb[sel]) - 0.01 * d.entropy().sum()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()), 0.5)
            opt.step()
            with torch.no_grad():
                actor.log_std.clamp_(max=math.log(1.0))
    if (it + 1) % 25 == 0:
        curves_s.append(evaluate(actor, deterministic=False, seed0=5000 + it))
        curves_d.append(evaluate(actor, deterministic=True, seed0=5000 + it))
        logstd_hist.append(actor.log_std.detach().clone().tolist())
        print(f"it {it+1:4d}  stochastic={curves_s[-1]:8.2f}  deterministic={curves_d[-1]:8.2f}  elapsed={time.time()-t0:5.1f}s", flush=True)
env.close()
t_end = time.time() - t0
print(f"\n총 스텝 {N_ITER*STEPS:,} | 소요 {t_end:.0f}s")
print(f"deterministic: first={curves_d[0]:.2f}  last={curves_d[-1]:.2f}")
print(f"stochastic:    first={curves_s[0]:.2f}  last={curves_s[-1]:.2f}")
print(f"최종 sigma: {[round(math.exp(v), 3) for v in logstd_hist[-1]]}")
print(f"베이스라인: random={np.mean(rand):.2f}  prop05={np.mean(prop05):.2f}  zero={np.mean(zero):.2f}")

# ---------- 학습 곡선 SVG ----------
x = np.arange(25, N_ITER + 1, 25)
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(x, curves_d, color="tab:blue", linewidth=2.2,
        label="결정론적 정책 (평균 μ 출력) — 로봇에 배포되는 형태")
ax.plot(x, curves_s, color="tab:orange", linewidth=1.4, alpha=0.85,
        label="확률적 에피소드 (샘플링 노이즈 포함)")
ax.axhline(float(np.mean(prop05)), color="gray", linestyle="--", linewidth=1.2,
           label=f"손으로 만든 비례제어기 k=0.5 ≈ {np.mean(prop05):.1f}")
ax.axhline(float(np.mean(rand)), color="silver", linestyle=":", linewidth=1.4,
           label=f"무작위 정책 ≈ {np.mean(rand):.1f}")
ax.set_xlabel("반복 (회당 1024 스텝)")
ax.set_ylabel("10에피소드 평균 리턴 (0에 가까울수록 좋음)")
ax.set_title(f"PPO on Reacher-v5 — 두 개의 학습 곡선 (시드 42, {N_ITER*STEPS:,} 스텝)")
ax.invert_yaxis()
ax.legend(fontsize=8, loc="lower right")
plt.tight_layout()
p = os.path.join(IMG, "ch13_3_ppo_reacher_curves.svg")
plt.savefig(p, bbox_inches="tight")
print(f"SVG 저장: {p}")
