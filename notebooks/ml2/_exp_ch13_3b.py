"""13.3 본문용: 20K 스텝(100반복×200스텝) PPO@Reacher — 기존 예제의 재현 확인.
같은 시드 42, 같은 클래스/하이퍼파라미터. 단 한 번만 실행해 숫자 채택.
"""
import math
import numpy as np
import torch, torch.nn as nn
from torch.distributions import Normal
import gymnasium as gym

torch.manual_seed(42); np.random.seed(42)

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
N_ITER, STEPS = 100, 200   # 총 20,000 스텝

pre_s = evaluate(actor, False, 5000)
pre_d = evaluate(actor, True, 5000)
print(f"학습 전: stochastic={pre_s:.2f}  deterministic={pre_d:.2f}", flush=True)

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
        print(f"it {it+1:4d}  stochastic={evaluate(actor, False, 5000+it):8.2f}  "
              f"deterministic={evaluate(actor, True, 5000+it):8.2f}", flush=True)
env.close()
post_s = evaluate(actor, False, 5900)
post_d = evaluate(actor, True, 5900)
print(f"학습 후: stochastic={post_s:.2f}  deterministic={post_d:.2f}")
