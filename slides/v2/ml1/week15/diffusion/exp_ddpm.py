"""작은 DDPM 으로 손글씨 숫자 생성 → assets/ddpm.npz · ddpm.json
쇼츠 lessons/ml1/diffusion/src/train_ddpm.py 와 같은 모델 · 학습 (MLP 512 + 시간 임베딩, T = 1000, β 1e-4 → 0.02, 30,000걸음).
평가: 2,000장 생성 (DDPM 1,000단계 · DDIM 50단계 · DDIM 10단계), 숫자 분류기 (로지스틱 회귀) 로 종류 · 확신 (≥ 0.9) 비율.
VAE · GAN 덱과 같은 판정 방법."""
import os, json, math, time, numpy as np, torch, torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
HERE = os.path.dirname(os.path.abspath(__file__))
torch.manual_seed(0); np.random.seed(0)
d = load_digits(); X = torch.tensor(d.data / 8.0 - 1.0, dtype=torch.float32)
clf = LogisticRegression(max_iter=2000).fit(d.data / 16.0, d.target)
T = 1000; beta = torch.linspace(1e-4, 0.02, T); alpha = 1 - beta; abar = torch.cumprod(alpha, 0)
def temb(t, dim=64):
    f = torch.exp(-math.log(10000) * torch.arange(dim // 2) / (dim // 2)); a = t[:, None].float() * f[None]; return torch.cat([a.sin(), a.cos()], 1)
net = nn.Sequential(nn.Linear(128, 512), nn.SiLU(), nn.Linear(512, 512), nn.SiLU(), nn.Linear(512, 512), nn.SiLU(), nn.Linear(512, 64))
f = lambda x, t: net(torch.cat([x, temb(t)], 1))
opt = torch.optim.Adam(net.parameters(), 1e-3); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 30000); L = []
t0 = time.time()
for it in range(30000):
    x0 = X[torch.randint(0, len(X), (256,))]; t = torch.randint(0, T, (256,)); e = torch.randn_like(x0)
    xt = abar[t].sqrt()[:, None] * x0 + (1 - abar[t]).sqrt()[:, None] * e
    loss = ((f(xt, t) - e) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step(); sch.step(); L.append(loss.item())
print("train sec", round(time.time() - t0), "loss", round(np.mean(L[-500:]), 4), flush=True)

@torch.no_grad()
def ddpm(n, seed):
    g = torch.Generator().manual_seed(seed); x = torch.randn(n, 64, generator=g)
    for t in range(T - 1, -1, -1):
        eps = f(x, torch.full((n,), t)); x = (x - beta[t] / (1 - abar[t]).sqrt() * eps) / alpha[t].sqrt()
        if t > 0: x = x + beta[t].sqrt() * torch.randn(n, 64, generator=g)
    return x.clamp(-1, 1)

@torch.no_grad()
def ddim(n, steps, seed):
    g = torch.Generator().manual_seed(seed); x = torch.randn(n, 64, generator=g)
    ts = np.linspace(T - 1, 0, steps).round().astype(int)
    for i, t in enumerate(ts):
        eps = f(x, torch.full((n,), int(t))); x0 = (x - (1 - abar[t]).sqrt() * eps) / abar[t].sqrt()
        if i + 1 < len(ts):
            tn = ts[i + 1]; x = abar[tn].sqrt() * x0 + (1 - abar[tn]).sqrt() * eps
        else: x = x0
    return x.clamp(-1, 1)

def judge(x):
    p = clf.predict_proba(((x.numpy() + 1) / 2)); return np.bincount(p.argmax(1), minlength=10).tolist(), float((p.max(1) >= 0.9).mean())
res = {}; imgs = {}
for name, fn in [("ddpm1000", lambda: ddpm(2000, 1)), ("ddim50", lambda: ddim(2000, 50, 1)), ("ddim10", lambda: ddim(2000, 10, 1))]:
    t0 = time.time(); x = fn(); sec = time.time() - t0; c, q = judge(x); res[name] = dict(counts=c, conf=q, sec=sec); imgs[name] = x[:16].numpy()
    print(name, c, round(q, 3), round(sec, 1), flush=True)
res["data_conf"] = float((clf.predict_proba(d.data / 16.0).max(1) >= 0.9).mean())
res["abar"] = {str(t): float(abar[t - 1]) for t in (1, 100, 250, 500, 750, 1000)}
json.dump(res, open(os.path.join(HERE, "assets", "ddpm.json"), "w"), indent=1)
# 앞으로 섞기 그림: 숫자 하나
x0 = X[0:1]; e = torch.randn(1, 64, generator=torch.Generator().manual_seed(1))
FW = [0, 100, 250, 500, 750, 1000]
fwd = np.stack([(abar[s - 1].sqrt() * x0 + (1 - abar[s - 1]).sqrt() * e)[0].numpy() if s else x0[0].numpy() for s in FW])
np.savez_compressed(os.path.join(HERE, "assets", "ddpm.npz"), fwd=fwd, fwd_t=FW, loss=np.array(L[::100]), **{f"img_{k}": v for k, v in imgs.items()})
