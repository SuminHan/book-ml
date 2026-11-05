"""GAN 실험 → assets/gan.npz · gan.json
A. 고리 (쇼츠 lessons/ml1/gan/src/train_gan.py 와 같은 설정): 반지름 2 고리, 생성자 · 판별자 MLP 128, Adam 2e-4 (β1 0.5), 5,000걸음, 단계별 생성 점 · 판별자 평균 확률.
B. 봉우리 8개 (반지름 2 원 위, 표준편차 0.05): 같은 구조로 시드 5개 × 5,000걸음. 1,000걸음마다 덮은 봉우리 수 (생성 점 2,000개 중 봉우리 3σ 안에 20개 이상).
C. 손글씨 숫자 (sklearn digits 8×8): 생성자 z 16 → 256 → 256 → 64 (시그모이드), 판별자 64 → 256 → 256 → 1, 10,000걸음, 배치 128.
   뽑은 2,000장을 숫자 분류기 (로지스틱 회귀) 로 판정 → 숫자 종류 분포, 최대 확률 ≥ 0.9 비율."""
import os, json, numpy as np, torch, torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
HERE = os.path.dirname(os.path.abspath(__file__))
bce = nn.BCEWithLogitsLoss()

def mlp(i, o, h=128, out=None, leaky=False):
    act = (lambda: nn.LeakyReLU(0.2)) if leaky else nn.ReLU
    layers = [nn.Linear(i, h), act(), nn.Linear(h, h), act(), nn.Linear(h, o)]
    return nn.Sequential(*(layers + ([out] if out else [])))

def train(real, zdim, xdim, steps, h=128, seed=0, snap=(), out=None, lr=2e-4, cb=None):
    torch.manual_seed(seed)
    G = mlp(zdim, xdim, h, out); D = mlp(xdim, 1, h, leaky=True)
    oG = torch.optim.Adam(G.parameters(), lr, betas=(0.5, 0.999)); oD = torch.optim.Adam(D.parameters(), lr, betas=(0.5, 0.999))
    Z = torch.randn(200, zdim); snaps, dlog, extra = {}, {}, {}
    for it in range(steps + 1):
        if it in snap:
            with torch.no_grad():
                snaps[it] = G(Z).numpy(); xr = real(2000); xf = G(torch.randn(2000, zdim))
                dlog[it] = (torch.sigmoid(D(xr)).mean().item(), torch.sigmoid(D(xf)).mean().item())
                if cb: extra[it] = cb(xf.numpy())
        xr = real(128 if xdim > 2 else 256); n = len(xr); xf = G(torch.randn(n, zdim))
        lD = bce(D(xr), torch.ones(n, 1)) + bce(D(xf.detach()), torch.zeros(n, 1)); oD.zero_grad(); lD.backward(); oD.step()
        lG = bce(D(xf), torch.ones(n, 1)); oG.zero_grad(); lG.backward(); oG.step()
    return G, snaps, dlog, extra

# A. 고리
def ring(n):
    a = torch.rand(n) * 2 * np.pi; r = 2 + 0.1 * torch.randn(n); return torch.stack([r * a.cos(), r * a.sin()], 1)
SNAP = [0, 300, 1000, 2000, 5000]
_, sA, dA, _ = train(ring, 2, 2, 5000, snap=SNAP)
print("ring D(real), D(fake):", {k: [round(x, 3) for x in v] for k, v in dA.items()}, flush=True)

# B. 봉우리 8개
C8 = torch.tensor([[2 * np.cos(2 * np.pi * k / 8), 2 * np.sin(2 * np.pi * k / 8)] for k in range(8)], dtype=torch.float32)
def eight(n): return C8[torch.randint(0, 8, (n,))] + 0.05 * torch.randn(n, 2)
def modes(x):
    d = np.linalg.norm(x[:, None, :] - C8.numpy()[None], axis=2); near = d.min(1) < 0.15; k = d.argmin(1)
    return int(sum((k[near] == m).sum() >= 20 for m in range(8))), float(near.mean())
snapB = list(range(0, 5001, 500)); B = {}
for seed in range(5):
    _, sB, _, eB = train(eight, 2, 2, 5000, seed=seed, snap=snapB, cb=modes)
    B[seed] = dict(modes=[eB[s][0] for s in snapB], quality=[eB[s][1] for s in snapB], final=sB[5000], snaps=np.stack([sB[s] for s in snapB]))
    print("eight seed", seed, B[seed]["modes"], flush=True)

# C. 숫자
d = load_digits(); Xd = torch.tensor(d.data / 16.0, dtype=torch.float32)
clf = LogisticRegression(max_iter=2000).fit(d.data / 16.0, d.target)
def digits_real(n): return Xd[torch.randint(0, len(Xd), (n,))]
def judge(x):
    p = clf.predict_proba(x); return np.bincount(p.argmax(1), minlength=10).tolist(), float((p.max(1) >= 0.9).mean())
snapC = [0, 1000, 3000, 10000]
GC, sC, dC, eC = train(digits_real, 16, 64, 10000, h=256, snap=snapC, out=nn.Sigmoid(), cb=judge)
print("digits:", {k: (v[0], round(v[1], 3)) for k, v in eC.items()}, {k: [round(x, 3) for x in v] for k, v in dC.items()}, flush=True)
np.savez_compressed(os.path.join(HERE, "assets", "gan.npz"), ring_snap=SNAP, ring_gen=np.stack([sA[s] for s in SNAP]), ring_real=ring(200).numpy(),
                    eight_c=C8.numpy(), eight_final=np.stack([B[s]["final"] for s in range(5)]), eight_snaps=np.stack([B[s]["snaps"] for s in range(5)]), eight_real=eight(400).numpy(),
                    digit_snap=snapC, digit_gen=np.stack([sC[s] for s in snapC]))
json.dump(dict(ring_d={str(k): v for k, v in dA.items()}, eight={str(k): {kk: vv for kk, vv in v.items() if kk not in ("final", "snaps")} for k, v in B.items()}, eight_steps=snapB,
               digits={str(k): v for k, v in eC.items()}, digits_d={str(k): v for k, v in dC.items()}), open(os.path.join(HERE, "assets", "gan.json"), "w"), indent=1)
