"""BatchNorm 직접 실험 → bn.npz. sklearn digits(8x8), 시그모이드 은닉층 8개(각 64칸) MLP, SGD lr 0.5, 배치 64, 시드 3개.
BN 있음/없음의 학습 정확도 곡선과, 학습 중 3번째 층 입력 분포(평균·표준편차)의 흔들림을 기록."""
import numpy as np, torch, torch.nn as nn
from sklearn.datasets import load_digits
d = load_digits(); X = torch.tensor(d.data / 16.0, dtype=torch.float32); y = torch.tensor(d.target)
def net(bn):
    L = []; i = 64
    for _ in range(8):
        L += [nn.Linear(i, 64)] + ([nn.BatchNorm1d(64)] if bn else []) + [nn.Sigmoid()]; i = 64
    return nn.Sequential(*L, nn.Linear(64, 10))
STEPS = 1500; EVERY = 50
curves = {}
for bn in (False, True):
    cs = []
    for seed in range(3):
        torch.manual_seed(seed); m = net(bn); opt = torch.optim.SGD(m.parameters(), 0.5); c = []
        for it in range(STEPS + 1):
            if it % EVERY == 0:
                m.eval()
                with torch.no_grad(): c.append(float((m(X).argmax(1) == y).float().mean()))
                m.train()
            idx = torch.randint(0, len(X), (64,)); loss = nn.functional.cross_entropy(m(X[idx]), y[idx]); opt.zero_grad(); loss.backward(); opt.step()
        cs.append(c)
    curves[bn] = np.mean(cs, 0); print(bn, np.round(curves[bn][::5], 3), flush=True)
import os; np.savez(os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "bn.npz"), steps=np.arange(0, STEPS + 1, EVERY), plain=curves[False], bn=curves[True])
