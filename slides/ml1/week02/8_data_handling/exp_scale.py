"""정규화 실험 → assets/scale.json
와인 데이터(UCI Wine, sklearn load_wine): 178병, 성분 13가지, 품종 3개. 학습 70% · 시험 30% (층화, 시드별).
PyTorch MLP 13-16-3 (tanh), 기본 초기화 그대로, Adam lr 0.01, 전체 배치 300번. 입력: 그대로 · Min-Max · Z-score (둘 다 학습 데이터로만 계산).
시드 0~9 평균 시험 정확도. + 첫 층 기본 초기 가중치 범위(±1/√13) · 처음 은닉층 tanh 가 포화(|값|>0.99)된 비율."""
import json, os, numpy as np, torch
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
X, y = load_wine(return_X_y=True)
names = load_wine().feature_names
res = {m: [] for m in ("raw", "minmax", "z")}; sat = {m: [] for m in res}; curves = {m: [] for m in res}
for s in range(10):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=s)
    for m in res:
        if m == "raw": A, B = Xtr, Xte
        elif m == "minmax":
            lo, hi = Xtr.min(0), Xtr.max(0); A, B = (Xtr - lo) / (hi - lo), (Xte - lo) / (hi - lo)
        else:
            mu, sd = Xtr.mean(0), Xtr.std(0); A, B = (Xtr - mu) / sd, (Xte - mu) / sd
        torch.manual_seed(s)
        net = torch.nn.Sequential(torch.nn.Linear(13, 16), torch.nn.Tanh(), torch.nn.Linear(16, 3))
        A, B = torch.tensor(A, dtype=torch.float32), torch.tensor(B, dtype=torch.float32)
        ya, yb = torch.tensor(ytr), torch.tensor(yte)
        with torch.no_grad(): sat[m].append(float((net[1](net[0](A)).abs() > 0.99).float().mean()))
        opt = torch.optim.Adam(net.parameters(), lr=0.01); c = []
        for t in range(300):
            opt.zero_grad(); loss = torch.nn.functional.cross_entropy(net(A), ya); loss.backward(); opt.step(); c.append(loss.item())
        curves[m].append(c)
        with torch.no_grad(): res[m].append(float((net(B).argmax(1) == yb).float().mean()))
acc = {m: float(np.mean(v)) for m, v in res.items()}; sat = {m: float(np.mean(v)) for m, v in sat.items()}
curve = {m: np.mean(v, 0)[::10].tolist() for m, v in curves.items()}
rng = {n: [float(X[:, i].min()), float(X[:, i].max())] for i, n in enumerate(names)}
torch.manual_seed(0); lin = torch.nn.Linear(13, 16); W0 = lin.weight.detach().numpy().ravel(); bound = float(abs(W0).max())
emb = torch.nn.Embedding(1000, 64).weight.detach()
# 뉴런 하나 들여다보기: 와인 1번 병을 seed 0 nn.Linear(13, 16) 의 4번 뉴런에 — 그대로 vs Z-score(전체 178병 평균 · 표준편차, 그림용)
x0 = X[0]; z0 = (x0 - X.mean(0)) / X.std(0)
wk, bk = lin.weight.detach().numpy()[4], float(lin.bias.detach()[4])
idx = np.linspace(0, 177, 12).astype(int); Zall = (X - X.mean(0)) / X.std(0)
neuron = dict(x=x0.tolist(), z=z0.tolist(), w=wk.tolist(), b=bk, raw=float(wk @ x0 + bk), zs=float(wk @ z0 + bk), many_raw=(X[idx] @ wk + bk).tolist(), many_z=(Zall[idx] @ wk + bk).tolist())
J = dict(neuron=neuron, names=names, w0=W0.tolist(), proline=X[:, 12].tolist(), hue=X[:, 10].tolist(), acc=acc, sat=sat, curve=curve, ranges=rng, bound=bound, theory=1 / np.sqrt(13), emb_std=float(emb.std()), emb_mean=float(emb.mean()))
json.dump(J, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "scale.json"), "w"), indent=1)
print(acc, sat, bound, J["theory"], J["emb_std"]); print({k: v for k, v in rng.items() if k in ("proline", "hue", "magnesium", "nonflavanoid_phenols")})
