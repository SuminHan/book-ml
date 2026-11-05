"""실제 데이터 비교 → assets/opt_digits.json. scikit-learn digits 1,797장, MLP 64-64-64-10 (ReLU), 배치 64, 30 에폭, 시드 0.
SGD lr 0.05 · 모멘텀 (lr 0.05, β 0.9) · Adam (lr 0.001, 기본값). 에폭마다 전체 학습 손실 (교차 엔트로피)."""
import os, json, torch, torch.nn as nn
from sklearn.datasets import load_digits
d = load_digits(); X = torch.tensor(d.data / 16., dtype=torch.float32); y = torch.tensor(d.target); out = {}
for name in ("sgd", "momentum", "adam"):
    torch.manual_seed(0); net = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 10))
    opt = {"sgd": lambda p: torch.optim.SGD(p, 0.05), "momentum": lambda p: torch.optim.SGD(p, 0.05, momentum=0.9),
           "adam": lambda p: torch.optim.Adam(p, 1e-3)}[name](net.parameters())
    g = torch.Generator().manual_seed(0); L = []
    for ep in range(30):
        for b in torch.randperm(len(y), generator=g).split(64):
            opt.zero_grad(); nn.functional.cross_entropy(net(X[b]), y[b]).backward(); opt.step()
        with torch.no_grad(): L.append(nn.functional.cross_entropy(net(X), y).item())
    out[name] = L
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "opt_digits.json"), "w"))
print({k: round(v[9], 3) for k, v in out.items()})
