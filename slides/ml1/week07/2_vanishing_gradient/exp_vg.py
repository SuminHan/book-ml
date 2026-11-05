"""기울기 소실 실험: MNIST, 은닉층 10개(폭 64) 신경망, 시그모이드 vs ReLU (같은 구조, 각자 표준 초기화: 시그모이드 = Xavier, ReLU = He).
① 학습 전 층별 기울기 크기(가중치 기울기 평균 절댓값) ② SGD(lr 0.05, 배치 64) 3 에폭 후 시험 정답률. 시드 0 → assets/vg.json"""
import json, os, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torchvision import datasets
HERE = os.path.dirname(os.path.abspath(__file__)); root = os.path.expanduser("~/Documents/manim_shorts/_data")
tr = datasets.MNIST(root, train=True, download=True); te = datasets.MNIST(root, train=False, download=True)
X = tr.data.float().div(255).view(-1, 784); y = tr.targets; Xt = te.data.float().div(255).view(-1, 784); yt = te.targets
def net(act):
    L = [nn.Linear(784, 64), act()]
    for _ in range(9): L += [nn.Linear(64, 64), act()]
    m = nn.Sequential(*L, nn.Linear(64, 10))
    for l in m:
        if isinstance(l, nn.Linear):
            (nn.init.kaiming_normal_(l.weight, nonlinearity="relu") if act is nn.ReLU else nn.init.xavier_normal_(l.weight)); nn.init.zeros_(l.bias)
    return m
out = {}
for name, act in (("sigmoid", nn.Sigmoid), ("relu", nn.ReLU)):
    torch.manual_seed(0); m = net(act)
    F.cross_entropy(m(X[:512]), y[:512]).backward()
    g = [float(l.weight.grad.abs().mean()) for l in m if isinstance(l, nn.Linear)][:10]   # 은닉층 10개 (입구 → 출구)
    m.zero_grad(); opt = torch.optim.SGD(m.parameters(), lr=0.05); acc = []
    for ep in range(3):
        perm = torch.randperm(len(y))
        for i in range(0, len(y), 64):
            b = perm[i:i + 64]; opt.zero_grad(); F.cross_entropy(m(X[b]), y[b]).backward(); opt.step()
        with torch.no_grad(): acc.append(float((m(Xt).argmax(1) == yt).float().mean()))
    out[name] = dict(grad=g, acc=acc); print(name, ["%.1e" % v for v in g], acc, flush=True)
out["ratio"] = {k: out[k]["grad"][-1] / out[k]["grad"][0] for k in ("sigmoid", "relu")}
print(out["ratio"]); json.dump(out, open(os.path.join(HERE, "assets", "vg.json"), "w"), indent=1)
