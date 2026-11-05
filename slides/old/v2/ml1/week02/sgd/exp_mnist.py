"""배치 크기 실험 (MNIST, 은닉층 1개 64 ReLU, 시드 0) -> assets/mnist_batch.json
배치 크기마다 학습률을 격자에서 골라 1에폭(6만 장) 학습: 걸음 수 · 시험 정답률 · 이 컴퓨터 CPU 시간.
그리고 학습 전, 배치 크기별 미니배치 기울기 100개와 전체 기울기의 코사인 유사도.
쇼츠 lessons/ml1/sgd 와 같은 설정 (B=32 lr 0.3 -> 96.0%, 전체 배치 1걸음 lr 2 -> 53.5%)."""
import json, os, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torchvision import datasets
torch.set_num_threads(4)
HERE = os.path.dirname(os.path.abspath(__file__)); root = os.path.expanduser("~/Documents/manim_shorts/_data")
tr = datasets.MNIST(root, train=True, download=True); te = datasets.MNIST(root, train=False, download=True)
X = tr.data.float().div(255).view(-1, 784); y = tr.targets; Xt = te.data.float().div(255).view(-1, 784); yt = te.targets
N = len(y)
def net():
    torch.manual_seed(0); return nn.Sequential(nn.Linear(784, 64), nn.ReLU(), nn.Linear(64, 10))
def acc(m):
    with torch.no_grad(): return float((m(Xt).argmax(1) == yt).float().mean())
def run(bs, lr):
    m = net(); opt = torch.optim.SGD(m.parameters(), lr=lr); perm = torch.randperm(N, generator=torch.Generator().manual_seed(0))
    t = time.time()
    for i in range(0, N, bs):
        b = perm[i:i + bs]; opt.zero_grad(); F.cross_entropy(m(X[b]), y[b]).backward(); opt.step()
    return acc(m), time.time() - t
GRID = {1: [0.003, 0.01, 0.03, 0.1], 8: [0.01, 0.03, 0.1, 0.3], 32: [0.03, 0.1, 0.3, 0.5, 1], 128: [0.1, 0.3, 1, 2],
        1024: [0.3, 1, 2, 3], N: [0.1, 0.3, 1, 2, 3, 5]}
out = {"batch": []}
for bs, grid in GRID.items():
    rs = {lr: run(bs, lr) for lr in grid}
    lr = max(rs, key=lambda k: rs[k][0]); a, sec = rs[lr]
    out["batch"].append(dict(B=bs, steps=-(-N // bs), lr=lr, acc=a, sec=sec)); print(bs, lr, a, round(sec, 1), flush=True)
m = net(); m.zero_grad(); F.cross_entropy(m(X), y).backward(); G = torch.cat([p.grad.flatten() for p in m.parameters()]).clone()
g = torch.Generator().manual_seed(1); out["cos"] = {}
for bs in (1, 8, 32, 128, 1024):
    cs = []
    for _ in range(100):
        b = torch.randint(0, N, (bs,), generator=g); m.zero_grad(); F.cross_entropy(m(X[b]), y[b]).backward()
        cs.append(float(F.cosine_similarity(torch.cat([p.grad.flatten() for p in m.parameters()]), G, dim=0)))
    out["cos"][bs] = cs; print("cos", bs, np.mean(cs), np.min(cs), flush=True)
json.dump(out, open(os.path.join(HERE, "assets", "mnist_batch.json"), "w"), indent=1)
