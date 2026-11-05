"""드롭아웃 실험 → assets/dropout.json
MNIST 학습 2,000장 (시드 0 무작위) · 시험 1만 장. MLP 784-512-512-10 (ReLU), Adam lr 1e-3, 배치 64, 100 에폭.
드롭아웃 p = 0, 0.2, 0.5, 0.8 (은닉층 두 곳). 에폭마다 학습 · 시험 정확도 (시험은 model.eval())."""
import os, json, torch, torch.nn as nn
from torchvision import datasets
HERE = os.path.dirname(os.path.abspath(__file__))
root = os.path.join(HERE, "..", "..", "week02", "sgd", "lab", "data")
tr = datasets.MNIST(root, train=True, download=True); te = datasets.MNIST(root, train=False, download=True)
g = torch.Generator().manual_seed(0); idx = torch.randperm(60000, generator=g)[:2000]
Xtr = tr.data[idx].reshape(-1, 784).float() / 255; ytr = tr.targets[idx]
Xte = te.data.reshape(-1, 784).float() / 255; yte = te.targets
out = {}
for p in (0.0, 0.2, 0.5, 0.8):
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(784, 512), nn.ReLU(), nn.Dropout(p), nn.Linear(512, 512), nn.ReLU(), nn.Dropout(p), nn.Linear(512, 10))
    opt = torch.optim.Adam(net.parameters(), 1e-3); hist = []
    for ep in range(100):
        net.train()
        for b in torch.randperm(2000, generator=torch.Generator().manual_seed(ep)).split(64):
            opt.zero_grad(); nn.functional.cross_entropy(net(Xtr[b]), ytr[b]).backward(); opt.step()
        net.eval()
        with torch.no_grad():
            hist.append(((net(Xtr).argmax(1) == ytr).float().mean().item(), (net(Xte).argmax(1) == yte).float().mean().item()))
    out[str(p)] = hist; print(p, hist[-1], max(h[1] for h in hist))
json.dump(out, open(os.path.join(HERE, "assets", "dropout.json"), "w"))
