"""평범한 깊은 신경망 vs 잔차 연결 → assets/resnet.json
MNIST 학습 1만 장 (시드 0), 블록 = Linear(64,64)-ReLU-Linear(64,64) (+ 입력 더하기) 다음 ReLU, 입력 784→64 · 출력 64→10.
블록 수 2 · 8 · 16 (층 4 · 16 · 32개), Adam lr 1e-3, 배치 64, 5 에폭. 에폭마다 학습 정확도 · 시험 정확도 (시험 1만 장)."""
import os, json, torch, torch.nn as nn
from torchvision import datasets
HERE = os.path.dirname(os.path.abspath(__file__))
root = os.path.join(HERE, "..", "..", "week02", "sgd", "lab", "data")
tr = datasets.MNIST(root, train=True, download=True); te = datasets.MNIST(root, train=False, download=True)
idx = torch.randperm(60000, generator=torch.Generator().manual_seed(0))[:10000]
Xtr = tr.data[idx].reshape(-1, 784).float() / 255; ytr = tr.targets[idx]; Xte = te.data.reshape(-1, 784).float() / 255; yte = te.targets

class Block(nn.Module):
    def __init__(self, res):
        super().__init__(); self.f = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 64)); self.res = res
    def forward(self, x):
        return torch.relu(self.f(x) + x) if self.res else torch.relu(self.f(x))

def net(blocks, res):
    return nn.Sequential(nn.Linear(784, 64), nn.ReLU(), *[Block(res) for _ in range(blocks)], nn.Linear(64, 10))

out = {}
for blocks in (2, 8, 16):
    for res in (False, True):
        torch.manual_seed(0); m = net(blocks, res); opt = torch.optim.Adam(m.parameters(), 1e-3); hist = []
        for ep in range(5):
            m.train()
            for b in torch.randperm(10000, generator=torch.Generator().manual_seed(ep)).split(64):
                opt.zero_grad(); nn.functional.cross_entropy(m(Xtr[b]), ytr[b]).backward(); opt.step()
            m.eval()
            with torch.no_grad():
                hist.append(((m(Xtr).argmax(1) == ytr).float().mean().item(), (m(Xte).argmax(1) == yte).float().mean().item()))
        out[f"{blocks}_{int(res)}"] = hist; print(blocks, res, [round(h[0], 3) for h in hist], round(hist[-1][1], 3), flush=True)
json.dump(out, open(os.path.join(HERE, "assets", "resnet.json"), "w"))
