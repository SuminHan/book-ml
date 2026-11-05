"""작은 CNN vs 완전연결 (MNIST, 2 에폭, Adam 1e-3, 배치 64, 시드 0) → assets/cnn.json + 첫 층 필터 (assets/filters.npy).
CNN: conv3x3(1→8) ReLU pool2 · conv3x3(8→16) ReLU pool2 · FC(16·5·5 → 10).  MLP: 784 → 128 → 10."""
import os, json, numpy as np, torch, torch.nn as nn
from torchvision import datasets
HERE = os.path.dirname(os.path.abspath(__file__))
root = os.path.join(HERE, "..", "..", "week02", "sgd", "lab", "data")
tr = datasets.MNIST(root, train=True, download=True); te = datasets.MNIST(root, train=False, download=True)
Xtr = tr.data.float().unsqueeze(1) / 255; ytr = tr.targets; Xte = te.data.float().unsqueeze(1) / 255; yte = te.targets
def cnn(): return nn.Sequential(nn.Conv2d(1, 8, 3), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(8, 16, 3), nn.ReLU(), nn.MaxPool2d(2), nn.Flatten(), nn.Linear(16 * 5 * 5, 10))
def mlp(): return nn.Sequential(nn.Flatten(), nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))
out = {}
for name, make in (("cnn", cnn), ("mlp", mlp)):
    torch.manual_seed(0); net = make(); opt = torch.optim.Adam(net.parameters(), 1e-3)
    for ep in range(2):
        for b in torch.randperm(60000, generator=torch.Generator().manual_seed(ep)).split(64):
            opt.zero_grad(); nn.functional.cross_entropy(net(Xtr[b]), ytr[b]).backward(); opt.step()
    with torch.no_grad(): acc = (net(Xte).argmax(1) == yte).float().mean().item()
    out[name] = dict(params=sum(p.numel() for p in net.parameters()), acc=acc)
    if name == "cnn": np.save(os.path.join(HERE, "assets", "filters.npy"), net[0].weight.detach().numpy()[:, 0])
    print(name, out[name])
json.dump(out, open(os.path.join(HERE, "assets", "cnn.json"), "w"))
