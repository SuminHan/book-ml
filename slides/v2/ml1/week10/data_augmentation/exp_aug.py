"""데이터 증강 실험 (MNIST 28x28, 작은 CNN): 숫자마다 10장(100장)만 학습, 증강 없음 vs 증강(매 에폭 무작위 ±10도 회전 · ±2칸 이동).
학습 세트 5개(시드 0~4) 평균, 시험 = MNIST test 10,000장 → assets/aug.json"""
import json, os, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torchvision import datasets, transforms as T
HERE = os.path.dirname(os.path.abspath(__file__))
root = os.path.join(HERE, "..", "..", "week02", "sgd", "lab", "data")
tr_ds = datasets.MNIST(root, train=True, download=True); te_ds = datasets.MNIST(root, train=False, download=True)
Xte = te_ds.data.float().div(255).unsqueeze(1); yte = te_ds.targets
dev = "cpu"
AUG = T.RandomAffine(degrees=10, translate=(2 / 28, 2 / 28))
def net():
    return nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                         nn.Flatten(), nn.Linear(32 * 49, 64), nn.ReLU(), nn.Linear(64, 10))
def run(seed, aug, epochs=400):
    g = np.random.RandomState(seed); torch.manual_seed(seed)
    idx = np.concatenate([g.choice(np.where(tr_ds.targets.numpy() == c)[0], 10, replace=False) for c in range(10)])
    X = tr_ds.data[idx].float().div(255).unsqueeze(1); y = tr_ds.targets[idx]
    m = net(); opt = torch.optim.Adam(m.parameters(), 1e-3)
    for _ in range(epochs):
        xb = torch.stack([AUG(x) for x in X]) if aug else X
        opt.zero_grad(); F.cross_entropy(m(xb), y).backward(); opt.step()
    with torch.no_grad(): return (m(Xte).argmax(1) == yte).float().mean().item()
R = {"base": [run(s, False) for s in range(5)], "aug": [run(s, True) for s in range(5)]}
out = dict(n_train=100, n_test=len(yte), base=float(np.mean(R["base"])), aug=float(np.mean(R["aug"])), base_all=R["base"], aug_all=R["aug"])
print(out)
json.dump(out, open(os.path.join(HERE, "assets", "aug.json"), "w"), indent=1)
