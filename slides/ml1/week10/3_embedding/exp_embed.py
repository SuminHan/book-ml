"""고양이 vs 개만 배운 CNN 의 중간층 (임베딩) 에 처음 보는 사진이 어디에 찍히나 → assets/embed.npz
쇼츠 lessons/ml1/embedding/experiment/pilot_all.py 와 같은 모델 · 학습 (CIFAR-10 고양이 5,000 + 개 5,000장, 15바퀴, 좌우 뒤집기).
데이터: lessons/ml1/embedding/experiment/data/*.parquet (CIFAR-10 · CIFAR-100, git 에 올리지 않음).
저장: 시험 정확도, 마지막 은닉층 64칸 벡터 (시험 고양이 · 개 각 1,000장, CIFAR-100 호랑이 · 표범 · 사자 · 오렌지 · 사과 · 구름 각 100장), 작은 사진들."""
import os, io, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.expanduser("~/book-ml/lessons/ml1/embedding/experiment/data")
torch.manual_seed(0); np.random.seed(0)
dev = "mps" if torch.backends.mps.is_available() else "cpu"

def load(f, col):
    d = pd.read_parquet(f"{DATA}/{f}.parquet")
    X = np.stack([np.array(Image.open(io.BytesIO(b["bytes"])).convert("RGB")) for b in d["img"]])
    return X, d[col].values

Xtr8, ytr = load("c10_train", "label"); Xte8, yte = load("c10_test", "label"); X1008, y100 = load("c100_test", "fine_label")
T = lambda X: torch.tensor(X).permute(0, 3, 1, 2).float() / 255 * 2 - 1
CAT, DOG = 3, 5
k = (ytr == CAT) | (ytr == DOG); Xtr, ytr2 = T(Xtr8[k]), torch.tensor((ytr[k] == DOG).astype(int))
UNSEEN = {"tiger": 88, "leopard": 42, "lion": 43, "orange": 53, "apple": 0, "cloud": 23}

class Net(nn.Module):
    def __init__(s):
        super().__init__()
        s.b1 = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2))
        s.b2 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2))
        s.b3 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.AdaptiveAvgPool2d(1))
        s.fc1 = nn.Linear(128, 64); s.out = nn.Linear(64, 2)
    def forward(s, x, emb=False):
        p = F.relu(s.fc1(s.b3(s.b2(s.b1(x))).flatten(1)))
        return p if emb else s.out(p)

net = Net().to(dev); opt = torch.optim.Adam(net.parameters(), 1e-3)
for ep in range(15):
    net.train(); perm = torch.randperm(len(Xtr))
    for i in range(0, len(Xtr), 128):
        idx = perm[i:i + 128]; x = Xtr[idx].clone(); flip = torch.rand(len(x)) < 0.5; x[flip] = x[flip].flip(3)
        opt.zero_grad(); F.cross_entropy(net(x.to(dev)), ytr2[idx].to(dev)).backward(); opt.step()
net.eval()

@torch.no_grad()
def embed(X):
    return torch.cat([net(X[i:i + 256].to(dev), emb=True).cpu() for i in range(0, len(X), 256)]).numpy()

kt = (yte == CAT) | (yte == DOG); Xt = T(Xte8[kt])
with torch.no_grad():
    pred = torch.cat([net(Xt[i:i + 256].to(dev)).argmax(1).cpu() for i in range(0, len(Xt), 256)]).numpy()
acc = float((pred == (yte[kt] == DOG)).mean()); print("acc", acc)
out = dict(acc=acc, cat=embed(T(Xte8[yte == CAT])), dog=embed(T(Xte8[yte == DOG])), img_cat=Xte8[yte == CAT][:200], img_dog=Xte8[yte == DOG][:200])
for n, c in UNSEEN.items():
    out[n] = embed(T(X1008[y100 == c])); out["img_" + n] = X1008[y100 == c][:20]
np.savez_compressed(os.path.join(HERE, "assets", "embed.npz"), **out)
