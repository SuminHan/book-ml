"""kNN 손글씨 숫자 실험 (MNIST: 학습 60,000장 · 시험 10,000장, 28×28 = 784 픽셀, 0~1 로 나눔) → assets/mnist.json, assets/mnist_examples.npz
유클리드 거리 전체 비교 (근사 없음). k = 1·3·5·7 시험 정확도, 장당 예측 시간, 예시 질의 3장 (맞힘 2 · 틀림 1) 과 가까운 5장."""
import json, os, time, numpy as np
from collections import Counter
HERE = os.path.dirname(os.path.abspath(__file__)); RAW = os.path.expanduser("~/Documents/manim_shorts/_data/MNIST/raw")
def load(img, lab):
    X = np.fromfile(f"{RAW}/{img}", np.uint8, offset=16).reshape(-1, 784).astype(np.float32) / 255
    return X, np.fromfile(f"{RAW}/{lab}", np.uint8, offset=8).astype(int)
X, y = load("train-images-idx3-ubyte", "train-labels-idx1-ubyte"); Xt, yt = load("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte")
K = 7; sq = (X ** 2).sum(1); NB, ND = np.zeros((len(Xt), K), int), np.zeros((len(Xt), K), np.float32)
t0 = time.time()
for i in range(0, len(Xt), 500):
    D = sq[None] - 2 * Xt[i:i + 500] @ X.T                                  # ‖q‖² 는 순서에 영향 없음
    idx = np.argpartition(D, K, 1)[:, :K]; d = np.take_along_axis(D, idx, 1); o = np.argsort(d, 1)
    NB[i:i + 500] = np.take_along_axis(idx, o, 1)
    ND[i:i + 500] = np.sqrt(np.maximum(np.take_along_axis(d, o, 1) + (Xt[i:i + 500] ** 2).sum(1, keepdims=True), 0))
sec = time.time() - t0
def vote(k):
    return np.array([Counter(r).most_common(1)[0][0] for r in y[NB[:, :k]]])     # 동점이면 더 가까운 쪽이 먼저 나온 라벨
acc = {k: float((vote(k) == yt).mean()) for k in (1, 3, 5, 7)}
p3 = vote(3)
ok = [i for i in range(len(yt)) if p3[i] == yt[i] and (y[NB[i, :5]] == yt[i]).all()]
bad = [i for i in range(len(yt)) if p3[i] != yt[i]]
Q = [next(i for i in ok if yt[i] == 3), next(i for i in ok if yt[i] == 7), next(i for i in bad if yt[i] == 4)]
np.savez_compressed(f"{HERE}/assets/mnist_examples.npz", q=Xt[Q].reshape(-1, 28, 28), ql=yt[Q], p=p3[Q],
                    nb=X[NB[Q, :5]].reshape(3, 5, 28, 28), nl=y[NB[Q, :5]], nd=ND[Q, :5])
json.dump(dict(acc=acc, ms_per_query=sec / len(Xt) * 1000, n_train=len(y), n_test=len(yt), wrong_k3=len(bad), Q=[int(q) for q in Q]),
          open(f"{HERE}/assets/mnist.json", "w"), indent=1)
print({k: round(v, 4) for k, v in acc.items()}, f"{sec:.1f}s", "틀림", len(bad), "Q", Q, "nl", y[NB[Q, :5]].tolist(), ND[Q, :5].round(2).tolist())

# --- 1-NN 한 장 (릴스형 그림): 시험 0번 글씨, 가장 가까운 학습 그림, 숫자별 학습 그림 40장씩 (가장 가까운 장 포함)
q0 = 0; n0 = int(NB[q0, 0]); G = 40
grid = []
for d in range(10):
    ids = [i for i in np.where(y == d)[0][:G]]
    if d == y[n0]: ids = ids[:G // 2] + [n0] + ids[G // 2:G - 1]       # 가장 가까운 장을 줄 가운데에
    grid.append(ids)
grid = np.array(grid)

# --- 비교용 신경망: 은닉층 64 (ReLU), 미니배치 32, 학습률 0.3, 시드 0 (week07 SGD 덱과 같은 설정). 1에폭 · 10에폭
import torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0); torch.set_num_threads(4)
Xtr, ytr, Xte = torch.tensor(X), torch.tensor(y), torch.tensor(Xt)
net = nn.Sequential(nn.Linear(784, 64), nn.ReLU(), nn.Linear(64, 10)); opt = torch.optim.SGD(net.parameters(), lr=0.3)
nn_acc, nn_steps = {}, 0
for ep in range(1, 11):
    perm = torch.randperm(len(ytr), generator=torch.Generator().manual_seed(ep))
    for i in range(0, len(ytr), 32):
        b = perm[i:i + 32]; opt.zero_grad(); F.cross_entropy(net(Xtr[b]), ytr[b]).backward(); opt.step(); nn_steps += 1
    if ep in (1, 10):
        with torch.no_grad(): nn_acc[ep] = dict(acc=float((net(Xte).argmax(1).numpy() == yt).mean()), steps=nn_steps)
np.savez_compressed(f"{HERE}/assets/mnist_one.npz", q=Xt[q0].reshape(28, 28), ql=yt[q0], nb=X[n0].reshape(28, 28), nl=y[n0], nd=ND[q0, 0],
                    grid=X[grid].reshape(10, G, 28, 28), pos=np.array([int(y[n0]), G // 2]))
J = json.load(open(f"{HERE}/assets/mnist.json")); J.update(nn=nn_acc, one_dist=float(ND[q0, 0]), one_label=int(y[n0]),
                                                       right_k1=int((vote(1) == yt).sum()), right_k3=int((vote(3) == yt).sum()))
json.dump(J, open(f"{HERE}/assets/mnist.json", "w"), indent=1)
print("1-NN", yt[q0], "->", y[n0], round(float(ND[q0, 0]), 2), "NN", nn_acc, "right", J["right_k1"], J["right_k3"])
