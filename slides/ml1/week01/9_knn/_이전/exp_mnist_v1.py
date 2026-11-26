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
