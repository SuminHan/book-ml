"""슬라이드·그림·실습이 공유하는 숫자 (쇼츠 lessons/ml1/xor_backprop 과 같은 XOR 2-2-1 신경망 학습).
손실 L = ½ Σ (y − t)², 활성 함수 시그모이드, 학습률 2.0, 4,000번 (전체 4개 한꺼번에), 초기 가중치 시드 3.
손 계산 예: 입력 (1, 1), 정답 0, W1 = [[1, −1], [1, −1]], b1 = 0, W2 = (1, 1), b2 = 0."""
import numpy as np

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float)
T = np.array([0, 1, 1, 0], float)
SNAPS = (0, 150, 400, 800, 1500, 4000)


def sig(z):
    return 1 / (1 + np.exp(-z))


def train(X, T, seed=3, lr=2.0, epochs=4000, snaps=SNAPS):
    r = np.random.default_rng(seed)
    W1, b1 = r.normal(0, 1, (2, 2)), np.zeros(2); W2, b2 = r.normal(0, 1, 2), 0.0
    out, losses = {}, []
    for e in range(epochs + 1):
        h = sig(X @ W1 + b1); y = sig(h @ W2 + b2)
        losses.append(0.5 * float(((y - T) ** 2).sum()))
        if e in snaps:
            out[e] = (W1.copy(), b1.copy(), W2.copy(), float(b2), float(((y > 0.5) == T).mean()), y.copy())
        d2 = (y - T) * y * (1 - y)                       # 출력층 오차 신호
        d1 = np.outer(d2, W2) * h * (1 - h)              # 숨은 층으로 거꾸로
        W2 -= lr * h.T @ d2; b2 -= lr * d2.sum()
        W1 -= lr * X.T @ d1; b1 -= lr * d1.sum(0)
    return out, np.array(losses)


def predict(params, pts):
    W1, b1, W2, b2 = params[:4]
    return sig(sig(pts @ W1 + b1) @ W2 + b2)


# 손으로 고른 해: h1 = OR, h2 = NAND, y = AND(h1, h2)
HAND = dict(W1=np.array([[1, -1], [1, -1]], float), b1=np.array([-0.5, 1.5]), W2=np.array([1, 1], float), b2=-1.5)
ACC = {0: 0.5, 150: 0.75, 400: 1.0, 4000: 1.0}
LOSS = {0: 0.519, 150: 0.255, 400: 0.008, 4000: 0.0003}
Y_END = [0.012, 0.986, 0.986, 0.013]
STEP = dict(z1=(2.0, -2.0), h=(0.881, 0.119), y=0.731, L=0.267, d2=0.144, gW2=(0.127, 0.017), d1=0.015)   # 손 계산 예

if __name__ == "__main__":
    out, losses = train(X, T)
    assert {e: out[e][4] for e in ACC} == ACC
    assert {e: round(float(losses[e]), 4 if e == 4000 else 3) for e in LOSS} == LOSS
    assert list(np.round(out[4000][5], 3)) == Y_END
    step = lambda z: (np.asarray(z) > 0).astype(float)
    hh = step(X @ HAND["W1"] + HAND["b1"]); assert (step(hh @ HAND["W2"] + HAND["b2"]) == T).all()
    x, t = np.array([1.0, 1.0]), 0.0; W1 = np.array([[1, -1], [1, -1]], float); W2 = np.array([1, 1], float)
    z1 = x @ W1; h = sig(z1); y = sig(h @ W2); L = 0.5 * (y - t) ** 2
    d2 = (y - t) * y * (1 - y); gW2 = d2 * h; d1 = d2 * W2 * h * (1 - h)
    got = dict(z1=tuple(z1), h=tuple(np.round(h, 3)), y=round(float(y), 3), L=round(float(L), 3), d2=round(float(d2), 3),
               gW2=tuple(np.round(gW2, 3)), d1=round(float(d1[0]), 3))
    assert got == STEP and abs(d1[0] - d1[1]) < 1e-12, got
    print("data ok")
