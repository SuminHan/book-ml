"""슬라이드·그림·실습이 공유하는 숫자 (쇼츠 lessons/ml1/cnn 과 같은 계산) + exp_cnn.py (assets/cnn.json, 작은 CNN vs 완전연결, MNIST 2 에폭)
- 그림: MNIST 테스트 첫 장(손글씨 7, 28×28) → assets/mnist_test0.npy (밝기 0~255 → 0~1)
- 필터: 사람이 고른 예시 필터(학습된 값 아님). 가로선 = 위 +1 · 가운데 0 · 아래 −1, 세로선 = 그 전치
- 합성곱: 패딩 0, 스트라이드 1 → 26×26, 음수는 0(ReLU), 2×2 최대 풀링 → 13×13
- 가중치 수: week10 p14 미니 LeNet(28×28 → conv 3층 10×10×16) vs 같은 출력의 완전연결층 (p15)"""
import os
import json
import numpy as np
IMG = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "mnist_test0.npy")).astype(float) / 255
KH = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], float)
KV = KH.T.copy()

def conv(im, k):
    n = im.shape[0] - k.shape[0] + 1
    return np.array([[(im[i:i + 3, j:j + 3] * k).sum() for j in range(n)] for i in range(n)])

def pool(m):
    n = m.shape[0] // 2
    return m[:2 * n, :2 * n].reshape(n, 2, n, 2).max(axis=(1, 3))

MH, MV = np.maximum(conv(IMG, KH), 0), np.maximum(conv(IMG, KV), 0)
PH = pool(MH)

conv_p = lambda k, ci, co: (k * k * ci + 1) * co
LAYERS = [conv_p(3, 1, 4), conv_p(3, 4, 8), conv_p(3, 8, 16)]
P_CNN = sum(LAYERS)
P_FC = 784 * 1600 + 1600
RATIO = P_FC / P_CNN

assert IMG.shape == (28, 28) and MH.shape == (26, 26) and PH.shape == (13, 13)
assert LAYERS == [40, 296, 1168] and P_CNN == 1504
assert P_FC == 1_256_000 and round(RATIO) == 835
assert KH.size == 9
PV = pool(MV)
I0, J0 = 8, 7                                          # s4 에서 손으로 보여 줄 위치 (가로획 윗변, 최댓값)
V0 = float((IMG[I0:I0 + 3, J0:J0 + 3] * KH).sum())
assert round(V0, 1) == 3.0 and MH[I0, J0] == V0 == MH.max()
assert np.allclose(IMG[I0, J0:J0 + 3], 1, atol=0.01) and np.allclose(IMG[I0 + 2, J0:J0 + 3], 0)

J = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "cnn.json")))
CMP = {k: (v["params"], round(v["acc"] * 100, 1)) for k, v in J.items()}
assert CMP == {"cnn": (5258, 97.5), "mlp": (101770, 96.4)}, CMP
OUT_SIZE = {(28, 3, 0, 1): 26, (28, 3, 1, 1): 28, (28, 3, 1, 2): 14}   # (n, k, p, s): 출력 크기
for (n, k, p, s), o in OUT_SIZE.items():
    assert (n - k + 2 * p) // s + 1 == o
