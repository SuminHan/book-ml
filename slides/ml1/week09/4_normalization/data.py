"""슬라이드·그림·실습이 공유하는 숫자 (쇼츠 lessons/ml1/batchnorm 과 같은 실험, exp_bn.py → assets/bn.npz).
scikit-learn digits (8×8, 1,797장), 시그모이드 은닉층 8개 (각 64칸), SGD lr 0.5, 배치 64, 시드 3개 평균, 학습 데이터 정확도.
논문: Ioffe & Szegedy (2015) arXiv:1502.03167 — 같은 정확도를 14배 적은 학습으로.
손 계산: 뉴런 하나가 배치 4개에서 받은 값 2, 4, 6, 8 · γ = 2, β = 1."""
import os
import numpy as np

D = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "bn.npz"))
STEPS, PLAIN, BN = D["steps"], D["plain"], D["bn"]
PLAIN_END, BN_250, FASTER = 10, 96, 14
HAND_X = [2.0, 4.0, 6.0, 8.0]
HAND = dict(mu=5.0, var=5.0, xhat=(-1.34, -0.45, 0.45, 1.34), y=(-1.68, 0.11, 1.89, 3.68))
GAMMA, BETA = 2.0, 1.0
SPREAD_PLAIN = ["1.5e-01", "2.1e-02", "3.0e-03", "4.4e-04", "5.9e-05", "8.3e-06", "1.2e-06", "1.8e-07"]   # 학습 전, 그림 512장 사이 층 입력 퍼짐 (시드 0)

if __name__ == "__main__":
    i250 = list(STEPS).index(250)
    assert STEPS[-1] == 1500 and round(float(PLAIN[-1]) * 100) == PLAIN_END and round(float(BN[i250]) * 100) == BN_250
    assert float(PLAIN.max()) < 0.11
    x = np.array(HAND_X); mu, var = x.mean(), x.var(); xh = (x - mu) / np.sqrt(var + 1e-5)
    assert (mu, var) == (HAND["mu"], HAND["var"]) and tuple(np.round(xh, 2)) == HAND["xhat"]
    assert tuple(np.round(GAMMA * xh + BETA, 2)) == HAND["y"], np.round(GAMMA * xh + BETA, 2)
    import torch, torch.nn as nn
    from sklearn.datasets import load_digits
    X = torch.tensor(load_digits().data[:512] / 16.0, dtype=torch.float32); torch.manual_seed(0); h = X; out = []
    for _ in range(8):
        z = nn.Linear(h.shape[1], 64)(h); out.append("%.1e" % float(z.detach().std(0).mean())); h = torch.sigmoid(z)
    assert out == SPREAD_PLAIN, out
    print("data ok")
