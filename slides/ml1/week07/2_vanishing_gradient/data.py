"""슬라이드·그림·실습이 공유하는 숫자 (쇼츠 lessons/ml1/vanishing_gradient 와 같은 실험, exp_vg.py → assets/vg.json).
MNIST, 은닉층 10개 (폭 64): 시그모이드 (Xavier 초기화) vs ReLU (He 초기화). 층별 기울기 = 학습 전 가중치 기울기 평균 절댓값 (배치 512).
정답률 = SGD lr 0.05, 배치 64, 3 에폭 뒤 시험 10,000장.
초기화 장면: 입력 1,000개 × 256차원 표준정규, ReLU 10층 (폭 256), 가중치 표준편차만 바꿔 층별 출력 표준편차 (시드 0)."""
import os, json
import numpy as np

J = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "vg.json")))
GS, GR = J["sigmoid"]["grad"], J["relu"]["grad"]               # 1층 (입구) → 10층 (출구)
ACC_S, ACC_R = J["sigmoid"]["acc"], J["relu"]["acc"]             # 에폭 1 · 2 · 3


def init_stds(scale, n=256, layers=10, seed=0):
    r = np.random.default_rng(seed); h = r.normal(0, 1, (1000, n)); out = []
    for _ in range(layers):
        h = np.maximum(0, h @ r.normal(0, scale, (n, n))); out.append(float(h.std()))
    return out


RATIO_MAN = 119                                                 # 출구 / 입구 기울기 = 119만 배
ACC_END = (10, 95)
QUARTER10 = 9.5e-7                                              # 0.25^10
INIT = {"0.01": (0.094, 3.04e-10), "1": (9.4, 3.04e10), "He": (0.831, 0.883), "Xavier": (0.588, 0.0276)}   # (1층, 10층) 출력 표준편차
N = 256

if __name__ == "__main__":
    assert round(GS[-1] / GS[0] / 1e4) == RATIO_MAN and (round(ACC_S[-1] * 100), round(ACC_R[-1] * 100)) == ACC_END
    assert abs(0.25 ** 10 - QUARTER10) < 1e-8 and 0.5 < GR[-1] / GR[0] < 2
    for k, s in (("0.01", 0.01), ("1", 1.0), ("He", np.sqrt(2 / N)), ("Xavier", np.sqrt(1 / N))):
        v = init_stds(s); assert (float(f"{v[0]:.3g}"), float(f"{v[-1]:.3g}")) == INIT[k], (k, v[0], v[-1])
    print("data ok")
