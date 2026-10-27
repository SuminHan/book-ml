"""슬라이드·그림·실습이 공유하는 숫자 (쇼츠 lessons/ml1/momentum_adam 과 같은 계산, exp_paths.py → assets/paths.npz).
골짜기: f(x, y) = (x² + 50y²)/2, 시작 (−4, 1.5), 도착 = 최솟값에서 0.05 이내. 학습률은 방법마다 격자에서 가장 빨리 도착하는 값.
실제 데이터: exp_digits.py → assets/opt_digits.json (digits, MLP, 30 에폭).
논문: Sutskever, Martens, Dahl, Hinton (2013) ICML · Kingma & Ba (2014) arXiv:1412.6980."""
import os, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D = np.load(os.path.join(HERE, "assets", "paths.npz"))
GD, MOM, ADAM = D["gd"], D["mom"], D["adam"]
STEPS = dict(zip(("gd", "mom", "adam"), [int(a) for a in D["arrive"]]))
LR = dict(zip(("gd", "mom", "adam"), [round(float(l), 3) for l in D["lr"]]))
DIG = json.load(open(os.path.join(HERE, "assets", "opt_digits.json")))
DIG10 = {"sgd": 0.817, "momentum": 0.066, "adam": 0.186}           # 10 에폭 뒤 학습 손실
DEFAULTS = (0.001, 0.9, 0.999)
V_HAND = [1.0, 1.9, 2.71, 3.44]                                     # 기울기가 계속 1 이면 v (β = 0.9)


def f(p):
    return (p[0] ** 2 + 50 * p[1] ** 2) / 2


if __name__ == "__main__":
    assert STEPS == {"gd": 111, "mom": 33, "adam": 27} and LR == {"gd": 0.039, "mom": 0.007, "adam": 0.18}
    assert np.allclose(GD[0], [-4, 1.5]) and {k: round(v[9], 3) for k, v in DIG.items()} == DIG10
    v, vs = 0.0, []
    for _ in range(4): v = 0.9 * v + 1; vs.append(round(v, 2))
    assert vs == V_HAND
    print("data ok")
