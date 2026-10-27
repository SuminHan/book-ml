"""슬라이드·그림·실습이 공유하는 숫자.
논문 수치: Hinton, Srivastava, Krizhevsky, Sutskever, Salakhutdinov (2012) arXiv:1207.0580 — MNIST 시험 1만 장 오답 160 → 130 (은닉 드롭아웃) → 110 (입력 20% 도).
우리 실험: exp_dropout.py → assets/dropout.json (MNIST 학습 2,000장, MLP 784-512-512-10, Adam, 100 에폭)."""
import os, json
J = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "dropout.json")))
PAPER = (160, 130, 110)
ERRORS = {p: round((1 - J[p][-1][1]) * 10000) for p in J}       # 100 에폭 뒤 시험 오답 수 (1만 장 중)
TRAIN_END = {p: J[p][-1][0] for p in J}
SUBNETS = 2 ** 12                                                # 뉴런 12개 켜고 끄는 조합
HAND = dict(h=(2, 4, 6, 8), mask=(1, 0, 1, 0), train=(4, 0, 12, 0), test=(2, 4, 6, 8))   # p = 0.5, 학습 때 ×2

if __name__ == "__main__":
    assert ERRORS == {"0.0": 770, "0.2": 663, "0.5": 726, "0.8": 680}, ERRORS
    assert SUBNETS == 4096 and all(m * h * 2 == t for h, m, t in zip(HAND["h"], HAND["mask"], HAND["train"]))
    print("data ok", {p: round(v, 4) for p, v in TRAIN_END.items()})
