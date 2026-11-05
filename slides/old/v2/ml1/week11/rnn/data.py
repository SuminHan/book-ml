"""슬라이드·그림·실습이 공유하는 숫자.
순서 예: 쇼츠 lessons/ml1/rnn 과 같은 작은 RNN (은닉 4칸, 무작위 가중치 시드 11, 학습 안 함) — Dog bites man vs Man bites dog.
손 계산: 1차원 RNN h_t = tanh(w_x x_t + w_h h_{t-1}), w_x = 1, w_h = 0.5, 입력 1, 0, 1.
언어 모델: exp_charrnn.py → assets/charrnn.json (tiny Shakespeare, 글자 65종, 은닉 128, 2,000걸음)."""
import os, json
import numpy as np

rng = np.random.default_rng(11)
D = 4
EMB = {w: rng.normal(0, 1, D) for w in ("dog", "bites", "man")}
WX, WH = rng.normal(0, 0.8, (D, D)), rng.normal(0, 0.8, (D, D))
S1, S2 = ["dog", "bites", "man"], ["man", "bites", "dog"]


def run(s):
    h, hs = np.zeros(D), []
    for w in s:
        h = np.tanh(WX @ EMB[w] + WH @ h); hs.append(h.copy())
    return hs


H1, H2 = run(S1), run(S2)
HAND_X, HAND_H = [1, 0, 1], [0.762, 0.363, 0.828]
J = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "charrnn.json")))
LOSS0, LOSS_END = 4.18, 1.58                                     # 첫 걸음, 마지막 50걸음 평균 (nats)
UNIFORM = 4.17                                                   # ln 65: 아무렇게나 찍을 때

if __name__ == "__main__":
    assert np.allclose(sum(EMB[w] for w in S1), sum(EMB[w] for w in S2)) and np.abs(H1[-1] - H2[-1]).max() > 0.5
    h, hs = 0.0, []
    for x in HAND_X: h = np.tanh(1 * x + 0.5 * h); hs.append(round(float(h), 3))
    assert hs == HAND_H, hs
    assert J["V"] == 65 and round(J["losses"][0], 2) == LOSS0 and round(sum(J["losses"][-50:]) / 50, 2) == LOSS_END and round(np.log(65), 2) == UNIFORM
    print("data ok", np.round(H1[-1], 2), np.round(H2[-1], 2))
