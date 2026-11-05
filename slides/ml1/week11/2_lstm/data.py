"""슬라이드·그림·실습이 공유하는 숫자.
곱셈 예 (쇼츠 lessons/ml1/lstm 과 같음, 설명용): 걸음마다 0.9 → 100걸음 0.003%, 0.999 → 90%.
논문: Hochreiter & Schmidhuber (1997) Neural Computation 9(8) — 1,000걸음 넘는 간격 · 잊는 문은 Gers, Schmidhuber, Cummins (2000).
실험: exp_char.py → assets/char_cmp.json (tiny Shakespeare, RNN vs LSTM, 은닉 128, 2,000걸음).
손 계산: c_{t-1} = 2, f = 0.9, i = 0.5, c~ = 0.8, o = 0.7."""
import os, json
import numpy as np
RNN_LEFT, LSTM_LEFT = 0.9 ** 100, 0.999 ** 100
J = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "char_cmp.json")))
VAL = {k: round(v["val"], 2) for k, v in J.items()}
HAND = dict(c_prev=2.0, f=0.9, i=0.5, c_tilde=0.8, o=0.7, c=2.2, h=0.68)

if __name__ == "__main__":
    assert round(RNN_LEFT * 100, 3) == 0.003 and round(LSTM_LEFT * 100) == 90
    assert VAL == {"rnn_64": 1.74, "lstm_64": 1.64, "rnn_128": 1.71, "lstm_128": 1.61}, VAL
    c = HAND["f"] * HAND["c_prev"] + HAND["i"] * HAND["c_tilde"]; h = HAND["o"] * np.tanh(c)
    assert round(c, 2) == HAND["c"] and round(float(h), 2) == HAND["h"]
    print("data ok")
