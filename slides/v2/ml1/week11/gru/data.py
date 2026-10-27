"""슬라이드·그림·실습이 공유하는 숫자.
논문: Cho 외 (2014) EMNLP "Learning Phrase Representations using RNN Encoder-Decoder for SMT" — 리셋 문 · 갱신 문 (GRU 라는 이름은 Chung 외 2014).
식은 원 논문 · PyTorch 와 같은 방향: z = 이전 기억을 '지키는' 비율, h = z * h_prev + (1 - z) * h_tilde.
실험: exp_char.py → assets/char_gru.json (GRU), RNN · LSTM 은 ../lstm/assets/char_cmp.json (같은 조건: tiny Shakespeare, 은닉 128, 2,000걸음).
손 계산 1 (갱신 문): h_prev = 0.8, z = 0.9, h~ = -0.2 → 0.9*0.8 + 0.1*(-0.2) = 0.70  (GRU 의 h 는 -1~1 이라 그 안의 값으로).
손 계산 2 (리셋 문): 후보 h~ = tanh(w x + u (r h_prev)), x = 1, w = 0.5, u = 2, h_prev = 0.8 → r = 1: tanh 2.1 = 0.97, r = 0: tanh 0.5 = 0.46."""
import os, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
G = json.load(open(os.path.join(HERE, "assets", "char_gru.json")))
C = json.load(open(os.path.join(HERE, "..", "lstm", "assets", "char_cmp.json")))
VAL = {k: round(v["val"], 2) for k, v in {**C, **{k: v for k, v in G.items() if k.startswith("gru_")}}.items()}
CURVE = {k: C[f"{k}_128"]["curve"] for k in ("rnn", "lstm")} | {"gru": G["gru_128"]["curve"]}
PARAMS = {k: G[f"params_{k}"] for k in ("rnn", "gru", "lstm")}     # 순환 층만 (입력 128 · 은닉 128, PyTorch 편향 2개)
HAND = dict(h_prev=0.8, z=0.9, h_tilde=-0.2, h=0.70)
RESET = dict(x=1.0, w=0.5, u=2.0, h_prev=0.8, r1=0.97, r0=0.46)
KEEP = 0.999 ** 100                                                  # z = 0.999 를 100걸음 → 90%

if __name__ == "__main__":
    assert VAL == {"rnn_64": 1.74, "lstm_64": 1.64, "rnn_128": 1.71, "lstm_128": 1.61, "gru_64": 1.66, "gru_128": 1.61}, VAL
    assert PARAMS == {"rnn": 33024, "gru": 99072, "lstm": 132096}, PARAMS
    assert PARAMS["gru"] * 4 == PARAMS["lstm"] * 3 and PARAMS["gru"] == 3 * PARAMS["rnn"]
    h = HAND["z"] * HAND["h_prev"] + (1 - HAND["z"]) * HAND["h_tilde"]; assert round(h, 2) == HAND["h"]
    for r, key in ((1, "r1"), (0, "r0")):
        assert round(float(np.tanh(RESET["w"] * RESET["x"] + RESET["u"] * r * RESET["h_prev"])), 2) == RESET[key]
    assert round(KEEP * 100) == 90
    print("data ok")
