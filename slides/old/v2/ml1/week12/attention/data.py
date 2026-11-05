"""슬라이드·그림·실습이 공유하는 숫자.
손 계산 (쇼츠 lessons/ml1/attention 과 같음): 직접 설계한 3차원 단어 벡터 (예시), Q = K = V = E, d = 3.
논문: Bahdanau, Cho, Bengio (2014, ICLR 2015) arXiv:1409.0473 — 영어→프랑스어 BLEU (전체 문장, 문장 길이 50 까지 학습)
  RNNencdec-50 17.82 vs RNNsearch-50 26.75 · Vaswani 외 (2017) 'Attention Is All You Need' 의 scaled dot-product.
실험: exp_attention.py → assets/attn.json (숫자열 베끼기, 은닉 32, 길이 5 ~ 30 학습)."""
import os, json
import numpy as np
WORDS = ["동물", "도로", "건너지", "그것은"]
WORDS_EN = ["animal", "street", "cross", "it"]
E = np.array([[3, 0, 0], [0, 3, 0], [0, 0, 3], [2.8, 0.3, 0]], float)
D = 3
SCORE = E @ E.T / np.sqrt(D)
W = np.exp(SCORE - SCORE.max(1, keepdims=True)); W /= W.sum(1, keepdims=True)
OUT = W @ E
IT = 3
BLEU = dict(encdec=17.82, search=26.75)
J = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "attn.json")))
LENS = [5, 10, 15, 20, 25, 30, 40, 50]
ACC = {k: {n: round(J[k]["acc"][str(n)]["tok"] * 100) for n in LENS} for k in ("plain_fwd", "plain_rev", "attn_fwd", "attn_rev")}
SEQ = {k: {n: round(J[k]["acc"][str(n)]["seq"] * 100) for n in LENS} for k in ("plain_fwd", "attn_fwd")}
MAP = J["attn_fwd"]["map"]; MAP_REV = J["attn_rev"]["map"]

if __name__ == "__main__":
    assert [round(x, 2) for x in SCORE[IT]] == [4.85, 0.52, 0.0, 4.58]
    assert [round(x, 3) for x in W[IT]] == [0.561, 0.007, 0.004, 0.427]
    assert [round(x, 2) for x in OUT[IT]] == [2.88, 0.15, 0.01]
    assert ACC["plain_fwd"][30] == 24 and ACC["attn_fwd"][30] == 97 and ACC["attn_fwd"][50] == 88 and ACC["plain_fwd"][50] == 18
    assert SEQ["attn_fwd"][30] == 92 and SEQ["plain_fwd"][30] == 0
    assert ACC["attn_rev"][30] == 63
    assert MAP["out"] == MAP["src"] and np.argmax(MAP["attn"], 1).tolist() == list(range(12))
    print("data ok")
