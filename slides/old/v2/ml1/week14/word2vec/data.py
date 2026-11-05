"""슬라이드·그림·실습이 공유하는 숫자.
논문: Mikolov 외 (2013) 'Efficient Estimation of Word Representations in Vector Space' (Skip-gram · CBOW, 비유 문제 19,544개: 의미 8,869 + 문법 10,675),
  Mikolov 외 (2013) 'Distributed Representations of Words and Phrases' (네거티브 샘플링, 흔한 단어 솎아 내기). 공개 벡터: Google News 약 1,000억 단어.
실험: exp_w2v.py → assets/w2v.npz (text8 1,700만 단어로 직접 학습, 약 2분), exp_analogy.py → assets/analogy.json.
손 계산: 3칸 벡터 v_고양이 = (1, 0, 1), u_잔다 = (1, 1, 0), 네거티브 u_자동차 = (0, −1, 1)."""
import os, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
Z = np.load(os.path.join(HERE, "assets", "w2v.npz")); A = json.load(open(os.path.join(HERE, "assets", "analogy.json")))
VOCAB = [str(w) for w in Z["vocab"]]; IX = {w: i for i, w in enumerate(VOCAB)}
EMB = Z["E"] / np.linalg.norm(Z["E"], axis=1, keepdims=True)
N_WORDS, N_VOCAB, N_PAIRS = int(Z["n_words"]), int(Z["n_vocab"]), int(Z["n_pairs"])
SEM, SYN = A["sem"], A["syn"]                       # (맞힘, 푼 문제, 전체)
EX, NB = A["examples"], A["neighbors"]
PAIRS = [("france", "paris"), ("germany", "berlin"), ("italy", "rome"), ("spain", "madrid"), ("russia", "moscow"), ("japan", "tokyo"), ("china", "beijing"), ("greece", "athens")]

sig = lambda z: 1 / (1 + np.exp(-z))
V_CAT, U_SLEEP, U_CAR = np.array([1, 0, 1]), np.array([1, 1, 0]), np.array([0, -1, 1])
POS = V_CAT @ U_SLEEP; NEG = V_CAT @ U_CAR
LOSS = -np.log(sig(POS)) - np.log(sig(-NEG))

if __name__ == "__main__":
    assert N_WORDS == 17005207 and N_VOCAB == 71290
    assert SEM[2] + SYN[2] == 19544 and SEM[2] == 8869
    assert round(SEM[0] / SEM[1] * 100, 1) == 23.3 and round(SYN[0] / SYN[1] * 100, 1) == 29.1
    assert EX["france paris germany"][0][0] == "berlin" and EX["japan tokyo korea"][0][0] == "seoul" and EX["man he woman"][0][0] == "she"
    assert [w for w, _ in EX["man king woman"]].index("queen") == 4 and EX["big bigger small"][0][0] == "smaller"
    assert all(w in IX for p in PAIRS for w in p)
    assert POS == 1 and NEG == 1 and round(sig(1), 2) == 0.73 and round(LOSS, 2) == 1.63
    print("data ok", round(LOSS, 3))
