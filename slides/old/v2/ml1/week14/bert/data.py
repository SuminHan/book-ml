"""슬라이드·그림·실습이 공유하는 숫자.
논문: Devlin, Chang, Lee, Toutanova (2018) arXiv:1810.04805 — 토큰 (WordPiece) 의 15% 를 고르고 그중 80% [MASK] · 10% 무작위 · 10% 그대로,
  BERT-BASE 12층 · 768칸 · 머리 12 · 110M, LARGE 24층 · 1024칸 · 340M, 11개 과제 최고 기록, GLUE 80.5% (+7.7%p), BooksCorpus 8억 + 영어 위키백과 25억 단어.
실험: exp_bert.py → assets/bert.json (Hugging Face bert-base-uncased 로 직접 잼: 빈칸 확률, 'bank' 문맥 벡터 유사도, 가중치 수)."""
import os, json
import numpy as np
J = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "bert.json")))
pct = lambda s: [(w, round(p * 100, 1)) for w, p in J["fill"][s]]
BREAD = pct("I went to the [MASK] to buy some bread.")
DOCTOR = pct("I went to the [MASK] to see a doctor.")
FRANCE = pct("The capital of France is [MASK].")
KOREA = pct("The capital of Korea is [MASK].")
SHE = pct("She is a [MASK] at the hospital.")
HE = pct("He is a [MASK] at the hospital.")
BANK = J["bank_sents"]; BANK_SIM = np.array(J["bank_sim"])
P = J["params"]

if __name__ == "__main__":
    assert BREAD[0] == ("market", 78.0) and DOCTOR[0] == ("hospital", 81.0) and FRANCE[0] == ("paris", 41.7) and KOREA[0] == ("seoul", 99.8)
    assert SHE[0] == ("nurse", 60.9) and HE[0] == ("doctor", 28.7) and HE[2] == ("nurse", 17.5) and SHE[2] == ("doctor", 10.9)
    assert np.round(BANK_SIM, 2).tolist() == [[1.0, 0.83, 0.54, 0.52], [0.83, 1.0, 0.42, 0.49], [0.54, 0.42, 1.0, 0.71], [0.52, 0.49, 0.71, 1.0]]
    assert P["vocab"] == 30522 and P["word_emb"] == 30522 * 768 and P["layers"] == 12 and P["layer"] == 7087872
    assert round(P["total"] / 1e6, 1) == 108.9 and round(12 * 768 ** 2 / 1e6, 2) == 7.08
    print("data ok")
