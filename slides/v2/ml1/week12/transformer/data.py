"""슬라이드·그림·실습이 공유하는 숫자.
논문: Vaswani 외 (2017) 'Attention Is All You Need' arXiv:1706.03762 — base: 인코더 6층 + 디코더 6층, d_model 512, 헤드 8 (64칸씩),
  FFN 2048, WMT'14 영어→독일어 BLEU 28.4 (big), 영어→프랑스어 41.8 (big, GPU 8개 3.5일), base 는 GPU 8개 12시간.
  Brown 외 (2020) GPT-3 175B: 96층, d_model 12288, 헤드 96.
예문 '오늘 점심 뭐 먹지 ?' (5토큰) 는 쇼츠 lessons/ml1/transformer 와 같음 (설명용).
실험: exp_tf.py → assets/tf.json (tiny Shakespeare, week11/lstm 과 같은 조건)."""
import os, json
J = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "tf.json")))
VAL = {k: round(v["val"], 2) for k, v in J.items()}
PARAMS = {k: v["params"] for k, v in J.items()}
MS = {k: round(v["ms"]) for k, v in J.items()}
TOKENS = ["오늘", "점심", "뭐", "먹지", "?"]
BASE = dict(layers=6, d=512, heads=8, ff=2048)
GPT3 = dict(layers=96, d=12288, heads=96, params=175e9)


def block_params(d):
    """블록 하나의 가중치 (편향 · 정규화 빼고): 어텐션 Q K V 출력 4d² + 피드포워드 d·4d·2 = 8d²."""
    return 4 * d * d + 8 * d * d


if __name__ == "__main__":
    assert VAL == {"rnn": 1.74, "lstm": 1.64, "tf": 1.61, "tf_nopos": 1.68}, VAL
    assert PARAMS == {"rnn": 49729, "lstm": 148801, "tf": 818241, "tf_nopos": 818241}, PARAMS
    assert BASE["d"] // BASE["heads"] == 64 and BASE["ff"] == 4 * BASE["d"]
    assert block_params(512) == 3145728                                  # 약 315만
    assert round(block_params(GPT3["d"]) * GPT3["layers"] / 1e9) == 174   # ≈ 175B
    assert block_params(128) * 4 == 786432
    print("data ok")
