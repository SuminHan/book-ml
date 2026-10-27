"""슬라이드·그림·실습이 공유하는 숫자.
논문: Sutskever, Vinyals, Le (2014) arXiv:1409.3215 — WMT'14 영어→프랑스어, LSTM 4층 × 1,000칸, 문장 = 숫자 8,000개 벡터,
입력 거꾸로 BLEU 25.9 → 30.6, LSTM 5개 앙상블 34.8 vs 통계 번역 33.3.
실험: exp_seq2seq.py → assets/s2s.json (A 숫자 말 → 숫자, B 숫자열 베끼기 길이별)."""
import os, json
J = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "s2s.json")))
NUM_TEST = round(J["numbers"]["test_acc"] * 100, 1)                    # 99.5 %
NUM_WRONG = J["numbers"]["wrong"]
LENS = [5, 10, 15, 20, 25, 30]
COPY = {n: round(J["copy"][str(n)]["tok"] * 100) for n in LENS}         # 글자별 정답률 %
COPY_REV = {n: round(J["copy_rev"][str(n)]["tok"] * 100) for n in LENS}
SEQ = {n: round(J["copy"][str(n)]["seq"] * 100) for n in LENS}          # 통째로 맞힌 비율 %
SEQ_REV = {n: round(J["copy_rev"][str(n)]["seq"] * 100) for n in LENS}
PAPER = dict(bleu_fwd=25.9, bleu_rev=30.6, ens=34.8, smt=33.3, layers=4, cells=1000, vec=8000)

if __name__ == "__main__":
    assert NUM_TEST == 99.5 and len(NUM_WRONG) == 1, (NUM_TEST, NUM_WRONG)
    assert COPY == {5: 93, 10: 60, 15: 42, 20: 32, 25: 27, 30: 24}, COPY
    assert COPY_REV == {5: 99, 10: 73, 15: 49, 20: 38, 25: 33, 30: 28}, COPY_REV
    assert SEQ[5] == 89 and SEQ_REV[5] == 99 and SEQ[10] == 9 and SEQ_REV[10] == 25 and SEQ[20] == 0
    assert PAPER["layers"] * 2 * PAPER["cells"] == PAPER["vec"]          # 층마다 h, c 각 1,000칸 → 8,000
    print("data ok")
