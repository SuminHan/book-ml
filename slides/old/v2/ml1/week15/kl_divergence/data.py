"""슬라이드·그림·실습이 공유하는 숫자 (단위 비트 = log₂, 쇼츠 lessons/ml1/kl_divergence 와 같은 예시).
뽑기 확률은 예시 (실제 게임 아님): 공지 Q = 당첨 20% · 꽝 80%, 실제 P = 당첨 4% · 꽝 96%. 사진 분류 확률 70% · 95% 도 예시.
실험: exp_kl.py → assets/kl.json (10만 명 뽑기 흉내, 봉우리 둘인 분포를 정규분포 하나로 맞추기: 앞 · 뒤 방향 KL)."""
import os, json
from math import log2
J = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "kl.json")))
H = lambda p: -sum(x * log2(x) for x in p if x > 0)
CE = lambda p, q: -sum(a * log2(b) for a, b in zip(p, q) if a > 0)
KL = lambda p, q: sum(a * log2(a / b) for a, b in zip(p, q) if a > 0)
Q, P = (0.20, 0.80), (0.04, 0.96)
S_P = [-log2(x) for x in P]; S_Q = [-log2(x) for x in Q]
H_P, CE_PQ, KL_PQ, KL_QP = H(P), CE(P, Q), KL(P, Q), KL(Q, P)
CAT = (1.0, 0.0); PRED = [(0.7, 0.3), (0.95, 0.05)]; KL_CAT = [KL(CAT, q) for q in PRED]

if __name__ == "__main__":
    assert [round(s, 2) for s in S_P] == [4.64, 0.06] and [round(s, 2) for s in S_Q] == [2.32, 0.32]
    assert (round(H_P, 2), round(CE_PQ, 2), round(KL_PQ, 2), round(KL_QP, 2)) == (0.24, 0.40, 0.16, 0.25)
    assert abs(KL_PQ - (CE_PQ - H_P)) < 1e-12 and [round(k, 2) for k in KL_CAT] == [0.51, 0.07]
    assert round(J["draw_mean"], 1) == 24.9 and J["draw_median"] == 17 and round(J["over50"] * 100) == 13 and round(J["over100"] * 100, 1) == 1.7
    assert round(J["draw_q_mean"], 1) == 5.0
    assert J["fwd"][1] == [0.0, 2.06] and J["rev"][1] == [-2.0, 0.5] and abs(J["rev"][0] - 0.6931) < 1e-3   # 뒤 방향 최소 = ln 2
    assert round(KL_PQ * 1000) == 160
    print("data ok")
