"""슬라이드·그림·실습이 공유하는 숫자 (쇼츠 lessons/ml1/knn 의 과일 예시 + lessons/ml1/curse_of_dim 의 실험).
과일 점 (크기, 단맛) 은 0~10 눈금의 예시. 귤 하나는 사과 무리 안에 잘못 적힌 점 (라벨 오류 예시).
차원의 저주 수치는 exp_dim.py 로 직접 잰 값 (assets/dim.json)."""
import os, json
from math import dist
from collections import Counter

APPLE = [(7.5, 3.5), (8.2, 4.5), (6.8, 3.0), (7.0, 5.2), (8.5, 3.2), (6.4, 4.0), (7.8, 5.5), (6.6, 5.4)]
TANG = [(2.5, 7.5), (3.2, 8.4), (4.0, 6.8), (2.0, 6.0), (3.5, 5.8), (4.6, 7.6), (2.8, 4.9), (4.2, 8.8),
        (5.0, 6.5), (1.8, 8.0), (3.0, 7.0), (4.8, 5.6)]
NOISE = (6.1, 4.8)
Q = (6.0, 4.6)
PTS = [(p, "사과") for p in APPLE] + [(p, "귤") for p in TANG] + [(NOISE, "귤")]


def knn(pts, q, k, sx=1.0, sy=1.0):
    near = sorted(pts, key=lambda t: dist((t[0][0] * sx, t[0][1] * sy), (q[0] * sx, q[1] * sy)))[:k]
    return near, Counter(l for _, l in near).most_common(1)[0][0]


def loo(pts, k):
    """한 점씩 빼고 나머지로 맞혀 본 정답 수 (leave-one-out)"""
    return sum(knn(pts[:i] + pts[i + 1:], pts[i][0], k)[1] == pts[i][1] for i in range(len(pts)))


NEAR3 = [((6.1, 4.8), "귤", 0.22), ((6.4, 4.0), "사과", 0.72), ((6.6, 5.4), "사과", 1.0)]
LOO = {1: 19, 3: 20, 5: 20, 7: 20, 9: 20, 11: 20, 15: 12, 19: 13}           # 21개 중 맞힌 수
SWEET10 = ["귤", "사과", "귤"]                                                 # 단맛을 0~100 점으로 재면 k=3 이웃
_J = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "dim.json")))
RATIO = {int(k): v for k, v in _J["ratio"].items()}
NOISE_ACC = {int(k): round(v * 100) for k, v in _J["acc"].items()}

if __name__ == "__main__":
    assert len(PTS) == 21 and sum(l == "귤" for _, l in PTS) == 13
    n3, v3 = knn(PTS, Q, 3); assert [(p, l, round(dist(p, Q), 2)) for p, l in n3] == NEAR3 and v3 == "사과"
    assert knn(PTS, Q, 1)[1] == "귤" and knn(PTS, Q, 5)[1] == "사과" and knn(PTS, Q, 21)[1] == "귤"
    assert {k: loo(PTS, k) for k in LOO} == LOO
    n, v = knn(PTS, Q, 3, sy=10); assert [l for _, l in n] == SWEET10 and v == "귤"
    assert (round(RATIO[2], 2), round(RATIO[1000], 1)) == (0.01, 0.9) and NOISE_ACC == {0: 90, 10: 80, 100: 66, 1000: 56}
    print("data ok")
