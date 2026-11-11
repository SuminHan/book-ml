"""교차 엔트로피 차시 공유 숫자 (슬라이드 · 그림 · 실습 일치). 쇼츠 lessons/ml1/cross_entropy/src/data.py 와 같은 예시.
벌점 단위: 슬라이드 비유는 비트(log₂), 실제 학습 코드(PyTorch·sklearn)는 자연로그 ln. 값은 상수배(1/ln2 ≈ 1.44) 차이."""
from math import log, log2, inf

TOTAL = 100                                   # 칩 100개
def pen(left):                                # 정답 쪽에 남은 칩 → 벌점(비트)
    return inf if left == 0 else -log2(left / TOTAL)

# (고양이, 강아지) 에 건 칩. 정답은 고양이
AI_A, AI_HALF, AI_C, AI_ALLIN = (90, 10), (50, 50), (1, 99), (0, 100)
LEARN = [1, 5, 20, 50, 80, 95]                # 학습하며 고양이 칩이 늘어난다
WORDS = [("먹지", 60), ("먹을까", 25), ("시킬까", 15)]   # "점심 뭐 ___ ?" 정답 = 먹지

# MSE 와 비교 (확신하고 틀림: 정답 고양이인데 고양이 확률 0.01)
Q_WRONG = 0.01
MSE_WRONG = (1 - Q_WRONG) ** 2                # 0.98 (최대 1)

assert pen(100) == 0 and pen(50) == 1 and pen(25) == 2
assert round(pen(90), 2) == 0.15 and round(pen(1), 2) == 6.64 and pen(0) == inf
assert round(pen(95), 2) == 0.07 and round(pen(60), 2) == 0.74
assert round(-log2(0.2), 2) == 2.32 and round(-log(0.1), 2) == 2.30 and round(6.64 * log(2), 2) == 4.60
