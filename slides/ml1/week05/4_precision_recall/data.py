"""슬라이드·그림·실습이 공유하는 숫자 (쇼츠 lessons/ml1/precision_recall 과 같은 예시: 학교 100명 중 독감 10명).
AI 가 낸 '독감일 확률' 점수 100개를 그대로 적어 두고, 모든 지표를 여기서 계산한다. 기준(임계값) 0.5 이상 = '독감'."""
N, POS, TH = 100, 10, 0.5
SICK = [0.95, 0.91, 0.86, 0.82, 0.77, 0.71, 0.63, 0.56, 0.44, 0.31]          # 독감 10명의 점수
HEALTHY = [0.81, 0.66, 0.58, 0.53, 0.33, 0.32, 0.27, 0.25, 0.25, 0.25, 0.24, 0.21, 0.21, 0.2, 0.2, 0.2, 0.19, 0.19,
           0.19, 0.17, 0.16, 0.16, 0.16, 0.16, 0.16, 0.16, 0.16, 0.16, 0.15, 0.15, 0.14, 0.14, 0.14, 0.14, 0.14, 0.14,
           0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.12, 0.12, 0.11, 0.11, 0.11, 0.11, 0.1, 0.1, 0.1, 0.1, 0.1, 0.09, 0.09,
           0.09, 0.09, 0.09, 0.08, 0.08, 0.08, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.06, 0.06, 0.06, 0.05,
           0.05, 0.05, 0.05, 0.05, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.03, 0.03, 0.03, 0.03, 0.02, 0.02, 0.01]   # 건강 90명


def confusion(sick, healthy, th):
    tp = sum(s >= th for s in sick); fp = sum(h >= th for h in healthy)
    return tp, fp, len(sick) - tp, len(healthy) - fp          # TP, FP, FN, TN


def metrics(tp, fp, fn, tn):
    acc = (tp + tn) / (tp + fp + fn + tn)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return acc, p, r, f1


TP, FP, FN, TN = 8, 4, 2, 86
ACC, PREC, REC, F1 = 94, 67, 80, 73                            # %
THS = {0.3: (62, 100), 0.5: (67, 80), 0.7: (86, 60), 0.9: (100, 20)}   # 임계값: (정밀도 %, 재현율 %)
ALL_FLU = (10, 10, 100, 18, 55)                                 # '모두 독감': (정확도, 정밀도, 재현율, F1, 정밀도·재현율 산술평균) %

if __name__ == "__main__":
    assert len(SICK) == POS and len(HEALTHY) == N - POS
    assert confusion(SICK, HEALTHY, TH) == (TP, FP, FN, TN)
    assert [round(v * 100) for v in metrics(TP, FP, FN, TN)] == [ACC, PREC, REC, F1]
    assert round(metrics(0, 0, POS, N - POS)[0] * 100) == 90                       # '모두 건강': 정확도 90%
    a, p, r, f = metrics(POS, N - POS, 0, 0)                                       # '모두 독감'
    assert (round(a * 100), round(p * 100), round(r * 100), round(f * 100), round((p + r) * 50)) == (10, 10, 100, 18, 55)
    for th, (pp, rr) in THS.items():
        _, p, r, _ = metrics(*confusion(SICK, HEALTHY, th)); assert (round(p * 100), round(r * 100)) == (pp, rr), th
    print("data ok")
