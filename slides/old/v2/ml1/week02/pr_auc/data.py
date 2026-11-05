"""슬라이드·그림·실습이 공유하는 숫자 (쇼츠 lessons/ml1/pr_auc 과 같은 예시: 학교 100명 중 독감 10명).
점수 100개는 precision_recall 덱과 같은 배열. ROC 곡선 · ROC-AUC · PR 곡선 · PR-AUC(Average Precision) 를 여기서 계산한다."""
import numpy as np

N, POS, TH = 100, 10, 0.5
SICK = [0.95, 0.91, 0.86, 0.82, 0.77, 0.71, 0.63, 0.56, 0.44, 0.31]          # 독감 10명의 점수
HEALTHY = [0.81, 0.66, 0.58, 0.53, 0.33, 0.32, 0.27, 0.25, 0.25, 0.25, 0.24, 0.21, 0.21, 0.2, 0.2, 0.2, 0.19, 0.19,
           0.19, 0.17, 0.16, 0.16, 0.16, 0.16, 0.16, 0.16, 0.16, 0.16, 0.15, 0.15, 0.14, 0.14, 0.14, 0.14, 0.14, 0.14,
           0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.12, 0.12, 0.11, 0.11, 0.11, 0.11, 0.1, 0.1, 0.1, 0.1, 0.1, 0.09, 0.09,
           0.09, 0.09, 0.09, 0.08, 0.08, 0.08, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.06, 0.06, 0.06, 0.05,
           0.05, 0.05, 0.05, 0.05, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.03, 0.03, 0.03, 0.03, 0.02, 0.02, 0.01]   # 건강 90명



def roc_points(sick, healthy):
    """기준을 1 → 0 으로 내리며 (거짓 양성 비율 FPR, 재현율 TPR)"""
    ths = sorted(set(sick) | set(healthy), reverse=True)
    pts = [(0.0, 0.0)]
    for t in ths:
        pts.append((sum(h >= t for h in healthy) / len(healthy), sum(s >= t for s in sick) / len(sick)))
    return pts


def roc_auc_pairs(sick, healthy):
    """독감-건강 짝 중 독감 점수가 더 높은 비율 (같으면 반)"""
    wins = sum(1.0 if s > h else 0.5 if s == h else 0.0 for s in sick for h in healthy)
    return wins / (len(sick) * len(healthy)), wins


def ranking(sick, healthy):
    """점수 순으로 줄 세운 (점수, 독감=1) 목록"""
    return sorted([(s, 1) for s in sick] + [(h, 0) for h in healthy], key=lambda t: -t[0])


def average_precision(sick, healthy):
    """독감 학생을 하나 찾을 때마다 그 순간의 정밀도를 적고, 평균"""
    found, precs = 0, []
    for k, (_, y) in enumerate(ranking(sick, healthy), 1):
        if y:
            found += 1; precs.append(found / k)
    return float(np.mean(precs)), precs


ROC_AUC, PAIRS_WIN, PAIRS = 0.98, 883, 900
AP = 0.85
AP_RANKS = [1, 2, 3, 4, 6, 7, 9, 11, 13, 16]                    # 독감 학생이 나온 줄 번호
FPR_05, TPR_05 = 4, 80                                           # 기준 0.5: 거짓 양성 비율 4/90 → 4%, 재현율 80%
HEALTHY_X10 = (0.98, 0.55)                                       # 건강 900명 (같은 점수 10배): ROC-AUC 그대로, AP 0.55

if __name__ == "__main__":
    auc, wins = roc_auc_pairs(SICK, HEALTHY)
    assert (round(auc, 2), wins, len(SICK) * len(HEALTHY)) == (ROC_AUC, PAIRS_WIN, PAIRS)
    ap, precs = average_precision(SICK, HEALTHY)
    assert round(ap, 2) == AP
    assert [k for k, (_, y) in enumerate(ranking(SICK, HEALTHY), 1) if y] == AP_RANKS
    x, y = zip(*roc_points(SICK, HEALTHY)); assert abs(np.trapezoid(y, x) - auc) < 1e-9
    assert (round(4 / 90 * 100), 80) == (FPR_05, TPR_05)
    a10, _ = roc_auc_pairs(SICK, HEALTHY * 10); ap10, _ = average_precision(SICK, HEALTHY * 10)
    assert (round(a10, 2), round(ap10, 2)) == HEALTHY_X10, (a10, ap10)
    from sklearn.metrics import roc_auc_score, average_precision_score
    yy = [1] * len(SICK) + [0] * len(HEALTHY)
    assert abs(roc_auc_score(yy, SICK + HEALTHY) - auc) < 1e-9 and abs(average_precision_score(yy, SICK + HEALTHY) - ap) < 1e-9
    print("data ok", [round(p, 2) for p in precs])
