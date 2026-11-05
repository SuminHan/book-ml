"""슬라이드·그림·실습이 공유하는 숫자 (쇼츠 lessons/ml1/kmeans 과 같은 예시: 집 30곳, 배달 기지 3곳).
집 좌표는 0~10 지도 눈금의 예시 값. k-means 반복과 거리² 합(SSE)은 실제 계산."""
import numpy as np

HOUSES = [[2.5, 7.5], [2.3, 6.4], [2.1, 6.4], [2.6, 8.3], [2.1, 6.7], [2.9, 7.5], [2.6, 6.4], [2.5, 7.8], [1.4, 6.8], [0.9, 6.1],
          [5.8, 7.4], [6.3, 7.8], [7.5, 7.4], [5.3, 7.1], [7.4, 7.7], [6.1, 7.2], [6.6, 6.9], [8.3, 6.9], [7.4, 8.4], [6.9, 7.5],
          [5.1, 2.7], [4.0, 2.7], [6.2, 1.3], [5.7, 2.7], [4.5, 4.3], [5.6, 1.6], [5.1, 3.1], [4.8, 3.2], [4.9, 3.2], [6.2, 2.0]]
INIT = [[1.5, 3.0], [4.5, 5.0], [8.5, 3.5]]              # 아무 데나 둔 기지 3곳
BAD_INIT = [[4.4, 3.0], [6.0, 2.2], [4.5, 8.6]]          # 두 곳이 같은 동네에서 시작


def assign(X, C):
    return ((X[:, None] - C[None]) ** 2).sum(-1).argmin(1)


def update(X, lab, C):
    return np.array([X[lab == j].mean(0) if (lab == j).any() else C[j] for j in range(len(C))])


def sse(X, C, lab):
    return float(((X - C[lab]) ** 2).sum())


def kmeans(X, init, n=20):
    C = np.array(init, float); hist = []
    for _ in range(n):
        lab = assign(X, C); hist.append((C.copy(), lab.copy(), sse(X, C, lab)))
        newC = update(X, lab, C)
        if np.allclose(newC, C): break
        C = newC
    return hist


def best_sse(X, k, tries=30, seed=0):
    r = np.random.default_rng(seed)
    return min(kmeans(X, X[r.choice(len(X), k, replace=False)])[-1][2] for _ in range(tries))


SSE_GOOD = [269, 134, 33, 29]
SSE_BAD = [167, 127]
BAD_SIZES = [6, 4, 20]
ELBOW = [275, 134, 29, 22, 16, 11]                       # k = 1 ~ 6, 30번 시작 중 최선
EX3 = [[2, 6], [4, 8], [6, 7]]                           # 갱신 예: 집 3채 → 평균 (4, 7)

if __name__ == "__main__":
    X = np.array(HOUSES)
    g = kmeans(X, INIT); b = kmeans(X, BAD_INIT)
    assert [round(s) for *_, s in g] == SSE_GOOD and np.bincount(g[-1][1]).tolist() == [10, 10, 10]
    assert [round(s) for *_, s in b] == SSE_BAD and np.bincount(b[-1][1]).tolist() == BAD_SIZES
    assert [round(best_sse(X, k)) for k in range(1, 7)] == ELBOW
    assert np.array(EX3).mean(0).tolist() == [4, 7]
    print("data ok", g[-1][0].round(2).tolist())
