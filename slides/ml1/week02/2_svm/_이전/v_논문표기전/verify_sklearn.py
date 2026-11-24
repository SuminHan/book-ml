"""직접 짠 학습 (data.train_maxmin) 이 sklearn SVC (하드 마진) 와 같은 답을 내는지 검증.
덱 데이터 20개 + 무작위 분리 가능 데이터 200개 (2~5차원, z-score). 비교: w 사이 각도, b 차이, m 비율, 서포트 벡터 집합.
옛 규칙 (w, b 둘 다 ‖w'‖ 로 나눔) 도 같이 돌려 왜 바꿨는지 보인다.
실행: ~/Documents/manim_shorts/_env/v/bin/python verify_sklearn.py   (약 1분)"""
import numpy as np
from sklearn.svm import SVC
from data import RED_PTS, BLUE_PTS, unit, MM_T, MM_LR


def train(X, y, scale_b, T=MM_T, lr0=MM_LR):
    """data.train_maxmin 과 같은 학습. scale_b=True 면 옛 규칙 (b 도 나눔). 반환: w, b, 마지막 1,000걸음에 민 점"""
    w, b = np.zeros(X.shape[1]), 0.0; pushed = np.zeros(len(X), bool)
    for t in range(1, T + 1):
        v = y * (X @ w + b); act = v <= v.min() + 1e-9
        if t > T - 1000: pushed |= act
        eta = lr0 / np.sqrt(t); w = w + eta * (y[act, None] * X[act]).sum(0); b = b + eta * y[act].sum()
        n = np.linalg.norm(w)
        if n > 0: w, b = w / n, (b / n if scale_b else b)
    return w, b, pushed


def compare(X, y, scale_b):
    sk = SVC(kernel="linear", C=1e8).fit(X, y); ws, bs, ms = unit(sk)
    w, b, pushed = train(X, y, scale_b)
    ang = np.degrees(np.arccos(np.clip(ws @ w / np.linalg.norm(w), -1, 1)))
    return ang, abs(b - bs), (y * (X @ w + b)).min() / ms, set(np.where(pushed)[0]) == set(sk.support_)


def datasets(n=200, seed=0):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        d = rng.integers(2, 6); X = rng.normal(size=(40, d)); wt = rng.normal(size=d); wt /= np.linalg.norm(wt)
        v = X @ wt + rng.normal() * 0.3; keep = np.abs(v) > 0.3                 # 경계 근처를 비워 분리 가능하게
        X, y = X[keep], np.sign(v[keep]); yield (X - X.mean(0)) / X.std(0), y


if __name__ == "__main__":
    X = np.array(RED_PTS + BLUE_PTS, float); y = np.r_[np.ones(10), -np.ones(10)]
    D = list(datasets())
    for scale_b, name in ((False, "지금 규칙: w 만 나눔"), (True, "옛 규칙: w, b 둘 다 나눔")):
        a, db, mr, sv = compare(X, y, scale_b)
        print(f"[{name}]\n  덱 데이터: 각도 차 {a:.3f}°, b 차 {db:.4f}, m 비율 {mr:.4f}, 서포트 벡터 같음 {sv}")
        R = np.array([compare(Xr, yr, scale_b) for Xr, yr in D], float)
        print(f"  무작위 200개: 각도 차 최대 {R[:, 0].max():.2f}°, b 차 최대 {R[:, 1].max():.3f}, m 비율 최소 {R[:, 2].min():.3f}, "
              f"m 99% 이상 {int((R[:, 2] > 0.99).sum())}/200, 서포트 벡터 같음 {int(R[:, 3].sum())}/200")
